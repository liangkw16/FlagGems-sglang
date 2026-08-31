<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/decode_attention -->
<!-- synced_at: 2026-08-31T12:37:00+08:00 -->

# decode_attention (attention/decode_attention)

## 任务描述

Single-query decode attention (MHA): computes `softmax(q @ K^T * sm_scale) @ V` for each batch element using paged KV cache。本题用例中 `H_Q == H_KV`（Multi-Head Attention）。

输入说明：
- `q`: `[B, H_Q, D]` — 单 token query（decode 阶段每次只有 1 个 token）
- `k_buffer` / `v_buffer`: `[num_pages, H_KV, D]` — paged KV cache pool
- `kv_indptr`: `[B+1]` — CSR 格式，`kv_indptr[b]:kv_indptr[b+1]` 指示第 b 个序列的 page 索引范围
- `kv_indices`: `[total_pages]` — 实际 page ID
- `sm_scale`: float — attention scale factor（通常为 `1/sqrt(D)`）

## 接口签名

```python
def reference(q, k_buffer, v_buffer, kv_indptr, kv_indices, sm_scale)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- 对每个 batch 元素 `b`:
  1. 通过 `kv_indptr[b]:kv_indptr[b+1]` 和 `kv_indices` 索引获取 KV 序列
  2. `logits = q[b] @ K.T * sm_scale`（float32 计算）
  3. `logits -= logits.max()`（数值稳定）
  4. `p = softmax(logits)`
  5. `o[b] = p @ V`
- 输出: `[B, H_Q, D_v]`，dtype float32

## 正确性判别标准

`atol=3e-2, rtol=1e-2`.


## 参考实现

```python
import torch


def reference(q, k_buffer, v_buffer, kv_indptr, kv_indices, sm_scale):
    B = kv_indptr.size(0) - 1
    _, H_Q, D = q.shape
    _, H_KV, _ = k_buffer.shape
    group_size = H_Q // H_KV

    o_ref = torch.empty((B, H_Q, v_buffer.shape[-1]), dtype=torch.float32, device=q.device)
    for b in range(B):
        start = int(kv_indptr[b].item())
        end = int(kv_indptr[b + 1].item())
        idx = kv_indices[start:end]

        k_seq = k_buffer.index_select(0, idx)
        v_seq = v_buffer.index_select(0, idx)
        if H_KV != H_Q:
            k_seq = k_seq.repeat_interleave(group_size, dim=1)
            v_seq = v_seq.repeat_interleave(group_size, dim=1)

        q_f32 = q[b].to(torch.float32)
        k_f32 = k_seq.to(torch.float32)
        v_f32 = v_seq.to(torch.float32)

        logits = torch.einsum("hd,lhd->hl", q_f32, k_f32) * float(sm_scale)
        logits = logits - logits.max(dim=-1, keepdim=True).values
        p = torch.softmax(logits, dim=-1)
        o_ref[b] = torch.einsum("hl,lhd->hd", p, v_f32)

    return o_ref
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
