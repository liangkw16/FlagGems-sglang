<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/extend_attention -->
<!-- synced_at: 2026-09-07T11:09:08+08:00 -->

# extend_attention (attention/extend_attention)

## 任务描述

分块预填充 / 推测性扩展注意力算子：新增（extend）query token 对已在 KV 缓存中的前缀 token 以及自身做因果注意力。每个 batch 请求的扩展 token 与其对应的前缀 KV 通过 `qo_indptr`、`kv_indptr`、`kv_indices` 三组索引拼接定位。仅覆盖标准因果路径（无自定义掩码、无滑动窗口、无 logit cap）。

## 接口签名

```python
def extend_attention(
    q_extend, k_extend, v_extend, k_buffer, v_buffer,
    qo_indptr, kv_indptr, kv_indices, max_len_extend,
):
```

> 选手实现的函数签名需与上述 `extend_attention(...)` 完全一致。

## 计算定义

- `q_extend`: `[E, H_Q, D]`，`k_extend`/`v_extend`: `[E, H_KV, D]`，`k_buffer`/`v_buffer`: `[T, H_KV, D]`
- batch `i` 的扩展 token：`q_extend[qo_indptr[i]:qo_indptr[i+1]]`
- batch `i` 的前缀 KV：`k_buffer[kv_indices[kv_indptr[i]:kv_indptr[i+1]]]`（v 同）
- `k_full = concat(k_prefix, k_extend_i)`，`v_full = concat(v_prefix, v_extend_i)`；`H_Q % H_KV == 0` 时对 KV 做 GQA 扩展
- 缩放因子 `scale = 1 / sqrt(D)`
- 注意力分数：`scores[q, h, k] = scale * q_extend[q, h, :] · k_full[k, h, :]`
- 因果掩码：位置 `k <= prefix_len + q_offset` 时保留，否则置 `-inf`
- `weights = softmax(scores, dim=-1)`，输出 `o[q, h, :] = sum_k weights[q,h,k] * v_full[k,h,:]`
- 输出 `[E, H_Q, D]`，float32

## 正确性判别标准

Per-dtype tolerance（对 fp32 cast 后的输出）：
- `atol=1e-2, rtol=1e-2`


## 参考实现

```python
import torch
import torch.nn.functional as F


def reference(
    q_extend, k_extend, v_extend, k_buffer, v_buffer,
    qo_indptr, kv_indptr, kv_indices, max_len_extend,
):
    B = qo_indptr.size(0) - 1
    _, H_Q, D = q_extend.shape
    _, H_KV, _ = k_extend.shape
    group_size = H_Q // H_KV
    scale = 1.0 / D**0.5

    o = torch.empty_like(q_extend, dtype=torch.float32)
    for i in range(B):
        q_start, q_end = int(qo_indptr[i].item()), int(qo_indptr[i + 1].item())
        kv_start, kv_end = int(kv_indptr[i].item()), int(kv_indptr[i + 1].item())

        prefix_indices = kv_indices[kv_start:kv_end]
        k_prefix = k_buffer[prefix_indices]
        v_prefix = v_buffer[prefix_indices]

        k_ext = k_extend[q_start:q_end]
        v_ext = v_extend[q_start:q_end]
        q_ext = q_extend[q_start:q_end]

        k_full = torch.cat([k_prefix, k_ext], dim=0).float()
        v_full = torch.cat([v_prefix, v_ext], dim=0).float()
        if group_size != 1:
            k_full = k_full.repeat_interleave(group_size, dim=1)
            v_full = v_full.repeat_interleave(group_size, dim=1)

        prefix_len = k_prefix.size(0)
        extend_len = k_ext.size(0)
        total_len = prefix_len + extend_len

        pos_keys = torch.arange(total_len, device=q_extend.device)
        t = prefix_len + torch.arange(extend_len, device=q_extend.device)
        causal_mask = pos_keys.unsqueeze(0) <= t.unsqueeze(1)

        attn_scores = (
            torch.einsum("qhd,khd->qhk", q_ext.float(), k_full) * scale
        )
        attn_scores = attn_scores.masked_fill(~causal_mask.unsqueeze(1), float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)
        o[q_start:q_end] = torch.einsum("qhk,khd->qhd", attn_weights, v_full)

    return o
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
