<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/seqlens_expand -->
<!-- synced_at: 2026-09-12T20:28:08+08:00 -->

# seqlens_expand (attention/seqlens_expand)

## 任务描述

把每请求的 `(qo_len, kv_len)` 对展开成 per-query-token 的 KV 长度向量 ——
extend batch 中每个 query 行能看到的 causal 长度。`clamp(..., min=0)` 是保命的：
DP-padded / idle 行可能有 `kv_len < qo_len`，下游消费者按 uint32 读这些长度，
负值会变成约 4e9 token 的长度并导致非法访问。

## 接口签名

```python
def reference(extend_seq_lens, seq_lens, total_len, max_q_len)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- `extend_seq_lens`、`seq_lens`：`[N]` int32 CUDA。
- 计算流程：

  ```
  out[offset[i] : offset[i] + qo_len[i]]
      = clamp(kv_len[i] - qo_len[i] + 1 + arange(qo_len[i]), min=0)
  offset = exclusive_cumsum(extend_seq_lens)
  ```

- 输出：`[total_len]` int32。

## 正确性判别标准

精确（整数运算）。

## 参考实现

```python
import torch


def reference(extend_seq_lens, seq_lens, total_len, max_q_len):
    # out[offset[i] : offset[i] + qo_len[i]] = clamp(kv_len[i] - qo_len[i] + 1 + arange(qo_len[i]), min=0)
    #   offset[i] = exclusive_cumsum(extend_seq_lens)
    # Vectorized: repeat each request's base to its qo_len rows and add the
    # in-request position.
    device = extend_seq_lens.device
    N = extend_seq_lens.shape[0]

    offsets = torch.zeros(N + 1, dtype=torch.int32, device=device)
    torch.cumsum(extend_seq_lens, dim=0, out=offsets[1:])

    req_idx = torch.repeat_interleave(
        torch.arange(N, device=device), extend_seq_lens
    )
    # torch.arange defaults to int64; keep every intermediate in int32 so the
    # output matches the spec'd int32 dtype.
    global_pos = torch.arange(total_len, device=device, dtype=torch.int32)
    local_pos = global_pos - offsets[req_idx]
    base = seq_lens - extend_seq_lens + 1
    vals = (base[req_idx] + local_pos).clamp(min=0)
    return vals.to(torch.int32)
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
