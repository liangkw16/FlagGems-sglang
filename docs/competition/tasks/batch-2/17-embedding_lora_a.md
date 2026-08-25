<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/embedding_lora_a -->
<!-- synced_at: 2026-08-25T00:19:05+08:00 -->

# embedding_lora_a (lora/embedding_lora_a)

## 任务描述

LoRA-A embedding lookup: gathers embedding rows per-segment according to LoRA adapter routing, producing the first-stage LoRA output.

## 接口签名

```python
def reference(input_ids, weights, batch_info, vocab_size, extra_embeddings=None)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- 对每个 segment（由 batch_info 指定的连续 token 区间）:
  1. 获取 adapter weight index 和 rank
  2. 用 `input_ids` 做 embedding lookup: `out[rows, :r] = weights[w_idx, :r, tokens].T`
  3. 若有 `extra_embeddings` 且 token >= vocab_size，则用 extra embedding 替换
- 输出: `[S, rank]`，与 weights 同 dtype

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`


## 参考实现

```python
import torch


def reference(input_ids, weights, batch_info, vocab_size, extra_embeddings=None):
    S = input_ids.shape[0]
    rank = weights.shape[1]
    out = torch.zeros(S, rank, dtype=weights.dtype, device=weights.device)

    seg_indptr = batch_info.seg_indptr
    weight_indices = batch_info.weight_indices
    lora_ranks = batch_info.lora_ranks

    for b in range(batch_info.bs):
        start = int(seg_indptr[b].item())
        end = int(seg_indptr[b + 1].item())
        if start == end:
            continue
        w_idx = int(weight_indices[b].item())
        r = int(lora_ranks[w_idx].item())
        if r == 0:
            continue

        tokens = input_ids[start:end].long()
        is_extra = tokens >= vocab_size
        clamped = tokens.clamp(max=vocab_size - 1)
        out[start:end, :r] = weights[w_idx, :r, clamped].t()

        if extra_embeddings is not None and bool(is_extra.any()):
            extra_idx = (tokens - vocab_size).clamp(min=0)
            extra_vals = extra_embeddings[w_idx, extra_idx, :r]
            out[start:end, :r] = torch.where(
                is_extra.unsqueeze(-1), extra_vals, out[start:end, :r]
            )

    return out
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
