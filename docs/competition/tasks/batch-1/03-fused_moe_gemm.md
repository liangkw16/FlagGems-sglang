<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/fused_moe_gemm -->
<!-- synced_at: 2026-08-24T02:17:26+08:00 -->

# fused_moe_gemm (moe/fused_moe_gemm)

## 任务描述

The core fused-MoE GEMM: each token's row is multiplied against the weight
matrix of whichever expert it was routed to (one call per top-k slot), with
the routing weight applied.

## 接口签名

```python
def reference(A, B, topk_weights, topk_ids, top_k):
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- `out[t, j] = (A[t] @ B[topk_ids[t,j]].T) * topk_weights[t, j]`.

The baseline's actual kernel call additionally needs `sorted_token_ids` /
`expert_ids` / `num_tokens_post_padded` (from `moe_align_block_size`) to
group tokens by assigned expert for the grid — that's scheduling metadata
that doesn't affect the output layout (the kernel writes each result back to
its own token's row via `offs_token`, not to its sorted/padded position), so
`baseline.py` computes it internally; it isn't part of the logical contract.

## 正确性判别标准

`atol=0.5, rtol=1e-2`.


## 参考实现

```python
import torch


def reference(A, B, topk_weights, topk_ids, top_k):
    T, K = A.shape
    E, N, _ = B.shape
    A32 = A.float()
    B32 = B.float()

    out = torch.empty(T, top_k, N, dtype=A.dtype, device=A.device)
    for t in range(T):
        for j in range(top_k):
            e = int(topk_ids[t, j].item())
            row = A32[t] @ B32[e].t()
            out[t, j] = (row * topk_weights[t, j].float()).to(A.dtype)
    return out
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
