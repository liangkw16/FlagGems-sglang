<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/post_reorder_deepgemm -->
<!-- synced_at: 2026-09-26T22:45:04+08:00 -->

# post_reorder_deepgemm (moe/post_reorder_deepgemm)

## 任务描述

DeepGEMM 变体的 MoE epilogue。与 `post_reorder_cutlass` 算术相同，但有效性门是 `expert_id >= 0` 而非 `!= num_local_experts` —— DeepGEMM 路径用 `-1` 表示 padding，并把 **fused shared expert** 放在 `num_experts`，所以 CUTLASS 的门会把它静默丢掉。

## 接口签名

```python
def reference(down_output, output, src2dst, topk_ids, topk_weights, topk, num_tokens, hidden_size, routed_scaling_factor)
```

> 选手实现的函数签名需与上述完全一致。

## 计算定义

- `-1` 标记 padded slot；`num_experts` 是 fused shared expert（有效）

```
output[t, :] = routed_scaling_factor *
               sum over slots i with topk_ids[t, i] >= 0 of
               down_output[src2dst[t, i], :] * topk_weights[t, i]
```

## 正确性判别标准

Per-dtype tolerance（standard `harness.correctness`）。

## 参考实现

```python
import torch


def reference(
    down_output,
    output,
    src2dst,
    topk_ids,
    topk_weights,
    topk,
    num_tokens,
    hidden_size,
    routed_scaling_factor,
):
    acc = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device=output.device)
    for i in range(topk):
        valid = (topk_ids[:, i] >= 0).float()
        rows = down_output[src2dst[:, i].clamp(min=0).long()].float()
        acc += rows * (topk_weights[:, i].float() * valid)[:, None]
    return (acc * routed_scaling_factor).to(output.dtype)
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
