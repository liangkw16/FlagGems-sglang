<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/zero_experts_identity -->
<!-- synced_at: 2026-09-26T22:45:04+08:00 -->

# zero_experts_identity (moe/zero_experts_identity)

## 任务描述

"Zero experts"（identity experts）：在把一部分 token 路由到 no-op 路径的模型中，任何 expert id `>= num_experts` 的路由 slot 都是 identity expert，其贡献就是 token 自己的 hidden state 乘路由权重。

## 接口签名

```python
def reference(expert_indices, expert_scales, num_experts, zero_expert_type, hidden_states)
```

> 选手实现的函数签名需与上述完全一致。

## 计算定义

- `expert_indices` / `expert_scales`: `[num_tokens, top_k]`
- `hidden_states`: `[num_tokens, hidden_dim]`，`hidden_dim` 是 256 的倍数

```
zero_scales = where(expert_indices >= num_experts, expert_scales, 0)
output[t, :] = hidden_states[t, :] * zero_scales[t, :].sum()
```

## 正确性判别标准

Per-dtype tolerance（standard `harness.correctness`）。

## 参考实现

```python
import torch


def reference(expert_indices, expert_scales, num_experts, zero_expert_type, hidden_states):
    zero_scales = torch.where(
        expert_indices >= num_experts, expert_scales, torch.zeros_like(expert_scales)
    )
    total = zero_scales.float().sum(dim=-1, keepdim=True)
    return (hidden_states.float() * total).to(hidden_states.dtype)
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
