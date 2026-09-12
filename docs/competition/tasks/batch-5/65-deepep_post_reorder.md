<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/deepep_post_reorder -->
<!-- synced_at: 2026-09-12T20:28:08+08:00 -->

# deepep_post_reorder (moe/deepep_post_reorder)

## 任务描述

DeepEP MoE epilogue：从置换布局把每个 token 的 `topk` 个专家输出 gather 回来，并用路由权重合并。

## 接口签名

```python
def reference(down_output, output, src2dst, topk_ids, topk_weights, topk, hidden_size, routed_scaling_factor)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- `down_output`: `[num_tokens * topk, hidden_size]`，`output`: `[num_tokens, hidden_size]`
- `src2dst`: `[num_tokens, topk]` int32，`-1` 表示跳过

```
output[t, :] = sum over slots i with src2dst[t, i] >= 0 of
               down_output[src2dst[t, i], :] * (topk_weights[t, i] * routed_scaling_factor)
```

累加器的精度并不固定：Triton 在 `routed_scaling_factor == 1.0` 时把 `sum_vec` 保持在 bf16，在 scale 乘法处提升为 fp32；reference 恒以 fp32 累加。

## 正确性判别标准

特殊策略（mismatch 比例）：`|actual - expected| > (3e-2 + 3e-2 * |expected|)` 的元素占比必须 `< 1e-2`，从而两种累加策略都是合法提交。

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
    hidden_size,
    routed_scaling_factor
):
    acc = torch.zeros(
        output.shape, dtype=torch.float32, device=output.device
    )
    for i in range(topk):
        dst = src2dst[:, i]
        valid = (dst >= 0).float()
        rows = down_output[dst.clamp(min=0).long()].float()
        w = topk_weights[:, i].to(down_output.dtype).float() * routed_scaling_factor
        acc += rows * (w * valid)[:, None]
    return acc.to(output.dtype)
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
