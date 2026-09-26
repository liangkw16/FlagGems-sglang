<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/indexed_scale_shift -->
<!-- synced_at: 2026-09-26T22:45:04+08:00 -->

# indexed_scale_shift (diffusion/indexed_scale_shift)

## 任务描述

Per-row AdaLN 调制，scale/shift 表通过索引 gather：每行从 `indices[row]` 取自己的调制向量。

## 接口签名

```python
def reference(x, shift, scale, indices)
```

> 选手实现的函数签名需与上述完全一致。

## 计算定义

- `x`: `[rows, hidden]` bf16
- `shift`/`scale`: `[num_variants, hidden]` bf16
- `indices`: `[rows]` int32/int64
- 公式：`out[row] = bf16_round(x[row] * bf16_round(1 + scale[indices[row]])) + shift[indices[row]]`
- 两次显式 bf16 round 复现 eager kernel 边界，是契约的一部分

## 正确性判别标准

标准 per-dtype tolerance

## 参考实现

```python
import torch


def reference(x, shift, scale, indices):
    idx = indices.long()
    sh = shift[idx].float()
    sc = scale[idx].float()
    one_plus = (1.0 + sc).to(torch.bfloat16).float()
    scaled = (x.float() * one_plus).to(torch.bfloat16).float()
    return (scaled + sh).to(x.dtype)
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
