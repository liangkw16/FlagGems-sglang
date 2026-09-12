<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/group_norm_silu -->
<!-- synced_at: 2026-09-12T20:28:08+08:00 -->

# group_norm_silu (diffusion/group_norm_silu)

## 任务描述

面向 diffusion U-Net / VAE 的 Fused GroupNorm + SiLU：`out = silu(group_norm(x, num_groups, weight, bias, eps))`。

## 接口签名

```python
def reference(x, weight, bias, num_groups, eps)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- `x`: `[N, C, ...]`，`C % num_groups == 0`
- `weight`/`bias`: `[C]`
- 先在 fp32 计算 group norm 统计量，再做 affine，最后施加 SiLU 并转回输入 dtype

## 正确性判别标准

标准 per-dtype tolerance（float32: atol=rtol=1e-4；bfloat16: 1.5e-2；float16: 1e-2）

## 参考实现

```python
import torch.nn.functional as F


def reference(x, weight, bias, num_groups, eps):
    y = F.group_norm(x.float(), num_groups, weight.float(), bias.float(), eps)
    return F.silu(y).to(x.dtype)
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
