<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/fused_rmsnorm -->
<!-- synced_at: 2026-08-24T02:17:26+08:00 -->

# fused_rmsnorm (activation_norm/fused_rmsnorm)

## 任务描述

Fused RMS normalization: `x * rsqrt(mean(x^2) + eps) * weight`, computed in fp32 and cast back.

## 接口签名

```python
def reference(x, weight, eps)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- `rms = sqrt(mean(x^2, dim=-1) + eps)`
- `out = (x / rms) * weight`
- 中间计算使用 float32，输出 cast 回输入 dtype

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`


## 参考实现

```python
import torch


def reference(x, weight, eps):
    x32 = x.float()
    rms = torch.sqrt((x32 * x32).mean(dim=-1, keepdim=True) + eps)
    out = (x32 / rms) * weight.float()
    return out.to(x.dtype)
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
