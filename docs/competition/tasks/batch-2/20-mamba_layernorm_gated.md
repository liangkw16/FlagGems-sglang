<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/mamba_layernorm_gated -->
<!-- synced_at: 2026-08-31T12:37:00+08:00 -->

# mamba_layernorm_gated (mamba/layernorm_gated)

## 任务描述

Gated LayerNorm for Mamba: applies layer normalization then multiplies by a gated activation `silu(z) * norm(x)`.

## 接口签名

```python
def reference(x, weight, bias, eps, z=None, group_size=None, norm_before_gate=True, is_rms_norm=True)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- 将输入按 `group_size` 分组: `x.view(M, ngroups, group_size)`
- 若 `norm_before_gate=False`: 先门控 `x = x * z * sigmoid(z)`
- RMS norm（`is_rms_norm=True`）或 LayerNorm:
  - RMS: `x_hat = x * rsqrt(mean(x^2) + eps)`
  - LN: `x_hat = (x - mean) * rsqrt(var + eps)`
- 应用 weight（和可选 bias）
- 若 `norm_before_gate=True`: 后门控 `y = y * z * sigmoid(z)`
- 输出: `[M, N]`，cast 回输入 dtype

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`


## 参考实现

```python
import torch


def reference(x, weight, bias, eps, z=None, group_size=None, norm_before_gate=True, is_rms_norm=True):
    M, N = x.shape
    if group_size is None:
        group_size = N
    ngroups = N // group_size
    out_dtype = x.dtype

    xf = x.float().view(M, ngroups, group_size)
    zf = z.float().view(M, ngroups, group_size) if z is not None else None

    if zf is not None and not norm_before_gate:
        xf = xf * zf * torch.sigmoid(zf)

    if is_rms_norm:
        var = (xf**2).mean(dim=-1, keepdim=True)
    else:
        mean = xf.mean(dim=-1, keepdim=True)
        xf = xf - mean
        var = (xf**2).mean(dim=-1, keepdim=True)

    rstd = torch.rsqrt(var + eps)
    x_hat = xf * rstd

    w = weight.float().view(ngroups, group_size)
    y = x_hat * w
    if bias is not None:
        b = bias.float().view(ngroups, group_size)
        y = y + b

    if zf is not None and norm_before_gate:
        y = y * zf * torch.sigmoid(zf)

    return y.reshape(M, N).to(out_dtype)
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
