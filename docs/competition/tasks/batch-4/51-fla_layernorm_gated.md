<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/fla_layernorm_gated -->
<!-- synced_at: 2026-09-12T00:03:19+08:00 -->

# fla_layernorm_gated (fla/layernorm_gated)

## 任务描述

FLA 系列融合门控归一化算子：先对输入 `x` 做 RMSNorm 或 LayerNorm，再乘以可学习的 weight/bias，最后用 gate 张量 `g` 施加门控激活。与 Mamba/SSD 组的 `layernorm_gated` 不同，本算子无通道分组，gate 始终在归一化之后施加，且门控激活函数可在 `swish`/`silu`/`sigmoid` 中选择。不涉及 residual 融合。

## 接口签名

```python
def fla_layernorm_gated(x, g, weight, bias, activation="swish", eps=1e-5, is_rms_norm=True):
```

> 选手实现的函数签名需与上述 `fla_layernorm_gated(...)` 完全一致。

## 计算定义

- `x`: `[T, D]` 输入张量；`g`: `[T, D]` 门控张量；`weight`: `[D]` 或 None；`bias`: `[D]` 或 None
- 归一化（在 float32 精度下）：
  - `is_rms_norm=True`（RMSNorm）：`x_hat = x / sqrt(mean(x²) + eps)`
  - `is_rms_norm=False`（LayerNorm）：先减均值，再 `x_hat = (x - mean) / sqrt(var + eps)`
- 仿射：`y = x_hat * weight + bias`（weight/bias 为 None 时跳过）
- 门控：
  - `activation in {"swish", "silu"}`：`y = y * g * sigmoid(g)`（swish gate）
  - `activation == "sigmoid"`：`y = y * sigmoid(g)`
- 输出 cast 回输入 dtype，shape 为 `[T, D]`

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`


## 参考实现

```python
def reference(x, g, weight, bias, activation="swish", eps=1e-5, is_rms_norm=True):
    out_dtype = x.dtype
    xf = x.float()

    if is_rms_norm:
        var = (xf**2).mean(dim=-1, keepdim=True)
        x_hat = xf * (var + eps).rsqrt()
    else:
        mean = xf.mean(dim=-1, keepdim=True)
        var = ((xf - mean) ** 2).mean(dim=-1, keepdim=True)
        x_hat = (xf - mean) * (var + eps).rsqrt()

    y = x_hat
    if weight is not None:
        y = y * weight.float()
    if bias is not None:
        y = y + bias.float()

    gf = g.float()
    if activation in ("swish", "silu"):
        y = y * gf * gf.sigmoid()
    elif activation == "sigmoid":
        y = y * gf.sigmoid()

    return y.to(out_dtype)
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
