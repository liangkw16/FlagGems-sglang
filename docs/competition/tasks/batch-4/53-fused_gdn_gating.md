<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/fused_gdn_gating -->
<!-- synced_at: 2026-09-07T11:09:08+08:00 -->

# fused_gdn_gating (fla/fused_gdn_gating)

## 任务描述

为 Gated DeltaNet 计算融合门控：给定每头的对数衰减参数 `A_log`、原始门控 logits `a` 和 `b`，以及偏置 `dt_bias`，在一个核中同时计算对数衰减门 `g` 和输出门 `beta_output`。该操作仅在 decode 阶段调用（每行一个 token），因此输出带有一个大小为 1 的前置 seq_len 维度。SGLang baseline 对应 `sglang.kernels.ops.attention.fla.fused_gdn_gating.fused_gdn_gating`。

## 接口签名

```python
def fused_gdn_gating(A_log, a, b, dt_bias, beta=1.0, threshold=20.0)
```

> 选手实现的函数签名需与上述 `fused_gdn_gating(...)` 完全一致。

## 计算定义

- `A_log`: `[H]` float32，每头的对数衰减参数
- `a`, `b`: `[B, H]` float32，原始门控 logits（分别用于计算 `g` 和 `beta_output`）
- `dt_bias`: `[H]` float32，时间步偏置，广播到 batch 维
- `beta`, `threshold`: 标量 softplus 参数，默认 `1.0` / `20.0`

中间量 `x`：

$$x = a + 	ext{dt\_bias}$$

数值稳定的 beta-softplus（高于阈值时线性通过）：

$$	ext{softplus\_x} = egin{cases} x & 	ext{if } eta cdot x > 	ext{threshold} \ frac{1}{eta}ln(1 + e^{eta x}) & 	ext{otherwise} end{cases}$$

对数衰减门：

$$g = -exp(A\_	ext{log}) cdot 	ext{softplus\_x}$$

输出门：

$$eta\_	ext{output} = sigma(b) = frac{1}{1 + e^{-b}}$$

两个输出均在 float32 精度下计算，并在最前面增加一个大小为 1 的维度：

$$g in mathbb{R}^{1 	imes B 	imes H}, quad eta\_	ext{output} in mathbb{R}^{1 	imes B 	imes H}$$

函数返回顺序为 `(g, beta_output)`。

## 正确性判别标准

float32 default
- float32: `atol=1e-4, rtol=1e-4`

## 参考实现

```python
import torch
import torch.nn.functional as F


def reference(A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
    x = a.float() + dt_bias.float()
    softplus_x = torch.where(beta * x <= threshold, F.softplus(x, beta=beta), x)
    g = -torch.exp(A_log.float()) * softplus_x
    beta_output = torch.sigmoid(b.float())
    return g.unsqueeze(0).to(torch.float32), beta_output.unsqueeze(0).to(torch.float32)
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
