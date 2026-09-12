<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/hc_head -->
<!-- synced_at: 2026-09-12T20:28:08+08:00 -->

# hc_head (activation_norm/hc_head)

## 任务描述

融合的 DSV4 "hc_head" LM-head 混合器：在单次 kernel 启动中完成 RMSNorm + 线性混合 + sigmoid 门控 + 加权求和，将多头码本 LM head 的 `hc_mult` 轴折叠为每个 token 的单个 `hidden_size` 输出向量。SGLang baseline 对应 `sglang.kernels.ops.layernorm.mhc_head.fused_hc_head`。

## 接口签名

```python
def hc_head(x, hc_fn, hc_scale, hc_base, norm_eps, hc_eps)
```

> 选手实现的函数签名需与上述 `hc_head(...)` 完全一致。

## 计算定义

- `x`: `[T, hc_mult, hidden_size]` bfloat16，主输入张量
- `hc_fn`: `[hc_mult, hc_mult * hidden_size]` float32，线性混合权重矩阵
- `hc_scale`: `[1]` float32，sigmoid 的缩放系数
- `hc_base`: `[hc_mult]` float32，sigmoid 的偏置
- `norm_eps`, `hc_eps`: 标量，分别为 RMSNorm 与 sigmoid 输出的数值稳定项

将输入展平后在 float32 下计算：

$$mathbf{x}_	ext{flat} = 	ext{flatten}(mathbf{x}, 	ext{dim}=1) in mathbb{R}^{T 	imes (hc\_mult cdot hidden\_size)}$$

RMSNorm 的倒数标准差（按行计算）：

$$r = frac{1}{sqrt{frac{1}{D}sum_{i} x_{	ext{flat},i}^2 + epsilon_	ext{norm}}} in mathbb{R}^{T 	imes 1}$$

线性混合得分（与 RMSNorm 系数相乘）：

$$	ext{mixes} = (mathbf{x}_	ext{flat} cdot mathbf{hc\_fn}^	op) odot r in mathbb{R}^{T 	imes hc\_mult}$$

sigmoid 门控（加入 `hc_eps` 保证正值）：

$$	ext{pre} = sigma(	ext{mixes} cdot hc\_scale + hc\_base) + epsilon_	ext{hc} in mathbb{R}^{T 	imes hc\_mult}$$

加权求和，折叠 `hc_mult` 轴：

$$y = sum_{j=1}^{hc\_mult} 	ext{pre}_{:,j} cdot mathbf{x}_{:,j,:} in mathbb{R}^{T 	imes hidden\_size}$$

结果 cast 回输入 `x` 的原始 dtype。

## 正确性判别标准

bf16 default
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`

## 参考实现

```python
import torch
import torch.nn.functional as F


def reference(x, hc_fn, hc_scale, hc_base, norm_eps, hc_eps):
    shape, dtype = x.size(), x.dtype
    x = x.flatten(1).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(x, hc_fn) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
    return y.to(dtype)
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
