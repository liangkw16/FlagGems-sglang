<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/l2norm -->
<!-- synced_at: 2026-09-12T20:28:08+08:00 -->

# l2norm (activation_norm/l2norm)

## 任务描述

沿最后一个维度做 L2 归一化：将输入除以其 L2 范数，不含可学习权重。该操作在 FLA 风格的线性注意力中用于 Q/K 归一化。计算在 float32 精度下进行，结果 cast 回输入的原始 dtype。SGLang baseline 对应 `sglang.kernels.ops.attention.fla.l2norm.l2norm_fwd`。

## 接口签名

```python
def l2norm(x, eps=1e-6):
```

> 选手实现的函数签名需与上述 `l2norm(...)` 完全一致。

## 计算定义

$$	ext{out} = frac{mathbf{x}}{sqrt{sum_{i} x_i^2 + epsilon}}$$

其中求和沿最后一个维度（`dim=-1`）进行，结果保持维度（`keepdim=True`），计算在 float32 精度下完成，最终 cast 回 `x.dtype`：

$$	ext{out} = left(mathbf{x}_	ext{float32} cdot frac{1}{sqrt{|mathbf{x}_	ext{float32}|_2^2 + epsilon}}ight) 	o 	ext{x.dtype}$$

## 输入输出规格

| 参数 | 形状 | dtype | 说明 |
|------|------|-------|------|
| `x` | `[..., D]` | float16 / bfloat16 / float32 | 任意前置维度，归一化沿最后维 |
| `eps` | 标量 | float | 数值稳定项，默认 1e-6 |
| **output** | `[..., D]` | 同 `x` | L2 归一化后的结果，形状与 `x` 相同 |

## 测试用例

| 形状 | dtype | eps |
|------|-------|-----|
| `[1, 64]` | bfloat16 | 1e-6 |
| `[4, 128]` | bfloat16 | 1e-6 |
| `[16, 256]` | bfloat16 | 1e-6 |
| `[8, 32, 128]` | bfloat16 | 1e-6 |
| `[4, 64]` | float32 | 1e-6 |

正确性判别：bfloat16 default，`atol=1.5e-2, rtol=1.5e-2`。

## 参考实现

```python
def reference(x, eps=1e-6):
    xf = x.float()
    rstd = (xf.pow(2).sum(dim=-1, keepdim=True) + eps).rsqrt()
    return (xf * rstd).to(x.dtype)
```

## 注意事项

- 计算须在 float32 精度下进行（`x.float()`），最终结果 cast 回 `x.dtype`，以保证数值稳定。
- 归一化沿 `dim=-1`，支持任意形状的输入（batch 维度任意）。
- 与 RMSNorm 不同，L2Norm 无可学习权重，也不除以维度数量，是纯粹的单位向量归一化。

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
