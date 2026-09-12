<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/fused_dual_residual_rmsnorm -->
<!-- synced_at: 2026-09-12T20:28:08+08:00 -->

# fused_dual_residual_rmsnorm (activation_norm/fused_dual_residual_rmsnorm)

## 任务描述

对输入张量 `x` 先做 RMSNorm（使用 `weight1`），将结果加到 `residual` 上得到中间张量 `mid`，再对 `mid` 做第二次 RMSNorm（使用 `weight2`）得到最终输出。两次 RMSNorm 均在 float32 精度下计算，结果分别 cast 回 `residual` 和 `x` 的原始 dtype。SGLang baseline 对应 `sglang.kernels.ops.elementwise.elementwise.fused_dual_residual_rmsnorm`（`autotune=False`）。

## 接口签名

```python
def fused_dual_residual_rmsnorm(x, residual, weight1, weight2, eps):
```

> 选手实现的函数签名需与上述 `fused_dual_residual_rmsnorm(...)` 完全一致。

## 计算定义

单行 RMSNorm 辅助定义（在 float32 下计算）：

$$operatorname{rms}(mathbf{v}) = sqrt{frac{1}{H}sum_{i=1}^{H} v_i^2 + epsilon}$$

$$operatorname{RMSNorm}(mathbf{v}, mathbf{w}, epsilon) = frac{mathbf{v}}{operatorname{rms}(mathbf{v})} odot mathbf{w}$$

整体计算流程：

$$	ext{mid} = 	ext{residual} + operatorname{RMSNorm}(mathbf{x}, mathbf{w}_1, epsilon)ig|_{	ext{cast to residual.dtype}}$$

$$	ext{output} = operatorname{RMSNorm}(	ext{mid}, mathbf{w}_2, epsilon)ig|_{	ext{cast to x.dtype}}$$

## 输入输出规格

| 参数 | 形状 | dtype | 说明 |
|------|------|-------|------|
| `x` | `[bs, hidden]` | float16 / bfloat16 / float32 | 主输入张量 |
| `residual` | `[bs, hidden]` | 同 `x` | 残差张量 |
| `weight1` | `[hidden]` | 同 `x` | 第一个 RMSNorm 的可学习权重 |
| `weight2` | `[hidden]` | 同 `x` | 第二个 RMSNorm 的可学习权重 |
| `eps` | 标量 | float | 数值稳定项，典型值 1e-5 或 1e-6 |
| **output** | `[bs, hidden]` | 同 `x` | 第二次 RMSNorm 的结果 |
| **mid** | `[bs, hidden]` | 同 `residual` | 第一次 RMSNorm + residual 的中间结果 |

## 测试用例

| bs | hidden | dtype | eps |
|----|--------|-------|-----|
| 1 | 4096 | bfloat16 | 1e-5 |
| 4 | 4096 | bfloat16 | 1e-5 |
| 16 | 8192 | bfloat16 | 1e-6 |
| 32 | 2048 | float16 | 1e-5 |
| 1 | 1024 | float32 | 1e-6 |

正确性判别：对 `output` 和 `mid` 两个输出均做 per-dtype tolerance 检查。
- float32：`atol=1e-4, rtol=1e-4`
- bfloat16：`atol=1.5e-2, rtol=1.5e-2`
- float16：`atol=1e-2, rtol=1e-2`

## 参考实现

```python
import torch


def _rmsnorm32(v32, w32, eps):
    rms = torch.sqrt((v32 * v32).mean(dim=-1, keepdim=True) + eps)
    return v32 / rms * w32


def reference(x, residual, weight1, weight2, eps):
    mid = residual + _rmsnorm32(x.float(), weight1.float(), eps).to(residual.dtype)
    out = _rmsnorm32(mid.float(), weight2.float(), eps).to(x.dtype)
    return out, mid
```

## 注意事项

- 两次 RMSNorm 的内部计算均须在 float32 精度下进行，最终结果再 cast 回目标 dtype，以保证数值稳定性。
- `mid` cast 目标为 `residual.dtype`，`output` cast 目标为 `x.dtype`，二者可能不同。
- 函数返回顺序为 `(output, mid)`，不可颠倒。

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
