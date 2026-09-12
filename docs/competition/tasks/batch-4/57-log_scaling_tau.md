<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/log_scaling_tau -->
<!-- synced_at: 2026-09-12T20:28:08+08:00 -->

# log_scaling_tau (attention/log_scaling_tau)

## 任务描述

对输入张量 `x` 的每一行乘以对应的标量 `tau[t]`，在 float32 下计算后 cast 回 `x.dtype`。该操作用于在一次 launch 中将对数空间的注意力缩放因子 `tau` 应用到融合 QKV 投影输出的 query 切片，替代独立的广播乘法算子。SGLang baseline 对应 `sglang.kernels.ops.attention.log_scaling_tau.apply_log_scaling_tau`。

## 接口签名

```python
def log_scaling_tau(x, tau):
```

> 选手实现的函数签名需与上述 `log_scaling_tau(...)` 完全一致。

## 计算定义

对每一行 `t` 独立缩放，`tau[t]` 广播到该行的所有元素：

$$	ext{out}[t, ldots] = left(x[t, ldots]_{	ext{float32}} 	imes 	au[t]ight) 	o 	ext{x.dtype}$$

等价的逐行向量形式：

$$	ext{out} = mathbf{x}_{	ext{float32}} odot oldsymbol{	au}_{	ext{reshape}} 	o 	ext{x.dtype}$$

其中 $oldsymbol{	au}_{	ext{reshape}} in mathbb{R}^{T 	imes 1 	imes cdots 	imes 1}$，广播覆盖所有尾部维度。

## 输入输出规格

| 参数 | 形状 | dtype | 说明 |
|------|------|-------|------|
| `x` | `[T, ...]` | float16 / bfloat16 / float32 | 主输入张量，支持任意尾部维度 |
| `tau` | `[T]` | float32 | 每行的缩放标量 |
| **output** | `[T, ...]` | 同 `x` | 缩放后的结果，形状与 `x` 相同 |

## 测试用例

| T | 尾部形状 | dtype | 说明 |
|---|---------|-------|------|
| 4 | `[64]` | float16 | 基本一维尾部 |
| 8 | `[128]` | float16 | 典型 query 头维度 |
| 16 | `[64, 128]` | float16 | 二维尾部 |
| 32 | `[256]` | float16 | 较大行尺寸 |
| 4 | `[64]` | bfloat16 | bf16 快速路径覆盖（8 对齐） |

正确性判别：fp16 default，`atol=1e-2, rtol=1e-2`。

## 参考实现

```python
import torch


def reference(x, tau):
    rows = x.shape[0]
    tau_r = tau.reshape(rows).float()
    shape = [rows] + [1] * (x.dim() - 1)
    return (x.float() * tau_r.view(shape)).to(x.dtype)
```

## 注意事项

- 计算须在 float32 精度下进行（`x.float()`），最终 cast 回 `x.dtype`。
- `tau` 需 reshape 为 `[T, 1, ..., 1]` 后才能正确广播到所有尾部维度。
- 实际实现中存在一条 bf16 快速路径（`row_scale_bf16`），要求 16 字节对齐且 `inner % 8 == 0`，与标量 Triton kernel 输出位完全相同；测试用例使用 float16 以覆盖通用路径，两条路径实现相同的数学公式。

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
