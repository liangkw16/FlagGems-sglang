<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/rotary_embedding -->
<!-- synced_at: 2026-09-02T11:21:57+08:00 -->

# rotary_embedding (diffusion/rotary_embedding)

## 任务描述

旋转位置编码（RoPE）应用：将预计算的余弦/正弦位置编码应用到输入 token 特征上，以交错配对方式对相邻维度对执行二维旋转变换，用于扩散模型的位置编码。

## 接口签名

```python
def reference(x, cos, sin, interleaved)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

输入：
- `x`: `[T, H, D]` — T 个 token，H 个注意力头，每头维度 D，bfloat16
- `cos`: `[T, D//2]` — 预计算的余弦值
- `sin`: `[T, D//2]` — 预计算的正弦值
- `interleaved`: bool — 本问题使用 False（半宽非交错形式）

计算步骤（float32 精度下执行）：

1. 拆分偶数/奇数维度：
   - `x1 = x[..., 0::2]`，形状 `[T, H, D//2]`（偶数下标维度）
   - `x2 = x[..., 1::2]`，形状 `[T, H, D//2]`（奇数下标维度）

2. 广播 cos/sin 到 head 维度：
   - `c = cos.reshape(T, 1, D//2)`
   - `s = sin.reshape(T, 1, D//2)`

3. 二维旋转（对每个维度对 `(x1[i], x2[i])` 应用旋转矩阵）：
   ```
   o1 = x1 * c - x2 * s
   o2 = x1 * s + x2 * c
   ```
   对应旋转矩阵：`[[cos, -sin], [sin, cos]]`

4. 交错重组：`out = stack([o1, o2], dim=-1).reshape(T, H, D)`
   — 将 `(o1[..., i], o2[..., i])` 还原为连续的偶奇维度对

输出转换为输入 dtype（bfloat16），形状与 `x` 相同 `[T, H, D]`。

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`

## 参考实现

```python
import torch


def reference(x, cos, sin, interleaved):
    xf = x.float()
    x1 = xf[..., 0::2]
    x2 = xf[..., 1::2]
    c = cos.float().reshape(cos.shape[0], 1, -1)
    s = sin.float().reshape(sin.shape[0], 1, -1)
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    out = torch.stack([o1, o2], dim=-1).reshape(xf.shape)
    return out.to(x.dtype)
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
