<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/gelu_and_mul -->
<!-- synced_at: 2026-09-01T13:44:50+08:00 -->

# gelu_and_mul (activation_norm/gelu_and_mul)

## 任务描述

门控 GELU 激活：将输入在最后一维对半分为 gate 和 up 两部分，对 gate 施加精确（erf-based）GELU 激活后与 up 逐元素相乘，输出维度为输入的一半。

## 接口签名

```python
def reference(hidden_states)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- 输入 `hidden_states`: `[bs, 2*d]`，任意浮点 dtype
- 输出：`[bs, d]`，与输入同 dtype
- 令 `d = hidden_states.shape[-1] // 2`：
  - `x1 = hidden_states[..., :d]`（gate 部分）
  - `x3 = hidden_states[..., d:]`（up 部分）
  - `out = gelu(x1.float(), approximate="none") * x3.float()`，转回输入 dtype
- GELU 使用精确 erf 公式：`gelu(x) = x * Φ(x) = x * (1 + erf(x / sqrt(2))) / 2`，不使用 tanh 近似

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`

## 参考实现

```python
import torch.nn.functional as F


def reference(hidden_states):
    d = hidden_states.shape[-1] // 2
    x1, x3 = hidden_states[..., :d], hidden_states[..., d:]
    out = F.gelu(x1.float(), approximate="none") * x3.float()
    return out.to(hidden_states.dtype)
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
