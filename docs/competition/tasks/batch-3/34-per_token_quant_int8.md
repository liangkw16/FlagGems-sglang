<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/per_token_quant_int8 -->
<!-- synced_at: 2026-08-29T00:54:30+08:00 -->

# per_token_quant_int8 (quantization/per_token_quant_int8)

## 任务描述

逐 token（整行）INT8 量化：对输入矩阵的每一行计算一个基于绝对最大值的缩放因子，将该行量化为 int8 整数；输出量化后的整数矩阵和每行一个的 float32 缩放因子向量。

## 接口签名

```python
def reference(x)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- 输入 `x`: `[M, N]`，任意浮点 dtype，连续存储
- 输出：`(x_q: [M, N] int8, x_s: [M, 1] float32)`
- 对每行 `row`：
  - `scale = max(|x[row]|, 1e-10) / 127`
  - `x_q[row] = clamp(round(x[row] / scale), -128, 127)`
- 等价于以整行作为一个 group 调用 per-token-group 量化（`group_size = N`）

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`

## 参考实现

```python
from flaggems_reference.per_token_group_quant_int8 import reference as _group_reference


def reference(x):
    return _group_reference(x, group_size=x.shape[-1])
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
