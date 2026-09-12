<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/residual_gate_add -->
<!-- synced_at: 2026-09-12T20:28:08+08:00 -->

# residual_gate_add (diffusion/residual_gate_add)

## 任务描述

LTX-2 transformer block 的逐元素 residual/gate 更新：`out = residual + T(update * gate)`。

## 接口签名

```python
def reference(residual, update, gate)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- 中间乘积先 round 到张量 dtype `T` 再做加法——双重舍入是契约的一部分，匹配未融合的 eager 序列
- `gate` 与 `residual` 同 shape，或为 row-broadcast `[1, ..., 1, D]`
- 所有张量 contiguous、同 dtype（fp16/bf16/fp32）、同 device；`residual.dim() >= 2`

## 正确性判别标准

标准 per-dtype tolerance

## 参考实现

```python
def reference(residual, update, gate):
    product = (update.float() * gate.float()).to(residual.dtype)
    return (residual.float() + product.float()).to(residual.dtype)
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
