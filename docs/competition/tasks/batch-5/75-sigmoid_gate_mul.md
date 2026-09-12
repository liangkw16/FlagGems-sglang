<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/sigmoid_gate_mul -->
<!-- synced_at: 2026-09-12T20:28:08+08:00 -->

# sigmoid_gate_mul (elementwise/sigmoid_gate_mul)

## 任务描述

逐元素门控乘，同形操作数：`out = x * sigmoid(gate)`。
kernel 是对 `x.numel()` 的扁平 1D 扫描，只要两个张量连续，任意 shape 均可。

## 接口签名

```python
def reference(x, gate)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- `x` 与 `gate` 同 shape 同 dtype。
- 计算流程：

  ```
  out = x.float() * sigmoid(gate.float())
  ```

  fp32 计算，cast 回 `x.dtype` 存储。

## 正确性判别标准

标准 per-dtype tolerance。

## 参考实现

```python
import torch


def reference(x, gate):
    return (x.float() * torch.sigmoid(gate.float())).to(x.dtype)
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
