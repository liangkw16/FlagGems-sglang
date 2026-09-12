<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/clamp_position -->
<!-- synced_at: 2026-09-12T20:28:08+08:00 -->

# clamp_position (attention/clamp_position)

## 任务描述

Decode 步位置计算：`positions = clamp(seq_lens - 1, min=0)`。
算术上极便宜，但每个 forward 在关键路径上启动一次，因此是纯 launch 开销 / 内存延迟问题。

## 接口签名

```python
def reference(seq_lens)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- `seq_lens`：1D int32 或 int64 CUDA 张量；输出同 shape 同 dtype。
- 计算流程：

  ```
  out = (seq_lens - 1).clamp(min=0)
  ```

## 正确性判别标准

精确（整数运算）。

## 参考实现

```python
def reference(seq_lens):
    return (seq_lens - 1).clamp(min=0)
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
