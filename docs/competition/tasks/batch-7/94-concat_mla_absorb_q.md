<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/concat_mla_absorb_q -->
<!-- synced_at: 2026-09-24T20:58:34+08:00 -->

# concat_mla_absorb_q (attention/concat_mla_absorb_q)

## 任务描述

沿最后一维拼接 absorbed-Q 的 NoPE 与 RoPE 两半：`out = cat([a, b], dim=-1)`。
关键点在存储模式：两个不同 stride 的源行喂给一个连续的 dest 行，MLA head 数规模。

## 接口签名

```python
def reference(a, b)
```

> 选手实现的函数签名需与上述完全一致。

## 计算定义

- `a`：`[dim0, dim1, a_last]` bf16；`b`：`[dim0, dim1, b_last]` bf16。
- 输出：`[dim0, dim1, a_last + b_last]` bf16。
- 计算流程：

  ```
  out = torch.cat([a, b], dim=-1)
  ```

## 正确性判别标准

精确（纯数据搬移）。

## 参考实现

```python
import torch


def reference(a, b):
    return torch.cat([a, b], dim=-1)
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
