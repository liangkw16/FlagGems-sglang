<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/gate_topk -->
<!-- synced_at: 2026-09-12T20:28:08+08:00 -->

# gate_topk (moe/gate_topk)

## 任务描述

面向小 `k` 的流式 top-k（`k <= 32`），是 `torch.topk(x, k, dim=-1)` 的稳定替代。内核把 `(value, index)` 打包成一个可排序 key，使整个 top-k 状态保持在寄存器中，并按 `BLOCK_SIZE_N = 32` 列块流式处理行 —— 当 `k` 远小于专家数时优于通用排序。

## 接口签名

```python
def reference(x, k)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- `x` 必须是 2D 连续张量，`x.numel() <= 2**31`
- `values`: `[M, k]`，dtype 同 `x`；`indices`: `[M, k]` **int32**
- 结果降序，tie-break 取较小列下标

```
values, indices = torch.topk(x, k, dim=-1, sorted=True)
```

## 正确性判别标准

Per-dtype tolerance 比较 values，indices 精确比较（检查函数分别比较两个输出）。

## 参考实现

```python
import torch


def reference(x, k):
    values, indices = torch.topk(x, k, dim=-1, sorted=True)
    return values, indices.to(torch.int32)
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
