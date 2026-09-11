<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/fill_padded_rows -->
<!-- synced_at: 2026-09-12T00:03:19+08:00 -->

# fill_padded_rows (moe/fill_padded_rows)

## 任务描述

把路由输出的每个 padding 行设为常数：`x[row, :] = fill_value`，其中 `row >= num_token_non_padded`。pad 数量位于 **device** 内存中且在内核内读取，grid 是静态的（每行一个 program），整个流程可被 CUDA graph 捕获 —— 这是该内核的意义所在。

## 接口签名

```python
def reference(x, num_token_non_padded, fill_value)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- `x`: 2D 张量（`x.stride(1) == 1`）
- `num_token_non_padded`: 单元素整数 CUDA 张量
- 若 `num_token_non_padded < x.shape[0]`，则 `x[num_token_non_padded:] = fill_value`

## 正确性判别标准

Exact（整数/常数填充，逐元素精确相等）。

## 参考实现

```python
import torch


def reference(x, num_token_non_padded, fill_value):
    out = x.clone()
    n = int(num_token_non_padded)
    if n < out.shape[0]:
        out[n:] = fill_value
    return out
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
