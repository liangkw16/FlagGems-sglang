<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/compute_src2dst -->
<!-- synced_at: 2026-09-12T00:03:19+08:00 -->

# compute_src2dst (moe/compute_src2dst)

## 任务描述

反转路由置换：给定 `topk_ids` 展平后稳定排序产生的 `reorder_ids`，写出每个源对的 destination slot。纯 scatter —— 每个元素一次 int32 load + 一次分散 int32 store，共 `num_tokens * top_k` 个元素。

## 接口签名

```python
def reference(reorder_ids, num_toks)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- `reorder_ids`: `[num_toks]` int64（`argsort` 的输出）
- 输出：`[num_toks]` int32

```
src2dst[reorder_ids[d]] = d
```

## 正确性判别标准

Exact（整数），逐元素精确相等。

## 参考实现

```python
import torch


def reference(reorder_ids, num_toks):
    src2dst = torch.empty(num_toks, dtype=torch.int32, device=reorder_ids.device)
    dst = torch.arange(num_toks, dtype=torch.int32, device=reorder_ids.device)
    src2dst[reorder_ids.long()] = dst
    return src2dst
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
