<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/pack_topk_ids -->
<!-- synced_at: 2026-09-26T22:45:04+08:00 -->

# pack_topk_ids (moe/pack_topk_ids)

## 任务描述

把每个 `(expert_id, weight)` 路由对打包进一个 int32 —— 这是 FlashInfer TRT-LLM routed-MoE 路径消费的布局。

## 接口签名

```python
def reference(topk_ids, topk_weights)
```

> 选手实现的函数签名需与上述完全一致。

## 计算定义

- `topk_ids`: int32，连续；`topk_weights`: float32，同 shape，连续
- 输出：int32，同 shape

```
out = (topk_ids.to(int32) << 16) | (topk_weights.to(bf16).view(int16).to(int32) & 0xFFFF)
```

bf16 截断是契约的一部分：低 16 位是 bf16 位模式，而不是舍入后的定点值。

## 正确性判别标准

Exact（整数）。

## 参考实现

```python
import torch


def reference(topk_ids, topk_weights):
    weight_bits = (
        topk_weights.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
    )
    return (topk_ids.to(torch.int32) << 16) | weight_bits
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
