<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/topk_sigmoid -->
<!-- synced_at: 2026-09-26T22:45:04+08:00 -->

# topk_sigmoid (moe/topk_sigmoid)

## 任务描述

Fused sigmoid MoE gate：对 router logits 做 sigmoid，取 top-k experts，可选 renormalize。内核是 destination-passing：原地写预分配的 `topk_weights`（fp32）与 `topk_ids`（int32）。

## 接口签名

```python
def reference(topk_weights, topk_ids, gating_output, renormalize, routed_scaling_factor)
```

> 选手实现的函数签名需与上述完全一致。

## 计算定义

- `gating_output`: `[num_tokens, num_experts]` fp32/fp16/bf16
- `correction_bias` 与 `num_fused_shared_experts` 融合不在范围内

```
scores = sigmoid(gating_output.float())
weights, ids = topk(scores, k)                  # descending
if renormalize:
    weights = weights * routed_scaling_factor / (weights.sum(-1, keepdim=True) + 1e-20)
```

## 正确性判别标准

Per-dtype tolerance 比较 weights，ids 精确比较（检查函数分别比较两个输出）；tie 不在契约内（case 用 fp32 logits，一行内没有两个专家得分相等）。

## 参考实现

```python
import torch


def reference(topk_weights, topk_ids, gating_output, renormalize, routed_scaling_factor):
    k = topk_weights.shape[1]
    scores = torch.sigmoid(gating_output.float())
    w, ids = torch.topk(scores, k, dim=-1)
    if renormalize:
        w = w * routed_scaling_factor / (w.sum(dim=-1, keepdim=True) + 1e-20)
    return w.contiguous(), ids.to(torch.int32).contiguous()
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
