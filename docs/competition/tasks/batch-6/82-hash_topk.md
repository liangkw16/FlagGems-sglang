<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/hash_topk -->
<!-- synced_at: 2026-09-23T13:37:15+08:00 -->

# hash_topk (moe/hash_topk)

## 任务描述

Hash 路由 MoE gate（DeepSeek-V4 风格）：token 的专家集合从预计算的 `tid2eid` 表按 token id 查找，因此没有 top-k 搜索 —— 只有 gather 路由 logits、`sqrt(softplus(.))` 打分、行重归一化，以及追加 fused shared experts。

## 接口签名

```python
def reference(router_logits, input_ids, tid2eid, num_fused_shared_experts, routed_scaling_factor, scoring_func)
```

> 选手实现的函数签名需与上述完全一致。

## 计算定义

- `router_logits`: `[num_tokens, num_routed_experts]`
- `input_ids`: `[num_tokens]`，索引 `tid2eid` 的 token id
- `tid2eid`: `[vocab, topk_routed]` int32 专家表
- `scoring_func` 必须为 `"sqrtsoftplus"`

```
logit_ij  = router_logits[i, tid2eid[token_id_i, j]]
w_ij      = sqrt(softplus(logit_ij))
out_ij    = w_ij / sum_j w_ij                          (routed slots)
out_ij    = 1 / routed_scaling_factor                  (fused shared slots)
id_ij     = tid2eid[token_id_i, j] | num_routed_experts + (j - topk_routed)
```

## 正确性判别标准

Per-dtype tolerance 比较 weights，ids 精确比较（检查函数分别比较两个输出）。

## 参考实现

```python
import torch
import torch.nn.functional as F


def reference(router_logits, input_ids, tid2eid, num_fused_shared_experts, routed_scaling_factor, scoring_func):
    assert scoring_func == "sqrtsoftplus"
    num_tokens, num_routed = router_logits.shape
    topk_routed = tid2eid.shape[1]

    expert_ids = tid2eid[input_ids.long()].long()
    logits = torch.gather(router_logits.float(), 1, expert_ids)
    w = torch.sqrt(F.softplus(logits))
    w = w / w.sum(dim=-1, keepdim=True)

    shared_w = torch.full(
        (num_tokens, num_fused_shared_experts),
        1.0 / routed_scaling_factor,
        dtype=torch.float32,
        device=router_logits.device,
    )
    shared_ids = (
        num_routed
        + torch.arange(num_fused_shared_experts, device=router_logits.device)
    ).expand(num_tokens, -1)

    weights = torch.cat([w, shared_w], dim=-1).float()
    ids = torch.cat([expert_ids.to(torch.int32), shared_ids.to(torch.int32)], dim=-1)
    return weights.contiguous(), ids.contiguous()
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
