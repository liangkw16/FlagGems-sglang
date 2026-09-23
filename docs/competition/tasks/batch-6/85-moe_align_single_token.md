<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/moe_align_single_token -->
<!-- synced_at: 2026-09-23T17:49:06+08:00 -->

# moe_align_single_token (moe/moe_align_single_token)

## 任务描述

`moe_align_block_size` 的 `M == 1` decode 特化：只有一个 token，其 `top_k` 个 expert 互异，每个 expert 独占一个 block。单个 warp 即可完成全部工作。

## 接口签名

```python
def reference(topk_ids, block_size)
```

> 选手实现的函数签名需与上述完全一致。

## 计算定义

- `topk_ids`: `[1, top_k]` int32，**expert id 互异**
- `sorted_token_ids`: `[top_k * block_size]` int32，`expert_ids`: `[top_k]` int32

```
sorted_token_ids = [0, pad, pad, ..., pad]  per expert block, pad = topk_ids.numel()
expert_ids       = the token's experts, ascending
num_tokens_post_padded = top_k * block_size
```

## 正确性判别标准

Exact（整数）；检查函数逐元素比较 `(sorted_token_ids, expert_ids, num_tokens_post_padded)` 三个输出。

## 参考实现

```python
import torch


def reference(topk_ids, block_size):
    topk = topk_ids.shape[1]
    numel = topk_ids.numel()
    device = topk_ids.device
    sorted_ids = torch.full(
        (topk * block_size,), numel, dtype=torch.int32, device=device
    )
    experts = torch.sort(topk_ids.flatten()).values.to(torch.int32)
    order = torch.argsort(topk_ids.flatten())
    for slot in range(topk):
        sorted_ids[slot * block_size] = int(order[slot])
    num_post = torch.full((1,), topk * block_size, dtype=torch.int32, device=device)
    return sorted_ids, experts, num_post
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
