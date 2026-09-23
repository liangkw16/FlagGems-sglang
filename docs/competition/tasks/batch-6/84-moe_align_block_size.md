<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/moe_align_block_size -->
<!-- synced_at: 2026-09-23T13:37:15+08:00 -->

# moe_align_block_size (moe/moe_align_block_size)

## 任务描述

每个 block-tiled fused-MoE GEMM 都依赖的 MoE sort/pad。给定 `topk_ids[num_tokens, top_k]`，把展平的 `(token, slot)` 对按 expert 分组，把每个 expert 的 run pad 到 `block_size` 的倍数，输出：
- `sorted_token_ids`：按 expert 顺序排列的展平 pair id，以 sentinel `topk_ids.numel()` 填充；
- `expert_ids`：每个 `block_size` 大小 block 的 expert id；
- `num_tokens_post_pad`：padding 后的总长度。

## 接口签名

```python
def reference(topk_ids, num_experts, block_size, sorted_token_ids, expert_ids, num_tokens_post_pad, cumsum_buffer, pad_sorted_token_ids)
```

> 选手实现的函数签名需与上述完全一致。

## 计算定义

- `topk_ids` int32 `[num_tokens, top_k]`
- `num_experts` 传的是模型 routed-expert 数 **加一**：末尾 slot 是 EP "filtered expert" bucket，不收 token 也不占 block。这与 `moe_runner.triton_utils` 的调用方式一致
- `cumsum_buffer` 是 `[num_experts + 1]` int32 的 caller 持有的 scratch
- `pad_sorted_token_ids=True` 时整个 `sorted_token_ids` buffer 预填 sentinel
- `expert_ids` 超过 `num_tokens_post_pad // block_size` 的条目未定义，检查只比较已定义前缀

对每个 routed expert `e`：
```
idx     = nonzero(flat == e)                # 展平 topk_ids 中属于 e 的下标
n       = idx.numel()
aligned = ceil(n / block_size) * block_size
sorted_ids[offset : offset + n] = idx
eids[offset // block_size : ...] = e
offset += aligned
```

## 正确性判别标准

Exact（整数），但 expert 内顺序不确定（atomic cursor 放置）：检查把每个 expert 的范围按 **multiset** 比较；`num_tokens_post_pad` 之后尾部必须保持 padding sentinel。

## 参考实现

```python
import torch


def reference(topk_ids, num_experts, block_size, sorted_token_ids, expert_ids, num_tokens_post_pad, cumsum_buffer, pad_sorted_token_ids):
    # ``num_experts`` counts the trailing "filtered expert" slot, which never
    # receives tokens and therefore gets no blocks; only 0..num_experts-2 do.
    num_routed = num_experts - 1
    flat = topk_ids.flatten()
    numel = flat.numel()
    sorted_ids = torch.full_like(sorted_token_ids, numel)
    eids = expert_ids.clone()

    offset = 0
    for e in range(num_routed):
        idx = (flat == e).nonzero().flatten()
        n = idx.numel()
        aligned = ((n + block_size - 1) // block_size) * block_size
        if n:
            sorted_ids[offset : offset + n] = idx.to(sorted_ids.dtype)
        nblocks = aligned // block_size
        if nblocks:
            beg = offset // block_size
            eids[beg : beg + nblocks] = e
        offset += aligned

    npost = torch.full_like(num_tokens_post_pad, offset)
    return sorted_ids, eids, npost
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
