<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/fused_moe_dispatch_index -->
<!-- synced_at: 2026-09-12T20:28:08+08:00 -->

# fused_moe_dispatch_index (moe/fused_moe_dispatch_index)

## 任务描述

一次 launch 构建 masked/DeepGEMM MoE 的置换 grouped-GEMM 目标索引：用 atomic cursor 把每个 `(token, slot)` 对按 expert 分桶，把 per-expert offset 变成 flat destination row。

## 接口签名

```python
def reference(topk_ids, num_local_experts, m_max)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- `topk_ids`: `[num_tokens, top_k]` int32；`-1` 标记 padding（跳过）
- `masked_m`: `[num_local_experts]` int32；`src2dst`: `[num_tokens * top_k]` int32

```
offset      = atomic_add(masked_m[expert], 1)
src2dst[i]  = expert * m_max + offset
masked_m[e] = number of pairs routed to expert e
```

## 正确性判别标准

Exact（整数），但桶内顺序来自 atomics 不确定：`masked_m` 精确比较，`src2dst` 按排序后的 multiset 比较（所有 case 都路由每个 slot，因此每个条目都被写入）。

## 参考实现

```python
import torch


def reference(topk_ids, num_local_experts, m_max):
    flat = topk_ids.reshape(-1)
    device = flat.device
    masked_m = torch.zeros(num_local_experts, dtype=torch.int32, device=device)
    src2dst = torch.empty(flat.numel(), dtype=torch.int32, device=device)
    counts = [0] * num_local_experts
    flat_cpu = flat.tolist()
    dst = []
    for e in flat_cpu:
        if e < 0:
            dst.append(0)
            continue
        dst.append(e * m_max + counts[e])
        counts[e] += 1
    src2dst.copy_(torch.tensor(dst, dtype=torch.int32, device=device))
    masked_m.copy_(torch.tensor(counts, dtype=torch.int32, device=device))
    return masked_m, src2dst
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
