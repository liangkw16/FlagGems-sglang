<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/pre_reorder_cutlass -->
<!-- synced_at: 2026-09-26T22:45:04+08:00 -->

# pre_reorder_cutlass (moe/pre_reorder_cutlass)

## 任务描述

CUTLASS-MoE 输入置换 + 融合的 per-tensor dequant scale：把每个 token 行复制到它的路由目标，途中按 `1 / a1_scales` 缩放。

## 接口签名

```python
def reference(input, gateup_input, src2dst, topk_ids, a1_scales, num_local_experts, topk, num_tokens, hidden_size)
```

> 选手实现的函数签名需与上述完全一致。

## 计算定义

- `a1_scales` 可为 `None`（scale 1.0）或 1 元素 fp32 张量
- `topk_ids[t, i] == num_local_experts` 标记路由到本 EP rank 之外的 slot，跳过

```
for each token t, for each slot i with topk_ids[t, i] != num_local_experts:
    gateup_input[src2dst[t, i], :] = (input[t, :] * (1 / a1_scales)).to(out_dtype)
```

## 正确性判别标准

Per-dtype tolerance（standard `harness.correctness`）。

## 参考实现

```python
def reference(
    input,
    gateup_input,
    src2dst,
    topk_ids,
    a1_scales,
    num_local_experts,
    topk,
    num_tokens,
    hidden_size,
):
    out = gateup_input.clone()
    inv = 1.0 / float(a1_scales) if a1_scales is not None else 1.0
    scaled = (input.float() * inv).to(out.dtype)
    flat_ids = topk_ids.reshape(-1)
    flat_dst = src2dst.reshape(-1)
    valid = flat_ids != num_local_experts
    src_rows = scaled.repeat_interleave(topk, dim=0)
    out[flat_dst[valid].long()] = src_rows[valid]
    return out
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
