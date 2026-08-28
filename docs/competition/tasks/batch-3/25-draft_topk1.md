<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/draft_topk1 -->
<!-- synced_at: 2026-08-29T00:54:30+08:00 -->

# draft_topk1 (speculative/draft_topk1)

## 任务描述

投机解码 Top-1 草稿生成：从下一个 token 的 logit 中选取概率最高的 token 作为草稿候选，返回其位置概率（恒为 1）、token 索引、更新后的位置编号，以及可选的草稿 token 缓冲区。

## 接口签名

```python
def reference(next_token_logits, positions, draft_tokens=None, draft_token_column=0)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

输入：
- `next_token_logits`: `[B, V]` — B 个序列各自下一个位置的 logit，V 为词表大小
- `positions`: `[B]` int64 — 各序列当前位置编号
- `draft_tokens`: `[B, D]` int 或 None — 草稿 token 缓冲区，D 为草稿步长（可选）
- `draft_token_column`: int — 将本步草稿写入 `draft_tokens` 的列索引（默认 0）

计算步骤：

1. Argmax 选择：`topk_index = argmax(next_token_logits, dim=-1, keepdim=True)`，形状 `[B, 1]`，int64
   — 贪心选取 logit 最大的 token，不经 softmax，直接取最大值下标

2. Top-1 概率（恒为 1）：`topk_p = ones([B, 1], dtype=float32)`
   — 贪心策略下 top-1 的验收概率为 1

3. 位置更新：`out_positions = positions + 1`，形状 `[B]`
   — 当前位置向前推进一步

4. 草稿缓冲区更新（当 `draft_tokens` 不为 None）：
   - `out_draft_tokens = draft_tokens.clone()`
   - `out_draft_tokens[:, draft_token_column] = topk_index.squeeze(-1)`
   — 将所选 token 写入草稿缓冲区的指定列

输出：`(topk_p, topk_index, out_positions, out_draft_tokens)`
- 当 `draft_tokens` 为 None 时，`out_draft_tokens` 为 None

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`

`topk_index` 和 `out_draft_tokens` 需与参考实现精确匹配（exact equality）。

## 参考实现

```python
import torch


def reference(next_token_logits, positions, draft_tokens=None, draft_token_column=0):
    bs = next_token_logits.shape[0]
    topk_index = next_token_logits.argmax(dim=-1, keepdim=True).to(torch.int64)
    topk_p = torch.ones(bs, 1, dtype=torch.float32, device=next_token_logits.device)

    out_positions = positions + 1
    out_draft_tokens = None
    if draft_tokens is not None:
        out_draft_tokens = draft_tokens.clone()
        out_draft_tokens[:, draft_token_column] = topk_index.squeeze(-1)

    return topk_p, topk_index, out_positions, out_draft_tokens
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
