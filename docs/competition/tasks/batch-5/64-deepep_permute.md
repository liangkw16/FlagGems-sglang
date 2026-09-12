<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/deepep_permute -->
<!-- synced_at: 2026-09-12T20:28:08+08:00 -->

# deepep_permute (moe/deepep_permute)

## 任务描述

DeepEP MoE gather/scatter prologue：把每个 token 的 hidden 行复制到置换后的 grouped-GEMM 输入中，每个路由 slot 一份。每个源 token 一个 program，hidden 维按 `BLOCK_SIZE` 分块循环，把同一 tile 重新写到 `topk` 个目标位置。

## 接口签名

```python
def reference(input, gateup_input, src2dst, topk_ids, topk, hidden_size)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- `input`: `[num_tokens, hidden_size]`，`gateup_input`: `[num_tokens * topk, hidden_size]`
- `src2dst`: `[num_tokens, topk]` int32 目标行，`-1` 表示跳过

```
for each token t, for each slot i with src2dst[t, i] >= 0:
    gateup_input[src2dst[t, i], :] = input[t, :]
```

## 正确性判别标准

Exact（纯数据搬运，逐元素精确相等）。

## 参考实现

```python
def reference(input, gateup_input, src2dst, topk_ids, topk, hidden_size):
    out = gateup_input.clone()
    flat_dst = src2dst.reshape(-1)
    valid = flat_dst >= 0
    src_rows = (
        input.repeat_interleave(topk, dim=0).to(out.dtype)
    )
    out[flat_dst[valid].long()] = src_rows[valid]
    return out
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
