<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/chunk_local_cumsum_scalar -->
<!-- synced_at: 2026-09-02T11:21:57+08:00 -->

# chunk_local_cumsum_scalar (fla/chunk_local_cumsum_scalar)

## 任务描述

Per-chunk (not global) cumulative sum of a scalar per-token, per-head gate
value — the FLA-family building block that turns raw log-decay values into
within-chunk cumulative decays for gated linear attention.

## 接口签名

```python
def reference(g, chunk_size, reverse=False, scale=None):
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- Reshape `g` to `(B, NT, BT, H)` (`NT = T // BT` chunks of width `BT =
  chunk_size`); cumsum along the within-chunk axis (reversed first if
  `reverse=True`, then flipped back), scaled by `scale` if given.
- Scope: `head_first=False` layout, `cu_seqlens=None` (fixed-size batching
  only); `T` is always an exact multiple of `chunk_size` (power of 2).

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`


## 参考实现

```python
def reference(g, chunk_size, reverse=False, scale=None):
    B, T, H = g.shape
    BT = chunk_size
    NT = T // BT

    g_c = g.float().view(B, NT, BT, H)
    if reverse:
        g_c = g_c.flip(2)
    out = g_c.cumsum(dim=2)
    if scale is not None:
        out = out * scale
    if reverse:
        out = out.flip(2)

    return out.reshape(B, T, H)
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
