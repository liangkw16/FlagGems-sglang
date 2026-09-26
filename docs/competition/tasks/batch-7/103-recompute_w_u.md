<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/recompute_w_u -->
<!-- synced_at: 2026-09-26T22:45:04+08:00 -->

# recompute_w_u (fla/recompute_w_u)

## 任务描述

chunked Gated DeltaNet 前向的 `w` / `u` 重计算步骤。给定每个 chunk 已求逆的 UT 变换矩阵 `A`，将其作用于 beta 缩放的 value 块与（gate 缩放的）key 块，重算出 `w` 与 `u`。

## 接口签名

```python
def reference(k, v, beta, g_cumsum, A, cu_seqlens)
```

> 选手实现的函数签名需与上述完全一致。

## 计算定义

- `k`: `[B, T, Hg, K]`，`v`: `[B, T, H, V]`，`beta` / `g_cumsum`: `[B, T, H]`，`A`: `[B, T, H, BT]`；`k`、`v`、`A` 为 bf16，`beta` 与 `g_cumsum` 为 fp32。
- `cu_seqlens` 此处为 `None`（varlen 路径不在范围内），`T` 是 `BT` 的倍数，不存在部分掩码的 chunk。
- 对每个 chunk `c`（`BT` 个 token）、batch `b`、head `h`：

```
Ac = A[b, rows_c, h, :]                                   # [BT, BT]
u[b, rows_c, h, :] = Ac @ (v[b, rows_c, h, :] * beta[b, rows_c, h])
w[b, rows_c, h, :] = Ac @ (k[b, rows_c, hk, :] * beta[b, rows_c, h] * exp(g_cumsum[b, rows_c, h]))
```

- GQA 映射：`hk = h // (H // Hg)`，即 `k` 只带 `Hg` 个 key-value head，而 `v`、`beta`、`g_cumsum` 带 `H` 个 query head。
- 返回 `w` `[B, T, H, K]` 与 `u` `[B, T, H, V]`。

## 正确性判别标准

Per-dtype tolerance：
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`

检查函数逐元素比较 `(w, u)` 两个输出。

## 参考实现

```python
import torch


def reference(k, v, beta, g_cumsum, A, cu_seqlens):
    assert cu_seqlens is None, "varlen path is out of scope"
    B, T, Hg, K = k.shape
    H, V = v.shape[-2], v.shape[-1]
    BT = A.shape[-1]
    heads_per_kv = H // Hg

    w = k.new_empty(B, T, H, K)
    u = torch.empty_like(v)

    for c in range(T // BT):
        rows = slice(c * BT, (c + 1) * BT)
        # [B, H, BT, BT]
        Ac = A[:, rows].permute(0, 2, 1, 3).float()
        b_beta = beta[:, rows].permute(0, 2, 1).float()          # [B, H, BT]
        b_g = torch.exp(g_cumsum[:, rows].permute(0, 2, 1).float())

        vb = v[:, rows].permute(0, 2, 1, 3).float() * b_beta[..., None]
        u[:, rows] = (Ac @ vb.to(v.dtype).float()).permute(0, 2, 1, 3).to(v.dtype)

        kh = k[:, rows].permute(0, 2, 1, 3).float()              # [B, Hg, BT, K]
        kh = kh.repeat_interleave(heads_per_kv, dim=1)           # [B, H, BT, K]
        kb = kh * b_beta[..., None] * b_g[..., None]
        w[:, rows] = (Ac @ kb.to(k.dtype).float()).permute(0, 2, 1, 3).to(k.dtype)

    return w, u
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
