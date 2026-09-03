<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/chunk_scaled_dot_kkt -->
<!-- synced_at: 2026-09-03T20:42:35+08:00 -->

# chunk_scaled_dot_kkt (fla/chunk_scaled_dot_kkt)

## 任务描述

Delta-rule 内 chunk 修正矩阵的构建算子：对每个 chunk 逐 head 计算 `beta * K @ K.T`，施加严格下三角掩码，并可选地按 `g_cumsum` 进行衰减缩放。输出用于后续 `solve_tril` 求解。本题仅覆盖固定 batch 路径（`cu_seqlens=None`，`T` 整除 `chunk_size`）；`H`（beta 的 head 数）可为 `Hg`（k 的 head 数）的整数倍（GQA 共享）。

## 接口签名

```python
def chunk_scaled_dot_kkt(k, beta, g_cumsum=None, chunk_size=64):
```

> 选手实现的函数签名需与上述 `chunk_scaled_dot_kkt(...)` 完全一致。

## 计算定义

- `k`: `[B, T, Hg, K]`；`beta`: `[B, T, H]`；`g_cumsum`: `[B, T, H]` 或 None；`chunk_size = BT`，`NT = T // BT`
- 将 `k` 在 head 维上 repeat `ratio = H // Hg` 倍，reshape 为 `[B, NT, H, BT, K]`
- 对每个 chunk：`A[b,n,h,i,j] = k[b,n,h,i,:] · k[b,n,h,j,:]`（点积）
- 若 `g_cumsum` 不为 None：`A[i,j] *= exp(g_cumsum[i] - g_cumsum[j])` 当指数 `<= 0`，否则置 `0`（safe-exp 保护）
- 乘以 `beta`：`A[b,n,h,i,j] *= beta[b,n,h,i]`
- 严格下三角掩码：`i > j` 保留，`i <= j` 置 `0`
- 输出 reshape 为 `[B, T, H, BT]`，dtype 为 float32

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`


## 参考实现

```python
import torch


def reference(k, beta, g_cumsum=None, chunk_size=64):
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    ratio = H // Hg
    BT = chunk_size
    NT = T // BT

    k_c = k.float().view(B, NT, BT, Hg, K)
    k_c = k_c.repeat_interleave(ratio, dim=3)  # (B, NT, BT, H, K)
    k_c = k_c.permute(0, 1, 3, 2, 4)  # (B, NT, H, BT, K)

    A = torch.einsum("bnhik,bnhjk->bnhij", k_c, k_c)

    if g_cumsum is not None:
        g_c = g_cumsum.float().view(B, NT, BT, H).permute(0, 1, 3, 2)  # (B, NT, H, BT)
        g_diff = g_c.unsqueeze(-1) - g_c.unsqueeze(-2)
        A = A * torch.where(g_diff <= 0, torch.exp(g_diff), torch.zeros_like(g_diff))

    beta_c = beta.float().view(B, NT, BT, H).permute(0, 1, 3, 2)  # (B, NT, H, BT)
    A = A * beta_c.unsqueeze(-1)

    causal = torch.tril(torch.ones(BT, BT, dtype=torch.bool, device=k.device), diagonal=-1)
    A = torch.where(causal, A, torch.zeros_like(A))

    # (B, NT, H, BT_i, BT_j) -> (B, NT, BT_i, H, BT_j) -> (B, T, H, BT)
    out = A.permute(0, 1, 3, 2, 4).reshape(B, T, H, BT)
    return out
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
