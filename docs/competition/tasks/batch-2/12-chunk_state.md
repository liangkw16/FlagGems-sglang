<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/chunk_state -->
<!-- synced_at: 2026-08-23T21:59:34+08:00 -->

# chunk_state (mamba/chunk_state)

## 任务描述

Mamba SSM chunk-state computation: computes per-chunk hidden states by accumulating `x * B * dt * exp(decay)` via einsum.

## 接口签名

```python
def reference(B, x, dt, dA_cumsum)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。
> 注意：参数 `B` 是 SSM 的状态投影矩阵（shape `[batch, seqlen, ngroups, dstate]`），不是 batch size。

## 计算定义

- `B`（SSM矩阵）: `[batch, seqlen, ngroups, dstate]`
- `x`: `[batch, seqlen, nheads, headdim]`
- `dt`: `[batch, nheads, nchunks, chunk_size]`
- `dA_cumsum`: `[batch, nheads, nchunks, chunk_size]`
- 计算 decay: `exp(dA_cumsum[..., -1:] - dA_cumsum)`
- scale = decay * dt
- 将 x 和 B 切块后做 einsum: `states = einsum("bcthp,bcthn->bchpn", x_c, B_scaled)`
- 输出: `[batch, nchunks, nheads, headdim, dstate]`

## 正确性判别标准

`atol=3e-2, rtol=3e-2`.


## 参考实现

```python
import torch


def reference(B, x, dt, dA_cumsum):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    ratio = nheads // ngroups

    x_c = x.reshape(batch, nchunks, chunk_size, nheads, headdim).float()
    B_c = B.reshape(batch, nchunks, chunk_size, ngroups, dstate).float()
    B_c = B_c.repeat_interleave(ratio, dim=3)

    dA_last = dA_cumsum[..., -1:].float()
    decay = torch.exp(dA_last - dA_cumsum.float())
    scale = (decay * dt.float()).permute(0, 2, 3, 1)

    Bs = B_c * scale.unsqueeze(-1)
    states = torch.einsum("bcthp,bcthn->bchpn", x_c, Bs)
    return states
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
