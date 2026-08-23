<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/chunk_cumsum -->
<!-- synced_at: 2026-08-24T02:17:26+08:00 -->

# chunk_cumsum (mamba/chunk_cumsum)

## 任务描述

Chunk-wise cumulative sum for Mamba SSM: computes `dt * A` cumsum within each chunk, returning processed dt and cumsum.

## 接口签名

```python
def reference(dt, A, chunk_size, dt_bias=None, dt_softplus=False)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- `dt`: `[batch, seqlen, nheads]` — 时间步长
- `A`: `[nheads]` — 衰减系数（负值）
- 可选 `dt_bias` 加到 dt 上，可选 `dt_softplus` 对 dt 做 softplus
- dt 经 clamp(min=0) 后，reshape 为 `[batch, nheads, nchunks, chunk_size]`
- `dA = dt * A`，在 chunk_size 维上做 cumsum
- 输出: `(dt_out, dA_cumsum)`，shape 均为 `[batch, nheads, nchunks, chunk_size]`

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`


## 参考实现

```python
import math

import torch
import torch.nn.functional as F


def reference(dt, A, chunk_size, dt_bias=None, dt_softplus=False):
    batch, seqlen, nheads = dt.shape
    nchunks = math.ceil(seqlen / chunk_size)

    dt_f = dt.float()
    if dt_bias is not None:
        dt_f = dt_f + dt_bias.float()
    if dt_softplus:
        dt_f = torch.where(dt_f <= 20.0, F.softplus(dt_f), dt_f)
    dt_f = dt_f.clamp(min=0.0)

    dt_out = dt_f.reshape(batch, nchunks, chunk_size, nheads).permute(0, 3, 1, 2).contiguous()
    dA = dt_out * A.float().view(1, nheads, 1, 1)
    dA_cumsum = dA.cumsum(dim=-1)
    return dt_out, dA_cumsum
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
