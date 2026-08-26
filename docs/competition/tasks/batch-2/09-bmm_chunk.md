<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/bmm_chunk -->
<!-- synced_at: 2026-08-27T00:31:02+08:00 -->

# bmm_chunk (mamba/bmm_chunk)

## 任务描述

Batched matrix multiply within chunks: reshape input into chunks and perform per-group batched inner product.输入 shape 为 `[batch, seqlen, ngroups, k]`，按 `chunk_size` 切块后做 einsum `bcigk,bcjgk->bcgij`（即每个 group 内的 chunk-local K*K^T）。

注意：`causal` 参数当前未使用（保留用于未来扩展），实现时可忽略。

## 接口签名

```python
def reference(a, b, chunk_size, causal=False)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- 输入 `a`, `b`: `[batch, seqlen, ngroups, k]`
- 按 `chunk_size` 切块: `a_c = a.reshape(batch, nchunks, chunk_size, ngroups, k)`
- 计算 per-group chunk-local 内积: `out = einsum("bcigk,bcjgk->bcgij", a_c, b_c)`
- 输出: `[batch, nchunks, ngroups, chunk_size, chunk_size]`
- 全程使用 float32 计算

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`


## 参考实现

```python
import math

import torch


def reference(a, b, chunk_size, causal=False):
    batch, seqlen, ngroups, k = a.shape
    nchunks = math.ceil(seqlen / chunk_size)

    a_c = a.reshape(batch, nchunks, chunk_size, ngroups, k).float()
    b_c = b.reshape(batch, nchunks, chunk_size, ngroups, k).float()
    out = torch.einsum("bcigk,bcjgk->bcgij", a_c, b_c)
    return out
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
