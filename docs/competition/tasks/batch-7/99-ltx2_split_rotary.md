<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/ltx2_split_rotary -->
<!-- synced_at: 2026-09-26T22:45:04+08:00 -->

# ltx2_split_rotary (diffusion/ltx2_split_rotary)

## 任务描述

LTX-2 的 split-half rotary（rotate-half 而非 interleaved pairs），使用 per-head 的 `cos`/`sin` 表。

## 接口签名

```python
def reference(x, cos, sin)
```

> 选手实现的函数签名需与上述完全一致。

## 计算定义

- `x`: `[batch, seq_len, num_heads * head_dim]` bf16
- `cos`/`sin`: `[batch, num_heads, seq_len, head_dim // 2]`
- `out[..., :h] = bf16(x[..., :h] * cos) - x[..., h:] * sin`
- `out[..., h:] = bf16(x[..., h:] * cos) + x[..., :h] * sin`
- 中间 `bf16(...)` 舍入是刻意保留的，复现原始 PyTorch 顺序（`x * cos` 物化为 bf16，再在 fp32 中 addcmul_）

## 正确性判别标准

标准 per-dtype tolerance

## 参考实现

```python
import torch


def reference(x, cos, sin):
    batch, seq_len, inner = x.shape
    _, num_heads, _, half = cos.shape
    head_dim = half * 2
    xv = x.reshape(batch, seq_len, num_heads, head_dim)
    # cos/sin are [B, H, T, half]; align to [B, T, H, half].
    c = cos.permute(0, 2, 1, 3)
    s = sin.permute(0, 2, 1, 3)

    x1, x2 = xv[..., :half], xv[..., half:]
    o1 = (x1 * c).to(torch.bfloat16).float() - x2.float() * s.float()
    o2 = (x2 * c).to(torch.bfloat16).float() + x1.float() * s.float()
    out = torch.cat([o1, o2], dim=-1)
    return out.reshape(batch, seq_len, inner).to(x.dtype)
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
