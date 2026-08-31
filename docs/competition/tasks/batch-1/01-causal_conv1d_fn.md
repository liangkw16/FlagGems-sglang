<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/causal_conv1d_fn -->
<!-- synced_at: 2026-09-01T06:22:57+08:00 -->

# causal_conv1d_fn (mamba/causal_conv1d_fn)

## 任务描述

Depthwise causal 1D convolution over variable-length, concatenated
("continuous batching") sequences, fused with an optional activation.

## 接口签名

```python
def reference(x, weight, bias, query_start_loc, seq_lens_cpu, activation="silu"):
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- Sequence `i` occupies columns `query_start_loc[i]:query_start_loc[i+1]`
  of `x` (concatenated left-to-right, no cross-sequence bleed).
- Per sequence, per channel `d`, per position `t` (0-indexed within the
  sequence): `conv[d,t] = sum_{k=0}^{width-1} weight[d,k] * x_pad[d, t+k]`,
  where `x_pad` is that sequence's channel row left-padded with
  `width-1` zeros (causal — no future or cross-sequence lookback).
- `out = conv + bias` (if given), then `out = silu(out)` if
  `activation in {"silu", "swish"}`.
- Scope: fresh prefill only (no cache, no `conv_states`, no `cache_indices`).

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`


## 参考实现

```python
import torch
import torch.nn.functional as F


def reference(x, weight, bias, query_start_loc, seq_lens_cpu, activation="silu"):
    dim, _ = x.shape
    width = weight.shape[1]
    out = torch.zeros_like(x)

    for i in range(len(seq_lens_cpu)):
        start = int(query_start_loc[i].item())
        end = int(query_start_loc[i + 1].item())
        seg = x[:, start:end].float()
        seg_len = seg.shape[1]
        padded = F.pad(seg, (width - 1, 0))

        conv = torch.zeros(dim, seg_len, device=x.device)
        for k in range(width):
            conv += weight[:, k : k + 1].float() * padded[:, k : k + seg_len]
        if bias is not None:
            conv += bias.float().unsqueeze(-1)
        if activation in ("silu", "swish"):
            conv = conv * torch.sigmoid(conv)

        out[:, start:end] = conv.to(x.dtype)

    return out
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
