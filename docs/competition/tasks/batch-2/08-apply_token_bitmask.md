<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/apply_token_bitmask -->
<!-- synced_at: 2026-08-25T00:19:05+08:00 -->

# apply_token_bitmask (sampling_grammar/apply_token_bitmask)

## 任务描述

Apply a token bitmask to logits: for each token, set logits to `-inf` where the corresponding bit is unset in the bitmask (used for constrained decoding / grammar sampling).

## 接口签名

```python
def reference(logits, bitmask)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- `logits`: `[B, V]` float tensor
- `bitmask`: `[B, ceil(V/32)]` int32 tensor，每 bit 对应一个 token
- 对于第 `v` 个 token：`word_idx = v // 32`, `bit_idx = v % 32`
- 若 `(bitmask[:, word_idx] >> bit_idx) & 1 == 0`，则置 `logits[:, v] = -inf`
- 输出与 `logits` 同 shape 同 dtype

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`


## 参考实现

```python
import torch


def reference(logits, bitmask):
    B, V = logits.shape
    v_idx = torch.arange(V, device=logits.device)
    word_idx = v_idx // 32
    bit_idx = v_idx % 32
    bits = (bitmask[:, word_idx] >> bit_idx) & 1
    return torch.where(bits == 0, torch.full_like(logits, float("-inf")), logits)
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
