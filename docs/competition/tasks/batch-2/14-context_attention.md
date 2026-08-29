<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/context_attention -->
<!-- synced_at: 2026-08-30T00:06:15+08:00 -->

# context_attention (attention/context_attention)

## 任务描述

Prefill/context阶段的注意力计算：对变长序列执行 scaled dot-product attention，支持 causal mask。输入为 packed 格式（多条序列拼接），通过 `b_start_loc` 和 `b_seq_len` 指示每条序列的起止位置。

## 接口签名

```python
def reference(q, k, v, b_start_loc, b_seq_len, max_input_len, is_causal)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- 输入 tensor layout: `[total_tokens, num_heads, head_dim]`（packed 格式）
- 对每条序列 `i`，取 `q[start:end]`, `k[start:end]`, `v[start:end]`
- 调用 scaled_dot_product_attention（含可选 causal mask）
- 输出 shape 与 `q` 相同，dtype 为 float32
- 注意：`max_input_len` 参数为 kernel 实现预留（用于分配临时空间），reference 中未使用

## 正确性判别标准

`atol=1e-2, rtol=1e-2`.


## 参考实现

```python
import torch
import torch.nn.functional as F


def reference(q, k, v, b_start_loc, b_seq_len, max_input_len, is_causal):
    o = torch.empty_like(q, dtype=torch.float32)
    B = b_seq_len.shape[0]
    for i in range(B):
        start = int(b_start_loc[i].item())
        length = int(b_seq_len[i].item())
        end = start + length
        o[start:end] = F.scaled_dot_product_attention(
            q[start:end].permute(1, 0, 2).float(),
            k[start:end].permute(1, 0, 2).float(),
            v[start:end].permute(1, 0, 2).float(),
            is_causal=is_causal,
        ).permute(1, 0, 2)
    return o
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
