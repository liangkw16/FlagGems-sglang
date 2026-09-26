<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/set_mla_kv_buffer -->
<!-- synced_at: 2026-09-26T22:45:04+08:00 -->

# set_mla_kv_buffer (kvcache/set_mla_kv_buffer)

## 任务描述

Scatter-write MLA paged KV cache：把每个 token 的 NoPE 和 RoPE key 部分拼接成一行存到 slot `loc[i]`。

## 接口签名

```python
def reference(kv_buffer, loc, cache_k_nope, cache_k_rope)
```

> 选手实现的函数签名需与上述完全一致。

## 计算定义

- `kv_buffer[loc[i], :nope_dim] = cache_k_nope[i]`；`kv_buffer[loc[i], nope_dim:] = cache_k_rope[i]`
- `kv_buffer`: `[num_slots, nope_dim + rope_dim]` bf16；`loc`: `[n_loc]` int64
- SGLang 原地写 `kv_buffer`；baseline 与 reference clone 后返回

## 正确性判别标准

exact（纯数据搬运）

## 参考实现

```python
import torch


def reference(kv_buffer, loc, cache_k_nope, cache_k_rope):
    out = kv_buffer.clone()
    row = torch.cat([cache_k_nope, cache_k_rope], dim=-1).to(out.dtype)
    out[loc.long()] = row
    return out
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
