<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/fixup_zero_kv -->
<!-- synced_at: 2026-09-17T20:50:10+08:00 -->

# fixup_zero_kv (attention/fixup_zero_kv)

## 任务描述

TRT-LLM ragged attention 的后处理：`kv_lens[i] == 0` 的请求没有任何可 attend 的 key，
其输出行是垃圾数据。该 kernel 在单次 launch 内把它们的输出清零、log-sum-exp 置为 `-inf`，
且不做 host sync —— 每个请求的 token 范围由 device 上的 `cum_seq_lens` 给出。

## 接口签名

```python
def reference(out, lse, kv_lens, cum_seq_lens, max_seq_len)
```

> 选手实现的函数签名需与上述完全一致。

## 计算定义

- `out`：`[total_tokens, num_heads, v_head_dim]` bf16/fp16。
- `lse`：`[total_tokens, num_heads]` fp32。
- `kv_lens`：`[batch_size]` int32；`cum_seq_lens`：`[batch_size + 1]` int32。
- 计算流程：

  ```
  for i where kv_lens[i] == 0:
      out[cum_seq_lens[i] : cum_seq_lens[i+1]] = 0
      lse[cum_seq_lens[i] : cum_seq_lens[i+1]] = -inf
  ```

## 正确性判别标准

精确。baseline 的 `_check` 对 out 用默认断言、对 lse 用 `equal_nan=True` 断言。

## 参考实现

```python
import torch


def reference(out, lse, kv_lens, cum_seq_lens, max_seq_len):
    out_c, lse_c = out.clone(), lse.clone()
    zero = (kv_lens == 0).nonzero().flatten().tolist()
    for i in zero:
        beg, end = int(cum_seq_lens[i]), int(cum_seq_lens[i + 1])
        out_c[beg:end] = 0
        lse_c[beg:end] = float("-inf")
    return out_c, lse_c
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
