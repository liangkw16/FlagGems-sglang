<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/unpad_draft_extend_output -->
<!-- synced_at: 2026-09-22T02:22:45+08:00 -->

# unpad_draft_extend_output (attention/unpad_draft_extend_output)

## 任务描述

`pad_draft_extend_query` 的逆操作：把 padded 的 `[bs, token_per_batch, H, D]` attention
输出中接受的行的 gather 成 ragged 的 `[total_tokens, H, D]` 张量。

## 接口签名

```python
def reference(raw_out, cu_seqlens_q, seq_lens_q, sum_seq_lens_q)
```

> 选手实现的函数签名需与上述完全一致。

## 计算定义

- `raw_out`：`[bs, token_per_batch, H, D]`。
- `seq_lens_q`：`[bs]` int32 接受长度；`cu_seqlens_q`：`[bs + 1]` int32。
- `sum_seq_lens_q`：接受 token 总数（输出行数）。
- 计算流程：

  ```
  output[cu_seqlens_q[b] + t] = raw_out[b, t]   for t < seq_lens_q[b]
  ```

## 正确性判别标准

精确（纯数据搬移）。

## 参考实现

```python
import torch


def reference(raw_out, cu_seqlens_q, seq_lens_q, sum_seq_lens_q):
    bs = seq_lens_q.shape[0]
    out = torch.empty(
        (sum_seq_lens_q, raw_out.shape[2], raw_out.shape[3]),
        dtype=raw_out.dtype,
        device=raw_out.device,
    )
    for b in range(bs):
        s = int(seq_lens_q[b])
        beg = int(cu_seqlens_q[b])
        out[beg : beg + s] = raw_out[b, :s]
    return out
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
