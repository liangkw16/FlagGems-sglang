<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/compute_position -->
<!-- synced_at: 2026-09-21T00:02:02+08:00 -->

# compute_position (attention/compute_position)

## 任务描述

extend（chunked-prefill）batch 的 fused position + start offset 计算。
对请求 i（prefix 长 p_i、extend 长 s_i）：start offset 是前 i 个 extend 长度之和，
positions 段填 `p_i + arange(s_i)`。SGLang 内核在 kernel 内部做排他前缀和，
大 batch 下是明显的优化点。

## 接口签名

```python
def reference(extend_prefix_lens, extend_seq_lens, extend_seq_lens_sum)
```

> 选手实现的函数签名需与上述完全一致。

## 计算定义

- `extend_prefix_lens`、`extend_seq_lens`：`[batch_size]` int32。
  `extend_prefix_lens` 为空时 prefix 视为 0。
- 计算流程：

  ```
  start = exclusive_cumsum(extend_seq_lens)
  positions[start[i] : start[i] + s_i] = p_i + arange(s_i)
  ```

- 返回 `(positions, start)`：positions 为 int64 `[extend_seq_lens_sum]`，
  start 为 int32 `[batch_size]`。

## 正确性判别标准

精确（整数运算）。baseline 的 `_check` 分别比较两个输出。

## 参考实现

```python
import torch


def reference(extend_prefix_lens, extend_seq_lens, extend_seq_lens_sum):
    bs = extend_seq_lens.shape[0]
    has_prefix = extend_prefix_lens.shape[0] == bs
    cumsum = torch.cumsum(extend_seq_lens, dim=0)
    start = torch.zeros_like(cumsum)
    start[1:] = cumsum[:-1]

    positions = torch.empty(
        extend_seq_lens_sum, dtype=torch.int64, device=extend_seq_lens.device
    )
    for i in range(bs):
        s = int(extend_seq_lens[i])
        p = int(extend_prefix_lens[i]) if has_prefix else 0
        beg = int(start[i])
        positions[beg : beg + s] = torch.arange(
            p, p + s, dtype=torch.int64, device=extend_seq_lens.device
        )
    return positions, start.to(torch.int32)
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
