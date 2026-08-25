<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/per_group_transpose -->
<!-- synced_at: 2026-08-25T00:19:05+08:00 -->

# per_group_transpose (quantization/per_group_transpose)

## 任务描述

Per-expert-group transpose: given a row-major `[M, K]` tensor whose rows are
partitioned into contiguous groups by `expert_offsets` (a grouped-GEMM
row layout), transpose each group's `[n_e, K]` block to `[K, n_e]`
independently, writing each transposed block back at the same flat byte
offset its input block occupied.

## 接口签名

```python
def reference(a, expert_offsets, m_alignment=1):
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- `expert_offsets` is a cumulative row-count array (`expert_offsets[0] == 0`,
  `expert_offsets[E] == M`); group `e` owns rows
  `expert_offsets[e]:expert_offsets[e+1]`.
- For each group: `out.view(-1)[start*K : start*K + n_e*K] == a[start:end].T.contiguous().view(-1)`,
  where `start, end = expert_offsets[e], expert_offsets[e+1]`, `n_e = end - start`.
- Output has the same total shape/size as `a` — it is not a global transpose,
  only a per-group one, laid out group-by-group.
- `m_alignment` is a compiler alignment hint on group boundaries in the
  baseline kernel; it has no effect on the output values.

## 正确性判别标准

精确相等（`atol=0, rtol=0`）。本算子为整数/字节搬运操作，无浮点计算，结果必须 bit-exact。


## 参考实现

```python
import torch


def reference(a, expert_offsets, m_alignment=1):
    m, k = a.shape
    out = torch.empty_like(a)
    flat_in = a.reshape(-1)
    flat_out = out.reshape(-1)
    num_experts = expert_offsets.numel() - 1
    for e in range(num_experts):
        start = int(expert_offsets[e].item())
        end = int(expert_offsets[e + 1].item())
        n = end - start
        if n <= 0:
            continue
        seg = a[start:end].t().contiguous().reshape(-1)
        flat_out[start * k : start * k + n * k] = seg
    return out
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
