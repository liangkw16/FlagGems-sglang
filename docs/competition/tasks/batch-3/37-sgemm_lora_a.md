<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/sgemm_lora_a -->
<!-- synced_at: 2026-09-01T13:44:50+08:00 -->

# sgemm_lora_a (lora/sgemm_lora_a)

## 任务描述

LoRA "A"（降维投影）矩阵的分段批量 GEMM：将输入按 segment 分组，每个 segment（属于同一请求的连续 token 行）与其对应 adapter 的权重切片相乘，输出各 segment 的投影结果。

## 接口签名

```python
def reference(x, weights, batch_info, stack_num=1)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- 输入 `x`: `[S, K]`；`weights`: `[num_lora, stack_num*r, K]`；`batch_info` 包含每个 segment `b` 的行范围、adapter 索引及可选的 permutation
- 输出：`[S, stack_num*r]`，与 `x` 同 dtype
- 对每个 segment `b`（行范围 `seg_indptr[b]:seg_indptr[b+1]`，adapter 索引 `w = weight_indices[b]`）：
  - 若存在 `permutation`：`rows = permutation[start:end]`，否则 `rows = arange(start, end)`
  - `out[rows] = x[rows].float() @ weights[w].float().T`，结果转回 `x` 的 dtype
- 每个 adapter 使用固定 rank `r`，输出宽度 `stack_num * r` 即权重矩阵的完整第一维

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`

## 参考实现

```python
import torch


def reference(x, weights, batch_info, stack_num=1):
    S, K = x.shape
    R = weights.shape[1]
    out = torch.zeros(S, R, dtype=x.dtype, device=x.device)

    seg_indptr = batch_info.seg_indptr
    weight_indices = batch_info.weight_indices
    permutation = batch_info.permutation

    for b in range(batch_info.bs):
        start = int(seg_indptr[b].item())
        end = int(seg_indptr[b + 1].item())
        if start == end:
            continue
        w_idx = int(weight_indices[b].item())
        if permutation is not None:
            rows = permutation[start:end].long()
        else:
            rows = torch.arange(start, end, device=x.device)

        x_seg = x[rows].float()
        w = weights[w_idx].float()
        val = x_seg @ w.t()
        out[rows] = val.to(x.dtype)

    return out
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
