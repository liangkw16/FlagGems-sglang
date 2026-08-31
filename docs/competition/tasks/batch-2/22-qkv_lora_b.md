<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/qkv_lora_b -->
<!-- synced_at: 2026-09-01T06:22:57+08:00 -->

# qkv_lora_b (lora/qkv_lora_b)

## 任务描述

LoRA-B for QKV projection: applies the second-stage LoRA matrix (B) to produce Q, K, V outputs per-segment.

## 接口签名

```python
def reference(x, qkv_lora_b, batch_info, output_offset, max_qkv_out_dim, base_output)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- 对每个 segment:
  1. 取 x 的对应行（支持 permutation 重排）
  2. 按 `output_offset` 分 n_slices（对应 Q/K/V 三个投影）
  3. 每个 slice: `out[rows, o_start:o_end] += scaling * (x_slice @ W_slice.T)`
- `base_output` 为已有的 dense 输出，LoRA 做增量叠加
- 输出与 `base_output` 同 shape 同 dtype

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`


## 参考实现

```python
import torch


def reference(x, qkv_lora_b, batch_info, output_offset, max_qkv_out_dim, base_output):
    out = base_output.clone().float()
    n_slices = output_offset.numel() - 1
    r = qkv_lora_b.shape[-1]

    seg_indptr = batch_info.seg_indptr
    weight_indices = batch_info.weight_indices
    lora_ranks = batch_info.lora_ranks
    scalings = batch_info.scalings
    permutation = batch_info.permutation

    for b in range(batch_info.bs):
        start = int(seg_indptr[b].item())
        end = int(seg_indptr[b + 1].item())
        if start == end:
            continue
        w_idx = int(weight_indices[b].item())
        if int(lora_ranks[w_idx].item()) == 0:
            continue
        scaling = float(scalings[w_idx].item())
        if permutation is not None:
            rows = permutation[start:end].long()
        else:
            rows = torch.arange(start, end, device=x.device)

        x_seg = x[rows].float()
        for i in range(n_slices):
            o_start = int(output_offset[i].item())
            o_end = int(output_offset[i + 1].item())
            x_slice = x_seg[:, i * r : (i + 1) * r]
            w_slice = qkv_lora_b[w_idx, o_start:o_end, :].float()
            out[rows, o_start:o_end] += scaling * (x_slice @ w_slice.t())

    return out.to(base_output.dtype)
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
