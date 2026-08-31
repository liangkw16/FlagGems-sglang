<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/chunk_state_varlen -->
<!-- synced_at: 2026-09-01T06:22:57+08:00 -->

# chunk_state_varlen (mamba/chunk_state_varlen)

## 任务描述

Variable-length version of chunk_state: 处理 packed（变长拼接）格式的输入，为每条序列计算其**最后一个 chunk** 对应的 SSM 隐状态。

输入张量布局：
- `x`: `[total_seqlen, nheads, headdim]` — packed tokens
- `B`: `[total_seqlen, ngroups, dstate]` — SSM 状态投影矩阵（packed）
- `dt`: `[nheads, nchunks, chunk_size]` — 无 batch 维
- `dA_cumsum`: `[nheads, nchunks, chunk_size]` — 无 batch 维
- `cu_seqlens`: `[batch+1]` — cumulative sequence lengths
- `chunk_states`: 仅用于确定输出 dtype（不参与计算）

输出 `[batch, nheads, headdim, dstate]`。

## 接口签名

```python
def reference(B, x, dt, dA_cumsum, cu_seqlens, chunk_states)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- 输入为 packed 变长格式（无 batch 维拼接）
- 对每条序列，只计算其**最后一个 chunk** 对应的 SSM 隐状态
- 计算逻辑同 chunk_state：`scale = exp(dA_last - dA_seg) * dt_seg`
- 最终: `states[bidx, h] = x_seg.T @ (B_seg * scale)`
- `chunk_states` 参数仅用于确定输出 dtype
- 输出: `[batch, nheads, headdim, dstate]`

## 正确性判别标准

`atol=3e-2, rtol=3e-2`.


## 参考实现

```python
import torch


def reference(B, x, dt, dA_cumsum, cu_seqlens, chunk_states):
    total_seqlen, nheads, headdim = x.shape
    _, nchunks, chunk_size = dt.shape
    _, ngroups, dstate = B.shape
    batch = cu_seqlens.numel() - 1
    ratio = nheads // ngroups

    states = torch.zeros(batch, nheads, headdim, dstate, dtype=chunk_states.dtype, device=x.device)

    for bidx in range(batch):
        start = int(cu_seqlens[bidx].item())
        end = int(cu_seqlens[bidx + 1].item())
        pid_c = (end - 1) // chunk_size
        chunk_start_tok = pid_c * chunk_size
        start_rel = start - chunk_start_tok
        end_rel = end - chunk_start_tok

        for h in range(nheads):
            g = h // ratio
            dA_cs_last = dA_cumsum[h, pid_c, end_rel - 1].float()
            x_seg = x[start:end, h, :].float()
            b_seg = B[start:end, g, :].float()
            dt_seg = dt[h, pid_c, start_rel:end_rel].float()
            dA_seg = dA_cumsum[h, pid_c, start_rel:end_rel].float()
            scale = torch.exp(dA_cs_last - dA_seg) * dt_seg
            b_scaled = b_seg * scale.unsqueeze(-1)
            states[bidx, h] = (x_seg.t() @ b_scaled).to(chunk_states.dtype)

    return states
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
