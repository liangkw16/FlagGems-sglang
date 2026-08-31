<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/silu_and_mul_masked -->
<!-- synced_at: 2026-09-01T06:22:57+08:00 -->

# silu_and_mul_masked (moe/silu_and_mul_masked)

## 任务描述

带掩码的 SiLU-and-mul 激活，用于 DeepGEMM 风格的分组 MoE 布局：每个专家的 token 块被填充到固定长度 `token_num_padded`，仅前 `masked_m[e]` 行有效。对有效行执行 `silu(gate) * up` 并以 bfloat16 输出，填充行不写入输出。

## 接口签名

```python
def reference(input, masked_m)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- 输入 `input`: `[E, T, H]`，dtype 为 bfloat16；`masked_m`: `[E]`，dtype 为 int
- 输出 `out`: `[E, T, H//2]`，dtype 为 bfloat16，初始化为零
- 对每个专家 `e`，令 `n = masked_m[e]`，`half = H // 2`：
  - `gate = input[e, :n, :half]`（转 float32）
  - `up   = input[e, :n, half:]`（转 float32）
  - `out[e, :n] = (gate * sigmoid(gate) * up)` 转回 bfloat16
- 行 `masked_m[e]:T` 为填充区，其输出值不作要求；正确性仅在每个专家的有效行 `[e, :masked_m[e]]` 上检验

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`

## 参考实现

```python
import torch


def reference(input, masked_m):
    E, T, H = input.shape
    half = H // 2
    out = torch.zeros(E, T, half, dtype=torch.bfloat16, device=input.device)
    for e in range(E):
        n = int(masked_m[e].item())
        if n <= 0:
            continue
        gate = input[e, :n, :half].float()
        up = input[e, :n, half:].float()
        val = gate * torch.sigmoid(gate) * up
        out[e, :n] = val.to(torch.bfloat16)
    return out
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
