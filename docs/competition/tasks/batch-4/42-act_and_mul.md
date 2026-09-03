<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/act_and_mul -->
<!-- synced_at: 2026-09-03T20:42:35+08:00 -->

# act_and_mul (moe/act_and_mul)

## 任务描述

MoE 路径下的门控激活算子：对 `gateup_output` 拆成 gate 和 up 两半，对 gate 施加激活函数后与 up 逐元素相乘，输出到独立缓冲区。支持 SiLU 和 GELU（tanh 近似），以及可选的 SwiGLU 输入裁剪。本题仅涵盖无专家过滤路径（`topk_ids=None, expert_ids=None`），每行均参与计算。

## 接口签名

```python
def act_and_mul(gateup_output, activation="silu", swiglu_limit=None):
```

> 选手实现的函数签名需与上述 `act_and_mul(...)` 完全一致。

## 计算定义

- `gateup_output`: `[M, 2H]` 张量，`gate = gateup_output[:, :H]`，`up = gateup_output[:, H:]`
- 若 `swiglu_limit` 不为 None：`gate = clamp(gate, max=swiglu_limit)`，`up = clamp(up, -swiglu_limit, swiglu_limit)`
- 激活函数：`silu(x) = x * sigmoid(x)`；`gelu(x)` 使用 tanh 近似
- `out = activation(gate) * up`，其中激活（及可选 clamp）在 float32 精度下计算，随后 gate 和 up 均 cast 回输入 dtype，逐元素相乘在输入 dtype 下完成
- 输出 shape：`[M, H]`

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`


## 参考实现

```python
import torch.nn.functional as F


def reference(gateup_output, activation="silu", swiglu_limit=None):
    hidden_size = gateup_output.shape[1]
    half = hidden_size // 2
    gate = gateup_output[:, :half].float()
    up = gateup_output[:, half:].float()

    if swiglu_limit is not None:
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)

    if activation == "silu":
        act = F.silu(gate)
    elif activation == "gelu":
        act = F.gelu(gate, approximate="tanh")
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    out = (act.to(gateup_output.dtype) * up.to(gateup_output.dtype)).to(gateup_output.dtype)
    return out
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
