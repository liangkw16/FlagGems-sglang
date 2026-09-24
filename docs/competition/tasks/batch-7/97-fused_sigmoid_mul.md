<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/fused_sigmoid_mul -->
<!-- synced_at: 2026-09-24T20:58:34+08:00 -->

# fused_sigmoid_mul (activation_norm/fused_sigmoid_mul)

## 任务描述

Attention 输出门控：`out = attn_output * sigmoid(gate)`。
两条路径：flat 路径下 attn_output 与 gate 同形 `[num_tokens, hidden]`；
strided 路径下 gate 是 3D `[num_tokens, num_heads, head_dim]`，attn_output 是 2D
`[num_tokens, num_heads * head_dim]`，内核直接按 stride 读取 gate 而无需 contiguous 拷贝。

## 接口签名

```python
def reference(attn_output, gate)
```

> 选手实现的函数签名需与上述完全一致。

## 计算定义

- `gate` 会被 reshape 到 `attn_output` 的形状。
- 计算流程：

  ```
  g = gate.reshape(attn_output.shape)
  out = attn_output * sigmoid(g)          # fp32 计算
  ```

  结果以输入 dtype 存储（inplace 变体不在范围内）。

## 正确性判别标准

标准 per-dtype tolerance。

## 参考实现

```python
import torch


def reference(attn_output, gate):
    g = gate.reshape(attn_output.shape).float()
    out = attn_output.float() * torch.sigmoid(g)
    return out.to(attn_output.dtype)
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
