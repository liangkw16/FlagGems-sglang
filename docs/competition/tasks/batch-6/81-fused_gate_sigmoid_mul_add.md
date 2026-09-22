<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/fused_gate_sigmoid_mul_add -->
<!-- synced_at: 2026-09-22T10:52:22+08:00 -->

# fused_gate_sigmoid_mul_add (activation_norm/fused_gate_sigmoid_mul_add)

## 任务描述

MoE 共享 expert 的门控融合：先算 hidden_states 与 gate_weight 的行点积得到标量门控，
再按行把 sigmoid(gate) 广播到 shared_output 上叠加到 final_hidden_states。
SGLang 内核原地写 final_hidden_states 并返回 None；本 baseline/reference 先克隆累加器再返回。

## 接口签名

```python
def reference(hidden_states, gate_weight, shared_output, final_hidden_states)
```

> 选手实现的函数签名需与上述完全一致。

## 计算定义

- `hidden_states`、`shared_output`、`final_hidden_states`：`[num_tokens, hidden]`，同 dtype。
- `gate_weight`：`[hidden]`。
- 计算流程：

  ```
  gate = sum(hidden_states * gate_weight, dim=-1)      # [num_tokens]
  out  = final_hidden_states + sigmoid(gate)[:, None] * shared_output
  ```

  行点积、sigmoid 与累加都在 fp32 中进行。

## 正确性判别标准

标准 per-dtype tolerance。

## 参考实现

```python
import torch


def reference(hidden_states, gate_weight, shared_output, final_hidden_states):
    gate = (hidden_states.float() * gate_weight.float()).sum(dim=-1)
    out = final_hidden_states.float() + torch.sigmoid(gate)[:, None] * shared_output.float()
    return out.to(final_hidden_states.dtype)
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
