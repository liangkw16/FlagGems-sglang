<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/rmsnorm_hf -->
<!-- synced_at: 2026-09-26T22:45:04+08:00 -->

# rmsnorm_hf (activation_norm/rmsnorm_hf)

## 任务描述

HuggingFace `LlamaRMSNorm` 语义的 RMSNorm —— 归一化激活在权重乘之前先 cast 回输入 dtype，
这与全程 fp32 的 `fused_rmsnorm` 不同。

## 接口签名

```python
def reference(input, weight, eps)
```

> 选手实现的函数签名需与上述完全一致。

## 计算定义

- `input`：`[num_tokens, hidden]` 2D fp16/bf16；`weight`：`[hidden]`。
- 计算流程：

  ```
  y = (x.float() * rsqrt(mean(x.float()**2, -1, keepdim=True) + eps)).to(x.dtype)
  out = weight * y
  ```

## 正确性判别标准

标准 per-dtype tolerance。

## 参考实现

```python
import torch


def reference(input, weight, eps):
    x = input.float()
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return weight * y.to(input.dtype)
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
