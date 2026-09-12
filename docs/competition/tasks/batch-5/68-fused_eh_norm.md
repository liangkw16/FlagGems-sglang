<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/fused_eh_norm -->
<!-- synced_at: 2026-09-12T20:28:08+08:00 -->

# fused_eh_norm (activation_norm/fused_eh_norm)

## 任务描述

EAGLE draft-model 前处理：用两套不同的权重分别对 token 嵌入（inputs_embeds）和上一时刻的
hidden state（previous_hidden）做 RMSNorm，再沿最后一维拼接 —— 一个 kernel 代替两次
norm 加一次 cat。归约在 fp32 中进行，权重乘也在 fp32 中完成，最后只在写回时做一次类型转换。

## 接口签名

```python
def reference(inputs_embeds, previous_hidden, enorm_weight, hnorm_weight, eps)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- `inputs_embeds` / `previous_hidden`：`[num_tokens, hidden]`，fp16 或 bf16。
- `hidden` 需在 `(256, 8192]` 且为 256 的倍数。
- 计算流程：

  ```
  e = rmsnorm(inputs_embeds, enorm_weight, eps)
  h = rmsnorm(previous_hidden, hnorm_weight, eps)
  out = cat([e, h], dim=-1)
  ```

  其中 `rmsnorm(x, w, eps) = x * rsqrt(mean(x^2, -1, keepdim=True) + eps) * w`。

## 正确性判别标准

标准 per-dtype tolerance（float32 1e-4/1e-4，bf16 1.5e-2/1.5e-2，fp16 1e-2/1e-2）。

## 参考实现

```python
import torch


def _rmsnorm(x, weight, eps):
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()


def reference(inputs_embeds, previous_hidden, enorm_weight, hnorm_weight, eps):
    e = _rmsnorm(inputs_embeds, enorm_weight, eps)
    h = _rmsnorm(previous_hidden, hnorm_weight, eps)
    return torch.cat([e, h], dim=-1).to(inputs_embeds.dtype)
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
