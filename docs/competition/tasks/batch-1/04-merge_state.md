<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/merge_state -->
<!-- synced_at: 2026-09-03T20:42:35+08:00 -->

# merge_state (attention/merge_state)

## 任务描述

Combine two partial (prefix/suffix) split-KV attention states — each an
output plus its log-sum-exp — into one, using the standard online-softmax
merge rule.

## 接口签名

```python
def reference(prefix_output, prefix_lse, suffix_output, suffix_lse):
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- `max_lse = max(prefix_lse, suffix_lse)` (elementwise; `inf` treated as `-inf`).
- `p = exp(prefix_lse - max_lse)`, `s = exp(suffix_lse - max_lse)`.
- `output = (prefix_output * p + suffix_output * s) / (p + s)`.
- `output_lse = log(p + s) + max_lse`.

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`


## 参考实现

```python
import torch


def reference(prefix_output, prefix_lse, suffix_output, suffix_lse):
    p_lse = torch.where(
        prefix_lse == float("inf"), torch.full_like(prefix_lse, float("-inf")), prefix_lse
    ).float()
    s_lse = torch.where(
        suffix_lse == float("inf"), torch.full_like(suffix_lse, float("-inf")), suffix_lse
    ).float()

    max_lse = torch.maximum(p_lse, s_lse)
    p_lse = p_lse - max_lse
    s_lse = s_lse - max_lse
    p_se = torch.exp(p_lse)
    s_se = torch.exp(s_lse)
    out_se = p_se + s_se

    output_lse = (torch.log(out_se) + max_lse).to(prefix_lse.dtype)

    p_scale = (p_se / out_se).unsqueeze(-1)
    s_scale = (s_se / out_se).unsqueeze(-1)
    output = (
        prefix_output.float() * p_scale + suffix_output.float() * s_scale
    ).to(prefix_output.dtype)
    return output, output_lse
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
