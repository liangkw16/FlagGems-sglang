<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/interleaved_rope -->
<!-- synced_at: 2026-08-30T00:06:15+08:00 -->

# interleaved_rope (rope/interleaved_rope)

## 任务描述

多模态交错 RoPE 流选择：给定三路并行的 cos/sin 已作用 RoPE 流（时间/高度/宽度），按维度下标 mod 3 将它们交错合并为一路输出，用于 Qwen2-VL 等多模态模型的位置编码合并。

## 接口签名

```python
def reference(x, mrope_section)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

输入：
- `x`: `[3, S, D]` — 三路 RoPE 流，S 为序列长度，D 为特征维度
  - `x[0]`：时间流（temporal）
  - `x[1]`：高度流（height）
  - `x[2]`：宽度流（width）
- `mrope_section`: `[s0, s1, s2]` int 列表，满足 `s0 + s1 + s2 = D // 3`，指定各模态维度段的长度

按维度下标 `d`（0 到 D-1）选择来源：

1. 若 `d % 3 == 1` 且 `d < mrope_section[1] * 3`：取高度流，`out[:, d] = x[1][:, d]`
2. 若 `d % 3 == 2` 且 `d < mrope_section[2] * 3`：取宽度流，`out[:, d] = x[2][:, d]`
3. 否则（`d % 3 == 0`，或超出段边界的维度）：取时间流，`out[:, d] = x[0][:, d]`

此操作为纯选择（gather/select），不含浮点运算，输出 dtype 与输入 `x` 相同，形状为 `[S, D]`。

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`

## 参考实现

```python
import torch


def reference(x, mrope_section):
    _, S, D = x.shape
    d = torch.arange(D, device=x.device)
    cond_a = (d % 3 == 1) & (d < mrope_section[1] * 3)
    cond_b = (d % 3 == 2) & (d < mrope_section[2] * 3)

    out = x[0].clone()
    out[:, cond_a] = x[1][:, cond_a]
    out[:, cond_b] = x[2][:, cond_b]
    return out
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
