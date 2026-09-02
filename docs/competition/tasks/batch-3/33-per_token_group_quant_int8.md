<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/per_token_group_quant_int8 -->
<!-- synced_at: 2026-09-02T11:21:57+08:00 -->

# per_token_group_quant_int8 (quantization/per_token_group_quant_int8)

## 任务描述

逐 token 分组 INT8 量化：将输入张量最后一维按 `group_size` 切分为若干组，每组独立计算基于绝对最大值的缩放因子并量化为 int8 整数；输出量化整数张量和每组一个的 float32 缩放因子张量。

## 接口签名

```python
def reference(x, group_size, dtype=torch.int8)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- 输入 `x`: `[..., K]`，任意浮点 dtype，连续存储；`K` 必须能被 `group_size` 整除
- 输出：`(x_q: [..., K] int8, x_s: [..., K // group_size] float32)`
- 将 `x` 在最后一维以 `group_size` 为单位分组，对每组 `g`：
  - `scale = max(|x[g]|, ε) / 127`，其中 `ε = 1e-10`
  - `x_q[g] = clamp(trunc(x[g] / scale), -128, 127)`（向零截断后转 int8）
  - `x_s[g] = scale`（float32）
- 缩放因子 shape 为 `x.shape[:-1] + (K // group_size,)`

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`

## 参考实现

```python
import torch

_EPS = 1e-10


def reference(x, group_size, dtype=torch.int8):
    iinfo = torch.iinfo(dtype)
    int8_min, int8_max = iinfo.min, iinfo.max

    x_ = x.reshape(x.numel() // group_size, group_size)
    amax = x_.abs().max(dim=-1, keepdim=True)[0].clamp(min=_EPS).to(torch.float32)
    x_s = amax / int8_max
    x_q = (x_ / x_s).clamp(min=int8_min, max=int8_max).to(dtype)
    x_q = x_q.reshape(x.shape)
    x_s = x_s.reshape(x.shape[:-1] + (x.shape[-1] // group_size,))
    return x_q, x_s
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
