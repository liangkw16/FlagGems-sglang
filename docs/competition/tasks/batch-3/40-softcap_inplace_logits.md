<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/softcap_inplace_logits -->
<!-- synced_at: 2026-08-30T00:06:15+08:00 -->

# softcap_inplace_logits (activation_norm/softcap_inplace_logits)

## 任务描述

对 logits 进行原地软截断（soft-capping）：对输入张量最后一行连续维度的每个元素应用 `tanh(x / cap) * cap`，输出与输入同 shape 同 dtype。实际算子会原地修改输入缓冲区，此处参考实现先克隆输入以保持纯函数式签名。

## 接口签名

```python
def reference(full_logits, final_logit_softcapping)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- 输入 `full_logits`: `[..., N]`，任意浮点 dtype；`final_logit_softcapping`: 标量 float
- 输出：与输入同 shape 同 dtype
- 逐元素计算：

  $$	ext{out}[i] = 	anh!left(frac{	ext{full\_logits}[i]}{	ext{final\_logit\_softcapping}}ight) 	imes 	ext{final\_logit\_softcapping}$$

- 该变换将输出值软限制在 `(-final_logit_softcapping, +final_logit_softcapping)` 区间内

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`

## 参考实现

```python
def reference(full_logits, final_logit_softcapping):
    return (full_logits / final_logit_softcapping).tanh() * final_logit_softcapping
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
