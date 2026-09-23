<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/moe_topk_sum -->
<!-- synced_at: 2026-09-23T17:49:06+08:00 -->

# moe_topk_sum (moe/moe_topk_sum)

## 任务描述

归约每个 token 的 per-expert MoE 输出：`out[m, k] = x[m, topk, k].sum(dim=1)`。纯带宽操作 —— 每个输出元素 `topk` 次读、一次写，hidden size 可达数千。

## 接口签名

```python
def reference(x, out)
```

> 选手实现的函数签名需与上述完全一致。

## 计算定义

- 两个张量均为 CUDA 上连续的 **bf16**
- 以 fp32 求和后转回 `out.dtype`

```
out[m, k] = x[m, topk, k].float().sum(dim=1).to(out.dtype)
```

## 正确性判别标准

Per-dtype tolerance（standard `harness.correctness`）。

## 参考实现

```python
def reference(x, out):
    return x.float().sum(dim=1).to(out.dtype)
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
