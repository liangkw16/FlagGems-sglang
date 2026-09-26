<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/tiny_n_gemm -->
<!-- synced_at: 2026-09-26T22:45:04+08:00 -->

# tiny_n_gemm (gemm/tiny_n_gemm)

## 任务描述

decode 阶段充满的小投影的瘦-M bf16 GEMM：`out = x @ w.T`，max_m（默认 16）行。
kernel 把 N 维拆成每 block `split_n` 列组，使 `n / split_n` 个 block 恰好填满
一波 SM —— launch config 搜索（`_default_split_n`）是 baseline 的一部分。

## 接口签名

```python
def reference(x, w, out_dtype)
```

> 选手实现的函数签名需与上述完全一致。

## 计算定义

- `x`：`[m, k]` bf16，m <= 16；`w`：`[n, k]` bf16。
- `out_dtype`：`torch.bfloat16` 或 `torch.float32`。
- 计算流程：

  ```
  out = (x.float() @ w.float().t()).to(out_dtype)
  ```

- 约束：`max_m * split_n <= k / vec_elems`，很小的 k 会限制可达到的拆分。

## 正确性判别标准

标准 per-dtype tolerance。

## 参考实现

```python
def reference(x, w, out_dtype):
    return (x.float() @ w.float().t()).to(out_dtype)
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
