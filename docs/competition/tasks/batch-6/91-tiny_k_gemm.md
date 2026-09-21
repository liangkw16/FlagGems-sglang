<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/tiny_k_gemm -->
<!-- synced_at: 2026-09-21T20:56:26+08:00 -->

# tiny_k_gemm (gemm/tiny_k_gemm)

## 任务描述

`tiny_n_gemm` 的小 K / 大 N 兄弟：`out = x @ w.T`。一个 warp 的 `K / 8` 条 lane
针对单个输出列归约 K 维；每个 block 覆盖 `split_n` 列，使 `N / split_n` 的 grid
恰好填满 SM 且无 tail block。

## 接口签名

```python
def reference(x, w, out_dtype)
```

> 选手实现的函数签名需与上述完全一致。

## 计算定义

- `x`：`[m, k]` bf16，m <= 16；`w`：`[n, k]` bf16。
- `k / 8` 必须是 2 的幂且 `<= 32` —— 即 `k` ∈ {128, 256}。
- `out_dtype`：`torch.bfloat16` 或 `torch.float32`。
- 计算流程：

  ```
  out = (x.float() @ w.float().t()).to(out_dtype)
  ```

## 正确性判别标准

标准 per-dtype tolerance。

## 参考实现

```python
def reference(x, w, out_dtype):
    return (x.float() @ w.float().t()).to(out_dtype)
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
