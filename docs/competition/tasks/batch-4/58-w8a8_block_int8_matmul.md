<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/w8a8_block_int8_matmul -->
<!-- synced_at: 2026-09-06T07:54:38+08:00 -->

# w8a8_block_int8_matmul (quantization/w8a8_block_int8_matmul)

## 任务描述

块量化 INT8 GEMM：输入矩阵 `A` 和 `B` 均为已量化的 int8 张量，各自携带块粒度的 float32 缩放因子。对每个输出元素，先在 float32 下累加 int8 乘积，再乘以对应的块缩放因子，最后 cast 到指定的输出 dtype。SGLang baseline 对应 `sglang.kernels.ops.quantization.int8_kernel.w8a8_block_int8_matmul`。

## 接口签名

```python
def w8a8_block_int8_matmul(A, B, As, Bs, block_size, output_dtype):
```

> 选手实现的函数签名需与上述 `w8a8_block_int8_matmul(...)` 完全一致。

## 计算定义

设 `block_size = [block_n, block_k]`，则块索引为：

$$i_k = lfloor k / block\_k floor, quad i_n = lfloor n / block\_n floor$$

输出元素：

$$	ext{out}[m, n] = left(sum_{k} A[m, k] cdot B[n, k] cdot As[m,, i_k] cdot Bs[i_n,, i_k]ight)_{	ext{float32}} 	o 	ext{output\_dtype}$$

等价地，对每个 `(k_block, n_block)` 块 tile：

$$C[:, n\_lo:n\_hi] mathrel{+}= left(A[:, k\_lo:k\_hi] cdot B[n\_lo:n\_hi,, k\_lo:k\_hi]^	opight) 	imes As[:, i_k] 	imes Bs[i_n, i_k]$$

所有累加均在 float32 精度下进行，最终 cast 到 `output_dtype`。

## 输入输出规格

| 参数 | 形状 | dtype | 说明 |
|------|------|-------|------|
| `A` | `[M, K]` | int8 | 量化后的激活矩阵 |
| `B` | `[N, K]` | int8 | 量化后的权重矩阵（已转置存储） |
| `As` | `[M, K // block_k]` | float32 | A 的块缩放因子，按行和 K 方向分块 |
| `Bs` | `[N // block_n, K // block_k]` | float32 | B 的块缩放因子，按 N 和 K 方向分块 |
| `block_size` | `[block_n, block_k]` | list[int] | N 方向和 K 方向的块大小 |
| `output_dtype` | — | torch.dtype | 输出张量的目标 dtype（通常为 bfloat16 或 float16） |
| **output** | `[M, N]` | `output_dtype` | 去量化后的矩阵乘结果 |

## 测试用例

| M | N | K | block_n | block_k | output_dtype | atol | rtol |
|---|---|---|---------|---------|--------------|------|------|
| 16 | 128 | 256 | 128 | 64 | bfloat16 | 0.5 | 1e-2 |
| 32 | 256 | 512 | 128 | 64 | bfloat16 | 0.5 | 1e-2 |
| 64 | 512 | 1024 | 128 | 128 | bfloat16 | 0.5 | 1e-2 |
| 128 | 1024 | 2048 | 128 | 64 | float16 | 0.5 | 1e-2 |

正确性判别：`atol=0.5, rtol=1e-2`（连续 GEMM 输出的 bf16/fp16 累加容差，非量化边界问题）。

## 参考实现

```python
import torch


def reference(A, B, As, Bs, block_size, output_dtype):
    A = A.to(torch.float32)
    B = B.to(torch.float32)
    block_n, block_k = block_size
    M, K = A.shape
    N, _ = B.shape

    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k

    C = torch.zeros(M, N, dtype=torch.float32, device=A.device)
    for i in range(k_tiles):
        k_lo, k_hi = i * block_k, min((i + 1) * block_k, K)
        a_tile = A[:, k_lo:k_hi]
        a_s = As[:, i : i + 1]
        for j in range(n_tiles):
            n_lo, n_hi = j * block_n, min((j + 1) * block_n, N)
            b_tile = B[n_lo:n_hi, k_lo:k_hi]
            s = a_s * Bs[j, i]
            C[:, n_lo:n_hi] += torch.matmul(a_tile, b_tile.t()) * s

    return C.to(output_dtype)
```

## 注意事项

- int8 输入须先 cast 到 float32 再做矩阵乘，不可在 int8 下直接 GEMM，以保证精度。
- `B` 的存储格式为 `[N, K]`（而非 `[K, N]`），做矩阵乘时需转置：`A @ B.t()`。
- 块缩放因子 `Bs` 的索引为 `[n_tile, k_tile]`，与 `As` 的索引 `[m, k_tile]` 在 K 维度对齐。
- 当 N 或 K 不能被块大小整除时，最后一个 tile 的范围由 `min` 截断处理，缩放因子仍取整块对应的标量。
- 容差 `atol=0.5` 较大，是因为 bf16/fp16 输出本身存在量化舍入误差，不反映实现错误。

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
