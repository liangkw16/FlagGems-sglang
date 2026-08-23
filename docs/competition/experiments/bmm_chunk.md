# Task 09 `bmm_chunk` 实验记录

## S0：generic baseline

状态：本地静态检查、NVIDIA 代理验证和不可变 ZIP 门禁通过；未提交平台
验证时间：2026-08-24 CST

### 契约

- 接口：`bmm_chunk(a, b, chunk_size, causal=False)`；`causal` 当前保留但
  不参与计算，`True/False` 必须产生相同完整矩阵。
- `a`、`b` shape 均为 `[B, T, G, K]`。
- 将 T 切成 `nchunks = T / chunk_size` 后，对每个 batch/chunk/group 计算
  `a_chunk @ b_chunk.T`。
- 输出是 out-of-place 的 `[B, nchunks, G, chunk_size, chunk_size]`，固定
  FP32；三种输入 dtype 均先转 FP32 再乘加。
- 题面虽然用 `ceil(T / chunk_size)`，紧接着的无 padding `reshape` 只有
  `T % chunk_size == 0` 才成立；S0 对不整除输入明确报 `ValueError`。
- 容差：FP32 `1e-4/1e-4`，BF16 `1.5e-2/1.5e-2`，
  FP16 `1e-2/1e-2`（atol/rtol）。
- 支持八芯：天数、沐曦、燧原、海光、昆仑芯、华为、国际通用 A/B。
- 缓存目录显示提交窗口为 2026-08-20 20:00 至 2026-08-27 19:59:59，
  最低加速比 `0.1x`；提交前以平台页面为准。

### 固定参考

- Mamba
  [`v2.2.4/ssd_bmm.py`](https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_bmm.py)：
  完整读取了 forward/backward、autotune、seq_idx、causal 和 wrapper。
- 赛题只保留四维 grouped forward；删除 3D、seq_idx、backward、contiguous
  copy、CUDA device context 和九档 NVIDIA autotune。
- 上游默认输出可跟随输入 dtype，且 FP32 dot 未固定 precision；赛题要求输出
  FP32，因此 S0 load 后显式转 FP32，并用 IEEE dot 禁用 TF32。

### 构建身份

| 项目 | 值 |
| --- | --- |
| 源文件 | `src/flaggems_sglang/ops/bmm_chunk.py` |
| 源文件 SHA-256 | `b533a3d59f883b01716297f603e344ad6d9399f011223a97ee5705309ed35843` |
| 测试 SHA-256 | `1f4d1bdc848afcb64c50d614c873784088a415821a7d30bbaec303a7558e06fd` |
| 源码 commit | `b05bfeb` |
| ZIP | `artifacts/competition/bmm_chunk/s0-b05bfeb/bmm_chunk.zip` |
| ZIP SHA-256 | `058b016c309c0affa5ecbbcb125de415a6565be93e2b76a9535473021169c4e3` |
| 远端证据目录 | `gpu:/tmp/flagos-batch2.SQaIX2`，mode 0700 |
| 平台 | 未提交；未经用户当次确认不得上传 |

### 唯一候选配置

- 固定 `BLOCK_M=32`、`BLOCK_N=32`、`BLOCK_K=32`、4 warps、1 stage；
  无 autotune 或 vendor 参数。
- grid 为 `(M/N tiles, B, nchunks*G)`；每个 program 计算一个输出 tile。
- a/b 四维 strides 与 output 五维 strides 全部参与 64-bit 地址计算；不做
  `contiguous()` copy。
- K 和 chunk_size 均使用完整 tail mask，支持非 2 次幂。
- `tl.dot(a_fp32, b_fp32, input_precision="ieee")` 显式禁 TF32，accumulator
  与输出均为 FP32。
- 无 PyTorch 核心计算、fallback、设备判断、autotune 或 vendor 文件。

### 正确性与静态检查

本地 Python `py_compile` 通过。远端 RTX 5070 Ti 16 GB 环境：Python
3.12.13、PyTorch 2.13.0+cu130、Triton 3.7.1、CUDA 13.0、compute
capability 12.0。

远端公开接口 unittest：4/4 通过，覆盖：

- FP16、BF16、FP32 输入与固定 FP32 输出；
- `K=13`、`chunk_size=7` 等非 2 次幂 tail；
- causal `True/False` 与完整 reference 同结果；
- 四维均带真实 stride 的非连续 a/b；
- 空序列输出和不整除 T 的显式契约；
- a/b shape、dtype、数值保持不变。

FP32 随机用例在 `1e-4` 容差通过，也验证了 IEEE dot 路径。Black 79、
isort、flake8 均通过；上述结果只证明 NVIDIA 代理路径。

### NVIDIA 代理性能

wrapper-inclusive；每项先验证正确性，再用
`triton.testing.do_bench(warmup=20, rep=50)`。

| dtype | `[B,T,G,K]` | CS | S0 (ms) | reference (ms) | speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| FP16 | `[1,128,1,64]` | 64 | 0.006903 | 0.012637 | 1.831x |
| BF16 | `[2,256,4,64]` | 64 | 0.009569 | 0.019537 | 2.042x |
| FP32 | `[1,128,2,31]` | 32 | 0.006243 | 0.010426 | 1.670x |
| FP16 | `[2,126,3,47]` | 63 | 0.007811 | 0.017180 | 2.199x |

ZIP 由 commit `b05bfeb` 的算子子树直接生成，仅含顶层 UTF-8
`bmm_chunk.py`。`unzip -t`、10 MB、成员名和逐字节 SHA-256 门禁均通过。

### 已知风险与下一步

- 平台未公开 correctness/benchmark shape；代理 shape 不能证明八芯性能。
- `input_precision="ieee"` 是标准 Triton API，但尚未由八种编译器全部验证。
- 3D grid、runtime K loop 和固定 32 tile 可能需要针对单芯编译/性能反馈调整。
- 若首次平台仅单芯失败，保持 generic 与已通过芯片不变，只做最小 vendor
  override；下一门禁是用户针对上述 ZIP 路径、哈希和实时额度作当次确认。
