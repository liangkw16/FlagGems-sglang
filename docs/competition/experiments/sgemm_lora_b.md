# Task 23 `sgemm_lora_b` 实验记录

## S0：generic baseline

状态：本地静态检查、RTX 5070 Ti 代理验证和不可变 ZIP 门禁通过；未提交平台

### 契约

- 接口：`sgemm_lora_b(x, weights, batch_info, base_output)`。
- `x` 为 `[tokens,K]`，`weights` 为 `[num_loras,N,K]`；每个 segment 通过
  `weight_indices` 选择 LoRA 权重，并计算 `scaling * x @ weights.T`。
- `permutation[start:end]` 存在时同时决定 x 的读取行与输出写回行；S0 按
  “permutation”本义假设行索引一一映射。
- `lora_ranks[w_idx] == 0` 时保留 base；题面 reference 对任意其他值都使用
  stored weights 的完整 K，不按 metadata rank 截断。
- 输出从 `base_output.clone()` 开始，FP32 累加后 cast 回 base dtype；全部输入
  保持不变。
- 容差：FP32 `1e-4/1e-4`，BF16 `1.5e-2/1.5e-2`，FP16
  `1e-2/1e-2`（atol/rtol）。
- 支持八芯，最低加速比 `0.1x`；提交窗口和额度在上传前以平台页面为准。

固定参考为 SGLang commit
[`8014d9d/sgemm_lora_b.py`](https://github.com/sgl-project/sglang/blob/8014d9d062c3cc5d393596ecdf2f7009191965df/python/sglang/kernels/ops/gemm/sgemm_lora_b.py)。
上游按 `min(stored K, metadata rank)` 截断，而题面 reference 只把 rank 0 当
no-op；S0 有意服从题面并对非零 rank 使用完整 K。

### 构建身份

| 项目 | 值 |
| --- | --- |
| 源文件 | `src/flaggems_sglang/ops/sgemm_lora_b.py` |
| 源文件 SHA-256 | `1f8bdc82d06dd2fe018b6323c14166e803ac7abd2c8328128135df5d8fe7fe23` |
| 测试文件 | `tests/test_sgemm_lora_b.py` |
| 测试 SHA-256 | `c25586ae38939c0dcdeb6c712c575fee75ccc045a9f4568fa81f990fad489eda` |
| 源码 commit | `b05bfeb` |
| ZIP | `artifacts/competition/sgemm_lora_b/s0-b05bfeb/sgemm_lora_b.zip` |
| ZIP SHA-256 | `d3a05c053120e9bf575125f28798eac0c5b5fdf9a9bf25f57fb83a8d1df2e348` |
| 远端证据目录 | `gpu:/tmp/flagos-batch2.SQaIX2`，mode 0700 |
| 平台 | 未提交；未经用户当次确认不得上传 |

### 唯一候选配置

- 固定 `BLOCK_S=16`、`BLOCK_N=64`、`BLOCK_K=32`，4 warps、1 stage；
  grid 为 `(token/output tiles, bs)`。
- x、weights、cloned output 和全部 batch metadata 都使用真实 stride。
- accumulator、base 和 scaling 均为 FP32；`tl.dot(...,
  input_precision="ieee")` 对 FP32 输入禁用 TF32。
- 除 `base_output.clone()` 外无 PyTorch 核心计算；无 fallback、设备判断、
  autotune 或 vendor 文件。

### 正确性与代理性能

远端环境：RTX 5070 Ti 16 GB、driver 610.57.04、Python 3.12.13、PyTorch
2.13.0+cu130、Triton 3.7.1、CUDA 13.0。精确同步后的第三波联合静态门禁
全部通过，公开接口回归共 14/14，其中本题 3/3，覆盖：

- FP16/BF16/FP32、正负 scaling、空 segment、rank0 和空输入；
- permutation、非连续 x/weights/base、非 2 次幂 K，以及 metadata rank 小于
  stored K 时仍使用完整 K；
- 输出 shape/dtype、输入不变性和题面逐项 reference。

当前 commit 的 wrapper-inclusive benchmark：

| dtype | `(tokens,bs,K,N)` | S0 (ms) | reference (ms) | speedup |
| --- | --- | ---: | ---: | ---: |
| FP16 | `(256,16,16,1024)` | 0.009691 | 1.004646 | 103.665x |
| BF16 | `(1024,32,32,2048)` | 0.025685 | 2.002376 | 77.959x |
| FP32 | `(256,16,16,1024)` | 0.014421 | 0.878912 | 60.948x |

ZIP 由 commit `b05bfeb` 的算子子树直接生成，仅含顶层 UTF-8
`sgemm_lora_b.py`。`unzip -t`、10 MB、成员名和逐字节 SHA-256 门禁均通过。

### 风险与下一步

- NVIDIA 代理不能证明其余七款芯片正确或达到门槛。
- 2D grid、runtime segment metadata、scalar control flow 和 IEEE precision 尚未
  由八种编译器全部验证。
- 若输入不是一一映射的真 permutation，并行 segment 可能写同一行；题面称其
  为“permutation 重排”，S0 不增加 GPU 去重开销。
- 未提交平台或消耗额度；上传前仍需用户确认上述路径、SHA-256 和实时额度。
