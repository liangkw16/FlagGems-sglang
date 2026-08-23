# Task 17 `embedding_lora_a` 实验记录

## S0：generic baseline

状态：S0 已打包并通过本地门禁；等待当次上传确认

验证时间：2026-08-24 01:28–01:37 CST

### 契约

- 接口：
  `embedding_lora_a(input_ids, weights, batch_info, vocab_size, extra_embeddings=None)`。
- `input_ids` shape `[S]`；`weights` shape
  `[num_loras, max_rank, vocab_size]`；输出是 out-of-place 的
  `[S, max_rank]`，dtype 与 `weights` 一致。
- 对 segment `b` 的 `[start:end)`，用 adapter `w_idx` 和有效 rank `r`
  写入 `weights[w_idx, :r, input_ids].T`；其余列保持零。
- token `>= vocab_size` 时，base 路径先 clamp 到 `vocab_size - 1`；若提供
  `extra_embeddings[num_loras, num_extra_tokens, max_rank]`，再以
  `[w_idx, token-vocab_size, :r]` 替换该行。
- 空 segment 与 `r == 0` 不写任何值，因此对应输出保持零；所有输入保持不变。
- 容差：FP32 `1e-4/1e-4`，BF16 `1.5e-2/1.5e-2`，
  FP16 `1e-2/1e-2`（atol/rtol）。
- 支持八芯：天数、沐曦、燧原、海光、昆仑芯、华为、国际通用 A/B。
- 缓存目录显示提交窗口为 2026-08-20 20:00 至 2026-08-27 19:59:59，
  最低加速比 `0.1x`；提交前以平台当次页面为准。

### `batch_info` 字段语义

| 字段 | 语义 |
| --- | --- |
| `bs` | 本次 segment 数；Triton grid 的第二维 |
| `seg_indptr[bs+1]` | segment 的累计 token 边界，`[b:b+2]` 给出 `[start:end)` |
| `seg_lens[bs]` | 每个 segment 的长度，应等于相邻 `seg_indptr` 之差 |
| `max_len` | 当前 batch 最大 segment 长度；Triton grid 的第一维 |
| `weight_indices[bs]` | segment 到 `weights` 第一维 adapter index 的映射 |
| `lora_ranks[num_loras]` | 每个 adapter 的有效 rank；按 `weight_indices[b]` 索引 |

`scalings`、`permutation` 等 `LoRABatchInfo` 其他字段不参与本题 reference。

### 固定参考

- SGLang
  [`8014d9d`](https://github.com/sgl-project/sglang/blob/8014d9d062c3cc5d393596ecdf2f7009191965df/python/sglang/kernels/ops/gemm/embedding_lora_a.py)：
  完整单文件参考的签名、batch 字段、地址公式、空 segment/rank-zero 和 extra
  分支与题面一致。
- S0 保留其 BLOCK_RANK 128 与 `max_len × bs` grid，移除未使用维度参数、
  contiguous 断言和 SGLang 类型依赖；补全所有 tensor 的真实 strides。

### 构建身份

| 项目 | 值 |
| --- | --- |
| 源文件 | `src/flaggems_sglang/ops/embedding_lora_a.py` |
| 源文件 SHA-256 | `f9fc288c06bd70793249f559d159f08f2be7adeb43913cc7aa00ce566abfc9e9` |
| 测试 SHA-256 | `1390ea035dd3ad9ede052484d9e557bff1acfa68e01b3bee2c3f7e00b6faebbc` |
| 源码 commit | `f431ba4` |
| ZIP | `artifacts/competition/embedding_lora_a/s0-f431ba4/embedding_lora_a.zip` |
| ZIP SHA-256 | `e0fd0124cece568d536efaa89d05779c1f7457d9f0abf13efba8d190c482567e` |
| ZIP manifest | 顶层 `embedding_lora_a.py`，4805 bytes |
| 远端证据目录 | `gpu:/tmp/flagos-batch2.SQaIX2`，mode 0700 |

### 唯一候选配置

- grid 为 `(batch_info.max_len, batch_info.bs)`；每个 program 处理一个
  segment 内 token。
- 有效 rank 以 128 为 block 分段；动态尾块 mask，覆盖 rank 大于 128。
- input、weights、output、extra embeddings 和四个 batch tensor 均使用真实
  strides；地址 stride 转为 64-bit。
- 输出仅用 `torch.zeros` 做允许的零初始化；base lookup、extra lookup 和写回
  全部真实运行 Triton。
- `HAS_EXTRA_EMBEDDINGS` 是 constexpr；显式使用 4 warps、1 stage；无设备
  判断、fallback、autotune 或 vendor 文件。

### 正确性与静态检查

本地 Python `py_compile` 通过。远端 RTX 5070 Ti 16 GB 环境：Python
3.12.13、PyTorch 2.13.0+cu130、Triton 3.7.1、CUDA 13.0、compute
capability 12.0。

远端公开接口 unittest：3/3 通过，覆盖：

- FP16、BF16、FP32 与题面容差；
- 普通 segment、空 segment、adapter rank=0 和未写列保持零；
- extra embedding 替换，以及未提供 extra 时 clamp 到最后 base token；
- rank=130 的第二 rank block 和尾 mask；
- 非连续 input/weights 的真实 strides；
- input IDs、weights、extra embeddings 与 batch metadata 全部不变。

Black 79、isort、flake8 均通过。上述结果只证明 NVIDIA 代理路径。

### NVIDIA 代理性能

wrapper-inclusive；每项先验证正确性，再用
`triton.testing.do_bench(warmup=20, rep=50)`。Torch reference 含题面中的
Python segment 循环。

| dtype | `(S,bs,rank,vocab)` | 路径 | S0 (ms) | reference (ms) | speedup |
| --- | --- | --- | ---: | ---: | ---: |
| FP16 | `(32,4,16,4096)` | base | 0.006502 | 0.164028 | 25.228x |
| FP16 | `(256,16,32,8192)` | base | 0.007944 | 0.630260 | 79.334x |
| BF16 | `(1024,32,64,32000)` | base | 0.021861 | 1.255000 | 57.409x |
| FP32 | `(256,16,32,8192)` | extra | 0.008288 | 1.111298 | 134.092x |

### 已知风险与下一步

- 平台未公开 correctness/benchmark shape；表中 shape 只是同类 LoRA 代理值。
- 固定实现要求 `max_len`/`seg_lens` 已提供且与 `seg_indptr` 一致；平台固定
  batch type 满足该条件，S0 不增加同步到 CPU 的修复逻辑。
- runtime rank loop、scalar control flow 和 2D grid 尚未被八种编译器验证。
- 题面未定义越界 extra token；S0 与固定参考一样，要求 extra index 有效。
- ZIP 由 commit `f431ba4` 直接生成；`unzip -t`、UTF-8、单一 `.py`、10 MB、
  basename 和 ZIP 内源码哈希门禁均通过。首次平台失败若只集中在单芯，保持
  generic 不变并加最小 vendor override；上述 ZIP 仍需当次用户确认。
