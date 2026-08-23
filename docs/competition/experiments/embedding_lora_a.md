# Task 17 `embedding_lora_a` 实验记录

## S0：generic baseline

状态：S0 已由 S1 正确性修复取代；保留为历史回退，不再建议上传

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

## S1：以 `seg_indptr` 为准并保护空 segment

状态：候选就绪，尚未提交平台

验证时间：2026-08-24 05:22–05:51 CST

### 根因与最小修复

S0 在判断 segment 是否为空前先读取 `weight_indices[b]` 和
`lora_ranks[weight_index]`。题面 reference 先检查
`seg_indptr[b] == seg_indptr[b+1]`；因此空 segment 的 metadata 即使不应被访问，
S0 仍可能越界读取并在部分后端 fault。S0 还额外用 `seg_lens[b]` 决定有效 token，
而 reference 的数据边界只由 `seg_indptr` 定义。

S1 的 kernel 先读取相邻两个 `seg_indptr`，对超出 `[start:end)` 的 program 立即
返回，然后才读取 adapter index 和 rank；同时删除 kernel 参数中的 `seg_lens` 及其
stride。`max_len` 仍作为 host grid 上界保留，避免为推导最大 segment 长度引入
device-to-host 同步。

### TDD 与 release 验证

- 新回归故意设置 `seg_lens=[1,0,1]`、`seg_indptr=[0,2,2,3]`，并给空 segment
  一个越界 adapter 哨兵；S0 在远端稳定触发 CUDA illegal memory access：
  `gpu:/tmp/flagos-embedding-lora-a.EJr2yH/tdd-old-source.log`。
- 修复后 5/5 unittest 通过：原三组回归，加 rank 127/128/129、metadata stride、
  输入不变性，以及 `seg_indptr` authoritative/空 metadata 不访问。
- release 证据目录：
  `gpu:/tmp/flagos-embedding-lora-a-release.JvZPhH`，mode 0700；
  `release.log` 绑定 source/test commit 字节，`release-probe.log` 绑定 15 组
  shape×dtype 探针。
- release 探针在 A1/A2/A3 和 rank 65/130 controls 上覆盖 FP16/BF16/FP32、
  base/extra、rank 0 与第二 rank block，正确性 15/15；wrapper-inclusive 中位延迟
  范围 `0.006144–0.018432 ms`。
- 共 12 个编译变体，均为 4 warps、1 stage；最大 32 registers/thread，shared、
  spill、global/profile scratch、local load/store 全为 0。

验证环境：NVIDIA GeForce RTX 5070 Ti 16 GB，driver 610.57.04，Python 3.12.13，
PyTorch 2.13.0+cu130，Triton 3.7.1，CUDA 13.0。该结果仅是 NVIDIA 代理证据。

### E1：小 rank 使用 2 warps（拒绝）

单变量候选只把 `weights.shape[1] <= 64` 的 launch 从 4 warps 改为 2；rank 65/130
controls 保持 4 warps。候选未提交，source SHA-256 为
`6caa508e964fd3277f68eab6934f04d59a8aa210b0eebefca49a579b71542989`，screening
目录为 `gpu:/tmp/flagos-embedding-lora-a-warp-screen.g0Q62D`。

- 候选 unittest 5/5、A/B correctness 15/15。
- 五轮交替 AB/BA，`warmup=25, rep=100`：affected 几何均值
  `0.999994x`；FP16/BF16/FP32 分别为 `0.999329x`、`1.000653x`、
  `1.000000x`；最差 affected 点 `0.997988x`。
- controls 几何均值 `1.000650x`，范围 `1.000000–1.003906x`。
- 2-warps base 路径为 30 regs/thread，对比 S1 的 20；extra 路径为 38，对比
  S1 的 32。CTA register footprint 仍下降，且两边均无 spill/shared/scratch/local，
  但没有可测性能收益。

预声明门槛要求 affected 总体 `>=1.05x`、每 dtype `>=1.02x`、单点
`>=0.98x`，controls 在 `0.98–1.02x`。E1 未达到收益门槛，已拒绝并回退；没有 E1
commit 或 ZIP。

### S1 构建身份

| 项目 | 值 |
| --- | --- |
| source commit | `d101ebe56cd37dc5fbe423bd4d029227181461f8` |
| verification commit | `d101ebe56cd37dc5fbe423bd4d029227181461f8` |
| ledger commit | 本节所在 commit |
| 源文件 SHA-256 | `fb29244a40cffdf0d585615cb1dc9f9272063c2028792ac914e57e7a562a5f92` |
| 测试 SHA-256 | `e8532f0c55ca1fcd534a626446b449d62ebcce5722ba9541f303c303ba6586c3` |
| ZIP | `artifacts/competition/embedding_lora_a/s1-d101ebe/embedding_lora_a.zip` |
| ZIP SHA-256 | `49d7a33648c31d2b13e46c7e3dba8e7a4b88ecadce7da444c2ed5bac6b0ac09f` |
| ZIP manifest | 顶层 `embedding_lora_a.py`，4722 bytes；ZIP 4858 bytes |

`unzip -t`、UTF-8/语法、唯一普通 `.py`、basename、大小、成员字节与 source
commit 均已复验；打包器第二次运行状态为 `verified-existing`。

### 剩余风险与停止点

- `max_len` 仍必须覆盖 `seg_indptr` 的最大 segment 长度；它是题面 batch metadata
  和 launch grid 上界。S1 不读取 `seg_lens`，但不为错误的 `max_len` 引入 host
  同步修复。
- runtime rank loop、scalar control flow 和 2D grid 仍需八芯平台验证；不把 RTX
  5070 Ti 结果外推到其他芯片。
- S1 当前为“候选就绪、未提交”。未打开浏览器、未读取或消耗平台额度；上传必须
  重新执行本地验签、平台只读预检并取得用户针对完整 tuple 的当次一次性确认。
