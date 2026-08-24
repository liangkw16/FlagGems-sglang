# Task 23 `sgemm_lora_b` 实验记录

## S0：generic baseline

状态：已被 S1 的 segment 边界修复替代；历史 ZIP 保持不可变，未提交平台

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

## S1：以 `seg_indptr` 为准并跳过无效 token blocks

状态：候选就绪，尚未提交平台

验证时间：2026-08-24 06:02–06:13 CST

### 根因与最小修复

S0 在确认 segment 是否为空前先读取 `weight_indices[b]` 和
`lora_ranks[weight_index]`。题面 reference 先比较相邻两个 `seg_indptr` 并跳过
空 segment；因此空段的 adapter metadata 即使不应被访问，S0 仍可能越界读取。
S0 还用 `seg_lens[b]` 决定有效 token，而 reference 的 segment 边界只由
`seg_indptr` 定义。

S1 先读取 `seg_indptr[b:b+2]`，对
`token_block * BLOCK_S >= end - start` 的 program 立即返回，再读取 adapter
index、rank、x 和 weights；同时删除 kernel 中的 `seg_lens` 参数。该 guard 与固定
SGLang 上游删除 ragged segment 无效 GEMM 的做法一致，并把空段保护提前到 metadata
读取前。`max_len` 仍只作为 host grid 上界，避免引入 device-to-host 同步。

### TDD 与 release 验证

- 新回归使用 strided `seg_indptr=[0,17,17,18]`，故意令
  `seg_lens=[1,0,1]` 失配，并给空 segment 一个越界 adapter 哨兵；S0 在独立进程
  稳定触发 CUDA illegal memory access，退出码 1：
  `gpu:/tmp/flagos-sgemm-lora-b-tdd.2A07xU/old-source.log`，日志 SHA-256
  `f859498a...6cbda46`。即使后端未 fault，S0 也只会更新首行而不是 reference 的
  17 行，因此回归不依赖越界行为。
- 修复后的 screening 与 release 均为 4/4 unittest 通过；新 case 同时覆盖
  `S=17` 两个 token tiles、`K=65` 三个 K tiles、`N=67` 两个 N tiles、metadata
  stride，以及空段 metadata 不访问。Black 79、isort、flake8、py_compile 和文件
  哈希门禁全部通过。
- release 证据目录：`gpu:/tmp/flagos-sgemm-lora-b-release.biIhDM`，mode 0700；
  静态/单测任务 PID/PGID `80558`，性能探针 PID/PGID `80652`。`release.log`、
  `release-ab.log`、`release-precision.log` 的 SHA-256 分别为
  `63b22c5f...ec6514`、`fa10de89...7cb20`、`2e20d756...f624`。
- release A/B 对四组 segment 分布的三种 dtype 共完成 12/12 reference correctness；
  五轮交替 AB/BA，`warmup=25, rep=100`。20 次 wrapper 批量计时消除短 kernel 的
  计时量化后，ragged affected 几何均值为 `2.9339x`；FP16/BF16/FP32 分别为
  `2.3901x`、`2.3833x`、`4.4336x`，范围 `1.8812–4.5407x`。
- 等长 controls 的几何均值为 `0.9845x`，范围 `0.9638–1.0012x`。FP16
  `(16 segments × 64 tokens, K=32, N=1024)` 约慢 `3.62%`，说明 guard 对完全
  等长的小 kernel 有可测控制流成本；ragged 分布收益和公开 reference correctness
  修复共同支持保留 S1，账本不把该 trade-off 隐藏为纯性能无回归。
- base/candidate 各编译 6 个变体；最大均为 106 registers/thread、10,240 bytes
  shared、4 warps、1 stage，spill、global scratch、local load/store 全为 0。

验证环境：NVIDIA GeForce RTX 5070 Ti 16 GB，driver 610.57.04，Python 3.12.13，
PyTorch 2.13.0+cu130，Triton 3.7.1，CUDA 13.0。该结果仅是 NVIDIA 代理证据。

### S1 构建身份

| 项目 | 值 |
| --- | --- |
| source commit | `222dd77bc855c16c8ce87b61a723cf5f573ef540` |
| verification commit | `222dd77bc855c16c8ce87b61a723cf5f573ef540` |
| ledger commit | 本节所在 commit |
| 源文件 SHA-256 | `34e752d5ce942d03c0c70e24c0a26b10df90a1fc841888b915c207219b11a269` |
| 测试 SHA-256 | `11e286d6ec8d954f7ae5f7d1b30df1bcb0c0250826264dbf46a6d9b86b179bb1` |
| ZIP | `artifacts/competition/sgemm_lora_b/s1-222dd77/sgemm_lora_b.zip` |
| ZIP SHA-256 | `4223927a48608887b322b87611001f65102cd0e6fa2bf432b4efb50a7773a03f` |
| ZIP manifest | 顶层 `sgemm_lora_b.py`，5091 bytes；ZIP 5219 bytes |

`unzip -t`、UTF-8/语法、唯一普通 `.py`、basename、大小和成员逐字节哈希均已
复验；打包器第二次运行状态为 `verified-existing`。

### 剩余风险与停止点

- `max_len` 必须覆盖 `seg_indptr` 的最大 segment 长度；S1 不再信任
  `seg_lens`，但不会为错误的 host grid hint 引入同步修复。
- 等长 FP16/BF16 小 shape 存在最多约 `3.62%` 的 NVIDIA 控制流成本；实际 LoRA
  batch 的 segment 分布和其余七类后端表现只能由平台验证。
- 2D grid、runtime segment metadata、permutation 和 IEEE dot 仍未在八芯实测；
  当前不预建 vendor 分支。
- S1 为“候选就绪、未提交”。未打开浏览器、未读取或消耗平台额度；上传前必须
  重新验签 ZIP、读取平台实时 tuple，并取得用户针对该精确产物的一次性确认。

## E2–E3：宽 `BLOCK_N` 候选，拒绝

固定 SGLang 上游对宽输出使用 `BLOCK_N=256`，因此 E2 只在
`output_dim>=1024 && output_dim%256==0` 时把 S1 的 N64 改为 N256；E3 是一次
预先声明的减半实验，在 `output_dim>=1024 && output_dim%128==0` 时使用 N128。
两者都保持 S1 的 kernel、segment guard、BLOCK_S/K、4 warps、1 stage、IEEE dot
和 fallback 路径不变。两次 screening 均未过门禁，未 commit、未生成 ZIP；
最终源码和测试已逐字节恢复 S1。

共同矩阵包含 4/4 unittest、21 个额外正确性 case、12 个 affected、6 个
control 和 15 组 affected resource。性能使用 batch20、六轮严格 3 AB/3 BA、
`warmup=25, rep=100`；晋级要求整体 `>=1.05x`、每 dtype `>=1.02x`、任一点/
单组 `>=0.98x`、controls `0.98–1.02x`，且不超过 160 registers、40 KiB shared、
至少 2 CTA/SM、0 spill/scratch/local load-store。三 dtype、permutation、ragged、
非连续输入、K65、空段越界 metadata 和完整 K 语义均对题面 reference 通过。

### E2：N256

候选源码/测试 SHA-256 为
`47ff05cb9cc5791946c5c7bba3a7072317e47c097f1c0bf817edfb7dba0f17d6` /
`a4874927b65b1c1fc2d91a20c9110cfeeb7af1985c6d61f6947aa1214eda5651`。

| 指标 | 结果 |
| --- | ---: |
| affected 几何平均 | `0.980470x` |
| FP16 / BF16 / FP32 | `0.904148/0.897662/1.161318x` |
| 最差 affected 点 / 单组 | `0.717499/0.715185x` |
| control 几何平均 / 范围 | `1.000000x / 1.000000–1.000000x` |

低精度 ragged-permutation 为 `0.717–0.723x`；FP32 虽有收益，但不允许按 dtype
事后切分。K65 FP16/BF16 编译物达到 255 registers、17 KiB shared 和 22 spills；
FP32 最高 254 registers、34 KiB shared，故性能和资源门禁均失败。

远端目录 `gpu:/tmp/flagos-sgemm-lora-b-e2.faH2bL`，PID/PGID `93118`，运行时间
09:13:47–09:14:22 CST。harness/gates、原始 JSON、gates 日志 SHA-256 分别为
`5689d5cbc5d4e31f88e54e54e9c05775078ca1f1878a59b43bc32c19e94fb816`、
`966089121417c17aa10c37d0edbf97f3db9ff3c68b53547b3c17245336a3b23e`、
`d3b11ee1d792197fc30977e348bfab9870b2ea512c07bed003d9c1ab2b002090`、
`e3e2efe5157eb69402b7d9887c18b75f4605e6b0e7b44e5bf65147da9ca8e58c`。

### E3：N128

候选源码/测试 SHA-256 为
`46b32c1a8ecbf362021848aea48d0ba7aaf0fba538fae6b51e05dee3411bdf88` /
`a4874927b65b1c1fc2d91a20c9110cfeeb7af1985c6d61f6947aa1214eda5651`。

| 指标 | 结果 |
| --- | ---: |
| affected 几何平均 | `0.894789x` |
| FP16 / BF16 / FP32 | `0.857401/0.854931/0.977342x` |
| 最差 affected 点 / 单组 | `0.584699/0.584614x` |
| control 几何平均 / 范围 | `0.999984x / 0.999901–1.000000x` |

低精度 ragged-permutation 进一步降到约 `0.585x`。候选最高 168 registers、
18 KiB shared；K7 permutation 的 FP16/BF16 仍有 2 spills，性能和资源均失败。
不再尝试 dtype/shape 特判或更多 N tile。

远端目录 `gpu:/tmp/flagos-sgemm-lora-b-e3.1DnsbT`，PID/PGID `93575`，运行时间
09:17:47–09:18:18 CST。harness/gates、原始 JSON、gates 日志 SHA-256 分别为
`c6e0aab3916d7b0d98db32e7772a70d9be3d852d592c079b77bd71b0f01f0fde`、
`41fa9e2146bc50b33fe163a0174d0d85479fd76f07e43c7e2fe3e34d26e33651`、
`711f0083250912590fc433ef492317093bc2f83f5d7e4ae5071b4a721916bd04`、
`45f04bbce04e80176bc66355a3b8ca9686c9e60e502ab37c1831437b8c3b02b5`。
