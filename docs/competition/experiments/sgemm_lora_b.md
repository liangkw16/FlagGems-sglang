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

## S2：64×128 tile generic + 三 vendor（首投候选，≤2 次预算）

状态：release 门禁通过，候选就绪

NVIDIA sweep（`gpu:/tmp/flagos-sglb-sweep.b7LokE`，PID `110215`，01:23:30，
wall 1200s，脚本 SHA-256
`d415ae925f2c2e757477fae705635b36a1185e70d065562e91f5c5345f6f7ea7`）：
3 个代表 shape（bs4/1024/r64/o1024/fp16、bs8/512/r256/o4096/bf16+perm、
bs2/2048/r16/o512/fp32）× block_s{16,32,64}×block_n{64,128}×block_k{32,64}
×warps{4,8}×stages{1,3} 共 72 配置。**64/128/32/4/3** 为全 case 最优：
1.274x/1.191x/1.578x（对 S1 的 16/64/32/4/1），几何均值约 1.34x。generic
据此升级为 64/128/32/4/3。

vendor（均按平台证据预防性预置）：

- `_ascend`：capped grid-stride 折叠（1D `min(total,4096)`，逻辑 id 按
  batch→tile 分解；early return 改循环内守卫），tile 同新 generic，ieee dot
  保留（华为可执行）。
- `_iluvatar` / `_enflame`（同字节）：fp32 路径 split-fp16 三点积
  （`x.dtype == tl.float32` 编译期分支），低精度路径裸 dot；tile 64/128、
  stages 3（满足燧原 stages≥2）。昆仑不加 vendor（Task 12/09 dot kernel
  generic 经 SDNN 路径平台通过）。

新增回归 `test_vendors_cover_fold_and_split_fp16`：bs=8、max_len 2048、
out 4096（总 program 8192 > 4096，覆盖两轮折叠）× 三 dtype（fp32 按 1e-4）
× permutation，四模块对 reference。screening
`gpu:/tmp/flagos-sglb-vend.rOMrhP`（两次 Black 原地格式化回拷 + 一次测试
shape 断言修正），最终 PID/PGID `110817`（01:39:51，wall 900s，脚本
SHA-256 与 /tmp/flagos-sglb-screening.sh 一致），5/5 unittest（2.451s），
`screening.log` SHA-256
`5c2e057c3d79b8c199b493f4ac55e3bffdbe0e239fbdd625fd246400b5a9ecc6`。release
`gpu:/tmp/flagos-sglb-release.FPKgsy`，source/verification commit
`1e834e2e6e10e88f08a9494727684b974d849123`，5/5 unittest，`release.log`
SHA-256
`7f22a25294163e492647fd352a2a27638d54760d36b17fe5cd7b531f4faa9a8d`（尾行
标签沿用 SCREENING_OK，门禁项齐全）。generic blob
`9b1a9a6c98b2cb7f9647a51276fda261f39be1eda06a866e3e3ad561a68bb355`，ascend
`41416a252cbf5aaa5bf57dcb6de3da20fd7cdc52e2a471d995c4c603f68fd2de`，
iluvatar=enflame
`34fa4c8def315c6277fd087f6c85692273b2bc5a13e46bc135c0a495f6db41e4`，测试
`4aa2133b16c9b321480ef06ac7c68d26bcd10200036671399c2f83485f1a036d`。
canonical ZIP `artifacts/competition/sgemm_lora_b/s2-1e834e2/sgemm_lora_b.zip`，
SHA-256
`a85c8ae3f1b6e82ebde27b5ab9675c105e2fc2b40f5b38568094ff597973fb1e`，成员
generic + ascend/enflame/iluvatar，`unzip -t` 通过。本任务预算 2 次提交。

### S2 平台首投：7/8（昆仑超时崩溃）→ S2b 昆仑保守 vendor

S2 于 2026-08-25 01:43:55 CST 提交（submission `4403`，当日序号 `9`，额度
`22/30`→`21/30`，`file_url_sha256` 为
`5d0229231d69f7d9786cea363a0de23fab6443b97f2702ec16ee0e92e7b332a6`）。
七芯成绩优异：海光 46.4090x、国际 A 46.6575x、天数 34.1355x（split-fp16
vendor 生效）、国际 B 29.0540x、华为 18.4110x（fold vendor 生效）、沐曦
17.5935x、**燧原 4.0475x**（64/128 + stages 3 修复了 Task 09 型慢路径，
为本队燧原 dot 最佳）。昆仑芯在验证执行阶段超时（1830s/1800s）并发生子进
程崩溃（`Fatal Python error: Aborted`），无逐 case 结果——判断为 generic 的
64/128 tile + `num_stages=3` 在 XPU SDNN 编译路径上编译爆炸；缓存源码亦
记录昆仑 `num_stages` 属 invalid 参数。S2b 新增 `_kunlunxin` vendor：kernel
与 generic 相同，launch 退回 Task 09/12 平台通过的保守配置 32/32/32、
warps 4、stages 1。screening `gpu:/tmp/flagos-sglb-s2b.ci3uNg`，PID/PGID
`111772`（02:17:37，wall 900s，脚本 SHA-256 同 /tmp/flagos-sglb-s2b-
screening.sh），5/5 unittest（含昆仑 vendor 回归），`screening.log`
SHA-256
`fef4f429cf4cdffe681027c32ca94602f938460d9e4bb0be75602fd4550489a9`；昆仑
vendor blob
`bdfe676a86e5ac718d8f0565cb78374395af3ec614ea1f2407ff37a11e47ade3`，测试
`2a2e8c5af3c5b923ca14b735bfb5ef4f152d04baf672c41d86c1dbb2e7ef3a21`。release
`gpu:/tmp/flagos-sglb-s2b-release.XXXXXX 系目录`，commit
`4c184b63f2b2452d76d53ef78f672b38b1df5147`，`RELEASE_OK`，`release.log`
SHA-256
`be2faedf1f2c779bae5a694f18143a42394f71779daadebf45fd692a6c81fb41`。
canonical ZIP `artifacts/competition/sgemm_lora_b/s2b-4c184b6/sgemm_lora_b.zip`，
SHA-256
`3b022a2b66b170c99d3aa0f94c9f5f878489df1fad729bc43d94ba09af993db0`，成员
generic + ascend/enflame/iluvatar/kunlunxin 四 vendor。本提交为 2 次预算的
最后一次。

### S2b 平台结果：仍 7/8，昆仑结构性超时 → Task 23 停止（2 次用尽）

S2b 于 02:21:15 CST 提交（submission `4443`，当日序号 `10`）。七芯维持
高分（同 S2）；昆仑保守 vendor（32/32/32/stages1）仍以同一形态失败：
验证执行阶段超时（1830s/1800s）+ 子进程 `Fatal Python error: Aborted`。
结论：昆仑 SDNN 路径对本 kernel 的 ragged permutation/segment 间接寻址
结构编译爆炸，与 tile/stages 无关（Task 09/12 的规整 dot kernel 同配置
可通过）。Task 23 两次预算用尽，停止；最佳成绩为 7/8（invalid）。后续
昆仑方向需改写为规整 batched-GEMM 形式（如按 segment 长度分桶后连续
GEMM），非单变量可及。

## E7：昆仑 ragged 隔离 + 规整 FP32 BMM（高倍数冲刺重开）

状态：release 门禁通过，候选就绪，待实时 preflight 单次提交。

2026-08-26 11:04:55 CST 实时状态为团队 `SoulCoder`、Task 23 可提交、当日
额度 `30/30`，截止时间 `2026-08-27 19:59:59`。在新的高倍数冲刺预算下重开
Task 23，但不再调整 tile/stages：S2/S2b 已证明昆仑失败是 ragged metadata 与
`tl.dot` 同处一个 SDNN 编译单元造成的结构性编译爆炸。

E7 只改 `_kunlunxin` vendor：

- 无 dot 的 pack-X 把 ragged/permutation 行整理为 FP32 `[B,L,K]`；
- 无 dot 的 pack-W 选择 adapter 并转置为 N 连续的 FP32 `[B,K,N]`；
- regular BMM 只读取连续临时张量，采用 32×64×32、FP32 IEEE dot、stages 1；
- scatter kernel 再按 permutation 写回，并以 FP32 执行
  `base + scaling * product` 后 cast；空 segment 在读取非法 adapter sentinel 前
  退出，非零 `lora_ranks` 仍覆盖完整 stored K。

所有 kernel 用 1D 逻辑 program 分块启动，单次物理 grid 不超过 65535。已知最大
回归 `B=8,L=2048,K=64,N=4096` 的 BMM/scatter program 从 32×32 的 65536
降为 32×64 的 32768。新增 Kunlun-only 组合回归覆盖三 dtype、K65/N67、
segment `[17,0,1,33]`、permutation、非连续数据/metadata、rank0、空段非法
adapter、非零小 rank 仍 full-K 以及输入不变性。

screening 位于
`gpu:/tmp/flagos-sgemm-lora-b-e7-screening.ktAbZF`，base commit
`c19883b3921a44e59578e4ff7197f6e99e2139b1`，PID/PGID `128978`，wall
900s；launcher 的 `setsid` 行为在同机实测为 PID=PGID，逐字重放脚本
`replay.sh` SHA-256
`1464dded128704d6636efad3b4be3cf014b6ac5bea1c29f15f4978631fe33594`。
6/6 unittest 通过（1.861s），`screening.log` SHA-256
`a67f788c751122d3813309d14305950b5ee470d679702f233cd97ce587b6289a`。
screening 输入中 Kunlun vendor SHA-256
`55f29037308d31dcf7c171c6d3cd39c555cbeb18692f772c2f025cc9d42cb839`，测试
`d53bc6d2cec921e68269ba73995160a861c3d7a69571ba9deeeeac8c79228736`；
generic/ascend/enflame/iluvatar 分别保持
`9b1a9a6c98b2cb7f9647a51276fda261f39be1eda06a866e3e3ad561a68bb355`、
`41416a252cbf5aaa5bf57dcb6de3da20fd7cdc52e2a471d995c4c603f68fd2de`、
`34fa4c8def315c6277fd087f6c85692273b2bc5a13e46bc135c0a495f6db41e4`、
`34fa4c8def315c6277fd087f6c85692273b2bc5a13e46bc135c0a495f6db41e4`。

同目录在 RTX 5070 Ti（driver 610.57.04，16 GiB；Python 3.12.13、
PyTorch 2.13.0+cu130、Triton 3.7.1、CUDA 13.0）无竞争 workload 时，对上述
最大回归做 5 轮交替 wrapper-inclusive 代理基准：S2b 中位 0.9050ms，E7 中位
1.4817ms（约 1.64× 代价），E7 峰值显存增量 415236096B；无 OOM/编译异常。
基准脚本 SHA-256
`14e1a963c9f4dc6b83db17c62cc14212ccdd039e9cb073e3e50aeecf01ddb505`，
`benchmark.log` SHA-256
`f9412d95ef6dd4aea3fd8fb1e52db819bb377f7e038929a27394bdbbd4dc739e`。
该结果只作 NVIDIA 资源代理；晋级依据是昆仑旧结构完全不可执行，而 E7 完成完整
正确性路径。

source/verification commit
`1953fde4c21727b68ddf121092b1adf5e2aa97b4` 的 Git-object release 位于
`gpu:/tmp/flagos-sgemm-lora-b-e7-release.jYTCOa`，PID/PGID `129403`，wall
900s，`replay.sh` SHA-256
`05bf2221145a45ed6234926063b8ecfa7d9c85de21d43c15cf7eab56fc07e716`；
6/6 unittest 通过（0.675s），尾行为 `RELEASE_OK`，`release.log` SHA-256
`f97f3e83de583b0ad16eac4b3fa8fab0827739b936d605faf8ae5e7fa2dc0be8`。
screening 的 source/test 字节与 Git blob 逐项一致。

canonical ZIP
`artifacts/competition/sgemm_lora_b/e7-1953fde/sgemm_lora_b.zip`，SHA-256
`f5ac39fba6b3c6f100b925fb7849174abbfbbeef1262451653ddb940052825da`，
成员为 generic + ascend/enflame/iluvatar/kunlunxin，`--verify-existing` 与
`unzip -t` 均通过。以下为 E7 提交时预设，已由后文 E7b 现行止损取代：E7
只提交一次；昆仑 ≥39x 即停止 T23，
15–39x 才进入 E8 scatter/output 融合，valid 但 <15x 则直接切换 T13；若编译
失败只允许一次无-dot FP32 multiply-sum fallback，不再调 tile/warps/stages。

### E7 平台结果：7/8；昆仑 pack-W uni_sram 明确失败

E7 于 2026-08-26 11:34:36 CST 单次提交（submission `4996`，当日序号
`1`，额度 `30/30`→`29/30`，`file_url_sha256` 为
`1d43fac66f49bafb0b75da0df618e9c507dba6a8ec74e763f9ef552bcf924c4d`）。
七个既有通过芯片保持稳定：天数 33.9675x、沐曦 17.7750x、燧原 4.0945x、
海光 43.0095x、华为 17.7485x、国际 A 44.6920x、国际 B 29.1200x。

昆仑在 8.407s 内完成返回，不再出现 1830s 编译超时；5/5 correctness case 均在
`_pack_weights_kernel` 首次编译处失败，错误一致为
`OutOfResources: uni_sram PassManager::run failed`，平台建议缩小 block 或
stages。traceback 落在提交文件第 342 行 pack-W launch，说明 regular BMM 尚未
执行到；结构隔离方向有效，但 64×32 的 FP32 pack tile 单 program 资源过大。

下一候选不重提 E7，也不改其他七芯：把 pack-W 改为 1D 256-element FP32 copy，
并把尚未平台执行的 regular BMM/scatter 收敛到 Task 09/12 昆仑已验证的 32×32
资源形态；现有 Python 分块启动继续保证总 program 超 65535 时拆成多次 launch。
若该结构第二次仍失败，则按止损不把它复用到 T22。

## E7b：昆仑 pack-W 资源收口 + 32×32 regular BMM

状态：release 门禁与不可变 ZIP 验签通过，候选就绪，待实时 preflight。

候选 source commit
`a5c5d0bd74d399716e1e614c9b0e897e30cda034` 只改昆仑 vendor 与对应回归：
pack-W 从 64×32 的 FP32 tile 收敛为每个 program 拷贝一个 rank row 的连续
N256，移除向量运行时除模；regular BMM/scatter 的 N tile 从 64 收敛到 32，
保持 32×32×32、FP32 IEEE dot、stages 1。其他七芯实现不变。Kunlun vendor
SHA-256 为
`242bc191e6134fdf1c0ef201c8fb54dee644a74893cbb72d035ce27d6dffd834`，测试为
`ee1e268bb4d6c6786ccd8fc2ca22f7050c3efac73d76390f3e897be02da72e2f`；
generic/ascend/enflame/iluvatar 仍分别为
`9b1a9a6c98b2cb7f9647a51276fda261f39be1eda06a866e3e3ad561a68bb355`、
`41416a252cbf5aaa5bf57dcb6de3da20fd7cdc52e2a471d995c4c603f68fd2de`、
`34fa4c8def315c6277fd087f6c85692273b2bc5a13e46bc135c0a495f6db41e4`、
`34fa4c8def315c6277fd087f6c85692273b2bc5a13e46bc135c0a495f6db41e4`。

最终 screening 位于
`gpu:/tmp/flagos-sgemm-lora-b-e7b-final-screening.DS1Rie`，PID `130015`、
runner PID `130016`、PGID `130015`，逐字重放脚本 `replay.sh` SHA-256
`f2dd7d54074789f1d0119c241116d097c88ed95b337ddc6337ec65e0a9dbd232`；
6/6 unittest 通过（0.851s），尾行为 `SCREENING_OK`，`screening.log` SHA-256
`956de13d3d52dc49e9f5922b19031c87092bce44068e992a3c81d3624086f68e`。
输入前后 manifest 文件 SHA-256 均为
`09ce895aecc4eba9e4bb77cd25f4a42dce7311d711f0268f6dddfd9f6b0ebb47`。
环境为 RTX 5070 Ti、Python 3.12.13、PyTorch 2.13.0+cu130、Triton
3.7.1、CUDA 13.0。

同目录最大回归的 5 轮 wrapper-inclusive 代理基准中，E7b 样本为
1.6183/1.6145/1.6081/1.5995/1.6152ms，中位 1.6145ms；S2b 中位
0.9035ms，E7b 峰值显存增量 415236096B，尾行为 `BENCHMARK_OK`。baseline、
基准脚本、重放脚本、日志 SHA-256 分别为
`bdfe676a86e5ac718d8f0565cb78374395af3ec614ea1f2407ff37a11e47ade3`、
`33d06022c5045c763c801e723d49d067bd7b74366197e08fdfe1c48bff9e8b89`、
`25ccf6f163d8e804ae584ff383b01420aeca4abda4d7741b7ec97a45a5e29771`、
`62f8ef8bca7643296d3d1036eec4b73c29b2ae5df215c5ba4fa570da6bdb7372`；
输入前后 manifest 文件 SHA-256 均为
`f812db0a1b7568070a3740e8fae4635f436ed8f20479d7cba5e22e1e956e2d2c`。

source/verification commit 均为
`a5c5d0bd74d399716e1e614c9b0e897e30cda034` 的 Git-object release 位于
`gpu:/tmp/flagos-sgemm-lora-b-e7b-release.fTJ385`，PID/PGID `130757`，wall
900s；`replay.sh` SHA-256
`c319e0ee991c0f274f888dad7422651d266cc31c2ab1f78fcc4a26930a0be1cc`。
6/6 unittest 通过（0.678s），尾行为 `RELEASE_OK`，`release.log` SHA-256
`2807e911980342ec315847186c1961aae987ba4a0dc4e0fce1e1c690fb747d0c`；
release 前后 manifest SHA-256 均为
`b8aff5dfa5498d858d43638d3ea0589d36ba58dca8dcec878f1559ebfad62693`，
逐字比较一致。

canonical ZIP 路径为
`artifacts/competition/sgemm_lora_b/e7b-a5c5d0b/sgemm_lora_b.zip`，大小
35490B，SHA-256
`19a46d4041d4b4b5ac7ffb73a5b692c66ff72a8def122cc30c805dbb19256ed4`，
成员为 generic + ascend/enflame/iluvatar/kunlunxin；`--verify-existing`、
`unzip -t` 和成员白名单均通过。E7b 定位为一次 validity/结构诊断：若仍
invalid，立即停止 T23 并跳过 T22；若 valid 但昆仑 <15x，停止 T23；只有昆仑
>=15x 且总分达到实时榜首 70% 才进入后续放大池。

### E7b 平台结果：仍 7/8；最小 pack-W 仍触发 uni_sram → T23/T22 止损

E7b 于 2026-08-26 14:01:22 CST 单次提交（submission `5038`，当日序号
`2`，额度 `29/30`→`28/30`，`file_url_sha256` 为
`281b8743c65e2489e7785d4774b123f391560309e9d6b4c6f72f8da8fa0b319b`）。
确认命令未继承远端 hostname 环境，因此工具内 remote verification 显示
unavailable；提交受理后从平台返回的可信 URL 只读下载验签，大小 35490B、
SHA-256
`19a46d4041d4b4b5ac7ffb73a5b692c66ff72a8def122cc30c805dbb19256ed4`，与本地
canonical ZIP 完全一致，未重试提交。

七个既有通过芯片仍稳定：天数 33.7595x、沐曦 18.0045x、燧原 3.9260x、
海光 43.9490x、华为 17.6765x、国际 A 46.0085x、国际 B 28.7700x。昆仑在
8.119s 内终态失败，5/5 correctness case 均在 `_pack_weights_kernel` 第一次
编译处报相同
`OutOfResources: uni_sram PassManager::run failed`；traceback 为提交文件第
337 行，case 0 的 grid 为 16，最大 case 的 grid 为 4096，regular BMM 仍未
执行到。

结论：即使 pack-W 已降为每 program 单 rank row × N256 的连续 FP32 copy，
该 kernel 仍不能通过昆仑 SDNN 编译；继续缩 tile/warps/stages 不再具有结构新意。
按 E7b 预设止损，Task 23 永久停止，不做 no-dot fallback；由于 regular BMM 从未
在昆仑执行成功，Task 22 同步跳过。高倍数主线转入 Task 13 两阶段规整 BMM。

## E8：移除失败 pack-W，framework layout prep + Triton GEMM

FlagOS 官方仓库复扫发现 Kunlun 上游 matmul 已使用 host-side transpose/layout
准备，而规整 batched GEMM 走直接 `[B,K,N]` 连续权重。E8 因而不是继续缩小两次
失败的 pack-W tile，而是彻底删除该 kernel：对 weights 无条件执行一次
`transpose(1, 2).contiguous()` 得到 `[A,K,N]`；pack-X、safe adapter、
FP32 IEEE GEMM、scale 与 scatter 仍全部由 Triton kernel 执行，无 fallback、
设备判断、异常捕获或 Torch GEMM。其余七芯字节冻结。

- source/verification commit：
  `e8cd539f6ec62c7fc1fcbd46aaceab1b5bdd669a`；Kunlun vendor/test SHA-256
  分别为
  `332a13ee7064d6ca8d58a586d07df4b41d7a6e1556e503d4ee5acad1ac2b11c6`、
  `4c0867258140d315bd95ac7c876da25edbcf29e5e02c6e43b4213d4967adf77e`。
- safe-adapter kernel 在读取 metadata 前先屏蔽空段，并把空段或 rank=0 映射
  到合法 adapter 0；`weights.shape[0]==0` 在 layout copy 前返回。新增 rank0
  adapter 超出 weights 和零 weights/全空段回归，消除隐藏 OOB。
- Git-object release：`gpu:/tmp/flagos-sgemm-layout-release.fN0LT4`；py_compile、
  Black、isort、flake8 与完整 unittest 10/10 通过，`release.log` SHA-256
  `87f823cd239b8ef1a658ec409d107de2da3c5b57e83829cefd41d534bfbd26bf`。
- K256/64 adapters 的 FP16、BF16、FP32 独立编译/数值门全部通过；regular
  BMM 为 52 registers、8 KiB shared、zero scratch。脚本 SHA-256
  `f13b2e2167781fa3fc1a5910bfc5ca42e5f61a45568d69e7d918c031d0f5cf9b`。
- RTX 5070 Ti 最大回归 `B8/L2048/K64/N4096` 的五轮独占代理中位
  1.6406ms，S2b 为 0.9025ms，峰值增量 411,042,304 bytes；比 E7b 的
  1.6145ms 仅慢 1.6%，资源不恶化。benchmark/baseline SHA-256 分别为
  `33d06022c5045c763c801e723d49d067bd7b74366197e08fdfe1c48bff9e8b89`、
  `bdfe676a86e5ac718d8f0565cb78374395af3ec614ea1f2407ff37a11e47ade3`。
- canonical ZIP：`artifacts/competition/sgemm_lora_b/e8-e8cd539/sgemm_lora_b.zip`，
  35,030 bytes，SHA-256
  `d0cabb0abca65e2d1db5cfbbe59e6a7621a4f6cb4e30116d24e45a05c9de0bb3`；
  五成员、`verified-existing`、`unzip -t` 全部通过。
- 一次提交止损：只有 8/8 且昆仑 ≥0.1x 才恢复 Task 23 有效分；按 E7b 七芯
  合计 192.094x 推算，门槛即约 24.024x 平均。若仍编译/资源失败，或 valid
  但昆仑 <15x，永久停止，不再调 tile/warps/stages。

### E8 平台结果：仍 7/8，但失败点推进到最终 scatter

- 2026-08-26 20:37:27 CST 单次提交，submission `5130`、当日序号 `16`；
  额度 `15/30`→`14/30`，平台回读 35,030 bytes 与 canonical SHA-256
  `d0cabb0abca65e2d1db5cfbbe59e6a7621a4f6cb4e30116d24e45a05c9de0bb3`
  完全一致；`file_url_sha256`
  `e13a2fcbcb56184c02f96045b7d6c16810e285cfe8aeb279e81131824d9d8b1d`。
- 七芯继续通过：天数 33.9385x、沐曦 17.8135x、燧原 3.9985x、海光
  47.3020x、华为 17.2215x、国际 A 46.0895x、国际 B 28.7610x，合计
  195.1245x。
- 昆仑 case 0 已越过被删除的 pack-W，并将 regular BMM 编译/launch 推进到
  后续 `_scatter_add_kernel`；该 kernel 在提交文件第 370 行首次编译时以
  grid `(2,)` 报 `uni_sram PassManager::run failed`。case 1–4 的 illegal
  memory/copy 错误均发生在首错污染设备之后，不作为独立根因。
- 这不是 E7/E7b 的失败复现：已证明 layout prep 绕过 pack-W 且 regular BMM
  不再是首个阻断点。下一且最后的结构候选仅把二维 ragged scatter 改成一维
  pointwise Triton scatter；不改 BMM、tile、warps 或其余七芯。按本轮七芯合计，
  昆仑只需 0.1x 即约 24.403x 平均。

## E9：昆仑 row-wise pointwise scatter

状态：Git-object release 与不可变 ZIP 门禁通过，待实时 preflight。

source/verification commit
`ee892715ba0c9ded9ac0b5fac43c97d1c7a45ff6` 只改昆仑最终 scatter 与回归；
E8 已在平台越过的 layout copy、safe-adapter、pack-X 和 regular BMM 均保持不变，
其余七芯源码逐字节冻结。旧 32×32、1024-lane 二维 scatter 改为每个 program
固定一个 `(batch, token, col_block)` 的 1×256 pointwise scatter；program-id 的
除模保持标量，唯一向量是连续列。wrapper 先完成全部 BMM，再按最多 65535 个
program 分块启动 scatter。空段、rank0、permutation 和尾列采用 staged mask，
不在无效 metadata 上形成实际 load。

- Kunlun vendor/test SHA-256 分别为
  `3595148d5c4d1aea4d6eff102675b61d754b14cd4724e006affe074f9141b1f2`、
  `f2318c963474fcb738092f759e555d2257aa2e0e7927789a44cc1fa6cd8d50d4`；
  generic/ascend/enflame/iluvatar 仍为 E8 的
  `9b1a9a6c98b2cb7f9647a51276fda261f39be1eda06a866e3e3ad561a68bb355`、
  `41416a252cbf5aaa5bf57dcb6de3da20fd7cdc52e2a471d995c4c603f68fd2de`、
  `34fa4c8def315c6277fd087f6c85692273b2bc5a13e46bc135c0a495f6db41e4`、
  `34fa4c8def315c6277fd087f6c85692273b2bc5a13e46bc135c0a495f6db41e4`。
- Git-object release 位于
  `gpu:/tmp/flagos-sgemm-flat-release.MTOu37`，PID/PGID `154498`、wall 420s；
  RTX 5070 Ti、Python 3.12.13、PyTorch 2.13.0+cu130、Triton 3.7.1、
  CUDA 13.0。py_compile、Black、isort、flake8 与完整 unittest 10/10 通过；
  `release.log` 尾行为 `RELEASE_OK`，SHA-256
  `e053bc70e5593cfa72413390dcb3f77d9347a7484db3433ec37ef0e8450babd6`。
- 定向脚本覆盖 N=33/64/255/256/257、permutation on/off 和三 dtype，共
  30 个组合；scatter 编译资源为 20 registers、0 shared、0 global/profile
  scratch，TTIR 无 vector div/rem 或 dot。脚本 SHA-256
  `957585f09163cef7c1b0bacc4970ebbb481b580ffc10840193363cd126897d31`。
- 最大回归 `B8/L2048/K64/N4096` 的 5 轮独占 E8/E9 代理中位分别为
  1.6387/1.6307ms，E9/E8=0.9951，峰值增量 411,042,304 bytes；benchmark
  脚本 SHA-256
  `d412e155db37040e5b7ce311a66efcdd8f122af695b34166c0ae9002f2641841`。
- canonical ZIP
  `artifacts/competition/sgemm_lora_b/e9-ee89271/sgemm_lora_b.zip`，35,057B，
  SHA-256
  `feaddd39d9472fd1d260968ad7bd3f3b98495ce51123da6fe5b10f9720cae92a`；
  五成员、`verified-existing`、`unzip -t` 与成员白名单全部通过。

E9 只提交一次：晋级门为 8/8 且昆仑 ≥0.1x；按 E8 七芯合计 195.1245x，
最低有效平均约 24.403x。若昆仑仍在该 scatter 编译/正确性失败，则永久停止
Task 23 与 Task 22，不再用额度试 BLOCK/warps/grid。

### E9 平台结果：7/8；scatter 越过编译但运行期非法内存

- 2026-08-26 20:54:44 CST 单次提交，submission `5132`、当日序号 `17`，
  额度 `14/30`→`13/30`；上传后无认证回读 35,057B、SHA-256
  `feaddd39d9472fd1d260968ad7bd3f3b98495ce51123da6fe5b10f9720cae92a`，
  与 canonical ZIP 完全一致。`file_url_sha256` 为
  `a012a7351cde56eef4ec99428bb44d294c2e070be55c5f798b8ea58db71efff9`。
- 七芯通过：天数 33.8060x、沐曦 18.0130x、燧原 4.1075x、海光
  51.0625x、华为 18.5675x、国际 A 45.8985x、国际 B 28.9615x，合计
  200.4165x；若昆仑仅过 0.1x，平均也会达到约 25.0646x。
- 昆仑不再出现 `uni_sram` 编译失败。case 0 在 7.485s 后于
  `assert_close` 的首次同步点报 error 700 illegal memory access；由于设备错误是
  异步上报，traceback 不能定位到具体 launch。case 1–4 均在首错后于 reference
  clone/copy 失败，只视为设备污染，不作为独立根因。
- 按 E9 预设门禁，Task 23 与依赖同结构的 Task 22 停止，不用平台额度继续猜
  BLOCK/warps/grid。后续只允许只读核对 FlagGems/FlagTree 一手实现；没有能锁定
  新根因的源码证据，不重开候选。

## E10：昆仑官方 mask-zero lowering（一次性重开）

状态：Git-object release 与不可变 ZIP 门禁通过，待实时 preflight。

E9 止损后的官方源码核对锁定了一个新根因。FlagTree XPU 后端默认关闭
`is_use_mask_zero`；官方验证文档说明默认 masked load/store lowering 可能无法保持
`other=0`，粗粒度 DMA 的被屏蔽尾 lane 还可能越界，而开启该选项会把真实 pointer
与 mask 传入 GM2LM、offset、legalize、mask、unroll 和 LLVM lowering。FlagGems
Kunlun 的 `nll_loss`、`nonzero`、sort 与 softmax 对同类尾 mask、间接地址和 masked
scatter 均在 launch 显式启用该选项。

E9 的四个 kernel 都在 `assert_close` 首次同步前异步启动，因此 error 700 不能只归因
于 traceback 最近的 scatter。E10 把这个**单一后端内存语义变量**加到三个非-dot
launch：safe-adapter、pack-X 与 scatter；regular BMM 的 `tl.dot`/SDNN 路径逐字节
冻结。safe-adapter 的 `B<256` 尾块、pack-X 的 token/rank 尾块以及 scatter 的
inactive row/N 尾块都因此使用官方 mask lowering。算法、BLOCK、grid、warps、
stages、stride 和其他七芯源码均不变。

- source/verification commit：
  `31a6789f9fdca7e032f6a9294c5adcb1540204da`；相对 E9 的 Kunlun vendor 只有三行
  `is_use_mask_zero=True`，vendor/test SHA-256 分别为
  `9a02a1cbb311246c1f45c2d2b14041cf597ed570f05eadb3bd289f1eab3b02fd`、
  `38f31fde3e7d354486a54230797a093ecb55eb5274a693aa66dd91ceb198287d`。
- generic/ascend/enflame/iluvatar 继续冻结为
  `9b1a9a6c98b2cb7f9647a51276fda261f39be1eda06a866e3e3ad561a68bb355`、
  `41416a252cbf5aaa5bf57dcb6de3da20fd7cdc52e2a471d995c4c603f68fd2de`、
  `34fa4c8def315c6277fd087f6c85692273b2bc5a13e46bc135c0a495f6db41e4`、
  `34fa4c8def315c6277fd087f6c85692273b2bc5a13e46bc135c0a495f6db41e4`。
- Git-object release：`gpu:/tmp/flagos-sgemm-maskzero-release.ll3Twr`，mode 0700，
  PID/PGID `154817`，wall 180s；py_compile、isort、flake8、2/2 CPU mock routing
  与独立 AST 门禁通过，尾行为 `RELEASE_OK`。`release.log`、`replay.sh`、输入 tar
  SHA-256 分别为
  `53b1e827b4d3bdba8b52fbefadead47c76ba32dc3d91c56651d775f80fcfeaa5`、
  `e951d1e5ff1d614ed2f2214f823828a8d4e68a708d7696edb4778313e255e262`、
  `b1fe643a495ce29e9518057bbc3eb7ff60f62e967644030058e09bd522b4afc8`；前后 manifest
  SHA-256 同为
  `a4e992b73ce30559452ebf6c15ffda51dfa33a98c76659826363a64e8b104613`。
- Black 25.12.0 在本地对 source/test 精确字节通过。NVIDIA Triton 不接受 XPU 私有
  launch option，因此 E10 vendor runtime 明确标为未验证；E9 已验证同字节 kernel
  body 的 10/10 correctness/30 定向组合和资源，本轮不以删掉 option 的代理源码冒充
  exact release。FlagTree 另要求足够新的 XRE 与 `dma_excp_mask`，只能由平台验证。
- canonical ZIP：
  `artifacts/competition/sgemm_lora_b/e10-31a6789/sgemm_lora_b.zip`，35,162B，
  SHA-256
  `505c73a8b6aab2e15ea7c3a40a35ea2ced7eb6b83ded58147dde586c44becfa5`；五成员、
  `verified-existing`、UTF-8/语法/成员白名单均通过。

E10 只提交一次：仅当 8/8 且昆仑 ≥0.1x 才恢复有效分；按 E9 七芯合计
200.4165x，最低有效平均约 25.0646x。若仍 error 700、编译失败或昆仑低于门槛，
Task 23/22 永久停止，不追加 BMM flag，也不再调 tile/grid。
