# Task 22 `qkv_lora_b` 实验记录

## S0：generic baseline

状态：已被 S1 的 segment/slice 边界修复替代；历史 ZIP 保持不可变，未提交平台
验证时间：2026-08-24 CST

### 契约

- 接口：
  `qkv_lora_b(x, qkv_lora_b, batch_info, output_offset, max_qkv_out_dim, base_output)`。
- `r = qkv_lora_b.shape[-1]`，且
  `n_slices = output_offset.numel() - 1`；x shape 必须为
  `[S, n_slices*r]`。
- slice i 独立读取 `x[:, i*r:(i+1)*r]`，写入
  `[output_offset[i]:output_offset[i+1]]`；slice 宽度允许不相等。
- 每个 segment 由 `weight_indices` 选择 adapter，由 `scalings` 选择缩放；
  `permutation[start:end]` 存在时同时决定 x 读取行和 output 写回行。
- `lora_ranks[w_idx] == 0` 时 no-op；题面 reference 对其他非零值使用完整 r，
  不截短 K。
- 结果从 `base_output.clone()` 开始，以 FP32 accumulator 计算 LoRA 增量并
  加到 FP32 base 值，最终 cast 回 base dtype；base 与其他输入都不变。
- 容差：FP32 `1e-4/1e-4`，BF16 `1.5e-2/1.5e-2`，
  FP16 `1e-2/1e-2`（atol/rtol）。
- 支持八芯：天数、沐曦、燧原、海光、昆仑芯、华为、国际通用 A/B。
- 缓存目录显示提交窗口为 2026-08-20 20:00 至 2026-08-27 19:59:59，
  最低加速比 `0.1x`；提交前以平台页面为准。

### `batch_info` 字段

`bs/max_len/seg_lens/seg_indptr` 定义 segment grid 与区间；
`weight_indices` 映射 adapter；`lora_ranks` 决定 rank0 no-op；`scalings`
提供 adapter 缩放；可选 `permutation` 提供物理 token 行。

### 固定参考

- SGLang
  [`8014d9d/qkv_lora_b.py`](https://github.com/sgl-project/sglang/blob/8014d9d062c3cc5d393596ecdf2f7009191965df/python/sglang/kernels/ops/gemm/qkv_lora_b.py)：
  完整读取 kernel、wrapper、Q/K/V packed grid、permutation 与 scaling。
- 只读复用了当前 `sgemm_lora_b.py` 已验证的真实 stride、permutation、
  base clone 与 IEEE dot 骨架，没有修改该文件。
- 相比固定 SGLang wrapper，S0 不接受外部 `n_slices=3` 默认值，严格从
  `output_offset.numel()-1` 推导；x slice 起点始终用完整 r，避免被 adapter
  metadata 的非零 rank 改变。

### 构建身份

| 项目 | 值 |
| --- | --- |
| 源文件 | `src/flaggems_sglang/ops/qkv_lora_b.py` |
| 源文件 SHA-256 | `f445a6ca930103beb8912218feba08c51430dec2e30b321d1444842245f7e04b` |
| 测试 SHA-256 | `39d5efb720a38ddc80d58f461d6d8bfbb42aae737826dd7b75fe5de8cb0ec238` |
| 源码 commit | `b05bfeb` |
| ZIP | `artifacts/competition/qkv_lora_b/s0-b05bfeb/qkv_lora_b.zip` |
| ZIP SHA-256 | `ec395510ac56ccd289f53f95dab584c9502950e7a8b5d30d0681a3e2a1ab8a30` |
| 远端证据目录 | `gpu:/tmp/flagos-batch2.SQaIX2`，mode 0700 |
| 平台 | 未提交；未经用户当次确认不得上传 |

### 唯一候选配置

- 固定 `BLOCK_S=16`、`BLOCK_N=64`、`BLOCK_K=32`，4 warps、1 stage。
- grid 为 `(token/output tiles, n_slices, bs)`；实际 slice 宽度由
  `output_offset` 动态 mask。
- x、weights、cloned output、output_offset 与所有 batch tensor 都使用真实
  strides；主数据地址 stride 转为 64-bit。
- accumulator、base load 和 scaling 均转 FP32；FP32 dot 使用
  `input_precision="ieee"` 禁用 TF32，store cast 回 base dtype。
- 除 `base_output.clone()` 外无 PyTorch 核心计算；无 fallback、设备判断、
  autotune 或 vendor 文件。

### 正确性与静态检查

本地 Python `py_compile` 通过。远端 RTX 5070 Ti 16 GB 环境：Python
3.12.13、PyTorch 2.13.0+cu130、Triton 3.7.1、CUDA 13.0、compute
capability 12.0。

远端公开接口 unittest：2/2 通过，覆盖：

- FP16、BF16、FP32 accumulator/cast 容差；
- 三个非等宽 slice `[5,3,7]` 与动态两个 slice `[4,7]`，证明未硬编码 3；
- 普通、空 segment、rank0、正负 scaling；
- permutation 同时用于 x 读取与 output 写回；
- 非连续 x/weights/base/output_offset 的真实 strides；
- 输出不 alias base，x、weights、base、offset 与 permutation 均不变。

FP32 非 2 次幂 rank=7 在 `1e-4` 容差通过，验证 IEEE dot。Black 79、
isort、flake8 均通过；上述结果只证明 NVIDIA 代理路径。

### NVIDIA 代理性能

wrapper-inclusive；每项先验证正确性，再用
`triton.testing.do_bench(warmup=20, rep=50)`。

| dtype | `(S,bs,r,offsets)` | rows | S0 (ms) | reference (ms) | speedup |
| --- | --- | --- | ---: | ---: | ---: |
| FP16 | `(32,4,16,[0,64,96,128])` | linear | 0.007779 | 0.620898 | 79.819x |
| BF16 | `(128,8,32,[0,128,192,256])` | linear | 0.008236 | 1.242405 | 150.855x |
| FP32 | `(256,16,64,[0,256,384,512])` | linear | 0.016469 | 2.232122 | 135.532x |
| FP32 | `(64,4,7,[0,33,80])` | permuted | 0.008696 | 0.427269 | 49.133x |

ZIP 由 commit `b05bfeb` 的算子子树直接生成，仅含顶层 UTF-8
`qkv_lora_b.py`。`unzip -t`、10 MB、成员名和逐字节 SHA-256 门禁均通过。

### 已知风险与下一步

- 平台未公开 correctness/benchmark shape；代理性能不能证明八芯排名。
- 3D grid、runtime slice offsets、scalar control flow 和 IEEE precision 尚未被
  八种编译器全部验证。
- S0 信任 `max_qkv_out_dim` 不小于最大 slice 宽度，与固定 wrapper 一致。
- 若首次平台仅单芯失败，保持 generic 与已通过芯片不变，只做最小 vendor
  override；下一门禁是用户针对上述 ZIP 路径、哈希和实时额度作当次确认。

## S1：以 `seg_indptr` 为准并跳过无效 QKV tiles

状态：候选就绪，尚未提交平台

验证时间：2026-08-24 06:16–06:22 CST

### 根因与最小修复

S0 在确认 segment 是否为空前先读取 `weight_indices[b]` 和
`lora_ranks[weight_index]`。题面 reference 先比较相邻两个 `seg_indptr` 并跳过
空 segment；因此空段不应访问的 adapter metadata 在 S0 仍可能越界。S0 还用
`seg_lens[b]` 判断 token block，而 reference 的 segment 边界只由
`seg_indptr[b:b+2]` 定义。

此外，wrapper 按最大 slice 宽度为每个 Q/K/V slice 启动相同数量的 output
blocks；较窄 slice 的越界 blocks 虽然最终 masked store，却仍执行完整 K-loop 和
dot。S1 先从 `seg_indptr` 计算 segment 长度，依次跳过无效 token block 和无效
output block，之后才读取 adapter index/rank 和矩阵数据；同时删除 kernel 的
`seg_lens` 参数。tile、数学顺序、FP32/IEEE 路径和地址语义均未改变。

### TDD 与 release 验证

- 新回归使用 strided `seg_indptr=[0,17,17,18]`，故意令
  `seg_lens=[1,0,1]` 失配，并给空 segment 越界 adapter 哨兵。S0 在隔离进程触发
  CUDA illegal memory access、退出码 1；即使后端未 fault，S0 也只会更新首行，
  与 reference 要求的 17 行不同。日志位于
  `gpu:/tmp/flagos-qkv-lora-b-tdd.SQMRBc/old-source.log`，SHA-256 为
  `1fa722b5f6de6a9e183657f67174fd1ae6b9943549592a6cbe6e1e3cd2aa4534`。
- 新 case 同时覆盖 `S=17` 两个 token tiles、`K=33` 两个 K tiles、slice 宽度
  `65/1` 的两个/一个 output tiles、strided metadata，以及空段 metadata 不访问。
  修复后的 screening 和 release 静态/公开接口门禁均为 3/3 通过；Black 79、
  isort、flake8、py_compile 与提交源码/测试哈希门禁通过。
- release 证据目录为 `gpu:/tmp/flagos-qkv-lora-b-release.eNynkq`，mode 0700；
  静态/单测任务 PID/PGID `81318`，A/B 任务 PID/PGID `81416`。release 源码和
  测试逐字节等于 commit；`release.log` 与 `release-ab.log` SHA-256 分别为
  `8f0d43bf08e3e0ccce9b9653740908bdeaace146fec1f35d2c4378d208c72fa6`、
  `7169d27969ec877c7d56372869bfcc337a31c218316a056ecb9563fa063ca639`；
  12/12 reference correctness 通过。
- 五轮交替 AB/BA，`warmup=25, rep=100`，每次 wrapper 批量 20 次。非等宽
  affected 几何均值 `1.1163x`；FP16/BF16/FP32 分别为 `1.0887x`、
  `1.0883x`、`1.1742x`，范围 `1.0056–1.3712x`。
- 等宽 controls 几何均值 `1.0297x`，范围 `1.0132–1.0615x`。metadata guard
  的执行顺序调整也覆盖 controls；未观察到 NVIDIA 代理回退。
- base/candidate 各编译 6 个变体，最大分别为 108/110 registers/thread；均为
  10,240 bytes shared、4 warps、1 stage，spill、global scratch、local
  load/store 全为 0。

验证环境：NVIDIA GeForce RTX 5070 Ti 16 GB，driver 610.57.04，Python 3.12.13，
PyTorch 2.13.0+cu130，Triton 3.7.1，CUDA 13.0。该结果仅是 NVIDIA 代理证据。

### S1 构建身份

| 项目 | 值 |
| --- | --- |
| source commit | `11ae343f3a5864fa0b175faff3b84e932a1b4a0f` |
| verification commit | `11ae343f3a5864fa0b175faff3b84e932a1b4a0f` |
| ledger commit | 本节所在 commit |
| 源文件 SHA-256 | `3906b26f941eebb2a30704b90120a838e052bb7f83474ebd1d2d92e6b4ec1d9c` |
| 测试 SHA-256 | `40aa5a140346825443bfd1cbe3728ec31fbd928f04766c095f33ec635a1755ce` |
| ZIP | `artifacts/competition/qkv_lora_b/s1-11ae343/qkv_lora_b.zip` |
| ZIP SHA-256 | `bec21ac8d198d0eefd3d7c0ef68bf3a2c654017c00656c230ab12bc04f0f4d9c` |
| ZIP manifest | 顶层 `qkv_lora_b.py`，6190 bytes；ZIP 6314 bytes |

`unzip -t`、UTF-8/语法、唯一普通 `.py`、basename、大小和成员逐字节哈希均已
复验；打包器第二次运行状态为 `verified-existing`。

### 剩余风险与停止点

- `max_qkv_out_dim` 必须不小于最大 slice 宽度；题面 reference 不使用该参数，
  但 canonical SGLang caller 把它作为正确的 host grid hint。S1 不为错误 hint
  改用总输出宽度过量启动，也不引入 device-to-host 同步。
- `max_len` 同样必须覆盖 `seg_indptr` 的最大 segment 长度；`seg_lens` 已不再参与
  kernel 正确性。
- 3D grid、runtime slice/segment metadata、permutation 和 IEEE dot 尚未在八芯
  实测；当前不预建 vendor 分支。
- S1 为“候选就绪、未提交”。未打开浏览器、未读取或消耗平台额度；上传前必须
  重新验签 ZIP、读取平台实时 tuple，并取得用户针对该精确产物的一次性确认。

## S2：64×128 tile generic + 四 vendor（首投候选，≤2 次预算）

状态：release 门禁通过，候选就绪

与 Task 23 同族同法：generic 由 S1 的 16/64/32/4/1 升级为 **64/128/32/4/3**
（配置依据 Task 23 的 NVIDIA sweep：同族 kernel 三 shape 全 case 最优，
1.19–1.58x）。vendor：

- `_ascend`：3D `(tiles, n_slices, bs)` 折叠为 1D `min(total, 4096)`，逻辑
  id 按 batch→slice→tile 分解，三个 early return 改循环内复合守卫；
- `_iluvatar`/`_enflame`（同字节）：fp32 路径 split-fp16 三点积（编译期
  `x.dtype == tl.float32` 分支），低精度路径裸 dot；
- `_kunlunxin`：预防性回退 32/32/32/4/1——Task 23 S2 平台实测 64/128+
  stages 3 在昆仑 SDNN 路径编译爆炸（1830s 超时 + 子进程崩溃），保守配置
  为 Task 09/12 平台通过形态。

新增回归 `test_vendors_cover_fold_and_split_fp16`：bs=4、max_len 4096、三
slice（768/512/1024）、permutation，总 program 6144 > 4096 覆盖两轮折叠，
三 dtype（fp32 1e-4）× 五模块。screening 首跑经远端 Black 原地格式化回拷
（`gpu:/tmp/flagos-qkv-vend.AiDiI8`，最终 PID `111568`，4/4）；加昆仑
  vendor 后复跑 `gpu:/tmp/flagos-qkv-s2b.xNlhQV`，PID/PGID `112362`
（02:22:05，wall 900s），4/4 unittest（0.943s），`screening.log` SHA-256
`8a135f0bde50cc5561b3812c1f2ef1026367403ae4cbcbf01830f01c6ef326c6`。release
`gpu:/tmp/flagos-qkv-release.*`（四文件版）与 s2b 版（含昆仑）均
`RELEASE_OK`；s2b 的 source/verification commit
`0d8511c28968fef6a4d3014278226052d891c045`。昆仑 vendor blob
`0270e3b489151eeb94d5edd57f5d4991a6f6a00d50677775b252659b2fbfe3e8`，测试
`27e78bd1cdfde0a1d9ea7ccdbd891b8d853aaec23fbc80d07db918dc9317b093`。
canonical ZIP
`artifacts/competition/qkv_lora_b/s2-0d8511c/qkv_lora_b.zip`，SHA-256
`6b1f604afcace52211d4ac1f4ff880df17c609d24412ecdb16758d55f6b8cf18`，成员
generic + ascend/enflame/iluvatar/kunlunxin 四 vendor，`unzip -t` 通过。

### S2 平台首投：燧原 case 2 编译失败 → S2c stages=1（第 2/2 次）

S2 于 02:25:26 CST 提交（submission `4451`，当日序号 `11`，额度
`20/30`→`19/30` 前后区间）。燧原 vendor（split-fp16 + 64/128 + stages 3）
在 correctness case 2 以 `Pipeline run failed: PassManager execution failed`
编译失败，其余 case 通过——shape 相关：特定 chunk_size 下 BLOCK_SIZE×
`tl.cumsum` 与 stages=3 的软件流水线组合触发 GCU Pipeline pass 崩溃（Task
23 同形式 dot kernel 无 cumsum，stages 3 平台通过；Task 17 亦证明分支
kernel 与 stages>1 不兼容）。昆仑 vendor 被选中时平台尚在
`waiting_callback`（保守 32/32/32/stages1 配置）。S2c 只把燧原 vendor 的
`num_stages` 改回 1（tile 与 split-fp16 不变），screening
`gpu:/tmp/flagos-qkv-s2c.RvcuB1`，PID/PGID `112580`（02:27:50，wall 900s），
4/4 unittest；release `gpu:/tmp/flagos-qkv-s2c-release.*`，commit
`7857dcaa071c3715813fb4723b003326ef1ad4a8`，`RELEASE_OK`，`release.log`
SHA-256
`228ecd30c6dc01c265293719bc32cec77982e03c1c80c7d2edba63dd16c77b81`。
canonical ZIP `artifacts/competition/qkv_lora_b/s2c-7857dc/qkv_lora_b.zip`，
SHA-256
`357e8a690cca68123aabebdbb5500a86ebd66fe328105a8b91f7c1afe489cb38`。
本提交为 2 次预算的最后一次。

### S2c 平台结果与 Task 22 停止（2 次预算用尽）

S2c 于 02:33:03 CST 提交（submission `4458`，当日序号 `12`，额度
`19/30`→`18/30`，`file_url_sha256` 为
`96eac69a78b858e333c958f2bbfdc70e63c24fec626adedb97d78798242b2ab8`）。
七芯中六芯高分通过：海光 82.6125x、国际 A 80.0090x、天数 47.8050x
（split-fp16 vendor）、国际 B 49.8585x、沐曦 43.7250x、华为 24.2540x
（fold vendor）；昆仑评测中。燧原 vendor（stages 1）仍在 case 2 以同一
`Pipeline run failed: PassManager execution failed` 编译失败——stages 不是
根因，该 case 的 kernel 特化在 GCU 编译器上无法通过 Pipeline pass，且无
GCU 环境可复现定位。Task 22 两次预算用尽（S2、S2c），按规则停止；已沉淀：
同族 kernel 中 cumsum/slice 元数据与 GCU Pipeline pass 的组合是燧原编译
失败的高发区，后续含 cumsum 的 dot kernel（Task 13/14/15/16）燧原 vendor
优先使用最简结构（无 cumsum、无 slice 标量载入）。
