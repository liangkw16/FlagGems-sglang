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

## S1a：预防性 Ascend/Kunlun token 折叠 + Enflame 配置 vendor（首投候选）

状态：release 门禁通过，候选就绪，等待 preflight 与提交

NVIDIA 调参 sweep（`gpu:/tmp/flagos-ela-sweep.575RVG`，PID `108768`，
00:56:20，脚本 SHA-256
`96be021c60f76e14d9a646e5ffecd7b8aa7bbca4b9d04bbd8ab28a1d0b6a13ca`，修正版
含路径/生成器两处修复）：4 个代表 shape × BLOCK_RANK{64,128,256,512} ×
warps{2,4,8} × stages{1,2} 共 66 个有效配置点，S1 的 128/4/1 在 3/4 case
处于噪声内最优（≤1.4% 差距），仅 bs2-len4096-r128-bf16 上 64/4/2 快
13.7%——generic 保持不变；64/4/2 作为燧原 vendor 配置依据。

grid 审计：`(max_len, bs)` 2D 展平总数可超 65535（华为 launch 越界、昆仑
编译失败，均已有平台证据）。vendor 设计：

- `_ascend` 与 `_kunlunxin`（逐字节相同文件）：`token_cap =
  max(1, 65535 // bs)`，物理 grid `(min(max_len, token_cap), bs)` 使展平总数
  ≤65535；kernel 把两个 early-return 改为 token 折叠循环
  `for token in range(pid, max_len, num_programs)` 内的 `token < seg_len and
  rank != 0` 守卫，segment/rank 元数据载入外提；数学与 BLOCK_RANK 128 不变。
- `_enflame`：仅配置换为 BLOCK_RANK 64 / warps 4 / stages 2（sweep + 燧原
  stages≥2 平台教训），grid 不变。

新增回归 `test_vendors_cover_token_fold_and_configs`：bs=8、随机 segment 长
9000–16384（max_len > token_cap=8191，覆盖两轮 token 折叠），四模块对
reference 逐字节相等。共 6/6 unittest。screening
`gpu:/tmp/flagos-ela-vend.UOxCaI`（首跑 Black 折行失败后以远端 Black 原地
格式化回拷修正），PID/PGID `109180`（01:06:07，wall 900s，脚本 SHA-256
`0b95a4c5b3d715339bd322b11fcbfaa9a7e902e84f5c167d71939803574bed17`），
`screening.log` SHA-256
`b905a375f476b59dd4812ee34f345ae2faf192c9dfd44ff7b45a62669e9d38b0`；ascend/
kunlunxin vendor blob
`1c4524e6f2c742d2d4950af8ecb53c6598a9a1f8eba0cb67ba79b309f6dee958`，enflame
vendor blob
`8151f7cbed9ccaa2d5e35dd4a85edd573fea5ccddda4380413a38c64d28bf592`，测试
`9d624b465b268b0211183fafa9fce8304a84799ae9aac576f59c4e0c82244338`。release
`gpu:/tmp/flagos-ela-release.cTDJau`，source/verification commit
`4db86a68f8e5038ece0d89b8387643b2c787c986`，PID/PGID `109366`（01:08:59），
`RELEASE_OK`，`release.log` SHA-256
`fa77a2f2cb18997a21f893eb9b887e87b72ebb8119e48eaff327196177bb1d5d`。
canonical ZIP
`artifacts/competition/embedding_lora_a/s1a-4db86a6/embedding_lora_a.zip`，
SHA-256
`3e5e15064613c9223491ed2745502ec5f535df1bc07c27698bb259dcb72f82c4`，成员
generic + ascend/enflame/kunlunxin 三 vendor，`unzip -t` 通过。本任务提交
预算 ≤3 次（本候选为第 1 次）。

### S1a 平台首投：7/8，燧原 vendor 编译失败

2026-08-25 01:10:36 CST 提交，submission ID `4372`、当日序号 `6`，额度
`25/30` 变 `24/30`，`file_url_sha256` 为
`8a7c10911a609ab7bb8b93c9552fdeb87a3a4e6dc3aa6c53e57e4c567d8a97f7`。01:11:23
终态 `completed` / `invalid_correctness`，7/8：天数 24.8645x、沐曦 7.3265x、
海光 27.1970x、国际 A 31.3045x、国际 B 17.1815x；华为 vendor 选中并通过
（1.9120x），昆仑 vendor 选中并通过（0.3320x）——token 折叠设计在两芯同时
成立。燧原 vendor（BLOCK_RANK 64/stages 2）编译失败：
`RuntimeError: Pipeline run failed: PassManager execution failed`（case 0 起
全部 case）。结论：`num_stages=2` 的软件流水线与该 kernel 的数据依赖分支
（extra/regular 载入二选一）在 GCU 编译器上不兼容；dot kernel 的 stages≥2
经验不可迁移到分支 gather kernel。第 2 次尝试（2/3）：燧原 vendor 改回与
generic 逐字节相同（128/4/1），其余文件不变。

### S1b/S1c 平台结果与 Task 17 停止（3 次预算用尽）

S1b（燧原 vendor 与 generic 逐字节相同，commit
`e7ea0645c357c5dc8eaf63e25b09b85f2b5eb9ff`，submission `4377`，01:13:16，
额度 `24/30`→`23/30`）：燧原对同字节 generic kernel 仍以同一
`Pipeline run failed` 编译失败，证明与 BLOCK_RANK/stages 配置无关，是
generic 的 kernel 结构本身（early return 与依赖加载数值的运行时分支）不被
GCU Pipeline pass 接受。其余七芯通过（天数 25.9825x、海光 24.8760x、国际 A
32.3185x；华为 vendor 1.9355x、昆仑 vendor 0.3260x）。

S1c（燧原 vendor 重写为全无分支：early return 改 token_valid mask、
is_extra 运行时分支改双路 masked load + `tl.where`，HAS_EXTRA_EMBEDDINGS
保持 constexpr 分支；commit `e12c7a9b02706a860ae9db8041e438558c405576`，
submission `4384`，01:21:18，额度 `23/30`→`22/30`，`file_url_sha256` 为
`9c96f7754d01f3d0c2cf3f6214ff756bca91a67f1c9f5c7ea030438128d0e397`）：
燧原仍以同一 `Pipeline run failed: PassManager execution failed` 失败（case
0–4）。剩余嫌疑：标量 masked load（`tl.load(ptr, mask=scalar)`）、seg_indptr
多次标量载入或该 gather 访存模式本身；需要 GCU 环境才能进一步定位。其余
七芯全部通过且分数稳定（天数 26.1420x、国际 A 31.7860x、华为 vendor
1.9845x、昆仑 vendor 0.3395x）。

Task 17 三次提交预算用尽（S1a/S1b/S1c 均 7/8，燧原编译失败），按规则停止，
额度转 Task 23。已沉淀可迁移知识：(1) 华为/昆仑 token 折叠 vendor
（`token_cap = 65535 // bs` + token 循环）两芯同时平台验证成功，直接适用于
Task 22/23 的同构 batch_info kernel；(2) GCU 编译器拒绝该 family 的三种
kernel 形态（含分支、无分支、generic 同字节），后续遇燧原 gather 类 kernel
失败时优先怀疑标量 masked load 与元数据标量载入，考虑把标量元数据改为
向量化载入或地址钳制后无 mask 载入。

## E2a：燧原 metadata route + 规整 rank gather（高倍数冲刺重开）

状态：原 `c7c184b` 产物在提交前经上游审计淘汰；后文 `E2a-i32` 已重新通过
release 门禁与不可变 ZIP 验签，候选就绪，待实时 preflight。

用户批准的高倍数冲刺重开 Task 17，但只允许一个结构性候选；generic、Ascend、
Kunlun 字节全部冻结。E2a 把原单 kernel 拆成两个职责单一的 kernel：

- route kernel 只读取 segment metadata，把每个 token 的 adapter/rank 展开为连续
  int32 数组。空 segment 的非法 adapter 在间接读取 rank 前钳为 0；route 数组
  零初始化，因此 segment 未覆盖 token 也保持 rank 0。
- gather kernel 不再读取 segment metadata，没有 runtime rank loop、early return
  或 masked scalar load；逻辑 grid 为 token × rank-block，物理 1D 展平并分批到
  65535。每个 program 对一个 token 做 128-rank 向量 gather；extra constexpr
  变体用两路安全地址 masked load + `tl.where`。

最终 source/verification commit 为
`c7c184b322a8cf683bdc2d3ba1c8082401431ea4`；Enflame/test SHA-256 分别为
`746bc29d5d0fe5daee91a3428ff85c03462f887e86560a4c4bdf1071ca5805e6`、
`5fc2a5ee86a1e877b9db0e439eec0a92489a4a87675a4e2dd3582b19bf7670da`。
组合回归覆盖 leading-empty 的非法 adapter、rank 129/0/65、base/extra token、
非连续 int64 input、int32/int64 非连续 metadata、非连续 weights/extra、输入不变
与精确 reference 相等；原 vendor 回归继续覆盖无 extra、超 65535 gather 分批和
大 segment route。

最终 screening 位于
`gpu:/tmp/flagos-embedding-lora-a-e2a-screening.qwCuS6`，base commit
`99bf5f7`，PID/PGID `132463`；`replay.sh` SHA-256
`91c60cff5c40d62026766cced6168065108eeb37de403d314088a0626f44877c`。
8/8 unittest 通过（0.689s），尾行为 `SCREENING_OK`，`screening.log`
SHA-256
`768896f2a58406b667cca3a991b3998dbfbdda45be15f895cd58dce5b8127d9f`；
输入前后 manifest SHA-256 均为
`3b1bec0c76afa4dc2115489eb5af6c54e470b081ec1119fb630552e28f4373b1`。
首次临时目录只因未携带项目 Black 79 列配置而在静态检查阶段停止，未执行候选；
修正重放脚本后才产生上述唯一 runtime screening 证据。

同目录 NVIDIA 代理基准脚本/日志 SHA-256 分别为
`feb29192c64643e01c6948ad2e0772677495120fe1182197248b1b759dda3704`、
`45637e890c97eea71bfec3dcdae7323f54a458ce82fbf1f4fde618c35b1d70c2`。
三个代表 shape 的 wrapper-inclusive 候选延迟为 0.00970/0.02405/0.01669ms，
相对旧 S1c 为 0.747/0.881/0.759x，但相对 reference 仍为
65.01/52.36/13.22x；峰值显存增量 18,432/139,264/692,736B。该数据只用于
排除代理侧阈值与资源灾难，不外推 GCU 性能。

Git-object release 位于
`gpu:/tmp/flagos-embedding-lora-a-e2a-release.BAhJEy`，PID/PGID `132975`；
`replay.sh` SHA-256
`89b3f8b7cab620ab454d872a9f4a747917f15e95f684252a6d3bd913b9eb6ced`。
8/8 unittest 通过（0.558s），尾行为 `RELEASE_OK`，`release.log` SHA-256
`9ed3064e91dac5d974deb490a8cad0d400895d60a24aba50867df1433b10637e`；
release 前后 manifest SHA-256 与最终 screening 相同。

canonical ZIP 为
`artifacts/competition/embedding_lora_a/e2a-c7c184b/embedding_lora_a.zip`，
22700B，SHA-256
`05240e744ba2f6bcd52fec0505c747efcde4f65eb4dfa51f448be85ab5edee97`，
成员为 generic + ascend/enflame/kunlunxin；`--verify-existing`、`unzip -t` 和
成员白名单均通过。

E2a 只提交一次：若 8/8 且燧原 ≥0.1x，立即停止 Task 17；若失败明确只落在
route 地址结构，最多允许一个 safe-address route E2b；若 gather 编译失败、低于
0.1x 或第二候选仍 invalid，则永久停止并转 Task 15，不调 BLOCK/warps/stages。

### E2a-i32：提交前消除 GCU300 64-bit IR

`c7c184b` 的 ZIP 尚未上传时，官方上游提供了可归因的新证据：FlagGems
[GCU300 embedding](https://github.com/flagos-ai/FlagGems/blob/d1c970e0c9ccb3c26d9fc8de906a7e21a64cc0a1/src/flag_gems/runtime/backend/_enflame/gcu300/ops/embedding.py)
会在 wrapper 将 int64 index 降为 int32；FlagGems
[#5345](https://github.com/flagos-ai/FlagGems/pull/5345) 与
libtriton_jit [#41](https://github.com/flagos-ai/libtriton_jit/issues/41)
均记录 GCU300 遇到 64-bit scalar/派生 IR 时在编译阶段失败。当前 E2a 虽已把
route 输出建为 int32，gather 内仍显式把九个 stride cast 为 `tl.int64`，因此旧
release 不能支撑本轮“消除 64-bit IR”的假设。旧 ZIP 保持不可变但标记为
superseded，未运行 preflight、未上传、未消耗额度。

最小修正只改 Enflame vendor：int64 `input_ids/seg_indptr/weight_indices/lora_ranks`
在 wrapper 有界降为 int32，并删除 gather 的九个显式 int64 stride cast；route、
gather、BLOCK/grid/warps/stages 与其余三份源码完全不变。公开与代理最大地址远低于
`2^31`；若隐藏 tensor 真需要超过该范围的元素偏移，GCU300 本身也不能用 i64
表示，作为已知风险保留，不增加会拒绝隐藏 case 的 host assert。

最终 source/verification commit 为
`fb1235d78f54699405624155d634f2a63ad209f0`；generic/Ascend/Enflame/Kunlun/test
SHA-256 分别为
`fb29244a40cffdf0d585615cb1dc9f9272063c2028792ac914e57e7a562a5f92`、
`1c4524e6f2c742d2d4950af8ecb53c6598a9a1f8eba0cb67ba79b309f6dee958`、
`962b8db7ce6fa1e3d317c309e77ef6069f97f0fa69888b760892c774098a25ac`、
`1c4524e6f2c742d2d4950af8ecb53c6598a9a1f8eba0cb67ba79b309f6dee958`、
`5fc2a5ee86a1e877b9db0e439eec0a92489a4a87675a4e2dd3582b19bf7670da`；
Enflame 源码中 `tl.int64`/`to(tl.int64)` 静态门禁为零命中。

最终 screening 位于
`gpu:/tmp/flagos-embedding-lora-a-e2a-i32-screening.Ry7zga`，base commit
`769ea73`，PID/PGID `133476`；沿用的 `replay.sh` SHA-256 为
`91c60cff5c40d62026766cced6168065108eeb37de403d314088a0626f44877c`。
8/8 unittest 通过（0.664s），尾行为 `SCREENING_OK`，`screening.log`
SHA-256 为
`d13548e1ab57b5abad984dcdfd4fb00d168d3303c73cf4bb7638d10f41ab9e34`；
输入前后 manifest SHA-256 均为
`e5ffb42167c5c14cccb4ef5cba03994c10dfe559fc670584ead6189508d5af9a`。

同目录 NVIDIA 代理基准脚本保持
`feb29192c64643e01c6948ad2e0772677495120fe1182197248b1b759dda3704`；
PID/PGID `133606`，日志 SHA-256
`0b42f43cb070b5fd9f808b7d6ffd39339af3a2300dcf00ef2f13505509f41e77`，
尾行为 `BENCHMARK_OK`。三个代表 shape 的 wrapper-inclusive 延迟为
0.01106/0.02488/0.01767ms，相对旧 S1c 为 0.647/0.851/0.712x，相对 reference
仍为 56.93/50.58/12.49x；峰值显存增量 19,456/143,360/703,488B。该数据只用于
排除代理侧阈值与资源灾难，不外推 GCU 性能。

Git-object release 位于
`gpu:/tmp/flagos-embedding-lora-a-e2a-i32-release.n9gpTt`，PID/PGID
`133729`；`replay.sh` SHA-256 为
`89b3f8b7cab620ab454d872a9f4a747917f15e95f684252a6d3bd913b9eb6ced`。
8/8 unittest 通过（0.554s），尾行为 `RELEASE_OK`，`release.log` SHA-256
为 `552417dfdb600d3e522a7647da5c0867b85575c3e23e1f489edc55aa0edda72d`；
release 前后 manifest SHA-256 与最终 screening 相同。

canonical ZIP 为
`artifacts/competition/embedding_lora_a/e2a-i32-fb1235d/embedding_lora_a.zip`，
22,719B，SHA-256
`eb4b40d4703f5c6ea8d9bc3e5c3b896310f5bfe7a9c0d40637dc0c746d126081`，
成员仍为 generic + ascend/enflame/kunlunxin；`--verify-existing`、`unzip -t`
与成员白名单全部通过。平台仍只允许这一个 E2a 正式提交，后续 stop gate 不变。

### E2a-i32 平台结果：8/8，13.8620625x

2026-08-26 15:03:33 CST 按一次性 preflight intent 提交，submission `5048`、
当日序号 `4`，额度 `27/30` → `26/30`。上传后无认证下载复验为 22,719B，
SHA-256 与本地 canonical ZIP 完全一致；本次 `file_url_sha256` 为
`338d841bc9fa5b4ee5c9a055b0c383f8a18b9ea21457964b480664fb13837d63`。

| 芯片 | selected file | 正确性 | 加速比 |
| --- | --- | --- | ---: |
| 天数 | generic | 通过 | 25.8475x |
| 沐曦 | generic | 通过 | 7.4465x |
| 燧原 | `_enflame` | 通过 | **0.3885x** |
| 海光 | generic | 通过 | 26.8050x |
| 昆仑 | `_kunlunxin` | 通过 | 0.3265x |
| 华为 | `_ascend` | 通过 | 1.8665x |
| 国际 A | generic | 通过 | 31.0115x |
| 国际 B | generic | 通过 | 17.2045x |

终态为 `valid`、8/8、平均 `13.8620625x`、`is_team_best=true`。燧原从此前
三轮全 case `PassManager execution failed` 恢复为正确且超过 0.1x；本轮平台
历史 traceback 也直接包含 `i64` loop/value IR，与公开 GCU300 verifier 证据吻合。
因此“消除 Enflame 64-bit IR”假设被平台证实，E2a stop gate 触发：Task 17
闭环完成，不再调 BLOCK/warps/stages 或追加提交。

## E3 offline：NVIDIA route + gather vendor（拒绝，不提交）

2026-08-26 在 E2a-i32 已 8/8 的前提下，只离线评估把 E2a 的 metadata route +
固定 rank gather 结构复制为 `_nvidia` vendor；generic 和四个既有 vendor 均冻结。
候选源码 SHA-256 为
`746bc29d5d0fe5daee91a3428ff85c03462f887e86560a4c4bdf1071ca5805e6`。

screening 位于
`gpu:/tmp/flagos-embedding-lora-a-nvidia-screening.68Ovmm`，base commit
`80c517b3da87fc7ea676f8946d2567e49b6339b6`，PID/PGID `156325`；8/8
unittest、py_compile、Black/isort/flake8 和前后哈希通过，尾行为
`SCREENING_OK`。`replay.sh` / `screening.log` SHA-256 分别为
`d39b9a0b7a54848e68444c89b759b8be7ecbe7b7f15373470f942850872f683f`、
`ea27732223e2e889f39cb4a8f8fd3538fd9b1139228ddb15063aaaa70dc7b520`。

同目录对当前 generic 做六轮交替 direct A/B；三点
`(tokens,batch,rank,vocab,extra)` 为 `(256,16,32,8192,False)`、
`(1024,32,64,32000,False)`、`(2600,3,129,1024,True)`，候选速度仅为
generic 的 `0.8292x/0.9300x/0.7896x`，几何均值 `0.8476x`，未达到
`>=1.35x` 晋级线。benchmark script/log SHA-256 分别为
`0eee4cf67504324d2e552325ad51dc451b310a6f59d91e07c22598239404959f`、
`6ec63639125be1dc4a39d49f4b120dc019f888c7ec665a74f6a4d59dd6909750`，尾行为
`BENCHMARK_OK`。

两阶段 route 的额外 launch/临时张量在 NVIDIA 上不能抵消原 kernel 的 metadata
分支成本，因此 E3 不晋升、不生成 ZIP、不运行 preflight、不消耗额度；候选源码和
测试已逐字节撤回，Task 17 继续保留 E2a-i32 的 13.8620625x team best。

## E4 Hygon：128-rank tile 使用 2 个 wave64

状态：commit-bound release 与 canonical ZIP 门禁通过，待唯一一次平台提交

E2a-i32 的停止结论针对 Enflame route/gather；E4 由固定 FlagTree HCU backend
对 legacy gfx928/gfx936 使用 64-lane wave，以及固定 FlagGems Hygon
`index_select_backward` 对 BLOCK 128 使用 `num_warps=2` 的新证据触发。E4 新增
自包含 `_hygon` vendor；它与已在海光平台通过的 generic 逐字节一致，唯一把
launch 的 `num_warps=4` 改为 `2`。BLOCK_RANK 128、stage 1、grid、地址、数学与
控制流全部冻结；generic、Ascend、Enflame、Kunlun 逐字节保持 E2a-i32。

| 项目 | 值 |
| --- | --- |
| source / verification commit | `d33b89dba8b0adc431731dae489673a50dfcdef3` |
| generic / Ascend SHA-256 | `fb29244a40cffdf0d585615cb1dc9f9272063c2028792ac914e57e7a562a5f92` / `1c4524e6f2c742d2d4950af8ecb53c6598a9a1f8eba0cb67ba79b309f6dee958` |
| Enflame / Kunlun SHA-256 | `962b8db7ce6fa1e3d317c309e77ef6069f97f0fa69888b760892c774098a25ac` / `1c4524e6f2c742d2d4950af8ecb53c6598a9a1f8eba0cb67ba79b309f6dee958` |
| Hygon / test SHA-256 | `cba1acc19bfda66ec5f7f8af977f335255728de9da51f3bb8a8d19a811b03c35` / `6f755e5a1f8cc2945566cd1aedd9e803f2241ff2894a846342a96bf73209a907` |
| source Git archive SHA-256 | `90467df961d4291292677267823efe5234733afa894136aa883bb2cba575963b` |
| screening | `gpu:/tmp/flagos-task17-hygon-warps-screen.JpZEkU`；8/8、静态检查、A/B 与 `SCREENING_OK` |
| release | `gpu:/tmp/flagos-task17-hygon-warps-release.QAIXhW`；8/8、0.559s、单变量字节断言、静态检查、A/B 与 `RELEASE_OK`；日志 mode 0600、SHA-256 `081663134b5c699ba5697d8c1db5d7b1330f87b86f84c60c7a64ef6ea4be6f8c` |
| canonical ZIP | `artifacts/competition/embedding_lora_a/e4-hygon-d33b89d/embedding_lora_a.zip`，27,567B，SHA-256 `74cff90ca2a055624647ee36b004372fe044282e9b1798b52de903bd33f31cfe` |

RTX 5070 Ti 上五个 shape × 三 dtype 的交替 A/B（5 组、warmup 25、rep 100）
先逐点与 E2a generic 精确比对；release 几何均值 `0.993774x`、最小点
`0.916905x`，通过预声明的 0.95/0.90 灾难门。candidate 为 30 regs、2 warps、
1 stage，零 spill/shared/scratch；baseline 为 22–24 regs、4 warps、1 stage，
同样零 spill/shared/scratch。该 wave32 代理不能验证 Hygon wave64 收益，只排除
正确性和资源灾难。

2026-08-27 04:19 CST 实时榜单中 E2a-i32 为第 `6`、`13.8620625x`，第 5 名
`14.19725x`。若其余七芯不变，海光需从 `26.805x` 严格升至
`>29.4865x`（+10.0037%）才能升名；公开第 5 名的海光为 `31.3005x`。一次提交
基础门为 8/8 valid、海光高于 26.805x 且平均高于 13.8620625x；升名门为海光
高于 29.4865x。提交后无论升降都永久停止 Hygon warps 轴，不重传相同字节。

2026-08-27 04:21:47 CST 经实时 preflight 执行 E4 唯一一次提交，submission
`5300`、daily seq `27`，额度由 `4/30` 变为 `3/30`；`file_url_sha256` 为
`bf71d63d5fdfbb9c8c408830bce4edca6de2c1fadc9359149a8c2d3945d03592`。对象
存储匿名回读为 27,567B，SHA-256 与 canonical ZIP 完全一致，五个成员通过
`unzip -t`；平台在海光选中预期 `_hygon` 文件。

04:22:38 CST 终态为 8/8、`valid`、平均 `14.1618125x`、team best；相对
E2a-i32 增加 `0.29975x`（+2.16%）。海光由 `26.805x` 提升至 `27.701x`
（+3.34%），过基础门但未达到 `29.4865x` 升名门：

| 芯片 | E4 speedup | 选中文件 |
| --- | ---: | --- |
| 天数 | `26.4850x` | `embedding_lora_a.py` |
| 沐曦 | `7.3035x` | `embedding_lora_a.py` |
| 燧原 | `0.3700x` | `embedding_lora_a_enflame.py` |
| 海光 | `27.7010x` | `embedding_lora_a_hygon.py` |
| 昆仑芯 | `0.3275x` | `embedding_lora_a_kunlunxin.py` |
| 华为 | `1.7670x` | `embedding_lora_a_ascend.py` |
| 国际 A | `32.0455x` | `embedding_lora_a.py` |
| 国际 B | `17.2950x` | `embedding_lora_a.py` |

未改字节的七芯存在测量波动；可归因目标芯只提升 3.34%。保留 E4 team best，
但平均仍低于第 5 名 `14.19725x`，永久停止 Hygon warps 轴，不试 1/4/8 warps。
