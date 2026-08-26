# Task 14 `context_attention` 实验记录

## Experimental S0：generic packed prefill attention

状态：P1 修复后 NVIDIA 代理正确性、编译资源和不可变 ZIP 门禁通过；**八芯高风险，未提交平台**

验证时间：2026-08-24 01:44–02:15 CST；打包门禁：2026-08-24 02:21 CST

源码 commit：`fbbf74fc8b39d69c5476ae6618cc6345fd0c763c`；未上传

### 决策

2026-08-24 02:15 CST 的公开 API 已更新为 97 次提交、19 支队伍、1 支达到
门槛；榜首 EvokeAgent 为 8/8、3.7924375x。此前 2026-08-23 21:59 CST 的
83 次提交、0 支 8 芯达标只是历史快照，不能继续当作当前状态。固定
`origin/pr31` 候选暴露过 shared-memory、超时和 grid 上限风险；本轮根据只读
交叉审查修复已证实的 `D < 16` 编译、O(batch) owner scan 和物理 grid 风险，
但 NVIDIA 代理仍不能证明本候选复制了榜首的八芯结果。

### 契约

| 项目 | 值 |
| --- | --- |
| 公开接口 | `context_attention(q, k, v, b_start_loc, b_seq_len, max_input_len, is_causal)` |
| packed 输入 | Q/K/V `[total_tokens,num_heads,head_dim]`；每条序列由 start 和 length 指定 |
| attention | 每条序列独立执行 `softmax(Q @ K.T / sqrt(D)) @ V`；causal mask 使用序列内局部位置 |
| `max_input_len` | 只供 kernel 规划；题面 reference 未使用，低报也不能改变正确性 |
| 数值 | Q/K/V 转 FP32，score、softmax 和输出累加均 FP32；FP32 dot 显式 IEEE |
| 输出 | 与 Q 同 shape，固定 FP32，out-of-place；输入不变 |
| 容差 | `atol=1e-2, rtol=1e-2` |
| 题面未公开 | input dtype、shape/head_dim 上界和 stride 范围；三 dtype 与非连续 stride 是代理验证范围 |
| 支持芯片 | 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用 A/B，共 8 款 |
| 截止 / 门槛 | 2026-08-27 19:59:59；每芯 `speedup >= 0.1` |

题面 reference 只明确同头数 MHA。S0 保留固定 SGLang/PR31 中的整除 GQA
head 映射，但本轮不把它当作平台已承诺的输入契约。

固定来源：本地 Task 14 题面；SGLang commit
`8014d9d062c3cc5d393596ecdf2f7009191965df` 的
`prefill_attention.py`；以及官方 PR #31 的 `origin/pr31`（tip `3f5aa55`，
实现 commit `73110e7`）。S0 删除设备识别、fallback、autotune、vendor、
reference/demo 和额外公开 alias。

### 唯一候选

- 单个 Triton online-softmax kernel。扁平一维 program 直接映射到
  `(query_slot, query_head, batch)`，不再让每个 program 扫描完整
  starts/lengths。program 内用 `query_block += QUERY_SLOTS` 扫描真实
  `seq_len`，所以低报 `max_input_len` 只降低并行度，不截断输出。
- `QUERY_SLOTS=max(ceil(max_input_len/16),1)`，最大 65,535；每次物理 launch
  也严格不超过 65,535 programs，超出的 batch-head 分片为多个互不重叠的
  Triton launch。logical grid 不使用受限的二维 `grid.y`。
- `BLOCK_M=BLOCK_N=16`；`BLOCK_D=max(16,next_power_of_2(head_dim))`，修复
  `D <= 8` 时 `tl.dot` 要求 K 至少 16 的编译错误，并降低 QK/PV tile 资源。
- 4 warps、1 stage。Q/K/V、metadata 与输出全部使用真实 stride；任意尾块和
  非 2 次幂 head_dim 均完整 mask。
- QK 和 probability-V 两个 `tl.dot` 的输入均显式转 FP32，并设置
  `input_precision="ieee"`；online maximum、denominator、probability 和
  accumulator 全部为 FP32，输出直接写 FP32。
- 非空核心路径只运行 Triton；wrapper 仅做 shape 契约检查和 FP32 输出分配。
  无 PyTorch 计算、设备判断、fallback、autotune 或私有后端 API。

### 验证

| 项目 | 值 |
| --- | --- |
| 源文件 SHA-256 | `23b9388fefd2adb333fbfe047b3d09945715ad3a181ea7175b85f9e8ca10002a` |
| 测试 SHA-256 | `644cc247e2fd7929578a3dfd5cf726e05ce0fa957e8582b4d15f3e30fed442fe` |
| ZIP | `artifacts/competition/context_attention/s0-fbbf74f/context_attention.zip` |
| ZIP SHA-256 | `38ce76db6fee2121a765a1cd741138b9c2ded2478fdd85b1bfb4bba3d0f97456` |
| ZIP 大小 / 成员 | 2,363 bytes；`context_attention.py` 7,861 bytes，SHA-256 与源文件相同 |
| ZIP 门禁 | `unzip -t` PASS；单个顶层 UTF-8 `.py`；ZIP 内源码与 commit 源码逐字节一致 |
| 远端证据目录 | `gpu:/tmp/flagos-context-attention.VQOW2U`、`gpu:/tmp/flagos-batch2.SQaIX2` |
| 远端环境 | RTX 5070 Ti 16 GB；PyTorch 2.13.0+cu130；Triton 3.7.1；CUDA 13.0 |
| 平台结果 | 未提交；submission ID、逐芯 speedup、均值、排名和额度均为 N/A |

- `D=8` 回归先于修复落盘，旧实现的 causal/non-causal 两个子例都以
  `tl.dot: Input shapes ... K >= 16` 编译失败。最终远端 unittest 5/5 通过；
  主测试内部覆盖 FP16/BF16/FP32 × causal/non-causal 六个子例、变长
  `1/5/33/7`、16-token block 边界、`D=37`、Q/K/V 与 metadata 非连续
  stride、输入不变，以及
  `max_input_len=1` 的低报；新增 `D=8`、长度 `3/33` 的双 causal 回归，以及
  257 条单 token 序列、`H=2,D=16,max_input_len=2048` 的分片回归：65,792
  个逻辑 programs 被拆为 `65,408 + 384` 两次 launch，均不超过 65,535；另
  覆盖空 packed batch。
- 本地 `py_compile`、`git diff --check` 通过；远端 Black 79、isort、flake8
  全部通过。本地与远端最终 source/test 哈希一致。
- 隔离 Triton cache `gpu:/tmp/flagos-context-direct.8F7Tg8` 产生 6 个 NVIDIA
  编译变体，均为 4 warps、1 stage、0 global scratch；`D=8/64/128` 的
  shared memory 分别为 3,072 / 9,216 / 17,472 bytes。PTX 中无 `tf32`，
  dot lowering 使用 `fma.rn.f32`。
- ZIP 由 commit `fbbf74f` 的算子源码直接生成，仅含顶层 UTF-8
  `context_attention.py`。文件小于 10 MB，`unzip -t`、成员名和 ZIP 内源码与
  commit 源文件逐字节一致门禁均通过。

wrapper-inclusive FP16 代理 benchmark 使用 PR #31 固定的三组公开 shape：

| lengths / heads / dim | causal | S0 ms | reference ms | 代理 speedup |
| --- | ---: | ---: | ---: | ---: |
| `[128,128,128,128] / 16 / 64` | false | 0.075585 | 0.281325 | 3.7220x |
| `[128,128,128,128] / 16 / 64` | true | 0.055571 | 0.356758 | 6.4198x |
| `[512,512] / 16 / 64` | false | 0.508648 | 0.319238 | 0.6276x |
| `[512,512] / 16 / 64` | true | 0.301512 | 0.380644 | 1.2625x |
| `[2048] / 16 / 128` | false | 7.466771 | 4.328212 | 0.5797x |
| `[2048] / 16 / 128` | true | 4.053038 | 5.231508 | 1.2908x |

对应 max absolute error 为 `9.24e-7` 到 `2.06e-6`，均远低于题面容差。

### 淘汰证据与风险

- 5070 Ti 只能证明 NVIDIA 代理路径；当前公开榜首 8/8 不能证明本地字节在
  MetaX/昆仑芯/Ascend 等后端编译或达到门槛。
- P1 owner scan 和无界二维 grid 已从结构上移除；17,472-byte NVIDIA shared
  峰值也显著低于旧候选。但各后端的 `tl.dot(input_precision="ieee")`、动态
  while 和多 launch 行为仍需平台逐芯验证。
- 题面未公开 head_dim 上界。NVIDIA 编译探针覆盖 `D=8/128/257/512`；
  `D=513/1024` 仍因 132,160-byte shared memory 超过 101,376-byte 上限失败，
  这是未解决的 P2，不能声称覆盖任意 D。
- 长序列 non-causal 代理速度为 0.5797x reference，说明 IEEE FP32 dot 和
  小 tile 的成本明显。虽高于 0.1 最低门槛，不能外推到其余七个 backend。
- 结论：保留该 experimental candidate 的不可变 ZIP；优先提交已有多芯成功
  证据的低风险任务。只有平台额度充足，且用户针对上述 Task、ZIP 路径、哈希
  和实时额度当次确认时，才用一次提交换取逐芯反馈。

## Experimental E1：NVIDIA `tf32x3` dot vendor

状态：NVIDIA 单变量优化通过提交字节发布门禁并生成不可变 ZIP；未提交平台

验证时间：2026-08-24 07:02–07:14 CST

源码 commit：`a085dc495b2ac84855784ff7694a83f050070908`

### 设计与边界

E1 不改 S0 generic，也不放大 tile。新增自包含
`runtime/backend/_nvidia/ops/context_attention.py`，仅把两个主导 FP32
`tl.dot` 的 precision 做成 compile-time 参数：

- `head_dim <= 128` 使用 `tf32x3`，让 NVIDIA 生成 TF32 MMA；online softmax、
  accumulator 和输出仍保持 FP32。
- `head_dim > 128` 原样使用 `ieee`。初筛中无门控版本在 D=257/512 虽数值正确，
  但出现约 960/1008-byte stack；门控后对应大 D 路径与 S0 一致，避免资源回退。
- grid、M/N/D tile、warps、stages、stride、mask、wrapper validation 和 generic
  七芯路径全部不变；没有设备判断、fallback、autotune 或额外依赖。

### 发布验证

| 项目 | 值 |
| --- | --- |
| generic SHA-256 | `23b9388fefd2adb333fbfe047b3d09945715ad3a181ea7175b85f9e8ca10002a` |
| NVIDIA vendor SHA-256 | `6ed12afdae00d614d7ed9e6cd00e9376e764df5c11467ec265cf2090f0962340` |
| 测试 SHA-256 | `63263732d5545254a5191c8539cbdf93c12423b716b61e97224a37f60551f375` |
| ZIP | `artifacts/competition/context_attention/e1-a085dc4/context_attention.zip` |
| ZIP SHA-256 | `1bd5f7483bac887f92c6be3e2aea81ac2c69f519aeafd28f267585f37a7da777` |
| ZIP 大小 / 成员 | 16,138 bytes；generic 7,861 bytes，NVIDIA vendor 8,009 bytes |
| 远端发布目录 | `gpu:/tmp/flagos-context-nvidia-release.QXbnko`，mode 0700 |
| 远端环境 | RTX 5070 Ti 16 GB；PyTorch 2.13.0+cu130；Triton 3.7.1；CUDA 13.0 |

- source commit 导出的发布字节 unittest 6/6 通过；generic 与 NVIDIA vendor
  都直接覆盖 FP16/BF16/FP32、causal/non-causal、D=8/37、非连续 stride、hint
  低报、65,535-grid 分片、空输入及输入不变。新增 D=257 回归验证 IEEE 回退。
- 发布前扩大筛选另覆盖输入尺度 `0.125/1/8`；24 个子例最大 normalized error
  `0.00855`、最大绝对误差 `1.90e-4`，远低于题面 `1e-2/1e-2` 容差。
- Black 79、isort、flake8、`py_compile` 和 `git diff --check` 通过；发布目录三份
  文件与 source commit SHA-256 一致。
- E1 PTX 已从 IEEE-only `fma.rn.f32` 转为 TF32 MMA。发布矩阵的 MMA 变体为
  91–150 registers/thread、2,560–12,288-byte shared；D=128 性能变体最高 186
  registers/thread、20,480-byte shared。全部为 0 stack、0 local、0 scratch；
  D=257 IEEE 变体也为 0 stack/local/scratch。

五轮交替、wrapper-inclusive FP16 配对结果：

| shape | S0 ms | E1 ms | paired E1/S0 | E1/reference |
| --- | ---: | ---: | ---: | ---: |
| `[128]x4,H16,D64`, non-causal | 0.0755 | 0.0418 | 1.8044x | 6.8207x |
| `[128]x4,H16,D64`, causal | 0.0555 | 0.0329 | 1.6851x | 11.1494x |
| `[512]x2,H16,D64`, non-causal | 0.5100 | 0.2481 | 2.0558x | 1.2897x |
| `[2048],H16,D128`, non-causal | 7.4892 | 4.0569 | 1.8461x | 1.0664x |
| `[2048],H16,D128`, causal | 4.0484 | 2.1731 | 1.8631x | 2.4095x |

确定性打包器从 commit 生成 generic 与 `context_attention_nvidia.py` 两个顶层
成员；重复验签得到相同 canonical ZIP SHA-256，`unzip -t`、UTF-8、10 MB 和
逐字节来源门禁均通过。

E1 只证明 RTX 5070 Ti 的 NVIDIA 路径；generic 在其余七芯的动态 while、IEEE
dot 和大 D shared-memory 风险没有因此消失。上传前必须重新读取实时额度，并取得
用户针对 Task 14、上述绝对 ZIP 路径和完整 SHA-256 的当次确认。

## E1a：天数/燧原 fp16-dot vendor（首投候选，≤2 次预算）

状态：release 门禁通过，候选就绪

E1 generic 自带 `_MAX_GRID_PROGRAMS = 65535` 分块 launch，华为/昆仑 grid
风险已由 generic 处理；其两个 ieee fp32 dot 在天数不可执行（平台证据）。
`_iluvatar`/`_enflame`（同字节）把两处 dot 操作数降为 fp16（容差 1e-2，
累加 fp32）。`_nvidia` vendor 保持 E1 原样。测试的
CONTEXT_ATTENTION_MODULES 元组扩入两 vendor，全回归覆盖。screening
`gpu:/tmp/flagos-ca-vend.OxccH9`（含一次 Black 回拷），最终 PID/PGID
`117722`（03:57 前后，wall 900s），6/6 unittest（5.084s），`screening.log`
SHA-256
`ceb071abe8b6ea0153c11f510a09935d17d978b43a04b68cfac297ba7259fd74`。
release `gpu:/tmp/flagos-ca-release.*`，source/verification commit
`6246fa87c4d6b0ecc35ca9cdb395f915d422b60e`，`RELEASE_OK`，`release.log`
SHA-256
`7ba7942c289ba47243f9b4f83cddd473e18beed27110c31f2070ea8c99fb5f4f`。
canonical ZIP
`artifacts/competition/context_attention/e1a-6246fa8/context_attention.zip`，
SHA-256
`8bfc8843bb6951de12160d83dbd56428c3697262ad331a05183217b9aa2d7861`，
成员 generic + enflame/iluvatar/nvidia，`unzip -t` 通过。

### E1a 平台结果（观测态收尾）

E1a 于 03:58:50 CST 提交（submission `4519`，当日序号 `22`，额度
`10/30`→`9/30`，`file_url_sha256` 为
`daaf9d7439f1b1cdf52d3a41ef900e71142186b76b0967ca6357e26191459e44`）。
五芯通过：天数 1.9943x（fp16-dot vendor 生效）、国际 A 6.3175x（nvidia
vendor 被选中）、海光 4.1352x、国际 B 1.6697x、沐曦 1.6620x。华为为
case 级异步错误（与 Task 15/16 的 Ascend flash 型边界 bug 同族，无单变量
解）；燧原/昆仑在评测队列滞留 40 分钟以上未出分（当晚该两芯评测器多次
超时崩溃），任务收尾时仍未终态。Task 14 按两次规则保留第 2 次额度，
以 5/8（观测态）+两芯待评记录；若后续出分为 7/8 亦无已知修复路径。

### E1a 终态（补记）

04:30:24 CST 终态 `completed` / `invalid_correctness`，5/8：天数 1.9943x
（iluvatar vendor）、沐曦 1.6620x、海光 4.1352x、国际 A 6.3175x（nvidia
vendor）、国际 B 1.6697x。三失败芯：燧原 vendor（fp16 dot）验证执行阶段
超时且"子进程仍在运行，疑似用户代码长时间运行或死循环"（与 Task 16 燧原
段错误同族：GCU 对该 flash kernel 的执行/编译不稳定）；昆仑为评测器
`Aborted` 崩溃（当晚该评测器对多任务同型崩溃）；华为为 case 级异步错误
（T15/T16 同族 Ascend flash 边界 bug）。Task 14 最终 5/8，保留第 2 次额度
（无已知单变量修复，三芯失败互独立）。

## E2-b951safe：官方 PR #31 四路径证据迁移（唯一一次重开）

状态：source/test、Git-object release 与不可变 ZIP 门禁通过，待实时 preflight；
只允许正式提交一次。

### GitHub 一手证据与决策

2026-08-26 对 `flagos-ai` 相关仓库的定向检索发现官方
[FlagGems-sglang PR #31](https://github.com/flagos-ai/FlagGems-sglang/pull/31)。锁定
作者有专项结果的 commit
[`b951a115`](https://github.com/flagos-ai/FlagGems-sglang/commit/b951a115a8cbd7ed81a2ea762c8825f8d26b7836)，
不采用 PR 当前可变 head。该证据组合为：

- 同源平台 submission `3942` 的六颗通过分数：天数 `3.94533333x`、沐曦
  `4.41016667x`、燧原 `0.574x`、海光 `7.10483333x`、国际 A
  `13.49016667x`、国际 B `1.27716667x`，合计 `30.80166667x`；旧失败为
  昆仑编译超时和华为 case 10 的 `blk=32774` grid 错误。
- `b951a115` 作者披露的专项 KernelGen 为 Kunlun 12/12、Ascend 16/16，另含
  Ascend 长序列 stress 1/1。它不是平台八芯官方结果，故只作迁移先验。
- 即使昆仑和华为都只有最低 `0.1x`，六芯冻结分数给出的有效均值下界仍为
  `3.87520833x`，略高于当时唯一有效榜首 `3.7924375x`。

原始 `b951a115` 不能直接提交：MetaX vendor 跨包导入 generic，不满足竞赛
self-contained 门禁；generic/MetaX/Ascend/Kunlun 又都在 `D=8` 令 `tl.dot`
得到小于 16 的 K 维。E2 只作最小安全改版：generic/Ascend/Kunlun 各增加
`BLOCK_D=max(16,next_power_of_2(D))`；MetaX 内联同一已修 generic，仅固定其
`BLOCK_N=16,num_warps=4` policy。对历史 `D>=16` shape，传入 kernel 的
constexpr 与证据字节完全相同。删除旧 Enflame/Iluvatar/NVIDIA vendor，让这三芯
与国际卡回到有六芯平台证据的 generic 路径。由于有上述派生修复，本候选明确不标为
exact-source evidence。

### 验证与发布

source/verification commit 为
`fc81bb8fbd6204d789d0b50efc09a43d4aa6b703`。generic/MetaX/Ascend/Kunlun/
test SHA-256 分别为
`e3698164cd1ca36b22822e48eb561c4165c73ef480e71725c97f9571dc475b4b`、
`58843e30a5c30f540591405953a58ae90cf34390048fb75a577fc5b4df17b69c`、
`2b98d4b985e637db792ca739f72e7a75e9e2760b1acaa51c9a7d3c09871fa6b8`、
`1cb2bac13e7e708bb3ee9ee8a31ac5754fab4df3ce2e0f353ba0569f80686eb4`、
`c1e628365a6b8d3c3a7c17fe7ea7690d69c33f1a4dc864f2ea89a5a9f55b04aa`。

- screening 位于 `gpu:/tmp/flagos-context-attention-ghscan.er90CC`，base commit
  `5e17af69ecbf573967aa13ee149bc94da7e64844`，PID/PGID `141956`；传输前后
  五文件哈希一致，静态门禁和 unittest 8/8 通过，`SCREENING_OK`；
  `screening.log` SHA-256 为
  `81fb1ce534c79466fe411b5171aa6928671b8e68cc44cafc6af318ad282fb76b`。
- Git-object release 位于
  `gpu:/tmp/flagos-context-attention-b951-release.mFgfh0`，PID/PGID `142872`；
  Black/isort/flake8/py_compile、哈希复验和 unittest 8/8（0.629s）通过，尾行为
  `RELEASE_OK`；`release.log` SHA-256 为
  `c3a9b25ac7d608a60ff3020dc1267a88377f6ecf17246720e6e8861d7c4d51ff`。
- 同目录 release benchmark PID/PGID `143278`；格式化后的独立 harness SHA-256
  `1d6cc4bff9e4bb29f16bcc3da68953d35c6fb0780dc18ff6b8df6cb8ac770761`，
  `benchmark.log` SHA-256
  `124329e542597eb260df965fcc9a325de54ac88267215944e1bf4c1be8e90783`，
  尾行为 `BENCHMARK_OK`。四个公开主 shape 的 wrapper-inclusive 代理 speedup 为
  `13.5016/20.2057/4.5583/2.9578x`，最大绝对误差 `4.12e-4`；四模块代表 shape
  延迟为 `0.02227/0.02699/0.03494/0.02752ms`。编译变体均为 0 global
  scratch，最大 shared 40,960B。
- 远端环境为 RTX 5070 Ti 16GB、Python 3.12.13、PyTorch 2.13.0+cu130、
  Triton 3.7.1、CUDA 13.0。NVIDIA 只作代理；Ascend/Kunlun 的主要晋级依据仍是
  上述固定 GitHub commit 的作者设备结果。

canonical ZIP 为
`artifacts/competition/context_attention/e2-b951safe-fc81bb8/context_attention.zip`，
32,808B，SHA-256
`63e3e0ddccf1493dfb484ee4a7f1310f4f91dae677b874afa68fe43798cac774`；
唯一成员为 generic + ascend/kunlunxin/metax。确定性重建、`--verify-existing`、
`unzip -t`、成员白名单和逐文件来源哈希全部通过。

停止规则：E2 只提交一次；昆仑或华为任一失败/低于 `0.1x`，Task 14 永久停止，
不再追加 tile/warps/dtype 猜测。
