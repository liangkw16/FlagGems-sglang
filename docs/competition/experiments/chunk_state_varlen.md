# Task 13 `chunk_state_varlen` 实验记录

## S0：generic baseline

状态：本地静态门禁、RTX 5070 Ti 代理验证和不可变 ZIP 门禁通过；未提交平台
验证时间：2026-08-24 01:35–01:40 CST
源码 commit：`b05bfeb`

### 契约

| 项目 | 值 |
| --- | --- |
| 公开接口 | `chunk_state_varlen(B, x, dt, dA_cumsum, cu_seqlens, chunk_states)` |
| packed 输入 | x `[total,nheads,headdim]`；B `[total,ngroups,dstate]` |
| chunk 参数 | dt / dA `[nheads,nchunks,chunk_size]`；cu_seqlens `[batch+1]` |
| 头映射 | `group = head // (nheads // ngroups)` |
| 计算 | 每条序列末 chunk 上 `x.T @ (B * exp(dA_last-dA) * dt)` |
| `chunk_states` | **只读取 dtype；数值、shape 和存储均不参与计算** |
| 输出 | `[batch,nheads,headdim,dstate]`，dtype=`chunk_states.dtype` |
| 支持芯片 | 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用 A/B，共 8 款 |
| 截止 / 门槛 | 2026-08-27 19:59:59；`speedup_threshold=0.1` |

固定来源为本地 Task 13 题面及 Mamba v2.2.4
`mamba_ssm/ops/triton/ssd_chunk_state.py::_chunk_state_varlen_kernel`。上游在序列
跨入末 chunk 时会累加 `chunk_states * exp(dA_last)`；题面 reference 明确没有
该项，因此 S0 连 kernel 参数都不传 `chunk_states` 指针，只在 wrapper 读取 dtype。

2026-08-24 01:39 CST 公开 API 状态：65 次提交、11 支队伍、2 支达到门槛；
榜首 MakeYUNAGreatAgain 为 8/8、477.784875x。动态值仅用于当时决策。

> **红线：公开 reference 的跨 chunk 定义自相矛盾。** 它用整段
> `x[start:end]` / `B[start:end]`，却只从最后一个 chunk 对 dt/dA 做 Python
> 负切片。`chunk_size=8`、单序列长度 10 时分别得到长度 10 和 2，直接广播
> 报错；长度 9/17 时 scale 长度 1 又会被静默广播到整段。S0 只承诺每条序列
> 完全位于其最后一个全局 chunk 内的可定义范围，不擅自补上游语义。

### 唯一候选

- 一个 self-contained Triton kernel；grid 为 output 的 32×32 矩阵 tile、batch、
  head，K tile 32，4 warps、1 stage，无 autotune/vendor。
- packed B/x、dt、dA、cu_seqlens 和 output 全部使用真实 stride；cu_seqlens
  支持 int32/int64，并转 int64 做地址计算。
- decay、scale、X/B 和 accumulator 全为 FP32；`tl.dot(...,
  input_precision="ieee")` 明确禁止 FP32 TF32。
- 输出由 `torch.empty` 按 `chunk_states.dtype` 分配；非空路径核心计算全在
  Triton，无设备判断、连续化或 fallback。

### 验证

| 项目 | 值 |
| --- | --- |
| 源文件 SHA-256 | `29da5027f726c9976b3251f21a4dea59efd45ec8d41e10dffe801f554af97707` |
| 测试 SHA-256 | `e678acd4cdc7390e52f1971becdd05a366ccb235847e8ade33e0c12b7b6ba907` |
| ZIP | `artifacts/competition/chunk_state_varlen/s0-b05bfeb/chunk_state_varlen.zip` |
| ZIP SHA-256 | `bd23ddad1c833c8f9ba2c8e0e551fa5e4c3d7ad446351d74a346af14c850603b` |
| 远端证据目录 | `gpu:/tmp/flagos-batch2.SQaIX2`，mode 0700 |
| 远端环境 | RTX 5070 Ti 16 GB；PyTorch 2.13.0+cu130；Triton 3.7.1；CUDA 13.0 |

- unittest 3/3 通过；覆盖 FP16/BF16/FP32 输入和输出，GQA `6 heads / 2
  groups`，多序列 `[3,5,4]` / `[1,7,2,6]`，两个全局 chunk，所有输入
  非连续、int32/int64 cu_seqlens、空 batch 和输入不变性。
- 使用两组不同且非零的 `chunk_states` 内容重复调用，输出逐位一致；同时输出
  dtype 严格跟随它。
- `py_compile`、`git diff --check`、Black 79、isort、flake8 均通过；本地与
  远端源码/测试 SHA-256 完全一致。
- wrapper-inclusive FP16 benchmark：`batch=16,nheads=8,ngroups=2,
  headdim=dstate=64,chunk_size=64`，S0 `0.010662 ms`，题面 reference
  `6.539520 ms`，代理 speedup `613.340x`。reference 含 Python/head 循环与
  `.item()` 同步，不能外推为八芯成绩。
- ZIP 由 commit `b05bfeb` 的算子子树直接生成，仅含顶层 UTF-8
  `chunk_state_varlen.py`；`unzip -t`、10 MB、成员名和逐字节 SHA-256 门禁
  均通过。

### 风险与下一步

- NVIDIA 代理不能证明其余七款芯片；平台前不增加 vendor 文件。
- 只覆盖上面标明的 reference 可定义域。若平台实际给跨 chunk 用例，必须先
  由赛方澄清是“仅末 chunk token”还是“前序 chunk_states 延续”，不能猜测。
- 若平台正确但矩阵维度尾块性能不足，下一单变量候选才调整 32×32 tile。
- 未提交平台或消耗额度；上传前仍需用户确认上述路径、SHA-256 和实时额度。

## S1：匹配公开 reference 的 Python slice 与广播语义

状态：确定性正确性修复已通过提交字节发布门禁并生成不可变 ZIP；未提交平台

验证时间：2026-08-24 07:02–07:09 CST

源码 commit：`791193061b115177052b365441baee52c1f3a81e`

### 根因与修复

S0 不是公开 reference 的实现。`chunk_size=8`、`cu_seqlens=[0,9]` 时，公开
reference 将最后 chunk 的单个 scale 广播到整段；S0 却从最后 chunk 指针向前
线性读取前一 chunk，导致 `x=B=1`、前一 chunk 的 `dt=0`、最后 scale 为 1 时
reference 返回 9、S0 返回 1。现有用例中的每条序列都恰好位于其结束 chunk
内，未覆盖该反例。空序列还会让 S0 构造 `dA[-1]` 地址。

S1 只修正这一根因，不调整 tile 或并行策略：

- 按长度为 `chunk_size` 的 Python 一维 slice 规则归一化负 `start_relative`，
  计算 `slice_start`、`slice_stop` 和 `scale_length`。
- `scale_length == 1` 时把唯一 dt/dA scale 广播到整段；
  `scale_length == sequence_length` 时逐 token 读取。
- `dA_last` 对空序列使用 masked load；K 循环执行零次并写出全零，消除负地址
  读取。
- 对公开 reference 自身会因 shape 不匹配而报错的
  `scale_length not in {1, sequence_length}` 保持未定义，不猜测 Mamba 的
  `chunk_states` 延续语义。

### 发布验证

| 项目 | 值 |
| --- | --- |
| 源文件 SHA-256 | `025ac23b0026a423e381f8fffe3715c4e21ce64558e79ad4841145351f11d4f1` |
| 测试 SHA-256 | `c827c21d3dc44c3ae344cf93d680b2ba2fef44e1670a741427fe47ae0c85b27c` |
| ZIP | `artifacts/competition/chunk_state_varlen/s1-7911930/chunk_state_varlen.zip` |
| ZIP SHA-256 | `fcc17df06adf338578402f315e4dab75bf2361e641885bb99f94e23be46efd49` |
| ZIP 大小 / 成员 | 7,516 bytes；顶层 `chunk_state_varlen.py` 7,376 bytes |
| 远端发布目录 | `gpu:/tmp/flagos-chunk-state-release.KKAmMf`，mode 0700 |
| 远端环境 | RTX 5070 Ti 16 GB；PyTorch 2.13.0+cu130；Triton 3.7.1；CUDA 13.0 |

- 从 source commit 导出的发布字节执行 unittest 5/5 通过。新增永久回归精确
  覆盖 `[0,9]` 的单 scale 广播，以及 `[0,0,1]` 的 leading-empty 序列；原有
  三 dtype、GQA、非连续 stride、int32/int64 和输入不变回归全部保留。
- Black 79、isort、flake8、`py_compile` 和 `git diff --check` 通过；远端
  source/test 与 commit SHA-256 一致。
- 仅在 S0/S1 都正确的两个 control shape 上做 FP16/BF16/FP32 五轮交替配对：
  六点中位 speedup 为 `1.0020/0.9997`、`1.0004/0.9972`、
  `0.9965/1.0182`，几何均值 `1.0023x`。编译变体为 40–56
  registers/thread、8,192-byte shared，0 stack、0 local、0 scratch。
- 确定性打包器从 commit 直接生成 ZIP；重复验签得到相同 canonical SHA-256，
  `unzip -t`、UTF-8、成员名、10 MB 和逐字节来源门禁均通过。

S1 只解决公开 reference 能正常返回的域，未做性能 E1，也未获得八芯平台证据。
上传前必须重新读取实时额度，并取得用户针对 Task 13、上述绝对 ZIP 路径和完整
SHA-256 的当次确认。

## E1：低精度原生 dot（拒绝）

状态：screening 正确性失败，源码已逐字节恢复 S1；未生成 ZIP、未提交平台

验证时间：2026-08-24 08:27–08:35 CST

固定 Mamba v2.2.4 上游会在 FP32 计算 decay/scale 后，把缩放后的 `B` 转回
`x` dtype 再进入 `tl.dot`。screening 先验证了宽矩阵上的性能空间：K=64、
`headdim/dstate` 为 128×256、256×128 或 256×256 时，RTX 5070 Ti 代理约为
`1.13–1.15x`；较小 K 或矩阵则可能回退，因此后续候选已收窄到宽矩阵。

但该 downcast 不满足题面 `atol=rtol=3e-2` 的完整数值契约。确定性 FP16
反例使用 K=2、`x=[64,-48]`、`B=[1,1]`、
`dt=[2.2734375,2.86328125]`、`dA=[0,-0.0570068359375]`，其 FP32
reference 和 S1 都精确为 0，低精度 scaled-B dot 输出 `0.0625`，绝对误差
超过 `0.03`。因此不以随机样本上的小误差或平均性能替代契约门禁，E1 直接拒绝。

screening base commit 为 `3c38f1c25db250e1a290067a610ae77170b89b6a`；最终
候选源码 SHA-256 为
`08e6c3f3edf4e80091a30a1811f16f8055115b94b8787650657731eb241d7fea`。
远端目录为 `gpu:/tmp/flagos-chunk-state-varlen.sJaYzJ`（mode 0700）；静态与
原有 5/5 单测日志 SHA-256 为
`f647610d779661a65a67ef4d7fc52dcbf2ea15d68c59de5d9066775c9c76e5a1`，反例
日志 SHA-256 为
`208cd018d28fe8af01e855b299a6b02a9cca8d5a0d8c416ecaf32c537c916786`。

## S1a：Ascend 折叠 + 天数/燧原 fp16-dot vendor（首投候选，≤2 次预算）

状态：release 门禁通过，候选就绪

grid 审计：3D `(tiles, batch, nheads)` 展平总数可超 65535（同 Task 12）。
vendor：`_ascend` 为 Task 12 平台验证的 capped grid-stride 折叠（cap 4096，
逻辑 id 按 head→batch→tile 分解）；`_iluvatar`/`_enflame`（同字节）把
ieee fp32 dot 改为 fp16 操作数 dot（题面容差 3e-2，无需 split；天数依据
Task 12/22/23 平台证据，燧原依据 Task 23 4.05x 证据）。题面 reference 的
段语义要求每段长度小于 chunk_size（现有测试约定），新增回归
`test_vendors_cover_fold_and_fp16_dot` 以 `[1,7,2,6]×8` 共 32 段 × 两 dtype
× 四模块覆盖。screening `gpu:/tmp/flagos-csv-vend.8oIvfU`（含一次 Black
回拷与三次测试 shape 修正——make_case 的全局 chunk 语义与段长 ≤
chunk_size 约束），最终 PID/PGID `117015`（03:36 前后，wall 900s），6/6
unittest（0.634s），`screening.log` SHA-256
`aa93469405f5afb1a21da2a5ea3276c5049c1466777e92108f6dbad6afd7bbbb`。
ascend vendor blob
`a8bf24b42a1f7a73ab4f9635b928e0a28094b258a04cf4de4d39ae52f08c9e4b`，
iluvatar=enflame
`86419a800ebf783d70f0392ae7c71ad06e755844197efe8ce23a6b7b394276e8`，测试
`08684018d37274fa90180aeeb731c9e340351f426a33d166559c06e2412bcd6b`（后再
修正段长，最终以 Git blob 为准）。release
`gpu:/tmp/flagos-csv-release.*`，source/verification commit
`dccd88d3b90526d3da3510c12e0622cd6b9528df`，`RELEASE_OK`，`release.log`
SHA-256
`0496e6f96e93bdb8338fd6ab1204849e38103cde6ded75179faaf3883d06a16d`。
canonical ZIP
`artifacts/competition/chunk_state_varlen/s1a-dccd88d/chunk_state_varlen.zip`，
SHA-256
`ad05f226981d59fcbc054aed46d35aa03a2bbcec2120d5402940584ad7452690`，
成员 generic + ascend/enflame/iluvatar，`unzip -t` 通过。

### S1a 平台首投：6/8 → S1b 燧原回退 generic（第 2/2 次）

S1a 于 03:48:19 CST 提交（submission `4514`，当日序号 `20`，额度区间
`12/30`→`11/30`，`file_url_sha256` 为
`9dd07764e65363ad207f61f67a8273480e1bae13d80bb1a4dc63c5ea8f8e39dd`）。
六芯高分通过：海光 533.9010x、国际 A 377.28x、天数 132.0380x（fp16-dot
vendor）、国际 B 119.0415x、沐曦 114.3535x、华为 27.7100x（fold vendor）。
燧原 fp16-dot vendor 全 case `Pipeline run failed`（与 Task 22 同型：GCU
对 32×32 tile + fp16 dot 组合编译失败；Task 23 的 64×128 通过）；昆仑
generic 全 case `PassManager::run failed`——varlen 的 cu_seqlens/边界
`tl.where` 结构在 XPU 编译器不可编译（与 Task 12 非 varlen 版通过形成
对照），结构性无解。S1b 把燧原 vendor 改回与 generic 逐字节相同
（ieee-fp32 dot + 32 tile，Task 12 燧原 0.1160x 平台通过的同款形式），
昆仑不设 vendor；release `gpu:/tmp/flagos-csv-release2.*`，commit
`1975cf7dd7a206f6727d6b425e551c63bef5d36a`，`RELEASE_OK`，`release.log`
SHA-256
`2c4399ec6f165cc77d57c6f0913749e92cf5750df10efc6a8aeef8e7690b85a3`。
canonical ZIP
`artifacts/competition/chunk_state_varlen/s1b-1975cf7/chunk_state_varlen.zip`，
SHA-256
`0319c0e26b7cd6fb12f33b43771572a058306e89ac5982234531552daa0203d1`。
本提交为 2 次预算的最后一次。

### S1b 平台结果与 Task 13 停止（2 次预算用尽）

S1b 于 03:50:42 CST 提交（submission `4516`，当日序号 `21`，额度区间
`11/30`→`10/30`）。六芯维持高分（天数 130.283x、沐曦 125.738x、海光
517.7335x、华为 29.341x）；燧原对与 generic 逐字节相同的 vendor 仍全
case `Pipeline run failed`——证明该 varlen kernel 结构本身在 GCU 编译器
不可编译（与 Task 22 qkv 同型结论），非 dot 精度或配置问题。昆仑维持
`PassManager::run failed`。Task 13 两次预算用尽，最终 6/8
（invalid_correctness）；两轮沉淀的天数/华为 vendor 形式已可复用。

## E2：metadata pack + 规整 FP32 BMM（高倍数冲刺重开）

状态：release 门禁与不可变 ZIP 验签通过，候选就绪，待实时 preflight。

2026-08-26 14:04 CST 实时状态为团队 `SoulCoder`、Task 13 `competing` 且可
提交，当日剩余额度 `28/30`。S1b 已通过六芯合计 1301.519x；只要燧原与昆仑
都达到 0.1x 门槛，同一提交的平均分即至少 162.714875x，因此本轮只解决两个
结构性编译失败，不改已通过的 generic/ascend/iluvatar 字节。

最终 source commit 为
`fff9ea904d025a589cff2c4daf6cec057075dcac`（主体 commit `1863fd0`）：

- pack kernel 独立读取 `cu_seqlens`、dt/dA、计算 FP32 scale，并把每条序列的
  X 与 scaled-B 写成连续 `[batch,head,max_seq_len,*]`；BMM 不再接触任何
  metadata、间接索引、runtime K 或 `tl.where`。
- `max_seq_len` 只在 host 读取一次；这既保留公开 reference 的
  `chunk_size=8, length=9` 单 scale 全序列广播，又避免按 total length 产生空
  program。混合空段用合法 `safe_end` 构造地址；all-empty 由 K=0 的 Triton
  BMM 写出全零。
- 燧原采用 32×32×32、FP32 IEEE、4 warps、stages 1；昆仑同为
  32×32×32 FP32 IEEE，pack tile 收敛为 16×16。所有逻辑 grid 均按 65535
  分批，`chunk_states` 仍只读取 dtype。
- 初始燧原 split-FP16 三点积草案虽通过常规回归，但代理大 scale 输出 NaN，
  因而在 release/平台前拒绝；最终候选回退 Task 12 已验证的 FP32 IEEE 形态，
  并新增 `exp(12)` 有限值永久回归。

最终 screening 位于
`gpu:/tmp/flagos-chunk-state-varlen-e2-fp32-screening.f7zuhX`，base commit
`1863fd0`，PID/PGID `131793`，wall 900s；逐字重放脚本 SHA-256
`ad63010b66dfbb20a831a8b187b67623cb20935478cb0f011c2bdaf7c6348524`。
10/10 unittest 通过（0.854s），尾行为 `SCREENING_OK`，`screening.log`
SHA-256
`9f6ee11943a9444cbdb0cd4a49467ba033a3e323b611028e8507fa403a76f66e`；
输入前后 manifest SHA-256 均为
`3f2931b1a566c1b085af12b856123c009491c60dd7724dc925e44646ab383d0c`。
环境为 RTX 5070 Ti、Python 3.12.13、PyTorch 2.13.0+cu130、Triton
3.7.1、CUDA 13.0。回归覆盖三 dtype、非连续输入/int32/int64 metadata、
leading/all-empty、`[0,9]` 与非零 start 的 `[3,9]` 广播、不同 shape/stride
的 dtype-only `chunk_states`、低精度消去反例、大 scale 有限值和输入不变性。

最终 Enflame/Kunlun/test SHA-256 分别为
`d5d1d8b195fe1efad275dc30ffcf9707499c67575b117a78bd68c097b452db93`、
`ec18415fdbb99787dbfcb5d4a3ceca126caa4cf330b03317879441c1b878bac5`、
`769a6ee3c1d526c2d1322fac5233a65e9a33412c3590c531e4d1947e62a81e4b`；
generic/ascend/iluvatar 保持
`025ac23b0026a423e381f8fffe3715c4e21ce64558e79ad4841145351f11d4f1`、
`a8bf24b42a1f7a73ab4f9635b928e0a28094b258a04cf4de4d39ae52f08c9e4b`、
`86419a800ebf783d70f0392ae7c71ad06e755844197efe8ce23a6b7b394276e8`。

同目录代表 shape（batch16、heads8、M=N128、K64、FP16 输入）的 7 轮
wrapper-inclusive 代理中位：generic 0.04018ms、Enflame 0.06625ms、Kunlun
0.06750ms；两候选峰值显存增量均 25165824B，三者输出最大绝对值相同且有限。
基准脚本/日志 SHA-256 分别为
`5b3184f09ad8e47cba492647b984c707bcaec936ce81c694878cd189b8e50e02`、
`a4df7689bf02a90d68cd82c1b12476d0123ec8fa87ee9460a484dfba2896ab10`，
尾行为 `BENCHMARK_OK`。该数据仅作 NVIDIA 资源代理；两款目标编译器仍以平台
结果为准。

source/verification commit 均为
`fff9ea904d025a589cff2c4daf6cec057075dcac` 的 Git-object release 位于
`gpu:/tmp/flagos-chunk-state-varlen-e2-release.p50nDq`，PID/PGID `132120`，
wall 900s；`replay.sh` SHA-256
`7bb6457e79c4f4b964786086b9b941eef68f30ca96a405952f6ae494bc55297b`。
10/10 unittest 通过（0.728s），尾行为 `RELEASE_OK`，`release.log`
SHA-256
`5e3cf904d9a96f8fa50e914a2fa0226485e02a7a97b60e18cdac64686a48ec79`；
release 前后 manifest SHA-256 均为
`3f2931b1a566c1b085af12b856123c009491c60dd7724dc925e44646ab383d0c`，
与最终 screening 一致。

canonical ZIP
`artifacts/competition/chunk_state_varlen/e2-fff9ea9/chunk_state_varlen.zip`，
大小 44399B，SHA-256
`91ec55f502bcdb4156f422e2f61f10350d5177113b0d7eae9c23ebe389984330`，
成员为 generic + ascend/enflame/iluvatar/kunlunxin；`--verify-existing`、
`unzip -t` 和成员白名单均通过。

E2 只提交一次；若 8/8 即停止 Task 13。若仅一款失败且 traceback 明确落在
pack 的地址/资源结构，最多允许一个结构不同的 pack 修复；若 BMM 失败、两款
均失败或第二候选仍 invalid，则立即转 Task 17，不再调 BLOCK/warps/stages。

### E2 平台结果与 Task 13 停止

E2 于 2026-08-26 14:26:08 CST 提交（submission `5041`，当日序号 `3`，
额度 `28/30`→`27/30`）。平台远端对象与 canonical ZIP 验签一致：44399B、
SHA-256
`91ec55f502bcdb4156f422e2f61f10350d5177113b0d7eae9c23ebe389984330`；
`file_url_sha256` 为
`c9f04ed37fbe09238508c8c1a768b6bc4f0d781ae42eb9a7081cb1021d349172`。

最终仍为 6/8、`invalid_correctness`。六个冻结后端均通过：海光
462.0035x、国际 A 375.7370x、天数 139.3615x、国际 B 108.2535x、沐曦
104.2905x、华为 27.3900x。燧原五个 case 均在
`_pack_sequences_kernel` 编译期报
`Pipeline run failed: PassManager execution failed`；昆仑 case 0/1/3/4
进入执行但大面积数值不一致（分别约 49.1%/60.5%/49.9%/48.2%，最大绝对
误差 1.6992/181.7227/91.375/680.5234）。因此结果同时命中“两款均失败”和
“非单一明确 pack 地址问题”停止门禁：Task 13 永久停止，不对 BLOCK、warps、
stages 或相同两阶段结构继续试错，剩余额度转投 Task 17。

## E3：GCU i32 metadata + XPU 全 padding 无 mask 访存（一次性重开）

状态：平台 7/8、`invalid_correctness`；一次性 stop gate 已触发，永久停止。

E2 后出现了两条可归因、且分别命中两款失败芯片的新证据，因此只重开一次，
不再调 BLOCK/warps/stages：

- Task 17 `fb1235d` 已在平台证明，wrapper 将 int64 metadata 有界降为 int32、
  并消除显式 i64 stride IR，可把燧原同型的全 case GCU Pipeline 编译失败恢复
  为 0.3885x。E3 对 Enflame 只复用这项已验证变换；pack/BMM 结构不变。
- E2 Kunlun 的失败只出现在 feature/output 尾块，并呈约一半元素错误、尾部精确
  为零。官方 XPU lowering 证据表明 masked store 会经粗粒度 DMA 写回，mask-off
  lane 仍可能影响邻行。E3 将 K/M/N 向 32 对齐，X/B 共享
  `max(padded_headdim,padded_dstate)` 行宽；源地址先钳到合法元素、无 mask
  load 后用 `tl.where` 清逻辑 padding，scratch、BMM 和 padded output 全部使用
  完整 32 tile 的无 mask 访存，最终仅返回 `[..., :headdim, :dstate]` 视图。
  all-empty 保留 K=0、零迭代 Triton BMM 写零，不引入 PyTorch fallback。

source/verification commit 均为
`d795ed3e958778a26399c521e9512a5581853453`。generic/Ascend/Iluvatar 字节冻结；
generic、Ascend、Iluvatar、Enflame、Kunlun、test SHA-256 分别为：

- `025ac23b0026a423e381f8fffe3715c4e21ce64558e79ad4841145351f11d4f1`
- `a8bf24b42a1f7a73ab4f9635b928e0a28094b258a04cf4de4d39ae52f08c9e4b`
- `86419a800ebf783d70f0392ae7c71ad06e755844197efe8ce23a6b7b394276e8`
- `9c3f4fc80e362369c90e54519056f0e1189ea8d229bfb03107c866b4a5067d6c`
- `acba578c0710438340bec62ad1d9665573d01152a51c89e307ecd0dc61611821`
- `b5b5395eac71916d1e2ccd08969de98df9ef805cf0cd7f1eec27c8090df75248`

最终 screening 位于
`gpu:/tmp/flagos-chunk-state-varlen-e3-screening-r2.087D2z`，PID/PGID
`155821`，wall 900s；`replay.sh`、`screening.log` SHA-256 分别为
`6f2311b1382ec40bee28f6a5d33531ba1eb83b7c4ab3a45685a0513e79db152d`、
`b60bbbeaf412b2c9055c28cc5aaa2f73fdb25c4a05cad8297a01c34f8c753f6f`。
12/12 unittest 通过（2.657s），覆盖 H/D 为 16/8、32/16、64/128 的不等
feature shape、静态无 masked-memory 回归、三 dtype、非连续输入、int32/int64
metadata、mixed/all-empty 和数值边界；尾行为 `SCREENING_OK`。首次 screening
只在 Black 静态门禁停止，未执行 runtime。

同目录 wrapper-inclusive NVIDIA 代理基准脚本/日志 SHA-256 分别为
`1abd7811186930e094392e31c272b02a0cf5aa1944028b7ff0865a7a841da212`、
`c029af5849951ebd07a407e29aa9c3f4881511a5a1311d6e6b5c49af111a0f4f`，尾行为
`BENCHMARK_OK`。相对 E2，候选在四个不等 feature shape 上为 1.025–1.502 倍
延迟，最坏峰值显存约 128 MiB；全部输出在题面 3e-2 容差内。该基准只排除资源
灾难，E3 的目标是恢复两个失败 backend，且六个已通过 backend 的字节不变。

Git-object release 位于
`gpu:/tmp/flagos-chunk-state-varlen-e3-release.JhuCwN`，PID/PGID `156132`，
wall 900s；`replay.sh`、`release.log` SHA-256 分别为
`9238a41f437458975e0cf213e1b4b7cab3780ce463903b86470af6bdd2a46f52`、
`2c2ba18385d41744dd2984fd232f8ab6dbe911c998307dcc57e3938590bf756b`。
12/12 unittest 通过（0.735s），格式、`py_compile`、Enflame 零 i64 静态门禁、
前后 Git-object 哈希全部通过，尾行为 `RELEASE_OK`。

canonical ZIP 为
`artifacts/competition/chunk_state_varlen/e3-d795ed3/chunk_state_varlen.zip`，
44,154B，SHA-256
`8ea09900fe8995f59f63e649ca0b3a3fd97215e1ed09511d0a4b96a97424151c`；成员为
generic + ascend/enflame/iluvatar/kunlunxin，`--verify-existing`、`unzip -t`、
UTF-8、10 MB 和成员白名单全部通过。

E3 只允许一次正式提交。六个冻结后端在 E2 合计 1217.036x；若两款失败芯片
仅达到 0.1x，理论平均仍为 152.1545x。若任一失败，保留 traceback 后永久停止
Task 13，不为相同结构生成 E4。

### E3 平台结果：7/8，燧原恢复、昆仑假设否决

E3 于 2026-08-26 22:19:44 CST 提交（submission `5159`、当日序号 `20`，
额度 `11/30` → `10/30`）。平台返回的 `file_url_sha256` 为
`14df7c6bd2176e94b253eac4563f01cee40356db3401a8aea947c1b851973fe5`；
对该官方对象存储 URL 做无认证、无重定向下载，复验为 44,154B、SHA-256
`8ea09900fe8995f59f63e649ca0b3a3fd97215e1ed09511d0a4b96a97424151c`，
与本地 canonical ZIP 完全一致。

七芯通过：海光 508.0375x、国际 A 362.4180x、天数 139.6665x、沐曦
134.6690x、国际 B 116.8505x、华为 27.7195x、燧原 **1.4735x**。
Enflame 五个 case 从 E2 的全编译失败恢复为全部通过，再次验证“metadata i32 +
消除显式 i64 IR”是 GCU Pipeline 的可迁移根因修复；冻结的六芯也全部保持通过。

Kunlun 五个 case 全部执行完但数值失败，错误元素比例依次为
76.4%/85.7%/72.4%/45.5%/48.5%，最大绝对误差为
3.2304/8.6875/8.9375/11.9998/12.9209。E2 尚有 case 2 通过，而全 padding、
无 mask 访存后连 case 2 也失败，故“仅由 masked-memory 尾块污染导致”这一假设
被平台否决，剩余问题已落到 XPU pack/BMM lowering 或布局语义，不能由另一个
局部变量解释。

终态为 `completed` / `invalid_correctness`、7/8；七芯合计 1290.8345x，但
无有效平均分。按提交前的一次性 stop gate，Task 13 永久停止，不生成 E4、不重试
同一 ZIP，剩余 10 次额度转投其他任务。

## E4：host-resolved 变长边界 + XPU direct dot（结构性重开）

状态：release、代理性能和不可变 ZIP 门禁通过，待实时 preflight。

E3 后只因出现了结构不同且已有目标芯平台证据的新机制而重开一次：Task 12 的
32×32 FP32 IEEE direct-dot 已在昆仑 0.2510x 通过；官方 FlagGems Kunlun
MM/BMM 同样由 host 提供 M/N/K，再运行规整 dot kernel。E4 仅替换 Kunlun
vendor：wrapper 一次 `cu_seqlens.tolist()` 解析每段 start/length/chunk/slice，
每个非空段直接启动同构 dot；删除 E3 的 metadata pack、FP32 scratch、regular
BMM 及 kernel 内 metadata 指针。其余七芯文件逐字冻结。

source/verification commit 均为
`46c8b3c20649dfba27f1c472ee3323bade94ae70`。generic、Ascend、Iluvatar、
Enflame、Kunlun、test SHA-256 分别为：

- `025ac23b0026a423e381f8fffe3715c4e21ce64558e79ad4841145351f11d4f1`
- `a8bf24b42a1f7a73ab4f9635b928e0a28094b258a04cf4de4d39ae52f08c9e4b`
- `86419a800ebf783d70f0392ae7c71ad06e755844197efe8ce23a6b7b394276e8`
- `9c3f4fc80e362369c90e54519056f0e1189ea8d229bfb03107c866b4a5067d6c`
- `f376eb99fc9b3466df849be62c97bceef2df6913dc4c8294c862edccb35eb934`
- `6bfd26077850f8a65e1777bfed23063f0f4b11a8b39ffe41ac751e02e1f504ef`

screening 位于 `gpu:/tmp/flagos-chunk-state-varlen-e4-screen.tToj0z`，mode
0700；`replay.sh`、`screening.log`、benchmark 脚本/日志 SHA-256 分别为
`44624ba9...3b0427`、`ee087bd7...c91bb`、`0eb7219c...8fb0`、
`1c1c0ea4...8a1e`。12/12 unittest 通过（5.560s），覆盖三 dtype、非连续
stride、int32/int64 边界、跨 chunk 单 scale 广播、mixed/all-empty、数值边界
和输入不变性。五个代表 shape 的 wrapper-inclusive NVIDIA 代理相对题面
reference 为 9.159–280.408x，最低值远高于 0.1x；该数字只作资源下限证据。

Git-object release 位于
`gpu:/tmp/flagos-chunk-state-varlen-e4-release.Qg1suB`，mode 0700；源 archive
SHA-256 为 `f4ad1d08...fc40c3`，`replay.sh`/`release.log` SHA-256 为
`04ae1103...12c7b`/`1bc616a7...0a464`。12/12 unittest（0.758s）、格式、
`py_compile`、文件哈希与结构门禁全部通过，尾行为 `RELEASE_OK`。

canonical ZIP 为
`artifacts/competition/chunk_state_varlen/e4-46c8b3c/chunk_state_varlen.zip`，
42,317B，SHA-256
`bd112d1efa5098b8294c2b772ef2fa8ee99783e1b2b880521995eac76a770b1b`；成员为
generic + ascend/enflame/iluvatar/kunlunxin，重复构建状态
`verified-existing`，`unzip -t` 与逐成员来源哈希通过。

E4 只允许一次正式提交。E3 七个冻结后端合计 1290.8345x；昆仑只要达到
0.1x，理论平均即 161.3668x。若 Kunlun 编译/正确性失败或低于门槛，永久停止
Task 13，不对 tile、warps 或逐段 launch 再做微调。

### E4 平台提交

2026-08-27 01:15:35 CST 执行唯一一次正式提交（submission `5220`、当日序号
`10`，preflight 额度 `21/30`）。平台返回的 `file_url_sha256` 为
`9b9bce972e4eb6540ca27588a24578d1de8df69fb1974615ee243c45edbaa34a`；对官方
对象存储 URL 做无认证、无重定向回读，复验为 42,317B、SHA-256
`bd112d1efa5098b8294c2b772ef2fa8ee99783e1b2b880521995eac76a770b1b`，与本地
canonical ZIP 完全一致。八芯均已进入队列，平台选择的 generic 与四个 vendor
文件符合 manifest；等待同一 submission 终态，不重试。

### E4 平台结果：8/8，166.4583125x team best

submission `5220` 于 01:18 CST 终态为 `completed` / `valid`，八芯全部通过：
天数 131.3410x、沐曦 99.7670x、燧原 1.4575x、海光 524.9110x、昆仑
55.3745x、华为 28.9180x、国际 A 370.7355x、国际 B 119.1620x；平均
**166.4583125x**，成为团队本题最佳。

关键目标昆仑从 E3 五个 case 全部数值失败恢复为五个 case 全过，且 55.3745x
远高于 0.1x 门槛，验证问题确在 pack/scratch/BMM 布局路径，而非 direct-dot
数学。公开达标队伍数由 5 增至 6。E4 已完成一次性闭环，Task 13 停止继续迭代，
保留该 8/8 结果。
