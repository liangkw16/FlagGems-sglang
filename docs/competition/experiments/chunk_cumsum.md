# Task 10 `chunk_cumsum` 实验记录

## 当前结论

状态：S1 已修复固定 Mamba 尾块前缀语义，通过提交字节发布门禁并生成规范 ZIP；
E1 `EXACT_SHAPE`、E2 `chunk_size<=8` 两 warps、E3 `chunk_size<=4` 两 warps及
E4 `chunk_size==3` 两 warps均未过各自预注册门禁，保留 S1；未提交平台。

当前候选：`s1-a4e84aa`，ZIP SHA-256
`f9fd0d595aeb5a4a4da76514321790815fbad9ccc39faa447c8bfa120f0e7db9`。

## S0：generic baseline

状态：历史基线；已被 S1 的尾块语义修复替代，不再建议上传

验证时间：2026-08-24 01:20–01:28 CST

源码 commit：`3fac516`

### 契约与固定来源

- 接口：`chunk_cumsum(dt, A, chunk_size, dt_bias=None,
  dt_softplus=False)`。
- 输入 `dt=[B,T,H]`；可选 FP32 bias/softplus 后 clamp 到非负，生成
  `[B,H,nchunks,chunk_size]` 的 FP32 `dt_out`，再按 head 乘 `A` 并沿
  chunk 维做 FP32 cumsum；返回顺序必须是 `(dt_out, dA_cumsum)`。
- 题面支持 FP16/BF16/FP32 输入和八款芯片；核心计算必须走 Triton。
- 固定来源：Mamba v2.2.4 `ssd_chunk_state.py`，以及公开本地引用
  `community/master` 的多芯实验。S0 没有复制后者的设备识别、cache hint、
  异常重试或 vendor 分支。

### 构建身份

| 项目 | 值 |
| --- | --- |
| 源文件 SHA-256 | `ce9ac83b61fe67c684060d7aaa1aac9238995b21179ddf80f48f019944c55a8d` |
| 测试 SHA-256 | `379396fb34b4eb27f3bed7196b5b7c508da1c61d5b37df0f0b42fed4d0738a1b` |
| ZIP | `artifacts/competition/chunk_cumsum/s0-3fac516/chunk_cumsum.zip` |
| ZIP SHA-256 | `81a1cff508d5ca8a7eb921d8644e4061b40382ea2ab9e4ce12a231118e48c607` |
| ZIP manifest | 顶层 `chunk_cumsum.py`，3974 bytes |
| 远端证据目录 | `gpu:/tmp/flagos-batch2.SQaIX2` |

### 候选与代理验证

- 单个标准 Triton kernel，3D grid `[head tile, chunk, batch]`；每 program
  处理最多 8 个 head 和一个 chunk，按真实 input stride 读取。
- `BLOCK_SIZE=next_power_of_2(chunk_size)`，tile 上限启发式 4096 元素，
  4 warps、1 stage；无 autotune、设备判断、fallback 或 vendor 文件。
- RTX 5070 Ti 16 GB 上 unittest 1/1 通过；循环覆盖三 dtype、chunk
  `5/16/64`、bias/softplus 开关、非连续输入、tail 和输入不变性。
- wrapper-inclusive 代表 shape 的代理加速比为 `3.184x–29.044x`；这只用于
  筛除明显慢候选，不能替代平台逐芯成绩。
- ZIP 由 commit `3fac516` 直接生成，`unzip -t`、UTF-8、单一 `.py`、
  10 MB、basename 和 ZIP 内源码哈希门禁均通过。

### 已知风险

- 公开 reference 的 `ceil(T/chunk_size)` 与后续 reshape 暗含 `T` 可整除；
  S0 对尾 chunk 做安全掩码和零初始化，但平台语义仍以真实 harness 为准。
- `tl.cumsum` 和二维 tile 尚未由其余七类编译器验证；单芯失败后才增加对应
  vendor，不预先推断 A/B 映射。
- 尚未上传或消耗额度；上述 ZIP 需要用户当次确认。

## S1：固定 Mamba 尾块语义修复

验证时间：2026-08-24 05:04–05:14 CST

### 根因与契约边界

- 题面 reference 使用 `ceil(T/chunk_size)`，但未补零便直接 reshape；正常非空且
  `T % chunk_size != 0` 时该代码本身不可执行，因此无法证明平台是否只给整除 shape。
- 固定 Mamba v2.2.4 commit
  `95d8aba8a8c75aedcaa6143713b11e745e7cd0d9` 明确把无效 `dt` lane 置零，
  再对零填充做 inclusive cumsum 并写满逻辑 chunk。因此 `dt_out` 尾部为零，
  `dA_cumsum` 尾部重复最后一个有效前缀值。
- S0 两路 store 都用了包含 `offsets_s < seqlen` 的输入 mask；零初始化使
  `dt_out` 恰好正确，却错误地把 `dA_cumsum` 尾部留成零。
- 新回归先在 S0 上失败：9/24 个元素不符，最大绝对误差 `0.5504680872`。
  S1 只把输出 mask 改为 `head valid && lane < chunk_size`；`values` 已将无效
  序列 lane 置零，所以两路输出与固定上游一致。平台真实隐藏 shape/ oracle 仍未知。

### 构建身份

| 项目 | 值 |
| --- | --- |
| source / verification commit | `a4e84aa834f0584420cd832ffc823641ee701593` |
| 源文件 SHA-256 | `5ee2294d4ad42c1bf355adc3d9418c8ae3dc7d43ccec477dcdc4e333dda0a6ef` |
| 测试 SHA-256 | `1bf761d43880f975ae155022c8bf625ab4a1680b8e2d6f6d8c172b1a6a9cc5e5` |
| ZIP | `artifacts/competition/chunk_cumsum/s1-a4e84aa/chunk_cumsum.zip` |
| ZIP SHA-256 | `f9fd0d595aeb5a4a4da76514321790815fbad9ccc39faa447c8bfa120f0e7db9` |
| ZIP manifest | 顶层 `chunk_cumsum.py`，4058 bytes；ZIP 4186 bytes |
| 开发/A-B 目录 | `gpu:/tmp/flagos-chunk-cumsum.lYW6xo`，mode 0700 |
| 最终发布门禁目录 | `gpu:/tmp/flagos-chunk-cumsum-release.oayxzR`，mode 0700 |
| 平台 | 未提交；未经用户针对该 Task/绝对路径/SHA/实时额度当次确认不得上传 |

打包器从 source commit 生成规范存储 ZIP；`unzip -t`、成员名、UTF-8、10 MB、
ZIP 内源码与 commit 逐字节一致等门禁均通过。S0 历史 ZIP 保持不变。

### 正确性与发布门禁

远端环境为 RTX 5070 Ti 16 GB、driver 610.57.04、Python 3.12.13、
PyTorch 2.13.0+cu130、Triton 3.7.1、CUDA 13.0。

最终提交字节的 Black 79、isort、flake8、`py_compile` 和公开接口 unittest
4/4 通过，覆盖：

- FP16/BF16/FP32 及题面容差；
- bias 与 softplus 四种独立组合、softplus 的 `20/21` 阈值分支；
- 非连续 `dt/A/dt_bias` 的互异真实 stride；
- 固定上游 partial-tail 的 `dt=0 / dA=carry`；
- 空 batch/seqlen/head 与全部输入不变性。

发布探针另有 9/9 个三 dtype 主要 shape 通过。五组交替、wrapper-inclusive
`warmup=25, rep=100` 相对题面等价 padded reference 的代理范围为
`2.682292x–31.803127x`。9 个编译变体最高 40 registers、2 KiB shared，
全部 0 spill/global scratch/local load-store，4 warps、1 stage，PTX 无 TF32。

| 证据 | SHA-256 |
| --- | --- |
| S0 预期失败日志 | `e5b41d370622c776322636cffc63aa166e8ead83e0450ff88d13abe8a24368c9` |
| S1 初筛日志 | `c720ff6b9a06a16435d642b846ebd898bb3523304fad378b350c204d65eb4e78` |
| 最终发布门禁日志 | `b824fab61ebdf1e611c33cb2a83b8f6603bb9097345431ac695d717fe9f64c4a` |
| 发布探针 / 日志 | `2cc935e607546179a0e546394dfa78b16d46f85c264bb959d29a6d71bdfef63f` / `ba1a0f9383d86093e26326df0859e0002a3d72372fda8250e1077bc8ea9fb42a` |

| 运行 | PID | 启动时间（CST） | 日志 |
| --- | ---: | --- | --- |
| S0 预期失败 | `77738` | `2026-08-24 05:04:21` | `tdd-old-source.log` |
| S1 初筛 | `77862` | `2026-08-24 05:04:52` | `correctness-fix.log` |
| S1 最终发布门禁 | `78570` | `2026-08-24 05:13:00` | `release-validation.log` |
| S1 发布探针 | `78664` | `2026-08-24 05:13:47` | `release-probe.log` |

## E1：`EXACT_SHAPE` 编译期去 mask，拒绝

E1 严格基于 S1，只新增一个 `EXACT_SHAPE: tl.constexpr`。仅当
`BLOCK_SIZE == chunk_size && T % chunk_size == 0 && H % BLOCK_H == 0`
时移除 `dt/A/bias` load、尾部清零和两路 store 的 predicate；fallback、grid、
tile、4 warps、1 stage、stride 和公式不变。固定公开 `community/master` commit
`0e8023da851c1a2917b628d5296d4f9e68b6ca56` 有同类 `NO_MASK` 条件，只支持
该实验形态，不能把其混合优化成绩归因于本变量。

候选先通过 4/4 unittest，再通过 27 个独立正确性 case（三 dtype，15 个 exact、
12 个 fallback）。PTX 证明 exact 的全局 load/store predicate 从 S1 的 4–9 个
降为 0；sequence-tail control 仍保留 8 个。五组 AB/BA 交替、public-wrapper
`warmup=25, rep=100` 的预设门禁为：affected 几何平均 `>=1.05x`、每 dtype
`>=1.02x`、稳定单点不低于 `0.98x`、control 位于 `0.98–1.02x`，且资源不增。

| 指标 | 结果 |
| --- | ---: |
| 12 个 affected 点几何平均 | `0.999484x` |
| FP16 / BF16 / FP32 affected | `0.999215/1.000000/0.999237x` |
| 最差 affected 点 / 单组 | `0.996865x / 0.932450x` |
| 12 个 control 几何平均 | `1.001088x` |
| control 范围 | `0.997389x–1.005236x` |

| affected shape | dtype | S1 (ms) | E1 (ms) | S1/E1 |
| --- | --- | ---: | ---: | ---: |
| `[2,4096,64]`, CS64 | FP16 | 0.010176 | 0.010208 | 0.996865x |
| `[4,1024,128]`, CS16 | FP16 | 0.010240 | 0.010240 | 1.000000x |
| `[8,512,32]`, CS128 | FP16 | 0.006144 | 0.006144 | 1.000000x |
| `[1,2048,8]`, CS64 | FP16 | 0.004096 | 0.004096 | 1.000000x |
| `[2,4096,64]`, CS64 | BF16 | 0.010208 | 0.010208 | 1.000000x |
| `[4,1024,128]`, CS16 | BF16 | 0.010240 | 0.010240 | 1.000000x |
| `[8,512,32]`, CS128 | BF16 | 0.006144 | 0.006144 | 1.000000x |
| `[1,2048,8]`, CS64 | BF16 | 0.004096 | 0.004096 | 1.000000x |
| `[2,4096,64]`, CS64 | FP32 | 0.010464 | 0.010496 | 0.996951x |
| `[4,1024,128]`, CS16 | FP32 | 0.012288 | 0.012288 | 1.000000x |
| `[8,512,32]`, CS128 | FP32 | 0.006144 | 0.006144 | 1.000000x |
| `[1,2048,8]`, CS64 | FP32 | 0.004096 | 0.004096 | 1.000000x |

资源也未过门禁：CS64 BF16 为 37→38 registers；CS128 FP16/BF16 为
40→42、FP32 为 39→40；低并行 FP32 为 37→38。shared 均不增且 0 spill/
scratch/local，但去 predicate 没有转化为代理性能。因此拒绝 E1，不 commit、
不生成 E1 ZIP，也不做事后 shape 阈值细分。

| E1 证据 | SHA-256 |
| --- | --- |
| 未提交候选源码 / 已提交测试 | `d6cd50e2fc1fa60528a3af8c07ded1f3cadc47e5a7486b50e1031ec19aed0296` / `1bf761d43880f975ae155022c8bf625ab4a1680b8e2d6f6d8c172b1a6a9cc5e5` |
| E1 unittest 日志 | `d59fb75809ebc690ffccf61e0ab41bb1009dd9953d7733db2877f13a342acebf` |
| A/B 脚本 / 日志 | `76f375ad9e0ca557ff622a2b365cc9b87c28212e98f4c8e7dec62ae20d53cc09` / `8b152def698583cdf7b530c2f9676f652d4e3d5b633a45c216952ac6a114f367` |

E1 unittest PID `78012`（05:07:41），A/B PID `78311`（05:09:57）。
独立只读复审确认本次 `promote=false` 不受 harness 配对问题影响；若未来候选可能
晋升，须把最差单组纳入 gate，并用五个配对 speedup 的中位数，而不是两个独立
时间中位数之比。

## E2–E4：tiny chunk 两 warps，拒绝

三次 screening 都只改 launch 参数，kernel、grid、tile、stride、公式、4-warps
fallback 和 1 stage 不变；均以 S1 源码为 baseline，未 commit、未生成 ZIP。
E2 的宽阈值先暴露资源退化，E3 缩到 `chunk_size<=4` 后仍未达到整体收益门槛；
E3 中事后表现最好的 CS3 点又全部开启 softplus，不能直接支持 `chunk_size==3`。
因此 E4 在运行前冻结全新 shape/seed、四个 bias×softplus cell、四个 shape family
和严格 3 AB/3 BA 门禁，只运行一次，避免继续按结果切片。

### E2：`chunk_size<=8`，性能与资源均失败

候选源码 SHA-256 为
`ba4cdac2ea848c1af834b2ac662ef9fdefbcb01d2dac8037c9c6f36724e451c3`，
screening base commit 为 `9dd09424e937503315cedb7a4214b70e214a381d`。
4/4 unittest 和 192 个扩展正确性 case 通过；72 个 affected、12 个 control 的
方向性计时结果如下。该首轮脚本没有 batch20，短 kernel 结果存在计时量化，只能
作为淘汰证据；候选同时明确未过资源门禁，结论不依赖边界计时精度。

| 指标 | 结果 |
| --- | ---: |
| affected 几何平均 | `1.039474x` |
| FP16 / BF16 / FP32 | `1.044884/1.036623/1.036935x` |
| 最差 affected 点 / 单组 | `1.000000/1.000000x` |
| control 几何平均 / 范围 | `0.998850x / 0.989378–1.000000x` |

CS5–8 的两-warps 编译物把 shared 从 0 增至 256 bytes；CS5–7 registers 也从
`17–20` 增至 `19–22`，虽无 spill/scratch/local load-store，仍违反资源不增门禁。
远端目录为 `gpu:/tmp/flagos-chunk-cumsum.8P8PPQ`；gates 和 screening PID/PGID
分别为 `91166`、`91386`，screening 时间为 08:46:29–08:48:11 CST。

| E2 证据 | SHA-256 |
| --- | --- |
| gates 日志 | `5072bb5bca3ce28b98e6d448523e3c1c678674e6f1553edd9ade4e555d8a1de7` |
| A/B 脚本 | `f4a254c5c903f00aeb7cd41b57d48ed1a0c5774331814f0c937c5ba83860b295` |
| 原始 JSON 日志 | `f8f7a85f91a7b909b318eeda89ada8151f2dc4d32856c310a229afb40b3c2c40` |

### E3：`chunk_size<=4`，性能失败

候选源码 SHA-256 为
`28c4c277c3600bbcf3068d16b22ff4c55b8289d3e7c8d90c467ce727b30ecc92`，
screening base commit 为 `ecd27066183af03e2f1207db29b2d2514f9111f6`。
4/4 unittest、216 个扩展正确性 case 和资源门禁通过；batch20、五组配对 AB/BA
覆盖 36 个 affected、18 个 control，但整体未达到预设 `1.05x`。

| 指标 | 结果 |
| --- | ---: |
| affected 几何平均 | `1.025243x` |
| FP16 / BF16 / FP32 | `1.026578/1.027181/1.021979x` |
| 最差 affected 点 / 单组 | `0.990538/0.984328x` |
| control 几何平均 / 范围 | `0.999709x / 0.995282–1.000000x` |

最高资源为 40 registers、4 KiB shared，全部 0 spill/scratch/local load-store。
远端目录为 `gpu:/tmp/flagos-chunk-cumsum-e3.i3OghE`；gates 和 screening
PID/PGID 分别为 `91969`、`92103`，screening 时间为 08:53:35–08:54:46 CST。

| E3 证据 | SHA-256 |
| --- | --- |
| gates 脚本 / 日志 | `a730ac849f696326deacf042c287aa1657612a61cdd784f7ab80be8013413f64` / `17586890f21d720b2f635294a12460b9363143f8cfcaffdf95d10c0e9595270c` |
| A/B 脚本 / 原始 JSON | `5e736ca533fc8ccb8b05a5e568604f3413fa89cae04aca3e0a30964a97543a51` / `87ae3d3feec4e38b3217bfbd228913e266cd1106883fbb9d06768ec28f332330` |

### E4：`chunk_size==3`，全新留出集 shape-family 失败

候选源码 SHA-256 为
`9bd77c2b19cc1445b84f1ff4eabbe3b440f27d8b857e55690efcdba7ea6bd9f6`。
运行前独立审查并冻结四个新 family：high exact `[2,1008,96]`、high tail
`[2,1007,63]`、medium exact `[1,384,32]`、low tail `[1,95,7]`；tail case 使用
非连续输入。每个 family 覆盖四个 bias×softplus cell 和三 dtype，共 48 个
affected；CS2/CS4 controls 共 24 个点。batch20、六轮严格 3 AB/3 BA，晋级除
整体 `>=1.05x` 外，还要求每 dtype、softplus 分层、四个 flag cell 和四个
family 各自 `>=1.02x`。

4/4 unittest、216 个扩展正确性 case、全部 control 和资源门禁通过。总体收益
达到 `1.057866x`，但只集中在大 shape；medium/low family 分别只有
`1.009096x/0.999916x`，故 `performance_gate=false`，不按结果继续缩窄条件。

| 指标 | 结果 |
| --- | ---: |
| FP16 / BF16 / FP32 | `1.058507/1.059066/1.056026x` |
| softplus false / true | `1.030372/1.086092x` |
| flag cells `b0s0/b0s1/b1s0/b1s1` | `1.028041/1.081526/1.032709/1.090678x` |
| family `high_exact/high_tail/medium_exact/low_tail` | `1.132476/1.095966/1.009096/0.999916x` |
| 最差 affected 点 / 单组 | `0.999421/0.984357x` |
| control 几何平均 / 范围 | `0.999974x / 0.999005–1.000497x` |

Affected 的 base/candidate 均最高 20 registers、0 shared；controls 均最高 22
registers、0 shared，全部 0 spill/scratch/local load-store。远端目录为
`gpu:/tmp/flagos-chunk-cumsum-e4.sJGwPL`；gates PID/PGID `92460`，screening
PID/PGID `92624`，运行时间为 09:03:44–09:05:35 CST。

| E4 证据 | SHA-256 |
| --- | --- |
| gates 脚本 / 日志 | `c9a6130973e76ea3be68dd03fddeda02f8dd704718969c60a10bfe5ff239f567` / `101775d5b6c00a9aca1c9d4c467d7bc7bf3a145da65fe2b45075b52badb32931` |
| 留出脚本 / 原始 JSON | `4bc520eacab419cfbed227bade368b7cc01e854c426cf4aebbcf24b9bb20ba93` / `283534b7a0222acb5b736afba9f1ccd7009fc86ef37c7b79b7231e933eda977b` |
| 空 stderr | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

### S1 剩余风险与下一步

- 题面 reference 的 partial-tail 矛盾与平台隐藏 shape 仍未知；S1 选择固定 Mamba
  语义，同时保持整除 shape 与 S0 逐字节等价路径。
- `tl.cumsum`、runtime mask 和 3D grid 尚未由其余七类编译器验证；没有平台逐芯
  反馈前不再追加本地 mask/tile 猜测。
- 当前可交付物是 S1，状态为候选就绪、未提交；下一门禁是用户针对 Task 10、
  ZIP 绝对路径/哈希和平台实时额度作当次确认。

## S2：hybrid grid Ascend/Kunlun vendor（首投候选，≤2 次预算）

状态：release 门禁通过，候选就绪

grid 审计同 Task 11：3D `(head_blocks, nchunks, batch)` 展平总数可超
65535。吸取 Task 11 E1a 教训（无条件折叠 vendor 使华为 0.0255x、昆仑
0.012x——benchmark 规模未超限时折叠循环纯属开销），本 vendor 采用
**hybrid**：`total ≤ 65535` 走与 generic 逐字节相同的 kernel 与 3D grid，
超限时才切换到折叠 kernel（1D `min(total,4096)` + div/mod 分解）。两个
kernel 共存于同一 vendor 文件，wrapper 按 total 分派。新增回归
`test_vendors_cover_folded_grid`（2×16384×96、chunk 64、总 program >4096，
fp32/fp16 × softplus 开关 × 三模块）。screening
`gpu:/tmp/flagos-cc-v2.1mWuse`（首跑 Black 折行经远端格式化回拷），最终
PID/PGID `113930`（02:41 前后，wall 900s），5/5 unittest（0.842s），
`screening.log` SHA-256
`3ac450cebd5ae38ab02e12329c01be4b91c3f88913ccd988941cce46406963b4`。
release `gpu:/tmp/flagos-dual-release.*/a`，source/verification commit
`5e776b5a9b22c77a547947291cacec45b4707ec2`，`RELEASE_OK`，`release.log`
SHA-256
`01146b80a3171c4a260ce013326cb40958a578ea8a887ac21878674096595f60`。
canonical ZIP `artifacts/competition/chunk_cumsum/s2-5e776b5/chunk_cumsum.zip`，
SHA-256
`d104ad83075e0b8f2bf34c8e82f5c6dbd396dc19dca65fc3b872a5eda15ec835`，
成员 generic + ascend/kunlunxin，`unzip -t` 通过。

### S2 平台首投：7/8（华为 UB 溢出）→ S2b tile 收缩（第 2/2 次）

S2 于 02:47:12 CST 提交（submission `4478`，当日序号 `14`，额度区间
`18/30`→`17/30`）。七芯通过（含昆仑 hybrid vendor 正确通过）；华为在
correctness case 3/4 以 `MLIRCompilationError` 失败：
`ub overflow, requires 2621696 bits while 1572864 bits available`——大
chunk_size（≥128）下 `block_h=8` 的 tile 触发 Ascend Unified Buffer 溢出，
与缓存的 Ascend vector operator 指南"UB overflow 时减小 tile"一致。
S2b 把华为 vendor 的 `block_h` 由 `min(8, 4096 // block_size)` 收缩为
`min(8, 512 // block_size)`（chunk 64 不变，128→4，256→2），并把回归扩为
三 shape：`(2,16384,96,64)`（原 grid 路径）、`(4,32768,288,256)`（总
program 73728 > 65535，折叠路径 + UB 边界）、`(1,8192,32,256)`。screening
`gpu:/tmp/flagos-cc-v3.6iw08X`（含一次 Black 回拷），最终 PID/PGID
`114629`（02:54:27，wall 900s），5/5 unittest（1.653s），`screening.log`
SHA-256
`ea89c0a70427d4feec12f387fc2bd49469bd9dd6d6e4eb76b4e91cc3e3785eb4`；ascend
vendor blob
`24b2a5d3d5aec1ac6bbc3dcf470bf578174e4b0889d1a2eafccb1a90610e1873`，测试
`4c448ed9ad90dc24d41de8b4a1af892c6a27e8300b756ffa6d57540135bfdb64`。
release `gpu:/tmp/flagos-cc-v3-release.*`，commit
`63e7943ab0f0ea2a9811731feab1ec5b234436bb`，`RELEASE_OK`，`release.log`
SHA-256
`ec659f6457f476e7c6b5f620a3c81aabbd0e0446de1b0bb8a0f5d103337527b1`。
canonical ZIP `artifacts/competition/chunk_cumsum/s2b-63e7943/chunk_cumsum.zip`，
SHA-256
`c822f75d719f8919269c7566b1210b6e31dc6ce3292229a838723f7945b15923`。
本提交为 2 次预算的最后一次。

### S2b 平台结果：8/8 正确、invalid_threshold → Task 10 停止（2 次用尽）

S2b 于 02:58:15 CST 提交（submission `4493`，当日序号 `16`，额度
`16/30`→`15/30`）。华为 UB 收缩生效：八芯 correctness 全部通过（S2 的
MLIR 编译失败消除），平均 `3.5502x`；但燧原 `0.0375x`、昆仑 `0.0120x`
低于 0.1x 门槛，`invalid_threshold`。结论：华为 tile 上限经验
（`block_h ≤ 512 // block_size`）沉淀成功；昆仑与燧原的 `tl.cumsum`
lowering 是 cumsum 家族（Task 10/11 双题证实）的固有瓶颈——昆仑无论
grid 形态稳定 0.012–0.016x、燧原 0.0035–0.0375x，非 grid/配置可解。
Task 10 两次预算用尽，停止。后续若重试需在 XPU/GCU 上改写 cumsum 的
算法形式（如分块两阶段扫描），非单变量可及。
