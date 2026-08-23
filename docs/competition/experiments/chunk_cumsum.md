# Task 10 `chunk_cumsum` 实验记录

## 当前结论

状态：S1 已修复固定 Mamba 尾块前缀语义，通过提交字节发布门禁并生成规范 ZIP；
E1 `EXACT_SHAPE` 去 mask 未过预设性能/资源门禁，保留 S1；未提交平台。

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

### S1 剩余风险与下一步

- 题面 reference 的 partial-tail 矛盾与平台隐藏 shape 仍未知；S1 选择固定 Mamba
  语义，同时保持整除 shape 与 S0 逐字节等价路径。
- `tl.cumsum`、runtime mask 和 3D grid 尚未由其余七类编译器验证；没有平台逐芯
  反馈前不再追加本地 mask/tile 猜测。
- 当前可交付物是 S1，状态为候选就绪、未提交；下一门禁是用户针对 Task 10、
  ZIP 绝对路径/哈希和平台实时额度作当次确认。
