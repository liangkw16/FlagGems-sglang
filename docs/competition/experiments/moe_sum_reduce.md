# Task 21 `moe_sum_reduce` 实验记录

## S0：generic baseline

状态：S0 已打包并通过本地门禁；等待当次上传确认

验证时间：2026-08-24 01:23–01:28 CST

### 契约

- 接口：`moe_sum_reduce(input, routed_scaling_factor)`。
- `input` shape 为 `[num_tokens, top_k, hidden_dim]`；factor 为 scalar。
- 结果是 out-of-place 的 `[num_tokens, hidden_dim]` tensor。
- 逐元素语义为
  `input.float().sum(dim=1).mul(routed_scaling_factor).to(input.dtype)`：
  FP32 累加，最终 cast 回输入 dtype。
- 容差：FP32 `1e-4/1e-4`，BF16 `1.5e-2/1.5e-2`，
  FP16 `1e-2/1e-2`（atol/rtol）。
- 支持八芯：天数、沐曦、燧原、海光、昆仑芯、华为、国际通用 A/B。
- 缓存赛题目录显示提交窗口为 2026-08-20 20:00 至
  2026-08-27 19:59:59，最低加速比 `0.1x`；提交前仍须以平台页面为准。
- 核心计算只走 Triton；无 PyTorch fallback、异常回退或设备判断。

### 固定参考

- SGLang
  [`8014d9d`](https://github.com/sgl-project/sglang/blob/8014d9d062c3cc5d393596ecdf2f7009191965df/python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py#L1163-L1249)：
  一 token 对一 hidden tile、FP32 accumulator、最后乘 scaling factor。
  上游要求 contiguous、BLOCK 2048、16 warps；这些 NVIDIA 倾向配置未复制。
- FlagGems
  [`ed2508b`](https://github.com/flagos-ai/FlagGems/blob/ed2508bcb5a03000e9774734201d840ba362cd11/src/flag_gems/fused/moe_sum.py#L24-L90)：
  使用 input/output strides 和 hidden tile，但带四档 autotune；S0 只抽取地址
  公式，不带 autotune。

### 构建身份

| 项目 | 值 |
| --- | --- |
| 源文件 | `src/flaggems_sglang/ops/moe_sum_reduce.py` |
| 源文件 SHA-256 | `52a2fc979784f2bd25e7e17b9822c23b4f438efdf062c70bfb09aba9ba732335` |
| 测试 SHA-256 | `7e9f0fc77d3a056b383ff29c17379867ba816eca5a9b858c365847cabf11e8ad` |
| 源码 commit | `3fac516` |
| ZIP | `artifacts/competition/moe_sum_reduce/s0-3fac516/moe_sum_reduce.zip` |
| ZIP SHA-256 | `ef3c30e416d24d8268a1c252261676f3e540910a8836a93d2520917580f514bf` |
| ZIP manifest | 顶层 `moe_sum_reduce.py`，2800 bytes |
| 远端证据目录 | `gpu:/tmp/flagos-batch2.SQaIX2`，mode 0700 |

### 唯一候选配置

- 2D grid：`(num_tokens, ceil(hidden_dim / 256))`；BLOCK 256。
- `TOP_K` 为 constexpr，循环在编译期展开；每个元素按 top-k 顺序做 FP32
  累加。
- 三个 input stride 和两个 output stride 全部参与 64-bit 地址计算；不调用
  `contiguous()`。
- 输出 tensor 与 input 同 dtype/device；store 自动 cast 回 input dtype。
- 显式使用 4 warps、1 stage；无 autotune 或 vendor 文件。
- 空 token/hidden 直接返回合法空输出；零 top-k 由 Triton 写出 FP32 零后 cast。

### 正确性与静态检查

本地 Python `py_compile` 通过。远端 RTX 5070 Ti 16 GB 环境：Python
3.12.13、PyTorch 2.13.0+cu130、Triton 3.7.1、CUDA 13.0、compute
capability 12.0。

远端直接执行公开接口 unittest：3/3 通过，覆盖：

- FP16、BF16、FP32，典型 `top_k=8` 和 hidden tail 257；
- 非连续 token/top-k/hidden 三维输入，验证全部真实 strides；
- 空 token、空 hidden、零 top-k；
- 输出 shape/dtype、负 scaling factor、输入不变性。

Black 79、isort、flake8 均通过。代理验证只能证明 NVIDIA 路径，不能替代
平台八芯结果。

### NVIDIA 代理性能

wrapper-inclusive；factor 0.75；每项 JIT 后使用
`triton.testing.do_bench(warmup=25, rep=100)`。结果只用于筛掉明显慢候选。

| dtype | shape | S0 (ms) | Torch reference (ms) | speedup |
| --- | --- | ---: | ---: | ---: |
| FP16 | `[1,8,4096]` | 0.004231 | 0.010365 | 2.450x |
| FP16 | `[32,8,4096]` | 0.007815 | 0.019175 | 2.454x |
| FP16 | `[128,8,7168]` | 0.027890 | 0.080839 | 2.899x |
| BF16 | `[1,8,4096]` | 0.004179 | 0.010532 | 2.520x |
| BF16 | `[32,8,4096]` | 0.007880 | 0.020411 | 2.590x |
| BF16 | `[128,8,7168]` | 0.027909 | 0.081410 | 2.917x |
| FP32 | `[1,8,4096]` | 0.004266 | 0.006441 | 1.510x |
| FP32 | `[32,8,4096]` | 0.010575 | 0.012560 | 1.188x |
| FP32 | `[128,8,7168]` | 0.050171 | 0.053337 | 1.063x |

### 已知风险与下一步

- 平台未公开 correctness/benchmark shapes；当前 shape 只是同类 MoE 代理值。
- SGLang 固定参考把 factor 标为 Python `float`，S0 也按 numeric scalar
  处理；只有平台证明会传 0-D tensor 时才增加对应 wrapper 语义。
- 2D grid、默认 launch 参数和 constexpr top-k 尚未被八种编译器验证。
- 首次平台结果若出现单芯 grid/编译失败，只为该芯加最小 vendor override；
  generic 与已通过芯片保持不变。
- ZIP 由 commit `3fac516` 直接生成；`unzip -t`、UTF-8、单一 `.py`、10 MB、
  basename 和 ZIP 内源码哈希门禁均通过。下一门禁是取得针对上述 ZIP 的当次
  用户确认；本记录不构成上传授权。

## S0 发布复核与 E1 BLOCK 512 负实验

状态：边界回归已扩展；E1 正确但收益不足且寄存器增加，已撤回；S0 与 ZIP 不变

复核时间：2026-08-24 03:12–03:15 CST

| 项目 | 值 |
| --- | --- |
| source commit | `3fac516a8d64c88b183801668a7857d969a05e37` |
| verification commit | `27da397fce11281549e2e03b88f621d67c972ac4` |
| S0 源码 SHA-256 | `52a2fc979784f2bd25e7e17b9822c23b4f438efdf062c70bfb09aba9ba732335` |
| 当前测试 SHA-256 | `bf63cfd9fd45889b3f5d05f38dfd0af600ca33128316315183394fccf55e4a03` |
| S0 ZIP SHA-256 | `ef3c30e416d24d8268a1c252261676f3e540910a8836a93d2520917580f514bf`，`verified-existing-legacy` |
| 规范 ZIP SHA-256 | `d000d65f081f1bebcd9e1f1193e9d6d2fa95886658c412df9d58aa4f47414e2d`，仅内存生成 |
| E1 临时源码 SHA-256 | `c623ee00793c27491346eed219cd0c466010fe07a15675ce5d15dc739e8d8c09` |
| 远端证据目录 | `gpu:/tmp/flagos-task21.WbcNfZ`，mode 0700 |
| baseline 门禁 | PID `72327`；03:13:03 CST；`baseline-gates.log`；SHA-256 `a8c53628c5930d2b1b55533012ad57053cbc51d8b11109eb617b33b7dd28961a` |
| E1 门禁 | PID `72442`；03:13:30 CST；`candidate-gates.log`；SHA-256 `86aced25ca207ee22b32c3889de530e58ed96849d9875cb0c8804910e5ec9ed2` |
| A/B | PID `72550`；03:14:30 CST；`ab.log`；SHA-256 `91405a825d329406015988db433dc5234e442f7139c69361952c099e689d80a6` |
| 平台结果 | 未提交；逐芯结果、均值、排名和实时额度均为 N/A |

新增第四个 unittest 方法，覆盖三 dtype ×
`hidden=255/256/257/511/512/513`、`top_k=3`、输入不变性和确定性数值；原有
top-k 0/4/8、非连续三维 stride、空维度和负 factor 回归保留。S0 与 E1 均通过
py_compile、Black 79、isort、flake8 和 4/4 unittest，远端源码/测试哈希与上表
一致。E1 另在下表 18 个组合上完整对 reference 验证，全部通过题面容差。

E1 只把 wrapper 中的 `block_size` 从 256 改成 512，grid、4 warps、1 stage、
constexpr top-k、FP32 累加、strides 和数学均不变。它把大 hidden 的 grid.y 约减半，
但编译变体的寄存器范围从 S0 的 7–31 增到 9–40；两者仍为
stack/shared/local 0。

wrapper-inclusive 五组轮换 A/B，组内 `warmup=25, rep=100`：

| dtype | shape | S0 / E1 | reference / E1 |
| --- | --- | ---: | ---: |
| FP16 | `[1,8,4096]` | 0.9985x | 2.4458x |
| FP16 | `[32,8,4096]` | 0.9885x | 2.4283x |
| FP16 | `[128,8,7168]` | 1.0230x | 2.8713x |
| FP16 | `[32,3,513]` | 1.0205x | 2.3573x |
| FP16 | `[32,1,4096]` | 1.0096x | 2.5189x |
| FP16 | `[32,16,4096]` | 1.0296x | 2.5247x |
| BF16 | `[1,8,4096]` | 0.9969x | 2.4467x |
| BF16 | `[32,8,4096]` | 1.0057x | 2.5654x |
| BF16 | `[128,8,7168]` | 1.0250x | 2.8845x |
| BF16 | `[32,3,513]` | 1.0325x | 2.4067x |
| BF16 | `[32,1,4096]` | 1.0049x | 2.5396x |
| BF16 | `[32,16,4096]` | 1.0234x | 2.5893x |
| FP32 | `[1,8,4096]` | 0.9879x | 1.4811x |
| FP32 | `[32,8,4096]` | 1.0090x | 1.1812x |
| FP32 | `[128,8,7168]` | 1.0027x | 1.0768x |
| FP32 | `[32,3,513]` | 1.0059x | 1.3533x |
| FP32 | `[32,1,4096]` | 1.0088x | 1.3449x |
| FP32 | `[32,16,4096]` | 1.0285x | 1.1293x |

18 点 S0/E1（E1 speedup）几何平均为 `1.011061x`，最差回归 `1.226%`。
它没有达到预设的 `>=1.05x` 晋级线，且寄存器上限增加，因此不提交 E1，也不
生成新 ZIP。工作树源码已恢复到 S0 SHA-256；扩大后的测试继续保留。下一次源码
迭代等待 S0 八芯结果，不用 NVIDIA 的约 1% 噪声收益换取跨芯占用风险。

## E2/E3：NVIDIA 大 hidden tile（否决）

状态：两项均未晋升；源码恢复 S0，未生成 ZIP，未提交平台

验证时间：2026-08-24 08:00–08:04 CST

在不改 kernel 数学、grid 维度、FP32 累加、stride、TOP_K constexpr 和 1 stage
的前提下，对 `num_tokens>=32 && hidden_dim>=2048` 做了两个相邻 screening：

- E2：`BLOCK_SIZE=2048, num_warps=8`，临时源码 SHA-256
  `427751040bde8c2cea9a8255c48186f3aee5bc8c7dc3dc3159cfd1bae25f3f54`。
- E3：`BLOCK_SIZE=1024, num_warps=8`，临时源码 SHA-256
  `19072de7644742aca488352e26d162316a2e64ce271b668c4c5c2074a12defbe`。

两项都通过 py_compile、Black 79、isort、flake8、4/4 unittest 和 18 个
dtype/shape reference 检查。screening 目录
`gpu:/tmp/flagos-moe-sum-reduce-e2.87tnS3`，mode 0700；环境仍为 RTX 5070 Ti、
PyTorch 2.13.0+cu130、Triton 3.7.1、CUDA 13.0。

五组轮换、wrapper-inclusive `warmup=25, rep=100` 的 screening 中：

| 候选 | 12 个 affected 几何平均 | dtype 几何平均 FP16/BF16/FP32 | 最差 affected | 6 个 controls 几何平均 |
| --- | ---: | --- | ---: | ---: |
| E2 2048/8 | `0.9886x` | `1.0184/0.9328/1.0171x` | `0.6833x` | `0.9961x` |
| E3 1024/8 | `1.0077x` | `1.0096/1.0075/1.0061x` | `0.9658x` | `0.9927x` |

E2 的明确失败点是 BF16 `[32,16,4096]`，从 S0 `0.01110 ms` 退化到
`0.01625 ms`；E3 虽消除该大回退，但总体远低于 `1.05x` 晋级线。差距足够大，
不再为失败候选追加 release/资源门禁或事后 dtype/shape 过拟合。

E2 gates/A-B、E3 gates/A-B、provenance 和 harness 的 SHA-256 依次为
`4d18192cf69f6b34ade12a9ff8fc69d25af53fbae2e2f33283ebeeef72289488`、
`2594280110e2fd1aa47347a922b6808784e29301a85c18383c8fd4ef9ddd17b9`、
`f1f677f714e597b1c7a91c78b28935c05b0288815484478c531f5db08d54c576`、
`901e1aaf3fc5ab275878a0a2c87dc933ff14f0b9d78f87e6daec222074bcf820`、
`91f8e504929ba7a52f2f7daad246d7f5fba3f8ea3b445fe6671aa1233b2f772e`、
`47cdb3c4e1cbb1c3cb2cce46893937997c2f2115a02793a9a50407a6a0ee4b35`。
当前源码已恢复 S0 SHA-256
`52a2fc979784f2bd25e7e17b9822c23b4f438efdf062c70bfb09aba9ba732335`；
不可变候选仍是 `s0-3fac516`。

## S0c：canonical 首投包与平台结果

状态：已提交；`invalid_correctness`，6/8，平均值与排名 N/A

生成与提交时间：2026-08-24 22:46 CST。不改 kernel，复用 source commit
`3fac516a8d64c88b183801668a7857d969a05e37` 与既有 4/4 回归证据。dry-run manifest
与账本完全一致后，canonical ZIP 为
`artifacts/competition/moe_sum_reduce/s0c-3fac516/moe_sum_reduce.zip`，2932
bytes，SHA-256
`d000d65f081f1bebcd9e1f1193e9d6d2fa95886658c412df9d58aa4f47414e2d`（即
2026-08-24 03:12 复核时仅在内存生成的规范字节）；唯一成员
`moe_sum_reduce.py` 2800 bytes，成员 SHA-256 与 source commit 一致，
`unzip -t` 通过。旧 `s0-3fac516` legacy ZIP 保持原字节不覆盖。

实时 preflight 通过：race `782kzq4m`、season 2、账号 `15600308080`、团队
`SoulCoder`、batch 2、Task 21、tid `s2t1op021`、task `competing`、提交窗口
2026-08-20 20:00 至 2026-08-27 19:59:59、最小间隔满足、额度 `21/30`。2026-08-24
22:46:50 CST 按当次 nonce 提交一次，submission ID `4274`、daily seq `10`，额度
变为 `20/30`。内置远端验签为 `verified`：平台对象 2932 bytes，SHA-256 与本地
canonical ZIP 完全一致；`file_url_sha256` 为
`ede4247c1b3737377466b4ee5ae85311672f5147df8532e094b16da3f8637d60`。

22:50:18 CST 终态 `completed` / `invalid_correctness`，6/8 通过：

| 芯片 | 结果 | speedup | 选中文件 |
| --- | --- | ---: | --- |
| 天数 | 通过 | 4.6870x | `moe_sum_reduce.py` |
| 沐曦 | 通过 | 3.5474x | `moe_sum_reduce.py` |
| 燧原 | 通过 | 2.3126x | `moe_sum_reduce.py` |
| 海光 | 通过 | 6.6266x | `moe_sum_reduce.py` |
| 昆仑芯 | 失败 | N/A | `moe_sum_reduce.py` |
| 华为 | 失败 | N/A | `moe_sum_reduce.py` |
| 国际通用 A | 通过 | 3.9612x | `moe_sum_reduce.py` |
| 国际通用 B | 通过 | 2.2598x | `moe_sum_reduce.py` |

两芯失败均在 correctness case 7（隐藏 shape `num_tokens=4096`、
`hidden_dim=7168`，2D grid `(4096, 28)`，逻辑 program 总数 `114688`）：

- 华为：`Invalid_Argument(EE1003) KernelLaunch failed because value 114688
  for parameter coreDim is invalid. Expected <= 65535`。与 Task 08 S0c、
  Task 20 E2 同一 Ascend 2D grid 展平越界模式；输出比较错误只是异步 launch
  失败的外层表现，算子数学未被证伪。
- 昆仑芯：XPU 编译失败于 `make_ttxir` 的 `pm.run`（arch=3、cluster_num=12、
  core_num=64、buffer_size_limit=512），包装为
  `OutOfResources: uni_sram PassManager::run failed`。这是 2D grid
  `(4096, 28)` 的编译期失败，不是运行期 grid 越界；Task 08 平台记录证明昆仑
  可正常执行 304128/2433024 规模的一维 grid pointwise kernel。

其余六芯全部通过且远高于 0.1x 门槛。下一轮保持 generic 字节不变，只加两个
自包含 vendor：昆仑改为与 generic 数学一致的一维展平 grid；华为沿用
Task 08/20 平台已验证的 capped grid-stride 模式。

## S1：昆仑一维展平 grid 与华为 capped grid-stride vendor

状态：release 门禁通过，候选就绪，等待 preflight 与提交

### 假设与单变量

generic 源码逐字节保持 S0 不变，只新增两个自包含 vendor：

- 昆仑 `_kunlunxin/ops/moe_sum_reduce.py`：kernel 数学、BLOCK 256、4 warps、
  1 stage、stride 与 FP32 累加均不变；把 2D grid
  `(num_tokens, hidden_blocks)` 展平为一维
  `(num_tokens * hidden_blocks,)`，program 内以
  `program // hidden_blocks`、`program % hidden_blocks` 还原 token 与
  hidden block。依据：S0c 失败是 XPU 编译期 `make_ttxir` 对 2D grid
  `(4096, 28)` 的 pass 失败，而 Task 08 平台记录中昆仑以 304128/2433024
  规模的一维 grid 正确执行 pointwise kernel。
- 华为 `_ascend/ops/moe_sum_reduce.py`：同样的一维逻辑 program 分解，外加
  物理网格 `min(total_programs, 4096)` 与 `tl.num_programs(0)` 步长的
  grid-stride 循环。依据：Task 20 E3 平台验证 cap 4096 在 Ascend 通过并达
  1.8838x，且其代理 cap 扫描中 4096 优于 48/256/1024/16384/32768/65535；
  S0c 华为失败仅为展平 `coreDim=114688 > 65535` 的启动越界。

失败规模回归 `test_vendors_cover_platform_failure_scale` 使用
`(num_tokens, top_k, hidden_dim) = (4096, 8, 7168)`，断言逻辑 program 总数
114688，并在三 dtype 下对 generic、昆仑、华为三个模块与 reference 精确比对，
同时验证输入不变性。其余四个既有回归方法不变。

### Screening 与 Release

screening 目录 `gpu:/tmp/flagos-moe-sum-reduce-s1.ge5pXH`（mode 0700）。
第一次运行 PID/PGID `103080`（22:55:42，wall 900s，脚本 SHA-256
`f675c56cab6910fd9b02d826b31692cc962075514ceda62654875353ca9aa2e9`）因测试
文件一处 Black 折行失败停止，无源码数学变化；本地修正折行（测试 SHA-256 由
`ea1641c64c6da6ca10f0266485ec2e2459471e76d07be737d5fd41d2b95ae7da` 变为
`f1ca0bcadb1393f2da5188a5597839abe71bd85eff996a230113d5872ca76dc8`）后以
PID/PGID `103179`（22:57:20，wall 900s，同脚本）重跑通过：
`screening.log` SHA-256
`69647fe844abf963c5bee3d22f3928a87a571af84b028397bb5cb9aae79f28c7`。远端
环境 RTX 5070 Ti 16 GB、driver 610.57.04、Python 3.12.13、PyTorch
2.13.0+cu130、Triton 3.7.1、CUDA 13.0。py_compile、Black 79、isort、flake8
与 5/5 unittest（0.748s）通过，前后 SHA-256 复核一致。同脚本附带的失败规模
代理计时（informative）三 dtype 下三模块全部正确：generic/昆仑/华为 p50 分别为
FP16 `0.694812/0.662263/0.685105 ms`、BF16 `0.705952/0.665458/0.688049 ms`、
FP32 `1.329962/1.337104/1.340918 ms`；一维昆仑不慢于 2D generic，华为
grid-stride 与 generic 持平，无代理回退。

source/verification commit 均为
`849527f184df53fe21150fd635e044c614dc9651`，其 Git blob SHA-256 与 screening
字节逐项一致（generic
`52a2fc979784f2bd25e7e17b9822c23b4f438efdf062c70bfb09aba9ba732335`、昆仑
`8b425964dc50d523506ec4b5379b2dd33490f0b0ca611787da52ff5e230ac079`、华为
`516979df657248bc5d6bd14dde9af06e21a09c6ed70e404998851af87aca229f`、测试
`f1ca0bcadb1393f2da5188a5597839abe71bd85eff996a230113d5872ca76dc8`）。
release 目录 `gpu:/tmp/flagos-moe-sum-reduce-s1-release.5BOySW`（mode 0700）
从该 commit 的 Git 对象建立，PID/PGID `103426`（23:00:21，wall 600s，脚本
SHA-256
`708f6ea5c2db1c4d753f88289674b87c651db4a14b160d953521a3511d0b915d`）；
py_compile、Black 79、isort、flake8 与 5/5 unittest（0.575s）全部通过并输出
`RELEASE_OK`，`release.log` SHA-256
`d7969825cd1fdc946fa987f170598db09da0720ea9ded31b1288bd94729cd8b2`。

canonical ZIP 为
`artifacts/competition/moe_sum_reduce/s1-849527f/moe_sum_reduce.zip`，9395
bytes，SHA-256
`a8416396f76c624ebbc06033b1daba88858fd97b65c6c74a5f9c83f1c30f25c9`，与
release 前 dry-run manifest 的 commit、成员集合、成员 SHA-256 和 canonical
ZIP SHA-256 完全相同。成员为 `moe_sum_reduce.py` 2800 bytes、
`moe_sum_reduce_ascend.py` 3253 bytes、`moe_sum_reduce_kunlunxin.py` 2956
bytes；`unzip -t` 通过。平台晋级门禁：8/8 通过、昆仑与华为各自选中对应
vendor 文件且高于 0.1x；其余六芯继续使用未变 generic。NVIDIA 只能代理编译与
数值，不能证明 XPU/Ascend runtime 实际行为。

### S1 平台结果：7/8

2026-08-24 23:02:23 CST 提交一次，submission ID `4283`、当日序号 `11`，额度由
`20/30` 变为 `19/30`。远端对象 9395 bytes，SHA-256 与本地 canonical ZIP 一致
（`verified`）；`file_url_sha256` 为
`f11dcbd706d6d3a7a75d0040af0a08be1f4464fac9f396a316fb3e9db05f0739`。23:03:08
CST 终态 `completed` / `invalid_correctness`，7/8：

| 芯片 | 结果 | speedup | 选中文件 |
| --- | --- | ---: | --- |
| 天数 | 通过 | 4.7174x | `moe_sum_reduce.py` |
| 沐曦 | 通过 | 3.5052x | `moe_sum_reduce.py` |
| 燧原 | 通过 | 0.2074x | `moe_sum_reduce.py` |
| 海光 | 通过 | 6.4758x | `moe_sum_reduce.py` |
| 昆仑芯 | 失败 | N/A | `moe_sum_reduce_kunlunxin.py` |
| 华为 | 通过 | 0.6496x | `moe_sum_reduce_ascend.py` |
| 国际通用 A | 通过 | 3.8572x | `moe_sum_reduce.py` |
| 国际通用 B | 通过 | 2.2328x | `moe_sum_reduce.py` |

结论分三部分：

- 华为 vendor 完成恢复：`_ascend` 文件被选中，case 7 的 `coreDim=114688`
  启动越界消除，八 case 全部通过且 `0.6496x` 远高于 0.1x 门槛；capped
  grid-stride 模式在 Ascend 第三次平台验证成功（Task 08/20/21）。
- 昆仑 vendor 失败且范围扩大：vendor 被选中后 correctness case 0–7 全部以
  同一 `OutOfResources: uni_sram PassManager::run failed` 编译失败收场；而
  S0c 中同数学的 2D grid generic 在昆仑通过 case 0–6。证明失败与 grid 规模
  无关，而是一维 `program // hidden_blocks` 分解这一 kernel 结构在该 XPU
  编译器（arch=3）上无法通过 `make_ttxir`。Task 08 同款 div/mod 能过，但彼
  kernel 无 int64 stride cast 与 TOP_K 静态累加循环，不能互相背书。
- 燧原本次 `0.2074x`，而 S0c 同字节 generic 为 `2.3126x`；其余六芯数值接
  近。generic 字节未变，判定为平台侧 GCU 当次测量波动，不据此改燧原路径，
  待下一轮平台数据复核。

下一轮 S2 只改昆仑 vendor：保留平台已证明可编译的 2D grid 结构，把 grid.x
（num_tokens 维）封顶并让 program 内按 `tl.num_programs(0)` 跨步遍历 token；
generic、华为 vendor 与全部既有回归保持不变。

## S2：昆仑轴交换 vendor

状态：release 门禁通过，候选就绪，等待 preflight 与提交

S1 的证据表明昆仑失败与 grid 规模无关：一维 div/mod 结构在该 XPU 编译器上
对全部 case 都无法编译，而同数学的 2D generic 只在 `(4096, 28)` 失败。固定
FlagTree `c1ea8285` 的 XPU compiler.py 没有 Python 层 grid 校验或钳制，
`uni_sram` 报错来自 C++ ttxir pass（core tiling / loop grid / alloca），由
kernel IR 与 grid 形状共同触发；其对 grid 的唯一特判是 `(12, 1, 1)` 开关
interleave pass。FlagGems `ed2508b` 的 kunlunxin heuristics 亦无 grid 上限，
pointwise 通用策略为 BLOCK 1024 / 8 warps。

S2 因此选择与已编译 IR 距离最小的单变量：昆仑 vendor 的 kernel 体与 generic
逐字相同（2D program_id、无 div/mod、无循环、同样的 int64 cast 与 TOP_K 静态
累加），仅交换两个 program 轴——`hidden_block = tl.program_id(0)`、
`token_offset = tl.program_id(1)`，wrapper 的 grid 由
`(num_tokens, hidden_blocks)` 改为 `(hidden_blocks, num_tokens)`。case 7 的
grid 由 `(4096, 28)` 变为 `(28, 4096)`：若 XPU 的 core tiling 只对大 grid.x
敏感，该交换可恢复编译；若对大 grid.y 同样失败，则下一轮改为静态 2-token
展开把 token 维减半。generic、华为 vendor 与测试文件字节均不变。

screening 目录 `gpu:/tmp/flagos-moe-sum-reduce-s2.32Dk8V`（mode 0700），
PID/PGID `103640`（23:08:19，wall 900s，脚本与 S1 相同，SHA-256
`f675c56cab6910fd9b02d826b31692cc962075514ceda62654875353ca9aa2e9`）。
py_compile、Black 79、isort、flake8 与 5/5 unittest（0.658s）通过，
`screening.log` SHA-256
`7c0e46bf6213b66eac44bff18ac4ad15977df997b4c6f075aa3ed36df1cb14b1`。失败规模
代理计时三 dtype 下 generic/昆仑/华为 p50 为 FP16
`0.695689/0.664483/0.685894 ms`、BF16 `0.706038/0.663793/0.688375 ms`、
FP32 `1.331204/1.334055/1.337605 ms`，无回退。

source/verification commit 均为
`67fb4fcd388c1c139dc117dd3500270c5a9cd2b8`；昆仑 vendor Git blob SHA-256
`a380ce3135df6c84a3c1640e7386275c26d7cb0b80396642ecd920139b16c92a`，其余三
文件与 S1 相同。release 目录
`gpu:/tmp/flagos-moe-sum-reduce-s2-release.w34D3a`（mode 0700）从该 commit
的 Git 对象建立，PID/PGID `103874`（23:11:09，wall 600s）；复用 S1 release
脚本（SHA-256
`708f6ea5c2db1c4d753f88289674b87c651db4a14b160d953521a3511d0b915d`，日志
头部 commit 标签仍为 849527f，属脚本模板陈旧；权威绑定以日志内文件
SHA-256 表对 67fb4fc blobs 逐项一致为准）。静态门禁与 5/5 unittest 通过并
输出 `RELEASE_OK`，`release.log` SHA-256
`113f7e3e3bb375432dce656e30cddf21632c0282cc8d9e36910c5798f0c2760f`。

canonical ZIP 为
`artifacts/competition/moe_sum_reduce/s2-67fb4fc/moe_sum_reduce.zip`，9305
bytes，SHA-256
`b78e408d9e2ee430430889da25f1b12f33d9e72a650a3adf17eb4fdb9deac72d`；成员为
`moe_sum_reduce.py` 2800 bytes、`moe_sum_reduce_ascend.py` 3253 bytes、
`moe_sum_reduce_kunlunxin.py` 2866 bytes，`unzip -t` 通过。平台晋级门禁：
8/8 通过、昆仑选中轴交换 vendor 且华为继续选中 ascend vendor，两芯均高于
0.1x。NVIDIA 不能证明 XPU 编译行为。

### S2 平台结果：7/8，昆仑假设二否决

2026-08-24 23:13:00 CST 提交，submission ID `4288`、当日序号 `12`，额度由
`19/30` 变为 `18/30`；远端验签 `verified`（9305 bytes，SHA-256 一致），
`file_url_sha256` 为
`e2280198b218da3753194275906063a18a9921b52ab662acccaa8d2b345660ec`。23:13:47
CST 终态 `completed` / `invalid_correctness`，7/8。华为 vendor 继续被选中并
通过（`0.6610x`）；天数 4.7238x、沐曦 3.5172x、海光 6.3594x、国际 A
3.7584x、国际 B 2.2194x；燧原第二次出现 `0.2072x`（同字节 generic 在 S0c 为
2.3126x，已连续两次低读数，与 ZIP 从单成员变为三成员在时间上相关，原因
未明，作为开放观察保留）。

昆仑 vendor 被选中后 case 0–7 依旧全部以同一 `uni_sram PassManager::run
failed` 编译失败。S2 的 vendor kernel 与 generic 逐字相同、仅交换
`tl.program_id` 两轴，而同数学 generic 在 S0c 通过 case 0–6；结论：该 XPU
编译器对 program 轴语义敏感，token（外层标量维）必须位于 pid(0)、hidden
block（arange 向量维）必须位于 pid(1)，交换即无法编译，与 grid 规模无关。

至此昆仑假设矩阵为：2D 原始布局通过 total≤65535 的全部 case、仅 total=
114688>65535 的 case 7 失败；Task 24 generic 曾在昆仑以 256512 规模纯一维
grid 通过（排除 grid.x 上限）；Task 19 vendor 以 total≤2048×1 通过。最强
假设：XPU 将 2D grid 展平为总 program 数并在编译期以 65535 为上限（与
Ascend `coreDim` 同类约束、不同报错形态）。据此 S3 把昆仑 vendor 的 kernel
恢复为与 generic 逐字节相同（token=pid(0)），仅把 wrapper 的 BLOCK_SIZE 从
256 提到 1024：case 7 的 grid 由 `(4096, 28)` 变为 `(4096, 7)`，total 28672
≤ 65535；该配置同时有固定 FlagGems `ed2508b` kunlunxin pointwise 策略
（BLOCK 1024 / 8 warps）背书。残余风险：若隐藏 shape 出现
`num_tokens × cdiv(hidden,1024) > 65535`（如 num_tokens ≥ 16384 且 hidden
7168）仍会失败。S3 为本轮昆仑最后一次单变量尝试；若仍失败则保留 S1/S2 证据
并转其他算子。

## S3：昆仑 BLOCK 1024 vendor

状态：release 门禁通过，候选就绪，等待 preflight 与提交

S3 把昆仑 vendor 的 kernel 恢复为与 generic 逐字节相同（token=pid(0)、
hidden block=pid(1)、无循环、无 div/mod），仅将 wrapper 的 `block_size` 从
256 提升到 1024；generic、华为 vendor 不变。case 7 的 grid 由 `(4096, 28)`
变为 `(4096, 7)`，总 program 数 28672 ≤ 65535。新增回归
`test_kunlun_vendor_block1024_boundaries` 覆盖三 dtype × hidden
`1023/1024/1025/2049` 的 1024 边界与尾块。固定依据：FlagGems `ed2508b`
kunlunxin pointwise 通用策略即为 BLOCK 1024；XPU 编译证据链见 S2 结论。

screening 目录 `gpu:/tmp/flagos-moe-sum-reduce-s3.wjvB8H`（mode 0700），
PID/PGID `104064`（23:15:52，wall 900s，脚本同 S1，SHA-256
`f675c56cab6910fd9b02d826b31692cc962075514ceda62654875353ca9aa2e9`）；静态
门禁与 6/6 unittest（0.837s）通过，`screening.log` SHA-256
`dc7418a4d0971b610f56f45b03839ac62ff25113681b01620778a648a4997edd`。失败规模
代理计时 generic/昆仑/华为 p50 为 FP16
`0.695485/0.676984/0.685337 ms`、BF16 `0.707215/0.684037/0.687108 ms`、
FP32 `1.333114/1.327501/1.334707 ms`，无回退。

source/verification commit 均为
`1ca7dd280e4f795e81fc7954b8c2b6c3fdc17f4b`；昆仑 vendor Git blob SHA-256
`68b0abe07e3cf4f2b9cb86063e9ad1e18edd83d90341410cd281ec595c83406d`，测试
`118e1747ca47b7283a15cf36b075c4ae9ffccf81cd185d1504c3c0f3b3e2fa99`。release
目录 `gpu:/tmp/flagos-moe-sum-reduce-s3-release.RYZbso`（mode 0700）从该
commit 的 Git 对象建立，PID/PGID `104297`（23:18:57，wall 600s，脚本复用
S1 模板，头部 commit 标签陈旧，权威绑定以日志内文件哈希对 1ca7dd2 blobs
逐项一致为准）；静态门禁与 6/6 unittest 通过并输出 `RELEASE_OK`，
`release.log` SHA-256
`6102e562295e0767e42f7c1fca33d8c9f363df97ce09a9d3193f14638f35cb49`。

canonical ZIP 为
`artifacts/competition/moe_sum_reduce/s3-1ca7dd2/moe_sum_reduce.zip`，9240
bytes，SHA-256
`159911639601002f9be5e083309d9a5cac1d1d32617e1fe31207486cc267b2f8`；成员
`moe_sum_reduce.py`、`moe_sum_reduce_ascend.py`、
`moe_sum_reduce_kunlunxin.py`，`unzip -t` 通过。平台晋级门禁：8/8 通过、
昆仑选中 BLOCK 1024 vendor 且华为继续选中 ascend vendor。

### S3 平台结果：8/8，有效，团队当前最佳

2026-08-24 23:20:43 CST 提交，submission ID `4291`、当日序号 `13`，额度由
`18/30` 变为 `17/30`；远端验签 `verified`（9240 bytes，SHA-256 一致），
`file_url_sha256` 为
`6202dd08e2cc83dbb2a93ebfb583e5f4756e974d05bb7c6fd5d809dae7e7ac98`。
23:21:15 CST 终态 `completed` / `valid`，8/8 通过，平均 `2.7096x`，平台标记
team best：

| 芯片 | 结果 | speedup | 选中文件 |
| --- | --- | ---: | --- |
| 天数 | 通过 | 4.6448x | `moe_sum_reduce.py` |
| 沐曦 | 通过 | 3.5248x | `moe_sum_reduce.py` |
| 燧原 | 通过 | 0.2060x | `moe_sum_reduce.py` |
| 海光 | 通过 | 6.4578x | `moe_sum_reduce.py` |
| 昆仑芯 | 通过 | 0.1754x | `moe_sum_reduce_kunlunxin.py` |
| 华为 | 通过 | 0.5982x | `moe_sum_reduce_ascend.py` |
| 国际通用 A | 通过 | 3.8554x | `moe_sum_reduce.py` |
| 国际通用 B | 通过 | 2.2144x | `moe_sum_reduce.py` |

结论：昆仑 BLOCK 1024 vendor 全部 case 编译并运行通过，"XPU 将 2D grid
展平为总 program 数、编译期上限 65535"的假设被平台证实（S0c/S1/S2/S3 四组
反例与正例构成完整证据链）。华为 ascend vendor 第三次平台验证成功。Task 21
八芯全部通过 0.1x 门槛，闭环完成；S3 为当前团队最佳。

遗留观察与下一步：燧原已连续三次出现 `~0.207x`（同字节 generic 首投为
2.3126x），与 ZIP 从单成员变为三成员时间上相关，属平台侧持续状态而非单次
噪声；若后续继续优化 Task 21，优先做燧原性能 vendor（如 Task 08 式
BLOCK 4096/grid 12）并同时在账本跟踪三成员 ZIP 与首投的差异。昆仑 0.1754x
与华为 0.5982x 亦有较大提升空间，但本轮停止迭代，额度转向队列中未首投的
12 个算子。

## E4：燧原 BLOCK 4096 负实验

状态：严格 screening 拒绝；源码和测试已恢复 S3，未 commit 候选、未生成
ZIP、未提交平台

验证时间：2026-08-25 11:19–11:29 CST。候选只新增自包含 Enflame vendor，
该文件与 generic 逐字仅 `block_size = 256` 改为 `4096`；2D grid、4 warps、
1 stage、TOP_K constexpr、FP32 累加、stride 与数学均不变。generic、华为和
昆仑源码保持原 SHA-256。最大已知平台 shape 的 Enflame grid 从
`(4096, 28)` 降为 `(4096, 2)`；未迁移 Task 08/24 的 cap-12 grid-stride，
避免混入第二变量。

### Screening 身份与正确性

| 项目 | 值 |
| --- | --- |
| base commit | `514bcc16a009342940fca184e9aca0e369badaa8` |
| 临时 Enflame 源码 SHA-256 | `d42e7c32ec5e077367014a08374c080ba15f7ec3e48fb713419a5dee34d0f106` |
| 临时测试 SHA-256 | `a5420ea6d979ef358a1528240c9efa6d25210a37c29ac16b0bb7fc7915573ff8` |
| 远端目录 | `gpu:/tmp/flagos-moe-sum-reduce-e4.JP7eZH`，mode 0700 |
| gates 脚本 / hash manifest | `f98af586632e02dcb60e87a303bb44dad7c346746eb3161820fef3d4dd5782dd` / `bd753ca8c83c4c1fc13cde2e8633154d28710d7779b2d1bb43e1524e0cd070bd` |
| unittest 日志 SHA-256 | `2b19fcb1722550ac0ecc0f74b64639eac392048515aa375798e511aa3c814243` |

首次单日志 gates 进程 `119133` 的 2390-byte tmpfs 日志出现不可读异常且未保留
退出码，不作为证据。拆分阶段日志后的重跑 `119460` 依次通过 py_compile、
Black 79、isort、flake8、前后哈希复验和 7/7 unittest。永久回归临时扩展覆盖
Enflame 三 dtype 的 hidden `4095/4096/4097`，并把最大已知平台 shape
`(4096,8,7168)` 加入 Enflame 直达路径；screening harness 另验证非连续 stride、
空维和 zero-top-k。候选全部正确，输入不变。

### 六轮 AB/BA 与拒绝

GPU 运行前为 0% utilization、34 MiB/16303 MiB，且无其他 compute process。
脚本 `screening-ab.py` SHA-256 为
`9f421435ab849397ced37f1b01a8e4d639f4bfb03a0ce8a8d0b88c1f6d169820`；
完整原始 JSON SHA-256 为
`7f314392f1703d643b1f98ec6225a0295c1c01352a4dfc53a6af58b9f44c32c5`，
摘要日志 SHA-256 为
`4e01a73b6660314d8c29d7212acffa7872d0603df9e9b15571db54517283536f`。
环境仍为 RTX 5070 Ti、Python 3.12.13、PyTorch 2.13.0+cu130、Triton
3.7.1、CUDA 13.0。

三 dtype 覆盖 `(1,8,4096)`、`(32,8,4096)`、`(128,8,7168)`、
`(4096,8,7168)`、`(32,1,4096)`、`(32,16,4096)`，另以
`(32,3,513)` 和 `(32,8,4097)` 作回退 guard；每点严格六轮
`AB/BA/AB/BA/AB/BA`、`warmup=25, rep=100`：

| 指标 | 结果 | 门槛 |
| --- | ---: | ---: |
| affected 几何平均 | `0.555139x` | `>=1.05x` |
| FP16 / BF16 / FP32 affected | `0.588252/0.588966/0.493800x` | 各 `>=1.00x` |
| 最差 affected 点 | `0.161780x` | `>=0.95x` |
| 最差 guard 点 | `0.567265x` | `>=0.97x` |
| 最差单轮 | `0.152068x` | `>=0.90x` |

TOP_K `1/8/16` × 三 dtype 的 NVIDIA 编译资源门禁通过：不超过 128
registers/8 KiB shared，且未发现 spill、global scratch 或 PTX local load/store。
因此拒绝原因是稳定而全面的代理性能退化，不是正确性或资源失败。Task 08/24
的 pointwise BLOCK 4096 收益不能外推到保活 4096-wide FP32 accumulator 的
reduction；不做事后 shape 缩窗，不消耗平台额度。S3 继续作为 Task 21 唯一
有效候选。

## E5：华为 BLOCK 512

状态：平台 8/8、`valid`、`2.76185x`，团队当前最佳

source/verification commit：`f2e9fc90837dd5169186bdecbaba7869959c44a4`。
E5 只把 Ascend vendor 的 `block_size` 从 256 改为 512；physical worker cap
4096、grid-stride、4 warps、1 stage、TOP_K constexpr、FP32 累加和 stride
均不变。generic 与昆仑 vendor 继续冻结为 S3 字节。历史 E1 的 512 是 generic
NVIDIA 负实验，E4 是 Enflame4096；两者均未验证或否决 Ascend 的单芯 512。

现有 `test_block_boundaries_all_dtypes` 复用于 generic 与 Ascend，覆盖三 dtype ×
hidden `255/256/257/511/512/513`，并保留非连续 stride、空维/zero-top-k、昆仑
1024 边界和平台最大 `(4096,8,7168)` 三 dtype 回归。screening 和独立
Git-object release 均通过 py_compile、Black 79、isort、flake8 与 6/6 unittest。

| 项目 | 值 |
| --- | --- |
| generic SHA-256 | `52a2fc979784f2bd25e7e17b9822c23b4f438efdf062c70bfb09aba9ba732335` |
| Ascend vendor SHA-256 | `f740604cd4a0506a3e41776f3f9a001edffafef45f04aff78d1ee8a208f2132b` |
| Kunlun vendor SHA-256 | `68b0abe07e3cf4f2b9cb86063e9ad1e18edd83d90341410cd281ec595c83406d` |
| 测试 SHA-256 | `bd17ff77e569304df14f2cb563b5b64b2028db51b87d8b1689e46ceb000e3f6f` |
| screening | `gpu:/tmp/flagos-moe-sum-reduce-e5-screen.uYr6hp`；PID/PGID `159234`；6/6、0.929s；脚本 SHA-256 `1124518e8e1c7b24bbb319950e09c0142921fdc0e2bddb6935838c4b6a81a462`；日志 SHA-256 `886173429837bff17ae4c9aab33a6c1f178ad6370056a78890f75b631ea4d94c` |
| 交替 A/B | 同目录；PID/PGID `159332`；脚本 SHA-256 `8ffcbb21104086ece1c6f927dbe388046f1f14d63bbb3a0ab3387090fc345db7`；日志 SHA-256 `194f781de9b58ca7ef6f766872024c4b75bfadf19103d2f5ba6a9ed7c3ea8077` |
| release | `gpu:/tmp/flagos-moe-sum-reduce-e5-release.q3YOwb`；PID/PGID `159467`；6/6、0.595s、`RELEASE_OK`；脚本 SHA-256 `dfe62f5ee8aaaac68acb4215528a38ed09231959accbf5a8d2cdfcca8af8075c`；日志 SHA-256 `2db8fc21d3516c287968707924ee354b10b8f08890a75ab8a575de91ae5896b7` |
| release Git archive SHA-256 | `92ddfc2a9cf1de8015e9dbc58b529b7a74a437218d200d9b15d97eed54d78058` |
| ZIP | `artifacts/competition/moe_sum_reduce/e5-f2e9fc9/moe_sum_reduce.zip`，9240 bytes |
| ZIP SHA-256 | `9544a54ff611e69bd5c42fc3aba2d440d4bca431bd0b665a940a0e24f3d38035` |

五组 AB/BA、三 dtype、control/boundary/mid/wide/max 共 15 点均与 BLOCK256
baseline 精确一致。非 control 中位 speedup 几何平均 `1.000685x`，最差点
`0.962818x`；平台最大 shape 的 FP16/BF16/FP32 分别为
`1.019517/1.025810/0.993040x`。candidate 为 22–56 registers、0 spill、0
shared、0 global scratch。NVIDIA 性能只作资源/明显回退门禁，不声称是 Ascend
收益测量。

canonical ZIP 仅含 generic、Ascend、Kunlun 三个白名单成员，成员哈希与 commit
一致；`dry-run`、`verified-existing` 与 `unzip -t` 均通过。E5 只允许一次平台
提交，晋级门为 8/8 valid、华为高于 S3 的 `0.5982x`、平均高于 `2.7096x`；
任一不满足即保留 S3 team best 并停止 Ascend 大 tile 假设。

2026-08-26 23:56:55 CST 在账号 `15600308080`、团队 `SoulCoder` 下执行 E5
唯一一次提交，submission ID `5199`、daily seq `24`，额度由 `7/30` 变为
`6/30`。`file_url_sha256` 为
`c5804f2f80eecd120d276958ac74737995a2eab7a8e80f53dc130956ab7f57cb`；内置远端验签
因未配置可信 hostname 为 `unavailable`，随后从返回的已核实对象存储地址无认证
下载，得到 9240 bytes，SHA-256 与 canonical 值完全一致。平台已选中预期的
generic、Ascend、Kunlun 三条路径；该候选禁止重传。

2026-08-27 00:05:45 CST 最后一颗燧原完成；00:12:05 CST 只读复核为
`completed` / `valid`、8/8、平均 `2.76185x`，平台标记 team best。E5 相对
S3 平均增加 `0.05225x`（+1.93%），通过预注册晋级门；新日额度为 `30/30`：

| 芯片 | E5 speedup | 相对 S3 | 选中文件 |
| --- | ---: | ---: | --- |
| 天数 | 4.7638x | +2.56% | `moe_sum_reduce.py` |
| 沐曦 | 3.5040x | -0.59% | `moe_sum_reduce.py` |
| 燧原 | 0.4210x | +104.37% | `moe_sum_reduce.py` |
| 海光 | 6.4886x | +0.48% | `moe_sum_reduce.py` |
| 昆仑芯 | 0.1760x | +0.34% | `moe_sum_reduce_kunlunxin.py` |
| 华为 | 0.8104x | +35.47% | `moe_sum_reduce_ascend.py` |
| 国际通用 A | 3.7662x | -2.31% | `moe_sum_reduce.py` |
| 国际通用 B | 2.1648x | -2.24% | `moe_sum_reduce.py` |

结论：Ascend BLOCK 512 假设被平台证实，保留 E5。燧原的同字节 generic
读数同时恢复一倍，属于平台波动，不能归因于 Ascend-only 改动。下一候选必须
继续冻结 E5 的 generic、Ascend 和昆仑字节，只允许新增单芯 vendor。

## E6：MetaX 官方 Qwen launch policy

状态：平台 8/8、`valid`、`2.795625x`，团队当前最佳

E6 只新增 MetaX vendor。固定官方
`FlagGems-vllm@43624463db77618b6d0e3f47fac990cea8c51a30` 的 MetaX
`moe_sum.py` 对 contiguous、top-k 8、hidden 2048/4096 使用 BLOCK1024、
8 warps，并保留 backend 默认 stages；本题沿用同一 launch policy，同时保留
题面要求的 scalar scaling、真实 stride 和 FP32 累加。其他 shape、top-k 或
非连续输入在同一文件使用 E5 generic 的 BLOCK256、4 warps、1 stage。

独立审查发现首版错误地把 fast path 也锁为 stage1；在任何 commit、release 或
平台提交前已改为官方的 `launch_kwargs` 形式：fast path 不传 `num_stages`，
fallback 才传 1。generic、Ascend、Kunlun 三个已有成员与 E5 逐字节一致。

| 项目 | 值 |
| --- | --- |
| source / verification commit | `9d3c82a0433295c30f7fd2dcfe3896ddf57483cf` |
| generic / Ascend SHA-256 | `52a2fc979784f2bd25e7e17b9822c23b4f438efdf062c70bfb09aba9ba732335` / `f740604cd4a0506a3e41776f3f9a001edffafef45f04aff78d1ee8a208f2132b` |
| Kunlun / MetaX SHA-256 | `68b0abe07e3cf4f2b9cb86063e9ad1e18edd83d90341410cd281ec595c83406d` / `20db4f49aada2976723ab9dabd7013545a30a0998ee8fb21ad655375bd2cb794` |
| 测试 SHA-256 | `0c31a7a5f6ac5362f739d7b6b58f501f5fb2a35207122b75ba22a9d7c8bc7136` |
| screening | `gpu:/tmp/flagos-moe-sum-reduce-e6-metax-screen.NnUURp`；PID/PGID `160329`；8/8、0.756s、`SCREENING_OK`；脚本/日志 SHA-256 `e4709693d4f51ead44bc6a63f3b50c281a54d021eb2a9db45a599752118042d0` / `6e2b96372efd59875e649af9131dfab53384f826fdc6844a743023edb013be63` |
| 交替 A/B | 同目录；PID/PGID `160428`；脚本/日志 SHA-256 `2bc7226c7958cf8d8c943f3b503584262828fe8076726d154c0dc0bcb581c81d` / `d4c03e5837f47c6a50e14ae89e03b456aa48a6d0372c3bce0acf4d19e14997ef` |
| release | `gpu:/tmp/flagos-moe-sum-reduce-e6-metax-release.k2HsCz`；PID/PGID `160514`；8/8、0.599s、`RELEASE_OK`；脚本/日志 SHA-256 `3f5dd03ac21627eceae49391adffa86f602c09646a0cbc78288ae95c406c3eca` / `43900ba0016450d0025e363c8fcb91be3e40fec4d2d36ffa88d743cb60ae48c4` |
| Git archive SHA-256 | `ad0cd85b66468decacb8460fc100d114d783078c2d2fda00a39ba2a37e94d5b8` |
| canonical ZIP | `artifacts/competition/moe_sum_reduce/e6-9d3c82a/moe_sum_reduce.zip`，12,410 bytes，SHA-256 `cf5ccff1f3724f1f1561b1a174327a961ef44db037c0c1a15e8d2dc660e7782d` |

代理 A/B 覆盖 fallback hidden7168 与 fast hidden2048/4096、三 dtype、最大
`(4096,8,4096)`；affected 中位 speedup 几何平均 `1.006816x`，fallback
最差 `1.000000x`。候选最多 40 registers、0 spill、0 shared、0 scratch；
8-warp fast path实际编译为 3 stages。NVIDIA 只证明无明显回退和资源风险，
不代表 MetaX 收益。

平台 one-shot 门：8/8 valid、沐曦选中 `_metax` 且高于 E5 的 `3.5040x`、
平均高于 `2.76185x`；显著收益目标为沐曦至少 `3.6640x`（整题约 +0.02x）。
任一基础门不满足即保留 E5，停止该 launch 假设。

2026-08-27 00:23:05 CST 在账号 `15600308080`、团队 `SoulCoder` 下执行 E6
唯一一次提交，submission `5209`、当日序号 `2`，额度由 `29/30` 变为
`28/30`。`file_url_sha256` 为
`3642af4cd64d3ed41b8a62318148e6f3b0d789d65056e14c4d5d32fd39bb50f5`；从已核实
对象存储地址无认证回读 12,410 bytes，SHA-256 与 canonical ZIP 完全一致。
平台已选中预期的 MetaX、Ascend、Kunlun 与 generic 路径；禁止重传。

00:23:37 CST 终态为 `completed` / `valid`、8/8、平均 `2.795625x`，平台
标记 team best。相对 E5 平均增加 `0.033775x`（+1.22%）；沐曦由
`3.5040x` 提升到 `3.8630x`（+10.25%），超过显著收益目标：

| 芯片 | E6 speedup | 相对 E5 | 选中文件 |
| --- | ---: | ---: | --- |
| 天数 | 4.6986x | -1.37% | `moe_sum_reduce.py` |
| 沐曦 | 3.8630x | +10.25% | `moe_sum_reduce_metax.py` |
| 燧原 | 0.2090x | -50.36% | `moe_sum_reduce.py` |
| 海光 | 6.5252x | +0.56% | `moe_sum_reduce.py` |
| 昆仑芯 | 0.1754x | -0.34% | `moe_sum_reduce_kunlunxin.py` |
| 华为 | 0.8548x | +5.48% | `moe_sum_reduce_ascend.py` |
| 国际通用 A | 3.7792x | +0.35% | `moe_sum_reduce.py` |
| 国际通用 B | 2.2598x | +4.42% | `moe_sum_reduce.py` |

结论：官方 MetaX launch policy 被平台证实，保留 E6 为 Task21 team best，
停止该轴。燧原同字节 generic 再次从 0.421x 波动到 0.209x，不能归因于
MetaX-only 改动；若无新的官方单芯实现，不为平台波动重传。

## E7：AMD 官方四档 BLOCK autotune

状态：平台 8/8、`valid`、`2.92895x`，团队当前最佳

E7 从 E6 team best 分叉，只新增 AMD vendor；generic、Ascend、Kunlun 与
MetaX 四个已有成员逐字节冻结。AMD 路径采用官方 FlagGems 的四档配置：
BLOCK/warps 为 `128/2`、`256/4`、`512/8`、`1024/8`，autotune key 为
`hidden_size,topk`，grid 从 selected config 的 meta 读取 BLOCK。launch 不显式
传 BLOCK、warps 或 stages，避免 FlagTree autotuner 重复绑定。

| 项目 | 值 |
| --- | --- |
| source / verification commit | `a305b67646c1c587cbe7d23c4d10af6226b20b01` |
| generic / Ascend SHA-256 | `52a2fc979784f2bd25e7e17b9822c23b4f438efdf062c70bfb09aba9ba732335` / `f740604cd4a0506a3e41776f3f9a001edffafef45f04aff78d1ee8a208f2132b` |
| Kunlun / MetaX SHA-256 | `68b0abe07e3cf4f2b9cb86063e9ad1e18edd83d90341410cd281ec595c83406d` / `20db4f49aada2976723ab9dabd7013545a30a0998ee8fb21ad655375bd2cb794` |
| AMD / test SHA-256 | `3b0de225dbf5ffc1004096055da871c919870cc0ba6e4a0297551c5c7537e399` / `19d8513c7b3acdc6dd9f1529ab996526a4f9073b14b80c9733b526b54220fa03` |
| screening | `gpu:/tmp/flagos-moe-sum-reduce-amd-screen.0djeCb`；10/10；日志 SHA-256 `db093544aef468392f2cf55dd9e7bd2fb2c9f90072082ba05966257953013c8d` |
| 交替 A/B | 同目录；18 点几何平均 `1.007808x`、最差中位 `0.969231x`；日志 SHA-256 `14f1bc0805c8e26d34f31c5ad62711aa27a65c01de0d286e398e712539dc6a11` |
| release | `gpu:/tmp/flagos-moe-sum-reduce-amd-release.rbjscH`；10/10、`RELEASE_OK`；日志 SHA-256 `2876188ff26e49270da98da30e77dd56e695d17007d42d07cfe145b53fba525c` |
| canonical ZIP | `artifacts/competition/moe_sum_reduce/e7-a305b67/moe_sum_reduce.zip`，15,584 bytes，SHA-256 `6b981b19be0dd0496757c74bd959dbe682d2f28c0c5b48aafc06c52865db4b82` |

四档均实际编译运行且都被选中过，候选为 0 spill/shared/scratch；代理只证明
正确性、资源与无明显回退，不外推 AMD 收益。历史平台 selected-file 已确认
`_amd` 只路由国际通用 B。one-shot 基础门为 8/8 valid、国际 B 选中 AMD 且
高于 E6 的 `2.2598x`、平均高于 `2.795625x`；显著收益门为国际 B 至少
`2.4198x`。当前第 10 名为 `2.95495x`，冻结其余七芯时国际 B 需严格超过
`3.5344x` 才升一位。任一基础门不满足即保留 E6，停止 AMD autotune 轴。

2026-08-27 01:41:41 CST 经实时门禁执行 E7 唯一一次提交，submission `5230`、
当日序号 `16`，额度由 `15/30` 变为 `14/30`。`file_url_sha256` 为
`19481adce178a1d0fd818c68ad71fa76fd5dbd274b16893d8b5f039737dd1e62`；从已核实
对象存储地址无认证回读 15,584 bytes，SHA-256 与 canonical ZIP 完全一致，
五个成员均通过 `unzip -t`。平台选中预期的 AMD、MetaX、Ascend、Kunlun 与
generic 路径；禁止重传。

01:42:29 CST 终态为 `completed` / `valid`、8/8、平均 `2.92895x`，平台
标记 team best。相对 E6 平均增加 `0.133325x`（+4.77%）；国际 B 由
`2.2598x` 提升到 `3.4598x`（+53.10%），超过显著收益门，但仍比预估升一名线
`3.5344x` 低 `0.0746x`：

| 芯片 | E7 speedup | 相对 E6 | 选中文件 |
| --- | ---: | ---: | --- |
| 天数 | 4.7294x | +0.66% | `moe_sum_reduce.py` |
| 沐曦 | 3.5246x | -8.76% | `moe_sum_reduce_metax.py` |
| 燧原 | 0.2090x | 0.00% | `moe_sum_reduce.py` |
| 海光 | 6.6482x | +1.88% | `moe_sum_reduce.py` |
| 昆仑芯 | 0.1752x | -0.11% | `moe_sum_reduce_kunlunxin.py` |
| 华为 | 0.8868x | +3.74% | `moe_sum_reduce_ascend.py` |
| 国际通用 A | 3.7986x | +0.51% | `moe_sum_reduce.py` |
| 国际通用 B | 3.4598x | +53.10% | `moe_sum_reduce_amd.py` |

结论：官方 AMD autotune 被平台证实，保留 E7 为 Task21 team best，停止四档
autotune 轴。其余四成员字节冻结，逐芯波动不能归因于 AMD-only 改动；只有找到
新的 AMD 配置机制且离线门禁通过，才允许再做一个单变量候选。

01:45 CST 公开榜单刷新后，E7 仍为第 `11/11`；上邻第 10 名为
`2.95495x`，只差 `0.026x`。上邻国际 B 为 `3.4312x`，已经低于本队
`3.4598x`，当前名次缺口来自其他芯片，故不再为 AMD 盲加配置。
