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
