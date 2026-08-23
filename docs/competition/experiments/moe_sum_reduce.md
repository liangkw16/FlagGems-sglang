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
