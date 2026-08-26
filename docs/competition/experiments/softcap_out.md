# Task 24 `softcap_out` 实验记录

## S0：generic baseline

状态：平台首轮 6/8；已由 S1 vendor 修复至 8/8
验证时间：2026-08-24 00:26–00:27 CST  
源码 commit：`196ee005b4d18f388e112920332c1bd1abe7b921`

### 构建身份

| 项目 | 值 |
| --- | --- |
| 源文件 | `src/flaggems_sglang/ops/softcap_out.py` |
| 源文件 SHA-256 | `e6ab1c434aa793bc58357e3d45d2eec7fd2ec56bebb65538b2a6049ca9a37ddc` |
| 测试 SHA-256 | `e24c142453011af201dcdba3f3c490db2df3a71573e7f2f8e881bd493da73af8` |
| ZIP | `artifacts/competition/softcap_out/s0-196ee00/softcap_out.zip` |
| ZIP SHA-256 | `3bb1218d87b2b6148a7336a975fdc4e0960629dc735ceadace235ba75cfd2814` |
| ZIP 内容 | 单个顶层文件 `softcap_out.py`，2277 bytes；ZIP 10 MB 门禁通过 |
| 远端证据目录 | `gpu:/tmp/flagos-softcap-release.xKfJUI`，mode 0700 |

ZIP 中的源码与 commit 源文件逐字节一致。没有 vendor 文件，也没有测试、缓存
或仓库依赖。

### 唯一候选配置

- generic Triton kernel；BLOCK 256；普通一维 masked grid。
- FP32 计算和输出；`|x/cap| < 0.25` 使用五阶奇多项式，其余使用稳定 exp。
- 不显式设置 `num_warps`、`num_stages` 或厂商选项。
- 支持非连续输入、Python scalar 和单元素 tensor cap；无设备判断或
  PyTorch fallback。

### 正确性

本地与远端源码、测试 SHA-256 完全一致。远端环境：RTX 5070 Ti 16 GB、
driver 610.57.04、Python 3.12.13、PyTorch 2.13.0+cu130、Triton 3.7.1、
CUDA 13.0、compute capability 12.0。

直接执行 `tests/test_softcap_out.py -v`：10/10 通过。覆盖：

- FP16、BF16、FP32 计算及固定 FP32 输出；
- 长度 0、1、17、63/64/65、127/128/129、255/256/257、
  511/512/513、1023/1024/1025；
- 连续、非连续、输入不变和 out-of-place；
- cap 为 Python float/int、CPU/设备单元素 tensor、多元素拒绝；
- cap 为 0、负数、`1e6`、FP32 最小 subnormal、`2^-128` 临界值、
  FP32 max、±Inf 和 NaN；
- 近零、分支边界、饱和区、Inf/NaN，以及固定种子的正态/均匀输入。

Black 79、isort、flake8 和 Python 语法检查均通过。完整 package registrar
因远端环境缺少仓库既有 `triton_kernels` 依赖而未作为 S0 ZIP 门禁；该项留给
正式上游 PR 环境。

### 本地性能

wrapper-inclusive；cap=30；每个组合先 JIT 和同步，然后执行五组
`triton.testing.do_bench(warmup=25, rep=100, quantiles=[0.2, 0.5, 0.8])`。
S0 与 PyTorch reference 每组轮换先后顺序。表中时间为五组 p50 的中位数。

| dtype | numel | S0 p50 (ms) | Torch p50 (ms) | speedup |
| --- | ---: | ---: | ---: | ---: |
| float16 | 4096 | 0.004192 | 0.010176 | 2.427x |
| float16 | 65536 | 0.004160 | 0.010240 | 2.462x |
| float16 | 1048576 | 0.010112 | 0.026624 | 2.633x |
| float16 | 16777216 | 0.139328 | 0.665376 | 4.776x |
| bfloat16 | 4096 | 0.004192 | 0.010208 | 2.435x |
| bfloat16 | 65536 | 0.004192 | 0.010240 | 2.443x |
| bfloat16 | 1048576 | 0.010016 | 0.026624 | 2.658x |
| bfloat16 | 16777216 | 0.139328 | 0.665536 | 4.777x |
| float32 | 4096 | 0.004192 | 0.006176 | 1.473x |
| float32 | 65536 | 0.004192 | 0.008160 | 1.947x |
| float32 | 1048576 | 0.014368 | 0.020576 | 1.432x |
| float32 | 16777216 | 0.178272 | 0.528400 | 2.964x |

本地最小 speedup 为 1.432x。普通 cap=30 的 NVIDIA 编译产物为 4 warps、
3 stages、41 个 B32 寄存器、0 shared memory、0 global scratch；metadata
未报告 spill。PTX 使用 `div.full.f32` 和 `ex2.approx.f32`。

### 已知边界

- 上述结论只证明本地 NVIDIA 路径，不能替代比赛平台八芯结果。
- S0 实测燧原最大用例需要 `grid.x=256512`，超过硬件上限 65535；S1 需要
  `_enflame.py` 的大 block 或 grid-stride 方案。Ascend 另有正确性失败，且
  `numel=2^24`、BLOCK 256 的 65536-grid 风险仍需在 vendor 版本一并规避。
- 以下平台结果以比赛页面真实返回为准。

### GitHub 交付

- 研究分支已推送到
  `liangkw16/FlagGems-sglang:research/season2-batch2`。
- 为避免把竞赛资料提交带入上游，从 `origin/master@3946b9a` 建立了只含算子
  和测试的 `ci/softcap-out`，commit 为 `e38ac7c`。
- 上游 [Draft PR #32](https://github.com/flagos-ai/FlagGems-sglang/pull/32)。
- [`basic ci` run 32651948481](https://github.com/flagos-ai/FlagGems-sglang/actions/runs/32651948481)
  当前为 `action_required`：首次外部 fork workflow 等待上游维护者批准，
  并非测试失败；CLA check 同时等待本人签署。

### 平台结果

- 提交时间：2026-08-24 00:45 CST
- 提交文件：`softcap_out.zip`
- 平台基础校验：通过，识别 1 个 `.py`，generic 文件命名正确
- 当前状态：已完成，6/8；两芯失败，平台未生成平均加速比
- 本次提交后今日剩余额度：14/15
- 页面未展示独立提交 ID，以 Task24、文件名和提交时间联合定位本次流水
- 燧原失败：`grid.x` 需要 256512，硬件上限 65535
- 华为失败：Case 1 输出约为期望值的 `1/30`，表现为最终 cap 缩放丢失

| 芯片 | 正确性 | speedup | 决策 |
| --- | --- | ---: | --- |
| 天数智芯 | 通过 | 3.60x | 保留 generic |
| 沐曦 | 通过 | 1.96x | 保留 generic |
| 燧原 | 失败 | — | S1 增加 grid 上限内 vendor 实现 |
| 海光 | 通过 | 2.14x | 保留 generic |
| 昆仑芯 | 通过 | 0.44x | 正确性已过，后续做性能 vendor |
| 华为 | 失败 | — | S1 显式保留 cap 缩放并限制 grid |
| 国际通用 A | 通过 | 3.16x | 保留 generic |
| 国际通用 B | 通过 | 2.78x | 保留 generic |

## S1：Ascend / Enflame correctness recovery

状态：平台复测完成，8/8 通过；当前单题第 8 名

### 构建身份

| 项目 | 值 |
| --- | --- |
| 源码 commit | `fe2348eae40cb59ffe8945676cb56e1958e58706` |
| ZIP | `artifacts/competition/softcap_out/s1-fe2348e/softcap_out.zip` |
| ZIP SHA-256 | `698a7d9652d973868941e6e9e773d7d62ec1dceb87e0e392430b4b1c9cc69ded` |
| generic SHA-256 | `e6ab1c434aa793bc58357e3d45d2eec7fd2ec56bebb65538b2a6049ca9a37ddc` |
| Ascend SHA-256 | `cf79382068d6127548594c38ccc3951ff8f255e56eb73bad0c24dcf7a5871425` |
| Enflame SHA-256 | `9524cc41a0bf8f6a9b2cc011e5fc09d7c2d47382bf4dd797662981489b6e2af2` |
| ZIP 内容 | 三个顶层文件：generic、`_ascend.py`、`_enflame.py`；3589 bytes |

### 唯一改动

- generic 文件逐字节保持 S0，不影响已通过的六款芯片。
- Ascend 保持 BLOCK 256，把物理 grid 上限设为 48 并在 program 内遍历
  logical blocks；最终缩放改为 `softcap_const * normalized`，匹配 Ascend
  官方同类 softcap 写法，并避开 S0 中 scalar 位于乘法右侧的 lowering 路径。
- Enflame 保持 BLOCK 256 和数值路径不变，把 grid 上限设为 12 并在
  program 内遍历 logical blocks，消除 S0 的 `grid.x > 65535`。

### 代理验证

- 远端 RTX 5070 Ti 16 GB：11/11 unittest 通过；vendor 回归覆盖 FP16、
  BF16、FP32，长度 `48*256+17`，同时命中 Ascend 和 Enflame 的多轮循环。
- 以平台失败规模上界 `N=65,667,072` 完整生成并逐元素比对 reference：
  Ascend、Enflame 两份 vendor 均通过 FP32 `1e-4` 容差。
- Black 79、isort、flake8、Python 语法和 ZIP manifest 门禁通过。

### 平台复测

- 提交时间：2026-08-24 01:03:51 CST
- 团队：`SoulCoder`；8/8 通过，平均加速比 1.90x，当前单题第 8/8 名
- 本次提交后今日剩余额度：13/15
- 页面未展示独立提交 ID，以 Task24、文件名和时间联合定位

| 芯片 | 正确性 | speedup | 相对 S0 |
| --- | --- | ---: | ---: |
| 天数智芯 | 通过 | 3.66x | +0.06x |
| 沐曦 | 通过 | 1.93x | -0.03x |
| 燧原 | 通过 | 0.35x | 从失败恢复 |
| 海光 | 通过 | 2.13x | -0.01x |
| 昆仑芯 | 通过 | 0.45x | +0.01x |
| 华为 | 通过 | 0.77x | 从失败恢复 |
| 国际通用 A | 通过 | 3.14x | -0.02x |
| 国际通用 B | 通过 | 2.77x | -0.01x |

### 后续优化

- Ascend operand-order workaround 已由本次平台复测证明可用；不再引入
  `tl.constexpr` 候选。
- Enflame 正确性已恢复，但 0.35x 说明 BLOCK 256 的约 21,376 次/CTA 循环
  开销过高；S2 保持 grid 12，单变量优先测试 BLOCK 4096。
- 当前榜首 2.43x；最大差距依次在 Enflame（0.35x 对 3.12x）和 Hygon
  （2.13x 对 3.27x）。下一轮先优化 Enflame，再评估 Hygon BLOCK。

## S2：Enflame 4096 tile 性能候选

状态：远端 NVIDIA 代理正确性、性能筛选和不可变 ZIP 门禁通过；S2c 已于
2026-08-25 提交平台并 8/8 通过，见下方平台复测

### 构建身份

| 项目 | 值 |
| --- | --- |
| 源码 commit | `5cd60194ee4761b097f089b6eae96e54254cfada` |
| ZIP | `artifacts/competition/softcap_out/s2-5cd6019/softcap_out.zip` |
| ZIP SHA-256 | `3746930f19d1a255571906fd4defd59b4a7ee272a65343f519969cd265e3db20` |
| generic SHA-256 | `e6ab1c434aa793bc58357e3d45d2eec7fd2ec56bebb65538b2a6049ca9a37ddc` |
| Ascend SHA-256 | `cf79382068d6127548594c38ccc3951ff8f255e56eb73bad0c24dcf7a5871425` |
| Enflame SHA-256 | `f9cbdd5eb5b13e6b754bb2666e81e68523f17270246a006893db85be27a8c562` |
| 测试 SHA-256 | `28f5399a3b04afc5593bfbcd061a4c07072e0c89e241843f7ff0641ac2d2f273` |
| ZIP 内容 | 三个顶层 UTF-8 `.py`；7,289 bytes 源码、3,591 bytes ZIP |
| 远端证据目录 | `gpu:/tmp/flagos-softcap-s2.L88VNU`，mode 0700 |
| 平台结果 | 未提交；逐芯结果、均值、排名和本次额度均为 N/A |

### 单变量与固定依据

- generic 和 Ascend 文件逐字节保持 S1；Enflame 只把 grid 计算及 kernel 的
  `BLOCK_SIZE` 从 256 改为 4096，物理 grid 继续固定最多 12，数学和 launch
  其他参数不变。
- 固定 FlagGems commit `ed2508bcb5a03000e9774734201d840ba362cd11`
  的 Enflame pointwise policy 为最大 tile 4096、最大 grid 12、4 warps；这是
  上游软件策略，不是比赛 GCU 资源保证。
- 平台观测最大用例 `N=65,667,072` 从 256,512 个旧 logical blocks 变为
  16,032 个 blocks，每个 CTA 的循环次数由 21,376 降至 1,336，恰好减少 16 倍。

### 代理验证

- 远端 py_compile、Black 79、isort、flake8 全部通过；unittest 11/11 通过。
  vendor 回归使用 `N=12*4096+17`：Enflame 为 13 blocks / 12 CTA，Ascend
  为 193 blocks / 48 CTA；三种 dtype 均覆盖第二轮循环、17 元素 tail 和输入
  不变性。
- `N=65,667,072` 的 FP16、BF16、FP32 全量输出均与题面 reference 通过对应
  容差；该探针同时覆盖每 CTA 1,336 次动态循环。
- RTX 5070 Ti 编译产物均为 4 warps、3 stages、0 global scratch、0 spill；
  FP16/BF16 为 56 registers、4,096 bytes shared，FP32 为 93 registers、
  0 shared。这些资源数字不能替代真实 Enflame JIT。
- ZIP 的三个成员名、UTF-8、10 MB、`unzip -t` 和 ZIP 内源码与 commit 源码
  逐字节一致门禁均通过。

wrapper-inclusive NVIDIA 单变量对比；每项五组交替顺序，组内
`warmup=25, rep=100`，表中为五组 p50 的中位数：

| dtype | numel | S1 ms | S2 ms | S2 / S1 |
| --- | ---: | ---: | ---: | ---: |
| FP16 | 4,096 | 0.004529 | 0.004465 | 1.0145x |
| FP16 | 65,536 | 0.012925 | 0.006319 | 2.0453x |
| FP16 | 1,048,576 | 0.133449 | 0.028173 | 4.7367x |
| FP16 | 16,777,216 | 2.032990 | 0.374900 | 5.4228x |
| FP16 | 65,667,072 | 7.906453 | 1.435665 | 5.5072x |
| BF16 | 4,096 | 0.004463 | 0.004423 | 1.0092x |
| BF16 | 65,536 | 0.012896 | 0.006330 | 2.0374x |
| BF16 | 1,048,576 | 0.133126 | 0.027249 | 4.8856x |
| BF16 | 16,777,216 | 2.029226 | 0.359651 | 5.6422x |
| BF16 | 65,667,072 | 7.894272 | 1.394875 | 5.6595x |
| FP32 | 4,096 | 0.004402 | 0.005410 | 0.8136x |
| FP32 | 65,536 | 0.012706 | 0.007359 | 1.7266x |
| FP32 | 1,048,576 | 0.135484 | 0.038376 | 3.5304x |
| FP32 | 16,777,216 | 2.069782 | 0.501093 | 4.1305x |
| FP32 | 65,667,072 | 8.058147 | 1.928959 | 4.1775x |

NVIDIA 上仅 FP32 `N=4096` 回退 18.6%，其余代理点持平或提升；真实 GCU 的
编译资源、大小 shape 权重和 speedup 在下方 S2c 平台复测中确认。

### Canonical 提交产物

2026-08-24 11:39 CST 从同一 source commit `5cd60194ee4761b097f089b6eae96e54254cfada`
重新生成独立 stage，不覆盖或改写原 S2 legacy ZIP。源码、测试和既有 release
证据均未变化。

| 项目 | 值 |
| --- | --- |
| stage / ZIP | `s2c-5cd6019` / `artifacts/competition/softcap_out/s2c-5cd6019/softcap_out.zip` |
| ZIP SHA-256 | `999f2dea69774c2f9756748a2a113c7ad54d3e2fdce18bfd24b014a96fed1f46` |
| 大小 / 状态 | 7,653 bytes；`verified-existing`，与 canonical SHA 相同 |
| 成员 | `softcap_out.py`、`softcap_out_ascend.py`、`softcap_out_enflame.py` |
| 平台状态 | submission `4536`，8/8、1.9855x、valid、team best |

### S2c 平台复测

- 提交时间：2026-08-25 08:43:20 CST；平台 submission `4536`。
- preflight 实时核对 Task 24、`s2t1op024`、账号、团队 `SoulCoder`、
  source commit、三成员集合、ZIP 绝对路径及 SHA-256；提交前额度 `8/30`。
- 上传与正式提交各执行一次。平台对象存储回读为 7,653 bytes，SHA-256
  `999f2dea69774c2f9756748a2a113c7ad54d3e2fdce18bfd24b014a96fed1f46`，
  与本地不可变 ZIP 完全一致。
- 终态：8/8、`valid`、平均 1.9855x、team best；提交后额度 `7/30`。
  状态 API 不返回实时榜单名次，因此不记录推测排名。

| 芯片 | S1 speedup | S2c speedup | 选择文件 | 结果 |
| --- | ---: | ---: | --- | --- |
| 天数智芯 | 3.65525x | 3.62800x | `softcap_out.py` | 通过 |
| 沐曦 | 1.92817x | 1.73042x | `softcap_out.py` | 通过 |
| 燧原 | 0.34575x | 1.18917x | `softcap_out_enflame.py` | 通过 |
| 海光 | 2.13033x | 2.17225x | `softcap_out.py` | 通过 |
| 昆仑芯 | 0.44750x | 0.44442x | `softcap_out.py` | 通过 |
| 华为 | 0.76958x | 0.70975x | `softcap_out_ascend.py` | 通过 |
| 国际通用 A | 3.13533x | 3.20033x | `softcap_out.py` | 通过 |
| 国际通用 B | 2.77033x | 2.80967x | `softcap_out.py` | 通过 |

S2c 把唯一改动芯片燧原提升至 S1 的 3.439x；平均加速比绝对增加
0.08772x（+4.62%）。其余七芯复用原字节并保持通过。该受控候选成为团队当前
最佳，停止继续扩大同一 tile 假设。

## S2d：昆仑 BLOCK 1024 vendor（性能冲刺）

Task 24 S2c 已 8/8 有效（平均 1.9855x），昆仑 0.44442x 是最弱芯。昆仑
知识（Task 21 平台验证）：BLOCK 是唯一有效调参轴，1024 是唯一平台成功
先例。S2d 新增 `_kunlunxin/ops/softcap_out.py`：kernel 与 generic 逐行
相同，仅 `BLOCK_SIZE` 256→1024（1D 平铺 grid，num_warps 4、stages 1 与
Task 21 昆仑 vendor 同形）。

- 单变量：仅昆仑。generic、`_ascend`、`_enflame` 与 S2c ZIP 逐字节相同；
  `test_vendor_overrides_preserve_cap_scaling` 的 vendor 元组加入
  `kunlunxin`（49152+17 元素输入自动覆盖多 program 路径）。
- source/verification commit：
  `8928ef286b9f0b432d1f1140c0c9d0fa6d41269c`；本地 py_compile、black
  25.12.0、isort、flake8 通过。
- canonical ZIP：`artifacts/competition/softcap_out/s2d-8928ef2/softcap_out.zip`，
  ZIP SHA-256
  `113c5e7233c213dec6a54b73188acda22a011210876861dad487010dbd11126a`，
  成员 generic + `_ascend`/`_enflame`/`_kunlunxin`，`unzip -t` 通过。
- release 目录：`gpu:/tmp/flagos-multi-release.JfYAit/t24-stage`（与
  T12 E3 同批串行）。
- 平台门禁：昆仑 ≥0.1x 且平均较 S2c 1.9855x 提升；其余七芯文件不变。

<!-- T24_S2D_RELEASE_RESULT_PENDING -->

### S2d/S2e 平台结果：昆仑 BLOCK 1024→4096 两连升，平均 2.0179x

- S2d（BLOCK 1024）：2026-08-25 15:33:56 提交（submission `4652`，序号
  `27`，额度 `5/30`→`4/30`，`file_url_sha256`
  `2ca3bff66142dac19838ab626aa013e0128f320f7b5402d894cea2ce33921fc8`），
  8/8 **valid**，平均 `1.9943x`：昆仑 0.4444→**0.7645x**（+72%，
  BLOCK 是昆仑唯一有效调参轴的第三次平台验证）；release
  `gpu:/tmp/flagos-multi-release.JfYAit/t24-stage`（`Ran 11 tests in
  0.648s`、`RELEASE_OK`）。
- S2e（BLOCK 4096，commit `1a5ea268de23bc5e5ba7ca7e8c7acd5390ac3725`，
  ZIP `s2e-1a5ea26` SHA-256
  `8469beb23dbaa27fbfbd7f6f74b650ee899586f43bacfb7e3fdd40e8dd566ed0`，
  release `gpu:/tmp/flagos-multi3-release.x58bBH/t24s2e-stage`
  `RELEASE_OK`）：15:44:56 提交（submission `4657`，序号 `29`，额度
  `2/30`→`1/30`），8/8 **valid**，平均 **`2.0179x`（team best，
  `is_team_best=true`）**：昆仑 0.7645→**0.8637x**（+13%），燧原
  1.1881x、华为 0.7371x、天数 3.6039x、海光 2.1488x、card_a 3.1368x、
  card_b 2.7835x、沐曦 1.6817x。
- Task 24 闭环：平均 1.9855→2.0179x；昆仑 BLOCK 曲线 256（generic，
  0.444x）→1024（0.7645x）→4096（0.8637x）仍未饱和，4096 为当前
  团队昆仑 elementwise 最佳配置。今日额度用尽后停止，剩 1 次留作
  截止前回归储备。

## S3c：FlagTree XPU grid=12 interleave

FlagTree 固定提交 `367dc5794f678a70ec57bb8a1b3d24bf9b855ca6` 显示，XPU
JIT 会把实际 launch grid 写入编译选项，而编译器仅在 grid 为 `(12, 1, 1)`
时启用 interleave pass；该 pass 接受本题的 `pid * BLOCK + arange` 一维
pointwise 形态。初版 S3 `45267aa` 把 `pid` 放进 `range` 下界；独立审查发现
循环 body 的 block argument 没有 defining op，无法被 `findDefOpBwd` 追溯，
因此该版本未提交。S3b `e3c1fdf` 又被编译器元数据审查挡住：FlagTree 在
compile 前严格比较 `(12, 1, 1)`，而一维 tuple 的补位发生在 launch 阶段。
S3c 因此明确传入三维 grid，并保持 BLOCK 4096、四 warps、单 stage 和数学字节
不变，将 `pid * BLOCK + arange` 显式保留在循环外，再按 12 个 program 一窗
推进，使静态 matcher 链成立。

- source/verification commit：
  `89699c64134852d2dfbb94bbae6d46b8710422e0`；与 S2e 相比，仅
  `softcap_out_kunlunxin.py` 改变，其余三个 ZIP 成员 SHA-256 逐字节相同。
- release：`gpu:/tmp/flagos-softcap-interleave2-release.ct5rpi`；远端从 Git
  对象归档，Python 3.12.13、Torch 2.13.0+cu130、Triton 3.7.1、RTX 5070 Ti。
  py_compile、Black、isort、flake8 及完整 unittest 11/11 通过；
  `release5.log` SHA-256
  `1675d3c062e25af4c2d0833c2b936811057b189ecd63fe5fefad44f72ec2745f`。
- 额外边界门：FP16/BF16/FP32 分别覆盖 `12*4096-1`、`12*4096+17`、
  `24*4096+123`、`1,000,003`，全部与 PyTorch 参考一致；CUDA 编译资源为
  55 registers、4096-byte shared、zero scratch；TTIR 明确保留
  `get_program_id → muli 4096 → add make_range` 后再进入 `scf.for`。
  额外脚本 SHA-256
  `fc6dab5c5c7d01cb1ecbc797c72e3c20d1c25c3a03a4eb3f0353c36cc82ac555`。
- canonical ZIP：`artifacts/competition/softcap_out/s3c-89699c6/softcap_out.zip`，
  10,344 bytes，SHA-256
  `452f54b0da15c698c6eae56790ed7979739c2668cacb2f71eba9553bf35c2807`；
  `verified-existing` 与 `unzip -t` 均通过。
- 一次提交晋级门：8/8 valid、昆仑高于 S2e 的 0.8637x、平均高于
  2.0179x；任一不满足即保留 S2e 并停止此假设，不做 grid sweep。

### S3c 平台结果：正确但显著回退，永久停止 interleave 假设

- 2026-08-26 20:23:41 CST 单次提交，submission `5128`、当日序号 `15`；
  额度 `16/30`→`15/30`，上传对象 10,344 bytes 且回读 SHA-256 与本地
  `452f54b0da15c698c6eae56790ed7979739c2668cacb2f71eba9553bf35c2807`
  完全一致；`file_url_sha256`
  `82d3eb70bbe31d72b8e6894c8dfcb7439ec16837ae3f70ce82091b7a8a9ff670`。
- 终态 8/8、`valid`，但平均仅 `1.95161458x`、非 team best；昆仑
  `0.86366667x`→`0.24175x`（-72.0%）。其余冻结芯片为天数 3.66075x、
  沐曦 1.70716667x、燧原 1.182x、海光 2.13416667x、华为 0.74275x、
  card_a 3.13075x、card_b 2.81358333x。
- 未达到“昆仑 >0.8637x 且平均 >2.0179x”晋级门。保留 S2e 为平台
  team best，源码已恢复到 S2e 的 Kunlun SHA-256
  `f34b06168ec951453601f552d3b6aa7bef1dac954a592226d3ac76ec248066bb`；
  永久停止 grid=12/interleave 假设，不做 grid 或 round sweep。

## S4：Ascend BLOCK 512

状态：平台 8/8、`valid`、`2.04696875x`，团队当前最佳

S4 只把 Ascend vendor 的 capped grid-stride tile 从 256 改为 512；physical
worker cap 48、数学、operand order 和 launch 其余参数不变。generic、Enflame、
Kunlun 继续冻结为 S2e 字节。预注册晋级门为 8/8 valid、华为高于 S2e 的
`0.7371x` 且平均高于 `2.0179x`；任一不满足即保留 S2e，不扩 sweep。

| 项目 | 值 |
| --- | --- |
| source commit | `1dadfcdac849fdb72386016b15b19180d0a2e93e` |
| verification commit | `54c4f69629fc14f26eafea61a638499e24a105bb` |
| generic / Ascend SHA-256 | `e6ab1c434aa793bc58357e3d45d2eec7fd2ec56bebb65538b2a6049ca9a37ddc` / `bb98a5fda924e09954ce5778a859d064daaa560ebc7d49f6dbd1229dadabb50b` |
| Enflame / Kunlun SHA-256 | `f9cbdd5eb5b13e6b754bb2666e81e68523f17270246a006893db85be27a8c562` / `f34b06168ec951453601f552d3b6aa7bef1dac954a592226d3ac76ec248066bb` |
| 测试 SHA-256 | `83cd2bea1f52fa626126dc183c77ceb4566c1a788b59f3a83bec973bcf8713ca` |
| screening | `gpu:/tmp/flagos-softcap-out-s4-screen.HTXRbr`；PID/PGID `159847`；12/12、0.555s、`SCREENING_OK`；脚本/日志 SHA-256 `f89edbfda763585cf0bbab45c8bf8e974828ee74ccd6668a9857cd15a1caeb3a` / `35917f752770550aa29df2c2cb71886f5c33ee3fef6773b174c6b03e2011c725` |
| 交替 A/B | 同目录；PID/PGID `159961`；脚本/日志 SHA-256 `3eb7e6c8562c85006ea2acba15b81691524aa2950c9bd303bc871cb2e402bbb5` / `ca58571a79582e29fb8ca49b8fcfa0b623ba5df4cbd0bf3d2030295f1318dd76` |
| release | `gpu:/tmp/flagos-softcap-out-s4-release.shg1iC`；PID/PGID `160083`；12/12、0.556s、`RELEASE_OK`；脚本/日志 SHA-256 `424df31756ad0237a9432838e59456922e6bce364df50c1577912987b8992ace` / `82dd3d8fa487e0e9273ccb96f90e81445b6eaf17f72783d4bf9203e6f0035a38` |
| source Git archive SHA-256 | `c3a08acebde8fa3a216848907589231799db2416a6b0e0a2b96570cc1491b5c0` |
| canonical ZIP | `artifacts/competition/softcap_out/s4-1dadfcd/softcap_out.zip`，10,099 bytes，SHA-256 `592d5dc80568fc9883bba0939e9eb45604037cdfc0f181e0b1b57b885c0536f1` |

新增回归覆盖 Ascend 256/512 边界及平台最大 `N=65,667,072`。首次 screening
揭示 CUDA FP16 `torch.linspace` 在该长度只生成 131,040 个有限值、其余
65,536,032 个为 NaN；验证 commit 改为先生成 FP32 再 cast，输入全有限后重跑
完整门禁。此修复仅改变测试数据生成，不改变 ZIP 源码。

RTX 5070 Ti 上 BLOCK256↔512 五组 AB/BA 覆盖长度 255、513、49,169、
1,000,003、65,667,072 与三 dtype；非 control 中位 speedup 几何平均
`1.362511x`，最差中位 `1.000000x`。最大规模 FP16/BF16/FP32 分别为
`1.758058/1.757346/1.665585x`；candidate 23–32 registers、0 spill、0 shared、
0 global scratch。该性能只作资源与明显回退门禁，不外推为 Ascend 收益。
canonical ZIP 四个成员均与 source commit 一致，`dry-run`、
`verified-existing`、UTF-8、10 MB 和 `unzip -t` 门禁通过。

2026-08-27 00:16:50 CST 在账号 `15600308080`、团队 `SoulCoder` 下执行 S4
唯一一次提交，submission `5204`、当日序号 `1`，额度由 `30/30` 变为
`29/30`。`file_url_sha256` 为
`e8fad19f54654b50953dae5189cc17292417a96b187782c587305c0f1317139a`；从已核实
对象存储地址无认证回读 10,099 bytes，SHA-256 与 canonical ZIP 完全一致。
平台选中了预期的 generic、Ascend、Enflame、Kunlun 四条路径；禁止重传。

00:17:35 CST 终态为 `completed` / `valid`、8/8、平均 `2.04696875x`，平台
标记 team best。相对 S2e 平均增加 `0.02903125x`（+1.44%），华为由
`0.73708333x` 提升到 `0.88375x`（+19.90%），通过两项预注册门：

| 芯片 | S4 speedup | 相对 S2e | 选中文件 |
| --- | ---: | ---: | --- |
| 天数 | 3.63441667x | +0.85% | `softcap_out.py` |
| 沐曦 | 1.69900000x | +1.03% | `softcap_out.py` |
| 燧原 | 1.18416667x | -0.33% | `softcap_out_enflame.py` |
| 海光 | 2.16458333x | +0.74% | `softcap_out.py` |
| 昆仑芯 | 0.86508333x | +0.16% | `softcap_out_kunlunxin.py` |
| 华为 | 0.88375000x | +19.90% | `softcap_out_ascend.py` |
| 国际通用 A | 3.15191667x | +0.48% | `softcap_out.py` |
| 国际通用 B | 2.79283333x | +0.34% | `softcap_out.py` |

结论：Ascend BLOCK512 单变量被平台证实；保留 S4 为 Task24 team best，停止
Ascend tile 扩展。其余七芯使用冻结字节并全部过门槛。

## S5：Kunlun 原生 `tanh`

状态：平台 8/8、`valid`、`2.02248958x`，非 team best；保留 S4

S5 只把 Kunlun vendor 的手写 `exp` 多项式 `tanh` 换成 XPU 官方
`tl_extra_shim.tanh`；BLOCK 4096、grid、四 warps、单 stage、cap 分支及输出
缩放均不变，generic、Ascend、Enflame 继续冻结为 S4 字节。固定官方依据：
FlagGems-Experimental `c73617ac10535ba3140a3e41e0291556d68d41b2` 的 Kunlun
`tanh.py` 与 `flash_kernel.py` 均直接使用该 intrinsic；FlagTree
`c1ea8285a06e97afad9dd2644bc71f2efca072f4` 将其落到 XPU `tanhf`，向量化
pass 支持 16-lane `tanh`。

| 项目 | 值 |
| --- | --- |
| source / verification commit | `76551bc02a1f09763f04a5872627b440c45c8213` |
| generic / Ascend SHA-256 | `e6ab1c434aa793bc58357e3d45d2eec7fd2ec56bebb65538b2a6049ca9a37ddc` / `bb98a5fda924e09954ce5778a859d064daaa560ebc7d49f6dbd1229dadabb50b` |
| Enflame / Kunlun SHA-256 | `f9cbdd5eb5b13e6b754bb2666e81e68523f17270246a006893db85be27a8c562` / `7c6121b04f4960040296bbc13f911e583f2b3151a0f67e2589117a2323a8674e` |
| 测试 SHA-256 | `7ea3feab807eff27aa630101bd8ae0256df21af83ac6750f3e45b984fc308d57` |
| screening | `gpu:/tmp/flagos-softcap-out-s5-kunlun-tanh-screen.9ivSL9`；PID/PGID `160675`；13/13、0.980s、`SCREENING_OK`；脚本/日志 SHA-256 `31593ba3f641f6c0ed6662ae4e00975a32c42fd3cd396ba2d5d98d41b0efd1c1` / `4681a8d466c331a2fcae0ac284c9b6ec04b5eab190221ad75b9d3b7d66f81516` |
| 交替 A/B | 同目录；PID/PGID `160834`；脚本/日志 SHA-256 `a1204ab905441ea21d621f89ddc48ddb31f0e49a6dd4057bb73dbc68f88e2c8c` / `2c513ad71dfaef0b952f705db2c21dc14e08b52e74893e67a76d4fc511652356` |
| release | `gpu:/tmp/flagos-softcap-out-s5-kunlun-tanh-release.dCgFal`；PID/PGID `160962`；13/13、0.563s、`RELEASE_OK`；脚本/日志 SHA-256 `dd6e11b856f27c157ebfa4badcf482db35e4043d2271dc94a94d54e0c22c9712` / `c15a52e6882ac558fe15ec08a220322e7e062f99969ee2e02400437db36464d7` |
| source Git archive SHA-256 | `e6ef337ca1a7d461ef348b280ad494b8408448f3f19bb3ca82d5627607c0bbd3` |
| canonical ZIP | `artifacts/competition/softcap_out/s5-76551bc/softcap_out.zip`，10,043 bytes，SHA-256 `891c641685bbf7bf44e2fdac5df27c58e949372fe87e692082b74708653fb3a1` |

新增回归直接覆盖 NaN、正负无穷、零、边界值与极端 cap。RTX 5070 Ti 代理上
最大平台 shape 的 FP16/BF16/FP32 中位比为 `0.994555/0.992570/1.000554x`，
candidate 最大 72 registers（旧式最高 96）、zero spill/scratch；小 shape 因
CUDA libdevice 调用开销使非 control 几何平均为 `0.961117x`，只作为资源和
正确性代理，不把 CUDA 时延外推成 XPU 收益。

一次提交晋级门：8/8 valid、Kunlun 高于 S4 的 `0.86508333x`、平均高于
`2.04696875x`；显著收益目标为 Kunlun 至少 `1.02508333x`（对应整题平均约
`+0.02x`）。任一基础门不满足即保留 S4，并永久停止 native-`tanh` 轴。

2026-08-27 00:29:27 CST 在账号 `15600308080`、团队 `SoulCoder` 下执行 S5
唯一一次提交，submission `5210`、当日序号 `3`，额度预计由 `28/30` 变为
`27/30`。`file_url_sha256` 为
`d0f8f1e2aefcc568a24ed9345724e7b9388cf49e139ecdf77ceb15519399edd4`；对平台
返回的匿名对象限长回读为 10,043 bytes，SHA-256 与 canonical ZIP 完全一致。
平台选中了预期的 generic、Ascend、Enflame、Kunlun 四条路径；禁止重传。

00:30:03 CST 终态为 `completed` / `valid`、8/8、平均 `2.02248958x`，非
team best；额度确认为 `27/30`。Kunlun 原生 `tanh` 把目标芯从
`0.86508333x` 提升至 `0.97591667x`（+12.81%），但没有达到显著收益目标
`1.02508333x`。未改字节的 Huawei 同轮由 `0.88375x` 波动至 `0.70366667x`，
其余冻结芯片波动后使平均较 S4 下降 `0.02447917x`，未过基础晋级门：

| 芯片 | S5 speedup | 相对 S4 | 选中文件 |
| --- | ---: | ---: | --- |
| 天数 | 3.63900000x | +0.13% | `softcap_out.py` |
| 沐曦 | 1.68775000x | -0.66% | `softcap_out.py` |
| 燧原 | 1.18925000x | +0.43% | `softcap_out_enflame.py` |
| 海光 | 2.11850000x | -2.13% | `softcap_out.py` |
| 昆仑芯 | 0.97591667x | +12.81% | `softcap_out_kunlunxin.py` |
| 华为 | 0.70366667x | -20.38% | `softcap_out_ascend.py` |
| 国际通用 A | 3.07291667x | -2.51% | `softcap_out.py` |
| 国际通用 B | 2.79291667x | +0.00% | `softcap_out.py` |

结论：native `tanh` 的 Kunlun 单芯方向成立，但整题没有晋级且平台只认 8 芯
均值。保留 S4 为 Task24 team best，按预注册规则永久停止该轴；不以相同字节
赌测量波动，也不做 native-`tanh` 变体。

## S6：Enflame GCU 原生 `tanh`

状态：平台 8/8、`valid`、`2.05411458x`，团队当前最佳、公开第 13/14

S5 的停止门限定于 Kunlun/XPU vendor：本候选由 2026-08-27 00:31 CST 新同步
的同题逐芯榜单证据触发，是不同文件、compiler 和 libdevice 的 Enflame/GCU 新轴，
不是 S5 重传或变体。公开榜单 Enflame 前三为 `451.84558333x`、`15.5915x`、
`11.47741667x`，而 S4 仅 `1.18416667x`，显示手写 `exp` 路径之外存在数量级
更高的 GCU 路径。

S6 从 S4 best 分叉，只把 Enflame vendor 的分段 Taylor + `exp` 公式替换为
`triton.language.extra.gcu.libdevice.tanh`；BLOCK 4096、grid cap 12、四 warps、
cap 缩放和极小 cap 保护不变。generic、Ascend、Kunlun 与 S4 逐字节相同。
固定 FlagTree `c1ea8285a06e97afad9dd2644bc71f2efca072f4` 的 GCU libdevice
将 FP32 `tanh` 映射为 `__nv_tanhf`；固定 FlagGems `a7620cc191a0b42e040194622c5758b22a7a25dc`
的 GCU300 production `tanh`/GELU 路径也直接调用该 shim。

| 项目 | 值 |
| --- | --- |
| source / verification commit | `7c4ccc1ad6d52f00c24119806dd1c4a684a4f21f` |
| generic / Ascend SHA-256 | `e6ab1c434aa793bc58357e3d45d2eec7fd2ec56bebb65538b2a6049ca9a37ddc` / `bb98a5fda924e09954ce5778a859d064daaa560ebc7d49f6dbd1229dadabb50b` |
| Enflame / Kunlun SHA-256 | `84f553f2518f640a596f368d9eee6f49ace82c20eb7e61186fc3dead50d20bca` / `f34b06168ec951453601f552d3b6aa7bef1dac954a592226d3ac76ec248066bb` |
| 测试 SHA-256 | `821ef8e1e681e7cdf6b47d1ae58a39b0a346777e049eb027357bb9c25d2b8024` |
| screening | `gpu:/tmp/flagos-softcap-out-s6-enflame-tanh-screen.LZ7rLu`；PID/PGID `161284`；13/13、0.558s、`SCREENING_OK`；脚本/日志 SHA-256 `b47c1fc1a0422085446c66178782a10c71e00fd935451c3c40722fa9288c28a4` / `1495dde475dde12bce3f7be3c147a5a20e3152594b0163889d06c8cef893acb7` |
| 交替 A/B | 同目录；PID/PGID `161381`；脚本/日志 SHA-256 `13ab69779e0207e7ada59a7f5cfd35c9bb71d03b099ef8f46a0d1d997ce392a2` / `70aa62a79f845470bf1b38e9066b2a7760477a34ed6a3d98103d901a8d402809` |
| release | `gpu:/tmp/flagos-softcap-out-s6-enflame-tanh-release.lqSCXu`；PID/PGID `161544`；13/13、0.561s、`RELEASE_OK`；脚本/日志 SHA-256 `979e6048cf82bbea5b2e7e75f24127dddfb344f83e201709a466647a9faf6a30` / `fbac745f323994198ab14afa760241450f14ee61d2e0c597b5e284a2d3810b71` |
| source Git archive SHA-256 | `7063907f73c1f71b1dacb01aa455bc5841b439672d203bd462ba071d17543685` |
| canonical ZIP | `artifacts/competition/softcap_out/s6-7c4ccc1/softcap_out.zip`，10,023 bytes，SHA-256 `14de4b16762dffb63d14c99407d59d9dffce3707a4c6c4022b9b84a4a808df8d` |

RTX 5070 Ti 上 CUDA libdevice 与 GCU lowering 不同，非 control 几何平均仅
`0.739297x`、最差 `0.519232x`；该结果只说明不能把 CUDA 时延外推到 GCU。
candidate 最大 77 registers（baseline 93）、zero spill/scratch，且所有输出均按
题面容差对 PyTorch 参考通过。`dry-run`、`verified-existing` 与 `unzip -t` 全绿。

一次提交基础晋级门：8/8 valid、Enflame 高于 S4 `1.18416667x`、平均高于 S4
`2.04696875x`。显著收益门为 Enflame 至少 `1.34416667x` 且平均至少
`2.06696875x`（整题 `+0.02x`）；若冻结其余芯，要超过当前第 12 名均值
`2.18982292x`，Enflame 需约 `2.327x`。任一基础门不满足即保留 S4，并停止
Enflame native-`tanh` 轴。

2026-08-27 00:43:20 CST 在账号 `15600308080`、团队 `SoulCoder` 下执行 S6
唯一一次提交，submission `5213`、当日序号 `4`，额度预计由 `27/30` 变为
`26/30`。`file_url_sha256` 为
`4810f049fa6d82b99753d48d5275109838af35b6b739c938a336038e9ec55620`；对平台
返回的匿名对象限长回读为 10,023 bytes，SHA-256 与 canonical ZIP 完全一致。
平台选中了预期的 generic、Ascend、Enflame、Kunlun 四条路径；禁止重传。

00:43:54 CST 终态为 `completed` / `valid`、8/8，平均 `2.05411458x`，
平台标记 team best；额度确认为 `26/30`。Enflame 由 S4 的
`1.18416667x` 提升至 `1.38541667x`（+16.99%），通过单芯显著收益门；
整题较 S4 增加 `0.00714583x`，通过基础晋级门，但未达到预注册的
`2.06696875x` 整题显著收益门：

| 芯片 | S6 speedup | 相对 S4 | 选中文件 |
| --- | ---: | ---: | --- |
| 天数 | 3.66100000x | +0.73% | `softcap_out.py` |
| 沐曦 | 1.70966667x | +0.63% | `softcap_out.py` |
| 燧原 | 1.38541667x | +16.99% | `softcap_out_enflame.py` |
| 海光 | 2.04441667x | -5.55% | `softcap_out.py` |
| 昆仑芯 | 0.86383333x | -0.14% | `softcap_out_kunlunxin.py` |
| 华为 | 0.81816667x | -7.42% | `softcap_out_ascend.py` |
| 国际通用 A | 3.14566667x | -0.20% | `softcap_out.py` |
| 国际通用 B | 2.80475000x | +0.43% | `softcap_out.py` |

2026-08-27 00:45:48 CST 公开榜单刷新后，S6 仍为第 13/14；第 12 名为
`2.18982292x`，差 `0.13570834x`。结论：GCU native `tanh` 的目标收益真实，
但未复现榜首的数量级路径；保留 S6 为 Task24 team best，停止同一 native-`tanh`
轴，不以相同字节或微调赌平台波动。

## S7：Enflame full physical grid

状态：平台 8/8、`valid`、`2.00559375x`，非 team best；保留 S6

S7 从 S6 team best 分叉，只把 Enflame 物理 grid cap 从上游 pointwise 软件
策略的 12 提高到平台 launcher 已实证的 `grid.x` 上限 65535；BLOCK4096、
grid-stride、native `tanh`、四 warps、cap 缩放和极小 cap 保护不变。最大平台
用例 `N=65,667,072` 只需 16,032 个 program，低于硬件上限；generic、Ascend、
Kunlun 与 S6 逐字节相同。

| 项目 | 值 |
| --- | --- |
| source / verification commit | `fcbf4906a6f870aeb363f20f86392faf61386c9a` |
| generic / Ascend SHA-256 | `e6ab1c434aa793bc58357e3d45d2eec7fd2ec56bebb65538b2a6049ca9a37ddc` / `bb98a5fda924e09954ce5778a859d064daaa560ebc7d49f6dbd1229dadabb50b` |
| Enflame / Kunlun SHA-256 | `d4c3fef38460d0e1451577e87ce666bdd15201a3ffe0566234ee509f66f11fb2` / `f34b06168ec951453601f552d3b6aa7bef1dac954a592226d3ac76ec248066bb` |
| 测试 SHA-256 | `821ef8e1e681e7cdf6b47d1ae58a39b0a346777e049eb027357bb9c25d2b8024`（=S6） |
| screening | `gpu:/tmp/flagos-softcap-out-s7-grid-screen.sHS1ie`；13/13、0.564s；日志 SHA-256 `ee7f8510df9916f65883a17ebd2f4db2f90702f8aa8a952d31ab9f016e48c04c` |
| 交替 A/B | 同目录；5 组 AB/BA、warmup 25/rep 100；日志 SHA-256 `b8ab20e51157554832d6a94849567c5cadc496b90ce4baf5bb742e19b49f62d0` |
| release | `gpu:/tmp/flagos-softcap-out-s7-grid-release.Ci4rVT`；13/13、0.564s；日志 SHA-256 `ee7f8510df9916f65883a17ebd2f4db2f90702f8aa8a952d31ab9f016e48c04c` |
| source Git archive SHA-256 | `ec570b8c334b713cdc3fbbf1ce2ff9cac671177e511dc393e879ed1da9cb7381` |
| canonical ZIP | `artifacts/competition/softcap_out/s7-fcbf490/softcap_out.zip`，10,026 bytes，SHA-256 `b3c71c5f0bd90f103ae187e43b7a36deb64762ea1d599b0c52463c622425b898` |

RTX 5070 Ti 上 cap12↔full-grid 的 `(49,169)/(1,000,003)/(65,667,072)`
三 dtype 几何平均分别为 `1.211329/4.876166/4.163021x`；每个候选输出也按题面
容差逐元素通过参考。该结果只作为调度机制候选，不外推为 GCU 收益。

2026-08-27 00:58:33 CST 经实时门禁执行 S7 唯一一次提交，submission `5217`、
当日序号 `7`，额度由 `24/30` 变为 `23/30`。`file_url_sha256` 为
`596f3d8b7b8adb74b642d8723eade971959ffb50ca79954f5102d0b59d4a79b0`；匿名回读
10,026 bytes，SHA-256 与 canonical ZIP 完全一致。平台选中预期四条路径，
禁止重传。

00:59:06 CST 终态为 8/8、`valid`，平均 `2.00559375x`、非 team best；目标
Enflame 由 S6 `1.38541667x` 降至 `1.04616667x`（-24.49%）。其余芯片为天数
3.65741667x、沐曦 1.69733333x、海光 2.119x、昆仑 0.86316667x、华为
0.835x、国际 A 3.046x、国际 B 2.78066667x。结论：GCU 的 12-CTA 软件策略
贴近实际硬件并行度；CUDA 的 4–5x full-grid 信号不可迁移。保留 S6，永久停止
Enflame grid cap 轴。

## S8：Enflame 32K tile

状态：release 门禁通过，canonical ZIP 已冻结，待一次性平台提交

S8 从 S6 team best 分叉，只把 Enflame `BLOCK_SIZE` 从 4096 提高到 32768；
grid cap 仍为 12，native `tanh`、四 warps、grid-stride 和数值保护均不变。
这与固定 FlagGems `a7620cc191a0b42e040194622c5758b22a7a25dc` 的 GCU300
pointwise codegen 最大 tile 32K 对齐。S7 已否决的 full-grid 轴没有复用；generic、
Ascend、Kunlun 与 S6 逐字节相同。

| 项目 | 值 |
| --- | --- |
| source / verification commit | `b2a249b823f4b0301b63283d4b7b08c98ddb6897` |
| generic / Ascend SHA-256 | `e6ab1c434aa793bc58357e3d45d2eec7fd2ec56bebb65538b2a6049ca9a37ddc` / `bb98a5fda924e09954ce5778a859d064daaa560ebc7d49f6dbd1229dadabb50b` |
| Enflame / Kunlun SHA-256 | `7a1eee3c22be1df8cb5ac561152b981e45baaba8ccb3d677b6c47c07a9e7a2b6` / `f34b06168ec951453601f552d3b6aa7bef1dac954a592226d3ac76ec248066bb` |
| 测试 SHA-256 | `c6292e22084e0302f4591ff9cbaf3fd06bacd2825f43d03985a87f63a612381e` |
| release | `gpu:/tmp/flagos-softcap-out-s8-release.SzF7uL`；PID/PGID `163457`；14/14、10.561s、所有格式门禁及 `RELEASE_OK`；日志 SHA-256 `c98091d6e8698e6e6c238798b2a174197933cd52be15a02f21dfd97052829835` |
| canonical ZIP | `artifacts/competition/softcap_out/s8-b2a249b/softcap_out.zip`，10,025 bytes，SHA-256 `9fd897ad5b1e167c8c0a49826c295b2387b160b2d378ffacec61d74a6e469899` |

新增回归覆盖 `32767/32768/32769` 和 `12*32768+17`，远端 RTX 5070 Ti
14/14 通过；该卡只验证正确性，不把 CUDA 时延外推到 GCU。一次提交基础晋级门为
8/8 valid、Enflame 高于 S6 `1.38541667x` 且平均高于 `2.05411458x`；显著收益门
为 Enflame 至少 `1.54541667x`。任一基础门失败即保留 S6，并永久停止 Enflame
tile 轴，不尝试 64K、warps、grid 或数学变体。

2026-08-27 01:21:49 CST 经实时门禁执行 S8 唯一一次提交，submission `5222`、
当日序号 `12`，额度预计由 `19/30` 变为 `18/30`。`file_url_sha256` 为
`068ee11071f32b153e259693f637021533daa5bdf1fba15b24e7e5d78c04e151`；匿名对象
回读为 10,025 bytes，SHA-256 与 canonical ZIP 完全一致，四个成员均通过
`unzip -t`。平台已选中预期的 generic、Ascend、Enflame、Kunlun 路径；禁止重传。

01:22:31 CST 终态为 8/8、`valid`，平均 `2.24001042x`、team best，额度确认为
`18/30`。Enflame 从 S6 `1.38541667x` 提升至 `2.97566667x`（2.15 倍），整题
增加 `0.18589584x`，基础门和显著收益门均通过：

| 芯片 | S8 speedup | 选中文件 |
| --- | ---: | --- |
| 天数 | 3.59366667x | `softcap_out.py` |
| 沐曦 | 1.67191667x | `softcap_out.py` |
| 燧原 | 2.97566667x | `softcap_out_enflame.py` |
| 海光 | 2.04100000x | `softcap_out.py` |
| 昆仑芯 | 0.86433333x | `softcap_out_kunlunxin.py` |
| 华为 | 0.88166667x | `softcap_out_ascend.py` |
| 国际通用 A | 3.10975000x | `softcap_out.py` |
| 国际通用 B | 2.78208333x | `softcap_out.py` |

S8 已超过提交前公开第 12 名 `2.18982292x`。结论：GCU300 官方 32K 最大 tile
先验可迁移且产生显著收益；保留 S8 为 Task24 最终版本，按预注册规则关闭本题，
不再尝试 64K、warps、grid 或数学变体。

01:23 CST 公开榜单刷新后，S8 为第 `11/14`；上邻第 10 名 `2.4094375x`，
下邻第 12 名 `2.22059375x`。相较 S6 的第 13 名净升两位。

## S9：Enflame softcap 编译期常量

状态：commit-bound release 与 canonical ZIP 门禁通过，待唯一一次平台提交

S8 的停止结论只覆盖 tile、warps、grid 和数学变体。S9 由随后发现的固定
SGLang `8014d9d062c3cc5d393596ecdf2f7009191965df` 精确同题源码触发：其
`softcap_out_kernel` 把 `softcap_const` 声明为 `tl.constexpr`。S9 从 S8 team
best 分叉，只给 Enflame kernel 的同名参数增加该注解；32K tile、grid 12、
四 warps、native `tanh`、公式与数值保护全冻，generic、Ascend、Kunlun 和测试
逐字节保持 S8。

| 项目 | 值 |
| --- | --- |
| source / verification commit | `a9cf947d2e98ded6cac28e5ab98b29cb7e1d44ea` |
| generic / Ascend SHA-256 | `e6ab1c434aa793bc58357e3d45d2eec7fd2ec56bebb65538b2a6049ca9a37ddc` / `bb98a5fda924e09954ce5778a859d064daaa560ebc7d49f6dbd1229dadabb50b` |
| Enflame / Kunlun SHA-256 | `b759b4dfe1c64c86b4edd15a659d4ea520f5858a25968d239f4f4a8cb05d829a` / `f34b06168ec951453601f552d3b6aa7bef1dac954a592226d3ac76ec248066bb` |
| 测试 SHA-256 | `c6292e22084e0302f4591ff9cbaf3fd06bacd2825f43d03985a87f63a612381e`（=S8） |
| source Git archive SHA-256 | `158995a9914f46c1dd8d41a7de146a79a281122561f8978691463b06f85b2a7b` |
| screening | `gpu:/tmp/flagos-task24-constexpr-screen.RTHyBz`；14/14、格式门禁、A/B 与 `SCREENING_OK` |
| release | `gpu:/tmp/flagos-task24-constexpr-release.UvEIO3`；14/14、0.575s、格式门禁、A/B 与 `RELEASE_OK`；日志 mode 0600、SHA-256 `e7b957b077d86da5ee2b94d744e3d3cd4df8739a48a9cb8666466b04135a564b` |
| canonical ZIP | `artifacts/competition/softcap_out/s9-a9cf947/softcap_out.zip`，10,039 bytes，SHA-256 `453697383c8d4ee1e850a70f940f023c16205992e97ea003a445111247a2460d` |

RTX 5070 Ti 上 S8↔S9 的 15 点交替 A/B（5 组、warmup 25、rep 100）全部先按
题面 reference 验正；release 非 control 几何均值 `1.069032x`，最小点
`0.920392x`，最大平台规模 FP16/BF16/FP32 分别 `1.046218x`、`1.052428x`、
`1.173649x`。PTX 中 cap 从运行时 `ld.param.b32` 变为常量
`mov.b32 0f41F00000`（30.0），kernel 参数少一个；已存在的 spill 集合不扩大，
相同大 shape 的 FP32 spill 从 150 降至 144。该代理只证明常量专门化已生效，
不把 CUDA 倍数直接外推至 GCU。

2026-08-27 04:10 CST 实时榜单中 S8 为第 `11/14`、`2.24001042x`，第 10 名
已升至 `2.429125x`。若其余七芯不变，Enflame 需从 `2.97566667x` 升至严格
高于 `4.48858334x`（+50.85%）才升一名。一次提交基础门为 8/8 valid、Enflame
高于 S8 且平均高于 `2.24001042x`；显著门为 Enflame 高于 `4.48858334x`。
任一基础门失败即保留 S8，并永久停止 constexpr 轴；S9 字节只允许提交一次。

2026-08-27 04:13:35 CST 经实时 preflight 执行 S9 唯一一次提交，submission
`5299`、daily seq `26`，额度由 `5/30` 变为 `4/30`；`file_url_sha256` 为
`0e69afb5261305d03afd7a60b54e3f87224bdedc3d302d10cfa6891a6a4e786f`。对象
存储匿名回读为 10,039 bytes，SHA-256 与 canonical ZIP 完全一致，四个成员均
通过 `unzip -t`，平台选中预期四条路径。

04:14:23 CST 终态为 8/8、`valid`、平均 `2.22646875x`、非 team best；低于
S8 `2.24001042x`。唯一改动芯片 Enflame 仅由 `2.97566667x` 增至
`2.98191667x`（+0.21%），既未过基础平均门，也远低于升名所需
`4.48858334x`：

| 芯片 | S9 speedup | 选中文件 |
| --- | ---: | --- |
| 天数 | `3.58900000x` | `softcap_out.py` |
| 沐曦 | `1.67816667x` | `softcap_out.py` |
| 燧原 | `2.98191667x` | `softcap_out_enflame.py` |
| 海光 | `2.11525000x` | `softcap_out.py` |
| 昆仑芯 | `0.86450000x` | `softcap_out_kunlunxin.py` |
| 华为 | `0.81225000x` | `softcap_out_ascend.py` |
| 国际 A | `2.97800000x` | `softcap_out.py` |
| 国际 B | `2.79266667x` | `softcap_out.py` |

结论：编译期常量在 NVIDIA 代理的 6.9% 信号没有迁移到 GCU；保留 S8 team
best，永久停止 constexpr 轴，不重传或继续同类常量特化。
