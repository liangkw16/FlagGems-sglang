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

状态：平台 submission `5210` 评测中

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
