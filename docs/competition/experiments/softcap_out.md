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
