# Task 24 `softcap_out` 实验记录

## S0：generic baseline

状态：平台首轮完成，6/8 通过；S1 vendor 修复中
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

状态：本地代理验证通过，等待人工确认后上传比赛平台

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

### 仍需平台证明

- CUDA 代理不能证明 Ascend 的 operand-order workaround；若 S1 仍丢失 cap
  缩放，下一候选才把 cap 改为 `tl.constexpr`。
- Enflame 在最大用例中每个 CTA 循环约 21,376 次；先恢复正确性，若低于
  `0.1x`，保持 grid 12 后单变量比较 BLOCK 1024 和 4096。
- 本 ZIP 尚未上传；继续保留至少两次最终回归额度。
