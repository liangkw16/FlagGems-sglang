# Task 20 `mamba_layernorm_gated` 实验记录

## 契约

- 接口：`mamba_layernorm_gated(x, weight, bias, eps, z=None,
  group_size=None, norm_before_gate=True, is_rms_norm=True)`。
- 输入与输出为 `[M, N]`；`group_size=None` 等价于 N，否则 N 必须被
  `group_size` 整除。
- `norm_before_gate=False` 时先计算 `x * z * sigmoid(z)` 再归一化；为 True
  时先完成归一化及 affine，再乘 `z * sigmoid(z)`。
- 支持 RMSNorm 与 LayerNorm；输入、均值/方差、`rsqrt`、weight、可选 bias 和
  gate 全部按 FP32 计算，输出转换回 `x.dtype`，所有输入保持不变。
- 题面支持 FP16、BF16、FP32，容差分别为 `1e-2`、`1.5e-2`、`1e-4`；
  支持八类芯片，最低加速比为 0.1x。

固定参考为 SGLang `8014d9d` 的
`python/sglang/kernels/ops/attention/fla/layernorm_gated.py`。S0 只保留
1-pass forward 语义，删除 backward、SM count、PDL、NPU/CPU/XPU 分支、
autograd wrapper 和设备上下文。

## S0：generic row-group baseline

状态：S0 已打包并通过本地门禁；等待当次上传确认

验证时间：2026-08-24 01:28–01:37 CST

源码 commit：`f431ba4`

### 构建身份

| 项目 | 值 |
| --- | --- |
| 源文件 | `src/flaggems_sglang/ops/mamba_layernorm_gated.py` |
| 源文件 SHA-256 | `99c66cd49ad9e8f37ea8b087d63fd93fd3fab9c7602361b360a37e4dfaa6ca73` |
| 测试文件 | `tests/test_mamba_layernorm_gated.py` |
| 测试 SHA-256 | `1d66c36617728fd8b46b4ad2bdee0e53e75ca3a067652ab869e3f928723712a6` |
| ZIP | `artifacts/competition/mamba_layernorm_gated/s0-f431ba4/mamba_layernorm_gated.zip` |
| ZIP SHA-256 | `0bf5d8f26c6e3b3b827e2541bc58c058dc6b6fec05efe7bcff127492dfaedf76` |
| ZIP manifest | 顶层 `mamba_layernorm_gated.py`，4394 bytes |

### 唯一候选配置

- 每个 `(row, group)` 一个 Triton program；
  `BLOCK_SIZE=next_power_of_2(group_size)`。
- BLOCK 小于 2048 使用 4 warps，否则使用 8 warps；固定 `num_stages=1`。
- 显式使用 `x/z` 行列 stride 与 `weight/bias` stride，尾部完整 mask，输出为
  连续同 shape、同 dtype tensor；空输入直接返回。
- None 指针以已有合法 tensor 作为占位，实际读取由 constexpr 分支完全移除。
- 无 backward、autotune、vendor、设备判断、异常捕获或 PyTorch 计算 fallback。

### 正确性与静态检查

- 本地 `py_compile`、AST 解析和 79 字符行宽通过。
- 公开接口测试先于实现落盘。远端 RTX 5070 Ti 16 GB、driver 610.57.04、
  Python 3.12.13、PyTorch 2.13.0+cu130、Triton 3.7.1、CUDA 13.0、
  compute capability 12.0。
- 远端源码/测试 SHA-256 与本地一致；Black 79、isort、flake8 均通过。
- `tests/test_mamba_layernorm_gated.py -v` 为 2/2 unittest 方法通过，运行
  1.074 秒。内部 12 个组合覆盖三 dtype、RMS/LN、整维/分组、bias None/有、
  z None/有、门控前/后、非连续 `x/z/weight/bias`、输入不变和 hidden=259
  的非 power-of-two tail；另覆盖空输入。

### NVIDIA 代理性能

wrapper-inclusive；shape `[64, 4096]`；每个候选先做正确性检查和 JIT，再用
`triton.testing.do_bench(warmup=25, rep=100, quantiles=[0.5])`。

| dtype | 分支 | S0 p50 (ms) | Torch p50 (ms) | speedup |
| --- | --- | ---: | ---: | ---: |
| FP16 | RMS、整维、无 gate/bias | 0.006144 | 0.024672 | 4.016x |
| FP16 | LN、group 512、bias、后 gate | 0.006144 | 0.038048 | 6.193x |
| BF16 | RMS、整维、无 gate/bias | 0.006144 | 0.024672 | 4.016x |
| BF16 | LN、group 512、bias、后 gate | 0.006144 | 0.038912 | 6.333x |
| FP32 | RMS、整维、无 gate/bias | 0.006144 | 0.018528 | 3.016x |
| FP32 | LN、group 512、bias、后 gate | 0.008192 | 0.028768 | 3.512x |

六个编译产物均为 1 stage、4 或 8 warps、16 或 32 bytes shared memory、
0 global scratch；PTX 未出现 `ld.local`/`st.local`。最小代理 speedup 为 3.016x。

### 已知风险

- NVIDIA 代理不能证明其余七类后端正确或达到门槛。
- 单 program 保存整个 group；超过常见 8K group 后可能出现寄存器或本地内存
  压力。没有为未公开的大 group 推测两阶段实现。
- 二维 grid 的 y 轴等于 group 数；公开题面没有 shape 上界，极端小 group/大 N
  仍需隐藏 harness 或平台验证。
- ZIP 由 commit `f431ba4` 直接生成；`unzip -t`、UTF-8、单一 `.py`、10 MB、
  basename 和 ZIP 内源码哈希门禁均通过。没有平台提交授权，也未消耗额度。

## E1 拒绝与 E2 tail warp 候选

状态：E1 未过线；E2 已通过本地代理门禁并生成不可变 ZIP；未平台提交

验证时间：2026-08-24 03:24–03:39 CST

### 构建身份

| 项目 | 值 |
| --- | --- |
| source / verification commit | `345413d2e328ff3503f90af65a585b894869bf4d` |
| E2 源码 SHA-256 | `b7b81a64a9abfa5a9cf3d69e0ad066ae7dc6c1ed3aa5a34c900d811fc1fbc346` |
| E1 临时源码 SHA-256 | `b00f1e8874c10d4ab74ba2112dfb394e97de7817b40333a269cba68c3281addc` |
| 最终测试 SHA-256 | `b7bd60f3a2ce837b6cec9d2b2c7c7131c86a2434811d1e02a33994dce0cab474` |
| E2 ZIP | `artifacts/competition/mamba_layernorm_gated/e2-345413d/mamba_layernorm_gated.zip` |
| E2 ZIP SHA-256 | `78c56c2955981833242d9fc2ed13dca1373014fc49f12072d469f34987875f03` |
| ZIP manifest | 顶层 `mamba_layernorm_gated.py`，4463 bytes，成员 SHA 与 E2 源码相同 |
| S0 回滚 ZIP | `s0-f431ba4`，SHA-256 `0bf5d8f26c6e3b3b827e2541bc58c058dc6b6fec05efe7bcff127492dfaedf76` |
| 平台结果 | 未提交；逐芯结果、均值、排名和实时额度均为 N/A |

打包器从上述 commit 生成 4609-byte 规范 ZIP；创建后再次得到
`verified-existing`。`unzip -t/-l`、UTF-8、单一顶层 `.py`、10 MB、basename、
成员源码哈希和 ZIP 哈希门禁全部通过。

### Screening 回归与远端证据

以下证据均早于 source commit `345413d2e328ff3503f90af65a585b894869bf4d`，
按当前工作流只作为 screening；提交背书使用后文 commit-bound release。

新增第三个 unittest 方法，覆盖三 dtype ×
`group_size=1/255/256/257/511/512/513/1025`，包含 256/512 tail 和首个
2048-block/8-warp 转换；使用分组 LN、bias、前 gate 并逐项对题面 reference。
原 12 个语义组合、非连续四类输入、输入不变性、空输入和三 dtype 容差继续保留。

远端目录为 `gpu:/tmp/flagos-task20.02omgb`，mode 0700；环境仍是 RTX 5070 Ti
16 GB、driver 610.57.04、Python 3.12.13、PyTorch 2.13.0+cu130、Triton 3.7.1、
CUDA 13.0。源码和测试字节与 commit 相同；最终 S0 和 E2 均通过 py_compile、
Black 79、isort、flake8 和 3/3 unittest。

| 证据 | PID / 时间 | 脚本 SHA-256 | 日志 SHA-256 |
| --- | --- | --- | --- |
| 最终 S0 门禁 | `73519` / 03:30:54 | 仓库 unittest | `85d64604abecf5ab50b42e5165bacc64d4196306896f6eb85799e92d776a1273` |
| E1 门禁 | `73121` / 03:26:50 | 仓库 unittest | `e3b545a52e495cf168ce4e813f655eebcb522888db8372e6329f3d3ef6bc8c58` |
| E1 A/B | `73227` / 03:27:41 | `ab0033b70933eba3927ea146ec35a555146ca0f21672eef30421d02d2b3b9c0d` | `d274d13917a4a2a8aa97fdbb6f4c5836c35e59958abbf81ad70de32e4882cc91` |
| E2 门禁 | `73614` / 03:31:51 | 仓库 unittest | `667b3269398fb58d87059189ae76a2ad3dd6c077a82fc575f6fe094f3f8d60b5` |
| E2 高并行 A/B | `73744` / 03:32:21 | `0a5506b3c3b214dbb4503ce82b1f94c108c4edc6aa9dad8104355f10ada1acfb` | `39806dd056e6495c219313b624c1778836394f381d2e1aa3cec8f87e827b1648` |
| E2 资源 | `73902` / 03:33:49 | `0c63e6ad1cfa6ca6efcd052859ec96a082d4e19b6ab9da7c808a982ba1cc9465` | `2dd8de9b03ac5d632cd2e465ba6153d59eeb69495d8aaf2914d44a4baa1bdf42` |
| 大 group probe | `74015` / 03:37:22 | `00a4c2f6e7400bd845128330d0b590179b2ea2bed954cd5e97f2ca6d2033a0e2` | `6dc7c8a6e137e1348cc99d7e50a0aa57ede9f2f55b3f707fbb89720706d038a2` |
| E2 低并行 A/B | `74102` / 03:38:03 | `c843275943f2ca429b5014c6ca55c42835142f7e4f1673661cf44bfb06d79445` | `52edb650e1bcdfde76648e753482f12729926f28c523a8daf742081e3433c8a9` |

### 单变量实验

E1 只把全部 `BLOCK_SIZE=512` 变体从 4 warps 改为 2。27 点五组轮换 A/B
使用 `warmup=25, rep=100`；每点先对 reference 验证。目标点几何平均只有
`1.030229x`，全套 `1.016686x`，最差回退 `2.669%`，没有达到预设的
`>=1.05x`，因此拒绝。它只暴露出不满 512 的 `group_size=257` 三 dtype
一致获得约 1.13–1.15x。

E2 保留 S0 默认策略，仅对 `257 <= group_size < 512` 的 512-block 使用
2 warps；完整 512、256 和其他 block 均保持 S0。高并行集合为五组轮换、
21 个受影响点加 6 个 control：

| 指标 | 结果 |
| --- | ---: |
| 全 27 点 S0 / E2 几何平均 | `1.061458x` |
| 21 个受影响点几何平均 | `1.080317x` |
| control 几何平均 | `0.998006x` |
| 受影响点最差回退 | `0.890%` |
| control 最大偏移 | `0.476%` |

| group_size | 三 dtype 主分支 S0 / E2 几何平均 |
| ---: | ---: |
| 257 | `1.179013x` |
| 320 | `1.151706x` |
| 384 | `1.079529x` |
| 448 | `1.062594x` |
| 511 | `1.106206x` |

低并行补测覆盖 `M * ngroups=1/2`、group 257/384/511、三 dtype 共 18 个
受影响点。E2 与 control 几何平均分别为 `0.996430x` 与 `0.997107x`，相对
control 归一化约 `0.99932x`；最差回退 `1.998%`，处于 control 的 `1.252%`
波动量级，没有稳定回退。五轮轮换已经足以停止阈值搜索，不再做 E3 或 autotune。

资源抽查覆盖 group 257/384/511/512 × 三 dtype × S0/E2 共 24 个产物。S0
为 23–28 registers、shared 16 bytes；E2 为 24–40 registers、shared 8 或
16 bytes。两者均为 0 spill、0 global scratch 且 PTX 无 local load/store；
完整 512 在两者中均为 4 warps、24 registers、shared 16 bytes。

额外大 group probe 用最重的分组 LN + bias + 前 gate，覆盖三 dtype ×
`group_size=1025/2048/4096/4097/8192`，全部通过 reference。编译资源为
32–88 registers、shared 32 bytes，0 spill/scratch/local；最大实际验证 group
为 8192。大于 8192 的隐藏 shape 与极端二维 grid 仍未公开，继续作为平台风险。

固定 SGLang `8014d9d` 对 512-block 使用 2 warps，固定 FlagGems `ed2508b`
的六个国产后端也有合法 2-warp 实例，因此 E2 配置合法性风险较低；但八芯性能
仍必须由平台证明。原 S0 ZIP 保留为回滚。上传前必须重新读取实时额度，并针对
Task 20、E2 ZIP 的绝对路径和完整 SHA-256 取得当次确认；本记录不构成授权。

## E2 commit-bound release

状态：候选就绪，未提交平台

2026-08-24 20:09–20:15 CST 从 source / verification commit
`345413d2e328ff3503f90af65a585b894869bf4d` 的 Git 对象建立 fresh release；源码和
测试 SHA-256 分别为
`b7b81a64a9abfa5a9cf3d69e0ad066ae7dc6c1ed3aa5a34c900d811fc1fbc346`、
`b7bd60f3a2ce837b6cec9d2b2c7c7131c86a2434811d1e02a33994dce0cab474`，
与远端字节一致。目录
`gpu:/tmp/flagos-mamba-layernorm-gated-e2-release.BVy770` 为 mode 0700；环境仍为
RTX 5070 Ti 16 GB、driver 610.57.04、Python 3.12.13、PyTorch 2.13.0+cu130、
Triton 3.7.1、CUDA 13.0。ledger commit 为本节所在 commit。

| 证据 | 启动 / PID=PGID / wall limit | 脚本 SHA-256 | 日志 / SHA-256 |
| --- | --- | --- | --- |
| release 静态与 unittest | 20:09:23 / `101073` / 600s | `f7660a622e355f80be1a6c6edcbde6492d984d84623757925753ac07eb7e8850` | `release.log` / `af625c461d5705e90558872bfc19f926205a5e7d9341522a855a1b38c2af702b` |
| 补充高吞吐 A/B | 20:10:38 / `101201` / 900s | `c41af397ff69122957f3a84c5dcef1307be9238180456389f33ea955cca4d9d5` | `ab.log` / `4e030ad4330e327b69cc2be363e79ecfcf72ec3ee1e8c2ae28e086e2426276f3` |
| 固定 affected/control A/B | 20:12:56 / `101345` / 900s | `0a5506b3c3b214dbb4503ce82b1f94c108c4edc6aa9dad8104355f10ada1acfb` | `release-e2-ab.log` / `50fb83f4e7ca0b039ef1bf454e0384e7380cc830a56dd182952725520a0fc16f` |
| 资源 | 20:14:05 / `101459` / 600s | `0c63e6ad1cfa6ca6efcd052859ec96a082d4e19b6ab9da7c808a982ba1cc9465` | `release-resources.log` / `2dd8de9b03ac5d632cd2e465ba6153d59eeb69495d8aaf2914d44a4baa1bdf42` |
| 大 group | 20:14:55 / `101565` / 600s | `00a4c2f6e7400bd845128330d0b590179b2ea2bed954cd5e97f2ca6d2033a0e2` | `release-large-probe.log` / `6dc7c8a6e137e1348cc99d7e50a0aa57ede9f2f55b3f707fbb89720706d038a2` |

release 通过 py_compile、Black 79、isort、flake8 和 3/3 unittest（0.547s）。
GPU 无竞争 workload 时，固定脚本对 commit 字节做五轮 AB/BA：全体几何平均
`1.063577x`，affected `1.082205x`，control `1.000866x`；最差 affected 回退
`0.419%`，control 最大偏移 `0.655%`。各 group_size 跨三 dtype 的几何平均为：

| group_size | S0 / E2 几何平均 |
| ---: | ---: |
| 257 | `1.184073x` |
| 320 | `1.156675x` |
| 384 | `1.079199x` |
| 448 | `1.061655x` |
| 511 | `1.105799x` |

补充脚本 `flagos-mamba-e2-ab-r2.py` 使用 `M=64`、`ngroups=4`，affected
`group_size=257/384/511`、control `256/512`，覆盖三 dtype 与分组 LN + bias +
前 gate；affected / control 分别为 `1.000375x` / `0.998766x`。该补充集合上表现
中性，不改变固定 affected 集合的晋级结论。两个 A/B 日志均保留逐点五轮原始样本。
24 个资源产物复现 S0 23–28、E2 24–40 registers；均为 0 spill、0 scratch，
shared 不超过 16 bytes，PTX 无 local load/store。大 group 的 15 个 case 全部通过
reference，覆盖到 `group_size=8192`，32–88 registers、shared 32 bytes、0
spill/scratch/local。资源与大 group 脚本输出只含确定性结构化结果，fresh release
重跑后分别与 screening 日志逐字节同 SHA；新目录、启动时间和 PID 如上表。

release 后规范 ZIP 再次为 `verified-existing`：
`artifacts/competition/mamba_layernorm_gated/e2-345413d/mamba_layernorm_gated.zip`，
4609 bytes，唯一成员 `mamba_layernorm_gated.py` 4463 bytes，ZIP SHA-256
`78c56c2955981833242d9fc2ed13dca1373014fc49f12072d469f34987875f03`；成员哈希与
source commit 完全一致，`unzip -t` 通过。固定 21 点 affected 集合超过
`>=1.05x`，原 control 与补充高吞吐 affected/control 均超过 `>=0.97x` 无回退
门禁，且资源无 spill；E2 晋级首次平台提交候选，S0 作为回滚。

## E2 平台首投：7/8

账号 `15600308080`、团队 `SoulCoder` 于 2026-08-24 20:38:59 CST 提交 E2，
submission ID `4229`、当日序号 `8`；
20:53:09 CST 只读查询显示 `completed` / `invalid_correctness`，平均分和排名均为
N/A。提交前后额度由 `23/30` 变为 `22/30`。平台回传 ZIP 为 4609 bytes，SHA-256
`78c56c2955981833242d9fc2ed13dca1373014fc49f12072d469f34987875f03`，与本地 E2
不可变产物一致；规范化 URL 的 `file_url_sha256` 为
`748b2ab5a8c7263da69b3f62409f0907ecfe7875f6b273e78418c71d599c36db`。

| 芯片 | 结果 | 加速比 |
| --- | --- | ---: |
| 天数智芯 | 通过 | `9.3550x` |
| 沐曦 | 通过 | `3.3548x` |
| 燧原 | 通过 | `0.5106x` |
| 海光 | 通过 | `6.2808x` |
| 昆仑芯 | 通过 | `0.3628x` |
| card_a | 通过 | `7.4232x` |
| card_b | 通过 | `5.6172x` |
| 华为 | 失败 | N/A |

华为 case 8 选择 generic `mamba_layernorm_gated.py` 后建立二维 grid；Ascend 后端
将两维展平为总 `coreDim=rows*group_count=131072`，超过 `coreDim<=65535` 约束，
kernel 未启动。平台随后展示的数值无效，根因是启动网格越界而非算子数学错误。

## E3：Ascend capped grid-stride recovery

状态：平台 8/8，有效，团队当前最佳

E3 只新增 Ascend vendor，不改 generic 数学与 E2 tail-warp 策略。物理 grid 为
`min(total_groups, 4096)`，每个 program 按 `tl.num_programs(0)` 跨步遍历逻辑
group；因此 E2 失败规模的物理 `coreDim` 为 4096，同时完整覆盖 131072 个逻辑
group。

| 构建项 | 值 |
| --- | --- |
| source / verification commit | `374e06c05af98e24d151c6d5f178cd62dd2d82e6` |
| ledger commit | 本节所在 commit |
| generic SHA-256 | `b7b81a64a9abfa5a9cf3d69e0ad066ae7dc6c1ed3aa5a34c900d811fc1fbc346` |
| Ascend vendor SHA-256 | `07c87ed8f0e1a4f18ddcf2557c8b40d6c6829eca4279cf9a3c3bc561a4a9d4e6` |
| test SHA-256 | `768157f0b2814ca9b3768e9de901e5accc4eebe77b150a1439d2944ccf0e1fa2` |
| canonical ZIP | `artifacts/competition/mamba_layernorm_gated/e3-374e06c/mamba_layernorm_gated.zip` |
| ZIP size / SHA-256 | 9715 bytes / `afe450702c551fc83395432733dd22e98840125a31247c5e43117983ee30bb3d` |
| ZIP members | generic 4463 bytes；Ascend 4968 bytes |

测试同时加载 generic 与 Ascend vendor，覆盖既有 dtype、LN/RMS、bias、前后 gate、
非连续输入、空输入和边界；新增 BF16 `64 * 2048 = 131072` logical groups 的失败
规模回归。最终 release 从该 commit 的 Git 对象建立，目录
`gpu:/tmp/flagos-mamba-layernorm-gated-e3-release.dk5UUb`，mode 0700；环境为 RTX
5070 Ti 16 GB、driver 610.57.04、Python 3.12.13、PyTorch 2.13.0+cu130、Triton
3.7.1、CUDA 13.0。

| release 证据 | 启动 / PID=PGID / wall limit | 脚本 SHA-256 | 日志 SHA-256 |
| --- | --- | --- | --- |
| 静态与 unittest | 22:23:35 / `102729` / 600s | `d1d66769ce1a9f2ad22e5ced030e3ccce092162de0eaec120dfa372062950d71` | `543b061623e2c875090871f305ad8fbf9169302cd2dc9113eb1600ba1954ebad` |
| 三点性能与资源 | 22:23:49 / `102842` / 600s | `8564403a29c147d5f356769f7bf512b2cbb6fda36e22b577131d75f2a7e0086d` | `0f5424b8731c82c2703991ff0a76da4aabad6fb54ae561a248a1d30b5dfc529a` |

release 通过 py_compile、Black 79、isort、flake8 和 4/4 unittest（0.575s）。相对
Torch reference，故障规模的 wide-groups 与 many-rows 五轮中位数分别为
`0.498173x`、`0.461446x`，control 为 `5.012515x`。资源抽查：group 2 为 36
registers、0 shared；group 257 为 40 registers、8 bytes shared；group 8192 为
97 registers、32 bytes shared；三者均为 0 spill、0 global scratch、无 local PTX。

cap 单变量扫描在
`gpu:/tmp/flagos-mamba-layernorm-gated-e3-screen.UJO6UU` 启动于 22:17:43，
PID=PGID `102371`、wall limit 600s；脚本 SHA-256 为
`dcf204c8102ec06184209a075bc8a80fc5fb65af7a6d7d8720995725951e293f`，日志 SHA-256
为 `d6173b362f5473072c1a4966e338b8ba34629291ded6335f65acdf145a26f1c6`。相对 Torch
reference，结果为 48=`0.038828x`、256=`0.194729x`、1024=`0.365852x`、
4096=`0.496595x`、16384=`0.489792x`、32768=`0.459019x`、65535=`0.413324x`，
因此选择 4096。该 NVIDIA 扫描只能证明映射正确并排除明显代理回退，不能外推华为
性能；E3 的晋级目标是恢复华为启动与正确性，最终八芯结果仍以平台为准。ZIP 已通过
成员哈希和 `unzip -t` 验签。

## E3 平台结果：8/8

账号 `15600308080`、团队 `SoulCoder` 于 2026-08-24 22:34:19 CST 提交一次；
submission ID `4268`、当日序号 `9`，额度由 `22/30` 变为 `21/30`。远端对象为
9715 bytes，SHA-256
`afe450702c551fc83395432733dd22e98840125a31247c5e43117983ee30bb3d`，与本地
canonical ZIP 完全一致；`file_url_sha256` 为
`25964167ece218042f4cdaba5f4ed69ed68c873d331e0b95651281141205cd4b`。

| 芯片 | 选中文件 | 结果 | speedup |
| --- | --- | --- | ---: |
| 天数智芯 | `mamba_layernorm_gated.py` | 通过 | `9.3056x` |
| 沐曦 | `mamba_layernorm_gated.py` | 通过 | `3.4172x` |
| 燧原 | `mamba_layernorm_gated.py` | 通过 | `0.5090x` |
| 海光 | `mamba_layernorm_gated.py` | 通过 | `6.3094x` |
| 昆仑芯 | `mamba_layernorm_gated.py` | 通过 | `0.3360x` |
| 华为 | `mamba_layernorm_gated_ascend.py` | 通过 | `1.8838x` |
| card_a | `mamba_layernorm_gated.py` | 通过 | `6.7256x` |
| card_b | `mamba_layernorm_gated.py` | 通过 | `5.5342x` |

22:35:10 CST 终态为 `completed` / `valid`，8/8 通过，平均 `4.2526x`，平台标记
为团队当前最佳。华为确实选中 Ascend vendor，E2 的 `coreDim=131072` 启动失败已
消除并达到 `1.8838x`；其余七芯继续使用未变 generic。公开榜单
`as_of=2026-08-24T22:35:37.794558+08:00` 显示 SoulCoder 第 6/6，参与 8 队、
达标上榜 6 队，榜首 EvokeAgent 为 `6.5554x`。保留 E3 作为团队最佳，不再沿同一
Ascend grid-cap 假设试参；后续额度转其他算子。

## E4：Enflame 小 group 多行 tile

状态：commit-bound release、canonical ZIP 与一次性提交门禁已完成

2026-08-27 03:54 CST 实时榜单中，本队 E3 为第 `7/8`、`4.2526x`；第 6 名
`5.84695x`。逐芯差距最大的燧原仅 `0.509x`，而榜首该芯为 `12.056x`。固定
FlagGems commit `ed2508bcb5a03000e9774734201d840ba362cd11` 的 GCU300/400
LayerNorm 对 `N <= 128` 使用约 1024 元素预算的二维多行 persistent kernel；固定
SGLang commit `8014d9d062c3cc5d393596ecdf2f7009191965df` 也保留多行 forward
结构。E4 只新增 Enflame vendor，generic 与 E3 Ascend 文件逐字节不变。

Enflame 在 `group_size <= 128` 且逻辑 group 总数至少 4096 时，取
`TILE_M=min(next_power_of_2(rows), ceil(1024/BLOCK_SIZE))`，二维 grid 继续把
group 作为独立轴，保证每组 weight/bias 切片正确；其他 shape 调用与 E3 generic
逐行 kernel 相同。平台已知 case 8 为 `rows=64, group_count=2048,
group_size=2`，因此物理 program 数从 131072 降到 2048。没有修改公式、dtype、
stride、warps、stages 或其他芯片路径。

| 项目 | 值 |
| --- | --- |
| source / verification commit | `98cb62bef14b0bb914e65e6ca5854366c2a3ceba` |
| generic / Ascend SHA-256 | `b7b81a64a9abfa5a9cf3d69e0ad066ae7dc6c1ed3aa5a34c900d811fc1fbc346` / `07c87ed8f0e1a4f18ddcf2557c8b40d6c6829eca4279cf9a3c3bc561a4a9d4e6` |
| Enflame / test SHA-256 | `8c469806538b6ae4d400120a35711692bdf91e85d3e0227aa88d1c07f91e80ff` / `09a8e3992fe0961b63ed2382d61397d951158fda28e023a7131106e40b7d60d9` |
| release | `gpu:/tmp/flagos-task20-enflame-multirow-release.BsSPcJ`；PID/PGID `173029`；4/4、`RELEASE_OK` |
| release script / log SHA-256 | `d3b23b531e22c9946f5769c9904021f1a7c32f854de22fff4fa8325a9102c8e3` / `b1f157240081c4074c6df1c0f1fcf0a06c50f97dfad9adbfc84e7179f1b9a0d8` |
| source Git archive SHA-256 | `df4dda8dc639c38a232423fb2ee096c1e08025dd545bd1336a4c18787ed21805` |
| canonical ZIP | `artifacts/competition/mamba_layernorm_gated/e4-98cb62b/mamba_layernorm_gated.zip`，17410 bytes，SHA-256 `40a80fcafc213f6bd4f84903f3b1f0b612e23efca18602b1e92b6d371a7f641b` |

release 从 source commit 的 Git objects 独立展开；源码、测试和 archive 哈希与
screening 使用字节一致。远端 RTX 5070 Ti 上通过 py_compile、Black 79、isort、
flake8 与 4/4 unittest；覆盖三 dtype、RMS/LN、gate 前后、bias/z 可空、非连续
stride、空输入和平台 case 8。五轮 AB/BA 的受影响集合几何平均为
`3.052146x`、最差 `1.498438x`；case 8 的 FP16/BF16/FP32 分别为
`7.0000/7.0000/7.4286x`，两个未命中 control 均为 `1.0000x`。candidate
资源为 16–40 registers、最多 256 bytes shared、0 spill、0 global scratch。
NVIDIA 只能证明结构与调度收益，不能外推真实 GCU 倍数。

ZIP 三个成员与 source commit 一致，`dry-run`、`verified-existing`、UTF-8、
10 MB 与 `unzip -t` 门禁全绿。E4 只允许一次平台提交；基础晋级门为 8/8 valid、
Enflame 高于 E3 `0.509x` 且平均高于 `4.2526x`，显著目标为 Enflame 至少
`1.018x`。任一基础门失败即保留 E3，不重传相同字节。
