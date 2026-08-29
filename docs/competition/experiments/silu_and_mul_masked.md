# Task 39 `silu_and_mul_masked` 实验记录

## 契约

- 签名：`reference(input, masked_m)`。
- `input` 为 BF16 `[E,T,H]`，`masked_m` 为整数 `[E]`，输出 BF16
  `[E,T,H//2]`。
- 仅计算每个专家前 `masked_m[e]` 行的
  `gate.float() * sigmoid(gate.float()) * up.float()`；填充行不验值。
- BF16 容差 `atol=rtol=1.5e-2`；支持八芯，核心计算必须走 Triton。

## S0：KernelGen 基线

状态：已提交，终态 `invalid_correctness`，7/8 通过，仅昆仑失败。

- 2026-08-30 通过 `kernelgen-server.generate_kernel` 生成两轮：首轮报告
  `41.188x` 但 65535 grid cap 后无步进，静态否决；第二轮补齐步进但嵌套
  `@triton.jit` 且代理仅 `0.123x`，静态否决。
- 随后调用 `optimize_kernel`，产物加入 autotune 并再次丢失 grid cap，静态
  否决。按 KernelGen 重试门禁停止继续生成。
- S0 仅保留第二轮的单 pass 算法，复用 T29 已八芯验证的固定
  `BLOCK_COL=1024`、capped 1D grid + kernel 内 grid-stride 骨架；无
  autotune、vendor 分支、fallback 或 host `.item()`。
- 输出用 `torch.empty`，因为填充行不参与正确性判定；有效行完整写入。
- 回归覆盖 int32/int64 mask、0/1/T-1/T、1024 列尾块、三维非连续输入、
  空维、NaN/Inf/极值、输入与 mask 不变性，以及
  `total_blocks > 2 * 65535` 的三轮折叠路径。

### Screening 身份与正确性

| 项目 | 值 |
| --- | --- |
| base commit | `14d5133` |
| source SHA-256 | `bdafd313c6bb841a3334eca33e7bd1637c110d5edbf2c0180c00b127820c9cad` |
| test SHA-256 | `7037aac3c9f8410542b08acd65d70482d6f531855a49505ebabbb4c690ea5c85` |
| 远端目录 | `gpu:/tmp/flagos-silu-and-mul-masked.NeO5UW`，mode 0700 |
| screening 日志 SHA-256 | `3411ebf555fb25c3c5a2c086ae90aab848e6bad4135cadf70e2027d277133733` |

远端 RTX 5070 Ti 16 GB、Python 3.12.13、PyTorch 2.13.0+cu130、
Triton 3.7.1：py_compile、Black 79、isort 80、flake8 全过，unittest
**5/5** 通过，远端复验 source/test 哈希不变。

特殊值审查曾提出把负半轴改成稳定 sigmoid；代理反例实证题面原样的
`torch.sigmoid(float32)` 对 `-92/-90` 返回 0，半指数改写反而分别输出
约 `-3.46/-25` 而不匹配 oracle，故回退到与题面参考一致的
`gate / (1 + exp(-gate))`。诊断目录
`gpu:/tmp/flagos-silu-numeric.fV6zJu`，日志 SHA-256
`a5fc99a10e9b835c2fc1bd03979e309c89f8a1959fd807d14e452461e429b6ae`。

### NVIDIA 代理性能

目录 `gpu:/tmp/flagos-silu-and-mul-masked.u0iZ1U`；benchmark 脚本
SHA-256 `854c4df1ad3764c1367ab9ab744c86307cfb163fe4c6244f6a78c81d465e76b5`，
完整 AB/BA 原始样本日志 SHA-256
`00a1efcf08231deba2cc9afae11009db2b040fae7bebe881c95f02486c388a96`。
运行前 GPU 无竞争进程；wrapper-inclusive、5 轮交替、p50：

| E×T×half | BLOCK 512 | BLOCK 1024 | BLOCK 2048 | BLOCK 4096 |
| --- | ---: | ---: | ---: | ---: |
| 8×64×512 | 25.21x | **26.19x** | 25.74x | 25.73x |
| 64×128×2048 | 43.14x | 61.13x | **82.00x** | 49.50x |
| 256×64×3584 | 32.14x | **32.27x** | 31.99x | 31.81x |

最大 input 224 MiB；S0 选择 BLOCK 1024：两组最优、最大 shape 与 2048
持平，且同骨架已有 T29 八芯实证。2048 只作为平台逐芯结果后的单变量候选。
NVIDIA 结果仅为代理证据，不外推八芯。

### Release 与不可变 ZIP

| 项目 | 值 |
| --- | --- |
| source/verification commit | `bd5bf8b040b934797a7686bddef06b0093dc3481` |
| release 目录 | `gpu:/tmp/flagos-silu-release.K5JLHZ`，mode 0700 |
| release 日志 SHA-256 | `fd37e5cc7d057a68f73d2b44a5c8418f86e22c5402134d45e9f70d3e7e0d85f0` |
| ZIP | `artifacts/competition/silu_and_mul_masked/s0-bd5bf8b/silu_and_mul_masked.zip` |
| ZIP SHA-256 | `cc9da72e2ad6c551aeba2eac74dbe2d7882d3f489b285601223b926f2f9815e0` |
| ZIP 内容 | 顶层 `silu_and_mul_masked.py`，2708 bytes；ZIP 2850 bytes |

release 文件全部由 Git 对象生成；source/test 与 screening 哈希一致。lint、
unittest 5/5、末尾哈希复核全过。BLOCK 1024 的 release speedup p50 为
25.88x / 61.31x / 32.28x，与 screening 一致。打包器 create 后
`--verify-existing` 通过，actual 与 canonical ZIP SHA-256 一致，单成员、
普通 UTF-8 Python 文件、远低于 10 MB。

### 已知边界

- 题面参考隐含 `H` 为偶数；奇数宽度时参考自身 gate/up 维度不匹配。
- grid-stride 采用与既有八芯实现相同的 int32 tile 归纳变量；仅在接近
  `2^31` tiles 或单行 `H >= 2^31` 的非常规不可分配 shape 存在理论索引
  溢出，不为该题合理 MoE shape 增加全 int64 成本。

## 平台记录

### S0 首投（sub 6584，2026-08-30 00:53 CST）

- 实时 preflight tuple 全部匹配；当时额度 28/30，单次 confirm 成功，
  提交后剩 27/30。file URL SHA-256
  `8c9524d870052214bce8f264290a6baa438467a8388be0522932ba69b3151576`。
- state `submitted`，远端 ZIP 验签因未配置已信任对象存储 hostname 为
  `unavailable`；不改变已提交事实，按门禁未重试。
- 终态 `invalid_correctness`，**7/8 通过，仅昆仑数值失败**：

| 芯片 | speedup | 状态 |
| --- | ---: | --- |
| tianshu | 24.5783x | 通过 |
| muxi | 16.4183x | 通过 |
| enflame | 0.4620x | 通过 |
| haiguang | 34.2437x | 通过 |
| kunlunxin | - | 正确性失败 |
| huawei | 6.8197x | 通过 |
| card_a | 31.7577x | 通过 |
| card_b | 10.2353x | 通过 |

昆仑 case 0 为 45/48 元素错（最大绝对差 6.1074），case 1 为
13150/16384 元素错（最大绝对差 7.03125）；不是门槛或编译失败。

## E1：昆仑去 metadata gating 单变量修复

状态：screening 通过，待 commit 后 release。

`tl.fdiv` 候选在 commit 前否决：固定 FlagTree commit `c1ea8285` 的
`tensor.__truediv__` 和 `tl.fdiv` 对 FP32 最终都调用
`builder.create_fdiv`。RTX 5070 Ti 的 Triton 3.7.1 实编译也证明 int32、
int64 mask 两组 generic/vendor TTIR 去除源码定位后逐字相同，均只有
`arith.divf`；IR 目录 `gpu:/tmp/flagos-silu-ir.vlH2Pf`。该改写不产生
不同程序，故没有消耗平台额度。

E1 保持 generic S0 字节不变；Kunlun vendor 与 generic 的完整计算 diff
只有删除 `expert_id/token_id/masked_m` gating，并把 store mask 收窄为
`cols < half_width`。BLOCK 1024、capped grid-stride、row/col 地址、SiLU
公式、类型和 launch 参数全部不变。题面不检验 padding 输出，写满全部行
合法；同时直接排除标量 metadata load/broadcast 或谓词误判造成有效行未写。
T29 E5 已在昆仑实证同一 row/col-block 骨架正确，因此不先改 flat 结构。

- screening：`gpu:/tmp/flagos-silu-kunlun-gating-e1.w6kQOE`，mode 0700；
  generic SHA-256 `bdafd313c6bb841a3334eca33e7bd1637c110d5edbf2c0180c00b127820c9cad`，
  vendor SHA-256 `c8ba71d893d07a2380ab8c9ab79b09b03d0f6d6c65634bc80fd9060cee754d59`，
  test SHA-256 `776debdf846d79f04c14c58e195c4c36f98bfc86a2867a5122a764a646fec89a`；
  最终日志 SHA-256
  `cc9f3c2a1d3b629a2e1470fa5674657ce43071410179a1358361ca8de8866c13`。
- RTX 5070 Ti 上 py_compile、Black 79、isort 80、flake8 全过，unittest
  5/5。grid-fold 回归改为约 0.5 MiB 的 `512x257x2`，generic/vendor 都覆盖
  `131584 > 2 * 65535` 的三轮折叠；另覆盖 int32/int64、尾列、非连续输入、
  特殊值和输入不变性。NVIDIA 只能验证字节可编译及语义，昆仑平台是必要
  证伪步骤。

### E1 release 与不可变 ZIP

| 项目 | 值 |
| --- | --- |
| source/verification commit | `e126063be7a6295c540e02f526f8de0599f9c3d5` |
| release 目录 | `gpu:/tmp/flagos-silu-kunlun-e1-release.ommhz0`，mode 0700 |
| release 日志 SHA-256 | `277d3d8977403fd0612c1ab2a1fd24a85d04bcc9f4b29488e88091367292e1ed` |
| ZIP | `artifacts/competition/silu_and_mul_masked/e1-e126063/silu_and_mul_masked.zip` |
| ZIP SHA-256 | `886f04d53dc2fa2958d89767c11ec73bd9e006b8e339c91caf1af911bcd1558f` |
| ZIP 内容 | 顶层 generic 2708 bytes + `kunlunxin` 2541 bytes；ZIP 5531 bytes |

release 文件由 commit 的 Git 对象生成，三文件哈希与 screening 完全一致；
py_compile、lint、unittest 5/5 和末尾哈希复核全过。打包器 create 后
`--verify-existing` 通过，actual 与 canonical ZIP SHA-256 一致。

### E1 平台终态（sub 6587，2026-08-30 01:07 CST）

实时 preflight tuple 全匹配，单次 confirm 成功；file URL SHA-256
`f4d817745bd9a12001bbd6fe00f9061de7917a5d60ba82abfb99d91ea31677d6`，
提交后额度 26/30。远端 ZIP 验签仍因未配置可信对象存储 hostname 为
`unavailable`，提交 state 已为 `submitted`，未重试。

**八芯 correctness 全过**，证明 S0 昆仑失败来自 metadata gating；但昆仑
`0.052x < 0.1x`，终态 `invalid_threshold`，展示平均 `15.66275x`：

| 芯片 | speedup | 状态 |
| --- | ---: | --- |
| tianshu | 24.3957x | 通过 |
| muxi | 16.6023x | 通过 |
| enflame | 0.4657x | 通过 |
| haiguang | 34.6750x | 通过 |
| kunlunxin | 0.0520x | 正确，未过 0.1x 门槛 |
| huawei | 6.5243x | 通过 |
| card_a | 32.4377x | 通过 |
| card_b | 10.1493x | 通过 |

下一候选只改 Kunlun 性能轴；generic 与其余七芯继续冻结。优先在不恢复已证伪
scalar gating 的前提下减少写满 padding 的额外工作，先保持 BLOCK 不变验证
flat/调度结构，达到 0.1x 后再做逐芯冲榜。

## E2：昆仑 flat-full BLOCK 1024

状态：screening 通过，待 commit 后 release。

E2 保持 E1 已验证的 mask-free、写满 padding、SiLU 公式、BLOCK 1024 和
65535 grid cap，只把“一行一个 1024-lane tile”改为 flat output-element
grid-stride。对 `half_width=128` 的形态，E1 仍在每行执行 1024 lanes 的
exp/div，flat 理论可把约 8 行装进一个 tile；恢复门槛只需
`0.1 / 0.052 = 1.923x`。T29 已在昆仑平台证明同一 flat scaffold 正确并比
row/col-block 快 65%。按该实证模板使用 int32 flat 索引，避免每 tile 的
1024-lane int64 地址乘法；已知题面规模远低于 int32 元素上限。

- 最终 screening：`gpu:/tmp/flagos-silu-kunlun-flat-e2-final.QEwAQv`，
  mode 0700；generic SHA-256
  `bdafd313c6bb841a3334eca33e7bd1637c110d5edbf2c0180c00b127820c9cad`，
  vendor SHA-256
  `2698072998829ead430005697c2262bd2dc8712e9ee4d221d833541b01a72462`，
  test SHA-256
  `776debdf846d79f04c14c58e195c4c36f98bfc86a2867a5122a764a646fec89a`，
  日志 SHA-256
  `322ac62661b6243a5291f0c3a17bf9583288e75f56ccea302f44242b0a9dcde3`。
- py_compile、Black 79、isort 80、flake8 和 unittest 5/5 全过；三轮
  grid-fold、尾块、非连续、特殊值和输入不变性均覆盖 flat vendor。
- NVIDIA wrapper-inclusive 代理在 `4x3x16`、`4x64x256`、
  `288x768x256` 上相对 E1 分别为 1.010x、0.991x、1.016x；该后端没有
  Kunlun 的尾 lane 执行成本，结果中性，不作为否决依据。E2 不同时改
  BLOCK 2048；若平台仅差少量，再单独调常量。
