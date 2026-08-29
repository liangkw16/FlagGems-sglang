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

状态：平台 8/8 correctness，`invalid_threshold`、`15.66275x`。

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

状态：平台 8/8、`valid`、`15.91945833x`。

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

### E2 release 与不可变 ZIP

| 项目 | 值 |
| --- | --- |
| source/verification commit | `f87989599fe621ac03f8e8d46600ba207802b2c7` |
| release 目录 | `gpu:/tmp/flagos-silu-kunlun-e2-release.qnH9CU`，mode 0700 |
| release 日志 SHA-256 | `27201b39f405aef24cc840fdfacfc08f98d4e5999e42f029df432357dfa49049` |
| ZIP | `artifacts/competition/silu_and_mul_masked/e2-f879895/silu_and_mul_masked.zip` |
| ZIP SHA-256 | `43c3e08dfd2795132bb095775031b603ee48bb686a35b4cdff0ccc7656cc657b` |
| ZIP 内容 | 顶层 generic 2708 bytes + `kunlunxin` 2276 bytes；ZIP 5266 bytes |

release 文件由 commit 的 Git 对象生成；三文件哈希与 screening 相同，
py_compile、lint、unittest 5/5 与末尾哈希复核全过。canonical ZIP create
和 `--verify-existing` 均通过。

### E2 平台终态（sub 6588，2026-08-30 01:15 CST）

实时 preflight tuple 全匹配，单次 confirm 成功；file URL SHA-256
`8390b7af5e4bf31bd1bd30a78bbd3fbdc3c1a27e90008aa618b8b113888e008f`，
提交后额度 25/30。远端对象存储验签仍为 `unavailable`，state 已是
`submitted`，未重试。

终态 **valid、8/8、当前团队最佳**，平均 `15.91945833x`。昆仑由 E1 的
`0.052x` 提至 `0.241x`（4.63 倍），flat packing 假设兑现并越过 0.1x
门槛：

| 芯片 | speedup | 状态 |
| --- | ---: | --- |
| tianshu | 24.5400x | 通过 |
| muxi | 16.6513x | 通过 |
| enflame | 0.4550x | 通过 |
| haiguang | 34.6500x | 通过 |
| kunlunxin | 0.2410x | 通过 |
| huawei | 7.9120x | 通过 |
| card_a | 32.3667x | 通过 |
| card_b | 10.5397x | 通过 |

相对当时榜首 `19.2431x` 仍差约 `3.324x` 平均。E2 已建立有效锚点；后续
每轮冻结 generic 和其余 vendor，优先以 host-resolved mask + 同一 flat
kernel 只处理 `sum(masked_m)` 有效行，避免重新引入 device-side gating。

## E3：AMD 四档列 tile autotune

状态：平台 8/8、`valid`、`16.16041667x`，团队当前最佳。

E3 从 E2 team best 分叉，只新增 AMD vendor；generic 和 Kunlun source
逐字节冻结。AMD 默认 elementwise 路径的固定 `1024/8w` 被保留为保底，
另加入 `128/2w`、`256/4w`、`512/8w`，autotune key 仅为
`half_width`。四档来源是 T21 `moe_sum_reduce` 的 AMD 平台实证族，不把
该算子的 `+53.1%` 外推成本题保证；这里只复用已验证的 launch policy，
减少隐藏 `half_width=8/128` 时固定 1024 lanes 的尾浪费。

grid 从所选 config 的 `BLOCK_COL` 动态计算，kernel 内同步重算
`num_col_blocks/total_blocks`，保留 capped grid-stride、metadata gating、
SiLU 公式和地址逻辑；launch 不显式重复绑定 BLOCK/warps。

- screening：`gpu:/tmp/flagos-silu-amd-e3.YfVVNq`，mode 0700；generic
  SHA-256
  `bdafd313c6bb841a3334eca33e7bd1637c110d5edbf2c0180c00b127820c9cad`，
  Kunlun SHA-256
  `2698072998829ead430005697c2262bd2dc8712e9ee4d221d833541b01a72462`，
  AMD SHA-256
  `a662c81024ad41eb9cf6bbbdf55c83bebdd681da3d27e219609856ae5074429f`，
  test SHA-256
  `2a15893c25da97a5c5a27d3ac6b83f76f1c0e47a2a31cd1d640206624155ae81`，
  unittest 日志 SHA-256
  `b1df0b57a41ba4896d7e5bbb6f3be1c14cc7bb4538d1d47bd0cd5301eaaf5754`。
- 本地 py_compile、Black、isort、flake8、diff-check 全过；远端 unittest
  5/5，AMD 覆盖 h=8/128、h=129/1025、int32/int64、非连续输入、特殊值和
  `> 2 * 65535` grid-fold。
- RTX 5070 Ti 五轮交替 A/B 在 `4x3x16`、`4x64x256`、
  `288x768x256` 的中位比为 `1.0006x`、`0.9972x`、`1.1073x`，几何平均
  `1.0338x`；日志 SHA-256
  `daf0bd77b92f08df6d0780a704b21268fe5ba0b7378aae0c86b2927221504056`。
  该代理只排除明显回退，AMD 平台仍是必要证伪步骤。

### E3 release 与不可变 ZIP

| 项目 | 值 |
| --- | --- |
| source/verification commit | `9cee390cce4fc17582c77278857f1442585ec99f` |
| release 目录 | `gpu:/tmp/flagos-silu-amd-e3-release.08TwCx`，mode 0700 |
| release 日志 SHA-256 | `2bff6418bc820849e24334a07d2334742de894724b13abfc634aef17ce0c1fa2` |
| ZIP | `artifacts/competition/silu_and_mul_masked/e3-9cee390/silu_and_mul_masked.zip` |
| ZIP SHA-256 | `520232a36872ccd82b554666a0075bf3d22ff577b1c7ebfe76907c0de0806649` |
| ZIP 内容 | 顶层 generic 2708 bytes + AMD 3050 bytes + Kunlun 2276 bytes；ZIP 8444 bytes |

release 四文件由 commit 的 Git 对象生成，前后哈希与 screening 一致；
py_compile、unittest 5/5 和 `RELEASE_OK` 全过。canonical ZIP create 与
`--verify-existing` 一致，新 AMD vendor 已确认进入三成员归档。

### E3 平台终态（sub 6591，2026-08-30 01:27 CST）

实时 preflight tuple 全匹配，单次 confirm 成功；file URL SHA-256
`95421daf1bdfe72b4aaa483983b35a7444ba88717c5c46762129d3ef4ed5f600`，
提交后额度 24/30。远端对象存储验签仍因未配置可信 hostname 为
`unavailable`，state 已为 `submitted`，未重试。

终态 **valid、8/8、团队新最佳**，平均 `16.16041667x`，比 E2 提高
`0.24095834x`。AMD 路由确认只作用于国际 B，得分从 `10.53966667x`
提高到 `10.94333333x`（`+3.83%`）；四档方向有效，但没有外推 T21 的
53.1% 收益。其余七芯字节冻结，分数变化视为平台波动：

| 芯片 | speedup | 相对 E2 | 选中文件 |
| --- | ---: | ---: | --- |
| tianshu | 24.8080x | +1.09% | generic |
| muxi | 16.4883x | -0.98% | generic |
| enflame | 0.4570x | +0.44% | generic |
| haiguang | 35.2043x | +1.60% | generic |
| kunlunxin | 0.2407x | -0.14% | Kunlun |
| huawei | 7.3993x | -6.48% | generic |
| card_a | 33.7423x | +4.25% | generic |
| card_b | 10.9433x | +3.83% | AMD |

E3 是当前 team best，后续 vendor 从该 commit 分叉并冻结已有 AMD/Kunlun
成员；不再继续同一 AMD 四档轴，下一轮优先高基数芯片的直接 vendor 证据。

## E4：NVIDIA 四档列 tile autotune

状态：平台 8/8、`valid`、`15.76016667x`，非 team best；保留 E3。

E4 从 E3 team best 分叉，只新增 NVIDIA vendor；generic、AMD 与 Kunlun
逐字节冻结。NVIDIA 文件与 E3 已平台验证的 AMD 文件逐字节相同，复用
`128/2w`、`256/4w`、`512/8w`、`1024/8w` 四档和
`key=["half_width"]`。T18 已完整实证竞赛后缀路由为国际 A=`_nvidia`、
国际 B=`_amd`；本轮因此只改变国际 A。

- screening：`gpu:/tmp/flagos-silu-nvidia-e4.VOLP0J`，mode 0700；generic
  SHA-256
  `bdafd313c6bb841a3334eca33e7bd1637c110d5edbf2c0180c00b127820c9cad`，
  Kunlun SHA-256
  `2698072998829ead430005697c2262bd2dc8712e9ee4d221d833541b01a72462`，
  AMD/NVIDIA SHA-256 均为
  `a662c81024ad41eb9cf6bbbdf55c83bebdd681da3d27e219609856ae5074429f`，
  test SHA-256
  `86b9cb782754a4faa845d6c8b009dc276dde55b1b9d2b6f080cad7aa2b622ea8`，
  日志 SHA-256
  `3eadcdd9181603777bd8c753c0c200bec61675b9f81658257abab312e14514d6`。
- 本地 py_compile、Black、isort、flake8、diff-check 与 AMD/NVIDIA
  byte-identity 断言全过；远端 unittest 5/5，四条 runtime 路径均覆盖。
- 因 NVIDIA 与 AMD 源码完全相同，直接复用 E3 的 RTX 五轮交替 A/B：三形态
  中位比 `1.0006x`、`0.9972x`、`1.1073x`，几何平均 `1.0338x`；
  E3 平台又在国际 B 实证 `+3.83%`。两项都不保证国际 A 收益，平台仍是必要
  证伪步骤；stop gate 为 8/8 valid、国际 A 选中 NVIDIA 且国际 A 与平均均
  高于 E3。

### E4 release 与不可变 ZIP

| 项目 | 值 |
| --- | --- |
| source/verification commit | `e2153fa4d666d4bb4ec151e1e8f27a7f8387ce3c` |
| release 目录 | `gpu:/tmp/flagos-silu-nvidia-e4-release.kax5gk`，mode 0700 |
| release 日志 SHA-256 | `a93a3c14663a62f286de1308c5748e28e8d5479ed91eed5cc579ae8cab0d9b3a` |
| ZIP | `artifacts/competition/silu_and_mul_masked/e4-e2153fa/silu_and_mul_masked.zip` |
| ZIP SHA-256 | `f55d676416af5a158c9d091c00b853020aac8e6f3204c1b8bf6661611e48fac2` |
| ZIP 内容 | generic 2708 bytes + AMD/NVIDIA 各 3050 bytes + Kunlun 2276 bytes；ZIP 11628 bytes |

release 五文件由 commit 的 Git 对象生成，前后哈希一致；py_compile、
unittest 5/5、AMD/NVIDIA `cmp` 和 `RELEASE_OK` 全过。canonical ZIP create
与 `--verify-existing` 一致，NVIDIA 新成员已进入四成员归档。

### E4 平台终态（sub 6594，2026-08-30 01:37 CST）

实时 preflight tuple 全匹配，单次 confirm 成功；file URL SHA-256
`501937baafbd9688b4648017aa509f453a9c31925fc5f19c121cdd0623f560a3`，
提交后额度 23/30。远端对象验签为 `unavailable`，提交 state 已为
`submitted`，未重试。

终态 8/8、`valid`，但平均 `15.76016667x`，比 E3 低 `0.40025x`，非
team best。国际 A 正确选中 NVIDIA，但从 E3 generic 的 `33.74233333x`
降至 `33.178x`（-1.67%），未过单芯与整题 stop gate；永久停止 NVIDIA
四档轴，后续候选从 E3 分叉且不携带 `_nvidia`：

| 芯片 | E4 speedup | 相对 E3 | 选中文件 |
| --- | ---: | ---: | --- |
| tianshu | 24.3433x | -1.87% | generic |
| muxi | 16.5140x | +0.16% | generic |
| enflame | 0.4683x | +2.48% | generic |
| haiguang | 34.5873x | -1.75% | generic |
| kunlunxin | 0.2370x | -1.52% | Kunlun |
| huawei | 6.3463x | -14.23% | generic |
| card_a | 33.1780x | -1.67% | NVIDIA |
| card_b | 10.4070x | -4.90% | AMD |

## E5：program-uniform padding row skip

状态：平台 8/8、`valid`、`15.672x`，非 team best；保留 E3。

E5 从 E3 team best 分叉，删除 E4 已证伪的 NVIDIA vendor；AMD 与 Kunlun
逐字节回到 E3。generic 只把原本合入每个 lane mask 的
`token_id < valid_rows` 提升为 program-uniform `if`，把 loads、exp/div、
cast 和 store 包在有效行分支内；BLOCK 1024、row/col 映射、grid cap、
grid-stride、公式与地址全部不变。不能对无效行 `return`，因为同一 physical
PID 跨 grid stride 后可能继续处理有效行。

S0 昆仑失败分母与代理 shape 共同锁定前两 shape 均为 50% 有效密度：
`4x3xhalf8` 比较 48/96 元素，`4x64xhalf128` 比较 16384/32768 元素。
旧 mask 只阻止无效行内存事务，仍执行 1024 lanes 的 exp/div；uniform `if`
直接跳过这半数 row program。为隔离 GCU 动态 loop 分支兼容风险，新增
Enflame vendor，内容与 E3 generic 逐字节相同；因此候选只改变天数、沐曦、
海光、华为和国际 A 五芯。

- screening：`gpu:/tmp/flagos-silu-row-skip-e5.y15eYt`，mode 0700；generic
  SHA-256
  `d5a85a335c5701e2fe03f62295180075272cdc1a20a56c37dc3c2eaf17073fb5`，
  Enflame SHA-256
  `bdafd313c6bb841a3334eca33e7bd1637c110d5edbf2c0180c00b127820c9cad`
  （=E3 generic），AMD/Kunlun SHA-256 仍为
  `a662c81024ad41eb9cf6bbbdf55c83bebdd681da3d27e219609856ae5074429f` /
  `2698072998829ead430005697c2262bd2dc8712e9ee4d221d833541b01a72462`；
  test SHA-256
  `6daba54aaed95ca014204d9cc113ae0aeba837e98d7b3404a920f147fddc5969`，
  unittest 日志 SHA-256
  `8171a83c7b05d26a5376a40a4a20be17787fcfe1fdb292e3ded69def58e5f867`。
- 本地 py_compile、Black、isort、flake8、diff-check 全过；远端 unittest
  5/5。回归新增实际 h=8/128 的 int32/int64 50% 密度；grid-fold 显式断言
  PID 0 先处理 invalid block 0，再处理 valid block 65535，防止误用 return。
- RTX 五轮交替 wrapper A/B 在 50% 密度三形态的中位比为 `1.0196x`、
  `0.9912x`、`1.5527x`，几何平均 `1.1621x`；all-valid controls 为
  `0.9999x`、`0.9747x`、`0.9758x`，几何平均 `0.9834x`。日志 SHA-256
  `e33502d81238a637f6e3bc106061a214d9560925682b35e564eedbe7da83cdff`。
  候选通过 affected `>=1.05x`、control `>=0.98x`、control 单点
  `>=0.96x` 门；小 shape 收益仍接近噪声，平台五芯是必要证伪步骤。

### E5 release 与不可变 ZIP

| 项目 | 值 |
| --- | --- |
| source/verification commit | `85def8d1e8902ffd2bdadbe39b03292a0616a62b` |
| release 目录 | `gpu:/tmp/flagos-silu-row-skip-e5-release.vOqrsb`，mode 0700 |
| release 日志 SHA-256 | `d7006fc70de150ca52e5acdb693311d9e8482b3dab27ae2a4501dac82af8d15a` |
| ZIP | `artifacts/competition/silu_and_mul_masked/e5-85def8d/silu_and_mul_masked.zip` |
| ZIP SHA-256 | `fd95c9f3a21ed7d2df7a88157785ad660c2b0796051bcac0ffd9e7456afa9e7e` |
| ZIP 内容 | generic 2782 bytes + AMD 3050 bytes + Enflame 2708 bytes + Kunlun 2276 bytes；ZIP 11362 bytes |

release 五文件由 commit Git 对象生成，前后哈希一致；py_compile、
unittest 5/5 和 `RELEASE_OK` 全过。canonical ZIP create 与
`--verify-existing` 一致，E4 NVIDIA 成员已移除，Enflame 保险成员已进入
四成员归档。

### E5 平台终态（sub 6596，2026-08-30 01:42 CST）

实时 preflight tuple 全匹配，单次 confirm 成功；file URL SHA-256
`bd45b186c51d2c616fe235cca6fca36fbcd4fd3517bb604402e7603c3e2b763d`，
提交后额度 22/30。远端对象验签为 `unavailable`，提交 state 已为
`submitted`，未重试。

终态 8/8、`valid`，但平均 `15.672x`，比 E3 低 `0.48841667x`，非
team best。五颗受影响芯片全部未超过 E3；平台短 row/高 launch-overhead
形态中 runtime branch 税高于跳过 padding ALU 的收益。Enflame 保险路由正确，
AMD/Kunlun 也保持专用路径。永久停止 uniform-if 轴，后续从 E3 原 generic
分叉：

| 芯片 | E5 speedup | 相对 E3 | 选中文件 |
| --- | ---: | ---: | --- |
| tianshu | 23.3353x | -5.94% | generic-if |
| muxi | 16.4313x | -0.35% | generic-if |
| enflame | 0.4547x | -0.51% | Enflame E3 byte |
| haiguang | 34.1330x | -3.04% | generic-if |
| kunlunxin | 0.2413x | +0.28% | Kunlun |
| huawei | 7.1050x | -3.98% | generic-if |
| card_a | 32.8997x | -2.50% | generic-if |
| card_b | 10.7757x | -1.53% | AMD |

## E6：Ascend BLOCK 512

状态：screening 通过，待 commit 后 release。

E6 从 E3 team best 分叉，generic、AMD 与 Kunlun 逐字节恢复 E3，并移除
E5 Enflame 保险成员。只新增 Ascend vendor；它与 generic 的完整计算 diff
只有 `_BLOCK_COL = 1024` 改为 `512`，其余 capped grid-stride、metadata
gating、公式、地址和默认 launch 全部相同。平台固定先例为 T24 同类 pointwise
Ascend 256→512 使华为 `+19.90%`，T21 reduction 为 `+35.47%`；反例 T08
为 `-0.73%`，因此只验证 512，不扩 sweep。

- screening：`gpu:/tmp/flagos-silu-ascend-e6.EfOZUO`，mode 0700；generic
  SHA-256
  `bdafd313c6bb841a3334eca33e7bd1637c110d5edbf2c0180c00b127820c9cad`，
  Ascend SHA-256
  `5129f38abe9ce1b6eadd669da1f905ddc1a8bc17e5ec057263e9c1150f9603e4`，
  AMD/Kunlun SHA-256 仍为
  `a662c81024ad41eb9cf6bbbdf55c83bebdd681da3d27e219609856ae5074429f` /
  `2698072998829ead430005697c2262bd2dc8712e9ee4d221d833541b01a72462`；
  test SHA-256
  `8c680d08c079aa1f17805e04a7bd34301bfb051736a9ed16b6c5c86787d47578`，
  unittest 日志 SHA-256
  `ccadf2d37d27f52ee9393ffafd1f8ff95db7b1a0537502667ec12fb6dab8a652`。
- 本地 py_compile、Black、isort、flake8、diff-check 和单行 source diff
  断言全过；远端 unittest 5/5，覆盖 h=8/128、`2*512+1` 跨块尾、两种
  mask dtype 与三轮 grid-fold。
- RTX 五轮交替 wrapper A/B 三形态中位比为 `0.9943x`、`1.0165x`、
  `1.4693x`，几何平均 `1.1409x`；日志 SHA-256
  `d386964c49847aec836c1e04932e2c6a1c147491aad93702622f73f98cc43121`。
  NVIDIA 结果只排除明显回退；平台 stop gate 为 8/8 valid、华为选中 Ascend、
  华为高于 E3 `7.39933333x` 且平均高于 `16.16041667x`。
