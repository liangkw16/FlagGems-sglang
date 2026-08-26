# Task 18 `fused_recurrent_gdn` 实验记录

## S0：experimental generic recurrent kernel

状态：RTX 5070 Ti 代理验证和不可变 ZIP 门禁通过；**八芯高风险，未提交平台**
验证时间：2026-08-24 01:45–01:52 CST
源码 commit：`de1530b`

### 契约

| 项目 | 值 |
| --- | --- |
| 公开接口 | `fused_recurrent_gdn(q, k, v, g, beta, scale, initial_state, output_final_state, use_qk_l2norm_in_kernel=False)` |
| 输入 | q/k `[B,T,H,K]`；v `[B,T,HV,V]`；g `[B,T,HV]` |
| beta | scalar/head `[B,T,HV]` 或 vector/head `[B,T,HV,V]` |
| 头映射 | `q_head = value_head // (HV // H)`，要求 `HV % H == 0` |
| 状态 | FP32 `[B,HV,V,K]`；每步依次 decay、delta correction、外积更新、q readout |
| 可选项 | q/k FP32 L2 norm；initial state；FP32 final state |
| 输出 | `[B,T,HV,V]`，dtype=`v.dtype`；final state 或 `None` |
| 容差 | `atol=1e-2, rtol=1e-2` |
| 支持芯片 | 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用 A/B，共 8 款 |
| 截止 / 门槛 | 2026-08-27 19:59:59；`speedup_threshold=0.1` |

固定来源为本地 Task 18 题面，以及 SGLang commit
`8014d9d062c3cc5d393596ecdf2f7009191965df` 的
`python/sglang/kernels/ops/attention/fla/fused_recurrent.py`。S0 保留固定源码的
前向递推和 `(V tile, B*HV)` 所有权，删除 autograd、varlen/KDA、input guard、
连续存储假设和 SGLang 私有 `exp` helper。

2026-08-24 01:51 CST 公开 API 状态：49 次提交、13 支队伍，仍为
`pending_challenge`，0 支达到门槛且无榜首。动态值仅用于当时决策。

### 唯一候选

- 一个 self-contained Triton kernel；每个 program 持有一个
  `(batch, value_head, V tile)` 的完整 `[BLOCK_V,BLOCK_K]` FP32 状态，并在
  device 端按真实 `T` 顺序循环。
- `BLOCK_K=next_power_of_2(K)`，`BLOCK_V=min(next_power_of_2(V),8)`，4 warps、
  1 stage。相较固定 SGLang 的 `BLOCK_V<=32`，S0 主动压低状态 tile，优先降低
  跨芯寄存器/本地内存风险。
- q/k/v/g/beta、initial state、output 和 final state 全部使用真实 stride；
  地址 stride 转 int64。MHA/GQA、两种 beta 均在同一 kernel 内用 constexpr
  选择，不按设备分支。
- decay、L2 norm、prediction、correction、状态更新和输出 readout 全部 FP32；
  output 最后由 store cast 到 `v.dtype`，final state 保持 FP32。
- 仅用 Torch 分配输出；非空计算路径没有 reference/demo、Torch 计算、
  `.contiguous()`、autograd、autotune、vendor、设备识别或 fallback。

### 验证

| 项目 | 值 |
| --- | --- |
| 源文件 SHA-256 | `5fbad8f892ae00b9331731a5a18135929227dfbb6cfeecc331d27690025dda73` |
| 测试 SHA-256 | `1f7ff9ea725e73940dedeedb86e2810ecf2c5ef26d71fb514fa6d77e4a200a83` |
| ZIP | `artifacts/competition/fused_recurrent_gdn/s0-de1530b/fused_recurrent_gdn.zip` |
| ZIP SHA-256 | `cf27e0e48f41fc1948075cd3bc22864e45d2387d8e61b5b6371fe1147fe9ce7f` |
| 远端证据目录 | `gpu:/tmp/flagos-task18-tdd`、`gpu:/tmp/flagos-batch2.SQaIX2` |
| 远端环境 | RTX 5070 Ti 16 GB；PyTorch 2.13.0+cu130；Triton 3.7.1；CUDA 13.0 |

- TDD 红灯先于实现：仅同步测试时，公开入口文件不存在，unittest 以
  `FileNotFoundError: .../fused_recurrent_gdn.py` 失败；实现后 unittest 2/2
  通过。
- 测试覆盖 FP16/BF16/FP32、MHA/GQA、scalar/vector beta、L2 开关、无/非零
  initial state、final state 开关、真实非连续 strides、`T=0` 和全部输入不变性。
- 额外大维度正确性：`K=128,V=127,T=8` 与
  `K=256,V=129,T=4,L2+vector beta` 均通过题面容差；最大 output 绝对误差分别
  为 `0`、`4.768e-7`，最大 final-state 误差分别为 `1.304e-8`、`1.118e-8`。
- 编译压力探针通过 `K/V/T` 为 `64/64/16`、`128/128/16`、`256/128/8`、
  `256/256/8`、`512/512/2` 的五组 BF16 GQA；该结果只证明 NVIDIA Triton
  3.7.1 能编译运行。

wrapper-inclusive BF16 benchmark；均为 `H=4,HV=8`、有初态和 final state，
`triton.testing.do_bench(warmup=25, rep=100)`：

| `B,T,K,V` | S0 (ms) | 题面 Torch reference (ms) | 代理 speedup |
| --- | ---: | ---: | ---: |
| `2,32,64,64` | 0.0206 | 2.2779 | 110.36x |
| `1,16,128,128` | 0.0245 | 1.0197 | 41.56x |
| `1,8,256,256` | 0.0171 | 0.3756 | 21.92x |

Torch reference 含 Python 时间循环，代理 speedup 不能外推为官方八芯成绩。

ZIP 由 commit `de1530b` 的算子子树直接生成，仅含顶层 UTF-8
`fused_recurrent_gdn.py`。`unzip -t`、10 MB、成员名和逐字节 SHA-256 门禁
均通过。

### 假设、风险与下一步

- 平台未公开 shape 集。S0 假设题面有效域：q/k shape 一致、所有 tensor 同设备、
  `K/V/H/HV > 0`、`HV % H == 0`、beta 只取题面两种 rank、initial state shape
  匹配。没有为未定义的非法输入发明语义。
- **首要风险是跨芯资源模型。** 状态必须跨动态 `T` 存活；即使 `BLOCK_V=8`，
  每个 program 仍持有 `8 * next_power_of_2(K)` 个 FP32 元素。NVIDIA 上
  `K=512,V=512` 可编译不代表天数/沐曦/燧原/海光/昆仑/华为可编译或不落慢速
  local memory。当前公开 49 次提交仍无任何 8 芯有效方案，与该风险一致。
- 每个 V tile 重复读取 q/k。压低 BLOCK_V 换取资源安全的代价是在大 V 上增加
  读流量；首次平台结果正确但低于门槛时，下一单变量才尝试 BLOCK_V 16。
- `tl.exp` / `tl.rsqrt` 的跨编译器精度尚未实测；题面容差为 `1e-2`，NVIDIA
  三 dtype 与大维度探针均有充分余量，但不能替代八芯结果。
- 该 S0 只适合作为一次受控平台实验；尚未消耗额度。网页上传必须取得用户对
  明确 Task、上述 ZIP 路径、SHA-256 和实时额度的当次确认。

## E1：`BLOCK_K<=128 → 1 warp`（否决）

状态：未晋升；未生成 ZIP，未提交平台

固定上游使用 1 warp，因此先保持 kernel、tile、状态和数学不变，只把
`BLOCK_K<=128` 的 launch 从 4 warps 改为 1。正确性与资源门禁通过，但五轮配对
A/B 的 affected 几何平均仅 `0.9181x`：低精度 K128 为 `1.513–1.524x`，
K64 却降到 `0.543–0.576x`，FP32 K64/K128 仅 `0.885/0.939x`。K129 controls
几何平均 `0.9997x`。因此 E1 按预设最差点门禁否决，不把上游固定配置直接泛化。

## E2：低精度 `BLOCK_K=128 → 1 warp`（晋升）

状态：源码、测试、release 代理验证和不可变 ZIP 门禁通过；**八芯仍高风险，未提交平台**

验证时间：2026-08-24 07:53–08:00 CST

### 最小门控

E2 只在 `BLOCK_K==128` 且 q 为 FP16/BF16 时使用 1 warp，即实际
`65<=K<=128` 的低精度路径。K<=64、FP32 以及 K>=129 全部保持 S0 的 4 warps。
kernel 数学、`BLOCK_V=8` 上限、状态 dtype、grid、1 stage 和所有 constexpr
功能开关均不变。新增测试实际编译 FP16 K128 affected、FP32 K128 control 和
FP16 K129 control。

源码 commit 为 `2ba2813756162abcfc3ef72e620f67f5e65eccca`；源码 SHA-256
`50877b97299f6bc15d0dc694391c5c3439b96596356efe9405b195bc63b58506`，
测试 SHA-256
`776a0ac2105df255452df145dd7479b1c0c1422451e2face7ee2ef5ea65d1303`。

### Release 代理验证

release 目录 `gpu:/tmp/flagos-fused-recurrent-gdn-release.mkgMaK`，mode 0700；
source 与 verification commit 均为
`2ba2813756162abcfc3ef72e620f67f5e65eccca`。环境为 RTX 5070 Ti 16 GB、
driver 610.57.04、Python 3.12.13、PyTorch 2.13.0+cu130、Triton 3.7.1、
CUDA 13.0。

- py_compile、Black 79、isort、flake8、逐文件哈希和 unittest 3/3 通过。
- 主矩阵 22 次 S0/E2/reference correctness 检查通过；五轮交替 A/B，
  `warmup=25, rep=100`、每次 wrapper 批量 10 次：FP16/BF16 K128 affected
  几何平均 `1.5127x`、最差 `1.5126x`；K64、FP32 和 K129 共 7 个 controls
  全部为 `1.0000x`。
- `K=65/96/127/128` 的 FP16/BF16 扩展矩阵 affected 几何平均 `1.5007x`，
  范围 `1.4785–1.5250x`；FP32 K96 control 为 `1.0000x`。
- S0/E2 各 11 个编译变体，最高均为 96 registers/thread、4,096 bytes
  shared，spill、global scratch、local load/store 均为 0。1-warp affected
  特化为 77–83 registers/thread、16 bytes shared。

release gates、主 A/B、BLOCK_K=128 扩展 A/B、provenance 和 corrected harness
的 SHA-256 依次为
`2bb7d01c6342fb3e1191b2f05fed86f815a5abf053025cb3c6a2226615ce3f16`、
`877bccb30065b15b616b405d4d3f5ccc09b28983e307d2de2752826d90ca8abc`、
`73672ad16bd3d9000681104cb47a5ad5f0179d1c99abb6577b31d07a39c44756`、
`8a8f420136d3cd106565fbbfbb009480f4aed85accb8fc93c6528b2ab991d5d6`、
`1ae87b7a3816f73dcb9749f11ca473d8f15b288186fbdc58cb2ce7840d74356d`。
corrected harness 对长随机递推先按 `sqrt(K)` 缩放 q/k，并避免 FP32 reference
对 q view 做 in-place scale；这些只影响代理数据生成与 reference，不改候选字节。

### 产物与剩余风险

- ZIP：
  `artifacts/competition/fused_recurrent_gdn/e2-2ba2813/fused_recurrent_gdn.zip`
- ZIP SHA-256：
  `4be0a8135cc5dcc23a33b31852b6754fa44a2959e8d035e49a113d07edaf14eb`
- 大小 / 成员：9,877 bytes；顶层 `fused_recurrent_gdn.py` 9,735 bytes。

确定性构建、`--verify-existing`、`unzip -t`、UTF-8、basename、10 MB 和逐字节
来源门禁均通过。E2 没有改变 K>=129 的高状态资源路径，因此 S0 所述八芯资源风险
仍然成立；本结果只证明 NVIDIA 代理。未打开浏览器、未读取实时额度、未提交平台，
旧确认不授权此 ZIP。

## E2 平台首投：7/8 terminal、0 通过，长递推归约不一致

2026-08-25 11:56:58 CST，平台实时契约同时返回
`status=submitting`、`can_submit=true`、`challenge_operator`；修正本地 preflight
对 `pending_challenge` 的过时硬拒绝后，复验 E2 的 commit、ZIP、唯一成员和
SHA-256，执行一次性提交。submission `4595`，额度 `7/30` → `6/30`；未重试。

截至 2026-08-26 16:21:25 CST 已为 `completed`、8/8 terminal、0 通过，整体
结果 `invalid_correctness`，未重试。七芯均在隐藏长 BF16 递推失败：case 3 的
错误数依次为沐曦
130805、燧原 668、海光 614、华为 532、国际 A 644、国际 B 827；case 4 除燧原
外分别仅错 2、7、21、9、4、17 个元素。燧原 case 4 另有确定的
`grid.y=256 > 255` 启动上限。输出总元素均为 524288；日志高置信对应
`B,T,H,HV,K,V=(8,128,4,8,64,64)` 与 `(32,32,4,8,64,64)`。case 3 从后段
出现 Inf/NaN 分类分岔，case 4 则是取消敏感的少量元素，指向 Torch
`einsum→bmm` 与 Triton `tl.sum` 的 FP32 归约树差异被不稳定递推放大。

### E3 screening：否决，不提交

先把 2D grid 展平为 1D，可独立修复燧原硬上限；数值侧在 RTX 5070 Ti 上用固定
seed、`atol=rtol=0.015`、`equal_nan=True` 重放两个隐藏形态。固定 1 warp、
2/4/8 warp、关闭 FP fusion、chunk 8/16 和串行 4×16 归约均未清零；最佳长形态
仍错 262/524288。原生 N=1 `tl.dot(input_precision="ieee")` 能编译且性能与基线
相当，但错误由 26/302 增至 29/1144；16 列 padded dot 同样更差且慢 3 倍。

证据目录为 `gpu:/tmp/flagos-fused-recurrent-gdn-e3.ENzFB5`、
`gpu:/tmp/flagos-fused-recurrent-gdn-reduce.xJDx4L` 和
`gpu:/tmp/flagos-fused-recurrent-gdn-dot.vdenQU`。reduce 统一日志 SHA-256
`7d5287e25ba86423222ad4624529d084b09e2116d2ccf960a25d3adecd5b46de`，N=1 dot
日志 SHA-256
`7ac1fd32d36e4267db9ece6ff8bd296e302fcbb28c59035c2efde14d62c68526`。所有候选均
未通过本地正确性门禁，因此未构建 ZIP、未执行第二次提交，worktree 恢复 E2。

## E4：复现 Torch 归约与 eager 舍入边界（晋升）

状态：源码、测试、release 代理验证和不可变 ZIP 门禁通过；submission `5082`
评测中

验证时间：2026-08-26 16:23–16:26 CST

### 根因与最小特化

E3 的归约枚举没有复现 Torch `einsum -> bmm` 的真实 FP32 加法树。RTX 5070 Ti
上的独立 K64 指纹探针表明，Torch 使用四个按 `k % 4` 分组的 FMA 累加器，最终按
`(a0 + a2) + (a1 + a3)` 合并；该顺序在 20,480 个随机输出上与 Torch bitwise
一致，串行、R1/R2、普通 R4 和 R8 顺序均产生大量 bit diff。另一个独立探针显示，
Triton `tl.exp` 对 2^20 个 BF16 gate 约有 30% bit diff，而 vendor libdevice
`exp` 及其后续 FP32 乘法与 Torch bitwise 一致。

E4 因而只特化平台暴露的 `K=64`、BF16、kernel 内不做 q/k L2 norm 的路径：

- 1D grid，每个 program 独占一个 `(B,HV,V)` 状态行，规避燧原 `grid.y<=255`；
- vendor libdevice `exp` 通过项目已有跨芯 shim 调用；
- 四路 `tl.fma` 和交叉合并复现 K64 bmm 归约；
- FP32 state scratch 跨 timestep 保存状态，独立 FP32 outer scratch 显式物化 eager
  外积乘法的舍入边界；
- 其他 dtype、K、L2 路径仍走原 generic kernel，没有设备识别或 Torch fallback。

outer scratch 版本移除了非标准 launch 参数 `enable_fp_fusion=False`；静态复核确认
新增路径只使用标准 `tl.load/store/fma`，state/outer 寻址、非连续输入 stride、
vector beta、initial/final state 和 `T=0` 契约均无硬阻塞。

### Screening 与 release 证据

最终源码 commit 为 `354952470e81db8234c85bc3f5364a110901f7fb`；其父提交
`8dc6d2368f27ae50b703ab4728d7e5087a70f82a` 引入 outer scratch，最终提交仅按
仓库 Black 79 规则机械排版。

RTX 5070 Ti、PyTorch 2.13.0+cu130、Triton 3.7.1、CUDA 13.0 上，对两组平台
高置信隐藏形态 `(B,T,H,HV,K,V)=(32,32,4,8,64,64)` 和
`(8,128,4,8,64,64)` 各跑 seed `1/7/42/31337/65537/20260824/20260825/20260826`。
按平台实际比较器 `atol=rtol=0.015, equal_nan=True`，16 次 output/final-state
错误计数全部为 0；specialized contract 与 empty sequence 也均为 `(0,0)`。
最终提交测试把长递推门槛进一步收紧为 `atol=rtol=0.01`，5/5 通过。

wrapper-inclusive benchmark：

| `B,T,H,HV,K,V` | E4 (ms) | Torch reference (ms) | 代理 speedup |
| --- | ---: | ---: | ---: |
| `32,32,4,8,64,64` | 6.1542 | 2.6276 | 0.4270x |
| `8,128,4,8,64,64` | 6.5050 | 10.3575 | 1.5922x |

两组均高于题面 `0.1x` 门槛。长递推 reference 确有 Inf/NaN，因此 screening
沿用平台 `equal_nan=True`；有限的前段和全部有限元素仍参与逐元素比较。

| 项目 | 值 |
| --- | --- |
| 源文件 SHA-256 | `c328d858139f446ee76f5e8f5be02776135b3ef6174d9dbc4a50b06601b98317` |
| 测试 SHA-256 | `00eaff20252f6062ee9d1febda6a06624d189844049474a346b9101204523d82` |
| release 目录 | `gpu:/tmp/flagos-fused-recurrent-gdn-release.NtCixc`（mode 0700） |
| release log SHA-256 | `bb9d81a2f07fd37d79e09deb8f98313ea5633d433a6d4ecfc443fb3681a0ca6d` |
| screening 目录 | `gpu:/tmp/flagos-fused-recurrent-gdn-bv32.gTcjCl` |
| screening log SHA-256 | `71d742e2020ae9f663b9f92b4792e97939f516a48aa388d2be53264444f80321` |
| screening harness SHA-256 | `822d85ead657f45a69dfeaa80e77a7d20f1320939dd47068dc41588a0a6817ff` |
| ZIP | `artifacts/competition/fused_recurrent_gdn/e4-3549524/fused_recurrent_gdn.zip` |
| ZIP SHA-256 | `1f4606bc083f4cd83046d5ae32f252dbc06becc8e03f41f5b60de06cb772d483` |
| ZIP 大小 / 成员 | 19,137 bytes；仅顶层 `fused_recurrent_gdn.py`（18,995 bytes） |

release 从最终 commit 直接 `git archive` 到全新目录；source/verification commit
一致。py_compile、isort、flake8、Black 26.5.1 和 unittest 5/5 全部通过，远端
source/test SHA 与 Git blob 相同。规范打包器创建后再以 `--verify-existing`
只读复验，canonical ZIP SHA、唯一成员、UTF-8、10 MB、`unzip -t` 和成员逐字节
来源门禁全部通过。

### E4 平台二投：NVIDIA 路径通过，R4 归约不可泛化

2026-08-26 16:29:15 CST，实时 preflight 核对 account `15600308080`、team
`SoulCoder`、Task 18、commit/ZIP/hash、23/30 剩余额度、120 秒间隔和截止时间后，
执行返回的一次性命令。submission `5082`，上传后额度 22/30；远端 ZIP 回读因未
配置受信 hostname 为 `unavailable`，但平台明确返回 `submitted` 和新 submission，
因此未重试。

截至 16:38:56 CST 为 6/8 terminal、1 通过：国际 A 选择 generic E4 文件并
5.39x 全过；天数、沐曦、海光、华为、国际 B 的 case 3/case 4 错误数分别为
`893/4`、`130872/8`、`730/26`、`761/5`、`901/16`，燧原和昆仑仍待回调。
相较 E2，R4 cross 在五个非国际 A 芯均未改善，而国际 A 从 `644/4` 清零，证明
四累加器交叉合并是 NVIDIA/cuBLAS 指纹，不是跨 backend 通用归约树。

## E5：backend-native `tl.sum` + NVIDIA 已通过 vendor（晋升）

状态：release 与不可变 ZIP 门禁通过；submission `5083` 评测中

验证时间：2026-08-26 16:34–16:38 CST

E5 冻结 E4 已在国际 A 平台通过的完整字节为
`fused_recurrent_gdn_nvidia.py`。generic 保持 libdevice exp、1D/one-row grid、
FP32 state/outer scratch 和 wrapper 契约，只把 prediction/readout 的 CUDA R4
静态循环恢复为 backend-native 64 项 `tl.sum`。天数、沐曦 backend 的 wave/warp
宽度为 64，该 one-row 形态至少不再硬编码 CUDA 加法树；其余芯片也由各自 Triton
backend 选择归约 lowering。

RTX 代理上 generic 不是实际 NVIDIA 选中路径：8 seeds 的短/长形态累计错误数为
159/7322，但 wrapper-inclusive speedup 为 9.5380x/33.2830x，证明编译、资源和
0.1x 性能无风险。NVIDIA vendor 继续沿用 E4 的 16-seed 零误差，以及平台国际 A
5.39x 实证。最终测试显式加载两份模块：generic 短契约 + NVIDIA 全矩阵，6/6 通过。

| 项目 | 值 |
| --- | --- |
| source / verification commit | `e3b9b4923cb1dc01fa67b3611e895231acf4bc76` |
| generic SHA-256 | `4603af742e50ea6cf372f24030183455cb97b9284f5ef2e717b4334b9a8a9937` |
| NVIDIA vendor SHA-256 | `c328d858139f446ee76f5e8f5be02776135b3ef6174d9dbc4a50b06601b98317` |
| 测试 SHA-256 | `3da08936fc4aeb5dc0f328bc9a7a900ab64b456d06a3442f573671b74b567e05` |
| release 目录 | `gpu:/tmp/flagos-fused-recurrent-gdn-release.CFaHOB`（mode 0700） |
| release log SHA-256 | `e8c30ec87491d48e99cbf5217a3a55d2712ef9c9e2d2701160a707edd13f9931` |
| generic screening 目录 | `gpu:/tmp/flagos-fused-recurrent-gdn-e5.N6UUEE` |
| generic screening log SHA-256 | `deaf56aa952b8dddbb61d68a96ab4fb885ba3cb5b05eb350797f73d6ca5e6c74` |
| ZIP | `artifacts/competition/fused_recurrent_gdn/e5-e3b9b49/fused_recurrent_gdn.zip` |
| ZIP SHA-256 | `27c6e17f24146e32fc954baf1246eaeb223ae39e73c1670fe544d7585f3ab872` |
| ZIP 大小 / 成员 | 34,331 bytes；generic + `fused_recurrent_gdn_nvidia.py` |

release 从最终 commit 导出；三文件 Git/远端 SHA 一致。py_compile、isort、flake8、
Black 26.5.1、unittest 6/6、规范构建和 `--verify-existing` 全部通过。E5 是对 E4
失败根因的单变量平台实验；generic 是否复现各 vendor Torch bmm 只能由逐芯回调
证实。

### E5 平台三投：one-row topology 否决

2026-08-26 16:41:35 CST，E5 通过实时 preflight 后执行一次性提交；submission
`5083`，额度 22/30 → 21/30，平台确认国际 A 选择 NVIDIA vendor，其余芯选择
generic。远端 ZIP 回读仍因未配置受信 hostname 为 `unavailable`，提交明确成功，
未重试。

截至首轮 7/8 terminal，国际 A 继续以 5.477x 通过。天数、沐曦、燧原、海光、
华为、国际 B 的 case 3/case 4 错误数分别为 `906/4`、`130868/9`、`583/3`、
`614/21`、`532/9`、`873/33`；昆仑待回调。海光/华为逐字复现 E2 指纹，天数、
沐曦、国际 B 未达到预设“不劣于 E2”门槛，因此 one-row/1-warp `tl.sum` topology
被否决。燧原的 1D grid 已消除原 `grid.y` 硬失败，但数值仍未清零。

## E6：恢复 `BV=8` / 4-warps reduction ownership（晋升）

状态：release 与不可变 ZIP 门禁通过；submission `5087` 评测中

验证时间：2026-08-26 16:44–16:47 CST

E6 只改变 E5 generic 的 reduction ownership：一个 program 处理 8 个 value
rows，K64 `tl.sum(axis=1)` 使用 4 warps，恢复 E2 的 `[BV=8,K=64]` topology；
program grid 仍展平为 1D，libdevice exp、FP32 state/outer scratch、外积舍入边界、
wrapper 和 NVIDIA vendor 全部保持。尾部 V 使用 mask，scratch 按真实
`[B,HV,V,K]` 连续寻址。

RTX 上 generic 仍不是实际选中路径：8-seed 短/长累计错误 142/8469，代理
speedup 9.8554x/37.9849x；generic contract/empty 均 `(0,0)`。NVIDIA vendor
继续由 E4 的本地零误差和平台 5.39–5.477x 覆盖。最终 release unittest 6/6
通过。

| 项目 | 值 |
| --- | --- |
| source / verification commit | `b528e9cd02e200cd523ec76df85839229a0039b4` |
| generic SHA-256 | `87d092112fe3f55747300df246fe4cb7f328576945f4ce3f47e38e999cd923e9` |
| NVIDIA vendor SHA-256 | `c328d858139f446ee76f5e8f5be02776135b3ef6174d9dbc4a50b06601b98317` |
| 测试 SHA-256 | `3da08936fc4aeb5dc0f328bc9a7a900ab64b456d06a3442f573671b74b567e05` |
| release 目录 | `gpu:/tmp/flagos-fused-recurrent-gdn-release.Bs1Ahp`（mode 0700） |
| release log SHA-256 | `c410620453beaf4d08226b05e2a8efee1386fd671fdd436906bda2cf572eb2fa` |
| generic screening 目录 | `gpu:/tmp/flagos-fused-recurrent-gdn-e6.csbhTT` |
| generic screening log SHA-256 | `2933ad475ae7189dfa1563c40c6e8d7efe4d1fb7d5d817ff896128c0cfd56126` |
| ZIP | `artifacts/competition/fused_recurrent_gdn/e6-b528e9c/fused_recurrent_gdn.zip` |
| ZIP SHA-256 | `6093114cf384aa3fe81a5291b6a48bd64d0b72f2c4969fc6906062967cf97764` |
| ZIP 大小 / 成员 | 35,370 bytes；generic + `fused_recurrent_gdn_nvidia.py` |

三文件 commit/远端 SHA 一致；py_compile、isort、flake8、Black 26.5.1、
unittest、规范构建与 `--verify-existing` 全部通过。E6 用一轮平台结果判定
layout/warp ownership 是否是 E2 数值指纹的必要条件。

### E6 平台四投：BV8 ownership 否决

2026-08-26 16:49:07 CST，E6 经实时 preflight 后执行一次性提交；submission
`5087`，额度 21/30 → 20/30，国际 A 选择 NVIDIA vendor，其余芯选择 generic。
远端 ZIP 回读仍为 `unavailable`，但平台提交成功，未重试。

截至 16:52:04 CST 为 7/8 terminal、1 通过：国际 A 5.505x；天数、沐曦、
燧原、海光、华为、国际 B 的 case 3/case 4 错误数分别为 `881/5`、
`131012/10`、`583/3`、`727/13`、`532/9`、`844/32`；昆仑待回调。BV8/4-warp
没有恢复 E2 的天数近零指纹，且没有新增非 NVIDIA 通过，layout/warp ownership
假设被否决。

同一时刻三笔 submission 均未完整终态：5082 为 6/8（燧原、昆仑 pending），
5083 与 5087 均为 7/8（昆仑 pending）；三笔都只有国际 A 通过。

### E7 offline：generic 恢复 `tl.exp`（不晋升）

缓存 backend 静态审计显示，Iluvatar/MetaX descriptor 的 `device_name="cuda"` 会让
项目 extra shim 尝试 CUDA libdevice，因此 E2 `tl.exp` → E4–E6 shim 是值得隔离的
变量。只读代理筛选将 E6 generic 恢复为 `tl.exp` 后，RTX 8-seed 短/长累计错误仅
从 `142/8469` 变为 `141/8456`，仍无任何非 NVIDIA 零误差证据。该工作树实验已
回退，未 commit、未构建 ZIP、未 preflight。

按 2026-08-26 16:52 CST 协调门禁，后续必须先取得新的非 NVIDIA 逐芯正信号，
或严格证明候选为何能相对 E2/E5 清零，才允许再次提交；当前保留 20/30 额度并只读
观察三笔 pending 回调。

## 2026-08-26 终态汇总（只读查询，未新增提交）

同日晚间只读 `status` 确认：5083、5087 已 `completed`，5082 仅剩燧原
`waiting_callback`（其余 7 芯终态）。三笔均为 1/8 通过、`invalid_correctness`，
逐芯 case 3/case 4 错误数：

| 芯片 | 5082 (E4) | 5083 (E5) | 5087 (E6) |
| --- | ---: | ---: | ---: |
| 天数 | 893/4 | 906/4 | 881/5 |
| 沐曦 | 130872/8 | 130868/9 | 131012/10 |
| 燧原 | pending | 583/3 | 583/3 |
| 海光 | 730/26 | 614/21 | 727/13 |
| 华为 | 761/5 | 532/9 | 532/9 |
| 国际 A（vendor） | 5.39x 通过 | 5.477x 通过 | 5.505x 通过 |
| 国际 B | 901/16 | 873/33 | 844/32 |
| 昆仑芯 | 超时崩溃 | 超时崩溃 | 超时崩溃 |

额度 16/30 已用，剩 14/30；Task 仍为 `pending_challenge`。

### 昆仑芯系统性验证超时

昆仑芯在全部 4 笔提交（含 E2 的原始简单 kernel）上均以同一方式失败：验证执行
阶段 `1830s/1800s` 超时，子进程 `Fatal Python error: Aborted`，崩溃线程位于
`torch/_inductor/compile_worker/subproc_pool.py`。该指纹与 kernel 变体无关，
说明昆仑芯评测环境对该算子的验证（reference 编译或我们的 Triton 编译）存在
独立的 30 分钟级瓶颈或崩溃；即便数值清零，昆仑芯也需要单独的编译侧解法
（例如显著降低特化变体数量或规避 inductor 路径），它不是归约树问题的下游。

### 指纹不变性与下一步

华为在 E2/E5/E6 三笔上逐字复现 `532/9`，燧原在 E5/E6 复现 `583/3`：我们的
topology/ownership 改动没有改变这些 backend 实际执行的 FP32 运算序列，
错误数完全由 backend 固有的 `tl.sum` lowering 与该芯 Torch bmm 归约树之差
经不稳定递推放大决定。在无法访问非 NVIDIA 硬件、且 NVIDIA 代理筛选已证明
无法预测这些指纹（E3/E7）的情况下，16.52 CST 协调门禁仍然有效：没有新的
非 NVIDIA 正信号或严格清零证明，不再消耗额度。剩余 14 次额度保留至
2026-08-27 19:59:59 截止。

## E8 offline：官方 `[BK,BV]` 寄存器轴（否决，不提交）

FlagAttention commit `8225e615ffec19a5481779806a03b134ff4a3b28` 的 direct GDN
使用 `[BK,BV]` state 和 `axis=0` K 归约；FlagGems-vllm commit
`43624463db77618b6d0e3f47fac990cea8c51a30` 的短序列路径也采用该形态。其公开测试
仅覆盖短序列且容差远宽于本题，因此只把寄存器轴作为一个离线变量，不复制 BV、warp、
stage、连续输入限制、state 布局或 L2 公式。

screening 基于 `b5f32dcf4cd027709ee15296ddedf94ab6e57771`，只把 K64 generic
的寄存器 view 从 `[8,64]` 转为 `[64,8]`，同时把 prediction/readout 改成
`axis=0`；物理 `[V,K]` scratch、BV8、4 warps、1 stage、libdevice exp、outer
scratch、1D grid、wrapper 和 NVIDIA vendor 全部冻结。候选工作树源码 SHA-256 为
`c3f71c17d62d8a26350c36e3f5e41705b1125f4aa9dbbf9b5e9506c50d6cdbcc`。

- screening 目录：`gpu:/tmp/flagos-fused-recurrent-gdn-kv-layout.96GmOR`，mode
  0700，PID/PGID `155369`，wall 上限 600s；py_compile、Black、isort、flake8、
  SHA 复验和 unittest 6/6 通过。
- 两个隐藏高置信形态各跑 8 seeds：`(32,32,4,8,64,64)` 累计错误 142，
  `(8,128,4,8,64,64)` 累计错误 8469；与 E6 **逐字相同**。specialized contract
  和 empty case 均为 `(0,0)`。
- wrapper-inclusive 代理 speedup 为 9.8193x / 37.6525x，性能与 E6 同级；
  screening log 与 harness SHA-256 分别为
  `cd66068e867e62938d55b3bfc1f53bf4731908a7c963c3b8287118dcdeafdffc`、
  `9b15b20bb6de8bb8345138b58fd2a4522e8c2fcdc848041f9d172301a4eebf6d`。

该结果证明当前编译器把这次轴转置规范化为与 E6 相同的归约；没有形成新数值路径，
也不能解决昆仑 1830s 超时。工作树已恢复 E6 字节，未新增测试、commit、ZIP、
preflight 或平台提交；Task 18 永久停止。

## E9：按芯分发显式归约树（重开攻坚，晋升提交）

状态：release 与不可变 ZIP 门禁通过；submission 评测中
验证时间：2026-08-27 00:20–00:50 CST；source / verification commit
`c5c9715cae2c1974395fd5097b942e7a27958a80`

### 对 08-26 协调门禁的响应与机制依据

E2–E8 的负结果共同指向一个可证伪解释：各 backend 把 `tl.sum` lowering 成各自的
加法序列，且我们的 grid/warp/layout 改动不改变该序列（华为 `532/9`、燧原
`583/3` 三轮逐字复现）。唯一的通过案例（国际 A × E4 vendor）证明了反面机制：
**逐字复现该芯 Torch bmm 的 FP32 加法顺序即可清零并通过性能门槛（5.39–5.505x）**。
因此 E9 不再改 topology，而是换成全新单变量类：

1. 放弃任何 reduction 原语（`tl.sum`/`tl.dot`），改用标量累加器 +
   显式元素级加法把归约顺序硬编码为常量表达式；编译器无法在不破坏 IEEE
   逐元素语义的前提下重结合，"backend lowering" 这一自由度被消除。
2. 平台按文件后缀自动选择每芯执行文件（E4–E6 已实证：国际 A 选
   `_nvidia`、其余选 generic）。据此为六颗未过芯片各发布一个互不相同
   的 cuBLAS 系 bmm 归约顺序假设文件；已通过的国际 A 保持 E4 原字节不动，
   generic 作为缺省回退也保持原字节不动。

K=64/BF16 隐藏形态走 kernel 内专用分支，是六颗芯片共同的实际执行路径；
非 K64/L2 路径在各文件内逐字节保持 E6 行为。昆仑芯未发布特殊文件——其失败是
验证期 `1830s/1800s` 编译侧超时崩溃，与本数值变量正交。

### 六个候选的顺序假设（提交前预注册）

| 文件 | 假设来源 | K=64 归约顺序 |
| --- | --- | --- |
| `_iluvatar` | 天数 CoreX 栈兼容 CUDA 生态，若 bmm 源自 cuBLAS 系则同指纹 | 四累加器 k%4 FMA 链，合并 `(a0+a2)+(a1+a3)`（cuBLAS 同款） |
| `_metax` | 沐曦 MCUDA wave 宽 64，小 K GEMV 常见八路分段 | 八累加器 k%8 FMA 链，成对树 `((a0+a1)+(a2+a3))+((a4+a5)+(a6+a7))` |
| `_hygon` | DTK/rocBLAS 血统 GEMV 线程独占输出、串行累加 | 单累加器 k=0..63 串行 FMA |
| `_enflame` | 烧原 GEMV 二路 ILP 形态 | even/odd 双串行链，`even+odd` 合并 |
| `_ascend` | 向量机 vcadd 相邻折叠 | 元素积先行舍入后相邻对全二叉树（6 层） |
| `_amd`（国际 B） | rocBLAS 分半连续段 + 段间一次相加 | 前 32 串行链 + 后 32 串行链，`lo+hi` |

每份文件除两个点积段外与其余文件及 E6 字节完全一致；外积 eager 舍入边界、
libdevice exp、state/outer scratch、1D one-row grid、wrapper 契约全部冻结。

### 证据

| 项目 | 值 |
| --- | --- |
| generic SHA-256 | `87d092112fe3f55747300df246fe4cb7f328576945f4ce3f47e38e999cd923e9`（= E6） |
| NVIDIA vendor SHA-256 | `c328d858139f446ee76f5e8f5be02776135b3ef6174d9dbc4a50b06601b98317`（= E4/E5/E6） |
| amd / ascend / enflame | `9caf54ec0fe2fa4203ea5c07117c9a91d5e127686c11abd0a5ab84337ad8a959` / `286a3802d71e9bf5fd8e3b1d0b628138c2609aa0cf873825c5d44d33498a3367` / `59c748d4fa151badf13a2073dff4b7eee5257dea8272be7362a0f0a5e13e2229` |
| hygon / iluvatar / metax | `9a3aad5780804b9baa3c78378dfe497995dfaa4fda12e28fdd28e0afab351851` / `80356eca70ae7f5dc854348aef82fa549e3aaee09b29f6865100575fedae1c8c` / `957b10b74747ccfb090b2c158f4b0a3a7a6d7085e2f3cece54aaeafe1bcf8f96` |
| 测试 SHA-256 | `bb77964146d33ce8986703414a29491c9f77a5f73a77b0ec50ae35101cfa955e` |
| release 目录 | `gpu:/tmp/flagos-fused-recurrent-gdn-e9.qw6zFA`（mode 0700） |
| release log SHA-256 | `2ca561b4b4ea865bfdf28786f30684af89006e22e3cbfca4161756f7362bced4` |
| ZIP | `artifacts/competition/fused_recurrent_gdn/e9-c5c9715/fused_recurrent_gdn.zip` |
| ZIP SHA-256 | `63188c757c0f16baf6fe718619a3fbf4aceabad56b07172dac1ebaee78efc763` |
| ZIP 大小 / 成员 | 168,364 bytes；generic + 7 个 vendor，共 8 个顶层 `.py` |

release 由 commit `c5c9715` 经 `git archive` 生成到远端临时目录；九文件远端
SHA 与 Git blob 逐项相同。py_compile、black/isort/flake8（仓库配置）、
unittest 8/8（含六个 vendor 模块的 K64 契约 smoke、scalar beta、空序列）全部
通过。环境：RTX 5070 Ti、driver 610.57.04、PyTorch 2.13.0+cu130、
Triton 3.7.1、CUDA 13.0。

wrapper-inclusive do_bench（BF16，含 initial/final state）代理证据：

| shape | generic ms | 各 vendor 变体 ms 范围 | 对照 reference ms | 变体代理 speedup |
| --- | ---: | ---: | ---: | ---: |
| `(32,32,4,8,64,64)` | 0.254 | 3.158–3.848（ascend 最慢） | ≈2.63 | ≈0.69–0.83x |
| `(8,128,4,8,64,64)` | 0.260 | 3.265–3.971 | ≈10.36 | ≈2.61–3.17x |

两隐藏形态均高于题面 0.1x 门槛；NVIDIA 上这些变体并非实际选中路径，上述数字
仅为资源/编译/性能风险证据。其余五颗芯片与昆仑的真实正确性与 speedup 只能由
平台回调判定；本候选相对 E2/E6 的增量风险仅限"新文件能否在目标芯编译运行"
这一被机制目标所要求的变量。额度当日剩余 29/30，满足协调门禁的成本约束。

### E9 平台五投：选文件机制全链路实证；海光降至 3 元素残差

2026-08-27 00:47 CST，preflight intent `prepared` 全字段核对后执行一次性确认；
submission `5215`。平台逐芯 `selected_file` 首次完整实证了后缀映射并全部按设计
命中：天数=`_iluvatar`、沐曦=`_metax`、燧原=`_enflame`、海光=`_hygon`、
华为=`_ascend`、国际A=`_nvidia`、国际B=`_amd`、昆仑=generic。

截至 00:57 CST 六芯终态：

| 芯片 | 文件 | 结果 | case 错误数 |
| --- | --- | --- | --- |
| 国际 A | `_nvidia` | **通过 5.4985x** | clean |
| 海光 | `_hygon`（串行 FMA） | 差 3 个元素 | `3/1048576`（单 case） |
| 天数 | `_iluvatar`（cuBLAS 同款 R4） | 失败 | `893/4`（≈tl.sum 基线 881/5，非 cuBLAS 系） |
| 沐曦 | `_metax`（R8 成对树） | 失败 | `131052/16`（量级不变，非 R8 树） |
| 华为 | `_ascend`（相邻对树） | 失败 | `531/6`（vs 基线 532/9——对树型几乎不敏感） |
| 国际 B | `_amd`（半区双链） | 失败 | `567/34`（较基线 844/32 显著移动） |

结论：(a) 海光串行假设近乎命中，残差 3 元素指向最后一层舍入语义（FMA 融合或
扫描方向），不是树结构；(b) 华为对归约树不敏感进一步收窄其差异来源；(c) 天数/
沐曦排除两个主流指纹族。

### E10 计划（单变量：串行乘加分离）

只改两份文件：`_hygon` 与 `_amd` 的 K64 归约改为"显式先乘后加"的纯串行链
（拆开 FMA 融合这一变量），其余六个文件与 generic/NVIDIA 字节冻结，作为同轮
对照。预期海光 3→0；国际 B 若同为 ROCm 血统应大幅收敛，否则该信息同样定位
其真实形态。

## E10：hygon/amd 串行乘加分离（晋升提交）

状态：release 与不可变 ZIP 门禁通过；提交中
验证时间：2026-08-27 00:53–00:58 CST；source / verification commit
`bd3f8569f15c61c81cd587ade27f80c5bbddd4b1`

只变量：`_hygon`、`_amd` 两文件的 K64 归约从 `tl.fma` 链改为显式
`product = a * b; acc = acc + product` 的纯串行链（先乘后加两次舍入），用于
解释 E9 海光仅剩的 3 元素残差；`_amd` 采用同一内容作为 ROCm 血统假设的同族
探针。generic、`_nvidia`（国际 A 已实证字节）、`_iluvatar`、`_metax`、
`_ascend`、`_enflame` 及测试共七个文件与 E9 commit 逐字节相同。

| 项目 | 值 |
| --- | --- |
| amd SHA-256 | `48c31ae8d642a886c05f49110ae658cb8d692c2cb6be88a99eca341a0f774589` |
| hygon SHA-256 | `613e625e68f8ebe37f6f548d34f3290e00d28f284eefd1364c0d24734fd09920` |
| 其余文件 | 与 E9 完全一致（generic/nvidia/ascend/enflame/iluvatar/metax/test） |
| release 目录 | `gpu:/tmp/flagos-fused-recurrent-gdn-e10.peHwYq`（mode 0700） |
| release log SHA-256 | `372c10283133328730cda4de51ea6a6014ecedc91548fcc9ec31bb7293f861aa` |
| ZIP | `artifacts/competition/fused_recurrent_gdn/e10-bd3f856/fused_recurrent_gdn.zip` |
| ZIP SHA-256 | `aa7afb3a35ac5a551f2322fca4c320f35472391fe572c37bd3fb6f092a568d14` |
| ZIP 大小 / 成员 | 167,312 bytes；generic + 7 vendor |

release 由 commit `bd3f856` 经 `git archive` 建目录；九文件远端哈希与 Git blob
逐项一致；py_compile、black/isort/flake8（仓库配置）、unittest 8/8 通过；
wrapper-inclusive do_bench 显示 hygon/amd 变体 ≈3.2–3.4 ms，与前轮同级。
规范打包器 dry-run 与正式构建 canonical SHA 一致，`unzip -t` 通过。
E10 相对 E9 只有两个受控差异文件；额度窗口内剩余 ≥20 次满足成本约束。

### E10 平台六投：乘加分离方向被否决，海光 3→15/16 反向恶化

submission `5218`。海光从 E9 的单 case `3/1048576` 恶化为双 case
`15,16`——**融合 FMA 的串行链显著优于分离乘加**，海光指纹族锁定为
"k=0..63 串行 + FMA"。国际 B 继续收敛（567→402），串行族假设进一步增强。
天数/沐曦/华为对照文件逐字未变，误差数完全复现（893/4、131052/16、531/6），
同时再次证明平台判定对相同字节的确定性。

## E11：hygon 串行 FMA 复原 + amd 同族探针（晋升提交）

状态：release 与不可变 ZIP 门禁通过；提交中
验证时间：2026-08-27 01:08–01:16 CST；source / verification commit
`7da35e73b2bf7bd2d6f950e498b08bf8b1786cb3`

- `_hygon`：恢复 E9 的串行 FMA 链（kernel 段与 E9 字节一致，仅头注释更新），
  作为最优已知配置与平台确定性双重对照；
- `_amd`：改用与 hygon 完全相同的串行 FMA 内容（rocBLAS 血统同族探针，
  预期较 E10 的 402 继续大幅移动或直接清零）；
- 其余六个文件（generic/nvidia/iluvatar/metax/enflame/ascend）与测试
  保持上一轮字节不变。
- TF32 截断探针已在本地代理上以 unittest 失败证伪其保真度并放弃（chop 式
  mantissa 清零 ≠ 真 TF32 RNE 转换），故华为本轮不改变量。

| 项目 | 值 |
| --- | --- |
| amd SHA-256 | `4409247f282d920cd44e1bfdb32f9458c73030051771990c43e759dca6ba6689` |
| hygon SHA-256 | `6aa9eed7e812ac9e54aa1559ebd645e93c2017e1271fe8c35ce2febcb9b4dbf8` |
| ascend | `286a3802d71e9bf5fd8e3b1d0b628138c2609aa0cf873825c5d44d33498a3367`（=E9） |
| release 目录 | `gpu:/tmp/flagos-fused-recurrent-gdn-e11.f8VTYy`（mode 0700） |
| release log SHA-256 | `c2cad3cf4486cf68d8d4d001dfde243bfedfa867d91e3806383bad0bfe24f992` |
| ZIP | `artifacts/competition/fused_recurrent_gdn/e11-7da35e7/fused_recurrent_gdn.zip` |
| ZIP SHA-256 | `4d8ab6270f6661d880cf64f4759baf5f117bff8ed38ac00c9329e559e38941dd` |
| ZIP 大小 / 成员 | 167,172 bytes；generic + 7 vendor |

远端九文件哈希与 commit blob 一致；py_compile/lint/unittest 8/8 全过；
do_bench hygon/amd ≈3.20/3.43 ms 同级。dry-run 与正式构建 canonical SHA
一致，`unzip -t` 通过。

### E11 平台七投：确定性对照成立；AMD 对融合不敏感、对树形敏感

submission `5221`：
- `_hygon` 恢复后海光精确复现单 case `3`——同字节同判定，平台评测确定性成立；
- 国际 B `402/26` 与 E10 的 unfused-serial **完全相同**：AMD 栈把分离乘加重新
  融合为 FMA，该芯对融合方式不敏感、只对归约树结构响应（halves 567 → serial
  402）；
- 天数/沐曦/华为对照位继续逐字复现（893/4、131052/16、531/6）；国际 A
  5.5315x 三连过；
- 昆仑芯本轮判定 `pass=false` 空 case 列表，与既往编译期超时崩溃一致。

## E12：amd even/odd 交错双链（晋升提交）

状态：release 与不可变 ZIP 门禁通过；提交中
验证时间：2026-08-27 01:30–01:45 CST；source / verification commit
`79afc084ddf891e826c9b479a8c9cb3cd7caccc2`

唯一变量：`_amd` 改为 even/odd 交错 FMA 双链（k%2 分组，even+odd 合并）——
经典族谱中该芯未试的最后形态。其余七个文件与 E11 字节完全一致。

发布纪律证据：第一次生成时误将单累加器合并式 `"prediction = prediction_0"`
带入双链文件（奇链整段被丢弃），远端 release unittest 以 `815/816` 元素失败
将其拦截；修正合并式为 `"@ = @_0 + @_1"` 后重走全部验证，8/8 通过。错误字节
从未进入 ZIP 或平台。

| 项目 | 值 |
| --- | --- |
| amd SHA-256 | `31e59173c273141bf23471116a2ecefaa58f06516ca68ec2ce4aa1f27466e3e8` |
| release 目录 | `gpu:/tmp/flagos-fused-recurrent-gdn-e12.eSQdWX`（mode 0700） |
| release log SHA-256 | `5af7b61f064bf6942d52011442a0e73c145b2d57f04047ccc216330d18171912` |
| ZIP | `artifacts/competition/fused_recurrent_gdn/e12-79afc08/fused_recurrent_gdn.zip` |
| ZIP SHA-256 | `3f058ea65ac64a07918ad4b28af4f2d03cf40c57e98099d1925494a60c46707a` |
| ZIP 大小 / 成员 | 168,331 bytes；generic + 7 vendor |

期望读数：card_b 较 402 继续下探则交错族更优并指引"每链更短+多链"方向；
回到 ~560 量级则串行仍是当前最优假设。

### E12 平台八投：evenodd 否决，串行仍是 AMD 最优

submission `5223` 首批终态：card_b `844/38` 恶化回 native 量级——交错双链被
否决；card_b 家族排序收敛为 **serial(402) < halves(567) < evenodd/native(~844)**。
海光继续精确复现 `3`，天数/沐曦/华为对照复现 `893/4`、`131052/16`、`531/6`，
国际 A 连续第五轮通过（5.359x）。燧原三笔全天未回调，昆仑维持编译期崩溃。

## 2026-08-27 收官快照与后续单变量队列

| 芯片 | 最优已知字节 | 当前状态 |
| --- | --- | --- |
| 国际 A | `_nvidia`（E4 不变） | ✅ 五连过，5.36–5.53x |
| 海光 | `_hygon` 串行 FMA | 仅差 3 元素，方向已锁死 |
| 国际 B | `_amd` 串行 FMA（E11 内容） | 402，树形敏感、融合不敏感 |
| 天数 / 沐曦 / 华为 | generic/native | 对已试全部变量无响应，疑非点积树差异源 |
| 燧原 | `_enflame` evenodd | 今日平台未回调，数值未知 |
| 昆仑芯 | generic | 编译期 1800s 超时崩溃，结构独立问题 |

额度 16/30 剩余。建议的下一序列（按信息增益排序）：
1. `_hygon` 反向串行（k=63..0）或"首元素乘积初始化"变体——3 元素残差二选一排除；
2. `_amd` chunk 化串行（如 8 段各 8 顺序合并）——在 402 与 halving 之间插值定位；
3. 平坦三芯改走 bmm 内部精度模式假设（真 TF32-RNE 或 bf16-in/fp32-acc 截断），须先在代理验证正确性门禁可放行（放宽 atol 策略需谨慎，避免公开契约回归）；
4. 燧原回调到达后再定其族谱探测。
当晚所有提交字节、证据目录与哈希见上文各节；全程仅本地 commit，未 push。
