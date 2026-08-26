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
