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

截至 12:14:11 CST 为 `evaluating`、7/8 terminal、0 通过，昆仑仍为
`waiting_callback`。七芯均在隐藏长 BF16 递推失败：case 3 的错误数依次为沐曦
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
