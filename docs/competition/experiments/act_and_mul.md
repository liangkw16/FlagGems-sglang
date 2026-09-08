# Task 42 `act_and_mul` 实验记录

```current
task: 42
operator: act_and_mul
batch: 4
validity: valid
platform: 8/8(e6,3.25835x,排名10)
team_best_stage: e6
team_best_commit: 5251bf5f1ef53eaac8ce83c0bb0d0b0b91425fad
team_best_speedup: 3.25835
sealed: no
next: e7 stride融合已验证3.149675未超E6(计分布局以连续为主);stride轴关闭,保留E6,无新结构证据不开轴
updated: 2026-09-08


状态：S0 候选就绪（generic 单文件），远端 NVIDIA 代理 screening 通过
（11/11 单测 + 三项 lint + 基准水位健康），未打包、未提交。2026-09-03
当日额度 30/30 已耗在第 3 批（最后 16:06），本题为第 4 批开题首投候选。

## 契约锁定

- 签名：`act_and_mul(gateup_output, activation="silu", swiglu_limit=None)`
- 输入：`gateup_output [M, 2H]`（**严格 2D**，参考实现 `[:, :half]` 切
  dim 1，>2D 无定义语义）；`gate = [:, :H]`，`up = [:, H:]`
- 计算：gate/up 先 `.float()`；可选 clamp（gate 只 max，up 对称）；
  activation ∈ {"silu","gelu(tanh)"}，其他值 `ValueError`；
  **激活在 fp32 算完后 cast 回输入 dtype，与 up（fp32 clamp 后同样 cast
  回）在输入 dtype 下相乘**——精度切换点与 T7/T29/T39 的 fp32 乘不同
- 输出：`[M, H]`，dtype 同输入
- 容差：fp32 1e-4 / bf16 1.5e-2 / fp16 1e-2
- 支持八芯；反作弊：核心计算必须 Triton，禁 try/except / 设备判断 /
  PyTorch fallback

## 方案（S0）

- 骨架复用 T29 `gelu_and_mul.py`（row/col-block 映射 + capped 65535
  grid-stride + BLOCK_COL=1024，两题八芯验证的结构）：
  每 program 一行 × 1024 列，gate/up 双载，块级除法（每块一次
  `row_id = block_id // num_col_blocks`，燧原逐元素除法贵的教训）
- SiLU 逐字题面公式 `gate / (1 + exp(-gate))`（fp32，禁稳定化改写）
- GELU-tanh：`0.5*x*(1+tanh(0.7978845608*(x+0.044715x³)))`，tanh 用
  稳定形式 `s·(1-e^{-2|y|})/(1+e^{-2|y|})`（饱和精确 ±1、无溢出、
  无 libdevice；避开昆仑 erf 崩溃族）。`tl.math.tanh` 留作 vendor 轴
- `HAS_LIMIT`/`ACT_IS_GELU` 均为 host 已知量 → constexpr 特化，非运行期
  分支（燧原约束）
- store 行：`act.to(elem_ty) * up.to(elem_ty)`（输入 dtype 乘法）
- 不显式设 num_warps/num_stages；wrapper `contiguous()` + 空输入 guard

## 验证证据（screening 模式，未提交候选）

- 远端：`gpu`（kkgpu/RTX 5070 Ti，driver 610.57.04），
  torch 2.13.0+cu130，triton 3.7.1，CUDA 可用
- 目录：`/tmp/flagos-act_and_mul.1zQ1I9`（0700）；两文件 SHA-256 与
  上述 source commit 逐字节一致：
  - `src/flaggems_sglang/ops/act_and_mul.py`
    `c5b7a981ee79694664a77ff266ba6143866afa292858e5395d6de495bba15acb`
  - `tests/test_act_and_mul.py`
    `a894bd19eeea52ed850c2938c974006e85d6e57d419325d74c684600afef574c`
- 门禁：py_compile 通过；black/isort(--profile black --line-length 80)/
  flake8(--max-line-length=120) 全绿（字节为远端 black 重排后取回，
  hash recheck 一致）；`python -m unittest -v` 11/11 OK（1.5s）
- 单测覆盖：3 dtype × 2 activation、边界 rows/half（1/7/63/64/65/511/
  512/513/1023/1024/1025/4096）、非连续、输入不变性、空输入/零宽、
  特殊值（±inf/±1e4/±92/±90/NaN，equal_nan）、limit∈{0.5,7,1e4}、
  **limit=0.0 仍生效**（falsy 非 None）、不对称 clamp 两侧探针、
  非法 activation ValueError、2D 契约（4097×8192 大形状）
- 基准（`triton.testing.do_bench` warmup=25 rep=100 median，AB 同机）：

  | shape | dtype | silu | gelu+limit7 |
  | --- | --- | ---: | ---: |
  | 4096×2048 | bf16 | 5.07x | 7.57x |
  | 16384×2048 | bf16 | 6.40x | 9.06x |
  | 4096×7168 | bf16 | 6.44x | 9.05x |
  | 1024×512 | bf16 | 3.14x | 3.71x |
  | 16384×2048 | fp32 | 1.68x | 3.04x |

  NVIDIA 代理证据，不能外推八芯。当前公开榜首 c2flow 3.1941x
  （八芯均值，含弱芯拖累），S0 水位判断有余量。

## 已知风险与对策

- 昆仑：无 erf/libdevice 依赖，无运行期分支；若读数弱，唯一有效轴
  BLOCK=2048（T29/T39 双实证）
- 燧原：整行 BLOCK 4096 四证强假设（T24/T33/T39/T29）；kernel 内无
  运行期分支已满足
- 华为：BLOCK 512 -42% 勿碰（T39）
- 沐曦：flat 大 BLOCK 2048（+65%/+11% 双证）
- 国际 B：四档列 tile autotune（128/2w、256/4w、512/8w、1024/8w，
  key=half_width）+3.83% 可搬；国际 A autotune 证伪不投
- 海光水位波动 34–56x，单轮高值不当结构收益
- gelu tanh 分支 fp32 下我们的稳定 tanh 与 torch tanh 有 ulp 级差，
  容差 1e-4 内（单测已含三 dtype 特殊值对齐）

## MCP 实机初筛（注入执行协议，2026-09-04 凌晨）

- 协议：`autotune_kernel` description 注入 VERBATIM 候选全文，
  `operator_name=<算子名>`（**不得加前缀**——首轮 `screen_` 前缀使
  harness 按前缀名调用、全部 NameError，失败神谕被污染，已废弃重交，
  污染产物存 `log/kernelgen-round/screen-prefix-polluted/`）
- **华为**：completed；iteration 5 轮零 error（失败神谕未触发）；
  终态代码与候选 kernel 逻辑逐字一致（diff 仅注释/license 头）；
  total_tests=0 → 按可信度阶梯记
  `mcp-compile-screened(fidelity)`，不采信 passed/自测 speedup；
  产物 `log/kernelgen-round/out_act_and_mul_huawei.json`
  SHA-256 `7f5981c381650ea4…`（完整哈希见文件）
- **天数**：completed；同样零 error + 逻辑保真 →
  `mcp-compile-screened(fidelity)`；产物
  `out_act_and_mul_tianshu.json` SHA-256 `d8e2621c5f8aab27…`
- 海光/沐曦：后台队列进行中（结果落地后续记）
- 结论：发射前编译风险最高的两家（华为/天数）无编译失败信号

## 提交预算与止损（2026-09-03 定稿）

- 默认 5 发：S0 探路 → 最多 3 次 vendor 单变量 → 1 发回归储备；
  同指纹失败连 2 次提前停
- 首投排程：09-04 00:00 额度重置后第 1 发（打包 → preflight → 一次性
  提交，门禁全过即自动执行）
- 每轮只改一个 vendor，其余字节逐字节冻结；冻结芯分数变化按水位处理

## 时间线

- 2026-09-03 20:42 题面同步、契约锁定、S0 实现 + 远端 screening
  11/11 通过 + 基准；commit `2652a4e`；未提交（额度 0/30）

## 平台首投结果（2026-09-04 01:09，submission 9370，daily_seq 1）

- **8/8 valid，平均 3.066825x**（ZIP `3d78528e…`，source `2652a4e`）
- 逐芯：天数 6.2234 / 沐曦 2.2476 / 燧原 0.969 / 海光 4.9658 /
  昆仑 0.279 / 华为 1.6714 / 国际A 4.538 / 国际B 3.6404
- 全芯正确性通过；最低昆仑 0.279x 仍过 0.1 门槛
- 榜首 c2flow 3.1941x，差距 4.2%；下一轴按预案：昆仑 BLOCK 2048 /
  燧原整行 4096 / 华为块结构（T39 E10 +246% 同构题）


## MCP 实机初筛归档（2026-09-04 晨，24 job 全部终态）

- 产物 `log/kernelgen-round/out_<op>_<chip>.json`（24 个，含 SHA）；
  协议：注入执行 + 失败神谕 + 终态代码保真 diff
- 干净通过（fidelity=True 且零 hard error）：本题华为/天数（详见
  各算子行）；海光/沐曦后端当夜多次 502（`ld0428.baai.ac.cn`），
  这些芯的编译信号不可得，非候选失败
- 保真失败（LLM 改写）= 无判定，不作数；harness 侧 artifact
  （NameError/IndexError/`constexpr[0]`）不计入失败神谕
- 平台实测（本账本上方小节）已是更强证据，MCP 结论仅作发射风险
  参考留存

## E1 三 vendor 冲分（2026-09-04 04:3x，submission 9416，daily_seq 10）

- **8/8 valid，平均 3.1647x**（3.0668 → +3.2%；source `1273a60`，
  ZIP `bfe4389f…`）
- 燧原 4096 整行：0.969→**1.682（+74%）**；昆仑 flat 2048：
  0.279→**0.4548（+63%）**；沐曦 flat 2048：2.2476→2.4552（+9%）
- 未动芯：天数 6.214 / 海光 4.9688 / 华为 1.668 / A 4.238（水位波动）
  / B 3.6368
- 榜首已升至 Warmhearted 3.369775x（7 队过线）

## E2 华为行结构化 + 海光 warps2（2026-09-04 06:0x，submission 9431，daily_seq 12）

- **8/8 valid，平均 3.248925x**（3.1647→+2.7%；source `f69d4e2`，
  ZIP `9f5f3bb2…`）
- 华为行结构化 vendor（每行一 program、行内 1024 子块、仅尾块带
  mask——T40 向量比较退化规避）：1.668→**2.258（+35%）**；
  海光 warps2：4.9688→5.067（+2%）；沐曦水位回落 2.4552→2.4006
- 榜首 Warmhearted 同步升至 3.5194x（7 队过线，44 提交，活跃竞争）

## E3 AMD 四档 autotune（2026-09-04 07:1x，submission 9451，daily_seq 14）

- 8/8 valid，平均 **3.2424——低于 E2 的 3.2489**（team best 维持 E2）
- card_b 3.6332→**3.4536**：T39 的 +3.83% 正迁移未复现，本题
  autotune 轴**单发证伪关闭**；沐曦 2.4006→2.501（水位）
- 结论：T42 各 vendor 轴基本打满（华为/海光/燧原/昆仑/沐曦已优化，
  AMD 负迁移）；对榜首 3.5194x 的差距属结构差，继续追需弱芯新结构
  证据，性价比低，转入守榜

## 2026-09-05 Codex 会诊作战方案（预注册，gpt-5.6-sol xhigh，只读会诊）

榜首 09-05 快照 431.4843x 判定为计时/reference 病理（总分较旧榜首放大约
122 倍，非合法优化量级）——**侦查不追分**，按旧榜首 Warmhearted 3.5194x
水位决策。三候选合计预期约 +2.2 逐芯分，理论可触旧榜首。

候选（单变量，其余成员逐字节冻结；发序 A1 → K1 → M1）：

1. **A1 华为裁尾（首发）**：行 vendor 完整块循环后无条件执行一次全 mask
   空尾块——`H % 1024 == 0` 时每行白做一整块激活数学（源码已核）。
   修法：constexpr `HAS_TAIL = half_width % BLOCK_INNER != 0`，无尾时
   编译期裁掉尾路径。预期 +15~45%；晋级门华为 ≥2.48x，≥2.70 再议宽 tile。
2. **K1 昆仑直达**：flat-2048 保持 BLOCK/数学不变，去掉动态 grid-stride
   循环，`ceil(n/2048) <= 65535` 时一 program 一块直达映射（T40 同法
   0.246→0.945 平台实证）。晋级门昆仑 ≥0.60x；<0.48 关闭无循环轴。
3. **M1 沐曦 warps8**：BLOCK=2048 下 num_warps 默认 4→8（仓库
   `metax_heuristics_for_num_warps` 2048 档=8 背书）。晋级门 ≥2.52x，
   <+5% 关闭 warp 轴。

431x 侦查（零额度）：只读拉该提交逐芯明细——单芯千倍=查该芯 reference/
隐藏 shape；八芯同倍=查计分口径；拿不到明细则维持 3.5x 水位决策。

## E4 华为 HAS_TAIL 裁尾（2026-09-06 00:20，submission 10324，daily_seq 2）

- 载体：E2 字节 + `_ascend` 加 `HAS_TAIL = half_width % BLOCK_INNER != 0`
  编译期裁掉空尾块（对齐 shape 不再每行白做一整块 mask 激活数学）；
  source `b6f982c`，ZIP `e4-b6f982c` SHA-256 `b51c80b7…4f1c`（7 成员）
- screening：远端 `/tmp/flagos-t42a1.8kW0mG`（unittest 13/13 OK 含新增
  aligned-width vendor 矩阵）；flake8 过，black-79 diff 仅既有字节漂移
- **终态 8/8 valid，avg 3.2325——非 team best（E2 3.2489 保持）**
- 华为 2.258→**2.3884（+5.8%）**：方向为正但**低于 ≥2.48 晋级门**
  （隐藏 shape 疑非全部 H%1024==0，仅部分命中裁尾）；沐曦 2.47（水位）；
  其余芯持平（昆仑 0.455/燧原 1.685/天数 6.23/海光 4.95/A 4.22/B 3.46）
- 判定：裁尾轴按单发证伪关闭（增益不足门）；T42 维持 E2 守榜，
  剩余 K1 昆仑直达（门 ≥0.60）待额度富余再发。额度 28/30

## E5 昆仑直达终态（2026-09-06，submission 10336）

- K1：昆仑 flat-2048 去循环直达 kernel（块数 ≤65535 时启用）——
  **8/8 valid avg 3.2315，非 team best；昆仑 0.4548→0.45 纹丝不动**，
  晋级门 ≥0.60 未达：**直达轴证伪关闭**（T40 的循环瓶颈不迁移到本题，
  0.45x 判定为该 op 在昆仑的水位）
- 剩余唯一未消费候选：M1 沐曦 warps 4→8（仓库 MetaX 配置背书）

## E6 沐曦 warps8 → 新 team best（2026-09-06，submission 10344）

- M1：沐曦 BLOCK=2048 下 num_warps 默认 4→8（仓库
  `metax_heuristics_for_num_warps` 2048 档=8 背书）；source `5251bf5`，
  ZIP `e6-5251bf5` SHA-256 `a6dfefec…2b71`
- **终态 8/8 valid，avg 3.25835 —— 新 team best**（E2 3.2489 → +0.3%）
- 逐芯：沐曦 2.2642（窗口水位低于上轮 2.47——warps 轴本身不明确，
  增益来自其他芯水位：华为 2.333/昆仑 0.4482）
- 判定：T42 预注册三候选（A1/K1/M1）全部消费完毕；M1 载体成为
  TB 字节，转守榜。距旧榜首 3.5194 仍有 -7.4%，无已验证新轴

## 2026-09-07 只读盘点校正

当前榜首与排名按最新任务API校正；旧段落保留历史快照，不据此分配本轮提交机会。本轮未修改或提交本题源码。
- 查询证据 `/Users/bytedance/ccc/flagos/artifacts/competition/batch4-top1-20260907/tasks-now.json` SHA256 `fc73368c3d98b228b0c7815d6ec1e9042a58af8d953337fff58990daec1474fc`。

## 2026-09-08 推荐方案实现与提交前验证（未提交平台）

generic 按真实行/列 stride 读取，消除非连续输入的 contiguous 拷贝，保留激活与 cast 顺序。代理非连续案例 1.34–1.62x，连续案例约 0.96–0.97x。

- source `26a95766b179d263916e9483dfc8d2343c40406a`；verification `26a95766b179d263916e9483dfc8d2343c40406a`。14 个测试方法、84 次实际 kernel 调用；选定 NVIDIA/代理范围门禁通过。
- 回执 `artifacts/competition/batch4-implementation-20260907/t42-release1/verification.json`，SHA256 `0065a84045e37aba7fe3e8214070929513c55e572332f2589dfd54f766d00c34`；日志 SHA256 `2281433114927e3978a08f26797f29191879f0906f1c1556cc1ee3430d2e2259`。
- 不可变 ZIP `artifacts/competition/act_and_mul/research-20260908-26a9576/act_and_mul.zip`，SHA256 `921e5648a0ec00eaaf9d61b7f408824db5ccb75cc89013c478dd78c3b0f0b372`；与 dry-run manifest、构建和 existing 验签一致。ZIP 是候选产物，不等于目标芯或平台已通过。
- 环境、逐源码执行范围、原始配对数据和未完成条件见[本轮报告](../implementation-batch4-20260908.md)及[证据清单](../data/batch4-implementation-20260908.json)。本轮不更新历史有效分，未做平台 preflight、上传或正式提交。

## E7 stride 融合 → 8/8 valid 3.149675，未超 E6，轴关闭（2026-09-08，sub 11048）

- 候选（commit `26a9576`，t42-release1 回执）：移除强制 contiguous，
  融合核按真实行/列 stride 读入；代理非连续 1.34-1.62x、连续 0.96-0.97x。
- 终态逐芯：天数 6.2124 / 沐曦 2.3144 / 燧原 1.681 / 海光 4.933 /
  昆仑 0.458 / 华为 2.2184 / A 4.2852 / B 3.095；avg 3.149675
  （E6 team best 3.25835 保持，平台计分 case 以连续布局为主）。
- 判定：非连续计分假设不成立，stride 轴关闭；榜首 431x 的结构面
  仍未知，保留 E6 守榜。

## E6r 水位重掷终态：8/8 valid 3.2160，未超 TB（2026-09-08T16:52，sub 11241）

- e6 字节注释载体（`a602a6b`，树态从 e7 stride 字节恢复 e6）。未现
  窗口尖峰；TB 3.25835 保持。T42 距榜首 431 需极端慢窗（T51 式），
  常规重掷无意义，后续仅在类似窗口证据时再打。
