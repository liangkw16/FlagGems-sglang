# Task 43 `causal_conv1d_update` 实验记录

```current
task: 43
operator: causal_conv1d_update
batch: 4
validity: valid
platform: 8/8(e14,6.5204375x非最佳);e13最佳6.545875x
team_best_stage: e13
team_best_commit: 4fa854a376de167e76a1e5d1441c6cd82b5866d7
team_best_speedup: 6.545875
sealed: no
next: 滚动窗口已实现并通过代理;性能未晋级,守E13,需目标加载/占用证据
updated: 2026-09-08
```

状态：S0 候选就绪（generic 单文件），远端 NVIDIA 代理 screening 通过
（8/8 单测 + 三项 lint + 基准水位 3–16.7x）。本题为 pending_challenge
（0/9 队达标），首个有效解含金量高；0 达标提示真门大概率在弱芯正确性
或 0.1x 门槛，S0 首投即探针。

## 契约锁定

- 签名：`causal_conv1d_update(x, conv_state, weight, bias=None, activation="silu")`
- 输入：x `[batch, dim]`（视 seqlen=1）或 `[batch, dim, seqlen]`；
  conv_state `[batch, dim, state_len]`；weight `[dim, width]`（depthwise）；
  bias `[dim]` 或 None
- 计算：x_cat = concat(conv_state, x)（fp32）；对每个 t：
  `out[:,:,t] = Σ_k weight[:,k]·x_cat[:,:,t+state_len+1-width+k]`；
  +bias；activation ∈ {"silu","swish"} 时 `x·sigmoid(x)`（其他值不加激活，
  参考**不报错**）；out cast 回 x.dtype
- 状态前移：new_conv_state = x_cat[:, :, -state_len:] cast 回
  conv_state.dtype；**out-of-place（clone 语义），不改输入**；返回
  `(out, new_state)` 二元组，2D 输入返回 2D out
- 容差：fp32 1e-4 / bf16 1.5e-2 / fp16 1e-2（out 与 state 均判）
- 支持八芯；反作弊：核心计算必须 Triton，禁 try/except / 设备判断 /
  PyTorch fallback

## 方案（S0）

- 每 program 一条 (batch, dim-block) 条带（BLOCK_D=256，capped 65535
  flat 1D grid + grid-stride，块级除法一次）；seqlen/state_len 走标量
  device 循环，width `tl.static_range` constexpr 展开，FMA 标量累加
- **全程 1D load/store**（避开 T36 昆仑毒点组合：2D masked tile +
  超越函数 + axis-1 reduce）；虚拟 concat 用双路 masked load（安全
  clamp 索引）+ 标量 `tl.where` 选择，无数据依赖分支
- SiLU 逐字 `val / (1 + exp(-val))`（T39 平台实证形式）
- out `empty_like(x)`、new_state `empty_like(conv_state)` 全覆盖写
  （T36 E22 省一份 clone 拷贝的实证）；wrapper 全参 `.contiguous()`
- `WIDTH`/`HAS_BIAS`/`ACT_IS_SILU` 均为编译期已知（constexpr 特化，
  非运行期分支）；bias 缺省传 x 占位（T36 先例）
- 显式不抄 SGLang `_causal_conv1d_update_kernel`（本仓 vendor patch
  实证其在昆仑 XPU fp16/bf16 编译失败）

## 验证证据（screening 模式，未提交候选）

- 远端：`gpu`（RTX 5070 Ti，driver 610.57.04），torch 2.13.0+cu130，
  triton 3.7.1；目录 `/tmp/flagos-ccu.bCrJtB`（0700）
- SHA-256（与 source commit `07aaf2e` 逐字节一致）：
  - `src/flaggems_sglang/ops/causal_conv1d_update.py`
    `0e62c67bd41c3ebf3d1a8ee12fc3eb6a4ba264dd94841b657f44e9c8df39bf35`
  - `tests/test_causal_conv1d_update.py`
    `711df2cbd4c292ad8e4230565304a9ef54a3fcebc4977a77b4b9ee8b522730a6`
- 门禁：py_compile / black / isort / flake8 全绿（远端 black 重排后
  取回，hash recheck 一致）；unittest 8/8 OK（2.1s）
- 单测覆盖：3 dtype × activation {silu/swish/无激活} × bias 有无；
  width∈{2,3,4,5,8} × state_len∈{width-1, 更长} × seqlen∈{1,2,3,5}；
  2D/3D 输入；batch/dim 边界（1/64/100/255/256/2049）；非连续 x 与
  state；seqlen>state_len 的全量换态路径；特殊值（±inf/±1e4/±92/±90，
  equal_nan）；空 batch；**三输入不变性逐项断言**
- 基准（do_bench warmup=25 rep=100 median，含 bias+silu，seqlen=1、
  width=4、state_len=3）：

  | shape | bf16 | fp16 | fp32 |
  | --- | ---: | ---: | ---: |
  | 64×2048 | 7.46x | 7.17x | 3.01x |
  | 512×2048 | 7.72x | 6.60x | 3.61x |
  | 512×4096 | 10.44x | 10.48x | 5.28x |
  | 2048×4096 | 15.72x | 15.70x | 7.18x |
  | 4096×5120 | 16.69x | 16.59x | 7.65x |

  NVIDIA 代理证据，不能外推八芯。reference 无 `.item()` 同步（与 T1
  不同），天花板来自融合多个小 launch，代理大 shape 已见 16x 级。

## 已知风险与对策（按芯）

- 昆仑：已避开毒点组合；若 uni_sram 编译失败，第一刀
  `isCloseCoreTiling=True`（T36 E13 破墙先例），次选
  Vectorize/UnrollControl；不删逻辑恒真 mask（E23 idle-core 教训）；
  BLOCK 是唯一调参轴
- 燧原：kernel 内无运行期分支已满足；无 int64 metadata；若读数弱，
  检查标量循环展开度（T41 state_passing unrun 前科提示串行 scan 类
  结构风险，本题每条带独立、无跨 program 依赖，风险低于 T41）
- 华为：grid≤65535 已 cap；BLOCK_D 512 -42% 类教训（T39）不适用于
  1D 条带，若弱可试 BLOCK_D=128
- 天数/沐曦/海光：纯 1D 算术结构预期无障碍
- seqlen>1 场景平台若测大 seqlen，标量 t 循环为串行热点——届时
  vendor 轴：t 维并行化（窗口重叠用 shared 复用）

## 提交预算与止损（2026-09-03 定稿）

- 默认 5 发：S0 探路 → 最多 3 次 vendor 单变量 → 1 发回归储备；
  同指纹失败连 2 次提前停；昆仑崩溃族按平台侧故障协议处理（不计
  止损、封存等健康窗口、重载需用户当次授权）
- 首投排程：09-04 重置后第 2 发（T42 之后）

## 时间线

- 2026-09-03 21:51 契约锁定、S0 实现 + 远端 screening 8/8 通过 +
  基准；commit `07aaf2e`；未提交（额度 0/30）

## 平台首投结果（2026-09-04 01:12，submission 9372，daily_seq 2）

- 7/8 正确、`invalid_correctness`（昆仑 fail；失败 case 平台未透出）
- 逐芯：天数 13.728 / 沐曦 6.7885 / 燧原 0.0265（过正确性但远低门槛）/
  海光 11.096 / 昆仑 FAIL / 华为 0.0555（低于门槛）/ A 8.3215 / B 10.497
- 修复优先级：昆仑正确性（T36 预案：uni_sram→`isCloseCoreTiling=True`、
  BLOCK 唯一轴、不删恒真 mask）→ 华为性能（BLOCK_D 128/512 扫描）→
  燧原性能（展开度/launch）


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

## E1 三 vendor 重构（2026-09-04 05:0x，submission 9422，daily_seq 11）

- 结果**恶化**：燧原由「过但慢」转 correctness 失败（精确 2D grid +
  去 int64 cast 重构引入）；昆仑仍败（两连败，止损）；华为 0.0555→
  0.0255 更慢
- 逐芯：天数 13.723 / 沐曦 7.0445 / 燧原 FAIL / 海光 10.859 /
  昆仑 FAIL / 华为 0.0255 / A 9.002 / B 10.6205
- 判定：无失败 case 细节下的重构均为盲试，本题暂停；榜首
  EvokeAgent 6.8566x 证明可解，需要情报（case 详情/工单）再开工

## 失败情报破译 + E2/E3（2026-09-04 深夜，submissions 9520/9522）

- 情报（raw_result）：昆仑 S0/E1 = **1e35 级未初始化内存垃圾**（标量
  条件选址在该芯静默失效指纹）；燧原 E1 = grid.y>255；华为 S0 过
  正确性但 0.055x
- E2（9520）：全向量化 [BLOCK_D=256, BLOCK_T=64]——华为撞 UB 溢出
  （3745792 bits）、昆仑撞 uni_sram 编译墙、燧原 0.001x 大 tile 拖垮
- E3（9522，source `1e6ce02`）：三 vendor 缩到 BLOCK_D=64 + 昆仑
  isCloseCoreTiling 技法——**华为 correctness 翻绿 0.074x**（差门槛
  一步）；昆仑仍 uni_sram 墙；燧原仍 0.001x
- 判定：华为一步之遥（BLOCK_T=32/更多 program 可试）；燧原向量化
  形态错配需回退 S0 字节另寻性能轴；昆仑墙深（T36 同款从未过）

## E4 静态短轴形态（2026-09-04 深夜，submission 9550，daily_seq 26，Codex P1/P3/P4）

- SEQLEN/STATE_LEN/WIDTH 全 constexpr + static_range 展开——每 (t,k)
  单条 load 路径，零运行期标量选择（source `e4bd3a6`，ZIP
  `39c8c413…`；ascend 512-lane / kunlunxin 128-lane / enflame 512-lane
  纯 i32）
- **昆仑：1e35 垃圾 → 有界错误**（abs 7.7–17.4，rel 至 2.9e6；98%
  元素错）——单路选址已生效，读的是对的数据区域但值算错，疑
  bf16→fp32 cast 或非 2 幂 WIDTH 权重访存的 lowering 问题
- 华为 0.0675（< e3 向量化 0.074——静态形态未兑现 2–6x 预期，T 轴
  padding 论不成立于该芯）；燧原 0.0305（>0.001 但距 0.1 仍 3.3x）
- 判定：T43 今日停（4 发留给新假设）；下一假设候选：昆仑 weight
  访存改 [BLOCK_D, WIDTH_POW2] 2D tile 或 fp32 权重预转换；华为改
  persistent 整行结构；燧原回退 S0 后找 launch/并行轴

## E5/E6 昆仑核弹级修复尝试（2026-09-04 深夜，submissions 9559/9561）

- E5（9559）：权重转置 [WIDTH,dim] + 纯 int32 → **错误逐位不变**
- E6（9561）：wrapper 全量 fp32 预转换（kernel 零 cast）+ tl.sigmoid
  + 转置权重 → **错误仍逐位不变**（abs 7.765625, 94/96, rel 56.665）
- **结论：昆仑后端对 depthwise-conv FMA 标量链存在确定性数值错译**
  （e4 静态 constexpr/e5 转置+int32/e6 全 fp32+sigmoid 三种截然不同
  的实现产生相同错误值——不是选址、不是 cast、不是权重布局，
  是编译器生成了错误的算术逻辑）。T36 同族结论互证。
  **昆仑轴 conclusive 封轴**（六投：S0/E1/E2/E3/E4/E5+E6）
- 华为 0.0625（fp32 预转换有开销；e5 的 256-lane 静态 0.069 最优）
- 燧原 0.0165（BLOCK_D=64 降了——e4 的 512-lane 0.0305 更好，回退）
- 今日剩 3 发保留给明日 30 发弹药

## E7 宽度轴归约大突破（2026-09-05 凌晨，submission 9852，daily_seq 2）

- **华为 0.074→0.4435x（6x 跃升）+ 燧原 0.0305→0.325x（10.6x 跃升）——
  两大弱芯同时过 0.1x 门槛！**（source `7dae9ca`）
  宽度轴归约：wrapper cat state+x → [W_PAD, D_BLOCK] 2D tile →
  tl.sum(w*v, axis=0)——彻底替换标量 FMA 链 + 消灭 T 轴 padding
- 逐芯：天数 13.14 / 沐曦 7.06 / **燧原 0.325** / 海光 11.16 /
  **华为 0.4435** / A 7.74 / B 10.48；昆仑 uni_sram 编译墙
- **7/8**——此题全场 76 发仅 1 队 8/8，我们 7/8 已是第一梯队

## 2026-09-05 Codex 会诊作战方案（预注册）

平台证据修正：全场 EvokeAgent 已 8/8（6.8566x）——昆仑结构存在，
"7/8=最优可达"仅对我方历史形态成立。双墙重判：E4-E6 相同错误值锁定
loop-carried FMA 链错译；E7 的 uni_sram OOR 不能定位到宽度归约本身
（同 kernel 还含 [64,128] 状态 tile 与 sigmoid）。

**P1（首选，1-2 发）：昆仑三 kernel 拆分 + 3D 微 program rank-1 归约**
- 卷积 kernel：`grid=(seqlen, dim, batch)`，每 program 一个 (t,d,b)，
  仅持有 `[W_PAD]` 向量 + 一次 `tl.sum(v*w, axis=0)`；无 bias/silu/
  cast/状态更新、无 static_range(t/k)、无 grid-stride
- 后处理 kernel：flat BLOCK=1024（T53 昆仑骨架）做 bias/silu/cast
- 状态 kernel：`grid=(ceil(state_len/16), dim, batch)` rank-1 拷贝
- 防墙：无 loop-carried FMA（昆仑 rank-1 归约 T16 正确性实证）；最大
  活跃形状 `[W_PAD,128]`→`[W_PAD]`；3D 小 grid 有 pr40 昆仑先例
- 分流：正确且 ≥0.1x → 停；E4-E6 同款错误值 → 转 P2；正确但 <0.1x →
  转 P3；OOR → 读 raw_result 定位具体 kernel，不盲扫 BLOCK

**P2**：wrapper unfold 物化 `[D, B*T, W]` + 每通道权重扩展 + 规则 32³
IEEE `tl.dot`（只存 C[:,0]；T28/T37 昆仑通过范式）。
**P3**：`[BLOCK_D≤8, W_PAD]` 小二维归约（tile ≤64 元素）。
验证矩阵：width 2-8 / dim 7-2049 / seqlen 1-6 / 三 dtype / bias±silu /
非连续 + IR 检查（P1 无 scf.for；P2 含 ieee dot）。

## E8/E9 昆仑 rank-1 归约两连败（2026-09-06，submissions 10329/10332）

- E8（10329）：三 kernel 拆分 + 3D 微 program（每 program 一个 (t,d,b)
  仅 [W_PAD] + tl.sum）——**昆仑编译通过（uni_sram 墙破）**但 5 case
  数值失败（87-94% 元素错，abs ≤18.9，与 E4-E6 错译族同指纹）
- E9（10332）：3D grid 改 1D-flat + `//`/`%` 推导（grid 轴映射假设）
  ——昆仑仍数值失败，**假设证伪**：错译在 rank-1 窗口归约本身
- 其余七芯两发全过（E8 逐芯：天数 13.15/沐曦 7.02/燧原 0.327/
  海光 10.58/华为 0.4565/A 8.99/B 10.64）

## E10 规则 GEMM 形态（2026-09-06 00:49，submission 10337）

- Codex P2：wrapper unfold 物化窗口 [D, B*S, W] + 每通道一次规则
  32×32 ieee GEMM（B 操作数 = 权重向量跨 N 列广播，只存 C 第 0 列
  到 pre[b,d,t]）；bias/silu 与状态拷贝沿用 flat kernel。
  T45 e8/e9 已平台证明该 GEMM 家族在昆仑 correctness 通过
- screening：unittest 9/9 OK（首轮 unfold 切片漏 state_len+1-width
  起始偏移，sl>w-1 case 修错）；source `3a0b5cd`，ZIP
  `e10-3a0b5cd` SHA-256 `bbe24679…fbee`（4 成员）
- 晋级门：昆仑全 case 正确且 ≥0.1x → 8/8；同款错译 → 昆仑轴 conclusive

## E10 终态 + 共享 post kernel 破案（2026-09-06）

- E10（10337）：昆仑仍 5 case 数值失败；**决定性指纹证据——E8/E9/E10
  三种截然不同的 conv kernel（rank-1 3D / rank-1 flat / 规则 GEMM）在
  case 0 产生逐位相同的错误**（30/32、abs 4.47265625 @ (1,5)、rel
  7.619 @ (1,2)，case 1/2 亦高度同构）——错译不在 conv 计算，在三者
  **共享的 flat post kernel**
- 嫌疑收敛：post kernel 中唯一的向量 fp32 除法
  `v / (1.0 + tl.exp(-v))`——T45 的 tl.exp 在昆仑 correctness 全过
  （exp 无罪），T52 全程在纠缠除法形态（除法有前科）
- **E11（10338）**：单变量 silu 改 `v * tl.sigmoid(v)`（消除向量除法），
  其余字节冻结 E10；source `ccf229c`，ZIP `e11-ccf229c`
  SHA-256 `677c12f0…f6507`，screening 9/9 OK
- 判定门：指纹变化 → post kernel 坐实并继续分诊（下一变量=bias
  gather）；指纹不变 → 昆仑轴 conclusive 封存

## E12 终态与昆仑轴 conclusive 封存（2026-09-06，submission 10341）

- bias 折入 GEMM ones-列（post 无 gather，仅 silu+cast）——**指纹变化：
  case 0 从 30/32 → 32/32（100%），失败 case 5→4**：向量 gather
  确认是错误源之一，但有残留错误（嫌疑：GEMM 的 stride-0 广播 B
  操作数或 post 的 silu；两者均无昆仑单独正证据）
- 其余七芯全过（天数 13.71/沐曦 7.023/海光 10.9445/华为 0.3945/
  A 8.559/B 10.6055；燧原在评）
- **判定：T43 昆仑轴 conclusive 封存 7/8**。任务全史累计 10 种
  Triton 形态（标量 FMA 链×6、rank-1×2、规则 GEMM×2）在昆仑全部
  数值错译或编译墙；EvokeAgent 8/8 结构未破译。跨题知识：
  **昆仑向量 gather 与广播操作数均为错译高危面**（T53 标量 gather
  为唯一已证安全形态）

## E13 通道连续 affine 卷积（2026-09-07）

- 新证据：[FlagGems-sglang PR34 固定源码](https://github.com/flagos-ai/FlagGems-sglang/blob/e7f91a5f6c813d499275b3f3a6288e1b3b5dddc9/src/flaggems_sglang/runtime/backend/_kunlunxin/ops/causal_conv1d_fn.py)。沿其 `[time,channel]` 连续加载模式，将本题 state+x 物化为 `[B,L,D]`，weight 为 `[W,D]`。每 program 一个时间位置和连续 channel block，scalar 基址+arange，宽度静态累加，bias/sigmoid仍在 Triton。此为新数据布局，非旧FMA源码重投；PR不构成本候选昆仑通过证据。
- 保留 generic/Ascend/Enflame 字节，只替换 Kunlun 的窗口 gather+stride0广播GEMM+后处理；state保留独立Triton拷贝。无设备判断或torch算子计算fallback。
- source/verification commit `4fa854a376de167e76a1e5d1441c6cd82b5866d7`。screening 两方法（144边界subcases+12核心subcases）通过，提交前后源/测试SHA一致；完整release10方法通过，零fail/error/skip/xfail。
- 三dtype、identity/no-bias与silu/bias；W2/3/4/8、state较长、S>state、D127/128/129与1023/1024/1025、非连续x、state不变性及精确state输出。generic既有其他回归最大D2049；新Kunlun本轮最大D1025、S6，未将generic覆盖外推。
- 远端 `gpu:/tmp/flagos-b4-invalid.eYQz0q/t43-release`，RTX5070Ti/torch2.13.0+cu130/triton3.7.1；执行 `timeout 420 /home/kevin/notebook/.venv/bin/python .agents/skills/flagos-operator-race/scripts/verify_release.py run --directory /tmp/flagos-b4-invalid.eYQz0q/t43-release`。py_compile/Black/isort/flake8通过。
- 昆仑 MCP verify 本次HTTP502，未执行，`target-runtime-unverified`。七芯历史通过路径不能替代本次逐芯结果；E12燧原实际最终1830s运行态超时，纠正旧“7/8”措辞。
- 晋级：全芯正确且各≥0.1，先取得有效分；新布局需付transpose搬运开销，性能未知。止损：本候选最多一发；仍错则保存首个失败case/selected_file，不把同指纹当作平台reference故障，不用注释重投。
- ZIP `/Users/bytedance/ccc/flagos/artifacts/competition/causal_conv1d_update/e13-4fa854a/causal_conv1d_update.zip`，22281 bytes，SHA256 `465bd4baaa2c4251978cffc9f9ed8bbac467c1c82c466461f5ee7737ac642bba`；dry-run/release/final source manifest一致。
- 成员 `causal_conv1d_update.py` ← `src/flaggems_sglang/ops/causal_conv1d_update.py` SHA256 `0e62c67bd41c3ebf3d1a8ee12fc3eb6a4ba264dd94841b657f44e9c8df39bf35`。
- 成员 `causal_conv1d_update_ascend.py` ← `src/flaggems_sglang/runtime/backend/_ascend/ops/causal_conv1d_update.py` SHA256 `85cc6109f376ab786ca0a15cce634736617f60e625c5baa9d613c810eef38165`。
- 成员 `causal_conv1d_update_enflame.py` ← `src/flaggems_sglang/runtime/backend/_enflame/ops/causal_conv1d_update.py` SHA256 `57d4825f20b864d1722f8760a8707d0be5e90e5f12038a8fd05809f7f38d43d8`。
- 成员 `causal_conv1d_update_kunlunxin.py` ← `src/flaggems_sglang/runtime/backend/_kunlunxin/ops/causal_conv1d_update.py` SHA256 `8086def1f50ec326ab7634010b62edcf950e6e78f52995d8bde74320098d3f20`。
- 实际kernel launches `{'src/flaggems_sglang/ops/causal_conv1d_update.py': 69, 'src/flaggems_sglang/runtime/backend/_ascend/ops/causal_conv1d_update.py': 39, 'src/flaggems_sglang/runtime/backend/_enflame/ops/causal_conv1d_update.py': 39, 'src/flaggems_sglang/runtime/backend/_kunlunxin/ops/causal_conv1d_update.py': 78}`。
- 证据 `/Users/bytedance/ccc/flagos/artifacts/competition/causal_conv1d_update/e13-4fa854a/validation/verification.json` SHA256 `3f08d408cabffc4d93ef5e4429b0d5c284674c207fc686e5024d601577c82e03`。
- 证据 `/Users/bytedance/ccc/flagos/artifacts/competition/causal_conv1d_update/e13-4fa854a/validation/verification.log` SHA256 `6c2779725af453a05c2885a0305e285052d59286aff76c4079efe93b19857697`。
- 证据 `/Users/bytedance/ccc/flagos/artifacts/competition/causal_conv1d_update/e13-4fa854a/validation/verification-input.json` SHA256 `44e0417c0d258ef9e073a4dc99137b4628b615b81cace8bda2dd97249ad357ca`。
- 证据 `/Users/bytedance/ccc/flagos/artifacts/competition/causal_conv1d_update/e13-4fa854a/validation/43-kg-request.json` SHA256 `b53b6bdfd166ba3e9ffbe2ad46a07f53a785f735f3df8c56542e1732a17cd8cb`。
- 证据 `/Users/bytedance/ccc/flagos/artifacts/competition/causal_conv1d_update/e13-4fa854a/validation/43-kg-response.json` SHA256 `92afbbac89b844f85c246c3eeb9c290c53565de4ce8bded312b2553158970e9b`。
- 实时旧状态2026-09-07T12:08:44+08:00：submission10341，quota24/30；本轮其他候选已另消耗额度，正式submit以preflight为准。

## E13 平台结果（2026-09-07T12:29:32.414023+08:00）

- submission `10706`，daily_seq `9`，创建 `2026-09-07T12:28:30`；正式上传/提交各一次，远端ZIP SHA验签 `verified`。
- 状态 `valid`，平台average_speedup `6.545875`；quota `21/30`（本条观测时）。

| 芯片 | 正确性/状态 | 加速比 | 实际成员 |
|---|---|---|---|
| tianshu | True / completed | 13.7035 | causal_conv1d_update.py |
| muxi | True / completed | 7.0445 | causal_conv1d_update.py |
| enflame | True / completed | 0.326 | causal_conv1d_update_enflame.py |
| haiguang | True / completed | 11.177 | causal_conv1d_update.py |
| kunlunxin | True / completed | 0.407 | causal_conv1d_update_kunlunxin.py |
| huawei | True / completed | 0.3435 | causal_conv1d_update_ascend.py |
| card_a | True / completed | 8.837 | causal_conv1d_update.py |
| card_b | True / completed | 10.5285 | causal_conv1d_update.py |

- 平台实时题面接口确认 **my_rank=3**，my_best_speedup=6.545875；榜首AttentionImOnly2YearsOld 7.90325。PR34结构移植本次在昆仑过线：0.407x，燧原0.326x。只证明本候选和本题，不能把通道连续布局的收益泛化到所有算子。
- 证据 `/Users/bytedance/ccc/flagos/artifacts/competition/causal_conv1d_update/e13-4fa854a/validation/43-submit.json` SHA256 `dca04eaf49ae65b63e884cd85e4fcc3170cea9eabcadf89b4ff1cf7889a5896d`。
- 证据 `/Users/bytedance/ccc/flagos/artifacts/competition/causal_conv1d_update/e13-4fa854a/validation/43-status-now.json` SHA256 `a4cf5fc666a491c9950850e84a908bf6cdaf2b67316d22eea489c8a9cee21001`。

## E14 Top1 冲刺候选（2026-09-07，提交前）

多步卷积 `seqlen>1` 的 generic channel tile 从 256 改 128，单步仍 256；厂商文件不变。保持状态更新、累加和激活顺序。最终源码字节已与纯 wrapper 五轮交错 A/B 基准匹配。

代理端 `(B,D,S)`：`(4,1024,3)`、`(2,4096,16)` 为旧版的 1.50x；`(32,4096,4)` 1.00x；三个单步对照 0.985–1.00x。编译产物显示多步 tile 的 shared memory 512→0、寄存器 48→37。早期 constexpr dims/seqlen/state_len 方案在单步大 batch 回退至 0.753x，已丢弃；1 warp 方案也未采用。

预注册：本轴首投一次探测全芯实际收益；目标是超过 team best 6.545875，冲榜目标按当时榜首 7.90325×1.03≈8.14035。未接近均值目标则不因单 case 1.5x 盲目重复调参。

- source/verification commit：`5cce466f70090896819a3fb8f4abd93221faa14c`；ledger commit 为本节所属提交。
- 本地 py_compile、Black、isort、flake8 通过。NVIDIA release：10 方法、69 次 kernel launch，fail/error/skip/xfail 均为0。执行源：`src/flaggems_sglang/ops/causal_conv1d_update.py`。
- 测试源码 SHA256：`efad97ced382855e86dba4958362bdaf88a5be3c99fe12b04b1ddff329a48c7f`；各输入文件 SHA 见 verification-input.json。
- 远端 `gpu:/tmp/flagos-b4-top1.8nsvBC/t43-release`，RTX5070Ti / driver610.57.04 / Python3.12.13 / torch2.13.0+cu130 / triton3.7.1。串行后台执行：`timeout 600 /home/kevin/notebook/.venv/bin/python .agents/skills/flagos-operator-race/scripts/verify_release.py run --directory /tmp/flagos-b4-top1.8nsvBC/t43-release`。
- ZIP `/Users/bytedance/ccc/flagos/artifacts/competition/causal_conv1d_update/e14-5cce466/causal_conv1d_update.zip`，22363 bytes，SHA256 `120876dee438b4c3d37c84d2cbaea26196164ebd5094fa0f70eccad7d19d39bc`；dry-run 与最终 manifest 五项恒等字段全部匹配。
- ZIP member `causal_conv1d_update.py` SHA256 `ffa794b7cd13e604eef0c38fc897520eefaa63e1745c0cf3148ee00c6e67f464`。
- ZIP member `causal_conv1d_update_ascend.py` SHA256 `85cc6109f376ab786ca0a15cce634736617f60e625c5baa9d613c810eef38165`。
- ZIP member `causal_conv1d_update_enflame.py` SHA256 `57d4825f20b864d1722f8760a8707d0be5e90e5f12038a8fd05809f7f38d43d8`。
- ZIP member `causal_conv1d_update_kunlunxin.py` SHA256 `8086def1f50ec326ab7634010b62edcf950e6e78f52995d8bde74320098d3f20`。
- 证据 `/Users/bytedance/ccc/flagos/artifacts/competition/causal_conv1d_update/e14-5cce466/validation/verification.json` SHA256 `cebc75997528c24cfe212358988d827a24a0d39cda84306ab27429d3344c305b`。
- 证据 `/Users/bytedance/ccc/flagos/artifacts/competition/causal_conv1d_update/e14-5cce466/validation/verification.log` SHA256 `de83969cff940fd8b737decbbb998bc7591571b425c0e7e1c3ab58ddf42d0a17`。
- 证据 `/Users/bytedance/ccc/flagos/artifacts/competition/causal_conv1d_update/e14-5cce466/validation/verification-input.json` SHA256 `7b5afa43b903ba8fff7b87d33abb43e5f8212646f61469c878526d0d42a767be`。
- 附加基准、诊断及 MCP 证据清单 `/Users/bytedance/ccc/flagos/artifacts/competition/causal_conv1d_update/e14-5cce466/validation/evidence-sha256.json` SHA256 `b5ab669b6c45cfbcc0d1818f45f05c5aa3006459c28f0f0019cbe057f5a6cc52`。
- 首轮每题最多1次正式上传/提交；本次预算上限沿用批准的 T43/T51/T52 各4、T57 3、储备6，须有新证据才继续消耗。sending/uncertain/stale_after_upload 不自动重试。

## 本轮平台结果与止损（2026-09-07T14:55:36+08:00）

- submission `10745` / daily_seq `11` / created `2026-09-07T14:45:23`；preflight与上传/提交均只执行一次，远端ZIP验签 `verified`。
- 平台原始状态 `completed` / `valid`，通过8/8、终态8/8；average_speedup `6.5204375`，is_team_best `False`；观测时额度 `19/30`。
- E14 比最佳 E13 回退约0.389%，代理两个case的1.5x未迁移成全芯收益。后续离线 time-grid capped8 方案在既有10方法通过，但 `[32,4096,4]` 仅0.833x，三个其他常规多步case约1.0x，额外 `[1,128,65]` 3.475x不足以证明题目加速，故不发布该方案。

| 芯片 | 状态/正确性 | 加速比 | 实际文件 |
| --- | --- | ---: | --- |
| tianshu | completed/True | 13.6185 | `causal_conv1d_update.py` |
| muxi | completed/True | 7.013 | `causal_conv1d_update.py` |
| enflame | completed/True | 0.3255 | `causal_conv1d_update_enflame.py` |
| haiguang | completed/True | 11.299 | `causal_conv1d_update.py` |
| kunlunxin | completed/True | 0.409 | `causal_conv1d_update_kunlunxin.py` |
| huawei | completed/True | 0.3775 | `causal_conv1d_update_ascend.py` |
| card_a | completed/True | 8.452 | `causal_conv1d_update.py` |
| card_b | completed/True | 10.669 | `causal_conv1d_update.py` |

- 证据 `/Users/bytedance/ccc/flagos/artifacts/competition/causal_conv1d_update/e14-5cce466/validation/43-submit.json` SHA256 `c524d25b871c1c904c41e6e613313cbc253f60bafc1eb26bc8f70c874f77f420`。

- 证据 `/Users/bytedance/ccc/flagos/artifacts/competition/causal_conv1d_update/e14-5cce466/validation/43-status-first-round.json` SHA256 `1918d1d0d4e7e09860137d057b1b5c034a1a57e65628851501f08c555f86776f`。
- 未晋级后续实验 `/Users/bytedance/ccc/flagos/artifacts/competition/causal_conv1d_update/e14-5cce466/validation/followup-evidence-sha256.json` SHA256 `4e846306d3b972f1b6f196ab1eb1cb0f6314573c8aabfe51f2db4428940276fc`。

## E15 燧原通道连续布局 + 华为循环不变量外提（2026-09-07）

- 调研定位两处结构剩余：①燧原 width-reduce vendor 的窗口加载是
  `[W_PAD, BLOCK_D]` index-shift 形式（[B,D,L] 布局下逐 lane 跨 seqlen
  stride）——PR34 在 GCU 实测该形态比规范连续加载慢 ~100x，燧原 0.326x
  与此吻合；②华为 vendor 的 weight tile 与 bias 在 `tl.static_range
  (SEQLEN)` 内逐步重载，且输出走 fp32 缓冲 + wrapper `.to()` 整趟转换。
- 单变量两 vendor：`_enflame` 整体替换为昆仑 E13 同款通道连续
  `[B,L,D]` 双 kernel 形态（PR34 结构，输出转置 pass 保留）；`_ascend`
  weight/bias 外提 + 输出直存原 dtype（去掉 `.to()` 整趟）。generic 与
  `_kunlunxin` 字节冻结。
- source/verification commit `cbfae4f`；screening `/tmp/flagos-t43e15`
  （10/10）与 release `/tmp/flagos-t43e15-rel`（10/10 方法、0 fail/skip，
  generic 69 / ascend 39 / enflame 78 / kunlunxin 78 launch 实跑）双绿；
  black/isort/flake8 在两 vendor 字节上分别全过。
- 预注册门：平台 avg 超 team best 6.545875 才晋级（燧原 0.326→预期
  主受益；华为微优化幅度小）；未超则收轴。
- ZIP `e15-cbfae4f`，SHA256
  `59abbc45f97c4049d34d28e8d1351439c3b4ef31ccb57e0cb9950cba48136b94`；
  成员 4（generic/kunlunxin 字节与 e13 一致；ascend
  `a3a5cd981d492def26fef2745a527a22e4dce207429dfb8566386e927332a49d`、
  enflame `01801b0a91726f5c98b6c9f91e05a5ee35874b4a541f2e93b7718a9dbe454319`）。
- 证据 `validation/verification.json` SHA256
  `200ff974a7bd0536cd49bc68af3679083dbd98543c963d8ec01b5aef4fa7b080`、
  `validation/verification.log`（相邻完整日志随回执归档）。

## E15 平台终态与处置（2026-09-07T18:2x）

- submission `10789`：invalid_correctness——**燧原 5 case 全部
  `Pipeline run failed: PassManager execution failed`**（通道连续形态
  不过 GCU 编译器；与"燧原编译路径对 kernel 形态敏感"知识吻合，
  PR34 结构在 fn 算子可用不代表 update 算子形态可用）。
- 其余七芯全过：天数 13.4375 / 沐曦 7.0025 / 海光 11.382 / 昆仑 0.406 /
  **华为 0.484（微优化 +41% 兑现：weight/bias 外提 + 输出直存原
  dtype 去 `.to()` 整趟）** / A 8.0865 / B 10.47。
- 即使燧原按旧 0.326 通过，均值 6.449 也低于 team best 6.545875
  （A 芯 8.84→8.09 等水位下漂 +0.14 华为增益无法覆盖）→ **本轴收券，
  不追投**；树回滚 `_enflame` 至 e13 已验证字节（成员 SHA
  `57d4825f20b864d1722f8760a8707d0be5e90e5f12038a8fd05809f7f38d43d8`），
  **保留 `_ascend` 外提改进**（平台实证 +41%，为后续候选打底）。
- 跨题知识：①昇腾"wrapper 整趟 dtype cast 消除"是真实杠杆（本题为
  输出流量主导型算子）；②燧原 PassManager 对通道连续双 kernel 形态
  （含 `enable_fp_fusion=False` kwarg 与 permute 物化 wrapper）编译
  失败，update 形态迁移需单变量拆分验证。
- 证据 `validation/43-status-final.json`（原始逐芯记录）。

## 2026-09-08 推荐方案实现与提交前验证（未提交平台）

generic width2/3/4 多 token 用寄存器滚动历史和权重复用；长序列/stride/输入 state 不变测试通过。完整 wrapper 配对约 0.974–1.000x，未形成稳定性能收益。

- source `8ba31a102f4ef0430c08f12c4622b27430915071`；verification `8ba31a102f4ef0430c08f12c4622b27430915071`。11 个测试方法、96 次实际 kernel 调用；选定 NVIDIA/代理范围门禁通过。
- 回执 `artifacts/competition/batch4-implementation-20260907/t43-release1/verification.json`，SHA256 `9210e672569db770418288a82c50b69a89d2f7de90763de7867dda4cbda51212`；日志 SHA256 `5b6652e3d3229686e2a22a8dbc6283aced00127e7887971660e1463c05807706`。
- 不可变 ZIP `artifacts/competition/causal_conv1d_update/research-20260908-8ba31a1/causal_conv1d_update.zip`，SHA256 `9308dfc04f2b619f7071340b7e31020e3ff6a8e917e7f2d9748219b6d8124592`；与 dry-run manifest、构建和 existing 验签一致。ZIP 是候选产物，不等于目标芯或平台已通过。
- 环境、逐源码执行范围、原始配对数据和未完成条件见[本轮报告](../implementation-batch4-20260908.md)及[证据清单](../data/batch4-implementation-20260908.json)。本轮不更新历史有效分，未做平台 preflight、上传或正式提交。
