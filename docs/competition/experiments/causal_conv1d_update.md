# Task 43 `causal_conv1d_update` 实验记录

```current
task: 43
operator: causal_conv1d_update
batch: 4
validity: valid
platform: 8/8(e13,6.545875x,排名3;昆仑首次0.407x)
team_best_stage: e13
team_best_commit: 4fa854a376de167e76a1e5d1441c6cd82b5866d7
team_best_speedup: 6.545875
sealed: no
next: 保持有效解;后续只按全芯均值收益排序,勿重试旧昆仑gather/GEMM轴
updated: 2026-09-07
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
