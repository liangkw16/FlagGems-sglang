# Task 43 `causal_conv1d_update` 实验记录

```current
task: 43
operator: causal_conv1d_update
batch: 4
validity: invalid
platform: 7/8(s0,kunlun correctness失败;huawei 0.0555x/enflame 0.0265x低于门槛)
team_best_stage: s0
team_best_commit: 07aaf2e5a081e4bfc4bb0ce3207b3e78c91d69df
team_best_speedup: -
sealed: no
next: 昆仑正确性修复优先(isCloseCoreTiling/BLOCK轴);华为/燧原性能轴其后——三芯全修才有8/8
updated: 2026-09-03
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
