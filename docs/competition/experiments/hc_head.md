# Task 55 `hc_head` 实验记录

```current
task: 55
operator: hc_head
batch: 4
validity: invalid_correctness
platform: 7/8(e1,昆仑=评测器崩溃族第2次;七芯总分+40%)
team_best_stage: -
team_best_speedup: 0
sealed: no
next: 昆仑对该题reference确定性崩溃(s0/e1同指纹,同日昆仑评过T48/T49/T53/T56);等平台窗口同字节重试;燧原0.30x仍弱
updated: 2026-09-07
```

## S0 每 token 两遍融合（2026-09-06，远端 GPU 1/1 OK）

- 形态：grid=(T,)；pass1 单遍同时累积 sumsq（RMSNorm）与
  mixes_raw（x·hc_fn[j] 3D tile [HC,HC,BH] 嵌套 tl.sum）；
  sigmoid 门控；pass2 重读 x（L2 热）折叠 hc_mult 轴写 y。
- BLOCK_H 限 4096//(HC²)；m 轴全 mask 防 OOB。
- 远端 5070 Ti：1/1 OK（T∈{1,3,4,8,16,33}×hc_mult∈{2,4,8}×hidden∈
  {128,256,320,512,1024,2048}，bf16 1.5e-2）。

## S0 平台进行中（2026-09-06，seq26）

- 已过 7：天数 1.59 / 沐曦 1.23 / 燧原 0.257 / 海光 3.22 /
  华为 2.04 / A 0.97 / B 4.45；昆仑评测中。

## S0 昆仑终态（2026-09-07 核实）

- seq26 昆仑失败详情：`执行超时(1830s/1800s) + Subprocess crash:
  Fatal Python error: Aborted`（torch inductor compile_worker 栈），
  属已知昆仑评测器崩溃族，不计代码止损；七芯成绩维持
  天数 1.59 / 沐曦 1.23 / 燧原 0.257 / 海光 3.22 / 华为 2.04 /
  A 0.97 / B 4.45。

## E1 扁平累加器 + 循环外归约（2026-09-07 已提交）

- 参照官方 FlagGems `hc_head_fused_kernel` 结构：`sqr_acc[BLOCK_H]` 与
  `mix_acc[HC,BLOCK_H]` 逐 lane 累加，fn 以 `[HC,BLOCK_H]` 2D tile 按
  输入行 m 展平流式载入，全部 `tl.sum` 移出 hidden 循环；第二遍加权
  输出不变。BLOCK_H 预算从 `[HC,HC,BH]≤4096` 改为 `HC*BH≤2048`
  （HC=8→256 / HC=4→512 / HC=2→1024），保持单 kernel。
- NVIDIA 代理 wrapper 基准（RTX 5070 Ti，中位数）：
  (T=33,hc=4,h=2048) 1.46x、(T=64,hc=8,h=2048) 1.52x、
  (T=8,hc=8,h=1024) 1.39x、(T=16,hc=2) 1.05x，全 shape 无回退；
  screening 1/1、release 6 launches 0 失败。
- source `0052424`，ZIP `e1-0052424` SHA-256
  `cf34629620dd7b945a428a06ca0f022bbf7218b0e80e8cbdef77674649d45cde`；
  2026-09-07 提交评测中。燧原（0.257x 弱芯）目标 ≥20%；昆仑同时重试。

## E1 终态（2026-09-07，submission 10668）

- **7/8，昆仑第 2 次同一崩溃指纹**（`compile_worker Aborted`，与 s0
  seq26 完全一致）；同日昆仑评测器正常完成 T48/T49/T53/T56 本队提交，
  证明是 hc_head 题评测侧（reference/inductor）确定性崩溃，非我方
  Triton 内核问题。按崩溃族协议不计代码止损。
- 七芯 vs s0：天数 1.38（-13%）/ 沐曦 1.10（-11%）/ 燧原 0.2995
  （+16.5%，未过 20% 门）/ 海光 2.156（-33%）/ **华为 3.153（+54%）/
  A 2.1815（+125%）/ B 9.088（+104%）**；七芯总和 13.76→19.36（+40%）。
  若昆仑恢复，e1 字节即为大幅 team best 候选；同字节重试不需新候选。
- 结构结论：循环外归约在华为/A/B 大幅兑现，海光/天数/沐曦小回退，
  与官方 FlagGems 结构在该三芯的 lowering 差异待后续分轴。
