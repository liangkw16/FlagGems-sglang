# Task 55 `hc_head` 实验记录

```current
task: 55
operator: hc_head
batch: 4
validity: pending
platform: 7/8(s0,昆仑=评测器崩溃族非代码);e1已提交待裁
team_best_stage: s0
team_best_speedup: 0
sealed: no
next: e1(0052424)循环外归约重写已提交(2026-09-07)评测中;门:燧原>=0.31x(≥20%)
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
