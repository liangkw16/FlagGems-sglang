# Task 55 `hc_head` 实验记录

```current
task: 55
operator: hc_head
batch: 4
validity: pending
platform: 7/8(s0,昆仑pending)
team_best_stage: s0
team_best_speedup: 0
sealed: no
next: 昆仑pending;燧原0.257x弱;榜首6.25x
updated: 2026-09-06
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
