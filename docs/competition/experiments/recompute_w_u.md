# Task 103 `recompute_w_u` 实验记录

```current
task: 103
operator: recompute_w_u
batch: 7
validity: valid
platform: e2(21522)evaluating:六芯已过(天数6.47/沐曦3.47/海光14.5/A13.2/华为1.19/B0.79),昆仑waiting>25min(Kahan双核64列FMA或极慢/排队);一发IEEE全芯PassManager墙;二发fp64被XPU降级(37元素);generic=首发bf16-dot字节(6/7芯过)
candidate_stage: s0
team_best_stage: s0
team_best_speedup: 见platform行
sealed: no
next: watch 21522 昆仑;若超时/失败,Kahan列展开在昆仑不可行→封存该芯(6/7芯读数健康,generic字节最优)或减列分块再试
updated: 2026-09-26
```
