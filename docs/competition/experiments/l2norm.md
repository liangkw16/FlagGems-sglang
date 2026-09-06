# Task 56 `l2norm` 实验记录

```current
task: 56
operator: l2norm
batch: 4
validity: valid
platform: 8/8(s0,3.10497917x)
team_best_stage: s0
team_best_speedup: 3.10497917
sealed: no
next: 冲分轴:昆仑0.578x/华为1.38x/沐曦2.54x;榜首EvokeAgent 3.89x差20%
updated: 2026-09-06
```

## S0 → **8/8 VALID**（2026-09-06，submission 10405）

- **8/8 valid，avg 3.1050x team best**（一发命中！）
- 逐芯：天数 6.66 / 沐曦 2.54 / 燧原 1.19 / 海光 4.54 /
  昆仑 0.58 / 华为 1.38 / A 3.73 / B 4.21
- Per-row fp32 sum-of-squares → rsqrt → scale，cast back
- 榜首 EvokeAgent 3.89x，差距 20%
