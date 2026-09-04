# Task 51 `fla_layernorm_gated` 实验记录

```current
task: 51
operator: fla_layernorm_gated
batch: 4
validity: valid
platform: 8/8(e1,5.390025x)
team_best_stage: e1
team_best_speedup: 5.390025
sealed: no
next: 守榜;冲分轴:昆仑0.944/华为2.52
updated: 2026-09-05
```

## S0（2026-09-05，submission 9867）
- 7/8，燧原 1830s 超时（非 correctness），其余七芯全过

## E1 燧原 vendor → **8/8 VALID**（submission 9944）
- 燧原 vendor：一 program 一行（去 grid-stride 循环）+ tl.rsqrt/tl.sigmoid
- **燧原超时→2.3398x**；逐芯：天数 9.21 / 沐曦 4.57 / 燧原 2.34 / 海光 7.74 /
  昆仑 0.944 / 华为 2.52 / A 8.73 / B 7.07
