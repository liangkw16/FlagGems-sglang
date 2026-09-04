# Task 53 `fused_gdn_gating` 实验记录

```current
task: 53
operator: fused_gdn_gating
batch: 4
validity: valid
platform: 8/8(e1,1.769075x)
team_best_stage: e1
team_best_commit: TO_FILL
team_best_speedup: 1.769075
sealed: no
next: 守榜;冲分轴:昆仑0.72x/燧原0.17x/沐曦1.34x
updated: 2026-09-05
```

## S0（2026-09-05 凌晨，submission 9866）

- 全 8 芯 correctness 通过；昆仑 0.0625x < 0.1x → invalid_threshold
- 逐芯：天数 2.88 / 沐曦 1.31 / 燧原 0.165 / 海光 2.79 / 昆仑 0.0625 /
  华为 1.19 / A 2.53 / B 2.73

## E1 昆仑向量化 vendor → **8/8 VALID**（submission 9877）

- 昆仑 vendor：标量循环→1024-lane 向量化（flat [B*H] 索引 + wrapper
  contiguous+view(-1)）
- **昆仑 0.0625→0.7168x（11.5x 跃升）** → **8/8 valid，avg 1.7691x**
- 逐芯：天数 2.33 / 沐曦 1.34 / 燧原 0.17 / 海光 2.98 / 昆仑 0.72 /
  华为 1.26 / A 2.65 / B 2.70
