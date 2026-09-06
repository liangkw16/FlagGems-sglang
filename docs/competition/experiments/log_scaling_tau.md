# Task 57 `log_scaling_tau` 实验记录

```current
task: 57
operator: log_scaling_tau
batch: 4
validity: pending
platform: 7/8(s0,燧原pending)
team_best_stage: s0
team_best_speedup: 0
sealed: no
next: 燧原pending;昆仑0.47/华为0.53偏弱;榜首2.82x
updated: 2026-09-06
```

## S0 2D grid 行缩放（2026-09-06，远端 GPU 2/2 OK）

- 形态：grid `(rows, cdiv(n_cols, BLOCK))`，BLOCK=min(next_pow2(n_cols),1024)；
  fp32 乘 tau 后 cast 回 x.dtype；任意尾部维度按 contiguous 展平成行。
- 远端 5070 Ti：2/2 OK（平台 5 shape + fp32 附加）。

## S0 平台进行中（2026-09-06，seq25）

- 已过 7：天数 3.97 / 沐曦 2.71 / 海光 4.70 / 昆仑 0.53 /
  华为 0.47 / A 3.32 / B 2.61；燧原评测中。
