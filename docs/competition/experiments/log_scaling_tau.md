# Task 57 `log_scaling_tau` 实验记录

```current
task: 57
operator: log_scaling_tau
batch: 4
validity: pending
platform: 0/0
team_best_stage: s0
team_best_speedup: 0
sealed: no
next: s0 待提交;榜首 2.82x(8/8)已4队过线,空间有限但补有效数
updated: 2026-09-06
```

## S0 2D grid 行缩放（2026-09-06，远端 GPU 2/2 OK）

- 形态：grid `(rows, cdiv(n_cols, BLOCK))`，BLOCK=min(next_pow2(n_cols),1024)；
  fp32 乘 tau 后 cast 回 x.dtype；任意尾部维度按 contiguous 展平成行。
- 远端 5070 Ti：2/2 OK（平台 5 shape + fp32 附加）。
