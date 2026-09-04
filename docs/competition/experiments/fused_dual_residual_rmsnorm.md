# Task 52 `fused_dual_residual_rmsnorm` 实验记录

```current
task: 52
operator: fused_dual_residual_rmsnorm
batch: 4
validity: invalid
platform: 5/8(e2,天数/燧原/昆仑/card_b 差丝米级精度)
team_best_stage: -
team_best_speedup: -
sealed: no
next: 精度追踪边际递减(3种除法形式各自修一芯破另一芯);建议暂停转T49
updated: 2026-09-05
```

## S0（reciprocal: v*(1/rms)*w）: 5/8
- 天数✗(1/16.7M) 燧原✗(1/33.5M) 昆仑✗(4/33.5M) card_b✓

## E1（div_rn: div_rn(v,rms)*w）: 5/8（tianshu✓ card_b✗）
- 天数✓ 燧原✗ 昆仑✗ card_b✗(NEW!)

## E2（plain /: v/rms*w）: 5/8（tianshu✓ card_b仍✗）
- 天数✓ 燧原✗ 昆仑✗ card_b✗
- 结论：三种除法形式各自修一芯破另一芯，1/33M 元素在 bf16 舍入边界
