# Task 52 `fused_dual_residual_rmsnorm` 实验记录

```current
task: 52
operator: fused_dual_residual_rmsnorm
batch: 4
validity: invalid
platform: 6/8(e4,燧原两核拆分也不行;精度天花板)
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

## E3 AMD reciprocal vendor → card_b 翻绿（submission 10143）
- **card_b PASS 8.09x**（reciprocal 形式恢复 S0 通过路径）
- tianshu PASS 8.62（direct / generic 继续生效）
- **6/8**：天数/沐曦/海光/华为/A/card_b 全过
- 仅燧原+昆仑败（同款 1-4/33M 精度边界，两种除法形式均不过）

## E4 燧原两核拆分（submission 10151）
- 燧原仍败——memory 物化中间 bf16 也没修复 1/33M 边界翻转
- **判定：T52 6/8 是当前方法的天花板**（燧原+昆仑均不可修）
