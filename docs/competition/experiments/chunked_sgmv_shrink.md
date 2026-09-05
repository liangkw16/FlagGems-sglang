# Task 48 `chunked_sgmv_shrink` 实验记录

```current
task: 48
operator: chunked_sgmv_shrink
batch: 4
validity: invalid
platform: 7/8(e5,燧原评测机连续4次超时=平台侧;昆仑1.77x+七芯全过)
team_best_stage: e2
team_best_speedup: -
sealed: no
next: 燧原评测机恢复后重投(e1字节已验证0.58x);等平台侧窗口
updated: 2026-09-05
```

## S0: 6/8（燧原+昆仑败）
## E1（燧原 route/materialize）: 7/8（燧原翻绿0.58x，昆仑败）
## E2（+昆仑 route/materialize + long）: 7芯已过含昆仑1.764x，燧原评测中

## E3-E5 燧原超时系列（2026-09-05，submissions 9986/9998/10002）
- E3（BK=512 revert）：燧原 1830s 超时
- E4（最小变更重掷）：燧原 1830s 超时
- E5（恢复 e1 原版字节）：燧原仍在评（4th 连续超时）
- **判定：燧原评测机持续繁忙（e1 同字节已过 0.58x）**；等平台侧
  窗口恢复后重投
- 七芯稳定：天数 3.74 / 沐曦 5.39 / 海光 6.09 / 昆仑 1.77 /
  华为 6.26 / A 7.04 / B 6.69
