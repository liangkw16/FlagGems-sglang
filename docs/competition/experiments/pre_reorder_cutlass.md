# Task 102 `pre_reorder_cutlass` 实验记录

```current
task: 102
operator: pre_reorder_cutlass
batch: 7
validity: valid
platform: s0(21442)valid 7/7 avg9.60:昆仑5.8(#2)/华为4.6(#3);榜首21.2差2.2x(结构未破译)
candidate_stage: s0
team_best_stage: s0
team_best_speedup: 见platform行
sealed: no
next: E1轴=clone规避或批量行结构;稠密芯#5需带宽结构证据
updated: 2026-09-25
```

## 不可变身份（s0，2026-09-25 第二批释放夜）

- 四题同批提交（21439/21440/21442/21443），review 5 项 P2 全修后门禁全绿。
- release 回执：`artifacts/competition/b7b-s0-release-20260925/pre_reorder_cutlass/verification.json`。
- ZIP：`artifacts/competition/pre_reorder_cutlass/s0-*/pre_reorder_cutlass.zip`（T102 绑定 2cad7b0d，其余 f5b7add4）。
