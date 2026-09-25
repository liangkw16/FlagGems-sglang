# Task 101 `post_reorder_deepgemm` 实验记录

```current
task: 101
operator: post_reorder_deepgemm
batch: 7
validity: valid
platform: s0(21443)valid 7/7 avg10.02:昆仑11.4(#2)/华为12.3(#2)/海光15.0/天数11.0;均值距榜首11.10一步之遥
candidate_stage: s0
team_best_stage: s0
team_best_speedup: 见platform行
sealed: no
next: E1轴=天数#4/沐曦#4/A,B#5的gather带宽(BLOCK/warps微调);昆仑华为已前二
updated: 2026-09-25
```

## 不可变身份（s0，2026-09-25 第二批释放夜）

- 四题同批提交（21439/21440/21442/21443），review 5 项 P2 全修后门禁全绿。
- release 回执：`artifacts/competition/b7b-s0-release-20260925/post_reorder_deepgemm/verification.json`。
- ZIP：`artifacts/competition/post_reorder_deepgemm/s0-*/post_reorder_deepgemm.zip`（T102 绑定 2cad7b0d，其余 f5b7add4）。
