# Task 100 `pad_draft_extend_query` 实验记录

```current
task: 100
operator: pad_draft_extend_query
batch: 7
validity: valid
platform: s0(21440)valid 7/7 avg8.43:天数20.6/沐曦4.5/海光11.1/昆仑2.8/华为3.7/A8.5/B7.8;稠密芯rank13-15均匀落后榜首12.74约1.5x(与T95同型结构差)
candidate_stage: s0
team_best_stage: s0
team_best_speedup: 见platform行
sealed: no
next: E1轴=clone/scatter结构(1.5x均匀差之谜,同T95);kunlun#5/huawei#7可vendor
updated: 2026-09-25
```

## 不可变身份（s0，2026-09-25 第二批释放夜）

- 四题同批提交（21439/21440/21442/21443），review 5 项 P2 全修后门禁全绿。
- release 回执：`artifacts/competition/b7b-s0-release-20260925/pad_draft_extend_query/verification.json`。
- ZIP：`artifacts/competition/pad_draft_extend_query/s0-*/pad_draft_extend_query.zip`（T102 绑定 2cad7b0d，其余 f5b7add4）。
