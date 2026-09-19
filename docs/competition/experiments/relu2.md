# Task 89 `relu2` 实验记录

```current
task: 89
operator: relu2
batch: 6
validity: valid
platform: completed(18314,e1,8/8,2.189x新TB;燧原streaming 0.8→2.0 +150%)
candidate_stage: e1
team_best_stage: e1
team_best_speedup: 2.189
sealed: no
next: 流式elementwise;轴=燧原0.8/昆仑0.8/华为1.1 vendors(streaming配方);榜首差距55%较大
updated: 2026-09-20
```

## 2026-09-20 S0/E1 首发记录

- 开发+验证+codex-review 门后 00:12-00:27 发射（详见
  `artifacts/competition/nx-s0-20260920/`）。

## 2026-09-20 E1 平台终态：燧原 8192 档 +150%，TB 2.189

- submission **18314** completed/valid，8/8，均值 **2.189 新 TB**（vs 2.043，
  +7.1%）。**燧原 0.8→2.0（+150%，T76 8192 配方直移兑现）**；华为 1.1→1.0
  （persistent 无增益）；榜首 3.17 差 45%。
