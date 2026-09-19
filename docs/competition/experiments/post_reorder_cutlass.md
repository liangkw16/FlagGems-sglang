# Task 88 `post_reorder_cutlass` 实验记录

```current
task: 88
operator: post_reorder_cutlass
batch: 6
validity: valid
platform: completed(18312,e1,8/8,16.643x<TB;燧原streaming中性保s0)
candidate_stage: e1
team_best_stage: s0
team_best_speedup: 16.916
sealed: no
next: slot静态展开fp32合并;海光48.9领先;轴=燧原1.1vs?/昆仑3.1 vendor;TOPK展开已用
updated: 2026-09-20
```

## 2026-09-20 S0/E1 首发记录

- 开发+验证+codex-review 门后 00:12-00:27 发射（详见
  `artifacts/competition/nx-s0-20260920/`）。

## 2026-09-20 E1 平台终态：燧原 streaming 中性，TB 保 s0 16.916

- submission **18312** completed/valid，8/8，均值 16.643 < TB 16.916（保 s0）。
  燧原 1.1（24-SIP 配方在 token-row 形态无增益——与 T79 gather 反例一致）；
  沐曦 13.7→14.7；海光 48.9→45.3（窗口）。燧原 vs 榜首 18.4 结构未破译。
