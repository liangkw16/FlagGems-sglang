# Task 88 `post_reorder_cutlass` 实验记录

```current
task: 88
operator: post_reorder_cutlass
batch: 6
validity: valid
platform: completed(18320,e2,8/8,15.227x<TB;hygon warps16 -26%保e1)
candidate_stage: e2
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

## 2026-09-20 E2 平台终态：hygon warps16 大负，TB 保 16.916

- submission **18320** completed/valid，8/8，均值 15.227 < TB 16.916（保
  s0）。**海光 45.3→33.6（-26%，warps16 第二个负例——warps 分化确认按
  op×chip 组合**）；昆仑 3.1→3.5；华为 12.1→13.4。轴关闭。
