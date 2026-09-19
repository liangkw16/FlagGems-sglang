# Task 91 `tiny_k_gemm` 实验记录

```current
task: 91
operator: tiny_k_gemm
batch: 6
validity: valid
platform: completed(18345,e7,8/8,1.953x=TB;metax16/kunlunw8中性,阶梯到顶)
candidate_stage: e7
team_best_stage: s0
team_best_speedup: 1.953
sealed: no
next: metax BLOCK_N32修复smem后#2;距榜首3.4%:轴=BLOCK_N阶梯(64于非K256形状)/m16rows;昆仑1.1/华为0.8 vendor
updated: 2026-09-20
```

## 2026-09-20 S0/E1 首发记录

- 开发+验证+codex-review 门后 00:12-00:27 发射（详见
  `artifacts/competition/nx-s0-20260920/`）。

## 2026-09-20 E2 平台终态：双 vendor 中性，TB 保 1.953

- submission **18308** completed/valid，8/8，均值 1.951 ≈ TB 1.953（保 e1）。
  沐曦 1.8（warps8 无增益）、燧原 0.6→0.7（微）；B 2.8。距榜首 2.016 差
  3.3%，缺口在沐曦 1.8 vs 2.4 与 B 2.8 vs 3.0——无已知配方，关闭本轴。

## 2026-09-20 E7 平台终态：BLOCK_N/warps 阶梯到顶，TB 保 1.953

- submission **18345** completed/valid，8/8，均值 1.953 = TB（持平）。沐曦
  1.7（16 档无增益）、昆仑 1.1（warps8 无增益）、华为 0.8→0.9；BLOCK_N
  128 因 smem 149KB 不可发（K=256 tile 物理上限）。T91 全部已知轴关闭。
