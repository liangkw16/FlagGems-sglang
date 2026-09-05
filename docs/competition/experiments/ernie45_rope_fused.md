# Task 49 `ernie45_rope_fused` 实验记录

```current
task: 49
operator: ernie45_rope_fused
batch: 4
validity: invalid
platform: 7/8(e2,昆仑uni_sram墙已3投;conclusive)
team_best_stage: e1
team_best_speedup: -
sealed: no
next: 昆仑uni_sram墙(需更小tile/单head);7/8已是好成绩
updated: 2026-09-05
```

## S0: 6/8（燧原PassManager + 昆仑uni_sram）
## E1（precomputed-pos vendor）: **7/8**（燧原翻绿0.552x！昆仑仍uni_sram）
- 逐芯：天数 16.24 / 沐曦 8.85 / **燧原 0.552** / 海光 9.33 /
  昆仑 FAIL / 华为 2.48 / A 22.14 / B 9.88
- Precomputed-pos vendor：wrapper PyTorch 预计算 [T, half_rd] 位置
  张量（纯元数据），kernel 变纯 gather+RoPE 零分支——PassManager
  毒点彻底消除

## E2 昆仑 HEADS_TILE=1 + coreTiling（submission 10035）
- 昆仑仍 uni_sram（3 投：S0 generic / e1 precomputed-pos / e2 最小
  tile+coreTiling）→ **昆仑 conclusive 封轴**
- 逐芯：天数 16.26 / 沐曦 8.98 / 燧原 0.577 / 海光 9.32 /
  昆仑 FAIL / 华为 2.47 / A 21.54 / B 9.87
