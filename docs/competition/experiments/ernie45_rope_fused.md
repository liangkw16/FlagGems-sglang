# Task 49 `ernie45_rope_fused` 实验记录

```current
task: 49
operator: ernie45_rope_fused
batch: 4
validity: invalid
platform: 7/8(e2,昆仑uni_sram墙已3投;e3新结构已提交待裁)
team_best_stage: e1
team_best_speedup: -
sealed: no
next: e3(5a6b8cb)昆仑pair-per-program已提交(2026-09-07 seq2)评测中;门:昆仑>=0.12x
updated: 2026-09-07
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

## E3 昆仑 pair-per-program（2026-09-07 已提交）

- 借用官方 PR40 `mrope_fused` 昆仑工作划分：3D grid
  `(tokens, head_groups, pairs)` + 标量位置/cos/sin + `BLOCK_HEADS=8` +
  `num_warps=1` + 尾部独立 copy kernel；仅换轴规则为 ERNIE 的 H/W
  奇偶交替 + 尾部 T（mrope 的连续 T|H|W 段不适用）。
- 与 e2（HEADS_TILE=1，整段旋转向量仍驻留）不同，每 program 活跃状态
  缩至 8 lane 标量对，直接针对 uni_sram 墙；grid 总数继承 PR40 已过
  昆仑的证据（未在本地复证 65535 上限，风险随平台裁决记录）。
- variants 矩阵补非 128 head size、t 段切分与 bf16；NVIDIA 代理
  screening 5 tests / 3 sources 通过，release 回执（7 launches ×
  50）0 失败。source `5a6b8cb`，ZIP `e3-5a6b8cb` SHA-256
  `36db259a70a94e5a950f6a96b0c016c9cee139a7b1f3637d664e80aaba1e1c96`。
- 2026-09-07 seq2 提交评测中；晋级门：昆仑 ≥0.12x 且其余七芯不回退。
