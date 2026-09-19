# Task 90 `sigmoid_gate_mul_broadcast` 实验记录

```current
task: 90
operator: sigmoid_gate_mul_broadcast
batch: 6
validity: valid
platform: completed(18336,e2,8/8,2.495x微幅新TB;hygon回退后恢复)
candidate_stage: e2
team_best_stage: e2
team_best_speedup: 2.495
sealed: no
next: 行门控HDIM静态展开;轴=燧原0.8/华为1.2(warps/persistent配方);昆仑0.8贴门槛
updated: 2026-09-20
```

## 2026-09-20 S0/E1 首发记录

- 开发+验证+codex-review 门后 00:12-00:27 发射（详见
  `artifacts/competition/nx-s0-20260920/`）。

## 2026-09-20 E1 平台终态：燧原 +12% 但均值持平，TB 保 2.472

- submission **18315** completed/valid，8/8，均值 2.433 < TB 2.472（保 s0）。
  燧原 0.8→0.9（+12%）；**海光 4.2→3.8（-10%，warps16 在此题为负——与
  T86 正例互补，warps 分化按 op 而非只按芯）**；华为 persistent 无增益。

## 2026-09-20 E2 平台终态：hygon 回退兑现，TB 2.495

- submission **18336** completed/valid，8/8，均值 **2.495 微幅新 TB**（vs
  2.472）。海光 3.8→4.3（回退 warps16 后恢复）；燧原 0.9 持平。距榜首
  3.21 差 29%。
