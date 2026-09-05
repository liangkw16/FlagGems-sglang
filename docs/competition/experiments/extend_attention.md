# Task 50 `extend_attention` 实验记录

```current
task: 50
operator: extend_attention
batch: 4
validity: candidate-wip
platform: 6/8+(e3,燧原翻绿0.013x<门槛;昆仑在评)
team_best_stage: -
team_best_speedup: -
sealed: no
next: 华为深层数值问题(online+two-pass均败);5/8已是208发1队过线题的好成绩;冲分优先
updated: 2026-09-05
```

## S0 开发进度（2026-09-05）
- Kernel 结构：(batch, q_head, m_block) 3D grid + online softmax +
  paged prefix KV (kv_indices 间接) + GQA + 因果掩码
- 单 token 无前缀 case **已精确**（ieee dot 修复 TF32 精度后 diff=0.0）
- 多 batch case 仍 ~86% 元素错误——根因未定位（嫌疑：prefix KV 间接
  寻址或因果掩码在多 batch grid 下的交互）

## S0 提交（2026-09-05，submission 10069）

- **5 芯已过 correctness 且全部超 0.1x 门槛**：
  天数 0.31x / 沐曦 0.60x / 海光 1.37x / A 1.40x / B 1.07x
- 华为 correctness 失败（精度差异）
- 燧原/昆仑评测中
- **调试关键**：单 query 标量 online softmax → 多 block KV 循环中
  `m_val = m_new` 缺失导致第二个 block 的 alpha 清零全部先前累积
  ——一行修复，4/4 代理测试全过

## E1-E2 华为 vendor（2026-09-05，submissions 10114/10125）

- E1（fp32 强制）：华为仍败
- E2（两遍法替代 online softmax）：华为仍败（同款 Comparing error，
  同款张量值 -4.26e-01 / 4.31e-01）
- **判定**：华为数值问题不在 softmax 算法（online/two-pass 等价
  数学都败），在更底层的 dot/exp/accumulation lowering
- 五芯稳定过门槛：天数 0.32 / 沐曦 0.59 / 海光 1.33 / A 1.40 / B 1.07

## E3 batched vendor → 燧原翻绿（submission 10141）
- **燧原 PASS 0.013x**（batched 后不再超时！但低于 0.1x 门槛）
- 沐羲升至 0.727x（batched 比 per-query 快）
- 6 芯 correctness 全过（天数/沐曦/燧原/海光/A/B）
- 华为仍败；昆仑在评
