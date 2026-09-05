# Task 50 `extend_attention` 实验记录

```current
task: 50
operator: extend_attention
batch: 4
validity: candidate-wip
platform: 5/8+(S0在评;5芯过门槛;华为correctness败;燧原/昆仑在评)
team_best_stage: -
team_best_speedup: -
sealed: no
next: 调试多batch路径(单token已精确);榜首c2flow 3.60x,1队过线
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
