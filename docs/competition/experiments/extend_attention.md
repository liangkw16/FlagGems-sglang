# Task 50 `extend_attention` 实验记录

```current
task: 50
operator: extend_attention
batch: 4
validity: candidate-wip
platform: none(未提交;S0开发中,多batch路径~86%错)
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
