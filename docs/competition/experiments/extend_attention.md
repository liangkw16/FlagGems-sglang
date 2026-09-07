# Task 50 `extend_attention` 实验记录

```current
task: 50
operator: extend_attention
batch: 4
validity: candidate-wip
platform: 6/8(e4,昆仑conclusive封轴;华为数值不可修;燧原0.013x<门槛)
team_best_stage: -
team_best_speedup: -
sealed: no
next: 逐阶段诊断已实现但插桩有观察效应;需要目标原失败重放,不再将两种失败形态称数值永久不可修
updated: 2026-09-08
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

## E4 昆仑 per-query BLOCK_N=16（submission 10280）
- 昆仑仍败（uni_sram 未被 BLOCK_N=16 + coreTiling 清除）→ conclusive 封轴
- 6 芯 correctness 过；燧原 0.013x 低于门槛；华为数值问题
- **T50 最终定格 6 芯 correctness / 5 芯过门槛**

## 2026-09-08 推荐方案实现与提交前验证（未提交平台）

实现华为 QK/max/exp/sum/PV 独立诊断副本；同输入分进程执行。代理完整测试通过，但插桩改变818/6144个最终FP32位值，存在观察效应，不能用当前trace判定目标首差。

- source `c73f6c3f83ec38d5a2c40cfdef996e64e50ecd67`；verification `c73f6c3f83ec38d5a2c40cfdef996e64e50ecd67`。4 个测试方法、42 次实际 kernel 调用；选定 NVIDIA/代理范围门禁通过。
- 回执 `artifacts/competition/batch4-implementation-20260907/t50-release1/verification.json`，SHA256 `b49b9d20b0fb66f2685f193428ffc2465b4b4b136c9a5cbb2b677338cbaf421e`；日志 SHA256 `7e4ec08054da182e1ec9c6931edcaf04a6032a69274edaf609caa29a5af1e888`。
- 环境、逐源码执行范围、原始配对数据和未完成条件见[本轮报告](../implementation-batch4-20260908.md)及[证据清单](../data/batch4-implementation-20260908.json)。本轮不更新历史有效分，未做平台 preflight、上传或正式提交。
