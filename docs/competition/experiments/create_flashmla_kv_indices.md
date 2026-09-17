# Task 79 `create_flashmla_kv_indices` 实验记录

```current
task: 79
operator: create_flashmla_kv_indices
batch: 6
validity: valid
platform: completed(17209,s0,8/8,173.3643125x,#2;榜首HAiWORLD 187.47)
candidate_stage: s0
team_best_stage: s0
sealed: no
next: 燧原 34.6→51.3(T63 GCU persistent24+stages3+BLOCK4096 配方 vendor)+沐曦 120→164+昆仑 15.4→17.8;华为 97.5 已反超榜首 92.5
updated: 2026-09-18
```

## 契约与实现（S0）

- 题面：[Task 79](../tasks/batch-6/79-create_flashmla_kv_indices.md)。T63 姐妹题
  改 per-page：`slot = req_to_token[pool, start + p*page_size]`，
  `kv_indices[i,p] = slot // page_size`；输出 2D clone 基底 + 行尾保留；
  exact。上游 92d831d `create_flashmla_kv_indices_triton` 对照。
- 实现：T63 e8 族结构（grid-stride 行循环 + 页块 split、GCU 安全算术、
  i64 仅寻址）；page_size 走 constexpr（pow2 除法强度削减、昆仑向量除法
  风险规避）；out=empty + kernel 行尾回写（免 clone 全量拷贝）。
- 测试：page_size {1,3,16,64,128}、非整除长行、kv_start=None/非零、
  空 batch/零长度行、行尾哨兵保留与 base 变更重读、宽度==页数、
  split 几何边界（255/256/257、batch 257）、stride/dtype 矩阵。

## 不可变身份

- source commit：`370923bfb83a8b7e681051acb2a3b1585dde12fc`（五题同批）。
- verification commit：`370923bfb83a8b7e681051acb2a3b1585dde12fc`（含测试侧修复轮）。
- source SHA-256：`93d158ad535142e941d8a8f0000ebeeccb147df6c8ceb02edf5d78efb9d49f01`。
- test SHA-256：`9083d94d1729394865a7b4e6026472b0fc84fef3864538c544409044350a0612`。
- ZIP：`artifacts/competition/create_flashmla_kv_indices/s0-370923b/create_flashmla_kv_indices.zip`，SHA-256
  `8818616277c4bbd9b0ad408bdc164d6696ee7a5434b12eb7a2455e8da2cb235d`（单成员 `create_flashmla_kv_indices.py`，generic-only）。
- release 回执：`artifacts/competition/b6-s0-release-20260917/create_flashmla_kv_indices/verification.json`
  （7 测试 0 失败，37 次非 warmup kernel launch，
  NVIDIA RTX 5070 Ti / torch 2.13.0+cu130 / triton 3.7.1），
  SHA-256 `037423bff60d94b0b0f4624098a07fe980606f89cc26b3c3891b4a1df8d21f1d`；日志 SHA-256 `80afa45e6882833b218ea9e9090d76a4763461d00203047c139188240d8d2890`。

## 靶子与下一步

- 榜首 HAiWORLD 187.47x；我方 T63 banked 逐芯等效 ≈199x，直移争第一；昆仑轴（8 vs 17.8）是最大增量项。
- 验证三轮教训已固化在测试侧：numel%16 契约、token 边界与 kv_lens 解耦、
  归约噪声容差、行内连续 stride 构造。
- 八芯目标 `target-runtime-unverified`（NVIDIA 代理证据），裁决权在平台。

## 2026-09-18 S0 平台首回执（submission 17209，observed_at 01:0x +08）

- 状态：8/8 valid, #2/4；均值 173.3643125。
- 逐芯：天数 340.72 / 沐曦 120.46 / 燧原 34.61 / 海光 284.95 / 昆仑 15.42 / 华为 97.54 / A 309.56 / B 183.64。
- 首发即 #2（次席 cgzhou 171.25 仅高 2.1）。对榜首逐芯：华为 +5.0 / 天数 -9.0 / 海光 -19.0 / A -12.1 / B -15.0 / 沐曦 -43.7 / 燧原 -16.7 / 昆仑 -2.4。燧原与沐曦两轴合计可 +7.5 均值。
