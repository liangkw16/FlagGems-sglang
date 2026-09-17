# Task 80 `fixup_zero_kv` 实验记录

```current
task: 80
operator: fixup_zero_kv
batch: 6
validity: valid
platform: completed(17224,e1,8/8,136.0378x首个有效,#2距榜首2.8%)
candidate_stage: e1
team_best_stage: e1
sealed: no
next: 昆仑轴关闭(9.23 flat-1D 兑现,远超榜首3.2);缺口=沐曦 100→195/海光 188→248/A 226→265;华为164.7/燧原23.3/天数238/B 138.6 反超榜首;e2 候选=沐曦 vendor 或 clone+修补单 launch 融合
updated: 2026-09-18
```

## 契约与实现（S0）

- 题面：[Task 80](../tasks/batch-6/80-fixup_zero_kv.md)。TRT-LLM ragged
  attention 后处理：`kv_lens[i]==0` 的请求整段 out 清零、lse=-inf；reference
  带 `nonzero().tolist()` host sync（巨分来源）；精确 + lse equal_nan。
  上游 74338e9 CUDA 版对照。
- 实现：clone×2 + 单 kernel；1D grid（batch×ot 展平，燧原 grid.y≤255 免疫），
  标量早退 + token 块 [BLOCK_T=8 × BLOCK_V=512] 向量清零；`max_seq_len`
  仅做 launch 尺寸提示，覆盖以 cum_seq_lens 为准（tile stride 兜底）；
  行内两维连续断言。
- 测试：kv_lens 与 token 边界解耦（零 KV 请求仍占有 q-token 行）、
  谎报 max_seq_len、未触碰段 NaN/Inf/-0 位级保留、重复调用重读。

## 不可变身份

- source commit：`370923bfb83a8b7e681051acb2a3b1585dde12fc`（五题同批）。
- verification commit：`686092eb443f6a7b9514f8890fa9fdaa2d339a02`（含测试侧修复轮）。
- source SHA-256：`eebf1fcb449a663163c1f7bb6a58bfb7c0d25a72bb0113e0bf7dd222068b4adc`。
- test SHA-256：`10d9df525805ab46f429cb60a18d625c6222f251303b82d2e62c0c48c3a0c5ff`。
- ZIP：`artifacts/competition/fixup_zero_kv/s0-370923b/fixup_zero_kv.zip`，SHA-256
  `294b35e3f98c6d8d759e0a3992914b65a9bdbd8dedee230493afad36d23db6a0`（单成员 `fixup_zero_kv.py`，generic-only）。
- release 回执：`artifacts/competition/b6-s0-release-20260917/fixup_zero_kv/verification.json`
  （6 测试 0 失败，16 次非 warmup kernel launch，
  NVIDIA RTX 5070 Ti / torch 2.13.0+cu130 / triton 3.7.1），
  SHA-256 `82f4cc986cfb8cffa03ec88c29cd8b8cd65fc141a91eb31a46cadaa1617035f5`；日志 SHA-256 `664ba85938870c42081c219b81f350bb6a1acc13e7193190b93e94d72cc5c180`。

## 靶子与下一步

- 榜首 HAiWORLD 139.95x；门槛风险芯为燧原（cgzhou 0.1924）；华为 persistent 是最大杠杆（1.9→51.6）。
- 验证三轮教训已固化在测试侧：numel%16 契约、token 边界与 kv_lens 解耦、
  归约噪声容差、行内连续 stride 构造。
- 八芯目标 `target-runtime-unverified`（NVIDIA 代理证据），裁决权在平台。

## 2026-09-18 S0 平台首回执（submission 17218，observed_at 01:0x +08）

- 状态：7/8, 昆仑编译错；均值 -。
- 逐芯：天数 238.62 / 沐曦 95.06 / 燧原 19.90 / 海光 198.22 / 昆仑 None(编译错) / 华为 162.67 / A 223.75 / B 134.52。
- 昆仑全部 9 case 同指纹 `size mismatch when packing elements for LLVM struct expected 8 but got 1`（fixup_zero_kv.py:54 lse 2D store），ConvertTritonXPUToLLVM 阶段——代码侧可修，非崩溃族。

## 2026-09-18 E1 平台终态：昆仑 flat-1D vendor 修复兑现，8/8 首个有效 136.04x #2

- 结构（`7757a41c`）：新增 `_kunlunxin` vendor——lse/out 全部改纯 1D store
  （每 token 行 flat span + 分块 mask），替换 generic 的 2D broadcast store
  （昆仑 `ConvertTritonXPUToLLVM` packing mismatch 根因）。release v2 双路径
  （generic 16 + kunlun vendor 16 launch）全过；ZIP `e1-7757a41` SHA-256
  `39cea4f138e7417e0fb86c1e7fe3afa63ff4c22539507bdfcacc9ffdb2183b43`，2 成员。
- submission **17224** completed/valid，8/8，均值 **136.0378x**（vs 榜首
  HAiWORLD 139.95 差 2.8%）。逐芯：天数 238.23 / 沐曦 100.42 / 燧原 23.28 /
  海光 187.58 / **昆仑 9.23（vendor 修复，s0 编译错→榜首档 3.2 的 2.9 倍）** /
  **华为 164.68（榜首 51.6 的 3.2 倍）** / A 226.31 / B 138.58。
- 反超榜首的芯：华为/燧原/天数/B/昆仑；缺口：沐曦（100 vs 195）、
  海光（188 vs 248）、A（226 vs 265）。
