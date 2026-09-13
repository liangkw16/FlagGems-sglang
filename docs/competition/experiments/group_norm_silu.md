# Task 72 `group_norm_silu` 实验记录

```current
task: 72
operator: group_norm_silu
batch: 5
validity: candidate-wip
platform: submitted(13774,e1,评测中;s0=7/8 昆仑 uni_sram)
candidate_stage: e1
team_best_stage: -
sealed: no
next: S0 首发等窗口；目标 8/8 valid 后按逐芯读数定轴
updated: 2026-09-13
```

## 契约与实现（S0）

- 完整题面：[Task 72](../tasks/batch-5/72-group_norm_silu.md)（2026-09-12 晚新增五题之一）。
- division-free [C,S] tiles; scalar-carried stats; three-pass centered variance。
- 核心计算 Triton，无 fallback；八芯 0.1x；截止 2026-09-17 19:59:59。

## 不可变身份

- source / verification commit：`e6b450fd4beb001255079ade929d3b2de51796d6`（五题同批提交）。
- source SHA-256：`690aa6bd15ed593fa76f8b412457aa9d4a1cec4d18d8ae7f0519b213f50bc229`。
- test SHA-256：`80f5c46cc34626a26ce0f6bba354420daee33ae9cb511d8e2bb0da1ba34645fa`。
- ZIP：`artifacts/competition/group_norm_silu/s0-e6b450f/group_norm_silu.zip`，SHA-256 `9d6d79bfd0c1dc2a2bc5eb72cacb7d8c5f950aaca46c9d71ec9243e22fdff3a2`（单成员 `group_norm_silu.py`）。
- release 回执：`artifacts/competition/batch5-new5-validate-20260913/group_norm_silu/verification.json`，
  SHA-256 `563d574916eaef42da2edfb405e5472a01d705064052736e93cd9843d4221ec0`；日志 SHA-256 `4e85ba3445e569bf0584ce2e0be7a76c9f56e90a4b3e58eb8ebe487958d5ca1d`。

## 验证状态

- screening 多轮门禁拦下并修复的缺陷已记录于提交说明；
  最终 release：0 失败/错误/skip，非空 kernel launch，NVIDIA 代理范围。
- 所有八芯目标 `target-runtime-unverified`；裁决权在平台。

## 2026-09-13 E1：昆仑小 tile vendor（候选就绪后提交）

- S0 判决：七芯过（含燧原/华为），**昆仑 `OutOfResources: uni_sram`**
  ——[C,S] tile 超该栈 SRAM 预算（真实执行 7.5s 后报错，非崩溃族）。
- E1：`_kunlunxin` vendor 同 kernel，tile 上限 8192→2048 lane
  （BLOCK_S 下限 32，FlagGems 昆仑小 tile 注记）；generic 字节不动。
- source commit：`8b539234fc8d130dc8698ce68b57982e2000ad46`；ZIP `e1-8b53923`，
  SHA-256 `356e1012819dfc0bb4d553c569bb4d58300ef6d00bae76f594c09e3d8d25af09`。
- release 回执：`batch5-t72e1-validate-20260913/group_norm_silu/verification.json`，
  SHA-256 `1e430b40e9d733dfd25f3501d7d6878448929aedd98ca2c897ae762d4bedbc10`；
  4 方法 0 失败。
- submission 13774；裁决点=昆仑 uni_sram 解除。

## 2026-09-13 E1 平台终态：7/8（昆仑仍 uni_sram）

- 2048-lane cap 后昆仑仍 `OutOfResources: uni_sram`（exec 8611ms 真实
  执行）——2D [C,S] tile 形态本身超预算。E2 改 1D 形态（仅空间维
  lane，channel 维标量循环 + num_warps=1）。
