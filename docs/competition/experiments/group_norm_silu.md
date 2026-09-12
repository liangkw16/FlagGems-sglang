# Task 72 `group_norm_silu` 实验记录

```current
task: 72
operator: group_norm_silu
batch: 5
validity: candidate-wip
platform: s0 候选就绪待提交
candidate_stage: s0
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
