# Task 73 `residual_gate_add` 实验记录

```current
task: 73
operator: residual_gate_add
batch: 5
validity: valid
platform: completed(13872,e1,8/8,3.903x;TB s0 3.952x)
candidate_stage: e1
team_best_stage: s0
team_best_speedup: 3.952
sealed: no
next: e1 扁平轴判关（燧原/华为/昆仑 -71~-84%,2D 网格是这些栈正确形态）；TB s0 守榜；追 0.32 回 2D 路径内微调
updated: 2026-09-13
```

## 契约与实现（S0）

- 完整题面：[Task 73](../tasks/batch-5/73-residual_gate_add.md)（2026-09-12 晚新增五题之一）。
- 2D grid avoids gate modulo; product rounds to dtype before add (double rounding contract)。
- 核心计算 Triton，无 fallback；八芯 0.1x；截止 2026-09-17 19:59:59。

## 不可变身份

- source / verification commit：`e6b450fd4beb001255079ade929d3b2de51796d6`（五题同批提交）。
- source SHA-256：`9ec431b99fbfcb016299f1ede9fa370badfc0083b2be50b1a64fb046ed8ef6b4`。
- test SHA-256：`65b50e81bd82507f68a2e7d50463e1e98c61d0b9bc51c32393c6a5a320c027ac`。
- ZIP：`artifacts/competition/residual_gate_add/s0-e6b450f/residual_gate_add.zip`，SHA-256 `942643b4f828362fb5f97e4f9a14894e8986fc5950fe8e656f85f5dd1dce54e2`（单成员 `residual_gate_add.py`）。
- release 回执：`artifacts/competition/batch5-new5-validate-20260913/residual_gate_add/verification.json`，
  SHA-256 `bc29bd54560f16941293e46ea8f7fe2fc4f12452442ecb2f637626008d47b74a`；日志 SHA-256 `b0258bcd84a6a28f7274383004361bf559e819c51c17cd4120a2ead082638b59`。

## 验证状态

- screening 多轮门禁拦下并修复的缺陷已记录于提交说明；
  最终 release：0 失败/错误/skip，非空 kernel launch，NVIDIA 代理范围。
- 所有八芯目标 `target-runtime-unverified`；裁决权在平台。

## 2026-09-13 E1：同形 gate 扁平化（已发射 13872）

- Codex 咨询建议：非广播 gate 改 flat 1D（免 2D grid 窄行浪费，免
  %D），广播保持 2D——纯结构调整，双重舍入语义不变。
- source commit：`dd043007f917f3406566167cc54240df4f719887`；ZIP `e1-dd04300`，
  SHA-256 `2f2d64146c540cca957d13fc06a67a8c2ab6a5523063aad020210f7baaec020f`；
  release 回执 `batch5-r5-20260913/residual_gate_add/verification.json`
  SHA-256 `f301e7c62bae010808c13d6555bd8cd31ff35b46f60a77e6f95a5e33a5c2ae04`；
  3 方法 0 失败。

## 2026-09-13 E1 平台终态：8/8 valid 3.903（低于 TB，扁平轴判关）

- 八芯全过但均值 3.903 < s0 TB 3.952：天数 +94%（3.55→6.89）/沐曦
  +22%/A +31%/海光 +59% 的扁平化收益被 **燧原 -75%（3.99→1.01）/
  华为 -71%（3.81→1.09）/昆仑 -84%（1.42→0.23）** 完全吃掉。
- 结论：**扁平 1D 在 GCU/昇腾/昆仑上大幅回退**（窄 tile 引发重编译
  或调度恶化），2D 网格在这些栈上是正确形态。team best 保留 s0；
  扁平轴关闭。T73 差 0.32 的追法回到 2D 路径内微调（BLOCK/num_warps
  分档 vendor）。
