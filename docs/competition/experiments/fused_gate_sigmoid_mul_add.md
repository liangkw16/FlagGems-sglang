# Task 81 `fused_gate_sigmoid_mul_add` 实验记录

```current
task: 81
operator: fused_gate_sigmoid_mul_add
batch: 6
validity: candidate-ready
platform: none(s0 release 5/5 通过 NVIDIA 代理;额度 0/30,09-18 首发)
candidate_stage: s0
team_best_stage: -
sealed: no
next: 窗口 09-24 19:59;D1 首发 s0 后按逐芯回执开 e 轴
updated: 2026-09-17
```

## 契约与实现（S0）

- 题面：[Task 81](../tasks/batch-6/81-fused_gate_sigmoid_mul_add.md)。
  `gate = Σ(hidden*gate_weight,-1)` fp32；`out = final + sigmoid(gate)*shared`
  （fp32 乘加）→ final dtype；标准容差。上游 d4ad368 FUSE_GATE 路径对照。
- 实现：单 kernel 两阶段（阶段 1 行点积 fp32——hidden 只读一次；阶段 2
  final+shared FMA store），一 program 行粒度 grid-stride（≤2048），
  BLOCK_H=1024 + 尾 mask；内维连续断言、行 stride 透传。
- 测试：dtype 矩阵 × 形状（1×1 到 2049×7168、BLOCK_H 尾 1535）、门控饱和
  （±200）与 gate=0 行、fp32 抵消行、NaN 传播、行间 stride。
  本地容差覆盖归约重排噪声（抵消点相对差放大），远严于平台标准。

## 不可变身份

- source commit：`370923bfb83a8b7e681051acb2a3b1585dde12fc`（五题同批）。
- verification commit：`70a4a4d58fe727496001e029deac9ff3aa6fe93d`（含测试侧修复轮）。
- source SHA-256：`b8053e124d7309d80e108ac9ae69a6fc3dfd3060b0c8c1aac1bf606ac4ab853c`。
- test SHA-256：`7c0bae7f2c82eb87f89dcbeec844549f8c4cdd26eaa12fe724fe3cc5ebe20227`。
- ZIP：`artifacts/competition/fused_gate_sigmoid_mul_add/s0-370923b/fused_gate_sigmoid_mul_add.zip`，SHA-256
  `5327412cb23be84e62a287f4b3c9c7e0fe3dcca892c18193a66b9d44ad32c0bf`（单成员 `fused_gate_sigmoid_mul_add.py`，generic-only）。
- release 回执：`artifacts/competition/b6-s0-release-20260917/fused_gate_sigmoid_mul_add/verification.json`
  （5 测试 0 失败，22 次非 warmup kernel launch，
  NVIDIA RTX 5070 Ti / torch 2.13.0+cu130 / triton 3.7.1），
  SHA-256 `75e8667cd4aed70549dfce35a5827f1b5554142225743d695ddc809ff49d48ab`；日志 SHA-256 `48cf0b58e7ac5b0c5dec8361875dd6341cce5db3775e996a292a1f297c464401`。

## 靶子与下一步

- 榜首 c2flow 4.4852x（领先次席 +1.95%）；e 轴：华为 persistent（上行假设，方向性）→ 燧原（3.89 vs 次优 2.20）→ 多行宽度对照。
- 验证三轮教训已固化在测试侧：numel%16 契约、token 边界与 kv_lens 解耦、
  归约噪声容差、行内连续 stride 构造。
- 八芯目标 `target-runtime-unverified`（NVIDIA 代理证据），裁决权在平台。
