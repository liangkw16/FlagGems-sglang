# Task 68 `fused_eh_norm` 实验记录

```current
task: 68
operator: fused_eh_norm
batch: 5
validity: candidate-wip
platform: not-submitted
candidate_stage: s0
team_best_stage: -
sealed: no
next: 远端 GPU 恢复后补 release 回执；回执齐全即进入 09-12 窗口发射队列（把握序第 2）
updated: 2026-09-11
```

## 契约与范围

- 完整题面：[Task 68](../tasks/batch-5/68-fused_eh_norm.md)（2026-09-11 新增）。
- 接口 `fused_eh_norm(inputs_embeds, previous_hidden, enorm_weight, hnorm_weight, eps)`；
  两路 RMSNorm（不同权重）+ 末维 concat；hidden ∈ (256, 8192] 且 %256==0；
  fp16/bf16；fp32 归约与权重乘、写回时一次类型转换；per-dtype tolerance。
- 返回 `[T, 2*hidden]` 新张量；核心计算 Triton；八芯 0.1x。

## 实现（S0）

- 上游：SGLang 8014d9d `kernels/ops/layernorm/fused_eh_norm.py`（CUDA JIT
  版）→ Triton 移植：每 token 一个 program，`BLOCK = next_power_of_2(hidden)`
  （≤8192）单块；先 enorm 路载入→fp32 平方和→rsqrt→乘权重→cast 写出，
  再 hnorm 路同构写 `out[:, hidden:]`。权重每 program 重读（上游 CTA 同构）。
- `num_warps=8, num_stages=1`（对齐本仓 fused_rmsnorm 先例）。

## 不可变身份

- source / verification commit：`b4727f1`。
- source SHA-256：`5a19180d9b5257c43d194bbb6df804b53ba849cf6e5679c3ce1a4c303397caaf`。
- test SHA-256：`81ca6d485ef5dae8a4c56273b8c5122c1f3892984cd339c6f13dbb679f591934`。
- ZIP：`artifacts/competition/fused_eh_norm/s0-b4727f1/fused_eh_norm.zip`。
- ZIP SHA-256：`1d588933d9797c66ff7c417ae5817b8659825a56625f54a1b1c06a4e21554807`。

## 验证状态

- py_compile、格式与 lint 通过（本地）。
- 测试：4 方法 / dtype×eps、hidden 512~8192（含 768、7168 非二次幂）、
  tokens 1~8193、strided 输入、全零行与空 batch；输入不变断言。
- **远端 GPU 不可达（2026-09-11 晚）**：release 回执待补，
  `target-runtime-unverified`。

## 风险

- 单块 BLOCK=8192 的寄存器压力在窄带宽芯上可能偏高；若平台退化，E1 方向
  是按 1024~2048 列分块的两遍循环（sumsq 后再归一化写出），复用本仓
  fused_rmsnorm/fused_norm_rope_stacked 已验证形态。
- 每行两次权重载入可被 L2 吸收；不提前优化。

## 优化方向（按把握）

1. S0 直投。
2. E1（未开发）：列分块两遍式，针对 BLOCK=8192 退化场景；先看平台逐芯。
