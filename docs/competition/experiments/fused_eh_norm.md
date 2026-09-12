# Task 68 `fused_eh_norm` 实验记录

```current
task: 68
operator: fused_eh_norm
batch: 5
validity: valid
platform: completed(13301,s0,8/8,6.5562x)
candidate_stage: s0
team_best_stage: s0
team_best_speedup: 6.55616667
sealed: no
next: S0 首发 8/8 valid（seq 6，榜首 HAiWORLD 7.17 差 0.61）；昆仑 1.2075 最薄；E1 列分块两遍式可试抬昆仑
updated: 2026-09-12
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

## 2026-09-12 平台结果（submission 13301，daily_seq 6）

- **8/8 valid，均值 6.55616667x**。逐芯：天数 10.8214 / 沐曦 6.1777 /
  燧原 2.0079 / 海光 10.9223 / 昆仑 1.2075 / 华为 3.9125 / A 9.1589 /
  B 8.2411。榜首 HAiWORLD 7.1684（差 0.61）。
- 回执（b4727f1）：4 方法 0 失败、15 launch、20 组 shape；
  `batch5-ext6-validate-20260912/fused_eh_norm/`
  （SHA-256 `c7b2c0c7a5d4c75947afa1fd10794de44c152a4e78a650361e804b64f592457b`）。
