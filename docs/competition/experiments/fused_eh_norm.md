# Task 68 `fused_eh_norm` 实验记录

```current
task: 68
operator: fused_eh_norm
batch: 5
validity: valid
platform: submitted(13369,e1,评测中;s0 8/8 6.5562x)
candidate_stage: e1
team_best_stage: s0
team_best_speedup: 6.55616667
sealed: no
next: e1（_hygon 2D 路径分裂）已发射；海光扛 0.485/0.71 榜差（10.92 vs 14.80）；预注册海光 ≥1.10x 才保留
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

## 2026-09-12 E1：`_hygon` 2D 路径分裂 vendor（候选就绪后提交）

- 逐芯榜单（13:4x）：榜首 HAiWORLD 7.2622，榜差 0.71 中 **0.485 集中在
  海光**（我 10.9223 vs 14.7999）；昆仑/国际 B 我方反超，其余芯差
  0.09~0.78。单杠杆 = 海光 vendor。
- 形态（T59 海光 +22% 同款打法）：直接 2D 网格 `(tokens, 2)`，每 program
  只算一路 RMSNorm（enorm 或 hnorm），程序内串行工作量减半、program 数
  翻倍；BLOCK/num_warps=8/num_stages=1 与 generic 一致，generic 字节不动。
- source commit：`19e45c369445ee5f07a1af9acf4b7d387c07a616`。
- ZIP：`artifacts/competition/fused_eh_norm/e1-19e45c3/fused_eh_norm.zip`，
  SHA-256 `7e29e09e2a8b859afb49b12ed1d17177d79915635bf1c1d0e2bb0db11195193d`；
  成员 generic `5a19180d…` + `_hygon` `f95542db…`。
- release 回执（v2，绑定 19e45c3，proxy-vendor hygon）：
  `artifacts/competition/batch5-t68e1-validate-20260912/fused_eh_norm/verification.json`
  （SHA-256 见下方提交段）；4 方法 0 失败，generic 15 + hygon 15 launch。
- 预注册：海光中位收益 ≥1.10x 才保留；其余七芯读数与 S0 窗口一致。

## 2026-09-12 E1 平台提交（submission 13369）

- 上传与正式 POST 各一次；state submitted。额度：发后 12/30。
- 回执 SHA-256：`16e98305df74721afd66f8bfeb4531dee19570df5350a0981088a7a3a6aeb316`；
  日志 `9df42acc60ef133ea48195ce194d1ea11b3da54373391443677689642aa777a2`。
- 裁决点：海光 10.92→？（预注册 ≥1.10x 即 ~12.0+ 才保留）；其余七芯
  应与 S0 窗口一致（generic 字节不变）。
