# Task 71 `gelu_tanh_and_mul` 实验记录

```current
task: 71
operator: gelu_tanh_and_mul
batch: 5
validity: valid
platform: completed(13875,e1,8/8,2.62646667x team best)
candidate_stage: e1
team_best_stage: e1
team_best_speedup: 2.62646667
sealed: no
next: e1 valid 2.626 新 TB（天数+46%/华为+276%）；昆仑/燧原回退→vendor 冻结 1024 可再提；榜首 3.44
updated: 2026-09-13
```

## 契约与实现（S0）

- 完整题面：[Task 71](../tasks/batch-5/71-gelu_tanh_and_mul.md)（2026-09-12 晚新增五题之一）。
- T29 chassis; tanh via exp identity (no tl.math); fp32 compute。
- 核心计算 Triton，无 fallback；八芯 0.1x；截止 2026-09-17 19:59:59。

## 不可变身份

- source / verification commit：`e6b450fd4beb001255079ade929d3b2de51796d6`（五题同批提交）。
- source SHA-256：`19bc8de0beb978bda557dc97fc9600fa802f21a473516ab36e53e5b6abf73c8f`。
- test SHA-256：`1b6d62fb4b910d5c400640565f7194346fef5cc9b3e1f8a267aeae55bc2f725f`。
- ZIP：`artifacts/competition/gelu_tanh_and_mul/s0-e6b450f/gelu_tanh_and_mul.zip`，SHA-256 `b123f0114f244e0d7f0aa494ccd289eebf39f9ae5beb2ec4181c02b28f87c6c5`（单成员 `gelu_tanh_and_mul.py`）。
- release 回执：`artifacts/competition/batch5-new5-validate-20260913/gelu_tanh_and_mul/verification.json`，
  SHA-256 `0d4850777b93b1fd6349205ce2abd45bc48a1fc446f011269067ab520486084d`；日志 SHA-256 `771a8af8086ff4a38bde695cbd34400ea7beccb08b3460ed0cba1eeeb44cc803`。

## 验证状态

- screening 多轮门禁拦下并修复的缺陷已记录于提交说明；
  最终 release：0 失败/错误/skip，非空 kernel launch，NVIDIA 代理范围。
- 所有八芯目标 `target-runtime-unverified`；裁决权在平台。

## 2026-09-13 E1：BLOCK 1024→4096（13875）8/8 valid 2.6265 新 TB

- **新 team best 2.6265（s0 2.566）**：天数 **+46%（3.49→5.09）**/
  华为 **+276%（0.50→1.88）**/A +72%/B +6%/海光 +8%；代价：昆仑
  2.19→0.25（-89%，BLOCK=4096 对窄带芯过宽——与 T63 E2 的 BLOCK=256
  回退燧原同构：**BLOCK 增益与回退的芯片分界**）、燧原 2.95→1.48。
- 后续：昆仑/燧原走 vendor 冻结 s0 BLOCK=1024（T63 vendor 模式）。
