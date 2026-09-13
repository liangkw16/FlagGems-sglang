# Task 75 `sigmoid_gate_mul` 实验记录

```current
task: 75
operator: sigmoid_gate_mul
batch: 5
validity: valid
platform: completed(13878,e1,8/8,2.5107x;TB s0 2.538x)
candidate_stage: e1
team_best_stage: s0
team_best_speedup: 2.538
sealed: no
next: e1 订正=4096 对燧原+71%（此前回退记载系转录错误,已作废）；e2=燧原8192+昆仑no-loop
updated: 2026-09-13
```

## 契约与实现（S0）

- 完整题面：[Task 75](../tasks/batch-5/75-sigmoid_gate_mul.md)（2026-09-12 晚新增五题之一）。
- flat 1024-lane x*sigmoid(gate); SGLang triton_sigmoid_gate_mul form。
- 核心计算 Triton，无 fallback；八芯 0.1x；截止 2026-09-17 19:59:59。

## 不可变身份

- source / verification commit：`e6b450fd4beb001255079ade929d3b2de51796d6`（五题同批提交）。
- source SHA-256：`df083ec35d96b1f3d08a7db4fa4f051d5fee6e747ab30bfe068276634c7f06af`。
- test SHA-256：`1f8d9b035c8706c9c98f6c186924a3d158f4ae40d5c5dbdb203853635cdb481a`。
- ZIP：`artifacts/competition/sigmoid_gate_mul/s0-e6b450f/sigmoid_gate_mul.zip`，SHA-256 `f608dbaf766c96df37e60f5d0c6a6a7ef89ffcd789f617b23165c479eee265ff`（单成员 `sigmoid_gate_mul.py`）。
- release 回执：`artifacts/competition/batch5-new5-validate-20260913/sigmoid_gate_mul/verification.json`，
  SHA-256 `93ea7bbb4a56b52982b89d21447b4afe9529bad5756c9d38af48d6c23aeae1d5`；日志 SHA-256 `a8f10c821397f48f969b52b395b846fe627e6466eb2a83ae7d80eda01914a24b`。

## 验证状态

- screening 多轮门禁拦下并修复的缺陷已记录于提交说明；
  最终 release：0 失败/错误/skip，非空 kernel launch，NVIDIA 代理范围。
- 所有八芯目标 `target-runtime-unverified`；裁决权在平台。

## 2026-09-13 E1：BLOCK 1024→4096（13878）8/8 valid 2.511（API 真值订正）

**真值**：s0(13767) enflame 0.827 / kunlun 0.295 / huawei 1.028 / tianshu 4.769
（avg 2.5382）；e1(13878) enflame **1.417（+71%）** / kunlun 0.262 /
huawei 1.369 / tianshu 4.191（avg 2.5107）。

**订正结论**：4096 对燧原 **+71%**（与此前"回退"记载相反——同样是
转录错误）；但天数 -12%（4.769→4.191）吃掉净收益，均略低于 TB。
e2 假设：**燧原再推 8192 + 昆仑 no-loop 形态**（T61 no-loop 读 1.29
vs 本题 0.26）。
