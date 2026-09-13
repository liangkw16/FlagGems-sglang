# Task 71 `gelu_tanh_and_mul` 实验记录

```current
task: 71
operator: gelu_tanh_and_mul
batch: 5
validity: valid
platform: completed(13898,e2,8/8,2.621x;TB e1 2.62646667x)
candidate_stage: e2
team_best_stage: e1
team_best_speedup: 2.62646667
sealed: no
next: e2 vendor 冻结证伪（s0 读数=窗口方差非 BLOCK 形态）；TB e1 守；追 3.44 需窗口或新结构
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

## 2026-09-13 E2：燧原/昆仑 vendor 冻结 1024（已发射 13898）

- e1 逐芯教训：4096 增益天数/华为/A,回退燧原(-50%)/昆仑(-89%)→
  vendor 冻结 s0 BLOCK=1024 供此二芯,generic 4096 供其余。
- source commit：`755d832cdb8660faea7aa07857b58888b6d81ed0`；ZIP `e2-755d832`，
  SHA-256 `387c38819176b495d2502461fbc65462ee750c8cb1d57412094e1193fe17af4c`；
  3 成员。release 回执 SHA-256
  `9ec905365cbaf5a96c4ea28d61f2a7440c47c483821ba683a99e0adb7326fc99`；
  4 方法 0 失败，3 源 45 launch。
- submission 13898；裁决点=燧原回 2.9+ 昆仑回 2.0+（均值预期
  2.63→2.95+,距榜首 3.44 缩到 0.5 内）。

## 2026-09-13 E2 平台终态：8/8 valid 2.621（未过 TB，vendor 冻结假设证伪）

- **燧原 vendor 被选中但读数反而更低**（1.48→1.03，BLOCK=1024 未能
  恢复 s0 的 2.95）；昆仑 vendor 同样（0.25→0.26，未回 s0 的 2.19）。
  **s0 读数里的燧原 2.95/昆仑 2.19 不是 BLOCK 形态差异——是窗口
  方差**（s0 单发窗口 vs e1/e2 双发窗口）。
- 均值 2.621 ≈ e1 2.626（TB 不变）。**"冻结窄 BLOCK vendor"假设
  证伪**——BLOCK=1024 与 4096 在这两芯的当前窗口下等价。
- T71 轴收口：TB e1 2.626；追榜首 3.44 需窗口或新结构。
