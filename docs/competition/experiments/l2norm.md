# Task 56 `l2norm` 实验记录

```current
task: 56
operator: l2norm
batch: 4
validity: valid
platform: 8/8(s0,3.10497917x)
team_best_stage: s0
team_best_speedup: 3.10497917
sealed: no
next: 已修复高维stride,GPU screening 7 tests通过;待提交字节release复验,未提交平台;8/8仍指历史s0
updated: 2026-09-06
```

## S0 → **8/8 VALID**（2026-09-06，submission 10405）

- **8/8 valid，avg 3.1050x team best**（一发命中！）
- 逐芯：天数 6.66 / 沐曦 2.54 / 燧原 1.19 / 海光 4.54 /
  昆仑 0.58 / 华为 1.38 / A 3.73 / B 4.21
- Per-row fp32 sum-of-squares → rsqrt → scale，cast back
- 榜首 EvokeAgent 3.89x，差距 20%

## 2026-09-06 验证流程修复

- 静态定位：连续高维输入按 `numel / D` 展平为行后，应以 D 作为输入/输出行步长，不能沿用原张量 stride(0)。
- 修复两个行步长；多维用例扩至三种 dtype、1D/3D/4D 以及前导维转置。现有 s0 成绩和 ZIP 保留原样，不能给修复版背书。
- 用户已授权源码传输；RTX 5070 Ti screening 7 tests、24 个 test/subTest 记录全部通过，入口实际调用 21 次。提交字节的 release 复验另记。
