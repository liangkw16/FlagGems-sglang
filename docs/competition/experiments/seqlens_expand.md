# Task 74 `seqlens_expand` 实验记录

```current
task: 74
operator: seqlens_expand
batch: 5
validity: valid
platform: completed(13804,e2,8/8,10.5267x;TB s0 10.6444x)
candidate_stage: e2
team_best_stage: s0
team_best_speedup: 10.6444
sealed: no
next: e2 8/8 valid（昆仑重掷成功）；tile 轴收益<窗口方差，收口；对榜首 c2flow 24.68 的差距需新结构
updated: 2026-09-13
```

## 契约与实现（S0）

- 完整题面：[Task 74](../tasks/batch-5/74-seqlens_expand.md)（2026-09-12 晚新增五题之一）。
- attention pad.py kernel; per-request program, clamped base+arange; cumsum offsets in wrapper。
- 核心计算 Triton，无 fallback；八芯 0.1x；截止 2026-09-17 19:59:59。

## 不可变身份

- source / verification commit：`e6b450fd4beb001255079ade929d3b2de51796d6`（五题同批提交）。
- source SHA-256：`884da15e0f885cc01f31ee53198fbd61cbd045d4c0b71c3ee0106e6644c9de35`。
- test SHA-256：`83da626a4e5922220d9283748353bea7d7e1286a7cc00fc5b0a49561402152ce`。
- ZIP：`artifacts/competition/seqlens_expand/s0-e6b450f/seqlens_expand.zip`，SHA-256 `2a6ecb1cea7019b6154ef9d78aa5969797d0b4b5d60a6598cd77ee9d111fc9bc`（单成员 `seqlens_expand.py`）。
- release 回执：`artifacts/competition/batch5-new5-validate-20260913/seqlens_expand/verification.json`，
  SHA-256 `3a84097f235d313a00adc7e357ab7b1fb90c6d87f63cf7c26542acb0522356c6`；日志 SHA-256 `a022f73c47816e01f24df117e3e2e09ae12545e4915c7ad8373192b0017fcf3d`。

## 验证状态

- screening 多轮门禁拦下并修复的缺陷已记录于提交说明；
  最终 release：0 失败/错误/skip，非空 kernel launch，NVIDIA 代理范围。
- 所有八芯目标 `target-runtime-unverified`；裁决权在平台。

## 2026-09-13 E1：请求维 tile（候选就绪后提交）

- S0 整请求单 program（BLOCK=next_pow2(max_q_len)）在请求少而 q_len 大
  的 shape 欠填充（对榜首差距全芯均匀 +5~+48）。E1：kv_indices splits
  同款 2D tile——固定 1024-lane，(request, tile) 每 program。
- source commit：`114be7818888c9d8bfd1b36c7bb42845bc2c7fd4`；ZIP `e1-114be78`，
  SHA-256 `85ae694a999dca3b9657d0815391102a4a69cf0e47f859f1ba7c3989ee0dc884`；
  release 回执 SHA-256 `38662ae336ddab30deca33cca2fc719fd3d2b475601aee26c3818404ae1801a6`；
  3 方法 0 失败。
- submission 13796；裁决点=全芯均匀差距是否收窄。

## 2026-09-13 E1 终态与 E2 重掷（submission 13804）

- E1（13796）7/8:七芯全过且 tile 正收益（天数 27.0782 +6% / 沐曦
  8.2582 +8% / 其余持平），昆仑 exec 0ms 间歇崩溃（s0 时昆仑正常判过
  1.8150 ⇒ 按提交闪断）。
- E2：tile 维 255 封顶 + grid-stride（燧原 grid.y 硬限的真实加固）=
  新 ZIP 身份重掷。
- source commit：`40f82daea4780bdb9e99b4474e5546748ef7c117`；ZIP `e2-40f82da`，
  SHA-256 `7c57ef034849d4c7b39d0beb19bf29bd84085f9da5209c6f6be613c5730db008`；
  release 回执 SHA-256 `941cc373a8481dc0d5e8e9ac4e75ddd8231f6bef00035e84c7ccbe341b1df9df`。
- submission 13804；裁决点=昆仑窗口 + 七芯 tile 增益兑现（均值预期
  ≈11.2）。

## 2026-09-13 E2 平台终态：8/8 valid 10.5267x（低于 TB，轴收口）

- 昆仑窗口重掷成功（exec 8786ms 真实执行，1.5952 过线）——**8/8 valid
  但均值 10.5267 < s0 TB 10.6444**：e1 的天数/沐曦 tile 增益被本窗口
  沐曦（8.26→6.82）/海光（14.66→13.82）回落吃掉。
- 处置：team best 保留 s0；tile 轴收益确认存在但量级 < 窗口方差，
  不再追加同轴发射。T74 保持 valid 在榜。
