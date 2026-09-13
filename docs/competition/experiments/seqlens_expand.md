# Task 74 `seqlens_expand` 实验记录

```current
task: 74
operator: seqlens_expand
batch: 5
validity: valid
platform: submitted(13796,e1,评测中;TB s0 10.6444x)
candidate_stage: e1
team_best_stage: -
sealed: no
next: S0 首发即 8/8 valid 10.6444x（天数 25.5/海光 14.6/A 12.5/B 10.6/燧原 6.8/华为 5.8/沐曦 7.6/昆仑 1.8）；c2flow 19.98 为靶；E1 按逐芯定轴
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
