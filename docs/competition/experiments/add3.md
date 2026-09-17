# Task 76 `add3` 实验记录

```current
task: 76
operator: add3
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

- 题面：[Task 76](../tasks/batch-6/76-add3.md)。`out = bf16(bf16(a+b)+c)`，
  **双重舍入是契约**（与未融合对逐位一致，禁 fp32 单舍入）；bf16 连续、
  numel%16==0；标准 bf16 容差。
- 实现：flat grid-stride kernel，`(a.f32+b.f32).bf16` 显式中间舍入再
  `+c` 出 bf16（RTNE，与 torch eager 每算子语义对齐）；BLOCK 1024、
  grid≤4096；对照 SGLang 197832b add3（PDL 为 NVIDIA 专属未引入）。
- 测试：双舍入最小反例（a=1,b=2^-8,c=-1 → 双舍入 0 / 单舍入 2^-8）、
  尾块边界、±inf 相消与 NaN 位级对照。

## 不可变身份

- source commit：`370923bfb83a8b7e681051acb2a3b1585dde12fc`（五题同批）。
- verification commit：`686092eb443f6a7b9514f8890fa9fdaa2d339a02`（含测试侧修复轮）。
- source SHA-256：`5f89fb2445a68fad815a89f298a29ce207d4e3e4c11369baeeaf4f23b820d296`。
- test SHA-256：`c4f0a92ceaf002c88bd47564eb825b09d61cd483219e3dec03c0d69d196149a6`。
- ZIP：`artifacts/competition/add3/s0-370923b/add3.zip`，SHA-256
  `6a50c36f2787094b347c45f0ed466f14309b14162971b4405cf56e924312f9c7`（单成员 `add3.py`，generic-only）。
- release 回执：`artifacts/competition/b6-s0-release-20260917/add3/verification.json`
  （3 测试 0 失败，9 次非 warmup kernel launch，
  NVIDIA RTX 5070 Ti / torch 2.13.0+cu130 / triton 3.7.1），
  SHA-256 `e7a372866f53bc1f4a7ae4612c7467f2ead8df6a6aa4790789c3136366f4eb93`；日志 SHA-256 `ed0eb9ef80b57877051e0ccf409d8c29c67867bc04a50e52b7fac2e7585720da`。

## 靶子与下一步

- 榜首 GuanghuLab 1.1611x（9/9 队有效，无彩票指纹）；主轴华为（全员<0.5）与昆仑 vendor。
- 验证三轮教训已固化在测试侧：numel%16 契约、token 边界与 kv_lens 解耦、
  归约噪声容差、行内连续 stride 构造。
- 八芯目标 `target-runtime-unverified`（NVIDIA 代理证据），裁决权在平台。
