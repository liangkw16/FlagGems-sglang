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
next: e1(f83c73a)高维stride修复版已提交(submission 10665,2026-09-07)评测中;出分后更新8/8口径
updated: 2026-09-07
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
- 最终 release 7 tests/24 条记录通过：source `f83c73aeceaa6382c6ec6d93a082fc24e37090fb`，verification `73f6bb801607d2ba77ca8469136e568a43ccf262`。回执、ZIP、门禁复验及完整哈希见 [流程实测](../workflow-validation-20260906.md)；未正式提交。

## E1 修复版平台提交（2026-09-07，submission 10665）

- source `f83c73a`（高维展平行距修复字节），stage e1，ZIP
  `e1-f83c73a/l2norm.zip` SHA-256
  `c9032f096d7811dd375547989c2157d4420f7649d6f64f5589a58581ecbcda9a`（单成员）。
- 2026-09-06 的旧回执缺 v2 字段（release 目录内 runner 取自当时 verification
  commit，早于 0fafad0 门禁迁移）；以 HEAD `a685bd5` 为 verification commit
  重跑 release：7 tests / 20 kernel launches / 0 失败，回执与日志存
  `artifacts/competition/l2norm/e1-f83c73a/`，SHA-256
  `00506003196f9637dd7dba22a7fadbe6d34903986dc3c184f95292b8aefeb599`。
- preflight 全门禁通过（task=competing、额度充足、元组逐项核对）后按持续
  授权自动提交；评测中。s0 的 8/8 与 3.1050x 在 e1 出分前保持 team best。
