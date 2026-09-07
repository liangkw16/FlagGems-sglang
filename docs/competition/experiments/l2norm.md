# Task 56 `l2norm` 实验记录

```current
task: 56
operator: l2norm
batch: 4
validity: valid
platform: 8/8(e1,3.10772917x,排名8)
team_best_stage: e1
team_best_commit: f83c73aeceaa6382c6ec6d93a082fc24e37090fb
team_best_speedup: 3.10772917
sealed: no
next: 短D大行数候选最终代理通过/ZIP验签;计分形状和其他适用芯片复验,小形状保留旧核
updated: 2026-09-08
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
  授权自动提交；**8/8 复验通过，avg 3.10772917x 新 team best**（与 s0 同
  水位，证明修复只影响高维正确性不影响性能）。
- 逐芯：天数 6.68 / 沐曦 2.59 / 燧原 1.19 / 海光 4.57 / 昆仑 0.585 /
  华为 1.41 / A 3.63 / B 4.21。"账本 8/8 与产物证据"的冲突至此消除：
  当前平台记录即修复字节。

## 2026-09-07 只读盘点校正

当前榜首与排名按最新任务API校正；旧段落保留历史快照，不据此分配本轮提交机会。本轮未修改或提交本题源码。
- 查询证据 `/Users/bytedance/ccc/flagos/artifacts/competition/batch4-top1-20260907/tasks-now.json` SHA256 `fc73368c3d98b228b0c7815d6ec1e9042a58af8d953337fff58990daec1474fc`。

## 2026-09-08 推荐方案实现与提交前验证（未提交平台）

多行tile8限定 D<=128且rows>4096；小规模保留旧核。最终65536x64/128为4.97x/3.56x，小案例约持平；新增rows4095/4096/4097分流边界。

- source `c73f6c3f83ec38d5a2c40cfdef996e64e50ecd67`；verification `c73f6c3f83ec38d5a2c40cfdef996e64e50ecd67`。8 个测试方法、128 次实际 kernel 调用；选定 NVIDIA/代理范围门禁通过。
- 回执 `artifacts/competition/batch4-implementation-20260907/t56-release2/verification.json`，SHA256 `6ee0122303a516aa07bde4b47364e2d6d42821cd0ec33406b8cd89a791302ec2`；日志 SHA256 `a23392a257e8c60d95702ba0414007f5325547c54f5c3db81abd42d685ba4854`。
- 不可变 ZIP `artifacts/competition/l2norm/research-20260908-c73f6c3/l2norm.zip`，SHA256 `e31e9c826871d4d786fffbfe866590ac0d31e4c49dda8d6924e195b5b25c1a39`；与 dry-run manifest、构建和 existing 验签一致。ZIP 是候选产物，不等于目标芯或平台已通过。
- 环境、逐源码执行范围、原始配对数据和未完成条件见[本轮报告](../implementation-batch4-20260908.md)及[证据清单](../data/batch4-implementation-20260908.json)。本轮不更新历史有效分，未做平台 preflight、上传或正式提交。
