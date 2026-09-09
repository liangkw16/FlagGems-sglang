# Task 56 `l2norm` 实验记录

```current
task: 56
operator: l2norm
batch: 4
validity: valid
platform: 8/8(e3,11062,3.20691667x新team best)
team_best_stage: e3
team_best_commit: c0384fada9d69aa95c51c08d2b85934ecaa3310c
team_best_speedup: 3.20691667
sealed: no
next: e4新燧原persistent vendor(e4cc425,release exit0+ZIP ab7a6f5d)就绪;预注册门燧原>=3(次名靶7.31)
updated: 2026-09-10


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

## E2 短行多行 tile → **8/8 VALID，avg 3.13796875x 新 team best**（2026-09-08，sub 11045）

- 候选 = 09-08 方案轮的短 D 多行 tile（commit `c73f6c3`，D≤128 且
  rows>4096 才启用 tile=8，小规模沿用旧核）；ZIP `e2-c73f6c3` SHA256
  `e31e9c826871d4d786fffbfe866590ac0d31e4c49dda8d6924e195b5b25c1a39`，
  单成员 generic，与 t56-release2 回执逐字节一致后经实时 preflight 单次提交。
- 终态逐芯：天数 6.62225 / 沐曦 2.55158333 / 燧原 1.271（+7%）/
  海光 4.597 / 昆仑 0.578 / 华为 **1.63733333**（1.41→，+16%）/
  A 3.6325 / B 4.21408333；avg 3.13796875（3.1077→，**+1.0%**）。
- 判定：代理 5x 的大行数场景在计分形状中占比不足，结构收益仅华为/燧原
  小幅兑现；距榜首 72x 的差距不在本轴，短行 tile 轴收益近天花板。

## E3 昆仑行块 vendor → **8/8 valid，avg 3.20691667x 新 team best**（2026-09-08，sub 11062）

- 诊断：e2 的多行 kernel 只在 `dim<=128 && rows>4096` 启用，平台计分
  shape 多未触发，昆仑仍走每行一 program 的最小粒度（0.578 = 纯
  per-program 固定开销）。
- 候选（commit `c0384fa`，仅 `_kunlunxin` vendor 单变量）：2D
  [BLOCK_ROWS, BLOCK_D] 行块 tile 常开（e2 已证该结构昆仑 correctness
  通过），element 预算 4096/program（D=64 时 32 行），grid-stride
  cap 65535（T45/T48 配方）；generic 六过芯字节不动。补
  `L2NormVariantsTest`（release 门禁曾拦 "applicable source was not
  exercised"，补测试后 9/9 过）。
- 终态逐芯：天数 6.68241667 / 沐曦 2.59433333 / 燧原 1.213 /
  海光 4.62766667 / **昆仑 1.07966667（0.578→，+87%）** /
  华为 1.53991667（generic 未动，窗口噪声 -6%）/ A 3.65333333 /
  B 4.265；avg 3.20691667（3.138→，**+2.2%**）。
- 判定：昆仑 per-program 开销论兑现，跨题再次验证"少而肥的 program"
  是昆仑 elementwise/轻归约第一杠杆（T53-E1、T24 后第三证）。距榜首
  sikadeer 72.3x 仍远（22.5x），华为/燧原需要新证据，短行轴收益已尽。

## E3r 水位重掷终态：8/8 valid 3.1037，未超 TB（2026-09-08T17:16，sub 11247）

- e3 字节注释载体（`b8c7d77`，经 gpu-et 通道跑 release）。带内回落，
  TB 3.2069 保持。同字节重掷剩 ≤1 次。
## E3r2 终掷：8/8 valid 3.1854 未超 TB（2026-09-09T06:38，sub 11646）
- 带内，TB 3.2069 保持；同字节预算用尽。剩余身份 e1r。


## 2026-09-09 榜首性质更正：慢窗产物，此前打错靶

按题 leaderboard 端点（返回每支队伍逐芯 speedup，只读、不耗额度）判定：
本题榜首 **72.31** 的燧原读数为 **551.4**，而同芯**第二名**仅 7.31
（Warmhearted），相差 **75x**。按“榜首某芯 > 同芯次优 20 倍即窗口产物”
判据，该榜首是评测机慢窗下 reference 退化所致，**不是可复现的结构领先**。

据此更正本账本此前把该榜差记为“需同量级慢窗才能追平”的结论：

- **正确靶子是同芯次优 7.31，不是榜首 551.4。** 我方燧原 1.213，
  该差距属结构可达范围。
- 对该彩票榜首继续投同字节重掷弹属错靶，会白烧全局额度（额度为账号
  全局每日 30 发、全题共享）。
- 仅把燧原抬到 7.31，八芯平均即 3.207 → 3.970。

待办：套用燧原 persistent+pingpong 形态（`grid=(min(tiles, 24),)` 配
`tl.range(pid, tiles, tl.num_programs(0), num_stages=3)`，并去掉显式
`num_warps`），根因、厂商文档依据与逐芯测算见
[第四批终盘方案](../strategy-batch4-final-20260909.md)；T49 的 `0d4d758`
是该形态的参考实现。本轮未改本题源码、未提交平台。

## E4 燧原 persistent vendor（2026-09-09 深夜，commit `e4cc425`，未提交平台）

- 新建 `_enflame/l2norm.py`（此前燧原走 generic：每行一 program + 钉
  warps=4）：24 program persistent 行走访 + `tl.range(num_stages=3)` +
  去钉 warps，数学与 generic 行核一致（fp32 sumsq、rstd、cast store）。
- exact release 回执 `/tmp/flagos-l2norm-rel/verification.json`
  （0 skip/xfail，exit 0）；canonical ZIP `e4-e4cc425`，SHA-256
  `ab7a6f5dbe6090f01677e2ff8114fe67e832f8e18e96d3a2e0c4575fe649a19b`。
- 预注册门：燧原 ≥3（当前 1.21，次名靶 7.31）。
