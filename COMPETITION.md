# FlagOS 第二届算子挑战赛本地资料

本工作区已基于官方 `FlagGems-sglang` 仓库整理成可直接开发和检索的资料包。

- [比赛要求、提交规范与资料入口](docs/competition/README.md)
- [全五批赛题索引与动态榜单快照](docs/competition/task-index.md)
- **第五批终局三件套（09-17 收官口径）**：
  [终日作战与逐发判决](docs/competition/optimization-batch5-finalday-20260917.md)、
  [终局榜与下一赛季开题情报](docs/competition/batch5-final-review-20260917.md)、
  [全季复盘与跨芯知识库](docs/competition/season2-retrospective.md)
- [第二批候选、产物哈希与提交队列](docs/competition/experiments/README.md)（顶部为 09-17 终局收口）
- [参考仓库与本地 Git 引用](docs/competition/reference-repositories.md)
- 历史作战文档（按文件名日期归档，状态均已过期，逐题真相见
  [生成索引](docs/competition/experiments/INDEX.md)）：
  [第二批策略（08-23）](docs/competition/strategy-batch2.md)、
  [前两批 24 题图谱](docs/competition/operator-atlas.md)、
  [41 题上游打法（08-31 快照）](docs/competition/upstream-operator-playbook.md)、
  [第三批打法速查（08-29）](docs/competition/batch3-upstream-playbook.md)、
  [跨芯优化方案（第二批历史稿，边界以复盘 §14/§15 为准）](docs/competition/cross-chip-optimization-plan.md)、
  [题型学习与八芯资料入口](docs/competition/learning-path.md)、
  [八芯公开规格与编译期约束](docs/competition/chip-landscape.md)、
  [第四批逐芯调研（09-07）](docs/competition/research-batch4-vendor-20260907.md)、
  [第四批实现与候选 ZIP（09-08）](docs/competition/implementation-batch4-20260908.md)、
  [会话挖掘复盘（09-11）](docs/competition/session-mining-retrospective.md)、
  [第五批 R2（09-11）](docs/competition/optimization-batch5-r2-20260911.md)、
  [第五批 R3（09-11）](docs/competition/optimization-batch5-r3-20260911.md)、
  以及 09-12~09-16 的 r4/status/strategy/leaderboard/top1 系列过程文档
- 厂商 backend 源码缓存：`docs/competition/data/vendor-backends/`
- 完整题面：`docs/competition/tasks/`
- 结构化赛题数据：`docs/competition/data/task-catalog.json`
- 公开赛程/芯片/统计快照：`docs/competition/data/race-overview.json`

常用检索：

```bash
rg -n "接口签名|参考实现|正确性判别" docs/competition/tasks
rg -n "softcap_out|fused_rmsnorm" docs/competition src tests benchmark
git grep -n "decode_attention" community/master
jq '.tasks[] | select(.batch_no == 2)' docs/competition/data/task-catalog.json
```

更新公开题面和榜单快照：

```bash
python tools/sync_flagos_season2_docs.py
```

刷新或验签厂商 backend 源码缓存（跨芯约束审查用，离线可读）：

```bash
python tools/fetch_vendor_backends.py --verify
```
