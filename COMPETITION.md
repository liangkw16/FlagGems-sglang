# FlagOS 第二届算子挑战赛本地资料

本工作区已基于官方 `FlagGems-sglang` 仓库整理成可直接开发和检索的资料包。

- [比赛要求、提交规范与资料入口](docs/competition/README.md)
- [第一批、第二批赛题索引](docs/competition/task-index.md)
- [第二批快速开发策略](docs/competition/strategy-batch2.md)
- [第二批候选、产物哈希与提交队列](docs/competition/experiments/README.md)
- [参考仓库与本地 Git 引用](docs/competition/reference-repositories.md)
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
