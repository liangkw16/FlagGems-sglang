# Task 62 `concat_mla_k` 实验记录

```current
task: 62
operator: concat_mla_k
batch: 5
validity: pending
platform: evaluating(12897,7/8)
candidate_stage: s0
team_best_stage: -
sealed: no
next: 只读跟进 12897 昆仑芯回调；不重投、不计算部分均值
updated: 2026-09-11
```

> 下方 S0 开发记录是 2026-09-10 快照；当前平台结果见 CURRENT 和文末提交记录。

## 契约与范围

- 完整题面：[Task 62](../tasks/batch-5/62-concat_mla_k.md)；[开发契约](../strategy-batch5.md)。
- exact；八芯每芯 0.1x；截止 2026-09-17 19:59:59（北京时间）。
- 按公开 reference 返回新张量并保持输入不变；核心计算使用 Triton，无 fallback。
- generic S0；NVIDIA 代理已验证，平台及其他目标 runtime 标记 `target-runtime-unverified`。
- 本轮未执行平台 preflight、上传或正式提交，不消耗额度。

## 不可变身份

- source commit：`7b53fdeb58878f8eeaf08eaabbde8bb38015e523`。
- verification commit：`7b53fdeb58878f8eeaf08eaabbde8bb38015e523`。
- ledger commit：本文件的 Git 提交；用 `git log -1 -- docs/competition/experiments/concat_mla_k.md` 定位。
- source SHA-256：`484482bdd03e429de677294ba191474ee0411fb84505743731b43b3ed4a87738`。
- test SHA-256：`756fddfbd1d43e54a6dd7efa99f4f99d39b36beb32ca4f56ea34df5baade7f39`。
- ZIP：`artifacts/competition/concat_mla_k/s0-7b53fde/concat_mla_k.zip`，2422 bytes，仅含 `concat_mla_k.py`。
- ZIP SHA-256（等于 canonical）：`bb3d1f2f89cfe0a5b19c31e4f2ac9b66ee5ddc4c8a93d6f399b5c55c0f46396e`。
- dry-run manifest、最终构建、existing 验签、Git blob 逐字节比对及 unzip -t/-l 全部通过。

## 验证证据

- 5 个测试方法 / 60 条 test/subTest 记录 / 28 次实际 kernel launch；失败、错误、skip、xfail 均为 0。
- RTX 5070 Ti；Python 3.12.13；PyTorch 2.13.0+cu130；Triton 3.7.1；CUDA 13.0。
- 回执：`artifacts/competition/batch5-development-20260910/release-t62-address/verification.json`。
- 回执 SHA-256：`25ab3c60c07f2c16859b1b438144f51bd61eef00f10dca8458149b9d2813c9f9`。
- 完整日志：`artifacts/competition/batch5-development-20260910/release-t62-address/verification.log`。
- 日志 SHA-256：`59bc9109ee99dfaccf0d2ff7e31dced070ba4ce6181134f61af50d473ce0c751`。
- benchmark：`artifacts/competition/batch5-development-20260910/release-t62-address/benchmark.json`，SHA-256 `1f47b314a053788bcd84446d8babbe98b13f0d23280eb6a39b3bdff58b8c0c9b`。
- Black/isort/flake8、Python 编译、空输入、边界、stride 和输入不变性均已检查。
- 所有 source/test/helper/runner/benchmark 哈希见[证据清单](../data/batch5-development-20260910.json)，清单 SHA-256 `74b37287f5649dc044696565306c97111117c991b3549393a88bcb703eba46b1`。

## 代理性能

| 代理负载 | 五轮配对 speedup 中位数 | 候选中位耗时 |
| --- | ---: | ---: |
| 32x128x192 | 1.6763x | 6.363 μs |
| 1024x128x192 | 1.4573x | 114.030 μs |

测速含 wrapper；五轮交替 AB/BA，每次 warmup 20ms / rep 50ms。
这些是两种代理负载的结果，不是平台平均分；原始样本已保留。

## 重放

远端隔离目录：`/tmp/flagos-t62-address.d9FYAR`；复跑需 prepare 到新目录，旧回执不可覆盖。

```bash
python .agents/skills/flagos-operator-race/scripts/verify_release.py prepare concat_mla_k --source-commit 7b53fdeb58878f8eeaf08eaabbde8bb38015e523 --verification-commit 7b53fdeb58878f8eeaf08eaabbde8bb38015e523 --dependency tools/benchmark_batch5.py --directory /tmp/NEW_RELEASE_DIRECTORY
timeout 600 /home/kevin/notebook/.venv/bin/python /tmp/NEW_RELEASE_DIRECTORY/.agents/skills/flagos-operator-race/scripts/verify_release.py run --directory /tmp/NEW_RELEASE_DIRECTORY
```

## 最终地址修复

初版受测范围全过，静态复核发现列索引与 stride 乘法仍可在 int32 域溢出。
最终版先扩位到 int64；新增共享 4 GiB 稀疏 storage 的 NoPE/RoPE 视图，
实际元素偏移达到 2^31。5 个方法全部通过，包含该回归；最终 ZIP 只使用修复字节。
初版回执保留于本轮 release/concat_mla_k/，不作为最终候选证据。

## 2026-09-11 一轮优化后的待提交候选

- 候选 `s0`；本轮审查后保留既有 S0。首次提交前候选就绪。
- source commit：`7b53fdeb58878f8eeaf08eaabbde8bb38015e523`；verification commit：`7b53fdeb58878f8eeaf08eaabbde8bb38015e523`。
- ZIP：`/Users/bytedance/ccc/flagos/artifacts/competition/concat_mla_k/s0-7b53fde/concat_mla_k.zip`；2422 bytes；成员 `concat_mla_k.py`。
- ZIP SHA-256：`bb3d1f2f89cfe0a5b19c31e4f2ac9b66ee5ddc4c8a93d6f399b5c55c0f46396e`。
- 源码 SHA-256：`484482bdd03e429de677294ba191474ee0411fb84505743731b43b3ed4a87738`。
- 测试 SHA-256：`756fddfbd1d43e54a6dd7efa99f4f99d39b36beb32ca4f56ea34df5baade7f39`。
- 回执：`artifacts/competition/batch5-development-20260910/release-t62-address/verification.json`；SHA-256 `25ab3c60c07f2c16859b1b438144f51bd61eef00f10dca8458149b9d2813c9f9`。
- 日志：`artifacts/competition/batch5-development-20260910/release-t62-address/verification.log`；SHA-256 `59bc9109ee99dfaccf0d2ff7e31dced070ba4ce6181134f61af50d473ce0c751`。
- 完整 release：5 个测试方法 / 60 条记录 / 28 次 kernel launch，全通过；NVIDIA 代理范围。
- 优化和复现见[本轮报告](../optimization-batch5-20260911.md)；[完整证据](../data/batch5-optimization-20260911.json)，SHA-256 `87a560dd86ab7e9dfb9607f8f05a007060d48b3aab876b165bfc0fec591f77db`。

## 2026-09-11 首次平台提交（2026-09-11T03:42:29.113201+08:00）

- `s0` / submission `12897` / daily_seq `4`，提交于 `2026-09-11T03:32:09+08:00`。每候选上传与正式 POST 各一次，无自动重试。
- nonce：`18c5bff9bf55203d968c224b501fe54a`；上传 URL SHA-256：`011f6a1e8fcc0f0da62d0bb57577d40c62b8384bad25d47923041c7c50a1ddc5`。
- 远端 ZIP 回读 verified，2422 bytes，SHA-256 `bb3d1f2f89cfe0a5b19c31e4f2ac9b66ee5ddc4c8a93d6f399b5c55c0f46396e`，与本地候选逐字节一致。
- 平台 `evaluating` / `pending`；正确性通过 7/8，完成 7/8；未产生八芯均值；未入有效榜。
- 当次 status 额度：24/30，已用 6；observed_at `2026-09-11T03:42:29.113201+08:00`。排名查询时间 `2026-09-11T03:39:41.445754+08:00`。

| 芯片 | 状态 | 正确性 | speedup |
| --- | --- | --- | ---: |
| tianshu | completed | 通过 | 2.6114 |
| muxi | completed | 通过 | 0.9836 |
| enflame | completed | 通过 | 0.109 |
| haiguang | completed | 通过 | 3.1072 |
| kunlunxin | waiting_callback | 待定 | — |
| huawei | completed | 通过 | 0.2188 |
| card_a | completed | 通过 | 1.7242 |
| card_b | completed | 通过 | 1.8166 |

昆仑芯仍为 waiting_callback，不能计为失败；燧原已通过但仅 0.109x。此次额外只读等待 90 秒仍未收齐结果（watch 退出码 124），保留原提交继续查分，不重投。

下一步：只读跟进 12897 昆仑芯回调；不重投、不计算部分均值。

[提交结果证据](../data/batch5-submissions-20260911.json)，SHA-256 `63418b87e9249bef72751df2a1778c52e6d679a5d313ecf18d8ed64b96c175dd`；原始 status `artifacts/competition/batch5-submit-20260911/62-status-034229.json`，SHA-256 `6a47dc86e76134e17c1e2754ca9db72f2bf8069fa4775aa98b556fdbd1a70359`。
