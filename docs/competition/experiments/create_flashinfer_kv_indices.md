# Task 63 `create_flashinfer_kv_indices` 实验记录

```current
task: 63
operator: create_flashinfer_kv_indices
batch: 5
validity: candidate-ready
platform: not-submitted
candidate_stage: e1
team_best_stage: -
sealed: no
next: E1 已验签；实时 preflight 后首次提交
updated: 2026-09-11
```

## 契约与范围

- 完整题面：[Task 63](../tasks/batch-5/63-create_flashinfer_kv_indices.md)；[开发契约](../strategy-batch5.md)。
- exact；八芯每芯 0.1x；截止 2026-09-17 19:59:59（北京时间）。
- 按公开 reference 返回新张量并保持输入不变；核心计算使用 Triton，无 fallback。
- generic S0；NVIDIA 代理已验证，平台及其他目标 runtime 标记 `target-runtime-unverified`。
- 本轮未执行平台 preflight、上传或正式提交，不消耗额度。

## 不可变身份

- source commit：`4836fe0378d6290cbcaa91958693e13ed1acdf3e`。
- verification commit：`4836fe0378d6290cbcaa91958693e13ed1acdf3e`。
- ledger commit：本文件的 Git 提交；用 `git log -1 -- docs/competition/experiments/create_flashinfer_kv_indices.md` 定位。
- source SHA-256：`0507dd1780757b11f1f253699c7927c63f915631f8d0b11c5ad7065fe3271688`。
- test SHA-256：`3a5324770f4298207e893d77412fb65ac51a4926887c558b87ddfaf775ae5642`。
- ZIP：`artifacts/competition/create_flashinfer_kv_indices/s0-4836fe0/create_flashinfer_kv_indices.zip`，2782 bytes，仅含 `create_flashinfer_kv_indices.py`。
- ZIP SHA-256（等于 canonical）：`a72813fa2eed3767f2f79a3a6d3fa2d9a3aad07ffc1186e04d944ed6f7bf9489`。
- dry-run manifest、最终构建、existing 验签、Git blob 逐字节比对及 unzip -t/-l 全部通过。

## 验证证据

- 4 个测试方法 / 25 条 test/subTest 记录 / 13 次实际 kernel launch；失败、错误、skip、xfail 均为 0。
- RTX 5070 Ti；Python 3.12.13；PyTorch 2.13.0+cu130；Triton 3.7.1；CUDA 13.0。
- 回执：`artifacts/competition/batch5-development-20260910/release/create_flashinfer_kv_indices/verification.json`。
- 回执 SHA-256：`7c46aa0c45e86bfb1b849edf5301d7350136e2046f278aa59608fbbc4c5b711d`。
- 完整日志：`artifacts/competition/batch5-development-20260910/release/create_flashinfer_kv_indices/verification.log`。
- 日志 SHA-256：`7a4d6b0e6d83d8fe9c0d8512b9517afc3ac8e45a69536f1c8a1836c7830dcd3e`。
- benchmark：`artifacts/competition/batch5-development-20260910/release/create_flashinfer_kv_indices/benchmark.json`，SHA-256 `607b5c74a7ddd46cedfa7a359b3da49a63ef3e6e88dbc1d908aeb08347c23f4d`。
- Black/isort/flake8、Python 编译、空输入、边界、stride 和输入不变性均已检查。
- 所有 source/test/helper/runner/benchmark 哈希见[证据清单](../data/batch5-development-20260910.json)，清单 SHA-256 `74b37287f5649dc044696565306c97111117c991b3549393a88bcb703eba46b1`。

## 代理性能

<!-- ponytail: 每请求一个 program；长段利用率成为瓶颈时再将请求拆成并行 tile。 -->

| 代理负载 | 五轮配对 speedup 中位数 | 候选中位耗时 |
| --- | ---: | ---: |
| 1x8193 | 2.7824x | 12.375 μs |
| 32x1024 | 112.9101x | 7.566 μs |

测速含 wrapper；五轮交替 AB/BA，每次 warmup 20ms / rep 50ms。
这些是两种代理负载的结果，不是平台平均分；原始样本已保留。

## 重放

远端隔离目录：`/tmp/flagos-batch5.POXFt3/create_flashinfer_kv_indices`；复跑需 prepare 到新目录，旧回执不可覆盖。

```bash
python .agents/skills/flagos-operator-race/scripts/verify_release.py prepare create_flashinfer_kv_indices --source-commit 4836fe0378d6290cbcaa91958693e13ed1acdf3e --verification-commit 4836fe0378d6290cbcaa91958693e13ed1acdf3e --dependency tools/benchmark_batch5.py --directory /tmp/NEW_RELEASE_DIRECTORY
timeout 600 /home/kevin/notebook/.venv/bin/python /tmp/NEW_RELEASE_DIRECTORY/.agents/skills/flagos-operator-race/scripts/verify_release.py run --directory /tmp/NEW_RELEASE_DIRECTORY
```

## 2026-09-11 一轮优化后的待提交候选

- 候选 `e1`；按 tile 调度优化晋级 E1。首次提交前候选就绪。
- source commit：`29754c1dfcbc87c9b193e750145b668c8e35d852`；verification commit：`29754c1dfcbc87c9b193e750145b668c8e35d852`。
- ZIP：`/Users/bytedance/ccc/flagos/artifacts/competition/create_flashinfer_kv_indices/e1-29754c1/create_flashinfer_kv_indices.zip`；2992 bytes；成员 `create_flashinfer_kv_indices.py`。
- ZIP SHA-256：`6d9d37ebf39521f46697f8dc1b725c4bd6ad18e21c3fe7b6bc22e48bdb6c0ac8`。
- 源码 SHA-256：`ce7c07d20f2a4c585e27757e13f2efb32c8069f08767aacc4bbca1647c2bbe96`。
- 测试 SHA-256：`6a353892e59c29ca98efc7bfa4225c3f98d6424b85571cd7921a709aa95c5809`。
- 回执：`artifacts/competition/batch5-optimize-20260911/release/create_flashinfer_kv_indices/verification.json`；SHA-256 `4e5229ce738bac8b177f44295dcdb7b78f5da10f851176f8d2462106276c96ba`。
- 日志：`artifacts/competition/batch5-optimize-20260911/release/create_flashinfer_kv_indices/verification.log`；SHA-256 `04e747df25c1e016e2a5586f05fcd04942ef7912bcdebe0539d538ab2f02ce6f`。
- 完整 release：4 个测试方法 / 28 条记录 / 16 次 kernel launch，全通过；NVIDIA 代理范围。
- 优化和复现见[本轮报告](../optimization-batch5-20260911.md)；[完整证据](../data/batch5-optimization-20260911.json)，SHA-256 `87a560dd86ab7e9dfb9607f8f05a007060d48b3aab876b165bfc0fec591f77db`。
