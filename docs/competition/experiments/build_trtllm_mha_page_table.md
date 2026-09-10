# Task 59 `build_trtllm_mha_page_table` 实验记录

```current
task: 59
operator: build_trtllm_mha_page_table
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

- 完整题面：[Task 59](../tasks/batch-5/59-build_trtllm_mha_page_table.md)；[开发契约](../strategy-batch5.md)。
- exact；八芯每芯 0.1x；截止 2026-09-17 19:59:59（北京时间）。
- 按公开 reference 返回新张量并保持输入不变；核心计算使用 Triton，无 fallback。
- generic S0；NVIDIA 代理已验证，平台及其他目标 runtime 标记 `target-runtime-unverified`。
- 本轮未执行平台 preflight、上传或正式提交，不消耗额度。

## 不可变身份

- source commit：`4836fe0378d6290cbcaa91958693e13ed1acdf3e`。
- verification commit：`4836fe0378d6290cbcaa91958693e13ed1acdf3e`。
- ledger commit：本文件的 Git 提交；用 `git log -1 -- docs/competition/experiments/build_trtllm_mha_page_table.md` 定位。
- source SHA-256：`0b13b86253e8e07948c08426725d481614a0df5a4ed1937b9499ca98e5f2ca2b`。
- test SHA-256：`56acee3177490d4675aa3f3b824e9819604494113e0f111f352e4e6ab1cacd97`。
- ZIP：`artifacts/competition/build_trtllm_mha_page_table/s0-4836fe0/build_trtllm_mha_page_table.zip`，2782 bytes，仅含 `build_trtllm_mha_page_table.py`。
- ZIP SHA-256（等于 canonical）：`45ede2a69cea1900a3ac827a814cf4091fc59e4290163ab1f18a17ecca138450`。
- dry-run manifest、最终构建、existing 验签、Git blob 逐字节比对及 unzip -t/-l 全部通过。

## 验证证据

- 4 个测试方法 / 54 条 test/subTest 记录 / 25 次实际 kernel launch；失败、错误、skip、xfail 均为 0。
- RTX 5070 Ti；Python 3.12.13；PyTorch 2.13.0+cu130；Triton 3.7.1；CUDA 13.0。
- 回执：`artifacts/competition/batch5-development-20260910/release/build_trtllm_mha_page_table/verification.json`。
- 回执 SHA-256：`d5f5e8ac19c18c37b0a1ff30334400279e5899ee6c419c3ee40fcf66e3beb1e3`。
- 完整日志：`artifacts/competition/batch5-development-20260910/release/build_trtllm_mha_page_table/verification.log`。
- 日志 SHA-256：`4555b2407d0ece95d2aa57015857e2100e5e739f535a21119479c0de54da9912`。
- benchmark：`artifacts/competition/batch5-development-20260910/release/build_trtllm_mha_page_table/benchmark.json`，SHA-256 `6ec15ea98b01ef91de136bebe3e781d985c60d686a103cced807b6fcf058502c`。
- Black/isort/flake8、Python 编译、空输入、边界、stride 和输入不变性均已检查。
- 所有 source/test/helper/runner/benchmark 哈希见[证据清单](../data/batch5-development-20260910.json)，清单 SHA-256 `74b37287f5649dc044696565306c97111117c991b3549393a88bcb703eba46b1`。

## 代理性能

| 代理负载 | 五轮配对 speedup 中位数 | 候选中位耗时 |
| --- | ---: | ---: |
| 8x128 | 14.5732x | 4.859 μs |
| 64x2048 | 9.2933x | 8.351 μs |

测速含 wrapper；五轮交替 AB/BA，每次 warmup 20ms / rep 50ms。
这些是两种代理负载的结果，不是平台平均分；原始样本已保留。

## 重放

远端隔离目录：`/tmp/flagos-batch5.POXFt3/build_trtllm_mha_page_table`；复跑需 prepare 到新目录，旧回执不可覆盖。

```bash
python .agents/skills/flagos-operator-race/scripts/verify_release.py prepare build_trtllm_mha_page_table --source-commit 4836fe0378d6290cbcaa91958693e13ed1acdf3e --verification-commit 4836fe0378d6290cbcaa91958693e13ed1acdf3e --dependency tools/benchmark_batch5.py --directory /tmp/NEW_RELEASE_DIRECTORY
timeout 600 /home/kevin/notebook/.venv/bin/python /tmp/NEW_RELEASE_DIRECTORY/.agents/skills/flagos-operator-race/scripts/verify_release.py run --directory /tmp/NEW_RELEASE_DIRECTORY
```

## 2026-09-11 一轮优化后的待提交候选

- 候选 `e1`；按 tile 调度优化晋级 E1。首次提交前候选就绪。
- source commit：`29754c1dfcbc87c9b193e750145b668c8e35d852`；verification commit：`29754c1dfcbc87c9b193e750145b668c8e35d852`。
- ZIP：`/Users/bytedance/ccc/flagos/artifacts/competition/build_trtllm_mha_page_table/e1-29754c1/build_trtllm_mha_page_table.zip`；2830 bytes；成员 `build_trtllm_mha_page_table.py`。
- ZIP SHA-256：`432e1896094351992b41ce3dba289ec89780f200bbbb75100f31e18920b9fb9a`。
- 源码 SHA-256：`92d3c574e999a0bfb718010aca583ef26aeeea5b05ff53a8962d636cbcade0a7`。
- 测试 SHA-256：`558d5955014cd6e6c7eb0350e1aadd392b61867d12cefef7485ccda9cb6e554a`。
- 回执：`artifacts/competition/batch5-optimize-20260911/release/build_trtllm_mha_page_table/verification.json`；SHA-256 `4ac4aa0c4eb0e8dbbb66a105f9417ce67ce1b1e019843f6fc3ffe681e0776b40`。
- 日志：`artifacts/competition/batch5-optimize-20260911/release/build_trtllm_mha_page_table/verification.log`；SHA-256 `a616b02c5d58abde49ec9c7120ee6c91ab771a8bb97568fcead5054d0341e147`。
- 完整 release：4 个测试方法 / 55 条记录 / 26 次 kernel launch，全通过；NVIDIA 代理范围。
- 优化和复现见[本轮报告](../optimization-batch5-20260911.md)；[完整证据](../data/batch5-optimization-20260911.json)，SHA-256 `87a560dd86ab7e9dfb9607f8f05a007060d48b3aab876b165bfc0fec591f77db`。
