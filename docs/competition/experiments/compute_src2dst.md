# Task 61 `compute_src2dst` 实验记录

```current
task: 61
operator: compute_src2dst
batch: 5
validity: candidate-ready
platform: not-submitted
candidate_stage: s0
team_best_stage: -
sealed: no
next: S0 已验签；实时 preflight 后首次提交
updated: 2026-09-11
```

## 契约与范围

- 完整题面：[Task 61](../tasks/batch-5/61-compute_src2dst.md)；[开发契约](../strategy-batch5.md)。
- exact；八芯每芯 0.1x；截止 2026-09-17 19:59:59（北京时间）。
- 按公开 reference 返回新张量并保持输入不变；核心计算使用 Triton，无 fallback。
- generic S0；NVIDIA 代理已验证，平台及其他目标 runtime 标记 `target-runtime-unverified`。
- 本轮未执行平台 preflight、上传或正式提交，不消耗额度。

## 不可变身份

- source commit：`4836fe0378d6290cbcaa91958693e13ed1acdf3e`。
- verification commit：`4836fe0378d6290cbcaa91958693e13ed1acdf3e`。
- ledger commit：本文件的 Git 提交；用 `git log -1 -- docs/competition/experiments/compute_src2dst.md` 定位。
- source SHA-256：`f59a3ffa8e3a463be40f7e70e8036b656a6df689899928689d073ba983215051`。
- test SHA-256：`1708ab05b089f1ad702294338a252e62f1ae41b40517ddd58bdda3603d166887`。
- ZIP：`artifacts/competition/compute_src2dst/s0-4836fe0/compute_src2dst.zip`，1198 bytes，仅含 `compute_src2dst.py`。
- ZIP SHA-256（等于 canonical）：`607bc6f0bbfedcfe9783a7dd2dc0af606c0ed717a64094a10aae58f33909ea8e`。
- dry-run manifest、最终构建、existing 验签、Git blob 逐字节比对及 unzip -t/-l 全部通过。

## 验证证据

- 3 个测试方法 / 32 条 test/subTest 记录 / 19 次实际 kernel launch；失败、错误、skip、xfail 均为 0。
- RTX 5070 Ti；Python 3.12.13；PyTorch 2.13.0+cu130；Triton 3.7.1；CUDA 13.0。
- 回执：`artifacts/competition/batch5-development-20260910/release/compute_src2dst/verification.json`。
- 回执 SHA-256：`f53c9eec0f6b1d3688a73ff09e4b566d3b10c0fdc0614b9c22927cbf2fb78736`。
- 完整日志：`artifacts/competition/batch5-development-20260910/release/compute_src2dst/verification.log`。
- 日志 SHA-256：`faf9c0481adc62bbce9bb3072a076c4b194dff095fac21cb4d5a62726ce482fa`。
- benchmark：`artifacts/competition/batch5-development-20260910/release/compute_src2dst/benchmark.json`，SHA-256 `42a4b02465f17e30967fb236965ca9bf131170cde3f26b6062725a3d3d144bae`。
- Black/isort/flake8、Python 编译、空输入、边界、stride 和输入不变性均已检查。
- 所有 source/test/helper/runner/benchmark 哈希见[证据清单](../data/batch5-development-20260910.json)，清单 SHA-256 `74b37287f5649dc044696565306c97111117c991b3549393a88bcb703eba46b1`。

## 代理性能

| 代理负载 | 五轮配对 speedup 中位数 | 候选中位耗时 |
| --- | ---: | ---: |
| 256 | 1.4945x | 4.182 μs |
| 131072 | 1.3485x | 7.660 μs |

测速含 wrapper；五轮交替 AB/BA，每次 warmup 20ms / rep 50ms。
这些是两种代理负载的结果，不是平台平均分；原始样本已保留。

## 重放

远端隔离目录：`/tmp/flagos-batch5.POXFt3/compute_src2dst`；复跑需 prepare 到新目录，旧回执不可覆盖。

```bash
python .agents/skills/flagos-operator-race/scripts/verify_release.py prepare compute_src2dst --source-commit 4836fe0378d6290cbcaa91958693e13ed1acdf3e --verification-commit 4836fe0378d6290cbcaa91958693e13ed1acdf3e --dependency tools/benchmark_batch5.py --directory /tmp/NEW_RELEASE_DIRECTORY
timeout 600 /home/kevin/notebook/.venv/bin/python /tmp/NEW_RELEASE_DIRECTORY/.agents/skills/flagos-operator-race/scripts/verify_release.py run --directory /tmp/NEW_RELEASE_DIRECTORY
```

## 2026-09-11 一轮优化后的待提交候选

- 候选 `s0`；本轮审查后保留既有 S0。首次提交前候选就绪。
- source commit：`4836fe0378d6290cbcaa91958693e13ed1acdf3e`；verification commit：`4836fe0378d6290cbcaa91958693e13ed1acdf3e`。
- ZIP：`/Users/bytedance/ccc/flagos/artifacts/competition/compute_src2dst/s0-4836fe0/compute_src2dst.zip`；1198 bytes；成员 `compute_src2dst.py`。
- ZIP SHA-256：`607bc6f0bbfedcfe9783a7dd2dc0af606c0ed717a64094a10aae58f33909ea8e`。
- 源码 SHA-256：`f59a3ffa8e3a463be40f7e70e8036b656a6df689899928689d073ba983215051`。
- 测试 SHA-256：`1708ab05b089f1ad702294338a252e62f1ae41b40517ddd58bdda3603d166887`。
- 回执：`artifacts/competition/batch5-development-20260910/release/compute_src2dst/verification.json`；SHA-256 `f53c9eec0f6b1d3688a73ff09e4b566d3b10c0fdc0614b9c22927cbf2fb78736`。
- 日志：`artifacts/competition/batch5-development-20260910/release/compute_src2dst/verification.log`；SHA-256 `faf9c0481adc62bbce9bb3072a076c4b194dff095fac21cb4d5a62726ce482fa`。
- 完整 release：3 个测试方法 / 32 条记录 / 19 次 kernel launch，全通过；NVIDIA 代理范围。
- 优化和复现见[本轮报告](../optimization-batch5-20260911.md)；[完整证据](../data/batch5-optimization-20260911.json)，SHA-256 `87a560dd86ab7e9dfb9607f8f05a007060d48b3aab876b165bfc0fec591f77db`。
