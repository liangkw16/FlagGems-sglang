# Task 60 `clamp_position` 实验记录

```current
task: 60
operator: clamp_position
batch: 5
validity: candidate-ready
platform: not-submitted
candidate_stage: s0
team_best_stage: -
sealed: no
next: S0 已通过提交前本地门禁；正式提交时重新进行实时 preflight
updated: 2026-09-10
```

## 契约与范围

- 完整题面：[Task 60](../tasks/batch-5/60-clamp_position.md)；[开发契约](../strategy-batch5.md)。
- exact；八芯每芯 0.1x；截止 2026-09-17 19:59:59（北京时间）。
- 按公开 reference 返回新张量并保持输入不变；核心计算使用 Triton，无 fallback。
- generic S0；NVIDIA 代理已验证，平台及其他目标 runtime 标记 `target-runtime-unverified`。
- 本轮未执行平台 preflight、上传或正式提交，不消耗额度。

## 不可变身份

- source commit：`4836fe0378d6290cbcaa91958693e13ed1acdf3e`。
- verification commit：`4836fe0378d6290cbcaa91958693e13ed1acdf3e`。
- ledger commit：本文件的 Git 提交；用 `git log -1 -- docs/competition/experiments/clamp_position.md` 定位。
- source SHA-256：`fc637bc6d0bccd8a38d11100aa3589feda86568aa4fa8383aca7472001c9ed2b`。
- test SHA-256：`fcfe27bc9c551018822dd8649ffa03ae8077f810d77596e078e94d270e3d8a15`。
- ZIP：`artifacts/competition/clamp_position/s0-4836fe0/clamp_position.zip`，1085 bytes，仅含 `clamp_position.py`。
- ZIP SHA-256（等于 canonical）：`506af23ed82335d1a93965436545c7c4bd646b09b5f589c642fcfadecc1e7209`。
- dry-run manifest、最终构建、existing 验签、Git blob 逐字节比对及 unzip -t/-l 全部通过。

## 验证证据

- 3 个测试方法 / 45 条 test/subTest 记录 / 22 次实际 kernel launch；失败、错误、skip、xfail 均为 0。
- RTX 5070 Ti；Python 3.12.13；PyTorch 2.13.0+cu130；Triton 3.7.1；CUDA 13.0。
- 回执：`artifacts/competition/batch5-development-20260910/release/clamp_position/verification.json`。
- 回执 SHA-256：`ec3aa53405854fb342081f3dc9a60f1ab2b9d81fd2f8ed30444a782cf59adb96`。
- 完整日志：`artifacts/competition/batch5-development-20260910/release/clamp_position/verification.log`。
- 日志 SHA-256：`8f9cc4892f5224d878a14add6b989ebfd03c73bef69d5bdcd79e858ab50dd5eb`。
- benchmark：`artifacts/competition/batch5-development-20260910/release/clamp_position/benchmark.json`，SHA-256 `cfd447d2ad66b8bb7054ed2b1b119d740d8c15ea5486516b7e19f937d72a4678`。
- Black/isort/flake8、Python 编译、空输入、边界、stride 和输入不变性均已检查。
- 所有 source/test/helper/runner/benchmark 哈希见[证据清单](../data/batch5-development-20260910.json)，清单 SHA-256 `74b37287f5649dc044696565306c97111117c991b3549393a88bcb703eba46b1`。

## 代理性能

| 代理负载 | 五轮配对 speedup 中位数 | 候选中位耗时 |
| --- | ---: | ---: |
| 32 | 1.1747x | 4.260 μs |
| 32768 | 1.2864x | 4.886 μs |

测速含 wrapper；五轮交替 AB/BA，每次 warmup 20ms / rep 50ms。
这些是两种代理负载的结果，不是平台平均分；原始样本已保留。

## 重放

远端隔离目录：`/tmp/flagos-batch5.POXFt3/clamp_position`；复跑需 prepare 到新目录，旧回执不可覆盖。

```bash
python .agents/skills/flagos-operator-race/scripts/verify_release.py prepare clamp_position --source-commit 4836fe0378d6290cbcaa91958693e13ed1acdf3e --verification-commit 4836fe0378d6290cbcaa91958693e13ed1acdf3e --dependency tools/benchmark_batch5.py --directory /tmp/NEW_RELEASE_DIRECTORY
timeout 600 /home/kevin/notebook/.venv/bin/python /tmp/NEW_RELEASE_DIRECTORY/.agents/skills/flagos-operator-race/scripts/verify_release.py run --directory /tmp/NEW_RELEASE_DIRECTORY
```
