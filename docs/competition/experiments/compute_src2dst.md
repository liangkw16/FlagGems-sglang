# Task 61 `compute_src2dst` 实验记录

```current
task: 61
operator: compute_src2dst
batch: 5
validity: invalid_correctness
platform: completed(12900,7/8)
candidate_stage: e1
team_best_stage: -
sealed: no
next: 提交 e1 燧原 vendor（ids 取小端 lo 词，全 int32 scatter）；据逐芯结果迭代
updated: 2026-09-11
```

> 下方 S0 开发记录是 2026-09-10 快照；当前平台结果见 CURRENT 和文末提交记录。

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

## 2026-09-11 首次平台提交（2026-09-11T03:39:14.099324+08:00）

- `s0` / submission `12900` / daily_seq `6`，提交于 `2026-09-11T03:37:33+08:00`。每候选上传与正式 POST 各一次，无自动重试。
- nonce：`0349d4a4f3aa4521ef8ae8bdb8a1b0bb`；上传 URL SHA-256：`87fa9861195f0f4bcf841586f744a6bb159832bd9778a86de7cb9e1629e978fc`。
- 远端 ZIP 回读 verified，1198 bytes，SHA-256 `607bc6f0bbfedcfe9783a7dd2dc0af606c0ed717a64094a10aae58f33909ea8e`，与本地候选逐字节一致。
- 平台 `completed` / `invalid_correctness`；正确性通过 7/8，完成 8/8；未产生八芯均值；未入有效榜。
- 当次 status 额度：24/30，已用 6；observed_at `2026-09-11T03:39:14.099324+08:00`。排名查询时间 `2026-09-11T03:39:41.445754+08:00`。

| 芯片 | 状态 | 正确性 | speedup |
| --- | --- | --- | ---: |
| tianshu | completed | 通过 | 3.023 |
| muxi | completed | 通过 | 1.1614 |
| enflame | completed | 失败 | — |
| haiguang | completed | 通过 | 1.8188 |
| kunlunxin | completed | 通过 | 1.2898 |
| huawei | completed | 通过 | 1.6266 |
| card_a | completed | 通过 | 1.6388 |
| card_b | completed | 通过 | 1.782 |

燧原选择的文件为 `compute_src2dst.py`；case 0, 1, 2, 3, 4, 5, 6, 7, 8 均在 make_gcuir → Pipeline.run 报 `RuntimeError: Pipeline run failed: PassManager execution failed`。尚未执行到数值比较，触发构造未定位；相同错误文本不能证明三题同根因，也不证明平台基础设施故障。

完整失败详情：`artifacts/competition/batch5-submit-20260911/61-enflame-failure.json`，SHA-256 `a839689fc76a4d0157d90ba7651c74260df80ff4383a6f6d2a385d9a05ac0409`。

下一步：定位燧原 scatter 编译失败；先验证索引 lowering。

[提交结果证据](../data/batch5-submissions-20260911.json)，SHA-256 `63418b87e9249bef72751df2a1778c52e6d679a5d313ecf18d8ed64b96c175dd`；原始 status `artifacts/competition/batch5-submit-20260911/61-status-033914.json`，SHA-256 `a5efad83ff8f3604bbdd85e881b5a0647c052894d92794ad6d60f1e8b5b006bd`。

## 2026-09-11 E1 燧原 vendor 候选（lo 词 scatter）

- 根因定位：s0 的 TTIR 算子集是燧原通过的 kv_indices 的**严格子集**，
  排除算子种类问题；契约强制 reorder_ids 为 int64（argsort 输出），
  唯一特化即 i64 向量 load——与 clamp_position 仅 int64 case 失败互证。
  E1 vendor 在 wrapper 做小端 lo 词视图
  `reorder_ids.view(torch.int32)[0::2]`：reorder_ids 是 range(num_toks) 的
  置换且契约保证 `0 <= num_toks <= 2**31`，lo 词即精确值——**可证无损**，
  非盲目缩窄。kernel 全 int32：load 寻址形态与 clamp_position 燧原已过的
  int32 case 相同；store 索引为 load 值 × runtime stride（decode_attention
  经 load 的 int32 pages 寻址同型）。grid 封顶 24；不钉 num_warps。
- source commit：`238a41eff945782708aa61177272fc5b601b6a86`（generic 字节不变）。
- ZIP：`artifacts/competition/compute_src2dst/e1-238a41e/compute_src2dst.zip`；
  4304 bytes；成员 generic + `compute_src2dst_enflame.py`
  （SHA-256 `301cb0c4c63da9746227749f40dad3770265bec250b3863303e62fa3f5527364`）。
- ZIP SHA-256：`102067c7f0816a5a9793527d84667fbb5429a702f831aebb4820fe5bc38fc2b8`。
- 源码 SHA-256：generic `f59a3ffa8e3a463be40f7e70e8036b656a6df689899928689d073ba983215051`
  （与 s0 逐字节一致）；vendor `301cb0c4…`。
- 回执：`artifacts/competition/batch5-enflame-fix-20260911/compute_src2dst-verification.json`；
  SHA-256 `1955490c79ff0ba987a9080812a711fb6f850514afebee5241a8834d6286863a`。
- 日志：`artifacts/competition/batch5-enflame-fix-20260911/compute_src2dst-verification.log`；
  SHA-256 `d6648889ca074a1d812892fee480f6d9640260a3ae3558783ba61bc0cceee79f`。
- 完整 release（v2，`--proxy-vendor enflame`）：3 方法 / 0 fail/err/skip；
  generic 19 次 + enflame vendor 19 次真实 kernel launch；代理 benchmark
  vendor≈generic+20%（仅燧原使用 vendor）。
- 燧原目标 runtime 仍未验证（target-runtime-unverified）：GCU 编译门由平台裁决。
