# Task 59 `build_trtllm_mha_page_table` 实验记录

```current
task: 59
operator: build_trtllm_mha_page_table
batch: 5
validity: invalid_correctness
platform: completed(12896,7/8)
candidate_stage: e2
team_best_stage: -
sealed: no
next: 提交 e2 燧原 vendor（去 i64 向量 load/xori/select/trunci）；据逐芯结果迭代
updated: 2026-09-11
```

> 下方 S0 开发记录是 2026-09-10 快照；当前平台结果见 CURRENT 和文末提交记录。

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

## 2026-09-11 首次平台提交（2026-09-11T03:39:06.765926+08:00）

- `e1` / submission `12896` / daily_seq `3`，提交于 `2026-09-11T03:29:45+08:00`。每候选上传与正式 POST 各一次，无自动重试。
- nonce：`337996912a6d31a71419fb5bfb4b4701`；上传 URL SHA-256：`de88f9887a2bc01c2a7944cffa7bbf5df537d3bb5470a3fda12065d3fbff0533`。
- 远端 ZIP 回读 verified，2830 bytes，SHA-256 `432e1896094351992b41ce3dba289ec89780f200bbbb75100f31e18920b9fb9a`，与本地候选逐字节一致。
- 平台 `completed` / `invalid_correctness`；正确性通过 7/8，完成 8/8；未产生八芯均值；未入有效榜。
- 当次 status 额度：24/30，已用 6；observed_at `2026-09-11T03:39:06.765926+08:00`。排名查询时间 `2026-09-11T03:39:41.445754+08:00`。

| 芯片 | 状态 | 正确性 | speedup |
| --- | --- | --- | ---: |
| tianshu | completed | 通过 | 70.034 |
| muxi | completed | 通过 | 13.1595 |
| enflame | completed | 失败 | — |
| haiguang | completed | 通过 | 27.05625 |
| kunlunxin | completed | 通过 | 2.1595 |
| huawei | completed | 通过 | 11.03625 |
| card_a | completed | 通过 | 22.861 |
| card_b | completed | 通过 | 19.973 |

燧原选择的文件为 `build_trtllm_mha_page_table.py`；case 0, 1, 2, 3, 4, 5, 6, 7 均在 make_gcuir → Pipeline.run 报 `RuntimeError: Pipeline run failed: PassManager execution failed`。尚未执行到数值比较，触发构造未定位；相同错误文本不能证明三题同根因，也不证明平台基础设施故障。

完整失败详情：`artifacts/competition/batch5-submit-20260911/59-enflame-failure.json`，SHA-256 `e282727d3307c8722b0c0280d9ef51e379556102d5966942f61e20cdb56a08e2`。

下一步：定位燧原 GCU IR 编译失败；最小复现后开发 vendor 候选。

[提交结果证据](../data/batch5-submissions-20260911.json)，SHA-256 `63418b87e9249bef72751df2a1778c52e6d679a5d313ecf18d8ed64b96c175dd`；原始 status `artifacts/competition/batch5-submit-20260911/59-status-033906.json`，SHA-256 `a0f94d92c6db59dc52b9b183207144c1647d8d6fccfaa197de7d5b65aacf6061`。

## 2026-09-11 E2 燧原 vendor 候选（i64 向量 load 消除）

- 根因定位：对照同批燧原通过的 kv_indices/deepep_permute 做 TTIR 算子集差分。
  clamp_position 恰好只有 int64 case 编译失败、src2dst 契约强制 int64 ids，
  而 63 题面明确 req_to_token 为 int32——通过内核只做 **i64 标量 load**，
  从不做 i64 向量 load。E2 vendor 据此消除四个独占嫌疑构造：
  i64 向量 load（wrapper 小端 word 视图，int64 pool 转 int32 词寻址）、
  `~` 补码（改写为 `page >= n_pages` 显式比较）、
  `tl.where`（拆成两次掩码 store：计算值/复制旧值）、
  值域 trunci（WORD 视图替代 `.to(int32)`）。
  保留的构造全部有燧原平台实证：向量 `>>`/`&`（apply_token_bitmask 2.85x）、
  标量 load + `.to(tl.int64)`（decode_attention/kv_indices）、标量 i64 ceil-div
  （kv_indices cdiv）。grid 封顶 24（24-SIP 指南）；不钉 num_warps。
- 修复过程发现并改正一处掩码错误：初版 `active = page < n_pages` 丢失
  `valid &` 前缀，n_pages 超过列宽时越出列的 lane 会越界写坏下一行
  （bs=65537/columns=1 用例 4755 元素失配）；改为与 generic 一致的
  `valid & (page < n_pages)` 后全绿。
- source commit：`238a41eff945782708aa61177272fc5b601b6a86`（generic 字节不变）。
- ZIP：`artifacts/competition/build_trtllm_mha_page_table/e2-238a41e/build_trtllm_mha_page_table.zip`；
  8736 bytes；成员 generic + `build_trtllm_mha_page_table_enflame.py`
  （SHA-256 `d7fdda592e6f846f3ffc93e313abb5adbe1c3a2eb2f382a7caa664ceceb3d405`）。
- ZIP SHA-256：`8599269f88b2392ef6e34bf30b6d4214ccf32041c0b43c3e02452dc0e3455e03`。
- 源码 SHA-256：generic `92d3c574e999a0bfb718010aca583ef26aeeea5b05ff53a8962d636cbcade0a7`
  （与 e1 逐字节一致）；vendor `d7fdda59…`。
- 回执：`artifacts/competition/batch5-enflame-fix-20260911/build_trtllm_mha_page_table-verification.json`；
  SHA-256 `ebd0ef185a4ee48f6c5f876fc14099bdd9f74a02aa55a0ba52303cff466a5bd5`。
- 日志：`artifacts/competition/batch5-enflame-fix-20260911/build_trtllm_mha_page_table-verification.log`；
  SHA-256 `e05b972937d21abc6ba5fe652444733e104a8d33b0e2df277c019785f27b2fc3`。
- 完整 release（v2，`--proxy-vendor enflame`）：4 方法 / 0 fail/err/skip；
  generic 26 次 + enflame vendor 26 次真实 kernel launch；RTX 5070 Ti 代理范围。
- 燧原目标 runtime 仍未验证（target-runtime-unverified）：GCU 编译门由平台裁决。
- 已知风险与边界：①word 视图假设小端（全部平台芯为小端）；②slot 值按
  张量元素量级 < 2^31 取 lo 词（reference `.to(int32)` 亦为 int32 语义宽度），
  测试含 -4097 负 sentinel 通过；③若平台 59 用例 pool 实为 int32（则根因不在
  i64 load），E2 同时消除的 xori/select/trunci 仍覆盖其余嫌疑。
