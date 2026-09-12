# Task 59 `build_trtllm_mha_page_table` 实验记录

```current
task: 59
operator: build_trtllm_mha_page_table
batch: 5
validity: valid
platform: submitted(13360,e6,评测中;team best e4r 24.1284x)
candidate_stage: e6
team_best_stage: e4r
team_best_speedup: 24.1284375
sealed: no
next: e6（_ascend 掩码地址钳位，根因=昇腾 masked-lane 越界地址 507035 族，triton-ascend #16275/#1490 外部佐证）候选就绪；华为 8.68→≈20 即均值 ≈25.5 重夺第一（GuanghuLab 25.43）；e4r 守榜
updated: 2026-09-12
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

## 2026-09-11 E2 平台提交（2026-09-11T06:44:10+08:00）

- `e2` / daily_seq `7`。上传与正式 POST 各一次，无自动重试。
- nonce：`e9a2944ca8fc4fb02015173fa13ca5e8`。
- 外部佐证（调研）：FlagTree 燧原后端 `enable_i64: bool = False` 默认关闭，
  GCU300 gcuir 管线第一步 `gcu64-type-verifier` 对含 i64 的 IR 直接 PassManager
  失败；FlagGems PR #5345 对 int64 slot_mapping 的既定修复即 launcher 侧
  int64→int32 downcast。与本轮 TTIR 差分结论（i64 向量数据 load 为毒点）一致。
- 逐芯结果：见下方跟进记录。

## 2026-09-11 E2 平台终态（submission 12902）

- 7/8：天数 67.97675 / 沐曦 12.913 / 海光 27.3865 / 昆仑 2.16975 / 华为 9.72425 /
  A 22.7605 / B 20.05925（七芯读数与 E1 一致，generic 字节不变的水位复现）。
- **燧原 vendor 被选中（selected_file=`build_trtllm_mha_page_table_enflame.py`）
  并首次通过 make_gcuir 编译**——i64 向量数据 load 消除即解除编译阻断的假设
  得到平台实证。但数值失败：case 0 起断言 `82/128 (64.1%)` 元素失配
  （最大绝对差 63345），GCU 专属 lowering 数值问题，代理 NVIDIA 同字节全绿。
- 归因与下一步：E2 内核含两次同址掩码 store（active 计算值 / inactive 复制旧值）
  与 `slot >> SHIFT`；kv_indices（121x 通过）是单 gather + 单掩码 store 形态。
  E3 改为 wrapper `page_table.clone()` 预填（与 reference 完全一致的语义）+
  内核单 active store，形态与 kv_images 完全对齐。

## 2026-09-11 E3 提交（2026-09-11T07:03:25+08:00，daily_seq 11）

- source commit `10e38fe242b5dc35d773d62767209f92a199e3d6`；nonce `d096a0a75c019134a5092260605a12fa`。
- ZIP `e3-10e38fe`，8090 bytes，SHA-256 `0c1af160511ba9b7c5bc6c52371d04db41baa9e5bede9a7908ab4f74b95ed6bc`。
- 回执 `artifacts/competition/batch5-enflame-fix-20260911/release-r3/build_trtllm_mha_page_table/verification.json`，
  SHA-256 `880db0f4e2d327b6575c83ac954d27cde37d47376139231a6b667516308d5c88`；
  4 方法 0 失败，generic 26 + enflame 26 次 launch。

## 2026-09-11 E3 平台终态与 E3R 重掷（daily_seq 16）

- E3（daily_seq 11，07:03）：**燧原 27.81525x 首次通过**；天数 68.63925 /
  沐曦 12.9255 / 海光 26.8495 / 华为 10.25925 / A 21.7895 / B 20.03675 全过。
  昆仑芯在评测器自身 XMLIR 栈崩溃（"No test results found (empty report)"，
  error code -299）——generic 字节与 2.17x 通过轮完全一致，判**崩溃族**非代码回归。
- 根因链沉淀（平台四轮实证）：E2 双同址掩码 store 形态在 GCU 编译通过但
  数值 64% 错；E3 改 wrapper `page_table.clone()` 预填 + 单 gather 单掩码 store
  （kv_indices 121x 已证形态）后数值全对且 27.82x（超榜一 Nectar 26.64x）。
- E3R 载体（daily_seq 16，07:30:27）：内核语义与 E3 相同（注释载体
  `cbd35d8`），重掷昆仑窗口；回执
  `artifacts/competition/batch5-enflame-fix-20260911/release-r5/build_trtllm_mha_page_table/verification.json`
  （SHA-256 `964308bcec6df32e1a452231a9ef01efcad0f08d34f1a19a6675e4d09b07972b`，
  4 方法 0 失败）；ZIP `e3r-cbd35d8`，8442 bytes，
  SHA-256 `99436cb19e8d41926d80847d1c101f8ea42af08c819f720a981812c5fd7bfc49`。
  昆仑重掷计数 1/2。

## 2026-09-11 E3R 平台终态：8/8 VALID（daily_seq 16，07:30:27）

- **八芯全过，首次有效提交**：天数 68.40825 / 沐曦 12.965 / 燧原 26.83925 /
  海光 24.8525 / 昆仑 2.159 / 华为 9.94025 / A 22.132 / B 20.153。
  **平均 23.4311875x**（187.4495/8）。昆仑 XMLIR 崩溃族重掷 1 次即恢复。
- 榜单对标：榜首 Nectar 23.90034375（燧原 26.639）、次席 c2flow 19.5053；
  我方 23.43x 逼近榜首（差 0.47，约 2%），燧原单芯 26.84x **超过榜首同芯读数**。
- 攻坚路径复盘（五轮平台实证，燧原 GCU 规则集已沉淀入 skill 硬事实表）：
  generic i64 向量 load 编译死 → E2 消 i64 后双同址 store 数值错 →
  E3 clone 预填 + 单 gather 单 store 全对 → E3R 昆仑崩溃重掷。

## 2026-09-11 深度优化轮：E4/E4R —— 8/8 VALID 24.1284x 登顶

- 情报：逐芯榜显示差距全在海光（我 24.85 vs 榜首 Nectar 32.06），填平需
  海光耗时 −13.1%。Codex 咨询（gpt-6-astra/ultra）+ T17 先例（hygon
  num_warps 4→2 +3.34%）→ 新增 `_hygon` vendor：直接 2D grid
  （row=pid0, tile=pid1）去除 div/mod 与 grid-stride + num_warps=2 +
  双 stride 对（old 读/out 写分离）。generic 字节不动。
- E4（daily_seq 23，08:58）：**海光 24.85→29.6378（+19%）**，燧原 27.30 保持；
  昆仑 XMLIR 崩溃族（generic 字节与 2.16x 通过轮一致）。
  回执 `batch5-deepopt-20260911/release/build_trtllm_mha_page_table/verification.json`
  （SHA-256 `1c48f8e09c5544edfe38589ddf743c9406fc5e6de5755cbb88451e570937b8e4`）。
- E4R（daily_seq 29，10:14，载体 `dc3c2b5`）：**八芯全过**：
  天数 68.367 / 沐曦 12.8375 / 燧原 28.0875 / 海光 30.207 / 昆仑 2.1485 /
  华为 8.677 / A 22.46875 / B 20.23425。
  **平均 24.1284375x = 新 team best，超过当时榜首 Nectar 23.9003 —— 升第一**。
  ZIP `e4r-dc3c2b5`，12759 bytes，SHA-256
  `525b5b45e9eb0c6c131ade74933e7a3623a8cb30a2673995e0700ecf056bd8af`；
  回执 `release-r2/…`（SHA-256 `338331166fab30515820ad6f992feafe32758e4d58b9876e22e1ff4af935f5c8`，
  generic 26 + enflame 26 + hygon 26 次 launch）。

## 2026-09-11 榜单反击：E5 `_ascend` vendor 候选就绪（待下一额度窗口）

- 榜单复核（~10:40）：**被 GuanghuLab 反超**（其 25.4259 vs 我 24.1284，
  Nectar 23.90 降至第三）。逐芯差分：我方领先燧原 +4.80 / A +2.81 /
  海光 +0.45 / 昆仑 +0.29 / B +0.50；落后 **华为 -16.11（我 8.677 vs 其
  24.788）**、天数 -3.03、沐曦 -0.08。华为单项为均值差 -1.30 的 1.55 倍。
- E5（commit `169dd6e`）：新增 `_ascend` vendor——镜像 hygon 轮平台已证
  的直接 2D（rows, page blocks）网格（无扁平 task/div/mod/grid-stride、
  标量行元数据、单 gather + 单掩码 store；该形态海光 +22%），不钉
  num_warps，BLOCK=256（1KB int32 tile，远低于昇腾 UB 预算）。
  华为 8.68→≈20 即均值 +1.4 → ≈25.5 重登第一（GuanghuLab 25.43）。
- 代理矩阵 4 方法 0 失败；回执 `release-next/build_trtllm_mha_page_table/verification.json`
  （SHA-256 `64d0382f9740174b23e8f180fe25416386e72c04b4f106696817347f2759d3b4`，
  generic 26 + ascend 26 + enflame 26 + hygon 26 launch）；
  ZIP `e5-169dd6e`，16814 bytes（4 成员），SHA-256
  `f9eab06b945e8d64a4e9598c81ec0fee9e32382e95a5bece07125c4354457ec8`。
- 状态：候选就绪未提交（当日额度 30/30 用尽）；发射序调整为第四发
  （前序：62-e4、61-e5r、63-e3）。

## 2026-09-12 E5 平台结果（submission 13232，daily_seq 4）

- 7/8（华为失败）。其余七芯健康且 vendor 全部按预期被选中：
  天数 69.6605、沐曦 12.95、燧原 27.4923、海光 30.0975、昆仑 2.1808、
  A 22.4938、B 20.1123（对照 e4r：天数 68.4/海光 30.2/A 22.8/B 20.6，
  基本持平）。
- 华为失败详情（raw_result）：`selected_file=build_trtllm_mha_page_table_ascend.py`
  被选中、执行完成（25153ms）后，比较阶段报
  `AclrtSynchronizeStreamWithTimeout(copy_stream) error 507035` 与
  `aclnnInplaceCopy inner error`。属**昇腾 aclnn 原生库层错误家族**
  （T46 `aclnnCat` 同例，异步栈不可靠）：无法区分 vendor 触发 NPU 运行时
  异常 vs 平台间歇。按硬事实表"单发探针止损，不据此改结构"处置。
- 处置：e4r team best 24.1284x 守榜不受影响（本次 validity=invalid，不入榜）。
  昇腾轴后续两条路：① 崩溃族注释载体重掷 1 发（需用户当次明示授权，
  新 commit/新 ZIP + 自有 release 回执）；② 先取得昇腾侧真机验证再重投。
  在两者其一之前不自动重试。
- 额度：发后 26/30（observed_at 2026-09-12T01:0x）。

## 2026-09-12 E6：`_ascend` vendor 掩码地址钳位（候选就绪后提交）

- 根因假设（联网核实的外部证据）：e5 华为 507035（aclnnInplaceCopy 流同步
  超时）属 triton-ascend 已知缺陷族——issue #16275（`tl.where` 双臂真实访存）
  与 PR #1490（masked atomic 剥 mask）同报 507035（MTE illegal GM address）；
  e5 vendor 的三处访存对 `page ≥ columns` 的 lane 均用未钳位地址（page_table
  旧值读、pool gather、out store），masked lane 地址被真实求值即越界。
- E6 改动（单变量）：全部访址改走 `page_rd = where(page < columns, page, 0)`；
  in-range lane 语义逐字节不变，越界尾 lane 从不 store。generic/enflame/hygon
  字节不动。
- source commit：`dcff187106855312d7de87e3112322ddaf537ff5`。
- ZIP：`artifacts/competition/build_trtllm_mha_page_table/e6-dcff187/build_trtllm_mha_page_table.zip`，
  SHA-256 `f5c675be674f6e2c16fd7ff0a2598dc8426a92acd66b05ae3ae7b4f794af0ffd`；
  4 成员：generic `92d3c574…`、`_ascend` `c7112f67…`（新）、`_enflame`
  `aaba86ae…`、`_hygon` `cdd9e2cb…`（后三者中仅 ascend 变化）。
- release 回执（v2，绑定 dcff187，proxy-vendor×3）：
  `artifacts/competition/batch5-e6-validate-20260912/build_trtllm_mha_page_table/verification.json`，
  SHA-256 `1b857fbeb85e332cb0c7cffdb7454550477924b1c5e1fbea89fa060048b07c51`；
  日志 SHA-256 `aa2970bc735dfe1e224a4afc94813eade813e0764aab9d40c9ae74682cd7c2f6`；
  4 方法 0 失败，generic 26 + ascend 26 + enflame 26 + hygon 26 次 launch。
- 昇腾真机仍 target-runtime-unverified（代理仅证数学/JIT）；裁决权在平台。
  预期：华为 8.68→≈20（e5 形态目标不变），七芯读数应与 e4r/e5 持平。

## 2026-09-12 E6 平台提交（submission 13360，daily_seq ~14）

- 上传与正式 POST 各一次，无自动重试；state submitted，13:12:21 入队评测。
- 额度：发后 16/30（observed_at 13:12:5x）。
- 裁决点：华为是否解除 507035（钳位假设验证）；其余七芯读数与 e4r/e5
  水位对比（vendor 字节仅 ascend 变化）。
