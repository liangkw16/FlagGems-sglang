# Task 60 `clamp_position` 实验记录

```current
task: 60
operator: clamp_position
batch: 5
validity: invalid_correctness
platform: completed(12898,7/8)
candidate_stage: e2
team_best_stage: -
sealed: no
next: 等待 E2 逐芯回调（整型 select 消除裁决）
updated: 2026-09-11
```

> 下方 S0 开发记录是 2026-09-10 快照；当前平台结果见 CURRENT 和文末提交记录。

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

## 2026-09-11 一轮优化后的待提交候选

- 候选 `s0`；本轮审查后保留既有 S0。首次提交前候选就绪。
- source commit：`4836fe0378d6290cbcaa91958693e13ed1acdf3e`；verification commit：`4836fe0378d6290cbcaa91958693e13ed1acdf3e`。
- ZIP：`/Users/bytedance/ccc/flagos/artifacts/competition/clamp_position/s0-4836fe0/clamp_position.zip`；1085 bytes；成员 `clamp_position.py`。
- ZIP SHA-256：`506af23ed82335d1a93965436545c7c4bd646b09b5f589c642fcfadecc1e7209`。
- 源码 SHA-256：`fc637bc6d0bccd8a38d11100aa3589feda86568aa4fa8383aca7472001c9ed2b`。
- 测试 SHA-256：`fcfe27bc9c551018822dd8649ffa03ae8077f810d77596e078e94d270e3d8a15`。
- 回执：`artifacts/competition/batch5-development-20260910/release/clamp_position/verification.json`；SHA-256 `ec3aa53405854fb342081f3dc9a60f1ab2b9d81fd2f8ed30444a782cf59adb96`。
- 日志：`artifacts/competition/batch5-development-20260910/release/clamp_position/verification.log`；SHA-256 `8f9cc4892f5224d878a14add6b989ebfd03c73bef69d5bdcd79e858ab50dd5eb`。
- 完整 release：3 个测试方法 / 45 条记录 / 22 次 kernel launch，全通过；NVIDIA 代理范围。
- 优化和复现见[本轮报告](../optimization-batch5-20260911.md)；[完整证据](../data/batch5-optimization-20260911.json)，SHA-256 `87a560dd86ab7e9dfb9607f8f05a007060d48b3aab876b165bfc0fec591f77db`。

## 2026-09-11 首次平台提交（2026-09-11T03:39:10.201484+08:00）

- `s0` / submission `12898` / daily_seq `5`，提交于 `2026-09-11T03:34:56+08:00`。每候选上传与正式 POST 各一次，无自动重试。
- nonce：`1680e98c76e2acab2eb754d0c4d564e1`；上传 URL SHA-256：`a8070d2094812cbe8226aedf736656b1c93a0b0ab632d9040271d3390ab19345`。
- 远端 ZIP 回读 verified，1085 bytes，SHA-256 `506af23ed82335d1a93965436545c7c4bd646b09b5f589c642fcfadecc1e7209`，与本地候选逐字节一致。
- 平台 `completed` / `invalid_correctness`；正确性通过 7/8，完成 8/8；未产生八芯均值；未入有效榜。
- 当次 status 额度：24/30，已用 6；observed_at `2026-09-11T03:39:10.201484+08:00`。排名查询时间 `2026-09-11T03:39:41.445754+08:00`。

| 芯片 | 状态 | 正确性 | speedup |
| --- | --- | --- | ---: |
| tianshu | completed | 通过 | 1.73733333 |
| muxi | completed | 通过 | 1.1535 |
| enflame | completed | 失败 | — |
| haiguang | completed | 通过 | 1.56733333 |
| kunlunxin | completed | 通过 | 0.9185 |
| huawei | completed | 通过 | 0.4135 |
| card_a | completed | 通过 | 1.38183333 |
| card_b | completed | 通过 | 1.44933333 |

燧原选择的文件为 `clamp_position.py`；case 3 均在 make_gcuir → Pipeline.run 报 `RuntimeError: Pipeline run failed: PassManager execution failed`。尚未执行到数值比较，触发构造未定位；相同错误文本不能证明三题同根因，也不证明平台基础设施故障。

完整失败详情：`artifacts/competition/batch5-submit-20260911/60-enflame-failure.json`，SHA-256 `2a647cce72ff2f293da0adca92a6a5567fa3b37b4dcad45ebfbe3e4dc52d3c42`。

下一步：定位燧原 case 3 编译失败；核对 int64 路径，不盲目缩窄数据。

[提交结果证据](../data/batch5-submissions-20260911.json)，SHA-256 `63418b87e9249bef72751df2a1778c52e6d679a5d313ecf18d8ed64b96c175dd`；原始 status `artifacts/competition/batch5-submit-20260911/60-status-033910.json`，SHA-256 `e249722ab8c475509882b7146132045ea0d03ac294022d2fdf7e0a429d3982c8`。

## 2026-09-11 E1 燧原 vendor 候选（int64 换 lo/hi 词算法）

- 根因定位：s0 恰好只有 int64 case（case 3）编译失败，int32 case 全过；
  i32/i64 两份 TTIR 的算子集完全相同（subi/maxsi 都在）⇒ 毒点与元素类型
  绑定。同批燧原通过的内核（kv_indices req_to_token 契约 int32）只做
  i64 标量 load，从无 i64 向量 load。E1 vendor 把 int64 数据路径整体换成
  **小端 (lo, hi) int32 词对算法**（wrapper 零拷贝 `view(torch.int32)`，
  非数据缩窄——64 位补码代数逐位保持）：
  `lo1 = lo - 1`；借位 `hi1 = where(lo == 0, hi - 1, hi)`（i32 回绕下
  (hi1, lo1) 恰为 v-1 的真补码表示，hi1 符号位即 v-1 符号位）；
  负号时两组词清零（两次掩码 store，`&` 已证构造）。
  纯 Python 仿真 22 个边界值 + 20000 随机值全对（min_int64 溢出包绕与
  torch 语义一致，远端 torch 测试实证）。int32 输入路径保留与 generic
  逐字节相同的内核（燧原已证）。grid 封顶 24；不钉 num_warps。
- source commit：`238a41eff945782708aa61177272fc5b601b6a86`（generic 字节不变）。
- ZIP：`artifacts/competition/clamp_position/e1-238a41e/clamp_position.zip`；
  5716 bytes；成员 generic + `clamp_position_enflame.py`
  （SHA-256 `e221b7efe1be6e04a90d13ad3d81a26c4d05a4c688b6ce5397893d6c726ba35d`）。
- ZIP SHA-256：`5d96abed5e02a2e24de1a11741718428be7bb93d745c7bdbb97a0fc5bd861ada`。
- 源码 SHA-256：generic `fc637bc6d0bccd8a38d11100aa3589feda86568aa4fa8383aca7472001c9ed2b`
  （与 s0 逐字节一致）；vendor `e221b7ef…`。
- 回执：`artifacts/competition/batch5-enflame-fix-20260911/clamp_position-verification.json`；
  SHA-256 `f90d2fb533210231d5249dd1cec8db200fdc3e8b2786a60d3021fce706fc12d3`。
- 日志：`artifacts/competition/batch5-enflame-fix-20260911/clamp_position-verification.log`；
  SHA-256 `173e0790531481508427122956009d9ca0d3ccf383089184a9fd65f15962523a`。
- 完整 release（v2，`--proxy-vendor enflame`）：3 方法（含 int64 极值/stride/
  尾块矩阵）/ 0 fail/err/skip；generic 22 次 + enflame vendor 22 次真实
  kernel launch；代理 benchmark vendor≈generic+10%（仅燧原使用 vendor）。
- 燧原目标 runtime 仍未验证（target-runtime-unverified）：GCU 编译门由平台裁决。
- 残余风险：vendor 内单个 `tl.where(lo == 0, hi-1, hi)` 为 i32 select
  （燧原实证仅有 fp select——apply_token_bitmask 2.85x）；若平台仍编译失败，
  下一候选把它替换为 `.to(tl.int32)` 布尔转换或算术消除，单独隔离该变量。

## 2026-09-11 E1 平台提交（2026-09-11T06:46:34+08:00）

- `e1` / daily_seq `8`。nonce：`d23f23520942e9558a4fa53cf7414f0d`。
- 外部佐证（调研）：FlagTree 燧原编译器 `enable_i64` 默认 False +
  `gcu64-type-verifier`（GCU300）与 i64 向量 load 毒点假说吻合；
  FlagGems PR #5345 的 int64→int32 launcher downcast 是社区既定模式。
- 逐芯结果：见下方跟进记录。

## 2026-09-11 E1 平台终态（submission 12903）

- 6 芯已过（等天数回调）：沐曦 1.14983333 / 海光 1.56216667 / 昆仑 0.90983333 /
  华为 0.20566667 / A 1.4585 / B 1.436；燧原 vendor 被选中但**仍编译失败**
  （case 3，vendor 第 99 行 launch）。
- 与同日 59 题 vendor（编译成功）的构造差分：E1 clamp vendor 独有 **整型
  `tl.where(lo == 0, hi-1, hi)`（i32 select）**——燧原语料中所有 tl.where
  通过案例均为 fp 数据（apply_token_bitmask 2.85x），整型 select 无一通过
  先例（draft_topk1 七结构全败亦含整型 where）。E2 消除 select：wrapper 零预填
  （clamp(min=0) 的零分支物化），内核 6 次互斥掩码 store，正判据纯 cmpi/andi
  （`hi>=0 & lo!=0` 与借位档 `hi>=1 & lo==0`、min_int64 包绕档
  `hi<-2147483647 & lo==0`）；50k 边界+随机值仿真 0 失配。

## 2026-09-11 E2 提交（2026-09-11T07:05:54+08:00，daily_seq 12）

- source commit `10e38fe242b5dc35d773d62767209f92a199e3d6`；nonce `d2527340ae6254d5c8504faf79541fe2`。
- ZIP `e2-10e38fe`，5883 bytes，SHA-256 `503ea8473f5fa1b10f62c64ce803c0325580b7f86529ee5da2722ca97114c527`。
- 回执 `artifacts/competition/batch5-enflame-fix-20260911/release-r3/clamp_position/verification.json`，
  SHA-256 `c4acb56402bf7e061b2beadd83a2f03c8769f66da1de21b7305c57a3505a3f58`；
  3 方法 0 失败，generic 22 + enflame 22 次 launch。
