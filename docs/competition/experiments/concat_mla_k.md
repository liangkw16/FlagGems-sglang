# Task 62 `concat_mla_k` 实验记录


## 2026-09-14 E6：燧原 concat 配方 vendor（候选就绪待发射）

- concat_mla_k 正是配方的 streaming elementwise 形态（192 元素行
  两 load 一 store）：BLOCK 覆盖行宽，行轴封 24 步进。generic 与
  昆仑两段式 vendor 字节不动。
- source commit：`9f5da6ec34987c72dd782dd377f87244c23556fe`；ZIP `e6-9f5da6e`，
  SHA-256 `90cef51ba50f4d8abb599f667c6014e1321e2c8fbb814b500f92da54e8d17b29`；
  release 回执 SHA-256
  `4e2f1f343eb003d0590d007725da0c07762044527d9895d0212b4fea7b6ad5bf`。

## 2026-09-14 E6 修订版（Codex 审查后，候选就绪待发射）

- 审查发现草稿的逐 head 行形态把 rope 读放大 16 倍（generic BH=16
  共享一次读）——修订为 generic 结构逐字节保留 + 唯一改动 task 轴
  封顶 24（配方真正贡献的单变量）。
- source commit：`220aa32d18a1c3a4aca829b09b79e2e906471e43`；ZIP `e6-220aa32`，
  SHA-256 `3dc76ba8cf0478ceb194fb9d400fb53a8006df2e7f3a1eccdff950ac457cb9b4`；
  release 回执前缀 `48687d55`；5 方法 0 失败。

## 2026-09-14 E6 平台终态：8/8 valid 1.1644x 新 team best

- 燧原 0.233→0.2744（+19%，封顶兑现）；天数 +4%/A +4%；华为 -28%
  （窗口）。均值 1.1644 > e5 1.1483 新 TB（+1.4%）。
- 榜首 EvokeAgent 1.814 差 0.65；燧原榜首 1.23 vs 我 0.27——该题
  燧原差距 4.5x，配方远不够，需要海光/天数多轴。

```current
task: 62
operator: concat_mla_k
batch: 5
validity: valid
platform: completed(14847,e10,8/8,1.1908x;燧原BH4反降-47%;TB e9 1.2458x)
candidate_stage: e10
team_best_stage: e9
team_best_speedup: 1.24575
sealed: no
next: e10 燧原BH4 0.15(-47%)跨芯平移证伪;TB e9 1.246守;收官仅守榜
updated: 2026-09-14
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

## 2026-09-11 昆仑回调终态与 S0R 重掷载体

- 12897 昆仑芯回调落地：`completed` / passed=False，raw_result errors 仅一条
  「服务线程卡死自动恢复，请重新提交」，failed_cases 为 0——**昆仑评测器崩溃族**
  指纹（T58 e6 → e6r 同字节 3 小时后通过的同型事件），非代码回归。
  12897 终态 7/8，invalid_correctness。其余七芯读数：天数 2.6114 / 沐曦 0.9836 /
  燧原 0.109（贴 0.1 门槛）/ 海光 3.1072 / 华为 0.2188 / A 1.7242 / B 1.8166。
- s0r 载体：内核语义与 s0（7b53fde）完全相同，仅头部注释变化以铸造新字节。
  source commit `3908f0cd05deb2218cc8d8d3a9090774144fa5fb`。
- ZIP：`artifacts/competition/concat_mla_k/s0r-3908f0c/concat_mla_k.zip`；
  2704 bytes；成员仅 `concat_mla_k.py`（SHA-256
  `834c0fc37e476629a72446cdd2bede8b94a3b6fe344ae237c2167de9bc358047`）。
- ZIP SHA-256：`a7d5f3ab56c9410f4dcac864d6c71ccdf1b23e4dc593d5bc5208c13ff5a4b764`。
- 回执：`artifacts/competition/batch5-enflame-fix-20260911/concat_mla_k-s0r-verification.json`；
  SHA-256 `2d141e4a32af178d40a80f3862d53d92abf8cc1ded21fb61100d45e46dc4f856`。
- 完整 release（v2）：5 方法 / 0 fail/err/skip / 28 次 kernel launch，
  绑定 3908f0c；重掷次数 1/2（崩溃族协议上限内）。
- 已知风险：燧原 0.109x 贴门槛，重掷窗口读数若下滑跌破 0.1 该发无效
  （当前 12897 本就无效，无净损失）；沐曦 0.9836、华为 0.2188 同样偏弱，
  后续优化方向为 wrapper/launch 开销与 tile。

## 2026-09-11 S0R 平台提交（2026-09-11T06:51:26+08:00）

- `s0r` / daily_seq `10`。nonce：`772eb61868a69bf91f3e452d737a1e75`。
- 逐芯结果：见下方跟进记录。

## 2026-09-11 S0R 终态与 S0R2 终投（daily_seq 20）

- S0R（daily_seq 10）：昆仑芯再次崩溃族（同「服务线程卡死自动恢复」错误，
  七芯第二次全过且读数稳定：燧原 0.11 / 华为 0.1134 / 沐曦 0.9906）。
- S0R2（daily_seq 20，~08:0x，commit `bb6faed`）：崩溃族重掷 2/2（最后一次）。
  回执 `release-r7/concat_mla_k/verification.json`（SHA-256
  `b9d1e831db48f602dcd88637b68939acc3b5d3935dc7b3f78bd856e1bf371b69`）；
  ZIP `s0r2-bb6faed`，2745 bytes，SHA-256
  `eeea677c98ca2c400a46c46b631fdd047372ca35c8239a430a17d3f9199094ac`。
- 弱芯观察：华为 0.2188→0.1134 水位波动大；燧原 0.109-0.11 贴门槛；
  后续优化方向为 launch/wrapper 开销（task 语义为纯内存单趟拷贝）。

## 2026-09-11 S0R2 观察中状态

- 昆仑回调仍在等待（提交 08:08:25）。注意华为读数 0.0932 已跌破 0.1 门槛
  （0.2188 → 0.1134 → 0.0932 三连降），即使昆仑恢复，本发大概率
  invalid_threshold；华为窗口回暖后需要真正优化轮（wrapper/launch 开销轴）
  而非重掷。崩溃族重掷额度已用尽（2/2）。

## 2026-09-11 深度优化轮：E1（BH16）/E2/E3（昆仑 vendor）——崩溃解除但数值未过

- E1（daily_seq 24，09:01）：generic BH 4→16 + 形状参数 do_not_specialize
  （昆仑/海光重编译风暴保险）。昆仑第 4 次崩溃族；其余读数待 E2 复测。
- E2（daily_seq 28，10:11，commit `0abaf28`）：新增 `_kunlunxin` vendor
  （每 program 一行的 flat 形态 + do_not_specialize）。**昆仑 XMLIR 崩溃
  首次解除（vendor 编译并运行）**，但数值 99.2% 垃圾（3.17e38 量级）。
  其余七芯大幅改善：**燧原 0.108→0.2328（+115%）**、华为 0.1594（回门槛上）、
  沐曦 1.022、天数 2.2892、海光 1.9216。代理曾抓出行寻址真 bug
  （`row64*os0` 误用 token stride，应为 head-dim stride），修复后全矩阵过。
- E3（daily_seq 30，10:26，commit `312f441`）：掩码 lane 负偏移钳制——
  昆仑仍 99% 垃圾且**垃圾值与 E2 逐位相同**（3.176854909925949e+38
  @(0,81,146)）⇒ 负载根本未读到输入数据。头号嫌疑：**XMLIR 对
  do_not_specialize 多参数的绑定/特化路径有错**（明日首查：去掉
  do_not_specialize 对照）。七芯读数保持（燧原 0.2314 / 华为 0.1594）。
- 回执：E2 `release-r2/concat_mla_k/…`（SHA-256
  `6359b8843358eba43c45c592b11b218520f02366a81507da26ef29d71d8dae5b`）；
  E3 `release-r3/…`（SHA-256
  `fa021c4e01f53cc17ddf3bc7f25e29df656ef13aec8652a35a7c018d46237666`）。
- 若昆仑修复：预期 8/8，均值 ≈ (2.29+1.02+0.23+1.92+K~0.9+0.16+1.61+1.65)/8 ≈ 1.22。

## 2026-09-11 E4 候选就绪（待下一额度窗口）

- 单变量对照 E3：**去掉 `do_not_specialize`**（E2/E3 平台昆仑读数逐位相同
  垃圾 ⇒ 唯一共享嫌疑为 XMLIR 非特化多参数绑定；重编译风暴是缓存代价非
  正确性问题）。代理全矩阵通过。
- source commit `95e529200a163575d060284efc01cc76d419a9ee`；ZIP `e4-95e5292`，
  7203 bytes，SHA-256 `545c629ec2f8d6556d5720b2d2de196b89ba1b231112691951cbbe2b42430e41`。
- 回执 `release-next/concat_mla_k/verification.json`（SHA-256
  `4c3f764767a735a794d88211161401ec2ffa184e6fb3860f693857fd477565f2`，
  5 方法 0 失败，generic 28 + kunlun vendor 28 launch）。
- 状态：候选就绪未提交（当日额度 30/30 用尽）。

## 2026-09-11 R2 待开发候选：B1/B2 弱芯轴（预注册门，未开发）

- 问题定性：T62 的**门槛风险高于均值风险**——燧原 0.2314 / 华为 0.1594 距
  0.1 门槛只有 1.6~2.3 倍，历史已出现华为 0.0932 跌破。昆仑 E4 裁决仍是
  一切晋级的前置。
- B1（有他题同芯平台先例）：`_concat_mla_k[(min(tasks, 65535),)]` 改为按
  物理资源封顶（燧原 24；华为按 vectorcore）。循环体本来就是 grid-stride，
  只改 cap 数值，不是重构。
- B2（假设）：生产 shape H=128 / N=128 / R=64 全为 2 幂，`n < nd`、
  `r < rd`、`h < heads` 三个谓词在窄守卫
  （`nd == 128 and rd == 64 and heads % BH == 0`）下走 constexpr 专化变体，
  让 2D store 恢复向量化；generic 路径保持运行时标量，重编译变体数固定为 2。
- 反向风险须在同一轮里量：`1c0381c` 把 BH 4→16 使 per-token program 数降到
  四分之一，代价是 rope 行按组重读；对燧原/华为是**增访存**，可能与目标相反，
  必须与 B1 分开做单变量，不要打包成一次提交。
- 预注册门：华为与燧原中位收益 **≥1.15x**，且其余六芯无任何一芯回退 >5%；
  完整正确性通过；spill/shared memory 为 0。
- 止损：若华为与燧原均 <1.05x，关闭该轴。
- 证据等级：B1 为他题同芯平台实证，B2 为**假设**。详见
  [r2 §3](../optimization-batch5-r2-20260911.md)。

## 2026-09-12 E4 平台结果（submission 13215，daily_seq 1）

- 7/8（昆仑失败），燧原 0.2306 / 华为 0.1796 仍贴近门槛；其余读数
  天数 2.287、沐曦 1.0258、海光 1.9032、A 1.6414、B 1.6284。
- 昆仑失败详情（raw_result）：`test_concat_mla_k[0]` 99.0% 元素失配，
  最大绝对差 3.18e38（bf16 近 max，未初始化读形态）；`[1]` 98.7% / 3.36e38。
  与 E2/E3 垃圾同指纹 ⇒ **去 `do_not_specialize` 对照证伪**：非特化多参数
  绑定不是根因。
- 结论：昆仑 vendor（flat 行形态）在 XMLIR 上第 3 次同指纹数值垃圾，
  按"两次同指纹止损"该轴停止重掷；下一步是根因分析（垃圾幅值指向
  未初始化/越界读——排查 vendor 的行寻址与 indptr 语义在 XMLIR 的
  lowering，或对照 generic 字节在昆仑跑 E1 前的原始读数）。
- 额度：发前 30/30，发后 29/30（observed_at 2026-09-12T00:5x）。

## 2026-09-13 E5：两段式 1D 昆仑 vendor（候选就绪后提交）

- 根因（调研实证）：四轮同指纹垃圾的共享配方 = 单 padded 256 宽
  range 盖非 2 幂行宽 192（含死区）+ 双 store 共享基址 + 逐 head 的
  rope 重读（stride-0 广播模式）——FlagTree #1147 OffsetAnalysis 误判
  触发配方 + FlagGems 昆仑"非 2 幂宽静默 miscompile"注记双重命中。
- E5：官方 `concat_and_cache_mla` 结构——两段完全独立 1D 循环
  （BLOCK_N=128、BLOCK_R=64 各自 2 幂），零共享基址/零死区/零 kernel
  内广播。
- source commit：`adfca1c670d449f208052ad2da3cfb7b29eb4f3b`；ZIP `e5-adfca1c`，
  SHA-256 `d295c7930808d05542d9f0c29bed5639b6f356492b3bb9ab21c85805b8a8c6c5`；
  release 回执 SHA-256 `6532be58d081828982d8c99144c44930be0c79f3f3a49051b2cbed5ace3188b7`；
  5 方法 0 失败，generic 28 + kunlun 28 launch。
- submission 13789（09-13 09:2x）；裁决点=昆仑四轮垃圾是否解除
  （若过即 8/8：七芯水位 2.29/1.02/0.23/1.92/0.16/1.64/1.63）。

## 2026-09-13 E5 平台终态：8/8 VALID 1.1483x 新 team best（submission 13789）

- **八芯全过，四轮同指纹垃圾收官**：天数 2.2320 / 沐曦 1.0156 /
  燧原 0.2330 / 海光 2.0290 / **昆仑 0.1840（两段式 vendor 被选中并
  通过——单 padded range+共享基址+stride-0 广播 = 垃圾触发配方的
  假设平台证实）** / 华为 0.2234 / A 1.6204 / B 1.6490。
  均值 1.1483 = 首 team best。
- 沉淀（昆仑垃圾黄金法则）：**两段独立 1D 循环 + 各自 2 幂块**，
  禁单 padded range 盖非 2 幂宽、禁双 store 共享基址、禁 kernel 内
  stride-0 广播读。
- 后续轴：燧原 0.233/华为 0.223 贴门槛（B1 grid 封顶/B2 constexpr
  去 mask 早已注册，昆仑定位后再排）；榜首 EvokeAgent 1.814。

## 2026-09-14 E7 候选就绪：海光 BH 16→4 隔离探针（Codex v3 队列槽 1，待发射）

- 依据：S0 BH=4 时海光读 3.11（历史最佳）；BH=16+dns 合并改动后海光
  降至 ~2.0，今榜单 3 队 3.3-4.03。BH 非已证根因（当时与 dns 混杂），
  本发是隔离实验：新建 _hygon vendor，与 generic 逐字节一致仅 BH=4、
  groups=cdiv(heads,4)；generic/其它芯字节不动。
- source / verification commit：`b60a46a…`；ZIP `e7-b60a46a`，
  4 members（generic+enflame+hygon+kunlunxin），SHA-256
  `e3eb9db890114178b8b2997e2076f4f93fb1b14e0173e95e359714362bacfae6`。
- release 回执 `batch5-submit-20260914/concat_mla_k/verification.json`
  SHA-256 `07a3522b1529441217cc4c6532ae4a55b01ddef1a2922973b63c83294505376b`
  （--proxy-vendor hygon：vendor 28 launch 0F0E0S）。
- 预注册晋级门：**海光 ≥ 2.60 且均值 ≥ 1.2144**；未过则 BH 轴停。

## 2026-09-14 E7 平台终态：8/8 VALID 1.2201x 新TB（submission 14686）

- **海光 vendor 被选中、passed 2.545**（BH16 的 2.04 → +25%）——门
  （≥2.60）差 0.055 未过，BH 轴按预注册停（不再试 BH=2 等梯度）。
  均值门（≥1.2144）过：1.2201 > TB e6 1.1644，新 TB = e7。
- 其余芯窗口内（天数 2.28/2.32、沐曦 1.03/1.02、燧原 0.28/0.27、
  昆仑 0.18/0.18、华为 0.19/0.16、A 1.63/1.69、B 1.64/1.64）。
- 距榜首 Evoke 1.8141 还差 0.594；剩余缺口在燧原/昆仑/华为/muxi
  四芯（第二高组合 1.8664），按 v3 队列继续槽 4（燧原去 mask）等。

## 2026-09-14 E8 候选就绪：燧原窄形状 constexpr 去 mask（v3 队列槽 4，待发射）

- 登记在案的 B2：heads%16==0 且 nd/rd 恰满 2 幂块时全部掩码编译期
  为真——NOMASK constexpr 路径去 mask；其它形状保留 e6 掩码路径，
  契约不缩。生产形状（128/128/64）命中快路径。codex-review（commit
  级）280 组边界检查无缺陷。
- source / verification commit：`5a97279f…`；ZIP `e8-5a97279`，
  SHA-256 `b9d5b5757a0f246e883f7eecc755ab253305f6bc5a0601406e979d8664ce1c54`。
- release 回执 `batch5-submit-20260914-finale/concat_mla_k-t62e8/verification.json`
  （enflame vendor 28 launch 0F0E0S）。
- 预注册晋级门：**燧原 ≥ 0.40 且均值 ≥ 1.2301**；未过不再改该芯 mask。

## 2026-09-14 E9 候选就绪：昆仑直线段化（待发射）

- 单变量：wrapper 已保证 BLOCK_N/BLOCK_R = next_pow2(nd/rd) ≥ nd/rd，
  内层两段循环各只跑一趟——删除脚手架，直线掩码射出语义等价；外层
  行 grid-stride 保留（任意 tokens 正确）。段寻址/2 幂块/无共享基址
  的 #1147 安全几何不变。
- source / verification commit：`cb13618e…`；ZIP `e9-cb13618`，
  SHA-256 `5754e59d0cb601aa2df3b49056e8fb6a03d87592080cd4e67e85053a4f93c392`。
- release 回执 `batch5-submit-20260914-finale/concat_mla_k-t62e9/verification.json`
  （kunlunxin vendor 28 launch 0F0E0S）。
- 预注册晋级门：**昆仑 ≥ 0.40 且均值上行**；未过则昆仑 no-loop 假设
  在本题关闭。

## 2026-09-14 E8/E9 平台终态：双探针目标芯均平，TB 靠窗口漂移至 e9 1.2458x

- **E8 燧原去 mask：0.2822 vs 0.2756（+2.4% 噪声）**——门（≥0.40）
  未过，掩码非燧原缺口主因。
- **E9 昆仑直线段化：0.1822 vs 0.1842（平）**——门（≥0.40）未过，
  内层循环脚手架非昆仑缺口主因（T75 的 no-loop 大增益再次确认是
  task 特有）。
- 均值 1.2458（e9）> e7 的 1.2201 主要来自海光窗漂移（2.671 vs
  2.545）——TB 按规则更新为 e9（有效且均值最高）。两假设证伪，
  T62 的燧原/昆仑缺口继续无已建杠杆。

## 2026-09-14 E10 候选就绪：燧原 tile 16→4 heads（海光 BH4 证据平移，待发射）

- 今日同任务海光 BH4 +25%（e7 新TB）；燧原 vendor 同构 BH=16——
  平移隔离实验。代理 enflame vendor 56 launch 0F0E0S（含 NOMASK
  快路径与掩码回退双路）。门：**燧原 ≥ 0.40**；未过轴停。
- source / verification commit：`d017032f…`；ZIP `e10-d017032`，
  SHA-256 `eca96c23e24fa0107adc2adbf3d4355b016b6d0c47be2a141ed663f2a9a7ced3`。

## 2026-09-14 E10 平台终态：8/8 VALID 1.1908x < TB（submission 14847）

- **燧原 BH4 passed 0.1474**（BH16 的 0.2802 → **-47%**）——海光 BH4
  的 +25% 不可跨芯平移，燧原 tile 轴反向；轴停。BH16 载体守。
- 均值 1.1908 < TB e9 1.2458。T62 四芯缺口全部无已建杠杆，收官期
  仅守榜。
