# Task 61 `compute_src2dst` 实验记录

```current
task: 61
operator: compute_src2dst
batch: 5
validity: invalid_correctness
platform: completed(12900,7/8)
candidate_stage: e2
team_best_stage: -
sealed: no
next: 等待 E2 逐芯回调（i32 scatter 索引裁决）
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

## 2026-09-11 E1 平台提交（2026-09-11T06:49:00+08:00）

- `e1` / daily_seq `9`。nonce：`a0572c3f2b737444bfba02a88cdccbfc`。
- 外部佐证（调研）：FlagTree 燧原后端 i64 默认拒绝（enable_i64=False /
  gcu64-type-verifier）；FlagGems enflame gcu300 的 index_put 等 190 个
  override 普遍做 int64→int32 索引 downcast（含 int32 溢出边界测试），
  lo 词 scatter 与社区方向一致。
- 逐芯结果：见下方跟进记录。

## 2026-09-11 E1 平台终态（submission 12904）

- 7/8：天数 3.0042 / 沐曦 1.1592 / 海光 1.7732 / 昆仑 1.3104 / 华为 1.5536 /
  A 1.6932 / B 1.7976（七芯与 S0 一致）。燧原 vendor 被选中但**仍编译失败**
  （vendor 第 64 行 launch，全部 case）。
- 与同日 59 题 vendor（编译成功）差分：E1 src2dst vendor 独有
  **load 值 `src.to(tl.int64)` 扩位后作 store 索引**——燧原通过内核的 store
  偏移全部来自 arange/标量 extsi（kv_indices），或保持 i32 算术（decode_attention
  经 load 的 int32 pages 寻址）。E2 把 scatter 索引保持纯 i32（`out + src`，
  clamp int32 case 燧原已过的 store 形态），load 寻址保持 i64 range 数学。

## 2026-09-11 E2 提交（2026-09-11T07:08:16+08:00，daily_seq 13）

- source commit `10e38fe242b5dc35d773d62767209f92a199e3d6`；nonce `ed0afee41889dbe0820d3edb73d6623e`。
- ZIP `e2-10e38fe`，4163 bytes，SHA-256 `04191509d38cfb6bd3ee9806a92ab42abcafc3ef28ab83bc4cfbe39da1ca2662`。
- 回执 `artifacts/competition/batch5-enflame-fix-20260911/release-r3/compute_src2dst/verification.json`，
  SHA-256 `e25d95ef7f7119249b7142ee090a4abd5d4f1c7ac61e9377d7e74131344fe05d`；
  3 方法 0 失败，generic 19 + enflame 19 次 launch。

## 2026-09-11 E2/E3 终态与 E4 提交

- E2（daily_seq 13，07:08）：燧原**首次编译通过**（i32 scatter 索引解除编译
  阻断）但数值 99% 错——`out` 为 `torch.empty`，错误指纹（inf 相对差）表明
  scatter 写错地址、未写元素留垃圾。其余七芯正常（天数 2.982 / 沐曦 1.147 /
  海光 2.017 / 昆仑 1.2902 / 华为 1.6086 / A 1.6222 / B 1.7558）。
- E3（daily_seq 15，07:19，extsi→i64 × runtime stride）：燧原**编译再次失败**
  ——与 E1 合并实证：**向量 load 值的 extsi→i64 进 store 寻址是编译毒点**；
  E2 的裸 i32 addptr 可编译但寻址错。
- E4（daily_seq 18，07:35:16，commit `cbd35d8`）：唯一未试形态——纯 i32
  `src * os`（runtime stride，do_not_specialize 防 1 特化折叠），镜像
  decode_attention 已证 gather 数据流（load 值 × i32 stride 寻址）用于 store 方向。
- 回执 `release-r5/compute_src2dst/verification.json`（SHA-256
  `4cb1ee8501173f78c8feaf91e623d951604be7fd370d1832911069ac23555a59`，
  3 方法 0 失败）；ZIP `e4-cbd35d8`，4415 bytes，SHA-256
  `36718e763de533d434f020b1e200ae4b1dbeef6fed7a568541478c5ce2acdc97`。

## 2026-09-11 E4 状态：燧原回调异常挂起

- E4（daily_seq 18，07:35:16）：其余七芯全过（天数 2.992 / 沐曦 1.1564 /
  海光 1.9016 / 昆仑 1.3104 / 华为 1.6078 / A 1.636 / B 1.7972）；
  燧原 waiting_callback 挂起超 50 分钟（历轮 ~10 分钟出结果）——怀疑
  `src * os` i32 scatter 在 GCU 运行时挂死或评测机窗口异常，等待终态。

## 2026-09-11 E4R 终态与止损

- E4R（daily_seq 22，08:20，commit `10fed92`）：燧原回调正常返回，
  **数值失败与 E2 指纹完全相同**（99%、diff 494@idx7、inf 相对差@476）
  ——`src * os`（i32 × runtime stride）与裸 i32 addptr 的 scatter 落点同样
  错误；E4 当次的「服务线程卡死」确为评测器崩溃族而非内核挂死。
  回执 `release-r8/compute_src2dst/verification.json`（SHA-256
  `7d2fd8b23ab69673c59fb0a737c5ff7a28672cf00287ccab3f89871d3ab055c7`）；
  ZIP `e4r-10fed92`，SHA-256
  `824722fa72ca38c2640cf4278b1be418f76bd3939823834da450a510f327bbf8`。
- **结论（GCU 规则集补充）**：数据依赖 scatter（load 值作 store 索引）在
  extsi-i64 / 裸 i32 / i32×runtime stride 三种形态下分别为编译死/错址/错址
  ——此 GCU 栈的 DMA store 不支持 load 索引寻址。重启候选方向：
  `tl.atomic_xchg`（非 DMA lowering 路径）。
- 七芯水位（E4R）：天数 3.0196 / 沐曦 1.171 / 海光 1.8468 / 昆仑 1.2956 /
  华为 1.6166 / A 1.6326 / B 1.7718。
