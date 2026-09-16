# Task 61 `compute_src2dst` 实验记录


## 2026-09-13 E8：燧原配方复刻（4096+封顶 24）——不适用 scatter（未过 TB）

- 真值：avg 2.0460 < TB 2.074；**enflame 3.8018 < 最佳 4.10（-7%）**，
  其余芯持平。**配方（grid 封顶+BLOCK 4096）对 streaming elementwise
  成立（T73 +92%/T75 +113%），对 scatter 无效**——散乱 store 需要
  程序级并行，封顶 24 后并行度不足。边界已清晰。
- 唯一残余轴：muxi 1.08 vs 榜首 2.72（2.5x），无已知形态证据。

```current
task: 61
operator: compute_src2dst
batch: 5
validity: valid
platform: completed(15835,e11,8/8,2.138475x新TB;Metax门未过)
candidate_stage: e11
team_best_stage: e11
team_best_speedup: 2.138475
sealed: no
next: 保留E11团队最佳；Ascend cap/tail候选仅隔离留档，代理未提速；等待Ascend同源运行证据
updated: 2026-09-16
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

## 2026-09-11 E5（2D 行段 scatter）：燧原评测器崩溃，内核未获裁决

- 情报：FlagGems 燧原 gcu300 `index_put` 在 GCU 上成功 scatter 的形态 =
  **2D 行段 store**（load 值 i32 × stride 作行基址 + arange(0,1) 列、
  rank 折叠分支、全程无 extsi）——与我方三种失败的逐 lane 1D scatter 不同。
- E5（daily_seq 27，09:55，commit `26eb3ac`）：`[BLOCK, 1]` 2D 行段 scatter
  （纯 i32 索引算术）。燧原 waiting_callback 挂起约 1.5 小时后判
  **「服务线程卡死自动恢复」评测器崩溃族**（与 E4 同指纹）——2D 形态在
  GCU 挂死或评测机连续异常，内核未获数值裁决。其余七芯正常（天数 3.0112 /
  沐曦 1.1632 / 海光 1.8674 / 昆仑 1.2924 / 华为 1.6286 / A 1.5972 / B 1.796）。
- 回执 `batch5-deepopt-20260911/release/compute_src2dst/verification.json`
  （SHA-256 `4a87d80c1a73c2b084175715e55976e5777f05c031bdd1731811be93d597dae7`）；
  ZIP `e5-26eb3ac`，4698 bytes，SHA-256
  `35421dd63107a4e6ade6df1291107a423c43b1e8c23fdb50d112011c67f0f014`。
- 明日：E5 字节重掷一次区分「GCU 挂死 vs 评测机窗口」；仍崩则 2D 形态
  也触发挂起，61 燧原轴与 60 一并封存等外部证据。

## 2026-09-11 E5R 候选就绪（待下一额度窗口）

- E5 同字节注释载体重掷（区分 GCU 挂死 vs 评测机窗口；崩溃族重掷 1/2）。
- source commit `95e5292…`；ZIP `e5r-95e5292`，5003 bytes，SHA-256
  `94d7e1745051048988e36fbf9d4427c201aeb6e7394457bfb1c7210a6bfd9969`。
- 回执 `release-next/compute_src2dst/verification.json`（SHA-256
  `3361018b37f6fa032f2d15c33ca4e0608282fd4b27496b612731291ad02fdb0d`）。
- 状态：候选就绪未提交。

## 2026-09-12 E5R 平台结果（submission 13221，daily_seq 2）

- 7/8（燧原失败）；七芯读数为历次最强：天数 3.012、沐曦 1.0712、
  海光 1.907、昆仑 1.2864、华为 1.7086、A 1.6186、B 1.7828。
- 燧原失败详情（raw_result）：`selected_file=compute_src2dst_enflame.py`
  被选中，**执行完成（8771ms）后数值失配**——`[1]` 511 元素中 506 个错
  （最大绝对差 494）、`[2]` 8192 中 8143 错。**不是挂死、不是窗口**：
  2D 行段 scatter 在 GCU 上确定性写错地址。
- 结论：e1/e3/e5r 三种 scatter 寻址形态在燧原全部失败（前两种编译失败/
  毒点，本种确定性错值）。按预注册止损，61 燧原轴与 60 一并封存，
  仅 precomputed-pos 全新结构（wrapper 预计算位置、kernel 纯掩码拷贝，
  T49 已证形态）可重开。
- 额度：发后 28/30。

## 2026-09-13 E7 平台终态：8/8 VALID 2.074025x（submission 13772）

- **八芯全过**——七轮燧原攻坚收官：宿主 int32 降位 + 纯 i32 地址链
  （FlagGems gcu300 scatter 纪律）首次通过；均值 2.074025。
- 沉淀：GCU scatter 黄金法则 = **kernel 内零 int64 数据**（宿主降位）+
  i32 计算偏移 + do_not_specialize 尺寸。
- 榜首 RSI 2.421，差 0.35；七芯水位本就是未过线最强，性能轴后续按
  逐芯读数定。

## 2026-09-14 E9 候选就绪：e7 字节恢复 + 宿主去零填充（待发射）

- Codex 审查发现：燧原 vendor 用 `torch.zeros` 分配输出，而题面契约是
  完整置换（每个位置必写一次）——零初始化是一整个白烧的设备写。
- 载体 = **恢复 e7 字节**（e8 配方已被平台证伪：3.80<4.10）+ `zeros→empty`。
  generic 七芯路径与 e7 完全同字节，读数应仅在窗口方差内波动。
- source commit：`e5a9a069bb33`；verification commit：`754d91e02ae6`。
- ZIP `e9-e5a9a06`，2 members（generic + enflame），SHA-256
  `5e3df87636ff47f0d343264405a3837b0397e4a39be8dda5c8494d06c3c796bc`。
- release 回执 `batch5-submit-20260914/compute_src2dst/verification.json`
  SHA-256 `fea9029fe330ac62e89621f36bcf3a5d0c69214a2d0cfca5c8fab0f647db9288`
  （release 模式、3 tests 0F0E0S、generic 19 次真实 launch；
  enflame vendor target-runtime-unverified）。
- 预注册晋级门：**燧原 ≥ 4.10（e7 水位）且均值 > TB 2.074**；燧原无增益
  则轴关，均值门失败但七芯无异动即收盘。

## 2026-09-14 E9 平台终态：8/8 VALID 2.099675x 新 TB（submission 14519）

- **预注册双门全过**：均值 2.0997 > TB 2.074；**燧原 4.4636 ≥ 4.10**，
  且较 e8 的 3.8018 +17.4%——e7 字节恢复 + zeros→empty 兑现。
- 逐芯（vs e8）：天数 3.0326（3.1358）/ 沐曦 1.177（1.1548）/
  **燧原 4.4636（3.8018）** / 海光 1.8508（1.8764）/ 昆仑 1.2908（1.289）/
  华为 1.4996（1.6354）/ A 1.7186（1.6936）/ B 1.7644（1.7814）——
  除燧原外均在窗口方差内，与"generic 同字节"预判一致。
- 榜首 RSI 2.42，差距收窄至 ~0.32。残余轴仍只有 muxi（1.18 vs 2.72）。
- 额度：发后 25/30（watch 实测）。

## 2026-09-16 Top1 差距与 E10 淘汰式筛选

- 实时 10:46：第6名，TB 2.099675，榜首2.606375，需 +24.13%。沐曦1.177→3.3554、燧原4.4636→5.8054两芯同时追平全榜已证水平，仅得2.539700，仍差0.066675；不能把单芯修复宣称为登顶方案。
- E10 source/verification `6b16ac1260e2f54887e026ce3010b351553d2d40`，只新增metax；release 3/3、0失败/skip，generic/metax各19次真实launch；回执 `artifacts/competition/batch5-verify-20260916/compute_src2dst/verification.json` SHA-256 `97d89f7c2b4ee5c7c31853765b01bcc89cd24b2ff81e5e9943ab9273b6ce3626`，本轮已执行完整本地验签。
- 不可变 ZIP `artifacts/competition/compute_src2dst/e10-6b16ac1/compute_src2dst.zip`，5837 bytes，SHA-256 `89f1b1ce369a455a23fa7e7d44f36350fb9410d021ab86ddac9122cd45e72a79`；成员generic/enflame/metax，完整成员哈希由规范打包器可重现。
- 本轮补五轮 AB/BA、wrapper-inclusive、8个case（int64/int32 × n256/131072 × contiguous/stride2），远端RTX5070Ti、无竞争进程、180秒上限。int64连续候选/基线速度比0.825/0.749，stride2为0.683/0.667；int32连续1.000/0.918，stride2为0.735/0.713。27寄存器/0spill/8192共享内存，generic18–22寄存器/0spill/0–2048共享内存。
- 原始样本、脚本、源码哈希、命令、PID保留 `artifacts/competition/t61e10-perf-20260916/`。结论仅是NVIDIA代理负结果：host cast/contiguous额外设备工作可测，不冒称Metax性能已证伪；为避免低价值试投，本轮不preflight E10。
- 下一候选 E11：保持Metax flat1024，直接读取原索引并在kernel内转int32寻址，支持原stride；移除wrapper转换。预注册：适用正确性全部通过，代理相对E10有稳定收益后才进入平台；平台沐曦≥2.7且八芯均值>2.099675晋级，全部芯≥0.1。若目标无收益则关本假说，保留E9。

## 2026-09-16 E11 候选就绪：去除 Metax wrapper 转换

- source / verification commit `cee98e52e87e77074d2d9a8eb612375d02c2e71c`；ledger commit 为本节所属提交。保留flat1024，直接读取原int64/int32索引，读取地址保留64位stride计算，store索引转int32；generic及enflame字节冻结。
- 完整screening3/3，五轮AB/BA：相对E10，int64连续n256/131072为1.2145/1.2399x，stride2为1.4499/1.4916x；相对generic约0.998/0.928（连续）、0.999/0.998（stride2）。代理只证明消除多余转换有益，不能宣称已经胜generic或Metax；目标专属单发依据为两队同芯2.92/3.3554的外部可达证据和新的flat/i32形态。
- screening原始样本 `artifacts/competition/t61e11-screening-20260916/`；相同source SHA `06585b64c2eba8febbd98d002428636d9fbbe0d24f80c7818baba70e167ea274`。
- exact release `artifacts/competition/t61e11-release-20260916/verification.json`，SHA-256 `a2c073a2c99780f2b8faca8800c6f715593ab7a073e93643e96178480ec8d66a`；日志SHA-256 `0a844cca87cce8e2495507ceb0af5c573fa843ae3c9cfbfb464e9fe3131ec706`。3/3、0失败/skip，generic/metax各19真实launch，RTX5070Ti；源码/测试/runner/Git对象/日志已验签。black25.12/isort/flake8通过。
- 测试SHA-256 `1708ab05b089f1ad702294338a252e62f1ae41b40517ddd58bdda3603d166887`；ZIP `artifacts/competition/compute_src2dst/e11-cee98e5/compute_src2dst.zip`，5206 bytes，SHA-256 `2d87e2fbc07b40790968cf22148376042edf8e5c52816500195bf8ed1616c4cc`。
- 成员 `compute_src2dst.py` SHA-256 `f59a3ffa8e3a463be40f7e70e8036b656a6df689899928689d073ba983215051`。
- 成员 `compute_src2dst_enflame.py` SHA-256 `2dbf3510c513687777efb04222007ecf276aeaafd739e7da684e74d038a736d8`。
- 成员 `compute_src2dst_metax.py` SHA-256 `06585b64c2eba8febbd98d002428636d9fbbe0d24f80c7818baba70e167ea274`。
- Metax真机 target-runtime-unverified；09-16实时KernelGen schema仍无固定候选字节执行接口，独立目标机无已授权记录。代理不能替代目标评测，平台将补齐证据。
- 预注册门：Metax≥2.7、全部8芯≥0.1、均值>2.099675；未达不重复同候选，保留平台团队最佳。

## 2026-09-16 E11 平台终态：8/8 valid，新TB但目标门未过

- submission **15835**，10:57:34；均值 **2.138475x**，较E9 +1.85%；远端ZIP回读verified，5206bytes/SHA与本地一致。nonce `662a23916f9823db729cd35ad44a5354` 终态submitted，禁止重投。
- 逐芯：tianshu 3.004 / muxi 1.2486 / enflame 4.4408 / haiguang 1.8854 / kunlunxin 1.287 / huawei 1.7128 / card_a 1.7436 / card_b 1.7856。Metax选中文件 `compute_src2dst_metax.py`，正确性已补齐；1.2486低于2.7预注册门，假说停止。generic/Enflame冻结字节的变化不计为本次代码收益。
- 状态原文 `artifacts/competition/top1-20260916/t61-e11-status.json` SHA-256 `577d9d9419bf0a3abc141877dc337d763710193b5f611ddb231a70dbdd1d182f`；提交原文 `artifacts/competition/top1-20260916/t61-e11-submit.json`。剩余额度 **27/30**。

## 2026-09-16 晚间：Ascend新结构隔离筛选，未晋级

- 基线提交 `50b91047515f9c45dd7397303996b1a5e9e359da`；本轮只做开发筛选，无平台preflight/上传/提交。候选保存在 `artifacts/competition/t61-ascend-20260916/bundle/`，不替换正式vendor。候选source SHA-256 `741c51fa8de5d6328f82e59078b2d40526953687f356c8992c8d570429c1979c`。
- RTX5070Ti / driver610.57.04 / Python3.12.13 / Torch2.13.0+cu130 / Triton3.7.1。screening同源完整回归 5 方法，0失败/错误/skip；明确执行generic+Ascend数学代理。Ascend仍为 `target-runtime-unverified`，NVIDIA结果不构成目标芯否定证据。
- 五轮交替wrapper-inclusive计时，16桶。候选相对generic几何均值：0.838433（仅cap）/0.832554（cap+完整块去mask）。没有正向性能证据，不晋级、不继续在NVIDIA扫配置；重开条件是Ascend同源实测或可绑定的目标编译产物。
- 首轮每次查询driver设备属性带来约1.7ms宿主开销；已用按device-index的标准库缓存修复并重新完整运行。旧uncached计时只保留诊断，不参与上述结论。缓存后大N封顶回退，baseline/cap-only逐桶PTX相同；完整块去mask未产生稳定改善。物理地址保持原64位链。
- 复现脚本、输入清单、回执、完整日志、逐桶逐轮原样本与资源记录保留在上述目录；screening不是release，不给旧ZIP或平台资格背书。
- `verification.json` SHA-256 `d0ce7661e2830f551de753da21516d53d72206f286dc9c8b66223a63270ca61a`。
- `verification.log` SHA-256 `9fc2da113f8aa25ff0198fa64730933a996d3d6a839d92d05f40bfaeee2fd259`。
- `benchmark.json` SHA-256 `903ab68f3862056bf44a7171475a2139c324deff34e4958e52cb7da5b649ccd4`。
- `run.log` SHA-256 `d992f044e551404b96b9f1ad3dff3d67b1e8243f70caeb92caa2c21f12d9de8c`。
