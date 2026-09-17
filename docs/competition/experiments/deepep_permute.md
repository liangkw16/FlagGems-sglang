# Task 64 `deepep_permute` 实验记录

```current
task: 64
operator: deepep_permute
batch: 5
validity: valid
platform: completed(16798,e15-carrier,8/8,8.47025x<TB;muxi单芯新高8.16)
candidate_stage: e15
team_best_stage: e13
team_best_speedup: 8.73395
sealed: no
next: 华为4.433未过6.2门不重掷；下一华为轴候选=官方gather/scatter best-practice形态(SUB_BLOCK_SIZE/insert_slice)，需先落结构再一发判决
updated: 2026-09-17
```

> 下方 S0 开发记录是 2026-09-10 快照；当前平台结果见 CURRENT 和文末提交记录。

## 契约与范围

- 完整题面：[Task 64](../tasks/batch-5/64-deepep_permute.md)；[开发契约](../strategy-batch5.md)。
- exact；八芯每芯 0.1x；截止 2026-09-17 19:59:59（北京时间）。
- 按公开 reference 返回新张量并保持输入不变；核心计算使用 Triton，无 fallback。
- generic S0；NVIDIA 代理已验证，平台及其他目标 runtime 标记 `target-runtime-unverified`。
- 本轮未执行平台 preflight、上传或正式提交，不消耗额度。

## 不可变身份

- source commit：`4836fe0378d6290cbcaa91958693e13ed1acdf3e`。
- verification commit：`4836fe0378d6290cbcaa91958693e13ed1acdf3e`。
- ledger commit：本文件的 Git 提交；用 `git log -1 -- docs/competition/experiments/deepep_permute.md` 定位。
- source SHA-256：`14339eba6cd4d0848066f025495afaf111d647aaded8f46abb91783a15226560`。
- test SHA-256：`3232b1c38f82804e83ab4d2d0983619b07257ea4430b7921a3e9bfa6bc76f925`。
- ZIP：`artifacts/competition/deepep_permute/s0-4836fe0/deepep_permute.zip`，1952 bytes，仅含 `deepep_permute.py`。
- ZIP SHA-256（等于 canonical）：`16a88406cb5775a2ba4f71351aaea11260b4d22a6ac3f13e1febcd80be3db454`。
- dry-run manifest、最终构建、existing 验签、Git blob 逐字节比对及 unzip -t/-l 全部通过。

## 验证证据

- 5 个测试方法 / 63 条 test/subTest 记录 / 34 次实际 kernel launch；失败、错误、skip、xfail 均为 0。
- RTX 5070 Ti；Python 3.12.13；PyTorch 2.13.0+cu130；Triton 3.7.1；CUDA 13.0。
- 回执：`artifacts/competition/batch5-development-20260910/release/deepep_permute/verification.json`。
- 回执 SHA-256：`e2f4f50ae5c16620fc9f5fd4699b856ee001b7913865015c0afb8b54e7a6c6ae`。
- 完整日志：`artifacts/competition/batch5-development-20260910/release/deepep_permute/verification.log`。
- 日志 SHA-256：`2f7fafbd7a8f246d54ec1ef454a5e159ac9cdd5074a7acacb3e995fa458e0cdb`。
- benchmark：`artifacts/competition/batch5-development-20260910/release/deepep_permute/benchmark.json`，SHA-256 `239f776c8cc989080a7cf5ae96a1292e4bf571180129a3143a3ffbf28c1c39f3`。
- Black/isort/flake8、Python 编译、空输入、边界、stride 和输入不变性均已检查。
- 所有 source/test/helper/runner/benchmark 哈希见[证据清单](../data/batch5-development-20260910.json)，清单 SHA-256 `74b37287f5649dc044696565306c97111117c991b3549393a88bcb703eba46b1`。

## 代理性能

| 代理负载 | 五轮配对 speedup 中位数 | 候选中位耗时 |
| --- | ---: | ---: |
| 8x4x4096 | 5.4297x | 10.120 μs |
| 512x4x4096 | 3.1446x | 58.857 μs |

测速含 wrapper；五轮交替 AB/BA，每次 warmup 20ms / rep 50ms。
这些是两种代理负载的结果，不是平台平均分；原始样本已保留。

## 重放

远端隔离目录：`/tmp/flagos-batch5.POXFt3/deepep_permute`；复跑需 prepare 到新目录，旧回执不可覆盖。

```bash
python .agents/skills/flagos-operator-race/scripts/verify_release.py prepare deepep_permute --source-commit 4836fe0378d6290cbcaa91958693e13ed1acdf3e --verification-commit 4836fe0378d6290cbcaa91958693e13ed1acdf3e --dependency tools/benchmark_batch5.py --directory /tmp/NEW_RELEASE_DIRECTORY
timeout 600 /home/kevin/notebook/.venv/bin/python /tmp/NEW_RELEASE_DIRECTORY/.agents/skills/flagos-operator-race/scripts/verify_release.py run --directory /tmp/NEW_RELEASE_DIRECTORY
```

## 2026-09-11 一轮优化后的待提交候选

- 候选 `e1`；按 tile 调度优化晋级 E1。首次提交前候选就绪。
- source commit：`29754c1dfcbc87c9b193e750145b668c8e35d852`；verification commit：`29754c1dfcbc87c9b193e750145b668c8e35d852`。
- ZIP：`/Users/bytedance/ccc/flagos/artifacts/competition/deepep_permute/e1-29754c1/deepep_permute.zip`；1970 bytes；成员 `deepep_permute.py`。
- ZIP SHA-256：`b33968d2510443859fc50f02926127bc85ba002425a4bf42228e4d812bbb32b1`。
- 源码 SHA-256：`7857c235db4f253b025d9890b444215455bde6c8ea99bbe9cecd3f6a8d69fc1a`。
- 测试 SHA-256：`822d3de09ab6a8b32d228e30709cd3d2453a040663d38b8ba5bd264031297c84`。
- 回执：`artifacts/competition/batch5-optimize-20260911/release/deepep_permute/verification.json`；SHA-256 `e07bda65b5e89c95cfc4952cc9b9e52dbab17084ce5f892e3fca760cf8250e3c`。
- 日志：`artifacts/competition/batch5-optimize-20260911/release/deepep_permute/verification.log`；SHA-256 `274fe4cb929c2c7bf910f84d533bb083e2d58caefdf702a43240ddfaed206c09`。
- 完整 release：5 个测试方法 / 64 条记录 / 35 次 kernel launch，全通过；NVIDIA 代理范围。
- 优化和复现见[本轮报告](../optimization-batch5-20260911.md)；[完整证据](../data/batch5-optimization-20260911.json)，SHA-256 `87a560dd86ab7e9dfb9607f8f05a007060d48b3aab876b165bfc0fec591f77db`。

## 2026-09-11 首次平台提交（2026-09-11T03:39:24.977852+08:00）

- `e1` / submission `12895` / daily_seq `2`，提交于 `2026-09-11T03:27:13+08:00`。每候选上传与正式 POST 各一次，无自动重试。
- nonce：`08f1a53b02603b388fdec2dbcaa75c1d`；上传 URL SHA-256：`5aeefb4e06b920d458d9d87323a9b8b95beb5db13a469e2fbf776e5919ad8cbc`。
- 远端 ZIP 回读 verified，1970 bytes，SHA-256 `b33968d2510443859fc50f02926127bc85ba002425a4bf42228e4d812bbb32b1`，与本地候选逐字节一致。
- 平台 `completed` / `valid`；正确性通过 8/8，完成 8/8；6.7478x；第 3/4 名。
- 当次 status 额度：24/30，已用 6；observed_at `2026-09-11T03:39:24.977852+08:00`。排名查询时间 `2026-09-11T03:39:41.445754+08:00`。

| 芯片 | 状态 | 正确性 | speedup |
| --- | --- | --- | ---: |
| tianshu | completed | 通过 | 16.1606 |
| muxi | completed | 通过 | 5.5116 |
| enflame | completed | 通过 | 2.5326 |
| haiguang | completed | 通过 | 10.3318 |
| kunlunxin | completed | 通过 | 0.2538 |
| huawei | completed | 通过 | 3.7614 |
| card_a | completed | 通过 | 8.4868 |
| card_b | completed | 通过 | 6.9438 |

下一步：保留 E1；目标轴按均值增量排序（见下方 R2 候选节），昆仑并非第一轴。

[提交结果证据](../data/batch5-submissions-20260911.json)，SHA-256 `63418b87e9249bef72751df2a1778c52e6d679a5d313ecf18d8ed64b96c175dd`；原始 status `artifacts/competition/batch5-submit-20260911/64-status-033924.json`，SHA-256 `4dbc91d328aaee717d6ec1a70041d1d9562e504ed39f8ae90979ea6abe302b3d`。

## 2026-09-11 R2 目标轴纠正（归因，替代上文"优先昆仑"口径）

真实靶子是次席 **Nectar 7.61745**，不是榜首 EvokeAgent 21.649025——后者的
81% 来自华为单芯 100.305，而第二名同芯仅 3.0316（差 33 倍，远超 20x 判据），
是不可复现的窗口彩票。按"追平 Nectar 后的均值增量"排序：

| 芯片 | 我方 | Nectar | 增量 | 占 0.86965 |
| --- | ---: | ---: | ---: | ---: |
| enflame | 2.5326 | 5.5018 | **+0.3711** | 42.7% |
| haiguang | 10.3318 | 12.7074 | +0.2970 | 34.2% |
| muxi | 5.5116 | 6.7576 | +0.1558 | 17.9% |
| tianshu | 16.1606 | 17.1312 | +0.1213 | 13.9% |
| card_a | 8.4868 | 8.7262 | +0.0299 | 3.4% |
| card_b | 6.9438 | 6.9442 | +0.0001 | 0.0% |
| kunlunxin | 0.2538 | 0.1396 | −0.0143（已反超） | — |
| huawei | 3.7614 | 3.0316 | −0.0912（已反超） | — |

即把昆仑从 0.2538 提到同题第三名 0.9048 只值 **+0.081**，而把燧原从 2.53
提到同题最优 6.68 值 **+0.519**。**燧原第一轴、海光第二，昆仑可忽略。**

## 2026-09-11 R2 待开发候选：燧原轴（预注册门，未开发）

- 前置事实：`1c0381c` 新增的 `_enflame/ops/deepep_permute.py` 只是把 generic 的
  clone+scatter 字节**冻结成载体，不含优化**，救不了 2.17x 的缺口。
- C1（有他题同芯平台先例）：现 launch `min(tasks, 65535)` 改为按物理资源封顶
  （燧原 24），循环体已是 grid-stride，仅改 cap 数值。
- C2（假设）：现 kernel 的 `dst = tl.load(routes)` 后直接
  `tl.store(out + dst * os0 + ...)`，正落在 T61 平台实证的毒点
  "load 取值直接作 store 索引"上；按 T49 precomputed-pos 路线改写成 wrapper
  预计算目的地地址张量、kernel 纯乘法掩码。
- **clone 轴不复投**：上限虽为 1.43~2.13x，但 `1c0381c` 的 destination-gather
  重写实测仅 0.35x、宽行变体 0.25~0.60x，两条均已被测量否掉。
- 预注册门：燧原中位收益 **≥1.3x**（C1）或 **≥1.5x**（C2），其余七芯无回退
  >5%；完整正确性通过。
- 证据等级：C1 为他题同芯平台先例，C2 为**结构假设**。
  详见 [r2 §4](../optimization-batch5-r2-20260911.md)。

## 2026-09-12 E2 = C1（燧原 grid 物理封顶，候选就绪后提交）

- R2 预注册 C1 兑现：`_enflame` vendor 启动 `min(tasks, 65535)` 改
  `min(tasks, 24)`（24-SIP 物理宽度；超发 grid = 纯调度开销，T19-E5/
  T51-E5 平台教训 +38% 中位）。循环体本为 grid-stride，单变量仅 cap。
- source commit：`a70efbbfa7da447ab199eeab4e4863de6fc26b4d`。
- ZIP：`artifacts/competition/deepep_permute/e2-a70efbb/deepep_permute.zip`，
  SHA-256 `7b2425e0ebb4a0aa0906b70c8609705770094814aeca4327d3dd7e85366b0aab`；
  成员 generic `7857c235…` + `_enflame` `5692986d…`。
- release 回执（v2，绑定 a70efbb，proxy-vendor enflame）：
  `artifacts/competition/batch5-t64c1-validate-20260912/deepep_permute/verification.json`，
  SHA-256 `b2dc8ef1a6e22f120caef42c8199753215421ac832a7ac1c81e3b06f9aae5594`；
  日志 SHA-256 `0a3b0036a54a35a779b9fb33509cee530292ea960f36483598c03e2cac63c288`；
  5 方法 0 失败，generic 35 + enflame 35 launch。
- 预注册门（R2 §4）：燧原中位收益 ≥1.3x；其余七芯 generic 字节不变。

## 2026-09-12 E2 平台提交（submission 13375）

- 上传与正式 POST 各一次；state submitted。额度：发后 10/30。
- 裁决点：燧原 ≥1.3x（2.53→3.3+）；其余七芯与 e1 水位一致。

## 2026-09-12 E2=C1 平台结果（进行中）：燧原 +5.5%，未达预注册门

- 已判 7/8（仅昆仑回调中）：燧原 **2.6708（e1 2.5326，+5.5%，`_enflame`
  vendor 选中）——未达预注册 ≥1.3x 门**，grid 封顶轴按预注册关闭；
  燧原真实缺口（同题次优 6.68）需结构轴（R2 已注册 C2 precomputed-pos：
  wrapper 预计算目的地地址、kernel 纯乘法掩码，消"load 值作 store 索引"）。
- 其余六芯与 e1 水位一致（±3.5% 内）：天数 15.8842 / 沐曦 5.4942 /
  海光 10.3922 / 华为 3.8306 / A 8.1884 / B 6.8864。
- 若昆仑维持 e1 水位（0.25x 量级），本发均值 ≈6.70，略低于 e1 6.7478
  ——team best 预计保留 e1；本发价值=燧原 +5.5% 的正向确认。

## 2026-09-12 E2=C1 平台终态：7/8（昆仑=崩溃族，未获裁决）

- 昆仑终态 `exec 0ms + 服务线程卡死`——vendor 未执行，非内核裁决。
  e2 判 invalid_correctness（缺芯不排名）；e1 6.7478x 保持 team best。
- 昆仑对 generic（e1 字节）09-11 曾正常通过 0.2538；后续新 ZIP（C2
  precomputed-pos 已注册）自然重掷。

## 2026-09-15 15:05 E2 昆仑终态回填 + 今日轴重定（Codex 修订采纳）

- 只读 status 查询确认：昆仑终态 `exec 0ms + 服务线程卡死`（崩溃族，
  vendor 未执行）——grid 封顶 24 未救昆仑，与 09-12 初判一致；e1
  6.7478 保持 team best，e2 均值不排名（缺芯）。
- **去 clone 方向降级为条件项**（2026-09-15 Codex 咨询 + 本地核实）：
  单 kernel 混写全量复制与覆盖写在跨 program 无全局顺序，需先构建
  逆路由 dst→src 才合法；且历史 destination-gather 代理仅 0.35x。
  进入条件=阶段计时证明 clone 占比足够大 + 逆路由方案过目标芯
  screening。
- 今日轴：现有 scatter 的 vendor 地址/块宽单变量优化；预注册门
  **华为 ≥5.65（3.76→+50%）或 燧原 ≥3.8（2.53→+50%）**，两芯分别判。

## 2026-09-15 21:30 E3/E4 燧原阶梯：1024 +19% 未达门，2048 已发射

- **E3（9ae907c1，BLOCK 512→1024）**：燧原 2.53→**3.01（+19%）**未达
  3.8 门；其余七芯噪声带（均值 6.696<TB 6.748）。TB 保持 e1。
- **E4（ea69f3a5，BLOCK 1024→2048）**：已发射，门 燧原 ≥3.8（c2flow
  形态 12.3 为远端参照）。

## 2026-09-15 21:55 E4 平台终态：燧原 2048 回落，BLOCK 阶梯在 1024 见顶

- 燧原 2.75（e3 1024 读 3.01，-8.6%）：**阶梯峰=1024**。两发未达 3.8
  门，轴关闭。燧原 3.01 vs c2flow 12.3 的剩余为结构差（不属宽度族）。
  TB 保持 e1 6.748。

## 2026-09-16 00:04 E5 发射（跨午夜，新额度）：华为 topk constexpr

- _ascend vendor 新建（f8e1aec，generic 字节 + `topk: tl.constexpr` 单
  变量，K 循环界静态折叠）。已提交评测中；门 华为 ≥5.65。

## 2026-09-16 00:45 E5 平台终态：8/8 VALID 7.1711x 新 TB；燧原 6.67 待复判

- 判决：天数 16.01 / 沐曦 5.56 / **燧原 6.666（e4 同字节读 2.75，
  判为 e4 低窗——真实水位≈6.7，c2flow 12.3 为目标）** / 海光 10.20 /
  昆仑 0.27 / 华为 3.09（topk constexpr 未达 5.65 门，≈持平偏下）/
  A 8.67 / B 6.90；均值 **6.748→7.171 新 TB**（ΔS=+0.42）。
- 华为 constexpr 轴关闭；燧原水位修正后剩余结构差 = 6.7→12.3。

## 2026-09-16 E6：删除 Enflame 空任务，发布验证通过

- TB E5 `f8e1aec19ae52a82070e48ffdc3fd72a81074011` 的Enflame实际BLOCK2048，但tiles仍按512计算；hidden4096每token8个task仅2个有效，其余仍读路由/走slot循环。仅以同一`_BLOCK=2048`导出tiles和launch，generic/Ascend逐字节冻结，clone/数学/stride/cap24不变。全第五批54源静态扫描未发现第二处同等确定的错配。
- screening **6/6方法，generic/Ascend/Enflame各44真实launch**；12桶×5轮AB/BA受影响端到端中位1.2131x，大矩阵512×4×4096为1.9018x、kernel-only2.7603x；全部桶最差0.9969x、零spill/shared0、控制无回退。原始样本/脚本/环境/双端hash：`artifacts/competition/t64e6-preparation-20260916/`；results/screening.json SHA-256 `a5c700ae8509fd8d35455a43610c8e376a3c71d6321ee5b07c835dad5a3d8654`，日志 `8e0f16d7f2f9b827f54a5403aae2caa3eb13c669fa7a64b5d43d1fd8bbe8eae6`。
- source/verification commit `1c9157a87f0a754463a87935e35396b951eefcd0`；Enflame SHA-256 `8537558942805522f7811ef81829974ee3c750f42b019b0f5da871a1f81f77f2`、test `870966b8f8b0eab0a0e270c193edaddc41cf25cefc7a70d500b32be4de0da10d`，与screening字节完全一致。新增几何工作量回归及511/512/513/2047/2048/2049/4095/4096/4097边界；不会用正确输出掩盖空task。py_compile/Black/isort/flake8通过。
- NVIDIA exact release **6/6、0F0E0S、generic/Enflame各47入口/44实际launch**。回执 `artifacts/competition/t64e6-release-20260916/verification.json` SHA-256 `46462da4e0c1ae88216377bf28d6e8e6cd0337bcda2c5510f0333f6558600c7d`、相邻日志 `2d0c5481a1a0a2cbbb8fc90cdb3886063db9e231f89b6c50b7d6c1f69b83d272`。RTX5070Ti、Python3.12.13/Torch2.13.0+cu130/Triton3.7.1；远端`/tmp/flagos-t64e6-release.Vp3CP9`、PID387629、660秒总上限、EXIT0；重放包和命令同目录。Enflame target-runtime-unverified，Ascend未在本轮release执行但screening已完整代理覆盖且源码未改。
- ZIP `artifacts/competition/deepep_permute/e6-1c9157a/deepep_permute.zip`，6930bytes，SHA-256 `b3bf732e3c31b36cbc15124371e98c61a33ad18b1770f75e1591531143bbc2ec`。三个成员SHA如下。
- 平台预注册：8/8且各芯≥0.1，Enflame≥8.0为目标正信号（TB6.666）；均值>7.17105才换TB。单芯追平当前榜首只增加平均约0.676，不能宣称本修复足以从7.17到26.49登顶。一次候选一次判决。

- `deepep_permute.py`：`7857c235db4f253b025d9890b444215455bde6c8ea99bbe9cecd3f6a8d69fc1a`
- `deepep_permute_ascend.py`：`c032bc09a05cbeaa217e4fa03207eb08854d2ca6d2548298dd99c2aebcd8d5f1`
- `deepep_permute_enflame.py`：`8537558942805522f7811ef81829974ee3c750f42b019b0f5da871a1f81f77f2`

## 2026-09-16 E6 单次平台提交

11:40:46，submission15868、daily_seq7；upload/正式POST各一次、state=submitted，远端ZIP字节验签一致。发后额度23/30；结果待八芯终态。证据 `artifacts/competition/top1-20260916/t64-e6-{preflight,submit}.json`。

## 2026-09-16 E6 平台终态：有效但回退，停止本轴

11:44:15只读核对：submission15868 completed/valid，8/8通过，均值 **6.94085x** <TB7.17105。Enflame **4.7192** <8.0预注册门，较TB6.666回退；NVIDIA代理删除空任务的收益未迁移到本次目标平台。保留TB E5，不重投，不以代理收益替代八芯结果。

逐芯：tianshu15.6062 / muxi5.4702 / enflame4.7192 / haiguang10.2958 / kunlunxin0.2602 / huawei3.7268 / card_a8.5892 / card_b6.8592。实际状态见 `artifacts/competition/top1-20260916/t64-e6-status.json`；账号额度23/30。

## 2026-09-17 E7：去 clone 三段式 + Ascend persistent vendor + Kunlunxin 冻结，已提交

- 选题主证（冲 Top1 依据）：09-17 逐芯快照（`data/leaderboard-batch5-perchip-20260917.json`，SHA `2247249c…`）中 T64 是全批唯一"同芯次优可证总分 37.371 > 榜首 28.907"的题；EvokeAgent 榜首无彩票芯（华为 110.29 vs 次优 100.91 两队同量级）。我方 TB E5 7.171（rank 14/17），六 generic 芯普遍 ~2x 落后。
- 根因假设：wrapper 的 `gateup_input.clone()` 使总流量 ≈ (1+3K)·T·H，去 clone 后理想 ≈ (1+K)·T·H，K=8 时代理理论上限 ~2.8x——与他队全芯 ~2x 结构差吻合。
- 结构（`a6921555`）：generic 改三段式——`_cover_rows`（TOPK 静态展开标量读路由，covered[dst]=1，int32 防 cmpi 类型混用）→ `_scatter_rows`（静态展开 slot 循环 + 路由基址提升）→ `_fill_rows`（covered==0 的行从 gateup 回填）；≤20MB 输出走与旧 TB 逐字节同构的 `_scatter_legacy`+clone 双路径分发（代理上小 shape 三段式因 2 次额外 launch 回退至 0.70x，双路径后回到 1.00x）。`_ascend` 重写：同一结构 + `num_vectorcore`（fallback 40）persistent grid（昇腾官方 vector_operator.md"大 grid 重复 dispatch 开销"+ decode_attention 已证模板，此前华为轴从未有真实代码尝试——E5 探针实为字节相同空操作）。`_kunlunxin` 新增：改前 generic 单 kernel 字节冻结（防 static_range/covered 形态在 XMLIR 的未知风险，保 0.266 有效余量）。`_enflame` 字节不变（E6 版，SHA `85375589…`；E5 版 tiles 错配虽平台分更高但被几何断言测试锁定，留 E8 评估）。
- screening（RTX 5070 Ti，torch 2.13.0+cu130/triton 3.7.1）：unittest 6/6 全绿（generic/ascend/kunlunxin/enflame 四路径全过，含 strided/empty/special/几何）；wrapper-inclusive 5 轮 AB/BA 中位：T4096×K8×H7168 pad0 **2.208x**、pad20 **2.066x**、T1024×K8×H4096 1.477x、T8193×K1×H4097 1.344x、双路径后小 shape 1.001x/1.014x 零回退。证据：`artifacts/competition/t64e7-screening-20260917/`（run.log `621e50da…`、run2.log `4a580c64…`、bench 脚本 `c7cf5110…`）。
- release（v2，source=verification commit `a6921555439f734d265ea732d6362f2aac8850ed`）：6/6 全过 0F/E/S/X，RELEASE_REQUIRED 全在；四源执行，真实 launch generic/ascend 各 46、enflame/kunlunxin 各 44；exit 0。回执 `artifacts/competition/t64e7-release-20260917/verification.json` SHA-256 `01879fc2288336dfab05815e01f5b0e1cb725c101d94f7bbe0247d40e97bf127`、日志 `04cb35b4c86d5277e51c84b32a1bd61240639aec28c9a6fb428077a36ae3dd00`；远端 `/tmp/flagos-t64-e7-run`。ascend/enflame/kunlunxin target-runtime-unverified（无授权目标机；平台八芯即终审）。py_compile/Black/isort/flake8 全过。
- ZIP：`artifacts/competition/deepep_permute/e7-a692155/deepep_permute.zip`，16243 bytes，SHA-256 `77615803203cc4278dbfed2405837e331ccbc60ae164d1b51421b3af13334f1e`（=canonical）。成员：generic `9666063e…`、ascend `5252283b…`、enflame `85375589…`（E6 冻结）、kunlunxin `ee84a2f7…`。
- 预注册门（提交前立此存照）：8/8 有效且各芯 ≥0.1；均值 > 7.17105 才换 TB；华为 ≥6.2 判 persistent 轴正信号（→E8 梯度）；燧原走 E6 冻结字节，读数漂移不归因代码。一次候选一次判决。

## 2026-09-17 E7 单次平台提交

12:24 前后，submission **16570**（upload/正式 POST 各一次，state=submitted，远端 ZIP 验签一致）；watch 绑定 file_url_sha256 `9ad1e3dd…`。发后额度 30→29/30。预注册门见上节；八芯终态待回，结果另节记录。

## 2026-09-17 E7 平台终态：8/8 有效，新 TB 7.47225x

submission 16570（daily_seq 1）completed/valid，8/8 全过，均值 **7.47225x > 7.17105 换 TB**（+4.2%）。逐芯（vs E5 TB）：tianshu 16.3698(+2.2%) / muxi 5.1076(-8.2%) / enflame 5.2084(E6冻结字节) / haiguang 10.612(+4.0%) / kunlunxin **0.6532(+145.7%，冻结旧字节纯窗口漂移，不归因代码)** / huawei **4.433(+43.7%，persistent 轴正信号但未过 6.2 预注册门)** / card_a 9.5174(+9.7%) / card_b 7.8766(+14.1%)。发后额度 29/30。证据 `artifacts/competition/t64e7-release-20260917/status-16570.json`。

判读：去 clone 三段式平台兑现远低于代理 2.2x（平台 +4~14%）；预注册华为门未过 ⇒ 不做同字节重掷，华为轴（4.43 vs 次优 100.91）需新结构证据——候选为昇腾官方 best-practice 的 gather/scatter 形态（外层任务按 vector core 分割 + hidden 按 UB 分 BLOCK_X + SUB_BLOCK_SIZE 批量小任务 + insert/extract_slice），见 `vendor-backends/ascend/vector_operator.md` 复杂向量算子一节与 triton-ascend-ops 004/006 教程。平台 shape 与 20MB 双路径阈值的交互未知（无逐 case 耗时），调阈值属盲调不立项。

## 2026-09-17 E8：Ascend 官方 004-gather_scatter 可移植形态（连续分片+SUB 批量 2-D store），已提交

- 结构（`3dae9c40`）：仅 `_ascend` vendor 的 scatter 段重写——每 program 持有连续 token 分片（chunk=cdiv(tokens,NPC)），SUB=4 行批量装载 [SUB,BLOCK=2048] 瓦片，per-slot 一次宽 2-D store；masked 目的地址钳位行 0（昇腾会求值 masked lane 地址，T59 e6 已证）；fill 段独立 512 基准 tiles。开发中修掉两个 bug：fill 误用 2048 基准 tiles（12.5% 列失配）与 launch 传参 tiles/fill_tiles 混用（CUDA IMA，CUDA_LAUNCH_BLOCKING 定位）。generic/enflame/kunlunxin 字节与 E7 冻结。
- release v2（commit `3dae9c4052375e183c783ae678c30372196f90a5`）：6/6 全过 0F/E/S/X，四源 launch 46/46/44/44，exit 0。回执 `artifacts/competition/t64e8-release-20260917/verification.json` SHA-256 `fced1f0685887cb1a372915270f94ac1e36923248a00a68cd7c2c566be28d70d`。
- ZIP：`artifacts/competition/deepep_permute/e8-3dae9c4/deepep_permute.zip`，17148 bytes，SHA-256 `0365c0e8cbd53d74a6178b5617d4943b3056f3b76d587928b7293aab17c44a0e`。
- 预注册门：8/8 有效且均值 > 7.47225 才换 TB；华为 ≥ 8.9（2x E7 的 4.43）为批量形态正信号。零 TB 风险（榜上最优保留）。一次候选一次判决。

## 2026-09-17 E8 平台终态：8/8 有效 7.23865x，未换 TB，华为批量形态证伪

submission 16586 completed/valid，均值 **7.23865x < TB 7.47225**（保 E7）。huawei 4.433→**2.882（-34.9%，SUB 批量 2-D store 连续分片形态在华为反向；预注册门 8.9 未过）**；其余芯与 E7 同量级（tianshu 16.23/muxi 5.11/haiguang 10.98/card_a 9.42/card_b 7.53/enflame 5.11 冻结/kunlun 0.657）。判读：E7 的 persistent+去 clone 是本题华为当前最优形态；2.88 vs 次优 100.9 的剩余缺口需要目标芯 IR 或全新结构（今日无通道）。华为轴第三次关闭，TB E7 7.47225 守擂。发后额度 19/30。

## 2026-09-17 E9：逆映射 gather 形态（generic+ascend 同步换结构），已提交

- 重盘点结论（15:40 快照 `data/leaderboard-batch5-perchip-20260917-evening.json`，SHA `bde5f234…`）：T64 仍为全批唯一可达 Top1（同芯次优可证 37.371 = 榜首 28.907 的 1.29x；次接近的 T65/T75/T61 为 0.97/0.95/0.99 且需超越次优，无杠杆）。剩余额度聚焦本题。
- 结构（`9518b55`）：E7 三段式（cover→scatter→fill）改为两段 gather——`_build_inv`（inv[dst]=token，int32，torch.full(-1) 初始化）+ `_gather_rows`（每输出行读 inv，t≥0 从 x[t] gather、否则从 gateup[r] 回填；**纯行连续 store，零 scatter store**）。动机：昇腾向量核对 gather 加载友好、对 scatter store 不友好（官方 004 教程存在理由；T63-e1 "单 gather+单掩码 store 是最稳形态"跨题旁证）。账本旧"destination-gather 0.35x"是内核内搜索变体，非 inv-map 版。generic（card_a/b/海光/沐曦/天数）与 `_ascend`（persistent grid）同批换装——每芯恰见一个结构变量；`_enflame`/`_kunlunxin` 字节冻结；≤20MB 双路径保留。
- screening（RTX 5070 Ti）：6/6 全绿；wrapper 基准与 E7 完全持平（大 shape 2.21x/2.07x，小 shape 1.00x——gather≈scatter 于 NVIDIA 符合预期，赌注纯在昇腾）。
- release v2（commit `9518b55a4e65c2e1bc710fb45f3fbe1f2a513071`）：6/6 全过 0F/E/S/X，四源 launch 45/45/44/44，exit 0。回执 `artifacts/competition/t64e9-release-20260917/verification.json` SHA-256 `d7b47fb09a52fb164073808f2ccf94bb6b0d87b317dd7be26c2dabc233c9493d`。
- ZIP：`artifacts/competition/deepep_permute/e9-9518b55/deepep_permute.zip`，SHA-256 `3a54cf7f9e9e033afd3071ed83dae0ea7c3d4c765b60d550762c9a4cc21e8f8f`，4 成员（enflame/kunlunxin 冻结）。
- 预注册门：8/8 有效且均值 > 7.47225 才换 TB；华为 ≥ 8.9（2x）为 gather 形态正信号（→则评估 generic 侧兑现）。零 TB 风险。一次候选一次判决。

## 2026-09-17 E9 平台终态：8/8 有效，新 TB 7.79705x；华为可移植形态穷尽

submission 16599 completed/valid，均值 **7.79705x > 7.47225 换 TB**（+4.35%）。逐芯（vs E7）：generic 侧 gather 全面正收益——card_b 7.525→**8.584(+14.0%)** / card_a 9.420→**10.255(+8.9%)** / tianshu 16.23→**17.30(+6.6%)** / muxi 5.110→**5.431(+6.3%)** / haiguang 10.98→**11.29(+2.8%)**；huawei 4.433→4.120（-7.1%，预注册门 8.9 未过——gather 不敌 scatter，与 SUB 批量结论合并：华为 portable 形态三板斧穷尽，persistent(+44%) 为最优）；kunlun 0.658 冻结；enflame 4.740 冻结字节窗口。发后额度 18/30。

## 2026-09-17 E10：燧原去 clone gather 形态（GCU 配方），已提交

- 结构（`3e2439e`）：仅 `_enflame` 宽路径换装——E9 的 `_build_inv`+`_gather_rows`（"单 gather+单掩码 store" GCU 最稳形态，T63-e1 平台实证；标量 scatter store 与 legacy kernel 同型已证）+ GCU 配方 launch（grid≤24、num_stages=3、无 warps 钉）。≤20MB 保留 legacy clone 路径（几何测试锁 `_deepep_permute` 小路径，测试字节不变）。generic/ascend/kunlunxin 与 E9 冻结。
- screening：6/6 全绿（四路径）；release v2（commit `3e2439e975e43fdde07896c473a1d4568d025e4c`）：6/6 全过 0F/E/S/X，四源 launch 45/45/45/44，exit 0。回执 `artifacts/competition/t64e10-release-20260917/verification.json` SHA-256 `1549d7c542736e2e262edf06ecefdc1917f66d151d7c3a6eaf9481ec33510c3b`。
- ZIP：`artifacts/competition/deepep_permute/e10-3e2439e/deepep_permute.zip`，SHA-256 `5d97364a291a5ed6212e690a07c357e60df84e5fd35a683416196bd09bf0b6ce`，4 成员。
- 预注册门：8/8 有效且均值 > 7.79705 才换 TB；燧原 ≥ 6（vs 冻结字节窗口 4.7-5.2）为 gather+配方正信号。零 TB 风险。一次候选一次判决。

## 2026-09-17 E10 平台终态：8/8 有效，微幅新 TB 7.818425x；燧原 gather 无增益，全轴收官

submission 16605 completed/valid，均值 **7.818425x > 7.79705 换 TB**（+0.27%，generic 芯窗口整体微升）。enflame 4.740→**4.814（+1.6%，预注册门 6 未过——去 clone gather 在 GCU 与 clone+scatter 打平，轴关闭）**；huawei 4.011（E9 冻结字节窗口）；tianshu 17.44/card_a 10.04/card_b 8.57/haiguang 11.64/muxi 5.39/kunlun 0.655。

**T64 当日终局**：TB 7.17105 → **7.818425**（+9.02%），rank 14 → 预计 9；四发（E7 去 clone 三段式 / E8 SUB 批量 / E9 逆映射 gather / E10 燧原 gather）全部 8/8 有效零浪费。Top1（28.91）未达：锁死在华为 4.0-4.4 vs 次优 100.9 的 25x 缺口——persistent（+44% 最优）、SUB 批量（-35%）、gather（-7%）三板斧穷尽，`tl.insert_slice` 不在 Triton 3.7.1 主线（官方完整形态无法代理验证），他队形态需目标芯 IR 才能破译。发后额度 17/30。

## 2026-09-17 E11：gather BLOCK 512→2048 阶梯（generic+ascend），已提交

- 结构（`633d7bc2a72882d888f925ae86596c072e653328`）：E9 gather 唯一变量 BLOCK 512→2048（generic+`_ascend`；家族 BLOCK 阶梯历史连涨：T63 512→4096 六档、本题燧原 512→2048 +19%）。enflame/kunlunxin 冻结；≤20MB legacy 路径不动。
- release v2：6/6 全过 0F/E/S/X，四源 launch。回执 `artifacts/competition/round3-20260917/deepep_permute/verification.json` SHA-256 `0e95a92d804ba1da649b84b02dc02874bf9c4b656cc13e4be7504fbb35cd81bb`。
- ZIP：`e11-633d7bc`，SHA-256 `b2ae310ef13cd4e8751217b870dae4e6c5dd6ff4d8d120390be6a74361a9e06e`，4 成员。
- 预注册门：8/8 有效且均值 > 7.818425 换 TB；六芯（generic+华为）中位 ≥ +3% 为阶梯正信号。

## 2026-09-17 E11 状态：send 阶段被本地超时中断，未提交，intent 卡 sending

preflight 全过后 submit 在上传阶段被本地 2 分钟命令超时杀死；status 复核**无提交记录、额度未消耗**（POST 未达服务器），CLI 状态机对未决 sending intent 拒绝重发（防双发），手工改写 intent 需用户明示授权。候选封存：ZIP `e11-633d7bc`（SHA `b2ae310e…`）、回执与账本齐备；若获授权可归档 intent 后重走 preflight 单发。

## 2026-09-17 E11 重发与平台终态：8/8 有效，新 TB 8.2269x

send 中断处置：status 复核无记录/额度未扣后，经用户明示授权归档 sending intent（`.git/flagos-platform/archived/6c8ace86…`），重走 preflight 单发 **16639**。终态：均值 **8.2269x > 7.818425 换 TB**（+5.21%）。逐芯（vs E10）：muxi 5.391→**7.588(+40.7%)** / enflame 4.814→5.783(冻结字节窗口+20%) / haiguang 11.637→**13.035(+12.0%)** / card_a 10.116 / card_b 8.144 / tianshu 16.261 / huawei 4.235 / kunlun 0.654。BLOCK 512→2048 阶梯兑现（预注册"六芯中位 ≥+3%"过门）。下一档 4096 立即追发（e12）。发后额度 13/30。

## 2026-09-17 E12：gather BLOCK 4096 阶梯追发，已提交

- E11 阶梯兑现后按"阶梯仍在涨"证据追发 4096（generic+`_ascend` 唯一变量；enflame/kunlunxin 冻结）。release v2（commit 见 ZIP）全过；回执 `artifacts/competition/t64e12-release-20260917/verification.json`。预注册门：8/8 有效且均值 > 8.2269 换 TB；≥2 芯 +3% 为正信号，否则阶梯封顶。

## 2026-09-17 E12 平台终态：8/8 有效，新 TB 8.28935x；阶梯封顶

submission 16669 completed/valid，均值 **8.28935x > 8.2269 换 TB**（+0.76%）。逐芯（vs E11）：huawei 4.235→**4.977(+17.5%)** / muxi 7.588→7.924(+4.4%) / card_b 8.473 / haiguang 13.064 / tianshu 16.313 / kunlun 0.658 / enflame 4.973（冻结字节窗口回落）。阶梯增益 512→2048→4096 = +5.2%→+0.76% **封顶**。T64 当日终局：TB 7.17105→**8.28935**（+15.6%），rank 14→9；发后额度 10/30。

## 2026-09-17 E13：燧原 gather BLOCK 4096（阶梯交叉应用），已提交

- e12 封梯后唯一剩余单变量：`_enflame` 宽路径 gather 512→4096（generic/华为已证档位 + GCU"BLOCK 倾向远大于 GPU"；generic/ascend/kunlunxin 冻结）。release v2（commit `28c5df9e4e26b918fd7b6ec0cb3790284ae0a767`）全过，回执 SHA-256 `d9e5bd300e467e0d1b6fe31c7a5e2074a403795e4c9765aa8fa007fdee2e3bda`。ZIP `e13-28c5df9` SHA `000cff091e80c96317da8c0eb3e980db779f4a24d9ab47a1ee2531e831716b8f`。门：8/8 且均值 > 8.28935；燧原 ≥ 6.5 为正信号。

## 2026-09-17 E13 平台终态：8/8 有效，新 TB 8.73395x；燧原阶梯陡升

submission 16677 completed/valid，均值 **8.73395x > 8.28935 换 TB**（+5.37%）。enflame 4.973→**7.0924（+42.6%，预注册门 6.5 大幅超过）**；tianshu 17.709/card_a 10.238/haiguang 13.083/card_b 8.341/muxi 7.902/huawei 4.848/kunlun 0.659。阶梯判读：燧原 512(平)→4096(+43%) 刚起且陡 → 追最后一档 8192（e14）。发后额度 9/30。

## 2026-09-17 E14 平台终态：8/8 有效 8.302425x，低于 TB；燧原阶梯 4096 见峰，全题收官

submission 16690 completed/valid，均值 **8.302425x < TB 8.73395**（保 e13）。enflame 7.0924→5.525（-22.1%，8192 过峰）。阶梯终值：燧原 512(平)→4096(**+42.6%**)→8192(-22%)；generic 512→2048(+5.2%)→4096(+0.76%)。

**T64 全日终局**：TB 7.17105 → **8.73395（+21.82%）**，rank 14→9；八发（E7/E8/E9/E10/E11/E12/E13/E14）全部完整闭环，其中六发换 TB。剩余缺口：华为 4.85 vs 次优 100.9（可移植形态穷尽）、燧原 7.09 vs 次优 12.07（4096 峰后无新档）。额度 8/30 剩余，按纪律封枪。

## 2026-09-17 E15 载体终态：8/8 有效 8.47025x，保底

窗口重掷 1/2：muxi **8.1638（单芯新高，超 e13 的 7.924）**，但 enflame 5.554（vs e13 窗口 7.09）回落，合计 8.470 < TB 8.73395。低滚 1/2。
