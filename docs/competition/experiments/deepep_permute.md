# Task 64 `deepep_permute` 实验记录

```current
task: 64
operator: deepep_permute
batch: 5
validity: valid
platform: submitted(13375,e2=C1,7/8已判;仅昆仑回调;燧原+5.5%未达1.3x门)
candidate_stage: e2
team_best_stage: e1
team_best_speedup: 6.7478
sealed: no
next: e2（燧原 grid 封顶 24）已发射；预注册燧原 ≥1.3x（目标轴 2.53→6.68 次优，值均值 +0.371）
updated: 2026-09-12
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
