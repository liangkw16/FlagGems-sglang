# Task 63 `create_flashinfer_kv_indices` 实验记录

```current
task: 63
operator: create_flashinfer_kv_indices
batch: 5
validity: valid
platform: submitted(13417,e7,评测中;TB e5 189.397x)
candidate_stage: e7
team_best_stage: e5
team_best_speedup: 189.39746875
sealed: no
next: e5 valid 189.40 新 TB(+43%,门 1.43x 过)；榜差 20；e6=三 vendor 刷新各自几何+无 clone（沐锡 +7.5/海光 +9.6/燧原 +2.6 潜力）
updated: 2026-09-12
```

> 下方 S0 开发记录是 2026-09-10 快照；当前平台结果见 CURRENT 和文末提交记录。

## 契约与范围

- 完整题面：[Task 63](../tasks/batch-5/63-create_flashinfer_kv_indices.md)；[开发契约](../strategy-batch5.md)。
- exact；八芯每芯 0.1x；截止 2026-09-17 19:59:59（北京时间）。
- 按公开 reference 返回新张量并保持输入不变；核心计算使用 Triton，无 fallback。
- generic S0；NVIDIA 代理已验证，平台及其他目标 runtime 标记 `target-runtime-unverified`。
- 本轮未执行平台 preflight、上传或正式提交，不消耗额度。

## 不可变身份

- source commit：`4836fe0378d6290cbcaa91958693e13ed1acdf3e`。
- verification commit：`4836fe0378d6290cbcaa91958693e13ed1acdf3e`。
- ledger commit：本文件的 Git 提交；用 `git log -1 -- docs/competition/experiments/create_flashinfer_kv_indices.md` 定位。
- source SHA-256：`0507dd1780757b11f1f253699c7927c63f915631f8d0b11c5ad7065fe3271688`。
- test SHA-256：`3a5324770f4298207e893d77412fb65ac51a4926887c558b87ddfaf775ae5642`。
- ZIP：`artifacts/competition/create_flashinfer_kv_indices/s0-4836fe0/create_flashinfer_kv_indices.zip`，2782 bytes，仅含 `create_flashinfer_kv_indices.py`。
- ZIP SHA-256（等于 canonical）：`a72813fa2eed3767f2f79a3a6d3fa2d9a3aad07ffc1186e04d944ed6f7bf9489`。
- dry-run manifest、最终构建、existing 验签、Git blob 逐字节比对及 unzip -t/-l 全部通过。

## 验证证据

- 4 个测试方法 / 25 条 test/subTest 记录 / 13 次实际 kernel launch；失败、错误、skip、xfail 均为 0。
- RTX 5070 Ti；Python 3.12.13；PyTorch 2.13.0+cu130；Triton 3.7.1；CUDA 13.0。
- 回执：`artifacts/competition/batch5-development-20260910/release/create_flashinfer_kv_indices/verification.json`。
- 回执 SHA-256：`7c46aa0c45e86bfb1b849edf5301d7350136e2046f278aa59608fbbc4c5b711d`。
- 完整日志：`artifacts/competition/batch5-development-20260910/release/create_flashinfer_kv_indices/verification.log`。
- 日志 SHA-256：`7a4d6b0e6d83d8fe9c0d8512b9517afc3ac8e45a69536f1c8a1836c7830dcd3e`。
- benchmark：`artifacts/competition/batch5-development-20260910/release/create_flashinfer_kv_indices/benchmark.json`，SHA-256 `607b5c74a7ddd46cedfa7a359b3da49a63ef3e6e88dbc1d908aeb08347c23f4d`。
- Black/isort/flake8、Python 编译、空输入、边界、stride 和输入不变性均已检查。
- 所有 source/test/helper/runner/benchmark 哈希见[证据清单](../data/batch5-development-20260910.json)，清单 SHA-256 `74b37287f5649dc044696565306c97111117c991b3549393a88bcb703eba46b1`。

## 代理性能

<!-- ponytail: 每请求一个 program；长段利用率成为瓶颈时再将请求拆成并行 tile。 -->

| 代理负载 | 五轮配对 speedup 中位数 | 候选中位耗时 |
| --- | ---: | ---: |
| 1x8193 | 2.7824x | 12.375 μs |
| 32x1024 | 112.9101x | 7.566 μs |

测速含 wrapper；五轮交替 AB/BA，每次 warmup 20ms / rep 50ms。
这些是两种代理负载的结果，不是平台平均分；原始样本已保留。

## 重放

远端隔离目录：`/tmp/flagos-batch5.POXFt3/create_flashinfer_kv_indices`；复跑需 prepare 到新目录，旧回执不可覆盖。

```bash
python .agents/skills/flagos-operator-race/scripts/verify_release.py prepare create_flashinfer_kv_indices --source-commit 4836fe0378d6290cbcaa91958693e13ed1acdf3e --verification-commit 4836fe0378d6290cbcaa91958693e13ed1acdf3e --dependency tools/benchmark_batch5.py --directory /tmp/NEW_RELEASE_DIRECTORY
timeout 600 /home/kevin/notebook/.venv/bin/python /tmp/NEW_RELEASE_DIRECTORY/.agents/skills/flagos-operator-race/scripts/verify_release.py run --directory /tmp/NEW_RELEASE_DIRECTORY
```

## 2026-09-11 一轮优化后的待提交候选

- 候选 `e1`；按 tile 调度优化晋级 E1。首次提交前候选就绪。
- source commit：`29754c1dfcbc87c9b193e750145b668c8e35d852`；verification commit：`29754c1dfcbc87c9b193e750145b668c8e35d852`。
- ZIP：`/Users/bytedance/ccc/flagos/artifacts/competition/create_flashinfer_kv_indices/e1-29754c1/create_flashinfer_kv_indices.zip`；2992 bytes；成员 `create_flashinfer_kv_indices.py`。
- ZIP SHA-256：`6d9d37ebf39521f46697f8dc1b725c4bd6ad18e21c3fe7b6bc22e48bdb6c0ac8`。
- 源码 SHA-256：`ce7c07d20f2a4c585e27757e13f2efb32c8069f08767aacc4bbca1647c2bbe96`。
- 测试 SHA-256：`6a353892e59c29ca98efc7bfa4225c3f98d6424b85571cd7921a709aa95c5809`。
- 回执：`artifacts/competition/batch5-optimize-20260911/release/create_flashinfer_kv_indices/verification.json`；SHA-256 `4e5229ce738bac8b177f44295dcdb7b78f5da10f851176f8d2462106276c96ba`。
- 日志：`artifacts/competition/batch5-optimize-20260911/release/create_flashinfer_kv_indices/verification.log`；SHA-256 `04e747df25c1e016e2a5586f05fcd04942ef7912bcdebe0539d538ab2f02ce6f`。
- 完整 release：4 个测试方法 / 28 条记录 / 16 次 kernel launch，全通过；NVIDIA 代理范围。
- 优化和复现见[本轮报告](../optimization-batch5-20260911.md)；[完整证据](../data/batch5-optimization-20260911.json)，SHA-256 `87a560dd86ab7e9dfb9607f8f05a007060d48b3aab876b165bfc0fec591f77db`。

## 2026-09-11 首次平台提交（2026-09-11T03:39:21.318114+08:00）

- `e1` / submission `12894` / daily_seq `1`，提交于 `2026-09-11T03:24:44+08:00`。每候选上传与正式 POST 各一次，无自动重试。
- nonce：`354996b527cb2509ad98954ec2c8a424`；上传 URL SHA-256：`7169d86d386524a1ee8791689f55018b74d9eb46b7bcebc4424cd0166c5c49ad`。
- 远端 ZIP 回读 verified，2992 bytes，SHA-256 `6d9d37ebf39521f46697f8dc1b725c4bd6ad18e21c3fe7b6bc22e48bdb6c0ac8`，与本地候选逐字节一致。
- 平台 `completed` / `valid`；正确性通过 8/8，完成 8/8；121.8698125x；第 5/5 名。
- 当次 status 额度：24/30，已用 6；observed_at `2026-09-11T03:39:21.318114+08:00`。排名查询时间 `2026-09-11T03:39:41.445754+08:00`。

| 芯片 | 状态 | 正确性 | speedup |
| --- | --- | --- | ---: |
| tianshu | completed | 通过 | 285.797 |
| muxi | completed | 通过 | 60.2795 |
| enflame | completed | 通过 | 21.708 |
| haiguang | completed | 通过 | 179.8935 |
| kunlunxin | completed | 通过 | 2.88025 |
| huawei | completed | 通过 | 66.19575 |
| card_a | completed | 通过 | 168.8675 |
| card_b | completed | 通过 | 189.337 |

下一步：保留 E1；后续优先定位昆仑和 wrapper 耗时。

[提交结果证据](../data/batch5-submissions-20260911.json)，SHA-256 `63418b87e9249bef72751df2a1778c52e6d679a5d313ecf18d8ed64b96c175dd`；原始 status `artifacts/competition/batch5-submit-20260911/63-status-033921.json`，SHA-256 `6d245cc1336493d5d89fc97f9db0c3651e0c04798581ebf001de196c5355ee9c`。

## 2026-09-11 E2：BLOCK=256 + 目标 ~512 协同 program（8/8 新 team best）

- 上游 SGLang AMD PR #37659 形态（2D (batch, token-block) 协同跨步、目标
  ~512 program、空块零迭代早退）+ Codex 建议合并：BLOCK 512→256，
  splits 上限 32→512（`512 // batch`），代理 L2 域不可分辨（全 11.7μs
  launch-bound），平台大芯片可分辨。
- 终态（daily_seq 25，09:07）：**8/8，平均 132.0990x 新 team best（+8.4%）**：
  天数 337.699（+18%）/ 海光 199.39 / A 183.87 / B 202.23 / 华为 67.09 持平；
  代价：沐曦 60.28→55.05、**燧原 21.71→9.01（-58%）**、昆仑 2.88→2.45。
- ZIP `e2-1c0381c`，3206 bytes，SHA-256
  `2b2c4f42248b534c38a1cced7edbcda2826cf53ec686b7bbded1d4cc368434ee`；
  回执 `batch5-deepopt-20260911/release/create_flashinfer_kv_indices/verification.json`
  （SHA-256 `09f7a29a0444f7724a8d4753f8e52ef4a0ed4c10d5605c1c1e21b85db566de45`）。
- 后续：BLOCK=256 对窄带宽芯（燧原/沐曦/昆仑）回退、对大芯片增益——
  下一候选 `_enflame`/`_kunlunxin`/`_metax` vendor 保 BLOCK=512+旧 splits，
  预期均值 ≈137+（GuanghuLab 135.6 可超）。

## 2026-09-11 E3 候选就绪（窄带宽 vendor，待下一额度窗口）

- `_enflame`/`_kunlunxin`/`_metax` vendor 冻结 E1 已证字节（BLOCK=512、
  splits≤32），恢复 E2 回退的窄带宽芯（预期燧原 9.0→21.7、昆仑 2.45→2.88、
  沐曦 55.1→60.3），大芯片保留 E2 新形态——预期均值 ≈134.4（vs E2 132.1）。
- source commit `95e5292…`；ZIP `e3-95e5292`，13416 bytes（4 成员），
  SHA-256 `ebb05e6a8353e5c08b0fddf4b37dce4451c479b65a7b9f72d094b8a7cb540486`。
- 回执 `release-next/create_flashinfer_kv_indices/verification.json`（SHA-256
  `814eb5c6ae5263fa0a2a67b453769c33884b7f80e4566c0dc007966d3a04ad90`，
  4 方法 0 失败，generic+3 vendor 各 16 launch）。
- 状态：候选就绪未提交。

## 2026-09-11 R2 待开发候选：去掉 wrapper 的 `kv_indices.clone()`（预注册门，未开发）

- 假设：E1 对榜首 Nectar 的逐芯比值八芯里六芯挤在 **0.49~0.63**
  （天数 0.559 / 沐曦 0.516 / 燧原 0.491 / 海光 0.628 / 国际 A 0.630 /
  国际 B 0.616，例外华为 0.959）。跨芯一致的比值是**加性固定开销**特征，
  而非算法差距。wrapper 的 `out = kv_indices.clone()` 流量为"读 N + 写 N"，
  与紧随的 gather kernel 同量级，相加约为纯 kernel 两倍 ⇒ 预测比值 0.5。
  旁证：`1c0381c` 为 T64 实测 "clone ceiling confirmed at 30-53%"。
- 未写区的正确形态：参考实现与测试把 `kv_indptr[i+1]` 构造成
  `kv_indptr[i] + len_i + 3`、`old` 长度 `pointers[-1] + 5`，所以未写区是
  **每两行之间 3 个元素的空隙 + 末尾 5 个**，不是连续尾巴——"拷贝尾部"的写法
  是错的。每行 program 已知道 `[indptr[r], indptr[r]+len_r)` 与 `indptr[r+1]`，
  由同一 program 再拷 `[indptr[r]+len_r, indptr[r+1])` 即可；末行补到
  `out.numel()`，另补 `[0, indptr[0])`。生产形状空隙为 0，补齐代价为零，
  而 clone 的 2N 流量被整体省掉。产出用 `torch.empty_like`（测试断言
  `data_ptr` 不同）。该改法**不改访问模式**，这是它与 T64 被证伪的
  destination-gather 重写的本质区别（后者代理 0.35x）。
- 预注册门：代理 wrapper-inclusive 实测 clone 占比 **≥25%**（任一代表负载
  `1×8193` / `32×1024`）；完整正确性全过；spill/shared memory 为 0；
  寄存器 ≤64。预期占比 30~50% 时均值 132.099→**185~260**。
- 止损：实测占比 <25% 直接关闭 clone 轴，不投平台。
- 同候选须一并处理：E2 起的 generic `splits = min(cdiv(width,256),
  max(1, 512//batch), 512)` 落在 grid 轴 1，而燧原 `grid.y` 硬限 255；
  `batch ≤ 2 且 width ≥ 65536` 时 splits 可达 256~512 会越界（E2 当次形状
  未触发，隐患对仍走 generic 的五芯持续存在；E3 三芯 vendor 冻结 E1 字节
  恰好绕开）。建议改用上游 `sgl-project/sglang#37659` 的
  `max(1, min(ceil(width/8192), 512 // max(1, batch)))` 并加 `min(..., 255)` 守卫。
- 证据等级：clone 占比与 2x 预测为**假设**，先做零额度代理测量再开发。
  详见 [r2 §2](../optimization-batch5-r2-20260911.md)。

## 2026-09-12 E3 平台结果（submission 13227，daily_seq 3）

- **8/8 valid，均值 131.9834**（未超 E2 team best 132.099，差 0.1%）。
- 逐芯：天数 337.5025 / 沐曦 53.654 / 燧原 **19.608**（E2 9.01 → 恢复 ✓，
  E1 为 21.71）/ 海光 201.1373 / 昆仑 2.7815（E2 2.45↑，E1 2.88）/
  华为 65.3153 / A 181.3925 / B 194.4765。
- 复盘：燧原/昆仑 vendor 冻结兑现；**沐曦（metax vendor）未回 E1 水位**
  （53.65 vs E1 60.28、E2 55.05）——同字节跨窗差异 ~12% ⇒ 沐曦差主要
  是窗口方差而非形态，"≈134.4" 预测中沐曦 +5 的成分不成立。A/B 亦较
  E2 让 1~4%（同因）。
- 结论：E3 与 E2 等价（均值差在窗口噪声内），team best 保留 E2 字节。
  下一主轴回到 R2 方向 A（去 wrapper clone，代理先测占比），不再在
  vendor 形态轴上加码。
- 额度：发后 27/30。

## 2026-09-12 clone 轴代理测量与 AB 负结论（零额度关轴）

- 占比测量（RTX 5070 Ti，wrapper-inclusive）：`1×8193` clone 3.0µs /
  全 11.7µs = **25.9%**；`32×1024` = **26.7%**——名义通过预注册 ≥25% 门。
- 但 e4（去 clone + 空隙唯一归属 `pid1==0` + grid.y≤255 守卫，正确性
  矩阵与 4 方法全绿）对 e3 字节的同负载 AB：**1.011x / 0.998x**。
  原因：wrapper 为 launch/CPU 开销主导（11.7µs 中 GPU 工作被掩盖），
  去掉 2N clone 流量不变现。
- 处置：e4 工作树字节已回退、不投平台；预注册门的目的是预测 e4 收益，
  AB 直接测得 ≈0 即按止损关闭。仅在目标芯出现 kernel-GPU-bound 证据
  （逐芯 exec_ms 与 kernel 流量强相关）时重开。

## 2026-09-12 E5：clone 轴重开（带宽域门通过，候选就绪后提交）

- 重开依据：榜首 RSI 逐芯全面 1.5-1.7x（天数+23.1/海光+18.6/B+12.4/
  A+10.5/沐曦+7.4 均值贡献）——量级吻合 clone 额外读+写一遍的流量；
  R2 关闭所依据的代理 shape 为 launch-bound 域（e4 AB 仅 1.011x），
  平台大芯为带宽 bound 域，负结论不可外推。
- 带宽域门（width=131072, batch=8, gapless）：输出相等，wrapper
  B/A=0.839（≥15% 门通过）。
- 实现：`empty_like` 出参 + kernel 补拷未写区（行 0 头 [0,indptr[0])、
  行间空隙 [indptr[r]+len, indptr[r+1])、末行尾到 numel；标量选择全
  算术、补拷加载算术钳位、独立 old-stride 参数（strided 用例）、
  batch=0 回退 copy_）；splits 加 255 封顶（燧原 grid.y 隐患，R2 注册）。
  首轮 screening 抓到两真 bug（空 batch 垃圾出参、old/out stride 混用），
  修复后 4 方法 0 失败，4 源 64 launch。
- source commit：`a8db85930074b0aa9b7dca29153e2bc069f50aca`。
- ZIP：`e5-a8db859`，SHA-256 `a7e75011e69133fddac184e36c45d241520ae26454ff281d381ffc5ea8e10d14`；
  4 成员（generic `47f8f7cd…` 变化，三 vendor 不变）。
- release 回执：`batch5-t63e5-validate-20260912/create_flashinfer_kv_indices/verification.json`，
  SHA-256 `31908b089e58583dfc4e9535b5492babe2f960db6807036449f8fcfa693e72ac`；
  日志 `092b51785a7a7981c3b4fb627926625d7e8b771ced4c71faf84ed34498851b6e`。
- 预注册：带宽 bound 芯（天数/海光/A/B）中位 ≥1.15x；燧原 vendor 不回退
  破 0.1；沐曦 vendor 读数对照。

## 2026-09-12 E5 平台提交（submission 13405）

- 上传与正式 POST 各一次；state submitted，15:3x 入队。额度：发后 5/30。
- 裁决点：带宽 bound 芯（天数/海光/A/B）中位 ≥1.15x；燧原/昆仑/沐曦
  vendor 读数不回退。

## 2026-09-12 E5 平台终态：8/8 VALID 189.40x 新 team best（submission 13405）

- **八芯全过，均值 189.39746875（e2 132.099，+43%）**：
  天数 **508.2520（+50%）** / 沐曦 54.4725 / 燧原 18.2643 / 海光
  **271.0018（+36%）** / 昆仑 2.7760 / 华为 **82.7750（+27%）** /
  A **259.1422（+41%）** / B **318.4960（+57%）**。
- 预注册门（带宽芯中位 ≥1.15x）以 **1.43x** 通过；clone=额外 2N 流量的
  假设平台证实；R2 关闭判据（launch-bound 代理域）确认为错误域外推。
- 榜差 77.3→**20.0**（RSI 209.44）。剩余缺口结构：沐曦 54 vs 115
  （+7.5 均值潜力，`_metax` vendor 仍冻结带 clone 的 E1 字节）、
  海光 271 vs 348（+9.6）、燧原 18 vs 39（+2.6）；天数/A/B 已接近或
  反超。
- 下一发 e6（已注册）：三 vendor 刷新为**各自几何 + 无 clone 的 e5
  kernel**（BLOCK=512/splits≤32 保留，gap 补拷全燧原已证原语）。

## 2026-09-12 E6：窄带 vendor 刷新（候选就绪后提交）

- e5 兑现后三 vendor 仍冻结带 clone 的 E1 字节：沐曦 54.5 vs 榜首 114.5
  （+7.5 均值）、燧原 18.3 vs 39、昆仑 2.78 vs 4.16。
- E6：e5 kernel（gap 补拷/算术选择/钳位 load/独立 old-stride/空 batch
  回退）装入各 vendor 自证几何（BLOCK=512、splits≤32、128//batch）；
  generic 字节不动。
- source commit：`0054ddccf1832c983f5a4507fe37300ebfcf1195`。
- ZIP：`e6-0054ddc`，SHA-256 `d97b9196173c93b3d83d4f8b6432582ff4ef79a466b3fbb5dbe6a930d655aed3`；
  4 成员（generic 不变；enflame `bf738fee…`/kunlunxin `aee2e744…`/metax
  `46826627…` 刷新）。
- release 回执：`batch5-t63e6-validate-20260912/create_flashinfer_kv_indices/verification.json`，
  SHA-256 `6c76aecea75c412126862a7f186e06fd40b19149e69228c05eb91e03a8f15769`；
  日志 `a5041acec69708319e6be449e6ed420465ae7dae3dd2ecb0711b9aaed286ac34`；
  4 方法 0 失败，4 源 64 launch。
- 预注册：沐曦中位 ≥1.15x（54.5→62+）；燧原/昆仑不回退破水位；带宽
  五芯维持 e5 水位。若全兑现均值 189→200+。

## 2026-09-12 E6 平台提交（submission 13415）

- 上传与正式 POST 各一次；state submitted。额度：发后 3/30（收盘余量）。
- 裁决点：沐曦 ≥1.15x；燧原/昆仑水位；带宽五芯维持 e5。

## 2026-09-12 E6 平台终态与 E7 修复（候选就绪后提交）

- E6（13415）7/8：**沐曦 82.7060（+52%，vendor 刷新兑现）**、海光
  299.9025（+11% 窗口）、昆仑 vendor 正常判决 2.7978、天数 511.73/
  A 263.62/B 305.84/华为 65.12（窗口波动）；**燧原 PassManager**——
  e6 vendor 的 `(i*m).to(int64)` 钳位 = **i64 向量乘法**，与 T67 e1
  轮怀疑构造一致（本轮二次实证该族毒点）。
- E7：燧原 vendor 标量标志改纯整型算术（`1 - min(row,1)` /
  `1 - min(batch-1-row,1)`，去 bool→i64 cast），gap load 去钳位改裸
  masked 偏移（燧原已证形态）；generic/metax/kunlunxin 字节不动。
- source commit：`40f6d6478137468e955d7a085738fd36c7b32f88`。
- ZIP：`e7-40f6d64`，SHA-256 `50012525b0a1308556d008566b13c161e3d264f2ca7db767e0dc610547fd0d82`；
  仅 `_enflame` `357d3820…` 变化。
- release 回执：`batch5-t63e7-validate-20260912/create_flashinfer_kv_indices/verification.json`，
  SHA-256 `16660b573d43071aaad0e85346b1bf0259939e1f37ad985dfdb5ff2aae450c74`；
  日志 `0cd2af829ff86b23283fe6c73e860cadfa1dc6968e552f311dcc08c4fc19e503`；
  4 方法 0 失败。
- 预期：燧原回 18-25 水平 ⇒ 七芯(e6 读数)+燧原 ≈ 均值 194+ 新 TB；
  沐曦 82.7 仍低于榜首 114.5（后续轴）。

## 2026-09-12 E7 平台提交（submission 13417）

- 上传与正式 POST 各一次；state submitted。额度：发后 2/30（收盘存底，
  今日首发到此为止）。
- 裁决点：燧原 PassManager 解除（i64 向量乘毒点二次实证的修复）；其余
  七芯 e6 水平。
