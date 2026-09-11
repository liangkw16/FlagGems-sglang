# Task 63 `create_flashinfer_kv_indices` 实验记录

```current
task: 63
operator: create_flashinfer_kv_indices
batch: 5
validity: valid
platform: completed(12894,8/8)
candidate_stage: e3
team_best_stage: e2
team_best_speedup: 132.099
sealed: no
next: 下一窗口第三发 e3 窄带宽 vendor（预期均值 ≈134.4）
updated: 2026-09-11
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
