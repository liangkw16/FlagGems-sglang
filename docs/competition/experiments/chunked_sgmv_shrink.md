# Task 48 `chunked_sgmv_shrink` 实验记录

```current
task: 48
operator: chunked_sgmv_shrink
batch: 4
validity: valid
platform: e8(11170)=7/8已通过,昆仑待回调;最佳仍e7(10808,4.7489375x)
team_best_stage: e7
team_best_commit: 094548df5da1075b8245b4af8ccddf024a319ae3
team_best_speedup: 4.7489375
sealed: no
next: 仅查11170昆仑回调,不得重复提交;总分与Top1未确定
updated: 2026-09-08
```

## S0: 6/8（燧原+昆仑败）
## E1（燧原 route/materialize）: 7/8（燧原翻绿0.58x，昆仑败）
## E2（+昆仑 route/materialize + long）: 7芯已过含昆仑1.764x，燧原评测中

## E3-E5 燧原超时系列（2026-09-05，submissions 9986/9998/10002）
- E3（BK=512 revert）：燧原 1830s 超时
- E4（最小变更重掷）：燧原 1830s 超时
- E5（恢复 e1 原版字节）：燧原仍在评（4th 连续超时）
- **判定：燧原评测机持续繁忙（e1 同字节已过 0.58x）**；等平台侧
  窗口恢复后重投
- 七芯稳定：天数 3.74 / 沐曦 5.39 / 海光 6.09 / 昆仑 1.77 /
  华为 6.26 / A 7.04 / B 6.69

## E6 → **8/8 VALID**（2026-09-05，submission 10111，第 6 个 8/8！）

- **燧原评测机恢复，e1 字节通过 0.58x**（5 次连续超时后终于恢复）
- **8/8 valid，平均 4.7198125x**
- 逐芯：天数 3.85 / 沐曦 5.31 / 燧原 0.58 / 海光 6.00 /
  昆仑 1.79 / 华为 6.54 / A 6.98 / B 6.70
- 关键 vendor：燧原 route/materialize + 昆仑 route/materialize + long-index

## 2026-09-06 SGLang 结构两发证伪（e2/e3，submissions 均败）

- e2（c7d2d06）：SGLang 生产形态 BLOCK_M=max_len、shape 自适应
  BLOCK_N/K → 平台 5/8 correctness 败。
- e3（885f9e1）：BLOCK_M 封顶 64 重试 → 7/8 败（仅昆仑 vendor 过）。
  根因：每 program 只装一个 BLOCK_M tile（`tl.arange(0,BLOCK_M)+seg_start`），
  段长 > BLOCK_M 的 token 直接丢失 → 75% mismatch 与 max_len=256/64=4 吻合。
- 结论：该移植缺"段内多 tile 循环"；上游还依赖调用方预切短 segment。
  两个失败版本不足以否定完整分块方案，先补长段覆盖再验证。
- **generic 已回退 E6 字节（6ed1fa9）**，vendor 不动；远端回归 5/5 OK
  （含 _op_variants 矩阵）。team best 仍 e6 8/8 4.7198x。

## 2026-09-05 冲分预注册（8/8 后；本会话基于同族资产拟定，未做专项会诊）

现状：e6 8/8 4.7198x，榜首 c2flow 21.63x。弱芯燧原 0.58/昆仑 1.79/
天数 3.85。同族 T47 结论：indirect+dot 在燧原/昆仑不受支持，
route/materialize 是 sgmv 族唯一可行形态（e8-e10 三投证伪）。

候选（按序，每轴 1 发不过门即关）：
1. 燧原 0.58x：e1 字节刚过线；T12/T47 燧原 dot 模板（64 tile +
   stages2）在本题未试过——单发 64³ route/materialize 变体
2. 天数 3.85x：核对现有实现 dot 操作数 dtype（天数 fp32-dot 静默错
   执行，必须 fp16 或 split-fp16 三点积）；若已是 fp32-ieee 则试
   T12 镜像 split-fp16
3. 昆仑 1.79x：BLOCK 唯一有效轴（T21 1024 唯一成功）——BK/BN/BM
   单档扫描 ≤2 发
止损：总额度优先让给 T42/T53/T52 的预注册候选。

## 2026-09-06 流程审查修正

- 63/64/65/256 行的 segment 用例已加入 generic 与 vendor 矩阵；验证状态见本次流程修复记录。
- E6 平台最佳结果保持历史原值；长段覆盖修复前不再将该方向称为结构证伪。
- 最终 GPU release 5 tests/27 条 test/subTest 记录通过，generic 与两个 vendor 入口均实际调用。source/verification 身份及完整回执见 [流程实测](../workflow-validation-20260906.md)。

## E4 自适应 BLOCK_S（2026-09-07 已提交）

- 重开 e2/e3 方向的修正版：上游 #10286 的关键前提是生产段很短
  （调用方先切成 16 行），BM=64 对 ≤16 行段是 4 倍填充浪费。
  单变量仅 BLOCK_S 随 max_len 取 16/32/64；BLOCK_N=128/BLOCK_K=32/
  warps=4/stages=3 保持 E6 平台已验证值，(token_tile, output_tile)
  网格覆盖不变（e2/e3 败因是每段单 tile，已由 E6 结构避免）。
- NVIDIA 代理 wrapper 基准：8 行段 1.47x、16 行段 1.47x、
  (8 行段,K=4096,bf16) 1.75x；64/256/257 行段 1.00x 不回退；
  screening 5/5（含 63/64/65/256 边界）、release 37 launches 0 失败。
- source `8b134f4`，ZIP `e4-8b134f4` SHA-256
  `6141f43ce475cba01498e9cefb6e82e015c303fdc6e7c3a649d2881dedf9b5b4`；
  2026-09-07 提交评测中。目标芯 ≥15% 才晋级 team best，否则保留 e6。

## E4 终态（2026-09-07，submission 10670）

- **7/8**：燧原 1830s R 状态超时（评测机忙）；其运行的是与 e6 通过
  0.58x 完全相同的 `_enflame` vendor 字节，非代码回归，同 2026-09-05
  e3–e5 拥堵前科。team best 仍 e6 8/8 4.7198x。
- 七芯 vs e6：天数 3.77（-2%）/ 沐曦 5.35（+1%）/ 海光 6.30（+5%）/
  昆仑 1.77（-1%）/ 华为 6.76（+3.4%）/ A 7.19（+3%）/ B 6.79（+1.3%），
  总和 +1.5%——**未过 ≥15% 晋级门，自适应 BLOCK_S 轴按预注册关闭**
  （平台隐藏 shape 的短段占比低于代理假设）。
- 结构结论：完整多 tile 覆盖下自适应 BM 方向正确但收益微小；
  e2/e3 的"结构证伪"改判为"覆盖缺陷 + 低收益"，账本留档。

## E7 燧原原生低精度 dot 操作数（2026-09-07）

- 预注册 vendor 轴 #1（T12 镜像）：e6 燧原 vendor 的 GEMM 把操作数
  cast 到 fp32 后 ieee dot——恰是 T12 实证的 GCU 病理配置（ieee-fp32
  操作数 + 小 tile 低 stages，T12 当时 0.116x；fp16 原生操作数 +
  64/64 tile + stages2 → 0.743x）。单变量：bf16/fp16 输入保持原生
  dtype 进 dot（fp32 累加；bf16×bf16/fp16×fp16 乘积在 fp32 内精确，
  与 reference 的 fp32 GEMM 在低精度容差内等价），fp32 输入维持 ieee
  路径；index_select 物化同步降为原生 dtype（省一半拷贝流量）。
  generic/kunlunxin 字节冻结。
- source/verification commit `094548d`；screening `/tmp/flagos-t48e7`
  （5/5）与 release `/tmp/flagos-t48e7-rel`（5/5 方法、0 fail/skip，
  generic 15 / enflame 11 / kunlunxin 11 launch 实跑）双绿；远端
  black/isort/flake8 全过（black 重排后取回 hash 一致）。
- 预注册门：燧原 ≥0.58x 基础上兑现 T12 量级（≥1.5x）且整题 avg 超
  team best 4.7198125 才晋级；燧原评测机忙超时（1830s R 态）不计
  代码失败、不重试同字节。
- ZIP `e7-094548d`，SHA256
  `84fe1c7df7c158ce5f0965aa22562b654f31bab6b31e0217bc8774af3a9d9307`；
  成员 3（generic/kunlunxin 与 e6 一致，enflame 为新字节）。
- 证据 `validation/verification.json` SHA256
  `0cafa617d5e848eb5b5d99d432e3c6cab0568719855b0db5eb34c89fb043b36d`。

## E7 平台终态（2026-09-07T2x:xx）

- submission `10808`：**valid，avg 4.7489375，新 team best（+0.62%）**；
  quota 观测 12/30。
- 逐芯：天数 3.7705 / 沐曦 5.305 / **燧原 0.528（预注册门未达：
  T12 原生操作数模板未迁移，0.58→0.53 噪声带内；燧原轴关闭）** /
  海光 6.223 / 昆仑 1.785 / 华为 6.622 / A 7.132 / B 6.626。
- 均值增量来自未动芯小幅水位（华为 +1.2%/A +2.2%/海光 +3.7%）。
- 跨题知识：**T12 燧原 fp16-操作数 dot 模板只在融合 kernel 形态
  兑现；route/materialize wrapper 的逐段 launch + 规则 GEMM 不受
  操作数 dtype 影响**——燧原 sgmv 瓶颈在逐段 Python 循环
  （index_select/GEMM/index_copy × B 段），而间接寻址 dot 又是
  该芯不可用形态，0.5x 接近该结构上限。csgmv 单 launch 分段 CSR
  是唯一结构出路但依赖间接权重寻址（燧原/昆仑不受支持），
  T48 结构轴定格。
- 证据 `validation/48-status-final.json`。

## 2026-09-08 推荐方案实现与提交前验证（未提交平台）

新增沐曦同数学路径 cpasync 候选，补长段跨tile和三dtype。NVIDIA 编译器拒绝 pipeline 参数（7条error），这是运行时覆盖缺口，未删除参数伪造通过，也未打包。

- source `26a95766b179d263916e9483dfc8d2343c40406a`；verification `26a95766b179d263916e9483dfc8d2343c40406a`。6 个测试方法、18 次实际 kernel 调用；选定 NVIDIA/代理范围门禁失败。
- 回执 `artifacts/competition/batch4-implementation-20260907/t48-release1/verification.json`，SHA256 `7f02b6837a843f77a35f04993b15eaf44c2305d8876860a4ac0276e3d3e44a74`；日志 SHA256 `f023c3e3b871b63720a2d6a4c1a75595b5049ed2faf0169893ab2a79678ec5b8`。
- 环境、逐源码执行范围、原始配对数据和未完成条件见[本轮报告](../implementation-batch4-20260908.md)及[证据清单](../data/batch4-implementation-20260908.json)。本轮不更新历史有效分，未做平台 preflight、上传或正式提交。

## 2026-09-08 E8 后续实施（独立分支）

- source / verification：`65b7fdd418d941ad4393ae9485541c74a55e5017`。
- release：8 方法、161 次 kernel launch，0 fail/error/skip。NVIDIA 代理范围通过；其他目标芯运行时未验证。
- 回执 `artifacts/competition/batch4-next-20260908/t48-release/verification.json`，SHA-256 `125c1980d9dd25a9ab3d7a12a25fd6691c9527ace0a96fba4204047bfea5019d`；完整日志 SHA-256 `ce7fc674e76f2e0d753b7a33ec794f74214ed30e37e5ae43449568eedb5614b2`。
- ZIP `/Users/bytedance/ccc/flagos-batch4-next/artifacts/competition/chunked_sgmv_shrink/e8-65b7fdd/chunked_sgmv_shrink.zip`，SHA-256 `b872fcb8beffb132d4ef9823d176919e1daa83d0f98ca7994cc0082cf0b0e6fa`，15849 bytes；成员：`chunked_sgmv_shrink.py`, `chunked_sgmv_shrink_enflame.py`, `chunked_sgmv_shrink_kunlunxin.py`。逐成员完整 SHA、测试/runner/依赖身份见[证据清单](../data/batch4-next-20260908.json)。
- GPU 读取段长，worker 遍历长段；相邻段优先排列，FP32 使用 BN64。补 CUDA Graph、非连续/未覆盖行、4097 行偏斜段、长 K 回归。generic/燧原/昆仑长 K 使用补偿累加，原容差不变。
- 移除从未取得目标执行证据的沐曦 cpasync 试验覆盖，恢复 E7 三成员集合，让沐曦使用新 generic；原失败仍保留在历史记录。
- 12 个 NVIDIA wrapper 样本 1.598–2.568x；6 轮 AB/BA 原始数据保留。3 个 profiler 样本候选均零 spill，DtoH 1→0，kernel 4→2。满足本轮代理晋级门，待实时平台门禁。

## E8 平台回执（2026-09-08T13:00:36.710258+08:00 快照）

- 单次提交 `11170`，提交时间 `2026-09-08T12:52:50+08:00`，daily_seq `17`。平台 ZIP 再下载验签 `verified`，SHA-256 与本地完全一致。
- 当前 **7/8 已通过，昆仑 waiting_callback，整体 pending**；不得提前记为有效、更新 team best 或宣称 Top1。平台下一次主动查询时间显示 `2026-09-08T14:52:53+08:00`，实际回调可能更早；不重复上传或提交。
- 额度快照：已用 17/30，剩 13，观测时间 `2026-09-08T13:00:36.710258+08:00`。

| 芯片 | E7 | E8 已返回 | 状态 |
| --- | ---: | ---: | --- |
| tianshu | 3.7705 | 6.391 | completed |
| muxi | 5.305 | 9.1925 | completed |
| enflame | 0.528 | 0.56 | completed |
| haiguang | 6.223 | 17.174 | completed |
| kunlunxin | 1.785 | 待定 | waiting_callback |
| huawei | 6.622 | 15.395 | completed |
| card_a | 7.132 | 12.482 | completed |
| card_b | 6.626 | 10.4835 | completed |

- 后续结构假设：燧原/昆仑保留的逐段 CPU 循环仍是下一处候选瓶颈。先验证“全批次一次 route → 连续地址分段 GEMM → 一次 scatter”的目标编译可行性，避免把融合间接寻址失败概括成所有分段 GEMM 都不可行。该假设尚未实现或验证。
- GitHub CI 仅在 master push/PR 或手动触发；本独立分支没有触发 CI。本地按 CI 固定 Black 24.8.0 / isort 5.12.0 / flake8 7.1.0 全部通过。benchmark 后续仅去除冗余括号，AST 相同；实际计时仍绑定 `65b7fdd` 的旧脚本 SHA，源码/测试/ZIP 不变。
