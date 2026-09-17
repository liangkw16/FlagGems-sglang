# Task 68 `fused_eh_norm` 实验记录


## 2026-09-14 E4：燧原 row-loop 4096+封顶（候选就绪待发射）

- 行归约 kernel 可用的配方轴：chunk 2048→4096 + grid 封 24 行步进
  （一 program 走多 token）。其余四源字节不动。
- source commit：`b4010e80ff717cec968d1e0bd893e3ea58813151`；ZIP `e4-b4010e8`，
  SHA-256 `95e685ec66e49e50452720badc226a3ba10b4219cb0fe18e482ad9de9e3e1e1a`；
  release 回执 SHA-256
  `42afb4731c8013b9fb9a0dcc63f5a0a470001159c21eba443e738a4964c8a67b`。

## 2026-09-14 E4（审查补测后，候选就绪待发射）

- 审查确认无四层嵌套（两层：行步进×4 顺序 chunk 循环），建议保留
  结构先补测——已加 tokens=25/49 × hidden=4352/8192 × fp16/bf16 组合
  覆盖（行步进重置累加器 + 多 chunk 内层）。
- source commit：`220aa32d18a1c3a4aca829b09b79e2e906471e43`；ZIP `e4-220aa32`，
  SHA-256 `95e685ec66e49e50452720badc226a3ba10b4219cb0fe18e482ad9de9e3e1e1a`；
  release 回执前缀 `aaca0ef3`；5 方法 0 失败。

## 2026-09-14 E4 平台终态：8/8 valid 7.0895x 新 team best

- 海光 **13.9942（+6%，vendor 持续兑现）**/A +3%；燧原 row-loop
  4096+封顶形态 **回退 -19%**（1.93→1.56——归约 kernel 的封顶与
  elementwise 配方不同性：24 程序行步进损失行级并行）；沐曦 -6%
  （窗口）。均值 7.0895 > e3 7.0530 新 TB（+0.5%）。
- 沉淀：**配方适用边界补充——row-reduction kernel 封顶回退**（燧原
  eh_norm 行归约需要行级并行度；与 streaming elementwise 区分）。
- 榜首 c2flow 8.054 差 0.96；燧原榜首 5.21 vs 我 1.56（3.3x）。

```current
task: 68
operator: fused_eh_norm
batch: 5
validity: valid
platform: completed(16802,e10-carrier,8/8,7.15726x<TB;窗口持平)
candidate_stage: e10
team_best_stage: e9
team_best_speedup: 7.20040833
sealed: no
next: 保留E8团队最佳；Ascend多行候选隔离留档，代理未提速；等待目标芯同源计时与IR
updated: 2026-09-17
```

## 契约与范围

- 完整题面：[Task 68](../tasks/batch-5/68-fused_eh_norm.md)（2026-09-11 新增）。
- 接口 `fused_eh_norm(inputs_embeds, previous_hidden, enorm_weight, hnorm_weight, eps)`；
  两路 RMSNorm（不同权重）+ 末维 concat；hidden ∈ (256, 8192] 且 %256==0；
  fp16/bf16；fp32 归约与权重乘、写回时一次类型转换；per-dtype tolerance。
- 返回 `[T, 2*hidden]` 新张量；核心计算 Triton；八芯 0.1x。

## 实现（S0）

- 上游：SGLang 8014d9d `kernels/ops/layernorm/fused_eh_norm.py`（CUDA JIT
  版）→ Triton 移植：每 token 一个 program，`BLOCK = next_power_of_2(hidden)`
  （≤8192）单块；先 enorm 路载入→fp32 平方和→rsqrt→乘权重→cast 写出，
  再 hnorm 路同构写 `out[:, hidden:]`。权重每 program 重读（上游 CTA 同构）。
- `num_warps=8, num_stages=1`（对齐本仓 fused_rmsnorm 先例）。

## 不可变身份

- source / verification commit：`b4727f1`。
- source SHA-256：`5a19180d9b5257c43d194bbb6df804b53ba849cf6e5679c3ce1a4c303397caaf`。
- test SHA-256：`81ca6d485ef5dae8a4c56273b8c5122c1f3892984cd339c6f13dbb679f591934`。
- ZIP：`artifacts/competition/fused_eh_norm/s0-b4727f1/fused_eh_norm.zip`。
- ZIP SHA-256：`1d588933d9797c66ff7c417ae5817b8659825a56625f54a1b1c06a4e21554807`。

## 验证状态

- py_compile、格式与 lint 通过（本地）。
- 测试：4 方法 / dtype×eps、hidden 512~8192（含 768、7168 非二次幂）、
  tokens 1~8193、strided 输入、全零行与空 batch；输入不变断言。
- **远端 GPU 不可达（2026-09-11 晚）**：release 回执待补，
  `target-runtime-unverified`。

## 风险

- 单块 BLOCK=8192 的寄存器压力在窄带宽芯上可能偏高；若平台退化，E1 方向
  是按 1024~2048 列分块的两遍循环（sumsq 后再归一化写出），复用本仓
  fused_rmsnorm/fused_norm_rope_stacked 已验证形态。
- 每行两次权重载入可被 L2 吸收；不提前优化。

## 优化方向（按把握）

1. S0 直投。
2. E1（未开发）：列分块两遍式，针对 BLOCK=8192 退化场景；先看平台逐芯。

## 2026-09-12 平台结果（submission 13301，daily_seq 6）

- **8/8 valid，均值 6.55616667x**。逐芯：天数 10.8214 / 沐曦 6.1777 /
  燧原 2.0079 / 海光 10.9223 / 昆仑 1.2075 / 华为 3.9125 / A 9.1589 /
  B 8.2411。榜首 HAiWORLD 7.1684（差 0.61）。
- 回执（b4727f1）：4 方法 0 失败、15 launch、20 组 shape；
  `batch5-ext6-validate-20260912/fused_eh_norm/`
  （SHA-256 `c7b2c0c7a5d4c75947afa1fd10794de44c152a4e78a650361e804b64f592457b`）。

## 2026-09-12 E1：`_hygon` 2D 路径分裂 vendor（候选就绪后提交）

- 逐芯榜单（13:4x）：榜首 HAiWORLD 7.2622，榜差 0.71 中 **0.485 集中在
  海光**（我 10.9223 vs 14.7999）；昆仑/国际 B 我方反超，其余芯差
  0.09~0.78。单杠杆 = 海光 vendor。
- 形态（T59 海光 +22% 同款打法）：直接 2D 网格 `(tokens, 2)`，每 program
  只算一路 RMSNorm（enorm 或 hnorm），程序内串行工作量减半、program 数
  翻倍；BLOCK/num_warps=8/num_stages=1 与 generic 一致，generic 字节不动。
- source commit：`19e45c369445ee5f07a1af9acf4b7d387c07a616`。
- ZIP：`artifacts/competition/fused_eh_norm/e1-19e45c3/fused_eh_norm.zip`，
  SHA-256 `7e29e09e2a8b859afb49b12ed1d17177d79915635bf1c1d0e2bb0db11195193d`；
  成员 generic `5a19180d…` + `_hygon` `f95542db…`。
- release 回执（v2，绑定 19e45c3，proxy-vendor hygon）：
  `artifacts/competition/batch5-t68e1-validate-20260912/fused_eh_norm/verification.json`
  （SHA-256 见下方提交段）；4 方法 0 失败，generic 15 + hygon 15 launch。
- 预注册：海光中位收益 ≥1.10x 才保留；其余七芯读数与 S0 窗口一致。

## 2026-09-12 E1 平台提交（submission 13369）

- 上传与正式 POST 各一次；state submitted。额度：发后 12/30。
- 回执 SHA-256：`16e98305df74721afd66f8bfeb4531dee19570df5350a0981088a7a3a6aeb316`；
  日志 `9df42acc60ef133ea48195ce194d1ea11b3da54373391443677689642aa777a2`。
- 裁决点：海光 10.92→？（预注册 ≥1.10x 即 ~12.0+ 才保留）；其余七芯
  应与 S0 窗口一致（generic 字节不变）。

## 2026-09-12 E1 平台终态：8/8 VALID（submission 13369，daily_seq 18）

- **八芯全过，均值 6.91475x = 新 team best（S0 6.5562，+5.5%）**：
  **海光 14.2632（`_hygon` 被选中，S0 10.9223 → +31%，预注册 ≥1.10x 门
  通过）** / 天数 10.8096 / 沐曦 6.2062 / 燧原 1.9993 / 昆仑 1.0861 /
  华为 3.6104 / A 9.2596 / B 8.0836。
- generic 字节未动，昆仑 -10%/华为 -8% 判窗口方差。榜差从 0.71 缩到
  0.35（HAiWORLD 7.2622）；剩余轴：沐曦 +0.75 / 天数 +0.61（逐芯情报，
  均非单芯主导，待下一杠杆）。

## 2026-09-12 E2：路径分裂复制到沐曦/天数（候选就绪后提交）

- e1 逐芯复盘后剩余榜差：沐曦 +0.75（6.21 vs 6.96）、天数 +0.61
  （10.81 vs 11.42）为非单芯主导的次级缺口；海光已兑现 +31%。将同一
  2D 路径分裂形态复制为 `_metax`（沐曦）与 `_iluvatar`（天数）vendor；
  generic 与 `_hygon` 字节不动。
- source commit：`76cb2cae80c9427df4f0bd50c53aef7c35bfc3b3`。
- ZIP：`artifacts/competition/fused_eh_norm/e2-76cb2ca/fused_eh_norm.zip`，
  SHA-256 `697d640cbf4535010dd2e8f56f3e4c3a3bb7256a239a87f4eeec2fd66caa992a`；
  4 成员（generic `5a19180d…` + hygon `f95542db…` + iluvatar `2ddf0e54…`
  + metax `a66edf31…`）。
- release 回执（v2，绑定 76cb2ca，proxy-vendor×3）：`batch5-t68e2-validate-20260912/fused_eh_norm/verification.json`，SHA-256 `bfcbbd4311e336205ebc105e7d8552e7e9c0dbf6badd0d32dbac1b0df81dd0bb`；日志 SHA-256 `ef9d08c2870e33d97ad156e919073d89c31b23fd4e8c89603b4a98543e01cb97`；4 方法 0 失败，4 源各 15 launch。
- 预注册：沐曦/天数中位收益 ≥1.05x 才保留；海光维持 e1 水位。

## 2026-09-12 E2 平台终态：8/8 VALID（submission 13377）

- **八芯全过，均值 7.021875x = 新 team best（e1 6.91475，+1.5%）**：
  天数 **12.2661（`_iluvatar` 兑现，+13%）** / 沐曦 **6.8055（`_metax`
  兑现，+9.7%）** / 海光 12.8831（`_hygon` 选中，较 e1 14.26 回落 9.7%，
  判窗口方差——vendor 字节与 e1 相同）/ 燧原 1.9817 / 昆仑 1.0901 /
  华为 3.6680 / A 9.2459 / B 8.2347。
- 两预注册门（≥1.05x）：天数 ✓（1.13x）、沐曦 ✓（1.10x）。
- 榜差 0.71（S0）→0.35（e1）→**0.24（e2）**；榜首 HAiWORLD 7.2622。
  剩余轴：海光窗口回暖（e1 字节已证 14.26）+ 燧原 1.98（+0.26 缺口）。

## 2026-09-12 E3：燧原两遍分块 vendor + 全芯重掷（候选就绪后提交）

- 目标：差 0.24 中的燧原 +0.26 轴（1.98 vs 榜首 2.24）+ 海光窗口回暖
  （e1 已证 14.26）→ 7.02→7.2+ 掷币 Top1。
- `_enflame` vendor：2048-lane 分块两遍式（sum 标量携带 = GCU 规则内；
  额外一遍读为寄存器减压的代价，0.1 门槛有 20x 余量）；其余四源字节
  不变，全芯全新评测。
- source commit：`8776b60d848509f307e2016fcc91f71579141d25`；ZIP `e3-8776b60`，
  SHA-256 `e7b7d597f58b0f84290ebde202409c077a235fa671c8d12c9523a3e314a4f7fe`；
  5 成员（仅 `_enflame` `62fcb339…` 新增）。
- release 回执：`batch5-final2-20260912/fused_eh_norm/verification.json`，
  SHA-256 `18bebed8dec7b7defb79a93e80aaae46b2df1347bcb10f88fa693c5f2caa00fd`；
  4 方法 0 失败，5 源 75 launch。
- submission 13443（额度最后一发，30/30 用满）；裁决=海光 ≥13.5 且燧原
  vendor ≥1.8 ⇒ 均值冲 7.26 掷币 Top1。

## 2026-09-12 E3 平台终态：8/8 VALID 7.0530x 新 team best（submission 13443）

- **八芯全过，均值 7.05304167（e2 7.0219 → +0.03）**：天数 12.2519 /
  沐曦 6.8378 / **燧原 1.9344（两遍分块 vendor 持平 generic 1.98——
  +0.26 轴未兑现，寄存器减压收益 ≈ 额外读代价）** / 海光 13.1783
  （窗口回暖一半）/ 昆仑 1.0913 / 华为 3.7159 / A 9.1642 / B 8.2504。
- **Top1 掷币未中**：7.053 vs HAiWORLD 7.2622（差 0.21，#6/6）。
  team best 仍前进一步；燧原两遍分块轴按等值判关。
- 额度：本发为今日最后一发（30/30 用满）。

## 2026-09-14 E5 候选就绪：仅去 24-program cap（待发射）

- Codex 审查 + 官方 `_enflame/gcu400/ops/layernorm.py` 对账：行归约族用
  flat grid（grid.x 上限 65535），不套 streaming 族的 24 cap；本 vendor
  kernel 内已有行 grid-stride，可自适应任意 grid 宽度。
- 载体 = e4 字节仅改 `min(tokens, 24)` → `min(tokens, 65535)`（单变量），
  其余四源（generic/hygon/iluvatar/metax）字节不动。
- source / verification commit：`754d91e02ae6`。
- ZIP `e5-754d91e`，5 members，SHA-256
  `78df07432437804cebcea79b9be36afaaf4249a74fd5f42ff9d9297491f77209`。
- release 回执 `batch5-submit-20260914/fused_eh_norm/verification.json`
  SHA-256 `dc831a5d6756c250174f1a785e3ae3f34c59df0452a70e4d7719c81ff854d1b1`
  （release 模式、5 tests 0F0E0S、generic 23 次真实 launch；
  四 vendor target-runtime-unverified）。
- 预注册晋级门：**燧原 ≥ 1.93（e3 水位）且均值 > TB 7.089**；恢复量
  预期 +0.04~0.05 均值（1.56→1.93 ≈ +0.046），不预期触及榜首 5.21 结构差。

## 2026-09-14 E5 平台终态：8/8 VALID 6.936x < TB（submission 14522）

- **预注册门未过**：均值 6.936 < TB 7.089；**燧原 1.528 < 1.93 门**，
  且与今晨 e4（capped）同窗读数 1.562 在噪声内（-2%）——去 cap 无恢复。
- 逐芯（vs e4）：天数 12.5906（12.2743）/ 沐曦 6.7401（6.4525）/
  **燧原 1.528（1.5623）** / 海光 12.8997（13.9942）/ 昆仑 1.0997（1.0791）/
  华为 3.4894（3.7074）/ A 9.0489（9.4509）/ B 8.0944（8.195）。
- 解读：本窗多芯整体下探（海光/华为/A/B 均低），1.93 是 09-12 窗的
  e3 水位；同窗对照下 cap 有无无差异。「24 封顶 -19%」在今天的窗里
  不可复现，行归约-配方边界结论修正为：**封顶对该形态无害亦无益**。
  TB 保持 e4（14385）；去 cap 轴按等值判关，不再花额度。
- 额度：发后 25/30。

## 2026-09-14 E6 候选就绪：燧原 split-row 两相形态（Codex top1 队列 2 号，待发射）

- 依据：15:39 榜单 5 队在燧原 4-6x（金狐狸 6.17/c2flow 5.21/Nectar
  4.58/Evoke 4.26/HAiWORLD 3.96），我方全部行打包形态只有 1.5-2.0
  ——计分 tokens 少时行级并行撑不满 24 SIP，结构实锤非彩票。
- 载体 = 照官方 gcu400 mean 两相模板重写：phase1 以 24-program
  grid-stride 算每 (row, 1024-lane 切片) 的平方和偏积（并行度随
  hidden 缩放），phase2 折 7 个偏积得 rstd 后逐切片归一化写出；
  num_stages=3 pingpong、不钉 num_warps；fp32 全程、store 转换不变。
- codex-review（--uncommitted）修复记录：本候选无缺陷；同 diff 的
  T63 vendor 被抓到 P1 实参错位（splits 位置）已修。
- source / verification commit：`797b7ea1…`；ZIP `e6-797b7ea`，
  SHA-256 `a8508d97fbfb4a9b6d8fb1f6ef47b6d8108fcd450a5a4ce07aae0666562312c8`。
- release 回执 `batch5-submit-20260914/fused_eh_norm/verification.json`
  SHA-256 `3989b87c7462bdfe7cd406ae15a94f5b031cf7d62c42e734ce5e6fca48133eaa`
  （--proxy-vendor enflame：vendor 46 次真实 launch 全矩阵 0F0E0S）。
- 预注册晋级门：**燧原 ≥ 2.0**（Codex 槽 2 门）；均值同时 > TB 7.089
  则进 top1 路径（追平 6.17 + 华为 2x ≈ 8.13）。燧原无增益则该形态
  关闭，T68 top1 主攻降级。

## 2026-09-14 E6 平台终态：8/8 VALID 7.042x < TB（submission 14589）——split-row 形态证伪

- **燧原 vendor 被选中、passed 0.874**——比行打包形态（1.56-1.98）
  更差，预注册门（≥2.0）未过。"tokens 少 → 24 SIP 饥饿"假说证伪：
  切片级并行没有兑现，两相额外 kernel 与偏积流量反而是净损失。
- 其余七芯窗口内（天数 11.99/12.27、沐曦 6.74/6.45、海光 14.66/
  13.99、华为 3.64/3.71、A 9.18/9.45、B 8.15/8.20、昆仑 1.11/1.08）。
  TB 保持 e4 7.089。
- 处置：split-row 形态关闭；燧原 4-6x 他队读数之因仍未破译（剩余
  假设=慢窗水位或未知单遍形态），T68 top1 主攻按预注册降级，转
  守榜 + 等逐芯情报刷新。

## 2026-09-15 23:20 夜间 warps 批扫终态：E7 负:燧原0.87(TB1.56减半);warps=4强负,轴关闭

- 单变量=燧原 vendor 钉 num_warps=4（夜间批 T71/T73/T68/T62 四题）。
  批量结论：燧原 warps 为逐题特性，非统一旋钮；本批 2 负 2 平，
  轴关闭。TB 各自保持。

## 2026-09-16 E8：独立 e/h 路径并行，发布验证通过

- 从TB E4 `220aa32d18a1c3a4aca829b09b79e2e906471e43` 独立分叉：仅将grid改为`(min(tokens,24),2)`，第二维选择e/h路径；保留cap24行步进、CHUNK4096、两遍读取、fp32顺序与warps8/stages1。不是历史E6两kernel split-row，不混入当前HEAD负候选。generic/Hygon/Iluvatar/Metax逐字节等于TB。
- 复用本仓固定Hygon `19e45c369445ee5f07a1af9acf4b7d387c07a616` 路径映射；不直接搬其无界grid。完整7方法含双输入/权重stride与tokens65535/65536/65537，screening baseline与generic/Enflame均通过。24桶×5轮AB/BA：14 primary桶几何均值1.30565015，五轮1.29871674–1.31365468，最差1.04206063；10个row-stride secondary桶1.26693–1.84899，后者也是受影响路径而非control。max40regs/32B shared/0spill。
- screening证据 `artifacts/competition/t68-path-split-cap24-screening-20260916/`：benchmark SHA-256 `6ddb744d3efc6a39c63caf05ef07c8686f7f5f9562684762051c981dda60a3b5`；verification SHA-256 `62f3b11d53e2e7e6e876eb20407e95d71aec46b51baa2f16b6327d978b36a468`。完整日志/原始样本/输入输出SHA在目录。
- source/verification commit `2f64c03f543b8f83d9234a684dbd8e091e3c008f`；候选Enflame SHA-256 `d5f8a4995d5cbb28ab385f169051be9e6bac6e0c8c27dadfb60e58e0ce77e515`、test SHA-256 `f9c02d2f392fdf3cfb99c16edbf93d5a4808973c77247f8af916a0ac314267f9`，与screening完全一致。py_compile/Black/isort/flake8通过。
- exact release **7/7、0F0E0S，generic/Enflame各29次入口、28次kernel launch**，RTX5070Ti/Python3.12.13/Torch2.13.0+cu130/Triton3.7.1。目录 `artifacts/competition/t68e8-release-20260916/`：verification SHA-256 `93598ab6a0ffaaa59594c4c45df8dde8bf69c8a3bbcd8ca851fbf84b14706ab9`，日志 `7a1170059f45e5aeb002d932c9d94f051c71807d32a04b0ecf63c53ece06c93d`。远端`/tmp/flagos-t68e8-release.C6Lg36`、PID387571、660秒总上限、EXIT0，重放包/命令/启动记录保留。Enflame target-runtime-unverified；其他未改vendor未在本轮release代理执行。
- ZIP `artifacts/competition/fused_eh_norm/e8-2f64c03/fused_eh_norm.zip`，SHA-256 `7c78a2686cb77cbfe391187a6d9679f7cd5d70c0e99269a66b07a357ca2cce4d`，17738 bytes；成员仅generic+enflame+hygon+iluvatar+metax，逐成员完整SHA如下。
- 平台晋级预注册：8/8且各芯≥0.1，均值>7.08945833才替换TB；Enflame≥2.0才算目标假说正信号。实时Top1为10.40585833，本改动尚无单步夺冠证据；一次提交判决，不重复同字节候选。

- `fused_eh_norm.py`：`5a19180d9b5257c43d194bbb6df804b53ba849cf6e5679c3ce1a4c303397caaf`
- `fused_eh_norm_enflame.py`：`d5f8a4995d5cbb28ab385f169051be9e6bac6e0c8c27dadfb60e58e0ce77e515`
- `fused_eh_norm_hygon.py`：`f95542dbcc1ca8fb4a2b30c66998ee7f954042cd50776b636faf98a209d048a5`
- `fused_eh_norm_iluvatar.py`：`2ddf0e54364ac27483e52e7ebeeaa6ba89243e548fe30dfa247007fc1aab9f48`
- `fused_eh_norm_metax.py`：`a66edf313a8c1db370057ae6b22143202b1bfdd306779380b7a83a1f250d6b89`

## 2026-09-16 E8 平台终态：8/8 valid，7.10215x微升

11:37:22单次提交15864（daily_seq6），upload/POST各一次，远端ZIP验签verified；11:40:42已8/8。逐芯天数12.61186667、沐曦6.80853333、燧原1.6814、海光13.55273333、昆仑1.0982、华为3.8108、A9.138、B8.11566667。均值7.08945833→7.10215（+0.1790%）新TB；目标燧原仅较1.56226667提升7.63%，未达2.0门，关闭本path分离轴。冻结源的芯片读数也变化，均分微升不全归因于代码。保留E8；不将代理1.30565x外推。证据 `artifacts/competition/top1-20260916/t68-e8-{preflight,submit,status}.json`，终态查询时额度24/30。

## 2026-09-16 晚间：Ascend新结构隔离筛选，未晋级

- 基线提交 `50b91047515f9c45dd7397303996b1a5e9e359da`；本轮只做开发筛选，无平台preflight/上传/提交。候选保存在 `artifacts/competition/t68-ascend-rowgroups-screening-20260916/bundle/`，不替换正式vendor。候选source SHA-256 `642df3afb2f6e1886adb91fcc963cdd6ca9c7eb5433d018e27eeaac8a82134b0`。
- RTX5070Ti / driver610.57.04 / Python3.12.13 / Torch2.13.0+cu130 / Triton3.7.1。screening同源完整回归 8 方法，0失败/错误/skip；明确执行generic+Ascend数学代理。Ascend仍为 `target-runtime-unverified`，NVIDIA结果不构成目标芯否定证据。
- 五轮交替wrapper-inclusive计时，20桶。候选相对generic几何均值：0.887450（候选）；BM1-flat 0.937139，BM1-cap 0.812031。没有正向性能证据，不晋级、不继续在NVIDIA扫配置；重开条件是Ascend同源实测或可绑定的目标编译产物。
- 候选为BM4/2/1、Hconstexpr、每行独立fp32双RMS、权重广播、物理核cap及安全grid-stride。BM2仅16桶/BM4仅10桶，不能跨不同集合比较均值选配置。已有tokens65535/65536/65537及双方stride回归通过；未把宽上界数学审计记成实际运行。
- 复现脚本、输入清单、回执、完整日志、逐桶逐轮原样本保留在上述目录；本轮未保存PTX或编译资源数据。screening不是release，不给旧ZIP或平台资格背书。
- `verification.json` SHA-256 `1607c402559b2204b767b7369dc8c89e675b41d0c91b71c0f09f6d4f419176e9`。
- `verification.log` SHA-256 `325144ff913b25eb10de56467f8edcff6736296e6fbfaea43ae15e0945d4f179`。
- `benchmark.json` SHA-256 `3390c41102e0356c58e13b99f0fe4b15bc9faac29b9c1abd377159cadd598a9b`。
- `run.log` SHA-256 `08c31659bbb12280a409b3e9f6031327ce7da8de3b1c76083373fb969989ebf7`。

## 2026-09-17 E9：Enflame GCU 配方（去钉+stages3），已提交

- 结构（`f8357985`，第二轮收割）：`_enflame` GCU 官方配方：去 num_warps=8 钉、num_stages 1→3 开 pingpong（T19/T51 两次平台 +38% 配方）。其余成员字节冻结；本题为已证配方家族单变量投放。
- screening（RTX 5070 Ti）：unittest 7 项全绿；black/flake8 过。
- release v2（commit `f83579853cc0163321fd11e6301a1ad61d46eb58`）：7 项全过 0F/E/S/X，changed-vendor 真实 launch 28 次，exit 0。回执 `artifacts/competition/round2-20260917/fused_eh_norm/verification.json` SHA-256 `a0da237ecd78b252cd2103776f04fe9d399cbeae0b62c5e5825d85b651646c13`。目标芯 target-runtime-unverified。
- ZIP：`artifacts/competition/fused_eh_norm/e9-f835798/fused_eh_norm.zip（17631B）`，SHA-256 `fceb2b67300cfa0b6548d3169618943349b1a64f0b190f28b0157edf8f815471`，5 成员（hygon/iluvatar/metax 冻结）。
- 预注册门：8/8 有效且均值 > 7.10215；燧原（1.68 起） ≥ 2.2 为正信号。一次候选一次判决。

## 2026-09-17 E9 单次平台提交

submission **16576**，evaluating；八芯终态另节记录。

## 2026-09-17 E9 平台终态：8/8 有效，新 TB 7.20040833x

submission 16576 completed/valid，均值 **7.20040833x > 7.10215 换 TB**（+1.38%）。逐芯（vs E8 TB）：enflame 1.6814→**2.338（+39.0%，GCU 配方三度平台实证：T19/T51/T68）**；muxi 7.046(+3.5%)/card_b 8.180(+0.8%)/kunlun 1.115(+1.6%)/haiguang 13.212(-0.9%)/tianshu 12.553(-0.5%)/huawei 3.750(-1.6%)。预注册燧原门 2.2 过。下一梯度：燧原仍距次优 41.7 有 17x，需 SUB 批量/best-practice 结构。

## 2026-09-17 E10 载体终态：8/8 有效 7.15725833x，保底

窗口重掷 1/2：全芯窗口持平（enflame 2.318 ≈ banked 2.338），7.157 < TB 7.200。低滚 1/2。
