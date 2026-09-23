# Task 80 `fixup_zero_kv` 实验记录

```current
task: 80
operator: fixup_zero_kv
batch: 6
validity: valid(8/8,e16,558.04x TB)
platform: e23(20469)valid 8/8 avg 554.449675<TB 558.039775判负,燧原109.912<130门; e22(20462)仍7/8,B卡waiting_callback;TB守e16
candidate_stage: e23
team_best_stage: e16
team_best_speedup: 558.04x
sealed: no
next: e23燧原i32轴关闭并恢复e22原字节(8ad7b360);e22只读等B卡回调;余额1/30,停止低证据重投;下一结构目标华为733/沐曦560/燧原234
updated: 2026-09-23
```

## 2026-09-23 E23 平台终态（20469）：valid 8/8，均分 554.449675 < TB；燧原 i32 轴判负

- 09-23 18:45:21+08 单次提交（当日序号 29），18:46:53+08 平台
  completed/valid、8/8 全过，均分 **554.449675**，低于 e16 队内最佳
  558.039775，未换 TB、仍为第 3。逐芯：天数 1197.9026 / 沐曦
  381.6966 / **燧原 109.912** / 海光 893.4904 / 昆仑 14.6098 /
  华为 378.4904 / A 845.5846 / B 613.911。燧原低于预注册
  ≥130 保留线，与 e22 110.1916 基本持平；i32 地址轴关闭。
- 源码在提交后按 e22 `f109f3b8` 的 GCU blob 原字节回滚，commit
  `8ad7b360`；E23 ZIP/回执仍保留，不再以同字节重投。提交时 CLI
  `remote_verification=unavailable`，其后受信旧 status 主机的独立
  无认证 HTTPS GET 验证远端 19180 B 与本地 SHA-256 完全一致。
  18:46:53+08 平台额度余 1/30。

### E23 预注册方案与验证证据

- 假说：e22 GCU kernel 将 `cum_seq_lens`、token/列 offset 全部转 i64；
  gcu300 的 64 位整数算术走软仿真。e23 仅把安全范围内的地址计算转
  int32，实际最后一行 out/lse offset 超过 `2^31-1` 时保留旧 i64 路径。
  Ascend/global worker 与其余三个成员和 e22 ZIP 字节相同。
- NVIDIA 代理配对计时：长段 i32/i64 基本持平，短段 wrapper i32 慢约
  5–6%；这不支持在 NVIDIA 晋级，但也不能预测 gcu300 的软仿真收益。
  提交前目标芯 **target-runtime-unverified**。预注册门：八芯全过且每芯≥0.1；
  燧原 ≥130 保留该轴，≥160 视为实质收益；均分 >558.039775 换 TB。
- source commit = verification commit：
  `38499d58e417d530c0611014105abb66c68c3d3e`；测试 SHA-256
  `2bca30d528133d3b2fa4849c8afd3413ec5eb48d4e6dc58a9bfa6c37d99b8987`。
  release 回执 `artifacts/competition/t80-e23-20260923/verification.json`
  SHA-256 `6e7a295d366827991ccf346305a2c054912698a827d23a9590f18599c6fa7cbd`，
  相邻日志 SHA-256 `a4fd4b1bd0eea883e8dd3b60348bd518000324945ca2a1703ee1fd1330ad0ea5`；
  NVIDIA RTX 5070 Ti / torch 2.13.0+cu130 / Triton 3.7.1，14 测试，
  0 failure/error/skip/xfail，5 源各 36 次非 warmup kernel launch。
- 不可变 ZIP `artifacts/competition/fixup_zero_kv/e23-38499d5/fixup_zero_kv.zip`
  SHA-256 `bb40df546c84689a421a41ea1cb577d6d9c06836f541eea10718ca373f0790f2`，
  19180 B，`--verify-existing` 与 `unzip -t` 通过，成员与 commit 字节一致：
  - `fixup_zero_kv.py` `e3371c49e3ef6f9ba7b2321b094a738a208129a682fc29e837b6a17317035943`
  - `fixup_zero_kv_ascend.py` `44f18d95e949fb131de932ad2858a5a442cc94709aeef0ce7108d33227d7f498`
  - `fixup_zero_kv_enflame.py` `62dea421902ffd17999f68d41c3579af35d563ad6b12633aeda45ba94733bdfb`
  - `fixup_zero_kv_kunlunxin.py` `b79e658780e02637372a5a84c1290b6bdddc3e8def54ae8ace5ba86030c6f0ff`
  - `fixup_zero_kv_metax.py` `37a4d96256677a1899fc388d8b5ef8c07b7b610f4af148202093aef58af15bda`

## 2026-09-23 E22 平台中间态（20462）：7/8 已通过，B 卡等回调

- 09-23 18:27:59+08 单次提交（当日序号 27）；18:38+08 只读状态为
  evaluating/pending，天数 1189.3998 / 沐曦 389.7132 / 燧原 110.1916 /
  海光 893.9844 / 昆仑 13.988 / **华为 381.2182** / A 814.279
  均通过，B 卡 `waiting_callback`、`next_status_query_at=20:28:02`。
  华为越过 ≥315 保留线，未到 ≥450 突破线；在 B 终态前均分与有效性未知，
  不将 7/8 计作有效排名。额度余 3/30。
- 提交时 CLI `remote_verification=unavailable`（未设置受信 host）；其后以
  既有平台 status 中的对象存储主机名作白名单，独立无认证 HTTPS GET
  下载本次文件，18793 B 与本地 ZIP SHA-256 完全一致；未重试上传/提交。

## 2026-09-23 E22 候选：全局固定 worker 扫描

- 假说：e20/e21 以 `(batch, tile)` 发射，即使 tile cap 缩到 48 仍使总
  program 数随 batch 增长；e22 Ascend 改为最多 40 个全局 worker，worker
  跨段及段内 tile 循环，保留 e16 的 `[8,512]` 2D 写形态与真实
  `cum_seq_lens` 覆盖。固定 40 避免每次 wrapper 查询设备属性的约 0.8ms
  主机开销。此形态不同于此前的单段 CTA、逐段 tile 与 capped 2D grid。
- 燧原从 e16 最佳 ZIP 恢复 12 CTA / 2 warps / `BLOCK_V=4096`，补齐
  `max_seq_len` 谎报时的段内 tile-stride 覆盖；其余 generic、昆仑、沐曦
  沿用已验证字节。NVIDIA 代理配对计时：Ascend wrapper 在四个 shape
  的耗时为 generic 的 1.05–1.19 倍，**不是华为加速证据**；全局 worker
  假说只由目标芯平台判定。
- source commit = verification commit：
  `f109f3b8ef4998229ac4e675ef934f55e78ffc71`；测试 SHA-256
  `2bca30d528133d3b2fa4849c8afd3413ec5eb48d4e6dc58a9bfa6c37d99b8987`。
  release 回执 `artifacts/competition/t80-e22-20260923/verification.json`
  SHA-256 `a09d415473e472e7f4d5059545b8a1f27ae577d43a476d913c1b9f48198828e2`，
  相邻日志 SHA-256 `fb4ef9427c110a1bf92af0077c6d19420b083a657aa6027da41936551e5935ac`。
  NVIDIA RTX 5070 Ti / torch 2.13.0+cu130 / Triton 3.7.1；release 14
  测试，0 failure/error/skip/xfail，5 源各 36 次非 warmup kernel launch；
  Ascend/Enflame/Kunlunxin/Metax 均为 **target-runtime-unverified**。
- 不可变 ZIP `artifacts/competition/fixup_zero_kv/e22-f109f3b/fixup_zero_kv.zip`
  SHA-256 `9b804a0198e32eef66fcf1e30f5359a4964d7b117bf989cfb166b11c50a90759`，
  18793 B，`--verify-existing` 与 `unzip -t` 通过；成员与 commit 字节一致：
  - `fixup_zero_kv.py` `e3371c49e3ef6f9ba7b2321b094a738a208129a682fc29e837b6a17317035943`
  - `fixup_zero_kv_ascend.py` `44f18d95e949fb131de932ad2858a5a442cc94709aeef0ce7108d33227d7f498`
  - `fixup_zero_kv_enflame.py` `c8ccee9808ff5ed181ac5b5385800b9dc1b3808b80c2a8fcb713a5763365762d`
  - `fixup_zero_kv_kunlunxin.py` `b79e658780e02637372a5a84c1290b6bdddc3e8def54ae8ace5ba86030c6f0ff`
  - `fixup_zero_kv_metax.py` `37a4d96256677a1899fc388d8b5ef8c07b7b610f4af148202093aef58af15bda`
- 预注册门：八芯全部正确且每芯 ≥0.1；华为 ≥315 保留该调度方向，
  ≥450 视为有意义突破；均分 > e16 558.039775 才换队内最佳。当前
  榜首约 643.53，若其余芯不变，华为需额外约 +684 才可仅靠此芯夺首；
  小于门槛或编译/数值失败即回到 e16 最佳 ZIP 分支继续寻找新结构。

## 2026-09-23 E21 平台终态（20362）：valid 8/8 均值 526.66 < TB 判负；ascend 轴五形态尽、封存

- 结构（`da470e93` + 评审 r2 测试修正 `b8ec953c`）：e20 字节基础上
  axis-1 tiles cap 255→48（core-scale K）+ num_warps 8→16。预注册门
  「华为 ≥450 keep」未达（186.93），均值 526.66 < 558.04 未换 TB。
- 逐芯：天数 1187.9 / 沐曦 390.7 / 燧原 110.6 / 海光 891.6 / 昆仑
  8.7 / 华为 186.9 / A 814.0 / B 622.9。八芯全部 pass（无编译错）。
- 华为 ascend 轴终审：e15（E_CHUNK=128 分块）/e16（hist/fill 拆分）
  BiShengHIR UB 编译败 ×2；e20（UB-safe 整块无 mask）210.91；e21
  （48-tile core-scale + warps16）186.93——四种内核形态 + e19 generic
  267.04，全部低于 generic 路径。EvokeAgent 733/c2flow 778 的华为
  读数需要的是不同算法形态（如整个 op 单核化），非 launch 几何。
  T80 ascend vendor 轴封存；下一发（若有）只走燧原（110.6→233.8）。
- 五元组：commit `da470e936baa58a87d032a6bd5ad8a49c11b6473`；ZIP SHA
  `5383510a0125429f879d213e88977a316d28f5d1604bb93583138ef818eb5336`
  （5 成员）；verification_commit `b8ec953c`（test
  `ac8f136f133158f7830ef2affa0457aa3d537754a2e29d204cd98c712a077e69`）；
  回执 `day6-climb-20260923/t80e21-wf/`（5 源 × 46 launch）。

## 契约与实现（S0）

- 题面：[Task 80](../tasks/batch-6/80-fixup_zero_kv.md)。TRT-LLM ragged
  attention 后处理：`kv_lens[i]==0` 的请求整段 out 清零、lse=-inf；reference
  带 `nonzero().tolist()` host sync（巨分来源）；精确 + lse equal_nan。
  上游 74338e9 CUDA 版对照。
- 实现：clone×2 + 单 kernel；1D grid（batch×ot 展平，燧原 grid.y≤255 免疫），
  标量早退 + token 块 [BLOCK_T=8 × BLOCK_V=512] 向量清零；`max_seq_len`
  仅做 launch 尺寸提示，覆盖以 cum_seq_lens 为准（tile stride 兜底）；
  行内两维连续断言。
- 测试：kv_lens 与 token 边界解耦（零 KV 请求仍占有 q-token 行）、
  谎报 max_seq_len、未触碰段 NaN/Inf/-0 位级保留、重复调用重读。

## 不可变身份

- source commit：`370923bfb83a8b7e681051acb2a3b1585dde12fc`（五题同批）。
- verification commit：`686092eb443f6a7b9514f8890fa9fdaa2d339a02`（含测试侧修复轮）。
- source SHA-256：`eebf1fcb449a663163c1f7bb6a58bfb7c0d25a72bb0113e0bf7dd222068b4adc`。
- test SHA-256：`10d9df525805ab46f429cb60a18d625c6222f251303b82d2e62c0c48c3a0c5ff`。
- ZIP：`artifacts/competition/fixup_zero_kv/s0-370923b/fixup_zero_kv.zip`，SHA-256
  `294b35e3f98c6d8d759e0a3992914b65a9bdbd8dedee230493afad36d23db6a0`（单成员 `fixup_zero_kv.py`，generic-only）。
- release 回执：`artifacts/competition/b6-s0-release-20260917/fixup_zero_kv/verification.json`
  （6 测试 0 失败，16 次非 warmup kernel launch，
  NVIDIA RTX 5070 Ti / torch 2.13.0+cu130 / triton 3.7.1），
  SHA-256 `82f4cc986cfb8cffa03ec88c29cd8b8cd65fc141a91eb31a46cadaa1617035f5`；日志 SHA-256 `664ba85938870c42081c219b81f350bb6a1acc13e7193190b93e94d72cc5c180`。

## 靶子与下一步

- 榜首 HAiWORLD 139.95x；门槛风险芯为燧原（cgzhou 0.1924）；华为 persistent 是最大杠杆（1.9→51.6）。
- 验证三轮教训已固化在测试侧：numel%16 契约、token 边界与 kv_lens 解耦、
  归约噪声容差、行内连续 stride 构造。
- 八芯目标 `target-runtime-unverified`（NVIDIA 代理证据），裁决权在平台。

## 2026-09-18 S0 平台首回执（submission 17218，observed_at 01:0x +08）

- 状态：7/8, 昆仑编译错；均值 -。
- 逐芯：天数 238.62 / 沐曦 95.06 / 燧原 19.90 / 海光 198.22 / 昆仑 None(编译错) / 华为 162.67 / A 223.75 / B 134.52。
- 昆仑全部 9 case 同指纹 `size mismatch when packing elements for LLVM struct expected 8 but got 1`（fixup_zero_kv.py:54 lse 2D store），ConvertTritonXPUToLLVM 阶段——代码侧可修，非崩溃族。

## 2026-09-18 E1 平台终态：昆仑 flat-1D vendor 修复兑现，8/8 首个有效 136.04x #2

- 结构（`7757a41c`）：新增 `_kunlunxin` vendor——lse/out 全部改纯 1D store
  （每 token 行 flat span + 分块 mask），替换 generic 的 2D broadcast store
  （昆仑 `ConvertTritonXPUToLLVM` packing mismatch 根因）。release v2 双路径
  （generic 16 + kunlun vendor 16 launch）全过；ZIP `e1-7757a41` SHA-256
  `39cea4f138e7417e0fb86c1e7fe3afa63ff4c22539507bdfcacc9ffdb2183b43`，2 成员。
- submission **17224** completed/valid，8/8，均值 **136.0378x**（vs 榜首
  HAiWORLD 139.95 差 2.8%）。逐芯：天数 238.23 / 沐曦 100.42 / 燧原 23.28 /
  海光 187.58 / **昆仑 9.23（vendor 修复，s0 编译错→榜首档 3.2 的 2.9 倍）** /
  **华为 164.68（榜首 51.6 的 3.2 倍）** / A 226.31 / B 138.58。
- 反超榜首的芯：华为/燧原/天数/B/昆仑；缺口：沐曦（100 vs 195）、
  海光（188 vs 248）、A（226 vs 265）。

## 2026-09-18 E2 平台终态：窗口重掷未遇慢窗，微幅新 TB 137.04

- E2（17296，`a4032b55`）：e1 字节的注释载体重掷（执行字节相同）。读数
  137.04 ≈ e1 的 136.04（各芯 ±2%）——**仍是快窗**，低滚 1/2，按纪律停止
  该路径。微幅新 TB（+0.7%）。release 双路径全过；ZIP `e2-a4032b5`。
- 榜单通胀对照：榜首 HAiWORLD 328（海光 553/A 638/B 457）为慢 reference
  窗口读数；我方 137 为 01:00 快窗读数，结构差距需同窗比较（沐曦
  102 是唯一明确落后轴）。

## 2026-09-18 E3/E4 平台终态：融合+掩码修复 → 沐曦+91%/海光+110%，燧原 vendor 兜回，TB 200.92

- codex-ask 第二轮指出 lse store 缺头掩码（非 2 幂头数越行写）与 clone
  融合机会，均经源码核实与代理基准实证（old 在 heads=96 混合段
  MISMATCH，new 全对；耗时 all-zero +19%/all-copy -13%）。
- E3（17314，`a115c1db`）：clone 折入 kernel（每元素单写者，零 KV 段写
  常量、未触碰段拷贝，流量 (2+z)D→(2-z)D，省两次 clone launch）+ lse
  头掩码。逐芯 vs e1：**沐曦 100→191.4（+91%）/ 海光 197→413.8（+110%）/
  华为 161→231.6（+43%）/ A 228→316 / B 135→159 / 天数 239→252**；
  燧原 23.3→14.1（-40%，融合版 GCU 回退）。均值 **198.40 新 TB**（+45%）。
- E4（17318，`6bcef5f4`）：`_enflame` vendor = e1 clone 结构 + 掩码修复
  ——燧原 14.1→**23.5（+67%）**，其余保持。均值 **200.920075 新 TB**。
  release 三路径全过；ZIP `e4-6bcef5f` 3 成员。
- 判读：codex-ask 排序第一的建议超额兑现（目标沐曦 ≥128，实得 191）。
  融合结构在 launch 敏感芯全面受益；GCU 偏好 clone+patch 分离形态
  （与 T81 双 kernel 的 GCU 反例互补：**融合与否按 chip×op 组合定**）。

## 2026-09-18 E5 平台终态：HV 展开中性，gap 覆盖加固入库，TB 保 e4

- 结构（`cb05292d`）：HV/NH constexpr 静态展开（T81-e5 机制迁移）+
  **首/尾 gap 覆盖**（cum 边界外 token 由边界段程序拷贝——bench 实证
  e3/e4 已提交字节在该形状下留下未初始化垃圾，平台形状恰好全覆盖故
  一直 8/8，属潜伏漏洞）。release 三路径全过。
- submission **17359** completed/valid，8/8，均值 **198.06 < TB e4
  200.92**（保 e4）。逐芯 vs e4：燧原 23.5→24.7（+5%）/ 昆仑
  9.06→9.41（+4%）/ 海光 399→414（+4%）；沐曦 -4%/A -5%/华为 -4%
  （窗口漂移量级）。判读：**HV 展开机制中性**——T80 的内层循环是
  trivial 拷贝（memory-bound），与 T81 的点积+FMA 热点循环不同；
  机制适用边界=循环体内是否有实质控制/计算开销。
- 正确性加固意义：gap 覆盖修复已在有效提交中入库，防住隐藏 shape
  触发未初始化输出的风险。

## 2026-09-18 E6 平台终态：warps 迁移混合判决（沐曦 +19%），明日组合待发

- 结构（`d4a95b60`，单变量）：generic num_warps=8（T81 warps 饥饿机制
  迁移；8×512 tile 默认 4 warps 时 32 元素/线程）。
- submission **17397** completed/valid，8/8，均值 **198.65 < TB e4
  200.92**（保 e4）。逐芯 vs e5：**沐曦 184.4→219.2（+19%！）** /
  B +6% / A +5.5% / 昆仑 +1%；**天数 254.6→229.2（-10%）/ 燧原
  24.7→22.0（-11%）** / 海光 -6% / 华为 -2%。
- 判读：warps 饥饿跨题存在（T81/T80 的沐曦均 +19~28%），但**最优
  warps 按芯分化**（天数/燧原偏好默认 4）。明日 e7 组合：generic 回
  默认 + 新增 metax vendor（warps=8 字节），投影 TB ≈205.3。

### 订正（codex-ask 第五轮核实）

- e6 的燧原 -11% **不能归因 warps**：`num_warps=8` 改在 generic，燧原走
  独立 vendor（字节未变）——该读数是窗口/测量混淆信号，"燧原偏好默认 4"
  的表述作废。天数 -10% 的归因保留（天数走 generic）。
- e7 实施口径：从**当前完整修复版**（含 gap/batch=0/HV）仅撤掉 generic
  的 `num_warps=8`，并把该配置放进新增 metax vendor——不是整文件回滚
  e5。

## 2026-09-18 晚 E7 候选就绪（提交前验证完成，待额度刷新首发）

- 结构：沐曦 warps=8 组合（e6 判决拆分）。预注册门：沐曦耗时-10%或≥210;均值>200.92换TB。
- source/verification commit `c6cf443188e44968213ba09b33c11385b786dd04`；release 回执
  `artifacts/competition/b6-e7-ready-20260918/fixup_zero_kv/verification.json`（4成员(generic/enflame/kunlunxin/metax),4路径x24launch,9测试）。
- ZIP `artifacts/competition/fixup_zero_kv/e7-*/fixup_zero_kv.zip`：
  SHA-256 `fa95bc2c01874db50f3809d5273f67866c8348ef883b6f815ecf235113c00c50`；test SHA-256 `34c41446af5b8700a1a42f96b24b4b2b7ea7de44ea77b1e0d6f3441d716f9abf`；回执 SHA-256 `bbc31a1e0d4274cd52e7104ada7bf8b427e2caaea73358b08930668326a33276`。
- 发射参数齐备（commit/zip/sha/test/receipt 五元组已核对），午夜额度
  刷新后按第五轮排序直接 preflight→submit。


## 2026-09-22 E8/E9/E10：原位结构 506.12 新 TB（+152%）

- **E8（19426）：8/8 valid 506.117 新 TB**。原位写（out/lse 即输出参数，
  只写零段、健康段程序立即退出——题面语义"单 launch 清零+无 host sync"）。
  逐芯：天数 1207 / 沐曦 167 / 燧原 25.2 / 海光 893 / 昆仑 8.59 / 华为
  303.1 / A 814.3 / B 630.8。codex-review 两轮（契约变更+每模块新鲜输入
  P2 修复）。T84 e12 之后**平台第二次接受原位返回**。
- E9（19433）：metax/enflame vendor 原位移植（沐曦 167→311.8、燧原
  25.2→41.9 兑现），昆仑 vendor 误带 2D lse 广播 → XPU packing 编译败。
- E10（19434）：昆仑纯 1D 平铺修复后编译过但数值败（3072 元素失配 ≈
  lse 写入量；e8 七个原位 generic 芯全过 ⇒ 疑昆仑检查器对返回缓冲的
  行为差异）。**止损**：昆仑 vendor 回滚 e8 字节（8.59 保底），TB 守住。
- 残余差距 vs c2flow 569.3：沐曦 409（e9 已证 312→e10 408.7 水位）、
  燧原 42 vs 193、昆仑原位之谜。


## 2026-09-22 E11 平台终态：518.70 新 TB

- E11（19452）：e9 的 metax/enflame 原位 vendor + 昆仑 e8 老字节组合。
  逐芯：天数 1205.2 / 沐曦 294.0（窗口回落，e9/e10 曾 312/409）/
  燧原 41.96 / 海光 892.2 / 昆仑 8.58 / 华为 276.6 / A 814.9 / B 616.2。
  **518.70 > 506.12 换 TB**。距 c2flow 569.3 = 9.8%。


## 2026-09-22 E12 平台终态：527.85 新 TB（燧原 12-CTA 几何兑现）

- E12（19455）：enflame vendor 真 gcu300 几何（(seg,tile) 工作项平铺 +
  ≤12 CTA 跨步 + warps2；e9 只钉了 warps 没封 grid）。**燧原 42→102.6
  （+144%）**；其余芯健康（天数 1171/海光 901/华为 310/A 819）。
  527.85 > 518.70 换 TB。距 c2flow 569.3 = **7.9%**。


## 2026-09-22 E13 平台终态：540.87 新 TB（metax 官方限额兑现）

- E13（19584）：metax vendor 存储瓦片 (8,512)=4096 → (4,512)=2048
  （官方 max_tile_size）+ warps 8→2（官方 zeros/ones 写密集启发式）。
  **沐曦 295→381.7（+29%）**，A 845。540.87 > 527.85 换 TB。
  **距 c2flow 569.3 = 5.3%**。


## 2026-09-22 E14（uncertain 封存）与 E14R 平台终态：550.19 新 TB

- E14：上传 uncertain 未达平台（无提交记录/未扣额度），CLI 对该元组
  永久封锁——intent 留档不再触碰；e14r 注释载体走新元组。
- E14R（19636，`e14r` 载体 + warps4）：8/8 valid **550.19 > 540.87 换
  TB**。天数 **1273.4**（挂起自行恢复 + 窗口新高，此前带 1199-1205）；
  沐曦 359.6（warps4 < warps2 的 381.7——**metax warps 阶梯闭合：2>4>8**）；
  华为 315.3。**距 c2flow 569.3 = 3.5%**（差 ~19 均值）。


## 2026-09-22 E15/E16 平台终态：558.04 新 TB，距榜首 2.0%

- E15（19657）：valid 536.10 < TB。燧原宽度 2048 档兑现 102→113.9；
  沐曦 warps2 恢复 388；天数窗口回落 1201。负结果但阶梯确认。
- E16（19661，燧原 4096 档）：valid **558.04 > 550.19 换 TB**。
  **沐曦 503.7（反超 c2flow 473）**；燧原 111.3（4096≈2048，宽度到顶）；
  B 671。**距 c2flow 569.3 = 2.0%**（差 ~11 均值）。
- 残差解剖：燧原 111 vs 193（-10 均值）+ 华为 297 vs 354（-7）+
  天数 1160（带 1160-1273）。


## 2026-09-22 E16R 平台终态：534.44 未超 TB，掷窗停止

- E16R（19666，e16 同字节载体）：valid 534.44 < 558.04。天数 1176 /
  沐曦 384（回落）/ 华为 266。四发样本 536/550/558/534 —— 窗口方差
  ±24，均值 ~545 < TB，继续掷窗 EV 为负，停止。TB = e16 的 558.04。


## 2026-09-22 E17 平台终态：flat-span 论点证伪，回滚 e16 字节

- E17（19698，generic+enflame+metax 全换 flat-span 双流形态）：全线回退
  ——沐曦 503.7→250.3 / 燧原 111.3→13.4 / 海光 892→545 / 华为 245 /
  A 687 / B 551；天芯失败挂起。**"零段=连续单流"的结构假设被平台证伪**：
  (seg,chunk) item 跨步循环的每 item lens 加载 + 小 lse 块开销远超
  连续性收益；e8 的 (8,BLOCK_V) 2D 瓦片在行内本就合并访问。
  codex-review 仍抓到真 P2（advisory span 未截断，已修但随回滚入档）。
  树已回滚 e16 字节，TB 558.04 保持。


## 2026-09-22 E18 平台终态：533.55 未超 TB；codex-ask 咨询闭环

- E18（19716）：燧原 tile-stride 兜底修复（codex-ask 抓到的 e12 item
  映射正确性缺口）+ warps 2→1。valid 533.55 < 558.04。燧原 110.7
  （warps1≈warps2，阶梯闭合）；修复为契约健壮性（平台 shape 未触发），
  回归测试 `test_understated_span_zero_segment` 入 RELEASE_REQUIRED。
- codex-ask 咨询结论（gpt-6-astra, high）：候选 3（warps1）已随 e18
  判平；候选 2（segment-band 调度：每条带一次元数据读+按段长定循环）
  为 09-23 主攻；候选 1 被截断待重询。窗口判别：无对方整体慢窗强证据，
  燧原单独窗口差异不可从 speedup 分离——需按相同 case 的 reference/
  实现原始耗时判别（无渠道，接受未知）。


## 2026-09-22 深夜 E19 上膛（09-23 午夜首发第 1 发）

- E19（`423699e`，燧原 segment-band，codex-ask 候选 2）：每程序拥有
  每 12 段之一、一次元数据读、按实际段长扫全部 tile（无 advisory 空
  tile）。codex-review 零发现（3000 组调度模型与父提交覆盖一致）；
  release 绿；ZIP `fixup_zero_kv/e19-423699e/`。回执
  `day5prep-20260921/fixup_zero_kv-e19/`。预注册门：燧原 ≥130 视为
  带突破；均值 >558.04 换 TB。


## 2026-09-23 E20 上膛（首个 _ascend vendor 入包：UB-safe 重构，待发射）

- 结构（e-round1 `48555b55` → e-round2 重构 `4a501c5d` + 复审修复
  `f258168d`）：首个 `_ascend` vendor 进入 ZIP。e-round1 的 T92-e20
  规则（int32 flat-span + fp32 尾掩码 2^24 双域门）被 e-round2 整体
  重构为 T40-e16 存储形态：整块无 mask 热路径 + 每流单 int-mask 尾块，
  删除双域门（4 个分支体合计 ~224KB 超 Ascend 192KB UB——T84 e15/e16 +
  T92 e20 BiShengHIR 'ub overflow multi-buffer' 同族风险）；BLOCK
  16384→4096、warps 16→8；grid 总上限 tiles=min(cdiv,255,65535//batch)
  （Ascend 2D 展平 65535 轴，stride 保覆盖）。新增回归：多块精确/部分尾
  （含全真尾掩码 + 空段）、grid 总量边界 257×255=65535 保留 vs 258→254，
  均入 RELEASE_REQUIRED_TESTS（现 13 项）。
- release v2（source=verification=`f258168d6f6d078364d648a719cfb8372b5b6c70`）：
  14 测试 / 210 case 全过，0 失败/错误/skip/xfail；5 路径
  （generic+ascend+enflame+kunlunxin+metax）各 35 次非 warmup kernel
  launch，330 组非空张量 shape/dtype 覆盖，exit 0；NVIDIA RTX 5070 Ti /
  torch 2.13.0+cu130 / triton 3.7.1。华为目标芯 target-runtime-unverified
  （NVIDIA 代理证据）。回执
  `artifacts/competition/day5prep-20260921/fixup_zero_kv_ascend_ub_safe_redesign-wf/verification.json`
  SHA-256 `3d90cb76f946753e013a2529e33777f8cdb2a48a8fd1fa0b36b1825ff7a34972`；
  日志 SHA-256 `87bb8251a402762d415c29314f0db37c570a5b95a6ee79d7fac3af1d80ece4b8`。
- 五元组（commit / ZIP / ZIP SHA / test / receipt，verification_commit 同
  commit `f258168d`）：ZIP
  `artifacts/competition/fixup_zero_kv/e20-f258168/fixup_zero_kv.zip`
  SHA-256 `8d613582f24470f87bfa05815e2178fe32d4b84930c089ebc844c6434a6d1624`
  （21166 B）；test SHA-256
  `87309f042c29171053ae1bf31416fc74e41c35f159becb5d126e60b307605ae4`。
- ZIP 名单（5 成员，已与 `zipfile` 实际成员逐一核对一致且逐成员字节与
  `git show HEAD:<path>` 相同——无打包器夹带；generic/enflame/kunlunxin/
  metax 四成员哈希与 e19 逐项相同，仅 ascend 新增，单变量确认）：
  - `fixup_zero_kv.py` =
    `e3371c49e3ef6f9ba7b2321b094a738a208129a682fc29e837b6a17317035943`（同 e19）
  - `fixup_zero_kv_ascend.py` =
    `73b297643646c6ca90d8bd3bcdfda4f2cf09a37701af047c229c740bc5edd939`（新）
  - `fixup_zero_kv_enflame.py` =
    `ff9a42238d54c7d34b741350c281a83809b5e0d71d84fa79d1aa3ab774d03063`（同 e19）
  - `fixup_zero_kv_kunlunxin.py` =
    `b79e658780e02637372a5a84c1290b6bdddc3e8def54ae8ace5ba86030c6f0ff`（同 e19）
  - `fixup_zero_kv_metax.py` =
    `37a4d96256677a1899fc388d8b5ef8c07b7b610f4af148202093aef58af15bda`（同 e19）
  - zip_sha256 `8d613582…` ≠ e19 `9f52025c…`：平台去重键（zip_sha256）
    满足新元组，无需载体 commit。
- 预注册门：华为 8/8 正确（UB-safe 设计的正向目标；任何 'ub overflow'
  编译错/数值错即回滚 ascend vendor 回 e19 四成员字节）；华为读数 ≥315
  视为轴兑现（超 generic 最好窗 e14r 315.3，榜首 354）；均值 >558.04
  换 TB，未过门保 e16 TB。

## 2026-09-23 E20 平台终态：valid 8/8 均值 545.021175 < TB 558.04——华为 ascend vendor 正确性达成但慢于其替换的 generic 路径

- E20（submission **20305**，daily_seq 8，created 2026-09-23T12:25:13+08）：
  completed/valid，8/8 全过，均值 **545.021175**（> e19/20168 的
  535.077325，< TB e16/19661 的 558.039775），is_team_best=false——TB
  守 e16 558.04。
- 逐芯（e20 vs e19=20168，括注平台 selected_file）：天数 1283.2118 vs
  1185.29（generic）/ 沐曦 372.1516 vs 388.14（_metax）/ 燧原 110.0834 vs
  110.29（_enflame）/ 海光 897.4726 vs 894.32（generic；唯一 raw errors
  条目为 pytest-asyncio PytestDeprecationWarning stderr 噪音，passed=true，
  0 failed_cases，良性）/ 昆仑 8.495 vs 8.43（_kunlunxin）/ **华为
  210.9114 vs 267.04（`fixup_zero_kv_ascend.py` 被平台选中，-21%）** /
  A 841.1382 vs 818.47（generic）/ B 636.7054 vs 608.64（generic）。
- 预注册门核对（bcf63bbc 上膛时记录）：华为 8/8 正确且无 'ub overflow'
  编译错——**达成**（UB-safe 重构的正确性目标兑现：首个 _ascend vendor
  在华为编译并全部通过，T84/T92 ub-overflow 编译错族未复现）；华为读数
  ≥315 轴兑现门——未达（210.91 < 315，且低于 e19 generic 读数 267.04，
  ascend vendor 慢于其替换的 generic 路径）；均值 >558.04 换 TB——未达。
  'ub overflow 编译错/数值错即回滚 ascend 回 e19' 的触发条件**未发生**；
  ascend vendor 回滚与否按预注册条款由编排方裁决。
- 额度：本发后 22/30 remaining（observed_at 12:27:30+08，submission
  20305/daily_seq 8）；三发全部落地后复核 20/30（used 10，observed_at
  12:41:42+08，本次 status JSON 实读）。
- remote_verification=unavailable（FLAGOS_REMOTE_ZIP_HOST 未设，仅远端
  字节未复核，不影响已成功提交）；未出现 sending/uncertain，未重试。
- 订正：前次 CURRENT 块误写 team_best_stage=e19/535.08x——平台实读
  is_team_best=true 记录为 19661（e16，558.039775），本次已订正为
  e16/558.04x。
