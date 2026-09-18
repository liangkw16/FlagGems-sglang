# Task 80 `fixup_zero_kv` 实验记录

```current
task: 80
operator: fixup_zero_kv
batch: 6
validity: valid
platform: completed(17397,e6,8/8,198.65x<TB;保e4;沐曦warps+19%但天数/燧原-10%)
candidate_stage: e6
team_best_stage: e4
team_best_speedup: 200.920075
sealed: no
next: 明日首发e7组合=generic回默认warps+沐曦独享warps=8 vendor(投影TB~205.3,muxi 219/其余e5水位);慢窗最后一掷信号不变;额度30/30今日用尽
updated: 2026-09-18
```

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
