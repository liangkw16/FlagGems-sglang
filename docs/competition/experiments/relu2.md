# Task 89 `relu2` 实验记录

```current
task: 89
operator: relu2
batch: 6
validity: valid(8/8,e8,2.7198x TB)
platform: e9(20379)valid 8/8 avg 2.6600<TB判负:华为1.02→1.29(+26%,drop-prefill+4096/16在1载1存op兑现!)但昆仑0.135(vs e8字节同0.79,平台劣化)与B 2.75(-18%)吃掉增益;_ascend已回滚e8 ZIP字节;TB 2.7198守
candidate_stage: e9
team_best_stage: e8
team_best_speedup: 2.7198x
sealed: no
next: 华为1.29→1.9(金狐狸带)配方已证方向,待昆仑/B水位恢复可重试组合(e9字节+昆仑回温);真实靶CosmosMind 4.71(enfl 14.7/华为4.9);昆仑今日三题字节同一崩读(T90/T87/T89)=平台劣化窗;额度12/30(used 18,observed 16:10+08)
updated: 2026-09-23
```

## 2026-09-23 E9 平台终态（20379）：valid 8/8 均值 2.6600 判负；华为方向兑现

- 结构（`f8e73c6d`）：`_ascend` = 去掉 masked-load `other=0.0` 预填
  + BLOCK 1024→4096 + warps 8→16；kernel 体/generic/_enflame 冻结
  e8 字节。
- 逐芯：天数 4.31 / 沐曦 2.39 / 燧原 3.95 / 海光 3.20 / 昆仑 0.135
  （generic sha `878e3ef9` 与 e8 相同——平台侧）/ **华为 1.289
  （+26%）** / A 3.27 / B 2.75（-18%）。
- 判定：均值 <TB 且昆仑/B 破水位带 → 预注册回滚条款触发，_ascend
  已回滚 e8 ZIP 成员字节。华为 +26% 是真实结构增益（1 载 1 存 op
  上 drop-prefill+宽档兑现，与 T87 pack 的 -11% 形成 op 形状对照）。

## 2026-09-20 S0/E1 首发记录

- 开发+验证+codex-review 门后 00:12-00:27 发射（详见
  `artifacts/competition/nx-s0-20260920/`）。

## 2026-09-20 E1 平台终态：燧原 8192 档 +150%，TB 2.189

- submission **18314** completed/valid，8/8，均值 **2.189 新 TB**（vs 2.043，
  +7.1%）。**燧原 0.8→2.0（+150%，T76 8192 配方直移兑现）**；华为 1.1→1.0
  （persistent 无增益）；榜首 3.17 差 45%。

## 2026-09-20 E2 平台终态：warps8 双 vendor 兑现，TB 2.344

- submission **18331** completed/valid，8/8，均值 **2.344 新 TB**（vs 2.189，
  +7.1%）。**海光 2.5→3.2（+28%）、沐曦 2.0→2.4（+20%，warps8 在 relu2
  双芯为正——与 sgmb 的海光 -10% 互补，确认按 op×chip 组合定）**；华为
  1.0→1.1。距榜首 3.17 差 26%。

## 2026-09-20 E3 平台终态：generic warps8 再兑现，TB 2.497

- submission **18350** completed/valid，8/8，均值 **2.497 新 TB**（vs 2.344，
  +6.5%）。天数 4.3→4.4 / A 2.5→3.3（+32%，generic 路径芯片受益 warps8）；
  海光 3.2→3.3 持平。距榜首 3.17 差 21%。

## 2026-09-21 夜 e4 候选就绪（午夜第 3 弹）

- 结构（`f3b8cad1`）：enflame BLOCK 8192→**16384 档** + ascend vendor 加
  num_warps=8——榜首燧原 4.0 vs 我方 2.0（2x）、华为 1.8 vs 1.1。预注册门：
  燧原 ≥2.6 或华为 ≥1.4；均值 >2.497 换 TB。
- 五元组：commit `f3b8cad1`；ZIP `e4-f3b8cad` SHA
  `df172a2c63c43e5de319b4e6790bfab311f011901af406ae5bb0567bf8176bd3`；
  回执 `ready2-wave-20260921/relu2/`（五路径）。

## 2026-09-21 00:10 E4 平台终态：2.5536 新高（+0.057），TB 刷新

- submission completed/valid。逐芯：天数 4.37 / 沐曦 2.37 / 燧原 2.61 /
  海光 3.24 / 昆仑 0.78 / 华为 1.0 / A 3.33 / B 2.72（enflame 16384 +
  ascend warps8 部分兑现）。榜首 3.241（EvokeAgent）：差 -0.69 均值，
  逐芯均匀（-0.4~-1.5），需广谱结构非单芯补洞。

## 2026-09-21 E5 候选就绪并发射（燧原回退 flat generic 网格）

- 结构（`4e5cdc02`）：_enflame 从 24-SIP 封顶 BLOCK 16384 回退 generic
  字节（flat 1D，全网格 2048，warps8）——行形式在 GCU 已双败（T90/T86），
  本弹验证 flat 形态全网格是否兑现（天数同字节读 4.37）。
  预注册门：燧原 ≥3.0；均值 >2.5536 换 TB。
- 五元组：commit `4e5cdc02b22f3c9654405160d648e02dc6182ab8`；ZIP `e5-4e5cdc0` SHA `ba5eba9ccb35651ed71519066b8b7fc86109c1361957d1e1b659f3bbfc727db2`；
  test `37b246090e6274a43e5c2968e5bfa400987c2926e82a81ca42a019ce9f2cfa30`；回执 `top1day-20260921b/relu2/` SHA `eef3070d6f8fc4a765b2f6c68756d140d393674e4d45def066047993b29e6a22`。

## 2026-09-21 00:47 E5 平台终态：燧原 0.5 三败，e4 的 24-SIP 形态才是 GCU 正解

- flat generic 全网格在 GCU 读 **0.5**（< e4 的 24-SIP BLOCK 16384 的 2.61）。
  至此 GCU 三种形态三败（行形式/flat 多程序/仅少程序+超宽可行）。
  TB 保 e4 2.5536。E6 轴：燧原回 24-SIP 形态 + 宽度 32768 阶梯。

## 2026-09-21 E6 候选就绪并发射（燧原回 24-SIP 形态，宽度 32768）

- 结构（`ad9f2a29`）：_enflame 恢复 e4 的 24-SIP 形态（grid
  min(cdiv(numel,32768),24)），BLOCK 16384→**32768**——e5 全网格判负
  （0.5）后按宽度阶梯续爬。预注册门：燧原 ≥3.2；均值 >2.5536 换 TB。
- 五元组：commit `ad9f2a298aea3675cade53171fb6c961c877be2e`；ZIP `e6-ad9f2a2` SHA `ec939dad275142380ec0c4ced44bd2805bd85fa5d619fbf638238074009e99b4`；
  test `37b246090e6274a43e5c2968e5bfa400987c2926e82a81ca42a019ce9f2cfa30`；回执 `top1day-20260921c/relu2/` SHA `d86d2cb4b17da0a5d3eb70c9ab0d91176e7eb1ffd1852ae9b45e8ba4dca62e90`。

## 2026-09-21 00:55 E6 平台终态：2.6568 新 TB（+0.103），燧原 3.3 兑现

- 逐芯：天数 4.4 / 沐曦 2.4 / 燧原 **3.3**（24-SIP 32768：2.61→3.3，
  宽度阶梯 65536 续爬）/ 海光 3.3 / 昆仑 0.8 / 华为 1.0 / A 3.4 / B 2.7。
  **E7 轴**：燧原 65536 + 海光 BLOCK 4096。

## 2026-09-21 E7 候选就绪并发射（燧原 65536 / 海光 4096）

- 结构（`36d980af`）：_enflame BLOCK 65536（3.3@32768 续爬）；_hygon
  BLOCK 4096（3.3@1024 宽度阶梯）。预注册门：燧原 ≥3.8 / 海光 ≥3.6。
- 五元组：commit `36d980af7a95eae8143ca3e9e0e687a378d93155`；ZIP `e7-36d980a` SHA
  `86dff4ad3743fcfd00a9a51e1a52e4c243cf6da083c4589b8f5a218d07a55b0a`；
  test `37b246090e6274a43e5c2968e5bfa400987c2926e82a81ca42a019ce9f2cfa30`；
  回执 `top1day-20260921d/relu2/` SHA
  `2b95d93159ed8a66ecc4d9f7d65ef7348c7d779e8786ec495e652ab5e7e1d572`。

## 2026-09-21 01:10 E7 平台终态：2.5369 判负，海光方向反转确认

- 海光 BLOCK 4096 = 3.3→**1.8**（海光要窄 1024，与华为宽度方向相反——
  宽度阶梯按 op×chip 分化的又一实证）；燧原 65536 = 3.3→**3.9** 进场带
  （3.78-4.12）但被海光回落吞掉。TB 保 e6 2.6568。E8 轴：海光回 1024
  + 燧原 131072 探顶。

## 2026-09-21 E8 候选就绪并发射（海光回 1024 + 燧原 131072 探顶）

- 结构（`8b47cce3`）：_hygon BLOCK 回 1024（e7 判负回滚）；_enflame
  131072（2.61→3.3→3.9 续爬）。预注册门：燧原 ≥4.05；均值 >2.6568 换 TB。
- 五元组：commit `8b47cce370af2555d7060be37e17095b02e6e1bf`；ZIP `e8-8b47cce` SHA `2e84243814b385ca51788edf5fc553839f3692ce1b4b252351f4ae8b416ec44c`；
  test `37b246090e6274a43e5c2968e5bfa400987c2926e82a81ca42a019ce9f2cfa30`；
  回执 `top1day-20260921e/relu2/` SHA `4389467907de88254d2b3171061598fb5ef9b0d855b0ca12bde95fb9e3b420e1`。

## 2026-09-21 01:27 E8 平台终态：2.7198 新 TB，燧原 3.86 顶确认

- 逐芯：天数 4.4 / 沐曦 2.34 / 燧原 **3.86**（131072 持平 65536——宽度顶
  在 65536，与场带 3.78-4.12 合）/ 海光 3.24（1024 回滚兑现）/ 昆仑 0.79 /
  华为 1.02 / A 3.33 / B 2.77。剩洞：沐曦 2.34 vs 3.14+、海光 3.24 vs
  3.5-4.3、华为 1.02 vs 1.25-1.9——结构未知，T89 进入 2.72-2.85 平台。


## 2026-09-21 E4 候选就绪（燧原官方 gcu300 规则集，待 09-22 发射）

- 官方源码调研（FlagGems _enflame gcu300 codegen / FlagTree enflame
  backend）：max_grid_size=(12,1,1)（grid-stride 吸收超额）、
  enflame_heuristics_for_num_warps 钉 2、stride 需编译期互整除才走 DMA。
  本题 vendor 应用：grid 24→12（warps 不钉——2-warp 131072 块在代理编译即病态）。
- 五元组：commit `245a7c65`；ZIP
  `artifacts/competition/relu2/e4-f3b8cad/relu2.zip`
  SHA `df172a2c63c43e5de319b4e6790bfab311f011901af406ae5bb0567bf8176bd3`；成员 generic/ascend/enflame/hygon/metax（generic 与非燧原 vendor 字节
  不变，账本明确列全）；回执 `day5prep-20260921/relu2/verification.json`
  SHA `698e909572e2d728…`。
- 预注册门：燧原 3.86→≥7.7(×2)；其余七芯不动（vendor-only 单变量）。codex-review
  零发现（grid-stride 边界模拟无漏算）。
