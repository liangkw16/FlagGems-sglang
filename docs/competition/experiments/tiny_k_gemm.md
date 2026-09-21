# Task 91 `tiny_k_gemm` 实验记录

```current
task: 91
operator: tiny_k_gemm
batch: 6
validity: valid
platform: completed(18345,e7,8/8,1.953x=TB;metax16/kunlunw8中性,阶梯到顶)
candidate_stage: e8
team_best_stage: s0
team_best_speedup: 1.953
sealed: no
next: metax BLOCK_N32修复smem后#2;距榜首3.4%:轴=BLOCK_N阶梯(64于非K256形状)/m16rows;昆仑1.1/华为0.8 vendor
updated: 2026-09-21
```

## 2026-09-20 S0/E1 首发记录

- 开发+验证+codex-review 门后 00:12-00:27 发射（详见
  `artifacts/competition/nx-s0-20260920/`）。

## 2026-09-20 E2 平台终态：双 vendor 中性，TB 保 1.953

- submission **18308** completed/valid，8/8，均值 1.951 ≈ TB 1.953（保 e1）。
  沐曦 1.8（warps8 无增益）、燧原 0.6→0.7（微）；B 2.8。距榜首 2.016 差
  3.3%，缺口在沐曦 1.8 vs 2.4 与 B 2.8 vs 3.0——无已知配方，关闭本轴。

## 2026-09-20 E7 平台终态：BLOCK_N/warps 阶梯到顶，TB 保 1.953

- submission **18345** completed/valid，8/8，均值 1.953 = TB（持平）。沐曦
  1.7（16 档无增益）、昆仑 1.1（warps8 无增益）、华为 0.8→0.9；BLOCK_N
  128 因 smem 149KB 不可发（K=256 tile 物理上限）。T91 全部已知轴关闭。

## 2026-09-21 夜 e8 候选就绪（午夜首发第 2 弹，Top1 冲刺）

- 结构（`3337be6b`）：metax vendor BLOCK_N 16→**64 + num_stages=1**——
  s0 时 BLOCK_N=64 需 86KB smem（默认 3 阶段）超沐曦 64KB 上限；
  单阶段把 tile 降到 ~40KB 物理可行，**榜首沐曦 2.3 vs 我方 1.8 是
  唯一有效缺口**（+0.5 芯分 ≈ +0.06 均值即够反超 2.032）。预注册门：
  沐曦 ≥2.1；均值 >1.953 换 TB，>2.032 冲 #1。
- 五元组：commit `3337be6b`；ZIP `e8-3337be6` SHA
  `1b0a69f1189478081b21ee253571dba938f701531773bf22c6dace9bc1ac5960`；
  回执 `top1-wave-20260921/tiny_k_gemm/`（四路径）。

## 2026-09-21 00:08 E8 平台终态：1.9739 新 TB（+0.021），沐曦 1.9 门未过

- submission completed/valid。逐芯：天数 2.73 / 沐曦 1.9（门 ≥2.1 未过，
  BLOCK_N64+单阶段仅 +0.1）/ 燧原 **0.67** / 海光 2.76 / 昆仑 1.09 /
  华为 0.83 / A 2.93 / B 2.88。A/B/海光已超榜首同芯。
- **榜首 2.032 → 2.2448**（EvokeAgent，天数 4.49 拉动）。Δ 归因：天数
  -1.76、燧原 -1.23 两洞合计 -0.37 均值 > 总差 0.27——**下一轴 = generic
  GCU 结构 + 燧原 vendor**。


## 2026-09-21 E8 候选就绪（燧原官方 gcu300 规则集，待 09-22 发射）

- 官方源码调研（FlagGems _enflame gcu300 codegen / FlagTree enflame
  backend）：max_grid_size=(12,1,1)（grid-stride 吸收超额）、
  enflame_heuristics_for_num_warps 钉 2、stride 需编译期互整除才走 DMA。
  本题 vendor 应用：grid 24→12+ num_warps=2 + 运行时 stride 参数 constexpr 化（DMA 判定编译期可证）。
- 五元组：commit `14437ad3`；ZIP
  `artifacts/competition/tiny_k_gemm/e8-14437ad/tiny_k_gemm.zip`
  SHA `14379b21bbf057d40c34581aaae829fabb0a9b67dd493ac3a955dfcdc5f69994`；成员 generic/enflame/kunlunxin/metax（generic 与非燧原 vendor 字节
  不变，账本明确列全）；回执 `day5prep-20260921/tiny_k_gemm/verification.json`
  SHA `8bb2c26447e6d7d4…`。
- 预注册门：燧原 0.67→≥1.34(×2)；其余七芯不动（vendor-only 单变量）。codex-review
  零发现（grid-stride 边界模拟无漏算）。
