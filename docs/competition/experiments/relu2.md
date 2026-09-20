# Task 89 `relu2` 实验记录

```current
task: 89
operator: relu2
batch: 6
validity: valid
platform: completed(18350,e3,8/8,2.497x新TB;天数/A +8% generic warps8)
candidate_stage: e3
team_best_stage: e3
team_best_speedup: 2.497
sealed: no
next: 流式elementwise;轴=燧原0.8/昆仑0.8/华为1.1 vendors(streaming配方);榜首差距55%较大
updated: 2026-09-20
```

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
