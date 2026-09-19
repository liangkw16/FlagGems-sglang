# Task 85 `moe_align_single_token` 实验记录

```current
task: 85
operator: moe_align_single_token
batch: 6
validity: valid
platform: completed(18349,e6,8/8,29.991x<TB;保e3;hygon vendor兑现30但窗口回落)
candidate_stage: e6
team_best_stage: e3
team_best_speedup: 30.591
sealed: no
next: e4并行化判负(28.0<TB,标量单程序本就正确);距榜首31.83差4%;昆仑标量1.5/海光22.2为可挖轴
updated: 2026-09-19
```

## 过程摘要（2026-09-19 凌晨，题面 09-18 晚随批 6 扩容上线）

- 开发：s0 generic + 全 unittest 矩阵（commit `540b574a` 家族），代理
  release 多轮修复后 8/8 验证；按回执逐芯定位失败根因，vendor 修复弹
  按 codex-review 门后发射（review 抓出 moe_align_single_token 填充轮
  越界读 P1 并修复）。
- 回执与产物：见 `artifacts/competition/b6ext-*-20260919/`（verification
  + log 逐弹归档），ZIP 在各 `artifacts/competition/moe_align_single_token/` 下按 stage。
- 今日新增跨芯硬事实：XPU 拒 f32→i16 位转换（经 i32 高半字绕过）；
  XPU 拒 tt.scan/tl.cumsum 与 atomic_rmw；XPU 上张量索引 gather 与
  runtime 宽度寻址产生垃圾（hash_topk 三形态同指纹，标量串行为唯一
  可用形态）；GCU make_gcuir 拒张量索引 gather（torch 预 gather 先例
  再证）；XPU bf16 downcast 刀刃值与 eager 差 1 ulp（rtne 显式钉仍差）。

## 2026-09-20 E5 平台终态：hygon 22.2→30.3（+37%）但均值回落，TB 保 e3

- submission **18347** completed/valid，8/8，均值 28.928 < TB 30.591（保
  e3）。**海光 22.2→30.3（+37%，scalar-if 单程序 ranks + 256 并行填充——
  已达榜首档 30.8）**；但燧原 15.6→7.7 / 华为 8.2→7.7 / 天数 117.6→107.8
  窗口回落抵消。组合弹（hygon vendor + e3 其余芯）为下一发明确候选。

## 2026-09-20 E6 平台终态：组合兑现但仍低于窗口期 TB

- submission **18349** completed/valid，8/8，均值 29.991 < TB 30.591（保
  e3）。海光 29.9 / 燧原 14.3（vendor+generic 组合均健康）；回落来自
  天数 109.7（e3 窗口 117.6）与华为 7.4（e3 窗口 8.2）——**判读：TB
  30.591 含窗口溢价，当前组合结构不低于 e3**。距榜首 32.89 差 9%。
