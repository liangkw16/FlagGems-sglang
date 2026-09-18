# Task 83 `indexed_scale_shift` 实验记录

```current
task: 83
operator: indexed_scale_shift
batch: 6
validity: candidate(7/8,昆仑1元素容差)
platform: completed(17729,e2,7/8;昆仑0.0156vs0.015刀刃)
candidate_stage: e2
team_best_stage: -
team_best_speedup: -
sealed: no
next: 三轮逼近(0.0215→0.0156 vs 0.015允许)但未过:两pass拆分+显式rtne vendor均差一元素;接受7/8不可有效,除非XPU bf16舍入语义新证据
updated: 2026-09-19
```

## 过程摘要（2026-09-19 凌晨，题面 09-18 晚随批 6 扩容上线）

- 开发：s0 generic + 全 unittest 矩阵（commit `540b574a` 家族），代理
  release 多轮修复后 8/8 验证；按回执逐芯定位失败根因，vendor 修复弹
  按 codex-review 门后发射（review 抓出 moe_align_single_token 填充轮
  越界读 P1 并修复）。
- 回执与产物：见 `artifacts/competition/b6ext-*-20260919/`（verification
  + log 逐弹归档），ZIP 在各 `artifacts/competition/indexed_scale_shift/` 下按 stage。
- 今日新增跨芯硬事实：XPU 拒 f32→i16 位转换（经 i32 高半字绕过）；
  XPU 拒 tt.scan/tl.cumsum 与 atomic_rmw；XPU 上张量索引 gather 与
  runtime 宽度寻址产生垃圾（hash_topk 三形态同指纹，标量串行为唯一
  可用形态）；GCU make_gcuir 拒张量索引 gather（torch 预 gather 先例
  再证）；XPU bf16 downcast 刀刃值与 eager 差 1 ulp（rtne 显式钉仍差）。
