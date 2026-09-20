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

## 2026-09-20 测试修正：真中点 tie 覆盖（咨询第六轮纠错落地）

- 旧 `test_double_round_boundary` 用 2^-9 —— bf16 在 [1,2) ULP=2^-7，
  2^-9 只是 1/4 间距，从未触及 tie。改为三个真中点：+2^-8（1.0 偶尾数
  vs 1+2^-7 奇）、+3·2^-8（1+2^-7 奇 vs 1+2^-6 偶）、-2^-9（1-2^-8 vs
  1.0，1 下方 ULP 减半）；x=1/shift=0 使第二/三轮舍入无 tie，用例纯隔离
  one_plus 轮。
- 核对内核（当前 generic）三处显式舍入链与 reference 逐轮一致；昆仑
  刀刃容差的舍入定位诊断（构造 scaled/final 轮 tie）留待下一 GPU 窗口。
- 无新候选：仅测试字节变化，不影响既有回执绑定。
