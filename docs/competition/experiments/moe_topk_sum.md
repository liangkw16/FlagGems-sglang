# Task 86 `moe_topk_sum` 实验记录

```current
task: 86
operator: moe_topk_sum
batch: 6
validity: valid
platform: completed(17752,e3,8/8,2.754x新TB;warps8第四次兑现)
candidate_stage: e3
team_best_stage: e3
team_best_speedup: 2.754
sealed: no
next: e1双vendor(燧原streaming+100%/华为persistent+27%)→e3 warps=8(+9.7%,海光+36%);e2 BLOCK2048回退/e4 warps16沐曦超限;距榜首3.729差26%;轴:天数4.8/沐曦2.6/海光4.9-6.0仍有空间
updated: 2026-09-19
```

## 过程摘要（2026-09-19 凌晨，题面 09-18 晚随批 6 扩容上线）

- 开发：s0 generic + 全 unittest 矩阵（commit `540b574a` 家族），代理
  release 多轮修复后 8/8 验证；按回执逐芯定位失败根因，vendor 修复弹
  按 codex-review 门后发射（review 抓出 moe_align_single_token 填充轮
  越界读 P1 并修复）。
- 回执与产物：见 `artifacts/competition/b6ext-*-20260919/`（verification
  + log 逐弹归档），ZIP 在各 `artifacts/competition/moe_topk_sum/` 下按 stage。
- 今日新增跨芯硬事实：XPU 拒 f32→i16 位转换（经 i32 高半字绕过）；
  XPU 拒 tt.scan/tl.cumsum 与 atomic_rmw；XPU 上张量索引 gather 与
  runtime 宽度寻址产生垃圾（hash_topk 三形态同指纹，标量串行为唯一
  可用形态）；GCU make_gcuir 拒张量索引 gather（torch 预 gather 先例
  再证）；XPU bf16 downcast 刀刃值与 eager 差 1 ulp（rtne 显式钉仍差）。
