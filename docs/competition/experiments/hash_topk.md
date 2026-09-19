# Task 82 `hash_topk` 实验记录

```current
task: 82
operator: hash_topk
batch: 6
validity: valid
platform: completed(18358,e6,8/8,6.316x新TB;e5泄漏字节教训+燧原24stages3)
candidate_stage: e6
team_best_stage: e6
team_best_speedup: 6.316
sealed: no
next: e4 constexpr-width vendor判负(昆仑垃圾指纹不变,标量唯一形态0.3);燧原torch-pregather vendor 0.8;距榜首7.15差17%,rank5;轴:昆仑向量形态XPU不可用已三证
updated: 2026-09-19
```

## 过程摘要（2026-09-19 凌晨，题面 09-18 晚随批 6 扩容上线）

- 开发：s0 generic + 全 unittest 矩阵（commit `540b574a` 家族），代理
  release 多轮修复后 8/8 验证；按回执逐芯定位失败根因，vendor 修复弹
  按 codex-review 门后发射（review 抓出 moe_align_single_token 填充轮
  越界读 P1 并修复）。
- 回执与产物：见 `artifacts/competition/b6ext-*-20260919/`（verification
  + log 逐弹归档），ZIP 在各 `artifacts/competition/hash_topk/` 下按 stage。
- 今日新增跨芯硬事实：XPU 拒 f32→i16 位转换（经 i32 高半字绕过）；
  XPU 拒 tt.scan/tl.cumsum 与 atomic_rmw；XPU 上张量索引 gather 与
  runtime 宽度寻址产生垃圾（hash_topk 三形态同指纹，标量串行为唯一
  可用形态）；GCU make_gcuir 拒张量索引 gather（torch 预 gather 先例
  再证）；XPU bf16 downcast 刀刃值与 eager 差 1 ulp（rtne 显式钉仍差）。

## 2026-09-20 E5/E6 平台终态：燧原 streaming vendor +12%，TB 6.316

- **E5（18357，7/8）教训**：56fb359 的 WIDTH-constexpr vector vendor 泄漏进
  ZIP（当时判负的实验字节未回退），昆仑复现 garbage-ids——**实验字节回退
  纪律**：判负的 vendor 改动必须立即回退或删除，否则后续组合弹携带已知
  坏字节。
- E6（18358）：scalar kunlun 恢复 + enflame 24-SIP+stages3 vendor。
  **燧原 0.8→0.9（+12%）**；均值 **6.316 新 TB**（+3%）。
