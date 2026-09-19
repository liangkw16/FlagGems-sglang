# Task 87 `pack_topk_ids` 实验记录

```current
task: 87
operator: pack_topk_ids
batch: 6
validity: valid
platform: completed(17740,e3,8/8,2.761x新TB#3距榜首0.8%)
candidate_stage: e3
team_best_stage: e3
team_best_speedup: 2.761
sealed: no
next: e1 i32位转换vendor修复昆仑→e3 BLOCK2048(+2.3%新TB);e2 4096回退/e4 warps8微回;距榜首2.783仅0.8%,BLOCK 2048为峰档
updated: 2026-09-19
```

## 过程摘要（2026-09-19 凌晨，题面 09-18 晚随批 6 扩容上线）

- 开发：s0 generic + 全 unittest 矩阵（commit `540b574a` 家族），代理
  release 多轮修复后 8/8 验证；按回执逐芯定位失败根因，vendor 修复弹
  按 codex-review 门后发射（review 抓出 moe_align_single_token 填充轮
  越界读 P1 并修复）。
- 回执与产物：见 `artifacts/competition/b6ext-*-20260919/`（verification
  + log 逐弹归档），ZIP 在各 `artifacts/competition/pack_topk_ids/` 下按 stage。
- 今日新增跨芯硬事实：XPU 拒 f32→i16 位转换（经 i32 高半字绕过）；
  XPU 拒 tt.scan/tl.cumsum 与 atomic_rmw；XPU 上张量索引 gather 与
  runtime 宽度寻址产生垃圾（hash_topk 三形态同指纹，标量串行为唯一
  可用形态）；GCU make_gcuir 拒张量索引 gather（torch 预 gather 先例
  再证）；XPU bf16 downcast 刀刃值与 eager 差 1 ulp（rtne 显式钉仍差）。

## 2026-09-19 夜 e5 候选就绪（午夜首发第 1 弹）

- 结构（`ced1bda`）：metax vendor = BLOCK 2048 + warps 8（e4 全局 warps8
  只败在燧原，隔离到 vendor）——榜首逐芯显示**沐曦 1.9 vs 2.4（+26%）单
  此一项即覆盖 0.8% 的榜差**。预注册门：沐曦 ≥2.2；均值 >2.783 夺 #1。
- 五元组：commit `ced1bda534a85743285bcfa6178e32df9c1447d2`；
  ZIP SHA `1cc3d95134785e9feac33520488beb3b9b56f2699c0d4ab3e842f609502ee06a`；
  test SHA `d493f7af59e7f3605d1713741549510b260aa35a7b6eaf64bb2e67e39e38575a`；
  回执 SHA `a51a032389eb653a0ac4ea1f383659643a77a6025bf557e0f8a5f666c0132697`
  （`ready-wave-20260919-night/pack_topk_ids/`，双路径 2 成员）。
- codex-review：本波 commit 唯一 P1 为已弃 T85 hygon 提取错版（OOB，
 亦解释 rw-mast 跨模块污染），与本弹无关；弃弹后重建于干净 commit。
