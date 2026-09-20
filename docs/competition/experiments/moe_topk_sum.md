# Task 86 `moe_topk_sum` 实验记录

```current
task: 86
operator: moe_topk_sum
batch: 6
validity: valid
platform: completed(18337,e6,8/8,2.84x<TB;保e5 2.908;燧原2D-tile中性偏负)
candidate_stage: e6
team_best_stage: e5
team_best_speedup: 2.908
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

## 2026-09-19 夜 e5 候选就绪（午夜首发第 2 弹）

- 结构（`ced1bda`）：hygon vendor = generic（warps 8）提升到 warps 16
  （T81 先例：海光 16-warp 恢复 +12%）——榜首逐芯显示**海光 4.9 vs
  6.4-7.1（+30-45%）**为最大缺口。预注册门：海光 ≥5.5；均值 >2.754 换 TB。
- 五元组：commit `ced1bda534a85743285bcfa6178e32df9c1447d2`；
  ZIP SHA `cab602040c838480c8b7f9eafc8d3c31f6b1b4751c81e2d077e2d93325239bee`；
  test SHA `93fc9dd8d253aebafb93078eb81c56c275d254d4f24e05d332fdc009d523c807`；
  回执 SHA `8a3b37216195275724a92fe846c2a6a9964c7cb8e78778ca4a57299ec215ed65`
  （`ready-wave-20260919-night/moe_topk_sum/`，四路径 4 成员）。

## 2026-09-20 E5 平台终态：hygon 16-warp +34%，TB 2.908

- submission **18255** completed/valid，8/8，均值 **2.908 新 TB**（vs 2.754，
  +5.6%）。**海光 4.9→6.1（+24%，e1 后再 +34% 累计）**；天数 4.9；距榜首
  3.73 差 22%。

## 2026-09-20 E6 平台终态：燧原 2D-tile 中性偏负，TB 保 e5 2.908

- submission **18337** completed/valid，8/8，均值 2.84 < TB 2.908（保 e5）。
  燧原 0.8→0.5（2D [TOPK_PAD, BLOCK] tile 形态在 GCU 为负——榜首 9.0 的
  结构另有来源，轴关闭）；天数 4.9→5.0 / 海光 6.0 窗口新高。

## 2026-09-21 E8 候选就绪并发射（燧原 vendor 回退 generic 扁平流）

- 结构（`76982f26`）：_enflame 从 2D [TOPK_PAD, BLOCK] tile（读数 0.53，
  跨步行 tile 违背 GCU flat streaming 偏好）回退 generic 字节（天数同源
  读 4.96）。预注册门：燧原 ≥2.0；均值 >2.840 换 TB。
- 五元组：commit `76982f269271ac4c0def557403940ddfe7610bd6`；ZIP `e8-76982f2` SHA
  `e2696eff1d325c0d7277d5d077c2a6cd4ff5c232f21808d359f5bc622e3adc84`；
  test `93fc9dd8d253aebafb93078eb81c56c275d254d4f24e05d332fdc009d523c807`；
  回执 `top1day-20260921/moe_topk_sum/` SHA
  `8276d759189e1dc7c5dc4b4f65e319761b76ec032fb8d643da71d0f706154c27`。

## 2026-09-21 00:40 E8 平台终态：燧原 0.24 判负（flat generic 字节同败）

- 7/8 完成时燧原已 0.24（< 2D tile 形态的 0.53）——flat 流 + TOPK 静态
  展开在 GCU 同样病理。与 T90 e5 判负互证：**generic 字节移植不是燧原
  答案，GCU 需逐 op 专属结构**。其余七芯与 e7 基线一致（沐曦 2.6/海光
  6.0/华为 1.83 等），均值回到 ~2.84 带。

## 2026-09-21 00:40 E8 平台终态终值：2.8210（燧原 0.24 判负），TB 保 e7 带

- 8/8：天数 5.0 / 沐曦 2.6 / 燧原 **0.24**（flat generic 也病理，与 T90 e5
  行形式 0.63 互证 GCU 需少程序+超宽块）/ 海光 6.0 / 昆仑 0.6 / 华为 1.83 /
  A 3.24 / B 3.10。燧原轴待 GCU 模型（24-SIP + BLOCK 16384 形态）再试。

## 2026-09-21 E9 候选就绪并发射（燧原 GCU 模型移植）

- 结构（`985db4e4`）：_enflame = generic 内核 + GCU 程序模型（grid
  min(rows,24) × splits≤4，BLOCK 16384，num_stages 3）——行形式/flat
  全网格两败后的第三形态。预注册门：燧原 ≥1.5；均值 >2.840 换 TB。

- 五元组：commit `985db4e49de59c57e5f6891af7ef0be4e20ae665`；ZIP SHA `641764b41de8035498a8b5ded89c75c171ca0cdf2c3f26ce243197593d5c06bf`；test `93fc9dd8d253aebafb93078eb81c56c275d254d4f24e05d332fdc009d523c807`；回执 SHA `478f07ac65c83a0fb95a7f5045953863967f09430f8213f42dc329daf6f97859`。
