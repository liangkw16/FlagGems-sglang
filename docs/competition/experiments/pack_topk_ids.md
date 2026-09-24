# Task 87 `pack_topk_ids` 实验记录

```current
task: 87
operator: pack_topk_ids
batch: 6
validity: valid(8/8,e5r2,3.43741667x TB)
platform: e5r2(20793)valid 8/8 avg 3.4374>TB 3.3715再刷新(+2%):昆仑7.137(水位稳住)/华为5.5745(上行)/天数3.51;距金狐狸3.578(#2)差4.1%未越;额度30/30用尽,窗口19:59关闭,本题收盘
candidate_stage: e5r2
team_best_stage: e5r2
team_best_speedup: 3.43741667x
sealed: yes
next: 收盘:rank#3(榜首CosmosMind 5.64华为27.3疑窗口,#2金狐狸3.578);本题全程2.77→3.437(+24%),全部来自水位重掷(字节未变)
updated: 2026-09-24
```

## 2026-09-24 E5R2 平台终态（20793，末发）：valid 8/8 均值 3.4374 再刷新 TB

- 结构：e5 字节 + 注释载体（第二抽）。逐芯：天数 3.507 / 沐曦 1.955 /
  燧原 2.285 / 海光 2.231 / **昆仑 7.137** / **华为 5.574** / A 2.228 /
  B 2.583。
- 免费赌博结算：昆仑水位第二抽 7.14（稳）+ 华为 5.17→5.57（上行）→
  +2% TB;#2 金狐狸 3.578 差 4.1% 未越。本题收盘 rank #3。

## 2026-09-24 E5R 平台终态（20786）：valid 8/8 均值 3.3715 新 TB；昆仑水位爆发

- 结构：e5 字节 + 注释载体（generic 头部 e5r 标记行，metax 冻结）。
- 逐芯：天数 3.496 / 沐曦 2.025 / 燧原 2.281 / 海光 2.151 /
  **昆仑 7.0395（e5 时代 1.27，e6 午后曾崩 0.394）** / 华为 5.171 /
  A 2.242 / B 2.568。
- 门判定：均值 >2.77 ✓ 换 TB;华为 ≥7.5 未达（水位论对华为不成立，
  对昆仑大幅超预期——预测的载体错、方向对）。

## 2026-09-23 E6 平台终态（20377）：valid 8/8 均值 2.5149 判负；ascend 形态反向

- 结构（`4dd7f577`）：首个 `_ascend` vendor——persistent Vector-Core
  网格 + BLOCK 4096 + warps 16 + 无 other 预填;kernel 用证明过的
  f32→bf16→f32 转换链（整数 RNE 原型被代理复现证伪：torch CUDA 把
  qNaN 规范化为 0x7FFF 而公式给 0x7FC0，NaN 语义芯片相关）。
- 逐芯：天数 3.30（-5%）/ 沐曦 2.01 / 燧原 2.30 / 海光 2.22（-7%）/
  昆仑 0.394（**字节同一崩读**，generic sha `83dc5944` 与 e5 相同）/
  华为 5.00（**-11%**）/ A 2.33 / B 2.57。
- 判定：预注册门（华为 ≥7.0）未达且均值 <TB；_ascend 移除，成员集回
  e5（generic+_metax）。
- 结构知识：**ascend launch 形态响应是 op 形状函数**——1 载 1 存的
  relu2 上 drop-prefill+宽档 +26%，2 载 1 存的 pack 上同配方 -11%；
  后续 ascend vendor 假设需按访存比分类。

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

## 2026-09-20 E5 平台终态：微幅新 TB 2.77，沐曦门未过

- submission **18252** completed/valid，8/8，均值 **2.77 新 TB**（+0.3%）。
  沐曦 1.9→2.0（未到 2.2 门——8 warps 在此题收益有限）；榜首升至 3.58
  （Sweetdeath 4096 档），差距扩至 29%。
