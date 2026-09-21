# Task 84 `moe_align_block_size` 实验记录

```current
task: 84
operator: moe_align_block_size
batch: 6
validity: candidate(7/8,华为reference崩)
platform: completed(17721,e2r,7/8;七芯健康93.7-95.3)
candidate_stage: e2r
team_best_stage: -
team_best_speedup: -
sealed: no
next: 华为两发同指纹reference侧torch_npu RuntimeError(score 0)=崩溃族,封存等健康窗;昆仑scalar-serial vendor 2.7-2.8兑现在库;燧原cumsum vendor 4.6-4.8;有效即~95x
updated: 2026-09-19
```

## 过程摘要（2026-09-19 凌晨，题面 09-18 晚随批 6 扩容上线）

- 开发：s0 generic + 全 unittest 矩阵（commit `540b574a` 家族），代理
  release 多轮修复后 8/8 验证；按回执逐芯定位失败根因，vendor 修复弹
  按 codex-review 门后发射（review 抓出 moe_align_single_token 填充轮
  越界读 P1 并修复）。
- 回执与产物：见 `artifacts/competition/b6ext-*-20260919/`（verification
  + log 逐弹归档），ZIP 在各 `artifacts/competition/moe_align_block_size/` 下按 stage。
- 今日新增跨芯硬事实：XPU 拒 f32→i16 位转换（经 i32 高半字绕过）；
  XPU 拒 tt.scan/tl.cumsum 与 atomic_rmw；XPU 上张量索引 gather 与
  runtime 宽度寻址产生垃圾（hash_topk 三形态同指纹，标量串行为唯一
  可用形态）；GCU make_gcuir 拒张量索引 gather（torch 预 gather 先例
  再证）；XPU bf16 downcast 刀刃值与 eager 差 1 ulp（rtne 显式钉仍差）。

## 2026-09-21 E5 候选就绪并发射（串行 scan 尾部并行化）

- 结构（`ac2cb311`）：_scan 只留 num_routed 偏移循环（每专家 5 标量
  操作）；新增 _fill 并行内核（每专家一程序写 expert_ids 段 + 全网格
  并行哨兵毯）。旧单程序串行填充整个 buf 是全芯 8-17×（燧原 100×）
  落后的主嫌疑。预注册门：天数 ≥400 / 海光 ≥400；均值 >95.28×2 即
  结构兑现。
- 五元组：commit `ac2cb3115e85e74baa9163257018e163d5031063`；ZIP SHA `ebe9bd4c4cf876b2660d4908f5332d71d24c42c907bd6b1b009803d53173d800`；test
  `f19789f3b167bea8cc936c5f86a285e2f8db5bce939e9a971c6ee182da37744c`；回执 `top1day-20260921k/` SHA `8ccb31fde66b089643e0f5bcc4e91a1f36867d4e8d41ad3b946f18883805f099`。

## 2026-09-21 08:06 E5 平台终态（7/8 时）：串行填充并行化兑现 +34%~+125%

- 逐芯：天数 185→**298.1** / 沐曦 63→**141.7** / 燧原 4.7（vendor 未动，
  原子禁令串行版）/ 海光 162→**217.1** / 昆仑 2.8 / A 161→**305.3** /
  B 170→**182.5**；华为评估中（历史有 reference 侧 torch_npu 崩溃族）。
- **E6**：wrapper 单次零填充合并 counts/cursor/nblk（减少设备端内核
  数）。燧原/昆仑无原子并行重设计（+60 均值潜力）入账为明日首轴。
- 五元组：commit `1dbebc9252817f167a0800c747e21ad89732515d`；ZIP SHA `b4b34419f4085b2c653542dea6241a754792cda52e2b14bf99337408fcf9b98f`；test
  `f19789f3b167bea8cc936c5f86a285e2f8db5bce939e9a971c6ee182da37744c`；回执 `top1day-20260921l/` SHA `23d5c23e7257abc496c321656cad60fcb6b0bbb0b18bfc756277eed703696d7d`。

## 2026-09-21 08:23 E6 平台终态：150.52（+6.5），华为 0.0 崩溃族第二次

- 逐芯：天数 295.7 / 沐曦 **165.9**（瘦身 +17%）/ 燧原 4.7 / 海光 223.2 /
  昆仑 2.8 / A 313.6 / B 198.2 / **华为 0.0**（e5/e6 连续两次 reference
  侧崩溃族，与 e2r 同族——七芯健康）。若华为健康 ~250：均值 ~182。
  **E7 = e6 字节载体弹重掷华为**（崩溃族协议 ≤2 次）。明日首轴：
  燧原/昆仑无原子并行重设计（+60 均值潜力）。

## 2026-09-21 E7 候选就绪并发射（华为崩溃族重掷 1/2）

- 结构（`746fa12b`）：e6 字节注释载体（七芯健康 150.5 基础上重掷华为
  reference 侧崩溃）。预注册门：华为 >0 即均值 ≥165；健康 ~250 则 ~182。
- 五元组：commit `746fa12bffcd427e023bd3be0d694f9c8f426c32`；ZIP SHA `5c3d051c5508548ae50352e3bdaea1be3add75a40c16b08a686574b52868f62d`；test
  `f19789f3b167bea8cc936c5f86a285e2f8db5bce939e9a971c6ee182da37744c`；回执 `top1day-20260921n/` SHA `82f1caf9b83c726a8c4deaa8431c9224960c119d4e738b915c9ac14037c178dc`。

## 2026-09-21 08:48 E7 平台终态：147.37，华为 0.0 第三次（确定性）

- 华为 completed 无 error 但 speedup 0.0——与 e2r 同签名，且 e5/e6/e7
  连续三次：**平台 reference 侧确定性失败**（非间歇崩溃族），重掷预算
  尽。七芯健康带：天数 296-298 / 沐曦 141-166 / 海光 217-223 / A 305-322 /
  B 182-198。T84 今日定格 **150.52**（e6，95.28 起 +58%）。
- 明日轴：燧原/昆仑无原子并行重设计（+60 均值潜力）+ 发射数压缩
  （榜首 883 的 5× 差主嫌疑）。

## 2026-09-21 E8 候选就绪并发射（华为 0.0 根因修复：去 sem="relaxed"）

- **根因链（联网取证）**：回执栈 `torch_npu/npu/utils.py:72` = synchronize()
  （异步设备错误在同步点浮出）+ Triton-Ascend 官方文档明确 sem
  acquire/release/**relaxed** 不支持——我方 atomic_add(sem="relaxed")
  在昇腾 lowering 出错，错误延迟到 reference 同步点才爆，归因帧误导为
  reference 侧。T85 华为健康（ascend vendor 无原子）为旁证。
- 结构（`56fcd785`）：仅去掉两处 sem="relaxed"（回默认 acq_rel）。
  预注册门：华为 >0（若仍 0 则 reference 平台侧当前损坏，转报告组委
  会）；七芯不回退（150.52 基线）。
- 五元组：commit `56fcd785d20da24afdc9884145eb4016ade80862`；ZIP SHA `10d6f634818171239de42b48e709de3dc333c08a4d79c287f5623f5319b66721`；test
  `f19789f3b167bea8cc936c5f86a285e2f8db5bce939e9a971c6ee182da37744c`；回执 `top1day-20260921p/` SHA `958d9f1f1935fbf1a2c3f116194ce360227d778f18fc706ff9f70459df8d26d6`。

## 2026-09-21 E9 候选就绪并发射（共享无原子 vendor：华为修复 + 双芯换血）

- 背景：e8 去 relaxed 后华为仍 0.0 同栈——原子本身在该栈上异步致错
  （batch-5 dispatch_index 先例："runtime degrades under atomic cursors
  (Ascend)"，当年改为无原子三内核后华为 3.6-3.8 健康）。
- 结构（`0d482cee`）：_ascend/_enflame/_kunlunxin 共享无原子版（batch-5
  模板移植）：32 块计数表→每专家前缀→微串行 scan→并行毯→O(32²) 段内
  rank 确定放置。修复了 _mabs_fill 的 block_size 换算（review 前自查）。
  预注册门：华为 >0（同时检验 reference 健康）；燧原 ≥50 / 昆仑 ≥10
  （对齐 4.7/2.8 旧串行版）；其余五芯 150 基线不回退。
- 五元组：commit `0d482ceeab5b6acda10bd71e162630ca0e3401df`；ZIP SHA `08404f5e29c3f724d4d6114535d186fceb1258af682c130c2edbe254c056951d`；test
  `f19789f3b167bea8cc936c5f86a285e2f8db5bce939e9a971c6ee182da37744c`；回执 `top1day-20260921q/` SHA `319acf002d4b312d91ac2370267f2be5a08bf85e8e69f6b66b4b873b97fc4628`。

## 2026-09-21 14:23 E9 平台终态：华为 87.5 通过——异步原子致错根因确认

- **华为 0→87.5**（e8 去 relaxed 仍 0.0 同栈 → e9 全无原子即通过：
  该栈上 atomic_add 本身异步致错，错误浮出于 reference 同步点；
  reference 本身健康，e2-e8 五发的"reference 侧"归因是异步故障的
  归因错位）。燧原 4.7→**11.7**（无原子版 2.5×，但 K1 标量专家循环
  grid 2048 未按 GCU 模型封顶——明日 24 程序化冲 462）。
  昆仑 tl.sum 归约 PassManager 失败（XPU ban 清单 +1：K1 形态的
  tl.sum）。
- **E10**：昆仑回滚旧串行 vendor（2.8 健康），ascend/enflame 保共享
  无原子版。预注册门：均值 >154（含昆仑 2.8 即 ~154.6）。
- 五元组：commit `1e7e17c585ccf20428fbe1621f9c56c0410afdb6`；ZIP SHA `5719ca7b095fd52c63f6d9bcc90dae48a900acff7806ee631ad20d7893115024`；test
  `f19789f3b167bea8cc936c5f86a285e2f8db5bce939e9a971c6ee182da37744c`；回执 `top1day-20260921r/` SHA `c7b046be209d38b023c191c3d14aeafe658244581d04ce60ef378aa352001fed`。

## 2026-09-21 14:41 E10 平台终态：156.91 新 TB，华为修复收束

- 逐芯：天数 291.9 / 沐曦 138.3 / 燧原 11.7 / 海光 231.6 / 昆仑 2.7 /
  **华为 92.0**（连续两发通过，修复稳定）/ A 296.3 / B 190.8。
- 当日 T84 战线：95.28 → **156.91**（+65%）。华为 0.0 五发悬案闭环：
  联网取证（torch_npu utils.py:72 = synchronize 异步浮出点 +
  triton-ascend 文档 sem 支持清单）→ e8 证伪 relaxed 假设 → e9 全无
  原子版通过确认"原子在该栈异步致错"→ e10 组装。
- 明日轴：燧原共享版按 GCU 模型封 24 程序（11.7→462 潜力 = +56 均值）；
  华为 92→286 场带；昆仑 K1 换无 tl.sum 形态。

## 2026-09-21 E11 候选就绪并发射（燧原 GCU 程序模型，当日末发）

- 结构（`e602b91a`）：_enflame = 无原子模板 + 三计算内核封 24 grid-stride
  程序（K2a 补专家 stride；毯填充保持宽流）。预注册门：燧原 ≥50（冲
  462 场带的第一步）；其余七芯 156.91 基线不动。
- 五元组：commit `e602b91a0ab68ff6381d408518b3ddfe759b1c7b`；ZIP SHA `fb7d91f95a019241de712a4d75147c99f0bc88a48b64bbc585769f1e6706a5e2`；test
  `f19789f3b167bea8cc936c5f86a285e2f8db5bce939e9a971c6ee182da37744c`；回执 `top1day-20260921s/` SHA `724037674faadd299dea529d2dd4c83d91f12164a8dc508b81efcf1a11b43823`。

## 2026-09-21 17:20 E11 平台终态：163.14 新 TB，当日额度打满（30/30）

- 逐芯：天数 293.2 / 沐曦 **178.8**（generic 字节未动，+29% 窗口/环境
  波动）/ 燧原 10.5（24 程序化 vs 2048 网格 11.7——**该算子 GCU 瓶颈
  不在程序数**，疑在 K1 标量专家循环 O(E×B) 或 O(32²) rank 本身，
  明日换宽度/批量形态再探）/ 海光 231.2 / 昆仑 2.8 / 华为 97.0 /
  A 295.1 / B 196.5。
- 当日 T84 终值：95.28 → **163.14**（+71%）。
