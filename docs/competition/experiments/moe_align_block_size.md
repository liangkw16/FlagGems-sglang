# Task 84 `moe_align_block_size` 实验记录

```current
task: 84
operator: moe_align_block_size
batch: 6
validity: valid(8/8,e14,492.17x TB)
platform: e14=492.17新TB(天数782/沐曦479/燧原253.9/海光904.1/昆仑2.70/华为91.7/A753/B671);e15/e16华为BiShengHIR UB溢出三连败,华为轴止损封存
candidate_stage: e16
team_best_stage: e14
team_best_speedup: 492.17x
sealed: no
next: 当日163.14→492.17(+202%);剩距c2flow 917.8=1.87x,缺口=华为91.7vs248(ascend新核三连编译败,封存)+昆仑2.7vs29+沐曦479vs718;明日再评估
updated: 2026-09-22
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


## 2026-09-21 E12 候选就绪（launch/分配结构重构，待 09-22 发射）

- 差距定性（逐芯快照）：c2flow 917.76 = 全芯均匀 5-8x（天数 2245/沐曦
  718/燧原 240/海光 1329/昆仑 29/华为 248/A 1407/B 1126），非单芯彩票。
  上游结构（GitHub agent 核对）：SGLang 现行 = 2 kernel + 0 memset + 0
  clone（单块共享内存直方图 + 3 级 warp scan + 二分填 expert_ids；scatter
  用 caller cumsum_buffer 当 cursor）。我方 e11 = memset + 4 kernel +
  clone + 3 alloc ⇒ launch/分配主导。
- e12（`eb1e2c6d`，generic-only 单变量，三 vendor 字节保持 e11 验证形态）：
  输出**原位写**（返回传入 buffer；eids 未定义尾=原值，与 reference clone
  语义逐位一致）；`_compute` 融合 launch（P 程序原子直方图 + ticket 最后
  程序做向量化 `tl.cumsum` scan + 逐 lane 变长填 expert_ids；其余程序并行
  毯填 sentinel）；cursor 落 caller `cumsum_buffer`；hist 原子加补 safe_e
  钳制（原版 e=-1 会越界）；`_scatter` 字节不动。发射 5→3，alloc 4→1。
- 五元组：commit `eb1e2c6d`；ZIP
  `artifacts/competition/moe_align_block_size/e12-eb1e2c6/moe_align_block_size.zip`
  SHA `02772ecd48f35516e9828890d2c8ce990a8bc52ed37c4f3f27fcab270bccd248`
  （4 成员）；回执 `day5prep-20260921/moe_align_block_size/verification.json`
  SHA `43502562067e7e0525d929f3b2bf4c1244a1f937bdce0dcc136ede7b9fb2838a`
  （2 测试 0 失败；generic 10 / ascend 25 / enflame 25 / kunlunxin 5 launch）。
- 预注册门：均值 >163.14 换 TB；任一走 generic 的芯（天数/沐曦/海光/A/B）
  相对 e11（293/179/231/295/196.5）回退 >10% 判负回滚；昆仑/华为/燧原走
  vendor 字节应复现 2.8/97/10.5。


## 2026-09-22 E12/E13 平台终态与 E14 发射

- **E12（19371）：8/8 valid 444.815 新 TB（+172%）**。逐芯：天数 783.7 /
  沐曦 567.8 / 燧原 12.5（vendor 老字节，窗口+19%）/ 海光 606.2 / 昆仑
  2.71 / 华为 91.2 / A 779.1 / B 715.2。三发射结构在全部 generic 芯
  翻倍兑现（293→784、179→568、231→606、295→779、197→715）——launch/
  分配主导假设完全验证。距 c2flow 917.8 剩 2.07x。
- E13（19388）：**燧原 253.1（反超 c2flow 的 240）**——无原子三发射核心
  + GCU 几何（12CTA/warps2/int32）兑现 12.5→253（20x）。generic 芯维持
  （783/475/658/791/676）。**华为 7/8 编译失败**：K1 的 32×512 compare
  矩阵要求 2686976 bits 超昇腾 UB 1572864 bits（192KB）——需专家轴分块。
  昆仑老 vendor 2.70 通过。判 invalid_correctness。
- E14（19394，`ad163b8a` 前身 commit）：enflame 保 e13 字节（253），
  ascend 回滚 e10 无原子模板（91.2 保底），kunlun 不动，generic 保 e12。
  预期 ~478（若各芯复现）。
- 剩余轴：华为 vendor 分块修复（e15，目标 91→248 场带）；昆仑 2.7→29
  （c2flow）；沐曦 568→718。


## 2026-09-22 E14 平台终态：492.17 新 TB；E15/E16 华为三连败止损

- **E14（19394）：8/8 valid 492.165 新 TB**。逐芯：天数 781.9 / 沐曦
  478.7 / 燧原 253.9 / **海光 904.1**（结构+窗口双兑现）/ 昆仑 2.70 /
  华为 91.7 / A 753.3 / B 671.0。当日 163.14 → 492.17（**+202%**）。
- E15（19402，E_CHUNK=128 分块）：华为同样 2686976-bit UB 溢出——分块
  未改变预算数字 ⇒ 消耗在分支联合缓冲，非 compare tile 本身。
- E16（19417，hist/fill 拆分 + epd constexpr + static_range）：华为仍
  BiShengHIR 失败（三连败）。**止损**：ascend 新核封存，华为轴以 e10
  老模板 91.7 保底；昆仑 vendor 老字节 2.70 稳定。
- 剩余差距解剖（vs c2flow 917.8）：华为 91.7 vs 248（+19.5 均值空间，
  需全新 ascend 内核形态——避开 2D tile 归约）、昆仑 2.70 vs 29（+33）、
  沐曦 478.7 vs 718（+30）。燧原 253.9 已反超 c2flow（240）。
