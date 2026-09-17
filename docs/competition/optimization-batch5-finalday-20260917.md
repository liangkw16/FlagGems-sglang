# 第五批最后一天作战方案（2026-09-17）

截止 19:59:59，今日额度 **30/30 全部可用**（10:59 `status` 实测，账号全局共享、120s 间隔）。
本方案由"更新榜单 → 逐芯对比 → 根因分析 → 联网/GitHub/本地经验"全链路调研产出；
执行时逐题建候选、过发布门禁后按既有授权自动提交。

## 1. 榜单快照与逐芯差距

- 全 17 题逐芯快照（每队每芯 speedup）：`data/leaderboard-batch5-perchip-20260917.json`，
  observed_at `2026-09-17T10:57:03`，SHA-256
  `2247249c555ad23e34cde481760c1a35e6a4885bd8142e8d46d2f284db095e1c`。
- 15 题有效、2 题无效（T60/T70）。我方名次：T59:8 T61:6 T62:12 T63:10 T64:14 T65:13
  T66:7 T67:12 T68:10 T69:5 T71:15 T72:6 T73:7 T74:8 T75:11。
- 彩票判据复核（榜首某芯 > 同芯次优 20x）：T63 华为 1255.97 vs 次优 91.95（13.7x，
  接近但未过线）；T73 燧原 18.29 vs 次优 5.18；T70 燧原 126.7 vs 次优 0.98（129x，纯彩票）。
  **真实可达目标一律取"同芯次优队读数"**，不用榜首极值定门。

### 逐芯差距表（ours → 同芯次优，Δ = 抬到次优可得的均值增量）

| 题 | 华为 | 燧原 | 天数 | 昆仑 | 其他大头 | 合计潜力 |
|---|---|---|---|---|---|---|
| T65 post_reorder | 17.5→244.7 | 8.2→35.7 | 31.5→57.0 | 0.38→5.07 | card_a 48.9→69.7 | **+40.6** |
| T64 permute | 3.1→100.9 | 6.7→12.1 | 16.0→30.8 | 0.27→1.49 | haiguang 10.2→22.1 | **+26.5** |
| T69 dispatch_idx | 3.8→7.9 | 0.76→24.6 | 77→210 | 0.102→7.8 | tianshu 主导 | **+26** |
| T59 page_table | 8.7→82.0 | 28.1→20.0(已高) | — | — | muxi 12.8→14.9 | +9.2 |
| T63 kv_indices | 73→92 | 6.2→83.1 | — | 2.7→13.0 | muxi 87→137 | +12 |
| T67 fill_padded | 1.9→17.5 | 1.5→5.7 | 11.9→18.4 | 0.59→1.2 | — | +3.4 |
| T68 eh_norm | 3.8→4.2 | 1.7→41.7 | — | — | — | +5.0(燧原) |
| T75 sigmoid_mul | 1.3→20.3 | 3.3→3.8 | — | — | — | +2.4 |
| T72 group_norm | 1.2→9.2 | 0.38→1.7 | 4.2→6.6 | — | — | +1.6 |
| T73 residual_add | 1.1→11.3 | 2.1→5.2 | — | — | muxi 4.8→5.8 | +1.9 |
| T74 seqlens | 7.6→8.2 | 9.2→22.8 | 59.5→87.4 | — | card_b 16.7→19.6 | +5.5(风险高) |
| T66 a_gemm | 1.8→3.4 | 0.42→3.1 | 1.3→4.4 | — | card_b 5.9→6.0 | +2.4 |
| T71 gelu_tanh | 1.9→7.3 | 2.0→4.4 | — | — | — | +1.0 |
| T62 concat_mla | 0.21→1.14 | 0.28→3.96 | — | 0.18→0.78 | muxi 1.0→1.6 | +0.8 |
| T61 src2dst | 1.7→4.3 | 4.4→5.5 | — | — | — | +0.6 |

## 2. 根因结论（瓶颈分析）

### 结论 A（主攻）：华为轴系统性"大 grid 反模式"，第五批从未应用 persistent 形态

昇腾官方 Vector Operator 指南（本地缓存 `data/vendor-backends/ascend/vector_operator.md`，
pin Ascend/triton-ascend `865691e`）原文：*"The key is not to create as many grid programs as
possible, but to keep the launch close to the number of physical Vector Cores and let each
program process multiple tiles in an inner loop… GPU-style small tiles with very large grids
often cause repeated dispatch overhead on NPUs"*；BLOCK 应在 UB 预算内尽量大。

我方第五批华为轴现状全部踩反模式（65535 级 grid 或每行一 program）：

| 题 | 华为跑的路径 | grid 形态 | body 已有 stride 循环？ |
|---|---|---|---|
| T75 | `_ascend`(=generic+warps8) | `min(cdiv(n,4096),65535)` | ✅ |
| T73 | generic | flat `min(cdiv,65535)` / 2D `(rows,d/B)` | flat ✅ / 2D 需核 |
| T71 | generic | `min(rows,65535)` | ✅ |
| T67 | generic | `(n_rows,)` 每行一 program | ❌（需加循环） |
| T64 | generic | `min(tasks,65535)` | ✅ |
| T65 | generic(E12 修复版) | 有界双轴 ≤65535 | ✅（双轴都 stride） |
| T63 | generic | `(min(batch,65535), splits)` | ✅（axis0 stride） |
| T59 | generic(TB zip 无 ascend 成员) | `min(tasks,65535)` | ✅ |
| T68 | generic（无 _ascend 文件） | `(tokens,)` | 需核 |
| T72 | `_ascend`(e9 算法) | 需核 | 需核 |

仓库内已证模板：`_ascend/ops/decode_attention.py:21-27`
`_get_num_vector_cores = driver.active.utils.get_device_properties(idx).get("num_vectorcore", 40)`
+ `for program_id in range(worker_id, total_programs, worker_count)`（T14/T15 平台通过）。
第四批同构证据：FLA persistent（vllm-ascend#7563）"物理 AI core grid + tl.range task stride"。
**对 body 已有 stride 循环的题，persistent 化是 launch 一行的单变量改动，零数值风险**（判分
均为标准 per-dtype tolerance，不触数学）。

注意：T59 HEAD 的 `_ascend` 文件仍是 e9 证伪字节（4.29x 行打包），新候选必须整体替换，
否则平台会选中旧字节；T64 HEAD `_ascend` 与 generic 逐字节相同（无害，直接重写）。

### 结论 B：GCU（燧原）官方配方未覆盖三题

配方（skill 硬事实，T19-E5/T51-E5 两次平台 +38%）：persistent `grid=(min(tiles,24),)` +
`tl.range(..., num_stages≥3)` 开 pingpong + **launch 不写 num_warps**（未钉组中位 2.077
vs 钉住组 0.732）。违规清单：
- T68 `_enflame`：钉 `num_warps=8, num_stages=1`（`_enflame/ops/fused_eh_norm.py:131-132`）；
- T63 `_enflame`（E15 字节）：grid `(min(batch,65535), splits)` 无 persistent、无 stages；
- T62 `_enflame`（E9 字节）：BH16 封顶形态，persistent/去钉从未应用（e11 只试过钉 warps=4 反向）。

### 结论 C：deepep 两题 slot 循环形态从未真实优化

T64/T65 kernel 为运行时界 slot 循环 + 逐 slot 标量 `tl.load` 路由/权重 + `dst.to(int64)`
进 store 寻址。账本事实：T64 的 E5 "constexpr 探针"实为字节相同的空操作（`git show` 双验），
即 **constexpr topk 从未真正上过平台**；T65 无任何华为代码级专项。上游 SGLang
`ep_moe_kernels.py` 的 cutlass/deepgemm 变体给出三条可移植差异：2D grid-stride +
`tl.range(num_stages=3)` 软件流水、目标指针提升（`dst_ptr_offs` 预构，循环内只加
`dst_idx*hidden`）、expert-id 判有效。deepep 原生变体还用 **native dtype 累加**（非 fp32）。
GCU 侧 i32 寻址规则（签名级限制、体内 i64 算术合法）已有完整本地矩阵，T64 注册未执行的
C2 precomputed-pos 可一并消除"load 值直接作 store 索引"毒点。

### 结论 D：T69 天数轴（77 vs 206-210）是原子结构问题，唯一大分轴但高投入

generic 单遍 atomic（E13 pair-lane）在 tianshu 77；他队 206-210。业界正统是 moe_align 式
**块聚合原子**：每 program 对 BLOCK 元素先算 per-expert 局部计数（compare+sum），每
(block,expert-tile) 一次 atomic 取基址，块内 rank 用 one-hot cumsum。代价是
[E_TILE,BLOCK] one-hot 矩阵寄存器压力大 + "向量+tl.sum" 是昆仑毒点（昆仑走冻结 vendor
不受影响）。账本已注册 local-bucket 候选（NVIDIA 代理 GM 4.19）从未目标验证。
**昆仑 0.1024 贴 0.1 门：任何 T69 候选必须冻结 `_kunlunxin` vendor 字节不动。**

### 结论 E：无效题与放弃项

- T60 clamp_position：燧原八轮全灭，根因候选"torch-gcu 逻辑 int64 物理窄化"需目标芯 probe，
  无设备入口，今日不解。维持封存。
- T70 gate_topk：昆仑 3630s 挂死×2 同指纹 = 崩溃族；重掷需用户当次明示授权 + 工单回应，
  不在自动计划内。generic v2 修复（323c3c9）仍留本地。
- T74：E4 TB 20.18 有效在榜；最近三发燧原 3630s 超时/昆仑编译错与数值错，validity 风险 >
  收益（潜在 +5.5 但需同时修两芯），今日不碰。
- T66 card_b（_amd 后缀，=AMD/HIP 路径）从未有专项 vendor（5.92 vs 榜首 12.07）：唯一
  未开发轴，但 GEMM 调参无设备证据，置信低，列为机动项。
- T69/T63/T65 的榜首极值芯（华为 1255/T63、T65 华为 255-279）中，T65 华为为两队同量级
  结构差（100.9 与 244.7），随结论 A 一并攻打；T63 华为次优仅 92（+2.4），优先级低于其
  燧原轴（+9.6）。

## 3. 联网与 GitHub 调研摘要

- 上游 `sgl-project/sglang` `kernels/ops/moe/ep_moe_kernels.py`：`deepep_permute/post_reorder`
  与我方同构（每 token 一 program、BLOCK 512、运行时 topk、i64 dst）；**cutlass 变体**的
  2D grid-stride + `NUM_STAGES=3` + 指针提升 + expert-id 判有效是可移植增量。
- `vllm-project/vllm-ascend` glm5next `fused_eh_norm.py`：纯 Triton、每 token 一 program、
  BLOCK=next_pow2(H)、无 NPU 魔法——T68 华为轴无上游特化可抄，走结论 A。
- flagos-ai/FlagGems-sglang PR 扫描（截至 09-17）：最新为 #79（09-15，均为前批），
  **第五批无 PR**（攻占后才开窗口），无他队结构泄露。
- 昇腾官方 `vector_operator.md` + `triton-ascend-ops` best_practice（gather_scatter /
  binned_gather_scatter / padded_gather_scatter）：deepep 类题的进阶形态（SUB_BLOCK_SIZE
  批量、insert/extract_slice、按 vector core 分外层任务）——结论 A 首轮不依赖，作为
  T64/T65 追击备选。

## 4. 候选清单与预注册门（按性价比排序）

规则：全部为 `_ascend`/`_enflame` vendor-only 改动（不动 generic、不动昆仑 vendor），
单变量（GCU 配方三件套按官方配方记为一变量组，账本注明）；NVIDIA 远端跑完整正确性
矩阵 + release 回执；华为/燧原轴标 target-runtime-unverified（无授权主机，KernelGen
 昨日不可用）；每候选一次上传一次提交。

| # | 题 | 改动 | 预注册门（全部芯片有效前提下） | 潜在Δ均值 | 弹药 |
|---|---|---|---|---|---|
| C1 | T75 | `_ascend` persistent：grid=min(cdiv, NVC=40)（body 已 stride） | 华为 ≥ 2.6 | +1.0~2.4 | 1 |
| C2 | T73 | `_ascend` persistent（flat 路径 launch 级） | 华为 ≥ 2.2 | +0.6~1.3 | 1 |
| C3 | T71 | `_ascend` persistent（rows grid 封顶 NVC） | 华为 ≥ 3.9 | +0.4~0.7 | 1 |
| C4 | T67 | `_ascend` persistent flat（body 加 stride 循环） | 华为 ≥ 3.9 | +1.0~1.9 | 1 |
| C5 | T64 | `_ascend` 重写：persistent grid + 大 BLOCK（保 stride loop） | 华为 ≥ 6.2 | +1.5~12 | 1 |
| C6 | T65 | `_ascend` 新增：persistent 双轴（E12 修复语义） | 华为 ≥ 35 | +2.3~28 | 1 |
| C7 | T63 | `_ascend` 新增：axis0 封顶 NVC | 华为 ≥ 100 | +0.5~2.3 | 1 |
| C8 | T59 | `_ascend` 替换 e9 字节：e4r generic 形态 + persistent | 华为 ≥ 17 | +0.9~9.2 | 1 |
| C9 | T72 | `_ascend` grid 复核 + persistent 封顶 | 华为 ≥ 2.4 | +0.4~1.0 | 1 |
| C10 | T68 | `_ascend` 新增 persistent（copy generic + stride） | 华为 ≥ 4.2 | +0.05~0.3 | 1 |
| C11 | T68 | `_enflame` GCU 配方：去 warps 钉 + stages=3 | 燧原 ≥ 3.4 | +1.5~4.5 | 1 |
| C12 | T63 | `_enflame` GCU 配方：persistent24 + stages3 + 去钉 | 燧原 ≥ 40 | +2.3~7.6 | 1 |
| C13 | T62 | `_enflame` GCU 配方（E9 字节上） | 燧原 ≥ 0.6 | +0.2~0.4 | 1 |
| C14 | T69 | generic 块聚合原子（昆仑 vendor 冻结） | tianshu ≥ 120 或均值 ≥ 58 | +3~16 | 2 |
| C15 | 机动 | T66 `_amd` card_b 专项 / 追击轮 | 按回执定 | +0.5~0.8 | 2 |

追击规则：C1-C13 中任何一发平台兑现（过门），同题允许最多 1 发梯度追击（如 T65 华为
过 35 后追 BLOCK 阶梯或 num_stages）；未过门的轴即封，弹药回池。

## 5. 30 发弹药分配与时间线

- **Phase 1 结构实验（即刻—16:00）**：C1-C13 共 13 发。开发顺序按 C1→C4（launch 级，
  半小时内可齐）先行成批验证提交，C5-C8（新 vendor 文件）随后，C11-C13 与 C5-C8 并行。
  每发间隔 ≥120s 由 CLI 保证；提交后不等八芯全回再发下一题（并行回执）。
- **Phase 2 追击（16:00—18:15）**：≤6 发，只追已过门轴的梯度。
- **Phase 3 收盘（18:15—19:50）**：≤3 发防御/回归；**保留 ≥4 发不耗尽**（第 4 批教训：
  最后 10 发中 8 发重复测量仅换 2 项小新高；同字节重掷只作防御储备且需预注册窗口判据）。
- 任何时刻发现同窗他队大幅跃升，先拉该题逐芯榜再定是否改打。

## 6. 风险与纪律

1. **T69 昆仑 0.1024**：vendor 字节冻结；generic 改动只影响走 generic 的芯。
2. **T59/T64/T67/T72/T75 HEAD 旧 `_ascend` 字节必须整体替换**，防平台选中证伪/空操作字节。
3. vendor 新文件必须接入 `tests/_op_variants.py` 矩阵（T35 样板）+ 完整 release 回执后
   才能打包提交；`build_submission.py` 从明确 commit 取字节。
4. 华为/燧原轴无授权目标机：全部标注 target-runtime-unverified，平台八芯即最终裁决；
   不以 NVIDIA 代理计时当作华为性能承诺（T65 教训：代理 GM 与平台无因果）。
5. 每发提交前 `status` 复核额度与间隔；`sending/uncertain/stale` 状态不自动重试。
6. 账本更新：每题平台终态写入对应实验账本 + CURRENT 块 + `gen_experiment_index.py` 刷新。

## 7. 快速核对数据（供执行会话直接引用）

- 我方今日零消耗，额度 30/30（`status` observed_at 2026-09-17T10:59:08）。
- T74 E14 终态 6/8（燧原 None + 昆仑 None），TB 仍 E4。
- T69 榜首 c2flow 82.74（tianshu 206.11/kunlun 2.86/huawei 20.88）；EvokeAgent 80.92
  （enflame 24.62/kunlun 7.84/tianshu 210.09）。
- T63 我方 TB=E8 199.70（燧原 6.18 为 E8 窗口值；E15 字节燧原最好 25.40）。

## 8. 当日执行与判决（2026-09-17 15:20 收盘记录）

9 发结构实验全部 8/8 有效（零无效提交），6 个新 TB、2 个保底、1 个微幅：

| 发 | 题 | 结构 | 终态 |
|---|---|---|---|
| E7 16570 | T64 | 去 clone 三段式 + ascend persistent + kunlunxin 冻结 | **新 TB 7.47225（rank 14→10）**，华为 +44% |
| E13 16572 | T65 | ascend persistent（归约 kernel） | 26.54 < TB，保 E10；华为 -19% |
| E10 16574 | T75 | ascend persistent | 微幅新 TB 2.89673；华为 +16% |
| E13 16575 | T73 | ascend persistent（broadcast 加 stride） | **新 TB 4.26670（rank 7→5）**，华为 +97% |
| E9 16576 | T68 | enflame GCU 配方（去钉+stages3） | **新 TB 7.20041**，燧原 +39% |
| E18 16577 | T63 | enflame persistent24+stages3 | 198.79 差 0.46% 保底；**燧原 6.18→36.0** |
| E10 16578 | T59 | ascend persistent（e4r 形态） | 23.88 < TB 保 e4r；华为 +2.9% 门未过 |
| E8 16579 | T67 | ascend persistent flat | **新 TB 4.43975**，华为 +30% |
| E11 16582 | T71 | ascend persistent | **新 TB 2.84881**，华为 +19% |
| E19 16585 | T63 | enflame BLOCK 8192 | 195.48 < TB；燧原 32.3 阶梯到顶 |
| E8 16586 | T64 | ascend SUB 批量 2-D store | 7.24 < TB 保 E7；华为 -35% 证伪 |

（11 发含并行会话 2 发；本会话 9 发。）

**配方判决（跨题复用结论）**：
1. `num_vectorcore` persistent（launch 级）对逐元素/行散布 kernel 稳定正收益（华为 +16%~+97%），对 slot 循环归约 kernel 无效甚至负向（T65 -19%）。
2. GCU 配方（去 warps 钉 + num_stages=3 + grid≤24）在 T68 (+39%) 与 T63 燧原 (+483%) 双双兑现，历史第三、四次平台实证。
3. 去 clone 三段式在 T64 平台兑现 +4.2%（代理 2.2x 只部分迁移）；双路径分发消除小 shape 回退。
4. SUB 批量 2-D store（官方 004 教程可移植子集）在 T64 华为反向 (-35%)——该形态的收益依赖 insert_slice/extract_slice 完整版或多行 UB 组装，纯 2-D scatter store 不等价。
5. 华为结构性大缺口（T64 2.9 vs 100.9、T65 14 vs 245、T59 8.9 vs 82）今日三板斧（persistent/行打包/批量 store）全部尝试完毕，需目标芯 IR 或授权主机才能再进一步。

额度：19/30 剩余；本会话产物 commits `a6921555..c144004c` 全部已推送。

## 9. 收盘补遗（2026-09-17 18:10）

重新盘点后剩余额度聚焦 T64（唯一可达 Top1），追加三发：

| 发 | 结构 | 终态 |
|---|---|---|
| E9 16599 | 逆映射 gather（generic+ascend，零 scatter store） | **新 TB 7.79705**：generic 全正（card_b +14%/card_a +8.9%/天数 +6.6%），华为 4.12 未过门 |
| E10 16605 | 燧原去 clone gather + GCU 配方 | 微幅新 TB **7.818425**：燧原 4.81 未过 6 门，与 clone+scatter 打平 |

T64 终局：7.171 → **7.818425**（+9.0%），rank 14→9；`tl.insert_slice` 不在 Triton 3.7.1 主线，官方完整 gather/scatter 形态无法代理验证，华为 25x 缺口留待目标芯 IR。当日全会话 12 发提交全部 8/8 有效、7 个新 TB、零无效；额度余 17/30。

## 10. 追加轮（2026-09-17 17:00 收盘）

用户授权 1/2/3 后追加五发（全部单发单判决）：

| 发 | 内容 | 终态 |
|---|---|---|
| T64 e11 16639 | gather BLOCK 2048（intent 归档重发） | **新 TB 8.2269（+5.2%）** |
| T66 e3 16645 | enflame GCU 配方 | 7/8 无效：昆仑在未变 generic 字节上编译失败=平台侧回归；enflame 0.437 无增益；TB 保 |
| T72 e11 16653 | enflame grid cap 24 | 8/8，2.4667 < TB 保；燧原 +22% 小正 |
| T69 e14 16658 | warp 聚合原子 generic | 6/8 无效：muxi/card_b 2-D cumsum 数值错 + tianshu 77→66.6 反降=假设证伪；TB 保 |
| T64 e12 16669 | gather BLOCK 4096 | **新 TB 8.28935（+0.76%），阶梯封顶** |

T64 当日 7.171→8.289（+15.6%，rank 14→9）。全天全会话 17 发提交、9 个新 TB、2 发平台侧无效（T66/T69，TB 均无损）。额度 10/30 剩余。T66 昆仑重掷（崩溃族协议）留用户单发授权决定。

## 11. 终局（2026-09-17 18:40 封枪）

T64 最后两发：e13 燧原 BLOCK 4096 **新 TB 8.73395（燧原 +42.6%）**；e14 8192 过峰保底。**T64 全日 7.17105→8.73395（+21.8%），rank 14→9，六次换 TB**。全会话当日总计：**19 发提交、10 个新 TB、2 发平台侧无效（TB 无损）**；额度 8/30 保留。T66 昆仑重掷经评估不再申请（enflame vendor 无增益，重掷期望值≈保底）。

## 12. 额度收尾（2026-09-17 17:45 终局）

最后两发 T62：e12 双配方 **新 TB 1.270875（华为 persistent +35.9%）**、e13 修正发射低于 TB 保底。**全天终局：21 发提交、11 个新 TB、2 发平台侧无效（TB 无损）**；T64 7.171→8.734（+21.8%，rank 14→9）为最大单项。所有有证据的轴（persistent 谱系、GCU 配方、BLOCK 阶梯、去 clone/gather 结构、块聚合）全部试尽或证伪并留档。**剩余 6 发按第 4 批教训不做无假设盲掷**，留作防御额度；赛后复盘资产：本文件第 8-12 节 + 各题账本终态。

## 13. 真终局（2026-09-17 17:00）

追加 T72 e12（while→tl.range stages 管线化）= 2.428 < TB 保底，GCU 配方最后一个元素试尽。**全天最终：23 发提交、11 个新 TB、2 发平台侧无效（TB 无损）、1 发低于 TB 保底**。剩余 5 发额度：全场不再存在任何有预注册证据的候选，按纪律不盲掷，留作防御。所有账本/方案/INDEX 已推送。
