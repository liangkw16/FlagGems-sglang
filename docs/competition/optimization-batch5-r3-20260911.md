# 第五批第三轮：平台扩题后的全批优化方案（2026-09-11 深夜）

本文是本轮只读+开发的收口：**平台于 09-11 晚把第五批从 6 题扩到 12 题**
（新增 T65–T70），截止不变（2026-09-17 19:59:59）。按用户要求：本轮
**不提交平台**，完成新六题 S0 开发、打包与账本，把全批 12 题的优化方向
按把握倒排，并做好 09-12 窗口的发射前准备。

## 1. 平台扩题事实（一手来源）

- `operator-tasks?batch_no=5` 现返回 12 题；旧六题题面逐字未变（仅
  synced_at 变化），新六题题面已落盘 `tasks/batch-5/65..70-*.md`。
- 同步脚本 `BATCHES` 已从 `(1,2,3,4)` 扩到 `(1,2,3,4,5)`（期望数 12），
  `task-index.md` 已含第 5 批 12 行。
- 新六题 2026-09-11T23:44 榜首/通过面（提交数/队伍数、达标队伍）：

| Task | 算子 | 榜首 | 榜首值 | 达标队伍 | 我方 |
| ---: | --- | --- | ---: | ---: | --- |
| 65 | deepep_post_reorder | EvokeAgent | 58.6088x | 5/14 | 未提交 |
| 66 | dsv3_fused_a_gemm | EvokeAgent | 4.1249x | 4/5 | 未提交 |
| 67 | fill_padded_rows | EvokeAgent | 8.3163x | 4/7 | 未提交 |
| 68 | fused_eh_norm | HAiWORLD | 7.1684x | 5/7 | 未提交 |
| 69 | fused_moe_dispatch_index | c2flow | 52.4826x | **1**/11 | 未提交 |
| 70 | gate_topk | c2flow | 3.2091x | **1**/10 | 未提交 |

> T69/T70 全场仅 c2flow 一队过线：**可过但难**。T69 的唯一通过队证明
> atomic-cursor 语义在八芯可达（外部校准，符合"全局止损需外部事实"纪律）。

## 2. 新六题 S0 开发（本轮已完成）

- source/verification commit：`b4727f1`（2026-09-11）；六题 ZIP 均为
  `artifacts/competition/<op>/s0-b4727f1/<op>.zip`（清单见各题账本）。
- 全部移植自 SGLang 8014d9d 固定 commit（T65/T69 出自 `ep_moe_kernels.py`、
  T67 出自 `moe/fill_padded_rows.py`、T70 出自 `moe/gate_topk.py`、
  T68 移植自 CUDA 版 `layernorm/fused_eh_norm.cuh`、T66 移植自 SM90
  CUDA 版 `gemm/dsv3_fused_a_gemm.cuh`，上游 CUDA 版为 2D N-tiling
  无 split-K，与我方结构同族）。
- py_compile / black / isort / flake8 全绿；六题测试矩阵共 24 方法，
  覆盖 dtype、tile 尾块、stride、空输入、特殊值与输入不变性。
- **验证债务（唯一未闭环项）**：远端 GPU 主机（192.168.5.204 物理口与
  EasyTier 10.126.126.6）09-11 晚全程不可达（ARP 无表项，主机离线），
  KernelGen MCP 本会话不可用。六个 release 目录已在本地
  `/tmp/flagos-<op>-release` 按 `b4727f1` 字节 prepare 完毕，GPU 恢复后
  一次传输、串行执行即可补齐回执。此前所有候选维持
  `candidate-wip / target-runtime-unverified`，不伪填验证状态。

## 3. 全批 12 题优化方向（按把握倒排）

### 第一梯队：已就绪、只等窗口（把握最高，无需再开发）

| 序 | 候选 | 依据 | 预期 |
| ---: | --- | --- | --- |
| 1 | T59 `e5-169dd6e`（`_ascend` vendor） | 华为 8.68→≈20 即重夺第一（现榜首 25.43，差距集中在华为） | 均值 24.13→≈25.5，重登 T59 第一 |
| 2 | T63 `e3-95e5292`（燧原/昆仑/沐曦冻结 E1 字节） | E2 窄带宽回退恢复（燧原 9.0→21.7） | 132.10→≈134.4 |
| 3 | T62 `e4-95e5292`（去 `do_not_specialize`） | 昆仑数值垃圾逐位稳定 ⇒ XMLIR 非特化绑定嫌疑 | 昆仑过 ⇒ 8/8 ≈1.22x 上榜 |
| 4 | T61 `e5r-95e5292`（2D 行段 scatter 重掷） | 崩溃族重掷先例 1/2 | 区分 GCU 挂死 vs 评测机窗口 |

（发射序按 T59 账本 CURRENT；四发均已过完整门禁，本节不重复 R2 论证。）

### 第二梯队：新六题 S0 首发序（按"首过八芯把握"倒排）

| 序 | Task | 把握依据 | 主要风险（已预注册对策） |
| ---: | ---: | --- | --- |
| 5 | **T67 fill_padded_rows** | 结构极简（clone+条件填充），exact 语义无浮点；上游生产内核 | 几乎无；超宽行寄存器压力留 E1 列分块 |
| 6 | **T68 fused_eh_norm** | 本仓 rmsnorm 家族 8/8 先例最厚；fp32 归约+单次转换正中规格 | BLOCK=8192 在弱芯的退化 ⇒ E1 两遍列分块 |
| 7 | **T65 deepep_post_reorder** | T64 同族同形态（已 8/8）；mismatch-ratio 宽容差；fp32 累加恒合法 | slot 标量循环开销 ⇒ E1 路由行向量化 |
| 8 | **T69 fused_moe_dispatch_index** | 上游同款 Triton 内核；c2flow 单队通过=八芯可达外部证据 | masked `tl.atomic_add` 昆仑/燧原未证 ⇒ E1 无原子双 kernel vendor |
| 9 | **T66 dsv3_fused_a_gemm** | tl.dot GEMM 本仓多题 8/8；bf16 语义与参考 fp32 一致 | M≤16 带宽受限、程序数不足 ⇒ E1 split-K/BLOCK_N 按逐芯读数单变量 |
| 10 | **T70 gate_topk** | 上游生产 Triton 内核近逐行移植 | `tl.topk/tl.sort/tl.bitonic_merge` 厂商 fork 未证 + fp32→uint64 体内 i64 ⇒ E1 手工 bitonic vendor |

### 第三梯队：旧六题追加方向（R2 已预注册，按均值增量期望排序）

1. **T63 方向 A：去 wrapper `kv_indices.clone()`**（最高期望，132→185~260）：
   先零额度代理测 clone 占比（门 ≥25%），过门才开发；同时按上游
   `sgl#37659` 公式修 `splits` 越燧原 `grid.y` 255 硬限的隐患（`batch≤2
   且 width≥65536` 触发）。
2. **T62 B1 grid 封顶**（一行改动；燧原/华为双双贴近 0.1 门槛的止血）；
   B2 constexpr 去 mask 独立单变量。
3. **T64 燧原 grid 封顶**（均值第一轴 +0.371）；precomputed-pos 改写其后。
4. **T59 e6 `_iluvatar`**（天数 −3.03 缺口）。
5. T60/T61 燧原轴维持封存（R2 §5/§6 证据订正后仍无独立证据不重启）。

## 4. 09-12 窗口发射纪律

- 今日额度已用至 `daily_seq 29`（30/30），下一窗口 09-12 00:00 重置。
- 前置条件：六个新 S0 的 release 回执必须先补齐（GPU 恢复 → 传输 →
  串行 run → 回收录证）；无回执不发射，不因窗口压力降低门禁。
- 建议顺序：第一梯队四发（00:00 起按 T59 账本序）→ 第二梯队 5–10 逐发，
  每发之间读逐芯结果再决定下一发；留 ≥20% 额度给"芯片信号为正但均值门
  未过"的重开。
- 同字节重掷纪律沿用 R2 §7（第四批最后 10 发教训）。

## 5. 证据分级

- **平台实测**：§1 榜单读数（2026-09-11T23:44 API）；旧六题全部逐芯
  数据见各题账本与 `data/batch5-*.json`。
- **上游源码级**：六个新算子的上游实现（SGLang 8014d9d 固定 commit）；
  dsv3 CUDA 版 tile 结构；sgl#37659/#35245（R2 已引）。
- **本地已验**：py_compile/格式/lint；ZIP canonical 哈希与成员清单。
- **假设（未测）**：第二/三梯队全部性能预期；T69 atomic 跨芯通过性；
  T70 tl.topk 家族跨芯编译。均不作为账本结论引用。
