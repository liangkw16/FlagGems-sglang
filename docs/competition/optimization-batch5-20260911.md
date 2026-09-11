# 第五批一轮优化与首次提交（2026-09-11）

用户要求“联网搜索 + 本地经验，先优化一轮，再尝试提交”。本轮检查 T59–T64，
T59/T63/T64 晋级 E1，T60/T61/T62 保留已验证 S0。本次六题各提交一次，结果见文末和各题 CURRENT。

## 依据与取舍

- [Triton 官方分块示例](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)说明按 program 划分独立输出块并使用尾部 mask；本轮沿用这一方式。
- [SGLang KV 索引源码](https://github.com/sgl-project/sglang/blob/8014d9d062c3cc5d393596ecdf2f7009191965df/python/sglang/kernels/ops/kvcache/kv_indices.py)的 FlashMLA 相邻实现用 grid 轴 1 拆页块；T63 将此结构用于长段，并保留长度驱动的 grid-stride 循环。
- [SGLang MoE 源码](https://github.com/sgl-project/sglang/blob/8014d9d062c3cc5d393596ecdf2f7009191965df/python/sglang/kernels/ops/moe/ep_moe_kernels.py)对每个 token 循环 hidden 块；T64 将 hidden tile 改为独立任务，仍在 tile 内向各有效目的行复用输入。
- 本地 T48 教训要求长段不能只处理第一个 tile；T21 的昆仑 grid 上限经验要求展平任务数受限，超出的任务继续 grid-stride；T58 的向量除法编译失败记录促使 T59 将行/块索引改为标量。跨芯收益均为待平台验证的假设。
- T60 已是单次融合，T61 已是直接逆排列 scatter，T62 已有 head 共享与 int64 地址修复；缺少新的可证实瓶颈，本轮保持。

实时逐芯榜单已读取，原始快照见证据清单。其分数用于识别目标芯风险；不将他队异常读数解释为确定的计时原因，不把 NVIDIA 比值外推为八芯均值。

## 配对筛选与发布验证

预注册晋级门：两个代表负载至少一个 E1/S0 中位收益 >=1.05，另一个 >=0.95；完整正确性通过，spill/shared memory 为 0，寄存器 <=64。
GPU 无其他计算进程；同一进程、同一输入五轮交替 AB/BA，含 wrapper、warmup 20ms / rep 50ms。

| Task | 负载 1 / E1 相对 S0 | 负载 2 / E1 相对 S0 | E1 寄存器 |
| --- | --- | --- | --- |
| T59 | 8×128 / 1.0663x | 64×2048 / 1.0083x | 38 / 40（S0 为 29） |
| T63 | 1×8193 / 1.9101x | 32×1024 / 1.1045x | 34 / 30（S0 为 40） |
| T64 | 8×4×4096 / 1.5346x | 512×4×4096 / 1.0027x | 19 / 19（S0 为 21） |

所有测得编译资源均无 spill 和 shared memory。T59 寄存器上升，目标芯表现仍需平台确认。
筛选只执行代表负载的参考对比，不能替代完整回执。源码/测试提交后从 Git 对象重新准备 release，
三个 E1 的完整 13 个测试方法全通过；加上未变更 S0 的既有回执，六题合计 **24 个测试方法、284 条 test/subTest 记录、146 次实际 kernel launch**，无失败/错误/skip/xfail。
新增覆盖 T59 65537 行、T63 32769 长段/129 请求、T64 超过 65535 个逻辑 tile；继续覆盖尾块、stride、空输入及原输出未写区域。

设备：NVIDIA RTX 5070 Ti，PyTorch 2.13.0+cu130，Triton 3.7.1，CUDA 13.0。目标芯 runtime 仍为 `target-runtime-unverified`，由首次平台提交补齐。

## 不可变证据

- E1 source / verification commit：`29754c1dfcbc87c9b193e750145b668c8e35d852`。
- T60/T61 S0：`4836fe0378d6290cbcaa91958693e13ed1acdf3e`；T62 S0：`7b53fdeb58878f8eeaf08eaabbde8bb38015e523`。
- [完整证据清单](data/batch5-optimization-20260911.json)：全部 ZIP/源码/测试/回执/日志/测速哈希、五轮样本、编译资源、启动记录和逐芯榜单。
- 清单 SHA-256：`87a560dd86ab7e9dfb9607f8f05a007060d48b3aab876b165bfc0fec591f77db`。
- 原始产物：`artifacts/competition/batch5-optimize-20260911/`；提交记录：`artifacts/competition/batch5-submit-20260911/`。

## 首次提交结果（截至 2026-09-11T03:42:29.113201+08:00）

六个候选各上传一次、正式提交一次，远端 ZIP 均回读验签通过；没有自动重试。

| Task | 候选 | submission / daily_seq | 正确性 | 平台有效性 | 八芯平均 | 排名 |
| --- | --- | --- | --- | --- | --- | --- |
| T59 | e1 | 12896 / 3 | 7/8 | invalid_correctness | 未产生八芯均值 | 未入有效榜 |
| T60 | s0 | 12898 / 5 | 7/8 | invalid_correctness | 未产生八芯均值 | 未入有效榜 |
| T61 | s0 | 12900 / 6 | 7/8 | invalid_correctness | 未产生八芯均值 | 未入有效榜 |
| T62 | s0 | 12897 / 4 | 7/8 | pending | 未产生八芯均值 | 未入有效榜 |
| T63 | e1 | 12894 / 1 | 8/8 | valid | 121.8698125x | 第 5/5 名 |
| T64 | e1 | 12895 / 2 | 8/8 | valid | 6.7478x | 第 3/4 名 |

T59/T60/T61 的燧原分别有 8/1/9 个用例在 GCU IR 编译的 PassManager 阶段失败；其余七芯通过。T60 失败为 case 3，栈中的 IR 包含 i64 指针；这只是定位线索，不能直接据此认定 64 位类型就是根因。三题同类错误文本不足以证明同一根因，后续需最小复现和厂商路径修复。

T62 仍为评测中，昆仑芯等待回调；不补造分数、不将缺失结果计零。其燧原已达 0.109x，余量较小。额外只读等待 90 秒仍未收齐结果（watch 退出码 124）；平台八芯闭环尚未全部完成。

额度 **24/30**（已用 6），来自上述带时间戳的 status；排名查询时间 `2026-09-11T03:39:41.445754+08:00`。

[完整提交结果与逐芯证据](data/batch5-submissions-20260911.json)，SHA-256 `63418b87e9249bef72751df2a1778c52e6d679a5d313ecf18d8ed64b96c175dd`。源码和账本均按项目授权 commit/push，ledger commit 由本文件 Git 历史定位；topic 分支不触发仅限 master 的 CI。

## 第二轮：燧原攻坚（06:30–08:30，daily_seq 7–22）

用户要求结合联网结果与 GitHub PR 优化后提交。TTIR 差分（对照同批燧原通过的
kv_indices/deepep）+ FlagTree 源码调研（`enable_i64=False` + `gcu64-type-verifier`）
定位 **i64 向量数据 load 为编译毒点**；随后 6 轮平台迭代收敛出 GCU 整数算子
规则集（已沉淀 skill 硬事实表）：i64 仅限寻址、向量 load 值不得 extsi→i64 进
store 寻址、逐 lane 1D scatter 不可用（2D 行段形态待裁决）、整型 select/
字面量多分支 store 均高危。

| 战果 | 内容 |
| --- | --- |
| **T59 首个 8/8** | E3「clone 预填 + 单 gather 单 store」（kv_images 形态）燧原 27.8x；E3R 昆仑 XMLIR 崩溃族重掷后八芯全过，**23.4312x 首次有效登顶** |
| T62 崩溃族处置 | s0r/s0r2 重掷（昆仑评测器「服务线程卡死」）；华为水位三连降破门槛 |
| T60/T61 止损 | 燧原轴分别七轮/五轮全负；60 失败指纹跨三代内核逐位相同 ⇒ 常数性 wrapper 级根因；61 三种 scatter 寻址形态均不可用 |

## 第三轮：深度优化（08:40–10:30，daily_seq 23–30）

逐芯榜单情报 + codex-ask（gpt-6-astra/ultra）+ GitHub 调研（vLLM expandInputRows、
FlagGems vendor 语料、上游 AMD kv_indices 并行化）。Codex 验证纪律生效：**三个
大胆候选被代理实测否证后当场丢弃**（T64 dst-row 重构 0.35x、宽行 0.25x、
nospec -24%），并证实 clone 天花板警告（52.7%）。

| Task | 候选 | 结果 |
| --- | --- | --- |
| T59 | E4 `_hygon` vendor（2D 直映 + warps=2）+ E4R 昆仑重掷 | **海光 24.85→30.21（+22%），8/8 均值 24.1284x 超当时榜首 23.90 登顶** |
| T63 | E2 BLOCK=256 + 目标 ~512 协同 program（上游 AMD PR #37659 形态） | **8/8 新 team best 132.0990x（+8.4%）**，天数 +18%；代价燧原 21.7→9.0 |
| T62 | E1 BH16+nospec；E2/E3 `_kunlunxin` vendor（flat 行形态） | **昆仑 5 连崩溃首次解除**（vendor 编译运行）；燧原 0.108→0.233、华为 0.159 回门槛上；昆仑数值垃圾逐位稳定 ⇒ 头号嫌疑 XMLIR 非特化参数绑定 |
| T61 | E5 2D 行段 scatter（FlagGems index_put 形态） | 燧原评测器崩溃未获裁决；其余七芯 1.16–3.01x |
| T60 | E7 值域受控 i32 快路径 | 平台 int64 case 值域不含于 [0,2^31) 或 XMLIR view 语义问题；七轮止损确认 |

## 榜单动态与差距归因（~10:40 复核）

- **T59 被 GuanghuLab 反超**（25.4259 vs 我 24.1284，Nectar 23.90 第三）。
  逐芯差分：我方 5 芯领先（燧原 +4.80/A +2.81/海光 +0.45/昆仑 +0.29/B +0.50），
  **差距集中于华为（8.677 vs 24.788，-16.11）**——单项为均值差 -1.30 的 1.55 倍。
- T63 榜单：Nectar 201.8 > EvokeAgent 184.0 > GuanghuLab 135.6 > c2flow 122.1
  > **我方 132.1（第 4）**。
- T64：EvokeAgent 21.65 > Nectar 7.62 > **我方 6.75（第 3）**（华为 100.3 差距
  需目标芯证据，本轮代理否证全部候选后搁置）。

## 下一窗口待发射队列（额度 30/30 用尽，00:00 重置；截止 09-17 19:59）

全部候选已过代理矩阵 + v2 release 门禁 + ZIP 构建，发射参数见各题账本：

| 序 | 候选 | 依据 | 预期 |
| --- | --- | --- | --- |
| 1 | 62-e4（去 do_not_specialize） | E2/E3 垃圾逐位相同 ⇒ XMLIR 绑定嫌疑 | 昆仑过 ⇒ 8/8 ≈1.22x 上榜 |
| 2 | 61-e5r（2D scatter 重掷） | 崩溃族重掷 1/2 | 区分 GCU 挂死 vs 窗口 |
| 3 | 63-e3（燧原/昆仑/沐曦 vendor 冻结 E1 字节） | E2 窄带宽回退恢复 | 均值 132.1→≈134.4 |
| 4 | 59-e5（`_ascend` vendor） | 华为 -16 缺口；镜像 hygon 已证形态 | 华为 8.68→≈20 ⇒ ≈25.5 重登第一 |

预注册后续方向：59-e6 `_iluvatar`（天数 -3.03）；62-e5 constexpr 烘焙
（FlagGems 昆仑 gather 风格，若 e4 无效）；60/61 燧原轴封存待外部证据。
