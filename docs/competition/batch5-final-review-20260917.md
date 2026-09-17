# 第五批终局复盘与下一赛季开题情报（2026-09-17）

数据源：终局逐芯快照 `data/leaderboard-batch5-perchip-20260917-final.json`
（observed_at 2026-09-17 17:00:55，SHA-256 `27de10cc60b9e705728712cc522b89e77f6668d354693676ddfb93f0291b496b`；
截止 19:59 前他队仍可变动，本表为该时刻口径）。

## 1. 终局榜（我方 rank / TB / 榜首）

| 题 | rank | 我方 TB | 榜首 | 主要缺口芯（我方 vs 同芯次优） |
|---|---:|---:|---:|---|
| T59 page_table | 8 | 24.128 | 39.002 | 华为 9.4x |
| T60 clamp_position | 无效 | — | 3.839 | 燧原 int64 平台侧（八轮未解） |
| T61 src2dst | 6 | 2.138 | 3.196 | 华为 2.4x、沐曦 2.4x |
| T62 concat_mla | 12 | 1.271 | 2.442 | 燧原 8.4x、华为 11.3x、昆仑 5.1x |
| T63 kv_indices | 10 | 199.695 | 416.474 | 燧原 17.6x、华为 17.2x、昆仑 4.7x |
| T64 deepep_permute | 9 | **8.734** | 30.595 | 华为 22.8x、燧原 13.3x、昆仑 2.6x |
| T65 post_reorder | 13 | 26.917 | 78.737 | 华为 16.0x、昆仑 13.7x、燧原 4.4x |
| T66 a_gemm | 8 | 3.099 | 5.727 | 燧原 7.4x、天数 3.2x、card_b 2.0x |
| T67 fill_padded | 12 | 4.440 | 12.464 | 华为 7.0x、燧原 4.7x |
| T68 eh_norm | 10 | 7.200 | 18.283 | 燧原 13.4x、华为 5.2x |
| T69 dispatch_idx | 5 | 51.338 | 85.017 | 昆仑 28.2x、燧原 11.9x、华为 4.3x、天数 2.6x |
| T70 gate_topk | 无效 | — | 18.880 | 昆仑驱动级挂死（崩溃族） |
| T71 gelu_tanh | 13 | 2.849 | 6.095 | 华为 6.3x、燧原 2.1x |
| T72 group_norm | 6 | 2.663 | 4.781 | 华为 7.7x、燧原 4.4x |
| T73 residual_add | 6 | 4.267 | 6.418 | 燧原 2.5x |
| T74 seqlens | 8 | 20.178 | 28.660 | 燧原 2.5x |
| T75 sigmoid_mul | 11 | 2.897 | 5.969 | 华为 10.9x、燧原 2.4x |

## 2. 当日战报（本会话）

- **25 发提交、7 题刷新 TB（12 次 TB 刷新事件）、2 发平台侧无效（TB 无损）、零自伤**；额度 25/30 → 5/30 剩余（**本节为 17:02 快照**；随后用户指令用完额度，终态 **30/30 用尽、本会话 28 发 + 并行会话 2 发**，末 4 发 T63 e22/T64 e15/T68 e10/T73 e14 全部 8/8 有效低于 TB 保底，见 [finalday §15](optimization-batch5-finalday-20260917.md)）。
- 提交序列：T64×9（16570/16586/16599/16605/16639/16669/16677/16690 + e8 重发链）、T65 16572、T75 16574、T73 16575、T68 16576、T63 16577/16585/16619、T59 16578、T67 16579、T71 16582、T61 16621、T66 16645、T72 16653/16770、T69 16658、T62 16754/16759。
- 名次：T64 **14→9**、T73 **7→5**；其余守位。
- 单题最大：T64 7.171→**8.734**（+21.8%，六次刷新）。

## 3. 配方判决（可复用结论，证据=当日平台终态）

| 配方 | 兑现 | 无效/反例 | 边界条件 |
|---|---|---|---|
| Ascend persistent（`num_vectorcore` 封顶，launch 级） | T64 +44%、T73 +97%、T67 +30%、T71 +19%、T75 +16%、T62 +36% | T65 -19%、T61 -19%、T63 持平、T59 +2.9% | 宽逐元素/行散布有效；slot 循环归约、窄 lane scatter、已宽 copy 无效 |
| GCU 去钉（删 num_warps pin） | T68 +39%（与 stages 同发） | T62 E11 warps4 -47%、E12 stages3 -46% | **任何 launch 钉在该 GCU 栈上高危**；纯净 launch 是默认最优 |
| GCU grid≤24 | T63 燧原 +483%（与 stages 同发）、T72 +22% | T64 持平、T61 不适用（本就个位数 grid） | 仅对超发（>10² program）形态有意义 |
| GCU num_stages=3 | T68/T63 同发兑现 | T72 scalar while 循环零效果、T62 反向 | pingpong 只在特定循环形态生效，无法事前判定 |
| BLOCK 阶梯（宽 tile） | T64 generic 512→2048 +5.2%、燧原 512→4096 **+42.6%** | generic 4096 仅 +0.76%（封顶）、燧原 8192 -22%（过峰）、T63 8192 -10% | GCU 峰值 4096；generic 峰值 2048；过峰即回退 |
| 去 clone / 逆映射 gather | T64 generic 五芯 +3~14%（gather>scatter） | 华为 gather -7%、SUB 批量 2-D store -35% | GPU 类芯 gather 优；华为两者皆非突破口 |
| warp 聚合原子（moe_align 式） | haiguang/card_a +5~9% | **天数 77→66.6 反降**；沐曦/AMD 2-D cumsum 数值错 | "逐 lane 原子是天数瓶颈"假设证伪；[E,B] cumsum 后端不可靠 |

## 4. 未破译缺口（下一赛季开题情报）

1. **华为 index/散布族 7~23x**（T64 22.8x、T65 16x、T62 11.3x、T75 10.9x、T59 9.4x、T72 7.7x、T67/T71 6~7x）：
   当日系统性排除 persistent（部分题 +16~97% 后仍是数量级差）、行打包（T59 e9 -51%）、
   SUB 批量 2-D store（T64 e8 -35%）、逆映射 gather（-7%）、warps/宽度/stages。
   EvokeAgent 在 T61 华为拿 6.50（我方 1.71）证明存在我们未知的 index-math/lowering 形态；
   突破需要授权昇腾主机做 IR/profile 对照（T65 诊断文档已列出实验设计）。
   `tl.insert_slice/extract_slice` 不在 Triton 3.7.1 主线（已核实），官方 004-006 教程完整形态
   无法代理验证——下一赛季若平台栈升级或取得主机应优先验证。
2. **燧原结构差 2~18x**（T63 17.6x、T68 13.4x、T64 13.3x、T69 11.9x、T62 8.4x、T66 7.4x）：
   配方四元素全部试尽后仍存。他队可能使用 SUB_BLOCK_SIZE + insert/extract_slice 完整 UB 组装
   形态（教程机制），纯 2-D scatter store 不等价（T64 e8 已证）。需燧原侧 profiling。
3. **昆仑**：安全形态极窄（标量分支 + 单次 store；向量+tl.sum、分支 RMW、向量除、2-D cumsum 均毒）；
   T69 昆仑 0.10 贴门是长期风险；T66 出现**同字节编译判定变化**（linalg.matmul 形状断言，
   09-16 通过 09-17 失败）= 平台栈在版本漂移，同类题提交前要预期这个变量。
4. **天数**：T66 3.2x、T69 2.6x——除已证伪的聚合原子外无线索；`_iluvatar` 后缀通道已建
   （T75 e2 曾用），可做天数控件实验。
5. **无效题**：T60（燧原 torch-gcu int64 物理窄化，证据待目标 probe 核实）、T70（昆仑驱动级挂死崩溃族，
   等工单）——两题失败面均指向平台侧/需设备探针，复盘材料在各自账本。

## 5. 资产指引

- 当日全程判决与预注册门：[作战方案 §8-13](optimization-batch5-finalday-20260917.md)
- 各题终态：[实验账本 INDEX](experiments/INDEX.md)（CURRENT 块为唯一真相）
- 三份逐芯快照：morning `…0917.json` / mid `…0917-evening.json` / final `…0917-final.json`
- 昇腾官方 vector operator 指南与 best-practice 教程索引：`data/vendor-backends/ascend/vector_operator.md`
