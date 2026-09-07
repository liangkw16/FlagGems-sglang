# 第四批冲榜方案与执行记录

## 2026-09-07 首轮实际结果与重新分配

最新快照：有效题 **11/17**，Top1 **1题（T47）**；本轮4次正式提交，实时剩17/30。T57新增有效2.36478125、排名5。T43 E14=6.5204375、T51 E4=5.195325均未超过既有最佳；T52 E6昆仑失败，燧原仍待回调，已经无八芯有效分可能。

本轮没有新增Top1。代理单case 1.5倍不等于全芯均值增长；后续以八芯算术均值和真实榜首为门。只有新候选达到发布门禁、不可变ZIP及实时preflight后才提交。预留6次，剩余11次也是上限，不能把额度当作必须用完的任务。

| 题 | 算子 | 当前最佳 / 排名 | 实时榜首 | 追榜所需增幅 | 下一项证据 / 方案 |
| ---: | --- | --- | ---: | ---: | --- |
| 42 | [act_and_mul](experiments/act_and_mul.md) | 3.258350 / #10 | 431.484300 | 13142.4% | 暂缓；旧M1/tile轴停止，需要整体内存访问结构的新证据 |
| 43 | [causal_conv1d_update](experiments/causal_conv1d_update.md) | 6.545875 / #3 | 7.903250 | 20.7% | 优先研究；保留E13，tile128平台无增益、时间并行常规case回退；后续须窗口复用/状态写回成本的新证据 |
| 44 | [chain_speculative_sampling](experiments/chain_speculative_sampling.md) | 无有效分 | 1.981275 | 先有效 | 先正确性；定位half概率归一化/扫描首分歧，保留失败回归，不能带缺口提交 |
| 45 | [chunk_scaled_dot_kkt](experiments/chunk_scaled_dot_kkt.md) | 无有效分 | 17.957125 | 先有效 | 先过线；昆仑0.063需至少1.59倍，改epilogue结构前先拿目标编译证据 |
| 46 | [chunked_embedding_lora_a](experiments/chunked_embedding_lora_a.md) | 14.105187 / #5 | 24.413875 | 73.1% | 储备；仅分段路由和权重复用的新结构，旧天数预路由水位内波动不重复 |
| 47 | [chunked_sgmv_expand](experiments/chunked_sgmv_expand.md) | 25.004812 / #1 | 25.004812 | 0.0% | 守榜；保持25.0048125，未被超越不为微小弱芯收益消耗机会 |
| 48 | [chunked_sgmv_shrink](experiments/chunked_sgmv_shrink.md) | 4.719812 / #3 | 23.743625 | 403.1% | 储备；段合批降低launch/重复权重加载，先证实整个wrapper收益，旧BLOCK_S轴关闭 |
| 49 | [ernie45_rope_fused](experiments/ernie45_rope_fused.md) | 8.582531 / #4 | 15.606438 | 81.8% | 储备；参考PR40的RoPE访存/跨头并行，保留已修昆仑；须同时提高多个高分芯 |
| 50 | [extend_attention](experiments/extend_attention.md) | 无有效分 | 10.723281 | 先有效 | 先诊断；PR38/47仅供LSE合并与归约参考，尚缺目标芯正确性和燧原≥0.1证据 |
| 51 | [fla_layernorm_gated](experiments/fla_layernorm_gated.md) | 5.393900 / #6 | 6.668225 | 23.6% | 优先研究；E4平台回退关闭本轴；若重开，先隔离BLOCK_R=1路径的IR变化并验证沐曦 |
| 52 | [fused_dual_residual_rmsnorm](experiments/fused_dual_residual_rmsnorm.md) | 无有效分 | 5.425125 | 先有效 | 待目标证据；RN仍昆仑4元素同指纹，需固定源码逐阶段var/rms/y1/mid/out对照，不再换等价公式盲投 |
| 53 | [fused_gdn_gating](experiments/fused_gdn_gating.md) | 2.126700 / #10 | 288.426175 | 13462.1% | 暂缓；榜首跳升至288.426175，旧小比例gate调参不足，需整体带宽结构证据 |
| 54 | [fused_norm_rope_stacked](experiments/fused_norm_rope_stacked.md) | 无有效分 | 10.223469 | 先有效 | 先诊断；已终态5/8，分开归一化、RoPE与输出布局定位三芯首分歧 |
| 55 | [hc_head](experiments/hc_head.md) | 无有效分 | 6.363313 | 先有效 | 先诊断；隔离reference/compile_worker/候选，撤回纯平台归因；即使单芯过线也离Top1很远 |
| 56 | [l2norm](experiments/l2norm.md) | 3.107729 / #8 | 72.306802 | 2226.7% | 暂缓；榜首72.30680208，先测全wrapper带宽/归约成本，旧短行调参不直接投 |
| 57 | [log_scaling_tau](experiments/log_scaling_tau.md) | 2.364781 / #5 | 2.816469 | 19.1% | 优先；E2新增有效并排名5，仍需+19.10%；warp/BLOCK扫描无收益，华为MCP两轮零测试且rank1语义错误，未晋级 |
| 58 | [w8a8_block_int8_matmul](experiments/w8a8_block_int8_matmul.md) | 122.661583 / #3 | 562.415908 | 358.5% | 储备；需高分芯GEMM整体结构变化，保留int8先castFP32再计算契约，弱两芯翻倍不足追榜 |

PR调研的实际落点： [PR34](https://github.com/flagos-ai/FlagGems-sglang/pull/34) 的channel连续布局帮助T43此前E13首次有效；[PR50](https://github.com/flagos-ai/FlagGems-sglang/pull/50) 的多行调度本轮迁移到T51后平台未获益。 [PR40](https://github.com/flagos-ai/FlagGems-sglang/pull/40)、[PR38](https://github.com/flagos-ai/FlagGems-sglang/pull/38)、[PR47](https://github.com/flagos-ai/FlagGems-sglang/pull/47) 继续作为RoPE/attention结构参考，不能当作本题通过凭证。

榜单原始证据 `/Users/bytedance/ccc/flagos/artifacts/competition/batch4-top1-20260907/tasks-after-round.json`，SHA256 `d4a03f5b13ef9c9784987cf14b6f634490e31565e7ef3ddea30b33eec91443eb`。逐芯结果、源码/测试/ZIP哈希和负向实验见各题账本及[自动生成索引](experiments/INDEX.md)。

## 以下为2026-09-06历史方案（已被上方盘点更新）

## 复盘：当前位置与可提升空间

### 已有 6×8/8 + 1 榜首的逐芯拖累分析

| 题 | 均值 | 榜首 | 差距 | 拖累芯（<均值/3） | 单芯提升→均值增益 |
| --- | --- | --- | --- | --- | --- |
| **T47** 榜首 | 25.00x | — | 守榜 | 燧原 0.18x | 0.18→1.0 = +0.10x |
| **T48** | 4.72x | c2flow 21.63x | -78% | 燧原 0.58x | 0.58→3 = +0.30x |
| **T46** | 14.11x | EvokeAgent 18.75x | -25% | 燧原 0.37x / 昆仑 0.23x / 华为 2.12x | 华为 2→5 = +0.36x |
| **T51** | 5.39x | c2flow 6.67x | -19% | 昆仑 0.94x | 0.94→3 = +0.26x |
| **T53** | 1.77x | EvokeAgent 3.54x | -50% | 燧原 0.17x | 0.17→1 = +0.10x |
| **T42** | 3.25x | 431.48x | 远 | 昆仑 0.46x | — |

### 未 8/8 题的卡点

| 题 | 当前 | 卡点 | 可修性 |
| --- | --- | --- | --- |
| T50 extend_attention | 6 芯 correctness 过 | 华为数值 / 燧原 0.013x<门槛 / 昆仑在评 | 华为已试 3 种方法均败；燧原需提速 8x |
| T52 dual_rmsnorm | 6/8 | 燧原+昆仑 1-4/33M 精度 | 已试 5 种方法，天花板 |
| T43/T45/T49 | 7/8 | 昆仑错译/崩溃/uni_sram | 硬件层不可修 |

## 明日 30 发分配方案（按期望值排序）

### P0：T48 冲分（4 发）— 最大单题提升空间
- 当前 4.72x vs 榜首 21.63x（差 78%），有 16.91x 提升空间
- **燧原 0.58x→3x**：当前 route/materialize 太慢（逐段 launch）；改用 batched 直接 kernel
- **天数 3.85→8x**：当前 generic 3D grid；试 flat + i32 路由（T46 沐曦方案）
- **昆仑 1.79→5x**：当前 route/materialize；试直接 kernel + long 索引
- 每提升一个弱芯 2x → 均值 +0.25x；三个同时提升 → 均值 +1x 以上

### P1：T46 冲分（3 发）— 排名第二接近
- 当前 14.11x vs 榜首 18.75x（差 25%），有 4.64x 空间
- **华为 2.12→5x**：T45 FLA persistent 结构迁移（同是 gather+计算型）
- **沐曦 6.87→10x**：已试 i32 路由；试 segment-owned（昆仑 +198% 同款）
- 华为 +2.88x → 均值 +0.36x；沐曦 +3.13x → 均值 +0.39x

### P2：T51 冲分（2 发）
- 当前 5.39x vs 榜首 6.67x（差 19%），有 1.28x 空间
- **昆仑 0.94→2.5x**：向量化 flat（T53 同款修复）
- 昆仑 +1.56x → 均值 +0.20x

### P3：T50 冲分（3 发）— 最大潜在新增
- 6 芯 correctness 已过；如果能修华为 → 7/8 或 8/8
- 华为：试 Ascend 官方教程结构（insert/extract_slice + 借轴转置）
- 燧原 0.013x 需提速 8x 过门槛：batched kernel 已建，需 tile/warps 调优
- 若 8/8 → 预估均值 ~1x（加分但量级不大）

### P4：T53 冲分（2 发）
- 当前 1.77x vs 榜首 3.54x（差 50%）
- **燧原 0.17→0.8x**：当前向量化 flat 太简单；试 BLOCK 4096
- 燧原 +0.63x → 均值 +0.08x（量级小）

### P5：T47 守榜（2 发）
- 榜首位置需防守（c2flow 紧追）
- 燧原 0.18→1.0x → 均值 +0.10x（安全边际）
- 仅在榜首被超时使用

### 保留（4 发）
- 应对意外（评测机故障、榜首变化、新发现）

## 冲分技术方案

### T48 燧原提速方案
当前 route/materialize wrapper 逐段 launch → 每个 segment 一次 GEMM。段多时 launch 开销大。
方案：改用 generic 3D grid 直接 kernel + i32 metadata + 燧原已证结构。

### T46 华为提速方案
当前 token 折叠 vendor 2.12x。T45 FLA persistent 在同类型 gather+计算上达 0.25x（从 0.045x，5.6x 提升）。
方案：persistent 物理 AI core grid + 每 task 一个 (token, head) + GQA 共享。

### T51 昆仑提速方案
当前 generic grid-stride 0.94x。T53 向量化 flat（1024-lane）在昆仑上从 0.06→0.72x（11.5x）。
方案：去 grid-stride 循环 → flat 向量化 + contiguous+view(-1)。

### T50 华为修复方案
已试 fp32 强制、two-pass softmax。均败。
方案：检查是否有与 T43 相同的"后端数值错译"——如果是，则 T50 华为也不可修，转攻燧原提速。

## 关键纪律

1. **单变量原则**：每发只改一个 vendor 或一个轴
2. **止损**：同指纹两连败即停该轴
3. **水位意识**：海光 34-56x 波动，单轮高值不当结构收益
4. **评测机窗口**：燧原会周期性繁忙，避开高峰
5. **优先 8/8 题冲分**（保底已有），再攻未 8/8 题（上限更高）
