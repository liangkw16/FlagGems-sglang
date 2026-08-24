<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

# 跨芯极致优化方案（第二批）

> 制定时间：2026-08-24。依据：5 个已平台 8/8 算子的逐芯结果、17 份实验账本的
> 全部已尝试记录，以及 [`data/vendor-backends/`](data/vendor-backends/README.md)
> 缓存的固定 commit backend 源码。本文只给可执行假设，不含未验证的性能承诺。

## 1. 瓶颈量化：三芯吃掉全部排名

5 个已 8/8 算子的逐芯加速比（Task 08 E2 / 19 E2 / 20 E3 / 21 S3 / 24 S1）：

| 芯片 | 5 题均值 | 最低 | 最高 | 提到 2.0x 可为每题平均分贡献 |
| --- | ---: | ---: | ---: | ---: |
| 天数 `_iluvatar` | 7.076 | 3.660 | 10.001 | +0.0000 |
| 海光 `_hygon` | 5.473 | 2.130 | 7.508 | +0.0000 |
| 国际 A | 4.818 | 3.140 | 6.946 | +0.0000 |
| 国际 B | 4.212 | 2.214 | 6.227 | +0.0000 |
| 沐曦 `_metax` | 3.807 | 1.930 | 5.386 | +0.0018 |
| **华为 `_ascend`** | **1.178** | 0.598 | 1.884 | **+0.1027** |
| **燧原 `_enflame`** | **1.084** | 0.206 | 2.851 | **+0.1358** |
| **昆仑 `_kunlunxin`** | **0.532** | 0.175 | 0.931 | **+0.1835** |

结论：**五强芯已无优化价值**（全部远超 2.0x，且平均分由算术平均计算，强芯再涨
对排名贡献被弱芯拉平）。全部优化预算应投向昆仑、燧原、华为三芯。三芯同时提到
2.0x 相当于每题平均分 +0.42，是任何 generic 微调都达不到的量级。

## 2. 已被否决的假设（不要重复）

| 假设 | 证据 | 结论 |
| --- | --- | --- |
| 昆仑 multi-row（Task 19 E2） | 平台 0.9308x，仅 +0.0119x | 本地代理收益不迁移，否决 |
| 昆仑轴交换（Task 21 S2） | `uni_sram PassManager::run` 全 case 失败 | 否决 |
| 昆仑 2-warps（Task 17 E1） | 0.999994x | **源码已解释**：`num_warps` 在昆仑无效 |
| 本地延迟作为跨芯依据 | Task 19 E2 反例 | 只能用资源指纹 |
| 燧原 0.207x 判为噪声（Task 21） | 连续三次 ~0.207x | 不是噪声，是真实瓶颈 |

`chunk_cumsum` E1–E4 全部拒绝、`chunk_state` E3 因最差 0.9591x + FP32 spill
拒绝、`decode_attention` E1 因 12 spills 拒绝——这些"负结果"的判据是正确的，
继续沿用。

## 3. 昆仑：源码已证伪三条常用手段

缓存源码给出的硬事实（详见
[chip-landscape.md §4.1](chip-landscape.md#41-backend-源码可证的编译期事实)）：

1. **`num_warps` / `num_ctas` / `num_stages` 全部无效**
   （`kunlunxin/compiler.py:140,172` 记为 `invalid_params`）。Task 17 E1 的
   2-warps 实验得到 0.999994x 是必然结果，不是测量噪声。
2. **物理并行度是编译期常量**：arch 3 固定 `nclusters=12, ncores=64`
   （`driver.py:41-53,703-710`），`gridX/Y/Z` 由 `LoopGrid` pass 在设备侧注入
   （`driver.py:554-556`）。**"加大 grid 提升并行度"在昆仑不成立。**
3. **`tl.dot` 默认 `tf32`，允许集合只有 `("ieee","tf32")`**
   （`compiler.py:135-136`），没有 `tf32x3`。

因此昆仑上**只剩三个真实可调轴**：

- `BLOCK_SIZE`（Task 21 S3 已验证：BLOCK 1024 使全部 case 编译通过，当前
  vendor 即 `block_size = 1024` + 2D grid）；
- `buffer_size_limit` / `TRITONXPU_BUFFER_SIZE`（默认 512，
  `compiler.py:107`）——这是 XPU 的片上 buffer 预算，是 BLOCK 上限的真实约束；
- grid 恰为 `(12,1,1)` 时才触发 `TTXPU_F_INTERLEAVE`（`compiler.py:255`）。

**昆仑候选假设 K1：继续沿 BLOCK 轴单变量抬升（1024 → 2048）。**
S3 已证明 BLOCK 1024 + 2D grid 在全部 case 编译并运行通过，且 BLOCK 是昆仑上
唯一被平台验证过有效的轴。BLOCK 2048 同时把展平总数进一步压低（case 7 由
`4096×7` 降到 `4096×4`，更远离 65535 上限），方向与已验证证据链一致。

前置检查：`buffer_size_limit` 默认 512，需确认 BLOCK 2048 下 FP32 累加器
（`accumulator = tl.zeros((BLOCK_SIZE,), tl.float32)`）不超 XPU 片上 buffer
预算；若编译报 buffer 超限，则该轴到 1024 为止，昆仑改为只跟随 generic。

**明确排除**：不要把昆仑 grid 强设为 `(12,1,1)` 去追 interleave。当前 vendor
是 2D grid `(num_tokens, cdiv(hidden, BLOCK))`，改一维会同时变更 kernel 数学
结构与 grid 形状，不是单变量；且 S2 的轴交换实验已证明"改昆仑 grid 结构"这条
路会触发 `uni_sram PassManager::run` 失败。interleave 条件只作为已知事实登记，
不作为本轮假设。

目标算子优先 `moe_sum_reduce`（当前 0.1754x，距 0.1x 门槛仅 0.075x 余量，
是全局最脆弱点）。

## 4. 燧原：唯一已验证的 6.6 倍手段，且可迁移

Task 08 E2 把燧原 vendor 的 `block_size` 从 256 改为 4096，平台
**0.4292x → 2.8510x（6.6 倍）**，grid cap 仍为 12。

**缓存源码解释了为什么**：燧原 `vector_length` 默认 512 字节，gcu400/410/500
为 **2048 字节**（`enflame/compiler.py:477,512,527`），并直接作为
`-convert-vector-to-gcu=vector-bit-width=` 传给编译器
（`compiler.py:268-274`）。BLOCK 256 在 2048B 向量宽度下严重欠填充；BLOCK 4096
才对齐向量化粒度。这不是玄学调参，是向量宽度对齐。

**燧原候选假设 E1（最高优先级）**：把 Task 08 已验证的 BLOCK 4096 + grid cap 12
迁移到当前仍为 BLOCK 256 的算子：

| 目标 | 当前燧原 | generic BLOCK | 状态 |
| --- | ---: | --- | --- |
| `moe_sum_reduce` | 0.2060x | 256（`ops/moe_sum_reduce.py:76`） | **无燧原 vendor** |
| `softcap_out` | 0.35x | 256（`ops/softcap_out.py:65`） | 有燧原 vendor，S2 未提交 |
| `mamba_layernorm_gated` | 0.5090x | — | 无燧原 vendor |

`moe_sum_reduce` 是最优起点：0.2060x 是全局最低读数之一，generic BLOCK 仍是
256，且账本已明确"若后续继续优化 Task 21，优先做燧原性能 vendor（如 Task 08 式
BLOCK 提升）"。同型手段 + 同型算子（都是 bandwidth-bound pointwise/reduction），
迁移风险低。

**同时必须注意**：燧原 `enable_i64=False` 默认关闭（`compiler.py:491`）。
BLOCK 提到 4096 后，`token * hidden + offset` 类地址计算的中间值会变大；若原本
依赖 int64，在燧原会静默降位。BLOCK 提升的回归必须覆盖最大公开 shape 的地址
边界。

**燧原 num_warps 上限是编译期 assert**：gcu400/410 上 `num_warps > 4` 直接
`assert False`（`compiler.py:102-108`）。任何燧原 vendor 都不得超过 4 warps。
另外 gcu300 的 `max_shared = 8 MiB * num_warps`，减 warps 会同时压缩 shared 预算。

## 5. 华为：官方范式已三次验证，下一步是 tile 而非 grid

capped grid-stride 已在 Task 08 / 20 / 21 三次平台验证成功，且与
triton-ascend 官方 Vector Operator 指南完全一致（原文：关键不是创建尽可能多的
grid program，而是让 launch 接近物理 Vector Core 数）。**结构问题已解决**，
华为当前 1.178 均值是性能问题，不是启动问题。

官方指南给出的下一层优化点（`ascend/vector_operator.md`）：

- `BLOCK_SIZE` 尽可能大但不超 UB 容量；UB 溢出则改用 sub-block；
- 尾轴需 **32B 对齐**；
- 不规则 GM 访问应转为"批量载入 UB 后在 UB 内选择"；
- 性能差时先查 grid 是否远大于物理 Vector Core 数。

**华为候选假设 A1**：在已验证的 capped grid-stride 骨架上，只把每 program 的
tile 增大到 32B 对齐的最大值（不改 grid cap）。目标 `moe_sum_reduce`
（0.5982x）。这是单变量，且有官方文档依据。

## 6. 执行顺序与额度分配

第二批截止 **2026-08-27 19:59:59**（按更严格口径），建议最晚 19:00 前完成最终
提交，并预留两次额度做最终回归。

**优先级 P0：未知 > 已知微调。** 12 个算子尚无任何八芯结果
（09/10/11/12/13/14/15/16/17/18/22/23）。首投拿到 8/8 逐芯数字的信息量远大于
已通过算子的小幅优化——因为弱三芯的洼地模式只有平台能告诉你。按
[`experiments/README.md`](experiments/README.md) 既定队列
`12 → 09 → 17 → 23 → 22 → 11 → 10 → 15 → 16 → 13 → 14` 推进。

**P1：三个弱芯定向修复**，每次只改一个变量、只改一芯 vendor：

1. 燧原 `moe_sum_reduce` BLOCK 4096（假设 E1，已验证同型手段，预期收益最大）；
2. 昆仑 `moe_sum_reduce` BLOCK 1024 → 2048（假设 K1，唯一已验证有效轴）；
3. 华为 `moe_sum_reduce` 32B 对齐大 tile（假设 A1）。

三条都落在同一算子上，便于隔离变量：generic 与其余七芯 vendor 逐字节不变。

**P2：新算子首投时预置弱三芯策略。** 对新的 bandwidth-bound pointwise/reduction
题，S0 直接按下表设默认值，避免重复交"结构学费"：

| 芯片 | S0 就该带的策略 | 依据 |
| --- | --- | --- |
| 燧原 | BLOCK ≥ 4096、grid cap 12、warps ≤ 4 | Task 08 E2 平台 + `vector_length=2048` |
| 华为 | capped grid-stride（cap 4096）、tile 32B 对齐 | Task 08/20/21 三次验证 + 官方指南 |
| 昆仑 | BLOCK 1024、不调 warps/stages | Task 21 S3 平台 + `invalid_params` |
| 天数/沐曦/昆仑 | FP32 严容差 GEMM 显式 `input_precision="ieee"` | 三芯默认 `tf32` |

## 7. 本地 5070 Ti 的角色边界

- **只做排除**：编译不过、数值错、资源退化（新增 spill / registers 上升 /
  shared 增长）、结构越界。`decode_attention` E1 因 12 spills 不晋级、
  `bmm_chunk` E1 因 154 registers + spill 双失败，都是正确用法。
- **不做证明**：任何"本地快 X%"都不得作为跨芯晋级依据（Task 19 E2 反例）。
- **晋级门禁**：资源不退化优先于延迟更低。
- 16 GB 显存装不下大 shape 不重要——隐藏 harness shape 本就未知。本地价值在
  **结构边界覆盖**（`0/1/BLOCK-1/BLOCK/BLOCK+1`、非连续 stride、空 segment、
  三 dtype、NaN/Inf），`embedding_lora_a` 与 `qkv_lora_b` 的空段越界都是这样
  抓到的。

## 8. 风险登记

- 燧原读数存在平台侧波动：同字节 generic 在 Task 21 出现 2.3126x 与 ~0.207x
  两种量级。但连续三次 ~0.207x 已排除偶发噪声，判定为真实瓶颈。
- 昆仑仅有 0.1754x（`moe_sum_reduce`），距 0.1x 有效门槛仅 0.075x 余量；任何
  昆仑改动都必须优先保证不跌破门槛，否则整题失去有效排名。
- 缓存 backend 的 commit 不等于比赛 worker 实际版本；`65535` 展平 grid 上限在
  全部 8 份源码中都不存在，属 runtime 约束，只能由平台报错反推。
