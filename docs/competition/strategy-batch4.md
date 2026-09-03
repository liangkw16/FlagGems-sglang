# 第 4 批作战策略（T42–T47）

> 制定时间：2026-09-03（题面同步 `docs/competition/task-index.md` 同批快照）。
> 输入：6 道新题题面 + 五路旧账本挖掘（激活族 / causal_conv+mamba / fla chunk 族 /
> LoRA 族 / 精确匹配采样族）。
> 本文档是批次级优先级与陷阱清单；各题当前状态以实验账本 `current` 块为准。

## 批次窗口与额度

- 提交窗口：2026-09-03 20:00 → 09-10 19:59:59（+08:00）；评审 09-11 → 09-17。
- 额度 30/日、00:00 重置、最小间隔 120s。09-03 当日 30 发已全部耗在
  第 3 批（最后 16:06），第 4 批我队 0 提交。
- 首日开火窗口：09-04 00:00 重置后。默认节奏：每晚一主一辅，留回归储备。

## 开题时榜单快照（2026-09-03 20:42 同步）

| 题号 | 算子 | 状态 | 达标队伍 | 当前第一 | 平均加速比 |
| ---: | --- | --- | ---: | --- | ---: |
| 42 | act_and_mul | competing | 1 | c2flow | 3.1941x |
| 43 | causal_conv1d_update | pending_challenge | 0 | - | - |
| 44 | chain_speculative_sampling | pending_challenge | 0 | - | - |
| 45 | chunk_scaled_dot_kkt | competing | 2 | EvokeAgent | 15.0261x |
| 46 | chunked_embedding_lora_a | competing | 1 | EvokeAgent | 18.7483x |
| 47 | chunked_sgmv_expand | competing | 2 | c2flow | 23.3266x |

## 优先级排序

| 优先级 | 题 | 判据（风险 × 收益 × 复用度） |
| --- | --- | --- |
| P0 | T42 act_and_mul | 风险最低：与 T29/T39 完全同构（row/col-block 骨架两题八芯验证）；1 队达标说明门槛可得；预计首个 8/8。 |
| P0 | T43 causal_conv1d_update | pending_challenge 差异化：0/9 队达标，首个有效解含金量高；T36 selective_state_update 的 clone/状态回写经验直接迁移；昆仑编译墙有现成绕法。 |
| P1 | T46 chunked_embedding_lora_a | T17 embedding_lora_a 的分段变体，模板直接改；gather 型低计算风险；榜首 18.7x 说明天花板高。 |
| P1 | T45 chunk_scaled_dot_kkt | bmm_chunk 骨架直接映射（K@K.T 同形态）；fla 族逐芯 dot 分派方案全部有平台实证；榜首 15x。 |
| P2 | T47 chunked_sgmv_expand | qkv_lora_b 骨架映射（slice_offsets 同构）；vendor 面大（昆仑 route/materialize 等），工作量最大但路径已知；榜首 23.3x。 |
| P3（门控） | T44 chain_speculative_sampling | 见「T44 前置实验门」；atol=0 + 昆仑采样族崩溃墙 + 燧原三毒点，无 go 实验不投入。 |

## 各题契约锁定与已知陷阱

### T42 act_and_mul（moe/act_and_mul）

契约：`act_and_mul(gateup_output, activation="silu", swiglu_limit=None)`；
`[M, 2H] → [M, H]`；fp32 1e-4 / bf16 1.5e-2 / fp16 1e-2。

1. **精度切换点与 T7/T29/T39 不同**：参考是
   `act.to(input dtype) * up.to(input dtype)`——激活（含 clamp）在 fp32 算完后
   **cast 回输入 dtype，再在输入 dtype 下做乘法**。旧模板是 fp32 乘完才 store
   cast，不能照抄 store 行。
2. clamp 不对称且在激活前：`gate = clamp(gate, max=limit)`（只 max），
   `up = clamp(up, -limit, limit)`（对称）；发生在 fp32 cast 之后。
3. SiLU 逐字写题面形式 `x / (1 + exp(-x))`（fp32），禁数值稳定化改写
   （T39 `-92/-90` 反例实证改写会错）。
4. gelu 用题面钦定 tanh 近似
   `0.5*x*(1+tanh(0.7978845608*(x+0.044715*x³)))`，纯算术路径无 erf 崩溃
   风险；S0 用稳定 tanh（`s*(1-e^{-2|x|})/(1+e^{-2|x|})`），`tl.math.tanh`
   留作后续 vendor 单变量（T24/T40 昆仑/燧原 +13%/+17% 实证）。
5. activation 非法值 host 层 `ValueError`（匹配参考；非设备 fallback）。
6. 模板：`src/flaggems_sglang/ops/gelu_and_mul.py`（row/col-block + capped
   65535 grid-stride + BLOCK_COL=1024）；`swiglu_limit`/activation 都是
   host 已知量 → constexpr 特化（HAS_LIMIT/ACT_IS_GELU），不构成燧原禁的
   运行期分支。
7. vendor 预案（平台实证，勿重复试错）：昆仑唯一有效轴 BLOCK=2048、
   禁 device 标量 gating；燧原整行 BLOCK 4096 四证 + 禁 kernel 内运行期
   分支；华为 BLOCK 512 -42% 勿碰；沐曦 flat 大 BLOCK；国际 B 四档列 tile
   autotune +3.83%；国际 A autotune 证伪不投。
8. Triton 3.7 禁 jit 内读模块级普通全局，常量内联或 constexpr。

### T43 causal_conv1d_update（mamba/causal_conv1d_update）

契约：`causal_conv1d_update(x, conv_state, weight, bias=None, activation="silu")`；
返回 `(out, conv_state_new)`；fp32 计算后 cast；2D x 视 seqlen=1。

1. **输出语义按题面不按 SGLang**：返回二元组；conv_state out-of-place
   （读原 state、写独立 buffer，T36 E21 source/destination 分离原则）；
   `empty_like` + 全覆盖替代 clone（E22 实证省一份拷贝，本题 state 全覆盖安全）。
2. kernel 结构：每 (batch, dim-tile) 一个 program；width constexpr 展开；
   state+新 token 一次载入寄存器，fp32 FMA 累加、bias+silu 融合、同时写出
   out 和前移 state——单 kernel 单次读写接近 I/O 下界。
3. **别抄 SGLang `_causal_conv1d_update_kernel` 原样当昆仑答案**：本仓 vendor
   patch 实证其在昆仑 XPU fp16/bf16 直接编译失败（类型断言/uni_sram）；fp32
   计算同时是精度要求和绕坑手段。
4. 昆仑预案：flat/direct 简单 grid、无动态 device loop、stages1；uni_sram
   先试 `isCloseCoreTiling=True`（T36 E13 破墙）；不删逻辑恒真 runtime mask
   （E23 idle-core 越界）。
5. 华为 grid≤65535：flat 1D capped grid。
6. wrapper 全参 `.contiguous()`；可选参数（bias/activation）constexpr 旗标 +
   占位张量（T36 L177–189 先例）。
7. 预期管理：reference 无 `.item()` 同步、seqlen≈1，天花板是「融合 N 个小
   launch」量级而非 T1 的 13x；0/9 队达标的真门大概率在昆仑/燧原正确性。
8. 3D x 的 seqlen>1：program 内 `tl.static_range` 顺序小循环 + 窗口递推；
   state 前移取 x_cat 尾部 state_len 列。

### T45 chunk_scaled_dot_kkt（fla/chunk_scaled_dot_kkt）

契约：`chunk_scaled_dot_kkt(k, beta, g_cumsum=None, chunk_size=64)`；
k `[B,T,Hg,K]`、beta `[B,T,H]`、输出 `[B,T,H,BT]` float32。

1. 主体复用 `bmm_chunk.py` 骨架：grid `(BT M/N tiles, B, NT*H)`；kernel 内
   `chunk_id*BT + m/n_offsets` 定位；k 一次加载同时供 dot 两侧（K@K.T）。
2. **输出直接写最终布局**：store 偏移 `b*sb + (n*BT+i)*st + h*sh + j*sl`
   四维 stride 显式传入，跳过 reference 的 permute/reshape；wrapper 断言
   `T % BT == 0`、`H % Hg == 0`。
3. 合成顺序：dot 累加（FP32）→ `A *= beta[i]`（beta 用自身 stride，与 k 不同）
   → safe-exp（`d=g[i]-g[j]`，`tl.where(d<=0, tl.exp(d), 0)`，FP32）→ store
   mask 加 `i > j`（严格下三角，对角线必须精确 0，专项回归锁定）。
4. GQA repeat 零物化：kernel 内 `head // (H//Hg)` 索引（chunk_state.py L85）。
5. dtype 分派：fp16/bf16 低精度直送 dot + fp32 累加（bmm_chunk E3
   USE_INPUT_DTYPE 模式）；fp32 输入 `input_precision="ieee"` 禁 TF32。
   32/32/32、4 warps、1 stage 起步求 8/8。
6. vendor 预案：天数 fp32 路径 split-fp16 三点积（1e-4 必须）；燧原
   fp16 dot + 64/64/128 + stages≥2 + capped fold（cap 64）；华为预防性
   capped grid-stride（`min(total,4096)`）；昆仑只许 fp32-ieee dot；沐曦/
   国际 B 低精度回退时保留 ieee 字节回退 vendor；国际 A 低精度预期大收益。
7. 回归清单：GQA ratio 1/2/3；g_cumsum 有无；safe-exp 正负边界（`g_diff>0`
   置 0）；对角线 0；BT/K 非 2 幂（31/33/65）；非连续 stride。

### T46 chunked_embedding_lora_a（lora/chunked_embedding_lora_a）

契约：`chunked_embedding_lora_a(input_ids, weights, batch_info, vocab_size)`；
weights `[num_lora, max_rank, vocab_size]`；输出 `[S, max_rank]` 未覆盖行为 0。

1. 直接改 `embedding_lora_a.py`（T17 平台 8/8 验证字节的后代）：grid
   `(max_len, bs)`、BLOCK_RANK=128 + runtime rank loop 尾 mask、`torch.zeros`
   起步、4 warps/1 stage。
2. **读序铁律**：先 `seg_indptr[b:b+2]` 判空 early-return，后读
   `weight_indices`/`lora_ranks`（空段越界哨兵实证）；删除 seg_lens 依赖。
3. rank < max_rank 的右侧列保持零（`rank_offsets < r` mask，T17 同款）。
4. permutation 管输出行；token 行由 `input_ids[permutation[...]]` 取。
5. 首投即带 `_ascend`/`_kunlunxin` token 折叠 vendor（`token_cap=65535//bs`）
   和 `_enflame` i32 route+gather vendor（wrapper 降 i32 + 删 int64 cast）；
   `_hygon` warps=2 低风险加分。
6. 本题与 T17 差异点：无 extra_embeddings/越界 token 语义（题面未提及则
   不实现，不臆造）；确认 harness 的 batch_info 字段集。

### T47 chunked_sgmv_expand（lora/chunked_sgmv_expand）

契约：`chunked_sgmv_expand(x, weights, batch_info, slice_offsets, max_slice_size, base_output)`；
`out[rows, slice] += scaling * x_slice @ W_slice.T`；返回新张量（clone 语义）。

1. 骨架映射 `qkv_lora_b.py`：`output_offset→slice_offsets`、`RANK→r`；
   grid `(cdiv(max_len,BLOCK_S)*output_blocks, n_slices, bs)`；64/128/32、
   4 warps、stages≤2。
2. `base_output.clone()` 起步、kernel 内 RMW（load base → fp32 →
   `base + acc*scaling` → cast 回）；输入不变性是平台测试项。
3. scaling per-adapter 标量（同一 adapter 所有 slice 共用）；rank==0 no-op；
   题面按 rank 维度「有效 rank」语义需确认是否像 T17 截断（T22/T28 族
   只把 0 当 no-op）。
4. K=r 很小：`mask_k` pad，dot 维度 <16 必须垫。
5. vendor 预案：天数 split-fp16 四点积（fp32-ieee 静默错是必踩坑）；燧原
   metadata 降 i32、失败即 route/materialize + 64³ GEMM；昆仑直接上
   route/materialize + 32³/stages1/`do_not_specialize=["M"]`（不要重走 T23
   五连败）；华为 3D grid 可用、超限折叠。
6. 尾块 mask 一律绝对列号（T37 E1 平台 99% 失配根因）；回归补 r/out 非
   2 幂（65/80/129）、非等宽 slice、空段、rank0、S=0。

### T44 chain_speculative_sampling（sampling_grammar/…）——门控投入

契约：整数输出 atol=0（三 dtype 全零容差）；B×S 串行接受链 + V 维逆 CDF。

**go/no-go 实验已完成（2026-09-04，NVIDIA 代理）——半精度判定 NEGATIVE**：
torch 半精度 `cumsum` 为半精度累加的内部树形扫描，与 fp32 串行 round /
dtype 逐步累加 / `round(cumsum(f32))` / blocked+Hillis 结构仿真全部
不一致（各 0/10–0/20）；端到端 fp32 0/10 失配、fp16 2/10、bf16 3/10
（仅最终 token 偏移，接受链全对）。八芯 torch 构建差异叠加风险。
**候选保留但带 limitation（fp32 全对）；提交与否留给用户门控**，详见
`experiments/chain_speculative_sampling.md`。

已知墙（五路调研结论）：
- atol=0 要求同时 bit-exact 复现 torch.sum/cumsum——账本零先例，逐芯
  归约序不同且无文档（最近先例 T33 单条除法用 div_rn 解决）；
- 昆仑 topk/采样族崩溃墙六题复现（T25/26/27/31/38 + T25 argmax mismatch）；
- 燧原三毒点叠加 argmax 硬限制（T25 七结构全灭）：cumsum、运行时分支、
  标量 masked load、i64 IR。
- 可行面：接受链可无损并行化（`cur_row` 恒为 `s-1`，接受数 = 前缀连续
  True 长度 = 首索引归约，draft_topk1 式）；性能无忧（reference 逐 `.item()`）。
- 若投入：零 while、零运行时分支、显式 IEEE 算子（div_rn 家族）、int32
  元数据、无混合 dtype where；5 发预算 + 同指纹 2 次止损，昆仑崩溃指纹
  出现即封存。

## 提交节奏与预算纪律

- 每题默认 5 发：S0 探路 → 最多 3 次 vendor 单变量 → 1 发回归储备；
  同指纹失败连 2 次提前停。
- 单变量原则：每轮只改一个 vendor，其余字节逐字节冻结；冻结芯分数变化
  按平台水位处理。
- 首日（09-04）排程建议：T42 S0 → T43 S0 → T46 S0 →（视结果）T45 S0。
- 中段（09-05 起）按 S0 结果逐芯迭代；09-09 前完成主要冲分，09-10 留
  终投回归。
- 海光水位波动大（T39 实证 34–56x），单轮高值不当结构收益。
- 昆仑崩溃族协议沿用：平台侧崩溃不计止损、封存候选等健康窗口、重载
  每发需用户当次明示授权。

## 与既有文档的关系

- 逐芯技术事实详单：`learning-path.md`、`chip-landscape.md`、
  `season2-retrospective.md`（毒点表）。
- 单题当前状态：`docs/competition/experiments/<operator>.md` 的 `current`
  块是唯一人工状态真相；本文档不承载单题状态。
- 打包与提交流程：skill 主文档 + `references/platform-workflow.md`。
