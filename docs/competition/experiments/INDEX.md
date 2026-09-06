# 实验状态索引（GENERATED）

> 由 `tools/gen_experiment_index.py` 从各账本顶部 ` ```current ` 块生成，
> 不要手改本文件；状态更新只改账本 CURRENT 块，然后重跑脚本。

| Task | 算子 | 有效性 | 平台 | 团队最佳 | 封存 | 下一步 | 更新 | 账本 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 25 | draft_topk1 | invalid | 6/8 | e6c | yes | 需外部证据(他人通过样例/平台澄清)才可重启 | 2026-08-31 | [draft_topk1](draft_topk1.md) |
| 26 | fused_moe_router_cudacore | invalid | 6/8 | e5-e8(等价) | yes | 平台 Q&A 澄清或他人华为方案公开 | 2026-08-31 | [fused_moe_router_cudacore](fused_moe_router_cudacore.md) |
| 27 | fused_moe_router_tensorcore | valid | 8/8(e9,1.039975x) | e9 1.016425x | yes | 昆仑需约6.5772x才登顶,无可信路径;转T36 | 2026-09-01 | [fused_moe_router_tensorcore](fused_moe_router_tensorcore.md) |
| 28 | gate_up_lora_b | valid | 8/8(e14,14.98025x) | e14 14.98025x | yes | 采样两连 TB(e13 14.4435/e14 14.98025);封存,明日 1-2 发守榜采样 | 2026-09-02 | [gate_up_lora_b](gate_up_lora_b.md) |
| 29 | gelu_and_mul | valid | 8/8(e9,2.805042x) | e9 2.805042x | yes | exact-erf 官方实现与 minimax 数值边界复核;预期收益远不足66.68%榜差,封存 | 2026-09-02 | [gelu_and_mul](gelu_and_mul.md) |
| 30 | interleaved_rope | valid | 8/8(s1,25.9236875x) | s1 25.9236875x | yes | 实时榜首37.7641;一读一写下界已达,MCP/官方实现复核无可信46.17%路径 | 2026-09-02 | [interleaved_rope](interleaved_rope.md) |
| 31 | moe_fused_gate | invalid_correctness | E9 sub8270 7/8;Kunlun 1833723ms 同指纹(第16例) | e7(=e6字节载体) 七芯~7.73x | yes | 永久封存;仅平台工单回应+他队结构公开或昆仑修复后以 e7 载体单发重验 | 2026-09-02 | [moe_fused_gate](moe_fused_gate.md) |
| 32 | moe_fused_mul_sum | valid | 8/8 | S0 4.4829x | yes | e5 三框架独立reduce同构;流量理想上限仅+22.7%,无法解释433%榜差 | 2026-09-01 | [moe_fused_mul_sum](moe_fused_mul_sum.md) |
| 33 | per_token_group_quant_int8 | valid | 8/8(e14,5.582775x) | e14 5.582775x | yes | e13 官方 constexpr/direct/subwarp/M8 家族全未过门;仅新 vendor subgroup 证据可重开 | 2026-09-02 | [per_token_group_quant_int8](per_token_group_quant_int8.md) |
| 34 | per_token_quant_int8 | valid | 8/8 | e1 4.7131x | yes | e4 persistent cap与SGLang launch参数均不过5%全矩阵门;已知轴尽 | 2026-09-01 | [per_token_quant_int8](per_token_quant_int8.md) |
| 35 | rotary_embedding | valid | 8/8(E10,7.047975x,team best) | e10 7.047975x | yes | 封存;仅燧原水位恢复信号时以 E10 字节重载(≤2 发,E10 ZIP 已验签在库) | 2026-09-02 | [rotary_embedding](rotary_embedding.md) |
| 36 | selective_state_update | invalid_correctness | 7/8(e29;昆仑compile-worker崩溃) | e22(correctness) 5.1200625x | no | persisted-slice轴已关闭;仅有全新源码级结构证据时重开 | 2026-09-01 | [selective_state_update](selective_state_update.md) |
| 37 | sgemm_lora_a | valid | 8/8(E5,5.3471x,rank6/6;E7 5.168非TB) | e5 5.3470625x | yes | E7 stages轴平台证伪(沐曦-20%/card_b-10%,代理+11%不迁移);树回滚E5字节,收盘 | 2026-09-02 | [sgemm_lora_a](sgemm_lora_a.md) |
| 38 | sigmoid_gate_topk_renorm | invalid_correctness | E5 sub 8170 7/8;Kunlun快速执行但9/9数值失败 | S0 | yes | T38封存;切换其他任务 | 2026-09-02 | [sigmoid_gate_topk_renorm](sigmoid_gate_topk_renorm.md) |
| 39 | silu_and_mul_masked | valid | 8/8 | E7 19.8698x | yes | - | 2026-09-03 | [silu_and_mul_masked](silu_and_mul_masked.md) |
| 40 | softcap_inplace_logits | valid | 8/8(e8,2.195604x,rank1) | e8 2.195604x | yes | 额度用尽收官;e16 华为字节(1.70)+TB 2.1956 #2;榜首 c2flow 2.2593 华为 3.40 未破译 | 2026-09-03 | [softcap_inplace_logits](softcap_inplace_logits.md) |
| 41 | state_passing | invalid | E7 sub8079 system_failed;7 pass,Kunlun 0.0065x,Enflame unrun | E5 diagnostic | yes | 本轮收盘;只保留平台工单与全新结构研究 | 2026-09-02 | [state_passing](state_passing.md) |
| 42 | act_and_mul | valid | 8/8(e2,3.248925x) | e6 3.25835x | no | M1载体成TB(3.25835,10344);三候选全消费;守榜,距旧榜首3.5194差-7.4%无新轴 | 2026-09-06 | [act_and_mul](act_and_mul.md) |
| 43 | causal_conv1d_update | invalid | 7/8(e12,昆仑10形态全错译;conclusive) | s0 -x | yes | 昆仑轴conclusive封存;仅全新结构证据(非gather/非广播/非FMA)或平台修复可重开 | 2026-09-06 | [causal_conv1d_update](causal_conv1d_update.md) |
| 44 | chain_speculative_sampling | invalid | 探针确认全芯失败(predicts mismatch=半精度bit-exact) | s0 -x | no | conclusive 封轴;天数/A/B同款517值=fp32也有差;平台测半精度 | 2026-09-04 | [chain_speculative_sampling](chain_speculative_sampling.md) |
| 45 | chunk_scaled_dot_kkt | invalid | 7/8(e7,仅昆仑失败;华为0.0455→0.2535x过门槛!) | e7 -x | no | E14 exp2证伪已回滚e13(0.063x八芯过correctness);T45暂停守invalid转T46 | 2026-09-06 | [chunk_scaled_dot_kkt](chunk_scaled_dot_kkt.md) |
| 46 | chunked_embedding_lora_a | valid | 8/8(e3,14.1051875x) | e3 14.1051875x | no | e6天数预路由水位带内无法归因,轴关闭;维持E3守榜(14.105x);后续按逐芯晋级纪律择机 | 2026-09-06 | [chunked_embedding_lora_a](chunked_embedding_lora_a.md) |
| 47 | chunked_sgmv_expand | valid | 8/8(e5,25.0048125x,榜首) | e5 25.0048125x | no | 守榜首(25.00x vs 前榜首23.33x);燧原0.178x/昆仑3.72x为余量轴;stale意图经用户授权归档重提(移至archived/) | 2026-09-04 | [chunked_sgmv_expand](chunked_sgmv_expand.md) |
| 48 | chunked_sgmv_shrink | valid | 8/8(e6,4.7198125x) | e6 4.7198125x | no | 守榜;冲分轴:燧原0.58x/昆仑1.79x/天数3.85x | 2026-09-05 | [chunked_sgmv_shrink](chunked_sgmv_shrink.md) |
| 49 | ernie45_rope_fused | invalid | 7/8(e2,昆仑uni_sram墙已3投;conclusive) | e1 -x | no | 昆仑uni_sram墙(需更小tile/单head);7/8已是好成绩 | 2026-09-05 | [ernie45_rope_fused](ernie45_rope_fused.md) |
| 50 | extend_attention | candidate-wip | 6/8(e4,昆仑conclusive封轴;华为数值不可修;燧原0.013x<门槛) | - -x | no | 华为深层数值问题(online+two-pass均败);5/8已是208发1队过线题的好成绩;冲分优先 | 2026-09-05 | [extend_attention](extend_attention.md) |
| 51 | fla_layernorm_gated | valid | 8/8(e1,5.390025x) | e1 5.390025x | no | e3华为constexpr-D仅+3.7%证伪关闭;昆仑0.97已修;剩沐曦rsqrt;燧原病态在评 | 2026-09-06 | [fla_layernorm_gated](fla_layernorm_gated.md) |
| 52 | fused_dual_residual_rmsnorm | invalid | 6/8(e5,五种数学形态同指纹边界失配;已分诊) | - -x | yes | 封存6/8;仅sqrt_rn+div_rn或torch归约树复刻等全新结构证据可重开 | 2026-09-06 | [fused_dual_residual_rmsnorm](fused_dual_residual_rmsnorm.md) |
| 53 | fused_gdn_gating | valid | 8/8(e1,1.769075x) | e1 1.769075x | no | e2七芯过,燧原(目标芯)卡病态盒子在评;出分即判轴;燧原停火直至恢复 | 2026-09-06 | [fused_gdn_gating](fused_gdn_gating.md) |
| 54 | fused_norm_rope_stacked | pending | 5/8(e2) | e1 0x | no | 当日额度耗尽;e2=5/8(行式vendor未修复三芯);代理fuzz仅覆盖已测输入;固定源码复现并定位首个分歧 | 2026-09-06 | [fused_norm_rope_stacked](fused_norm_rope_stacked.md) |
| 55 | hc_head | pending | 7/8(s0,昆仑pending) | s0 0x | no | 昆仑pending;燧原0.257x弱;榜首6.25x | 2026-09-06 | [hc_head](hc_head.md) |
| 56 | l2norm | valid | 8/8(s0,3.10497917x) | s0 3.10497917x | no | 高维stride修复版GPU release 7 tests通过;未提交平台;8/8仍指历史s0 | 2026-09-06 | [l2norm](l2norm.md) |
| 57 | log_scaling_tau | pending | 7/8(s0,燧原pending) | s0 0x | no | 燧原pending;昆仑0.47/华为0.53偏弱;榜首2.82x | 2026-09-06 | [log_scaling_tau](log_scaling_tau.md) |
| 58 | w8a8_block_int8_matmul | valid | 8/8(e1,118.15512x) | e1 118.15512x | no | 冲分轴:燧原4.48x/B7.68x;榜首209.3x差43%;skill沉淀昆仑向量整除崩溃教训 | 2026-09-06 | [w8a8_block_int8_matmul](w8a8_block_int8_matmul.md) |

缺 CURRENT 块（未计入索引）：apply_token_bitmask.md、bmm_chunk.md、chunk_cumsum.md、chunk_local_cumsum_vector.md、chunk_state.md、chunk_state_varlen.md、context_attention.md、decode_attention.md、decode_grouped_attention.md、embedding_lora_a.md、fused_recurrent_gdn.md、fused_rmsnorm.md、mamba_layernorm_gated.md、moe_sum_reduce.md、qkv_lora_b.md、sgemm_lora_b.md、softcap_out.md
