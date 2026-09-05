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
| 42 | act_and_mul | valid | 8/8(e2,3.248925x) | e2 3.248925x | no | 榜首Warmhearted升至3.5194x(活跃迭代中);剩余轴:昆仑0.456/沐曦2.40/华为2.258;AMD四档autotune可试(+3.83%实证) | 2026-09-03 | [act_and_mul](act_and_mul.md) |
| 43 | causal_conv1d_update | invalid | 7/8(e7,仅昆仑失败;华为0.4435x+燧原0.325x双双过门槛!) | s0 -x | no | 昆仑uni_sram编译墙(width-axis形态也触发了);7/8=此题最优可达;守榜 | 2026-09-03 | [causal_conv1d_update](causal_conv1d_update.md) |
| 44 | chain_speculative_sampling | invalid | 探针确认全芯失败(predicts mismatch=半精度bit-exact) | s0 -x | no | conclusive 封轴;天数/A/B同款517值=fp32也有差;平台测半精度 | 2026-09-04 | [chain_speculative_sampling](chain_speculative_sampling.md) |
| 45 | chunk_scaled_dot_kkt | invalid | 7/8(e7,仅昆仑失败;华为0.0455→0.2535x过门槛!) | e7 -x | no | 昆仑崩溃族(compile_worker Aborted,非代码);7/8=此题最优可达;守榜 | 2026-09-04 | [chunk_scaled_dot_kkt](chunk_scaled_dot_kkt.md) |
| 46 | chunked_embedding_lora_a | valid | 8/8(e3,14.1051875x) | e3 14.1051875x | no | 守榜:昆仑segment-owned已固化(0.23→0.64),e3=14.105x为team best;后续发收益被水位噪声(±1.5均值)淹没,停止 | 2026-09-03 | [chunked_embedding_lora_a](chunked_embedding_lora_a.md) |
| 47 | chunked_sgmv_expand | valid | 8/8(e5,25.0048125x,榜首) | e5 25.0048125x | no | 守榜首(25.00x vs 前榜首23.33x);燧原0.178x/昆仑3.72x为余量轴;stale意图经用户授权归档重提(移至archived/) | 2026-09-04 | [chunked_sgmv_expand](chunked_sgmv_expand.md) |
| 48 | chunked_sgmv_shrink | valid | 8/8(e6,4.7198125x) | e6 4.7198125x | no | 守榜;冲分轴:燧原0.58x/昆仑1.79x/天数3.85x | 2026-09-05 | [chunked_sgmv_shrink](chunked_sgmv_shrink.md) |
| 49 | ernie45_rope_fused | invalid | 7/8(e2,昆仑uni_sram墙已3投;conclusive) | e1 -x | no | 昆仑uni_sram墙(需更小tile/单head);7/8已是好成绩 | 2026-09-05 | [ernie45_rope_fused](ernie45_rope_fused.md) |
| 50 | extend_attention | candidate-wip | 5/8+(S0在评;5芯过门槛;华为correctness败;燧原/昆仑在评) | - -x | no | 调试多batch路径(单token已精确);榜首c2flow 3.60x,1队过线 | 2026-09-05 | [extend_attention](extend_attention.md) |
| 51 | fla_layernorm_gated | valid | 8/8(e1,5.390025x) | e1 5.390025x | no | 守榜;冲分轴:昆仑0.944/华为2.52 | 2026-09-05 | [fla_layernorm_gated](fla_layernorm_gated.md) |
| 52 | fused_dual_residual_rmsnorm | invalid | 5/8(e2,天数/燧原/昆仑/card_b 差丝米级精度) | - -x | no | 精度追踪边际递减(3种除法形式各自修一芯破另一芯);建议暂停转T49 | 2026-09-05 | [fused_dual_residual_rmsnorm](fused_dual_residual_rmsnorm.md) |
| 53 | fused_gdn_gating | valid | 8/8(e1,1.769075x) | e1 1.769075x | no | 守榜;冲分轴:昆仑0.72x/燧原0.17x/沐曦1.34x | 2026-09-05 | [fused_gdn_gating](fused_gdn_gating.md) |

缺 CURRENT 块（未计入索引）：apply_token_bitmask.md、bmm_chunk.md、chunk_cumsum.md、chunk_local_cumsum_vector.md、chunk_state.md、chunk_state_varlen.md、context_attention.md、decode_attention.md、decode_grouped_attention.md、embedding_lora_a.md、fused_recurrent_gdn.md、fused_rmsnorm.md、mamba_layernorm_gated.md、moe_sum_reduce.md、qkv_lora_b.md、sgemm_lora_b.md、softcap_out.md
