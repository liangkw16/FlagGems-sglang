# 实验状态索引（GENERATED）

> 由 `tools/gen_experiment_index.py` 从各账本顶部 ` ```current ` 块生成，
> 不要手改本文件；状态更新只改账本 CURRENT 块，然后重跑脚本。

| Task | 算子 | 有效性 | 平台 | 团队最佳 | 封存 | 下一步 | 更新 | 账本 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 25 | draft_topk1 | invalid | 6/8 | e6c | yes | 需外部证据(他人通过样例/平台澄清)才可重启 | 2026-08-31 | [draft_topk1](draft_topk1.md) |
| 26 | fused_moe_router_cudacore | invalid | 6/8 | e5-e8(等价) | yes | 平台 Q&A 澄清或他人华为方案公开 | 2026-08-31 | [fused_moe_router_cudacore](fused_moe_router_cudacore.md) |
| 27 | fused_moe_router_tensorcore | valid | 8/8(e8,1.016425x,rank3) | e8 1.016425x | yes | 昆仑需约6.5772x才登顶,无可信路径;转T36 | 2026-09-01 | [fused_moe_router_tensorcore](fused_moe_router_tensorcore.md) |
| 28 | gate_up_lora_b | invalid | 7/8 | e9 七芯~18.5x | yes | 工单 rerun 或用户授权注释载体重载 | 2026-08-31 | [gate_up_lora_b](gate_up_lora_b.md) |
| 29 | gelu_and_mul | valid | 8/8 | e8 2.7394x | yes | exact-erf 官方实现与 minimax 数值边界复核;预期收益远不足66.68%榜差,封存 | 2026-09-01 | [gelu_and_mul](gelu_and_mul.md) |
| 30 | interleaved_rope | valid | 8/8 | S0 25.8353x | yes | 实时榜首37.7641;一读一写下界已达,MCP/官方实现复核无可信46.17%路径 | 2026-09-01 | [interleaved_rope](interleaved_rope.md) |
| 31 | moe_fused_gate | invalid | 7/8 | e7(=e6字节载体) 七芯~7.73x | yes | 仅工单或结构改写(topk 形态) | 2026-08-31 | [moe_fused_gate](moe_fused_gate.md) |
| 32 | moe_fused_mul_sum | valid | 8/8 | S0 4.4829x | yes | e5 三框架独立reduce同构;流量理想上限仅+22.7%,无法解释433%榜差 | 2026-09-01 | [moe_fused_mul_sum](moe_fused_mul_sum.md) |
| 33 | per_token_group_quant_int8 | valid | 8/8 | e10 5.5720x | yes | 重新封存 e10 5.5720;e12 去 contiguous/warps 2,4,8 全部低于5%门,已知参数轴尽 | 2026-09-01 | [per_token_group_quant_int8](per_token_group_quant_int8.md) |
| 34 | per_token_quant_int8 | valid | 8/8 | e1 4.7131x | yes | e4 persistent cap与SGLang launch参数均不过5%全矩阵门;已知轴尽 | 2026-09-01 | [per_token_quant_int8](per_token_quant_int8.md) |
| 35 | rotary_embedding | valid | 8/8 | e3 5.8458x | yes | e5 generic heads_tile 1/2/8/16 全部低于5%门;tile4局部最优,收盘 | 2026-09-01 | [rotary_embedding](rotary_embedding.md) |
| 36 | selective_state_update | invalid_correctness | 7/8(e29;昆仑compile-worker崩溃) | e22(correctness) 5.1200625x | no | persisted-slice轴已关闭;仅有全新源码级结构证据时重开 | 2026-09-01 | [selective_state_update](selective_state_update.md) |
| 37 | sgemm_lora_a | invalid | 6/8(上限 7/8) | e2 | yes | 封存;候选可复用 | 2026-08-31 | [sgemm_lora_a](sgemm_lora_a.md) |
| 38 | sigmoid_gate_topk_renorm | invalid | 7/8 | S0 | yes | E2 七芯均值4.9415x(+4.17倍)但昆仑同族崩溃;仅平台runtime修复后重开 | 2026-09-01 | [sigmoid_gate_topk_renorm](sigmoid_gate_topk_renorm.md) |
| 39 | silu_and_mul_masked | valid | 8/8 | E7 19.8698x | yes | E14海光packing六轴八格均未过5%门;不提交,转向其它高登顶概率任务 | 2026-09-01 | [silu_and_mul_masked](silu_and_mul_masked.md) |
| 40 | softcap_inplace_logits | valid | 8/8 | e6 1.7679x | yes | e9 generic direct仅+1.63~3.40%;需+32.25%才登顶,官方可迁移轴已尽 | 2026-09-01 | [softcap_inplace_logits](softcap_inplace_logits.md) |
| 41 | state_passing | invalid | 7/8(8/8 terminal) | E3 | yes | e4 组合候选(fc6dd4f)已 staged;重载需用户逐发授权 | 2026-08-31 | [state_passing](state_passing.md) |

缺 CURRENT 块（未计入索引）：apply_token_bitmask.md、bmm_chunk.md、chunk_cumsum.md、chunk_local_cumsum_vector.md、chunk_state.md、chunk_state_varlen.md、context_attention.md、decode_attention.md、decode_grouped_attention.md、embedding_lora_a.md、fused_recurrent_gdn.md、fused_rmsnorm.md、mamba_layernorm_gated.md、moe_sum_reduce.md、qkv_lora_b.md、sgemm_lora_b.md、softcap_out.md
