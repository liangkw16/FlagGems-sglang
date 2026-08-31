# 实验状态索引（GENERATED）

> 由 `tools/gen_experiment_index.py` 从各账本顶部 ` ```current ` 块生成，
> 不要手改本文件；状态更新只改账本 CURRENT 块，然后重跑脚本。

| Task | 算子 | 有效性 | 平台 | 团队最佳 | 封存 | 下一步 | 更新 | 账本 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 25 | draft_topk1 | invalid | 6/8 | e6c | yes | 需外部证据(他人通过样例/平台澄清)才可重启 | 2026-08-31 | [draft_topk1](draft_topk1.md) |
| 26 | fused_moe_router_cudacore | invalid | 6/8 | e5-e8(等价) | yes | 平台 Q&A 澄清或他人华为方案公开 | 2026-08-31 | [fused_moe_router_cudacore](fused_moe_router_cudacore.md) |
| 27 | fused_moe_router_tensorcore | invalid | 7/8(七芯过,仅昆仑) | e7 | yes | 平台工单;README 早期 6/8 口径已过期 | 2026-08-31 | [fused_moe_router_tensorcore](fused_moe_router_tensorcore.md) |
| 28 | gate_up_lora_b | invalid | 7/8 | e9 七芯~18.5x | yes | 工单 rerun 或用户授权注释载体重载 | 2026-08-31 | [gate_up_lora_b](gate_up_lora_b.md) |
| 29 | gelu_and_mul | valid | 8/8 | e8 2.7394x | partial | 燧原 +42% 已兑现;华为轴 autotune 负结果,无已验证新杠杆 | 2026-08-31 | [gelu_and_mul](gelu_and_mul.md) |
| 30 | interleaved_rope | valid | 8/8 | S0 25.8353x | yes | 代理可见轴局部最优;榜首 36.4 未解释 | 2026-08-31 | [interleaved_rope](interleaved_rope.md) |
| 31 | moe_fused_gate | invalid | 7/8 | e7(=e6字节载体) 七芯~7.73x | yes | 仅工单或结构改写(topk 形态) | 2026-08-31 | [moe_fused_gate](moe_fused_gate.md) |
| 32 | moe_fused_mul_sum | valid | 8/8 | S0 4.4829x | yes | 收盘 S0 4.4829;e4 零权重跳过证伪(drop 面不存在+谓词代价),榜首 23.9 判定高水位产物 | 2026-08-31 | [moe_fused_mul_sum](moe_fused_mul_sum.md) |
| 33 | per_token_group_quant_int8 | valid | 8/8 | e10 5.5720x | no | e11 metax tile32 已提交待评测(ZIP 8f9ec1b3);单轴门沐曦 > 4.328 | 2026-08-31 | [per_token_group_quant_int8](per_token_group_quant_int8.md) |
| 34 | per_token_quant_int8 | valid | 8/8 | e1 4.7131x | yes | 收盘 e1 4.7131;e3 row-pack 平台中性+夹带 e2 昆仑坏字节致 invalid,树已回滚 | 2026-09-01 | [per_token_quant_int8](per_token_quant_int8.md) |
| 35 | rotary_embedding | valid | 8/8 | e3 5.8458x | yes | e4 燧原宽瓦片证伪(宽瓦片族首反例),收盘 | 2026-08-31 | [rotary_embedding](rotary_embedding.md) |
| 36 | selective_state_update | invalid | 7/8 | e8 七芯~5.8x | yes | 平台修复后一发转正(候选已封存) | 2026-08-31 | [selective_state_update](selective_state_update.md) |
| 37 | sgemm_lora_a | invalid | 6/8(上限 7/8) | e2 | yes | 封存;候选可复用 | 2026-08-31 | [sgemm_lora_a](sgemm_lora_a.md) |
| 38 | sigmoid_gate_topk_renorm | invalid | 7/8 | S0 | yes | 平台工单;候选封存可复用 | 2026-08-31 | [sigmoid_gate_topk_renorm](sigmoid_gate_topk_renorm.md) |
| 39 | silu_and_mul_masked | valid | 8/8 | E7 19.8698x | yes | 守榜(对 Fields);重载两滚低分关轴,榜首 23.85 结构性差距 | 2026-08-31 | [silu_and_mul_masked](silu_and_mul_masked.md) |
| 40 | softcap_inplace_logits | valid | 8/8 | e6 1.7679x | yes | 收盘 e6 1.7679(-12.6%);e7 generic2048/metax4096 双证伪,全部已知轴关闭,I/O 已贴下界 | 2026-08-31 | [softcap_inplace_logits](softcap_inplace_logits.md) |
| 41 | state_passing | invalid | 7/8(8/8 terminal) | E3 | yes | e4 组合候选(fc6dd4f)已 staged;重载需用户逐发授权 | 2026-08-31 | [state_passing](state_passing.md) |

缺 CURRENT 块（未计入索引）：apply_token_bitmask.md、bmm_chunk.md、chunk_cumsum.md、chunk_local_cumsum_vector.md、chunk_state.md、chunk_state_varlen.md、context_attention.md、decode_attention.md、decode_grouped_attention.md、embedding_lora_a.md、fused_recurrent_gdn.md、fused_rmsnorm.md、mamba_layernorm_gated.md、moe_sum_reduce.md、qkv_lora_b.md、sgemm_lora_b.md、softcap_out.md
