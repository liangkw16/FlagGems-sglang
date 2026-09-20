# 第二届 FlagOS 算子赛题索引

> 来源：[比赛页](https://flagos.io/race-detail-season2?id=782kzq4m)；同步时间：`2026-09-19T23:20:17+08:00`。
> 状态和榜单会变化，运行 `python tools/sync_flagos_season2_docs.py` 更新。

## 第 1 批

| 题号 | 算子 | 状态 | 提交/队伍 | 达标队伍 | 当前第一 | 平均加速比 |
| ---: | --- | --- | ---: | ---: | --- | ---: |
| 1 | [causal_conv1d_fn](tasks/batch-1/01-causal_conv1d_fn.md) | occupied | 114/21 | 1 | EvokeAgent | 13.6331x |
| 2 | [chunk_local_cumsum_scalar](tasks/batch-1/02-chunk_local_cumsum_scalar.md) | occupied | 111/18 | 6 | RSI | 2.3866x |
| 3 | [fused_moe_gemm](tasks/batch-1/03-fused_moe_gemm.md) | occupied | 104/21 | 1 | c2flow | 13.2849x |
| 4 | [merge_state](tasks/batch-1/04-merge_state.md) | occupied | 41/13 | 8 | RSI | 7.3011x |
| 5 | [mrope_fused](tasks/batch-1/05-mrope_fused.md) | occupied | 48/13 | 2 | c2flow | 24.9050x |
| 6 | [per_group_transpose](tasks/batch-1/06-per_group_transpose.md) | occupied | 33/14 | 8 | c2flow | 631.8800x |
| 7 | [silu_and_mul](tasks/batch-1/07-silu_and_mul.md) | occupied | 75/18 | 12 | EvokeAgent | 4.8633x |

## 第 2 批

| 题号 | 算子 | 状态 | 提交/队伍 | 达标队伍 | 当前第一 | 平均加速比 |
| ---: | --- | --- | ---: | ---: | --- | ---: |
| 8 | [apply_token_bitmask](tasks/batch-2/08-apply_token_bitmask.md) | occupied | 188/26 | 18 | hbmu9306 | 709.4368x |
| 9 | [bmm_chunk](tasks/batch-2/09-bmm_chunk.md) | occupied | 262/25 | 13 | RSI | 4.0070x |
| 10 | [chunk_cumsum](tasks/batch-2/10-chunk_cumsum.md) | occupied | 379/31 | 4 | c2flow | 5.5907x |
| 11 | [chunk_local_cumsum_vector](tasks/batch-2/11-chunk_local_cumsum_vector.md) | occupied | 179/26 | 5 | EvokeAgent | 2.1942x |
| 12 | [chunk_state](tasks/batch-2/12-chunk_state.md) | occupied | 170/26 | 12 | RSI | 17.7777x |
| 13 | [chunk_state_varlen](tasks/batch-2/13-chunk_state_varlen.md) | occupied | 128/21 | 6 | c2flow | 707.0045x |
| 14 | [context_attention](tasks/batch-2/14-context_attention.md) | occupied | 185/32 | 1 | EvokeAgent | 3.7924x |
| 15 | [decode_attention](tasks/batch-2/15-decode_attention.md) | occupied | 95/18 | 3 | RSI | 103.3379x |
| 16 | [decode_grouped_attention](tasks/batch-2/16-decode_grouped_attention.md) | occupied | 92/23 | 2 | RSI | 303.0163x |
| 17 | [embedding_lora_a](tasks/batch-2/17-embedding_lora_a.md) | occupied | 186/22 | 8 | RSI | 25.5830x |
| 18 | [fused_recurrent_gdn](tasks/batch-2/18-fused_recurrent_gdn.md) | invalid | 137/30 | 0 | - | - |
| 19 | [fused_rmsnorm](tasks/batch-2/19-fused_rmsnorm.md) | occupied | 114/23 | 18 | torpedo | 703.1127x |
| 20 | [mamba_layernorm_gated](tasks/batch-2/20-mamba_layernorm_gated.md) | occupied | 127/15 | 8 | RSI | 7.4590x |
| 21 | [moe_sum_reduce](tasks/batch-2/21-moe_sum_reduce.md) | occupied | 195/22 | 11 | HAiWORLD | 3.8270x |
| 22 | [qkv_lora_b](tasks/batch-2/22-qkv_lora_b.md) | occupied | 74/11 | 1 | c2flow | 181.7155x |
| 23 | [sgemm_lora_b](tasks/batch-2/23-sgemm_lora_b.md) | occupied | 137/17 | 5 | RSI | 40.5614x |
| 24 | [softcap_out](tasks/batch-2/24-softcap_out.md) | occupied | 175/20 | 14 | MakeYUNAGreatAgain | 58.5631x |

## 第 3 批

| 题号 | 算子 | 状态 | 提交/队伍 | 达标队伍 | 当前第一 | 平均加速比 |
| ---: | --- | --- | ---: | ---: | --- | ---: |
| 25 | [draft_topk1](tasks/batch-3/25-draft_topk1.md) | occupied | 235/29 | 10 | c2flow | 2.2311x |
| 26 | [fused_moe_router_cudacore](tasks/batch-3/26-fused_moe_router_cudacore.md) | occupied | 314/39 | 10 | c2flow | 2.2182x |
| 27 | [fused_moe_router_tensorcore](tasks/batch-3/27-fused_moe_router_tensorcore.md) | occupied | 172/30 | 8 | EvokeAgent | 2.4391x |
| 28 | [gate_up_lora_b](tasks/batch-3/28-gate_up_lora_b.md) | occupied | 178/28 | 4 | RSI | 48.1316x |
| 29 | [gelu_and_mul](tasks/batch-3/29-gelu_and_mul.md) | occupied | 273/32 | 17 | Nectar | 4.5662x |
| 30 | [interleaved_rope](tasks/batch-3/30-interleaved_rope.md) | occupied | 229/27 | 16 | HAiWORLD | 39.3048x |
| 31 | [moe_fused_gate](tasks/batch-3/31-moe_fused_gate.md) | occupied | 175/29 | 9 | RSI | 59.1920x |
| 32 | [moe_fused_mul_sum](tasks/batch-3/32-moe_fused_mul_sum.md) | occupied | 113/18 | 13 | YY-L | 23.9013x |
| 33 | [per_token_group_quant_int8](tasks/batch-3/33-per_token_group_quant_int8.md) | occupied | 283/36 | 15 | c2flow | 11.0389x |
| 34 | [per_token_quant_int8](tasks/batch-3/34-per_token_quant_int8.md) | occupied | 155/23 | 14 | c2flow | 9.2354x |
| 35 | [rotary_embedding](tasks/batch-3/35-rotary_embedding.md) | occupied | 125/24 | 15 | c2flow | 12.7588x |
| 36 | [selective_state_update](tasks/batch-3/36-selective_state_update.md) | occupied | 224/25 | 3 | EvokeAgent | 10.6046x |
| 37 | [sgemm_lora_a](tasks/batch-3/37-sgemm_lora_a.md) | occupied | 137/25 | 7 | EvokeAgent | 42.1385x |
| 38 | [sigmoid_gate_topk_renorm](tasks/batch-3/38-sigmoid_gate_topk_renorm.md) | occupied | 158/29 | 7 | sitraliqui | 9.0482x |
| 39 | [silu_and_mul_masked](tasks/batch-3/39-silu_and_mul_masked.md) | occupied | 276/30 | 13 | EvokeAgent | 26.1621x |
| 40 | [softcap_inplace_logits](tasks/batch-3/40-softcap_inplace_logits.md) | occupied | 264/29 | 17 | c2flow | 2.2593x |
| 41 | [state_passing](tasks/batch-3/41-state_passing.md) | occupied | 203/30 | 9 | EvokeAgent | 8.1376x |

## 第 4 批

| 题号 | 算子 | 状态 | 提交/队伍 | 达标队伍 | 当前第一 | 平均加速比 |
| ---: | --- | --- | ---: | ---: | --- | ---: |
| 42 | [act_and_mul](tasks/batch-4/42-act_and_mul.md) | occupied | 311/36 | 26 | gonzhanshishenmeganjue | 431.4843x |
| 43 | [causal_conv1d_update](tasks/batch-4/43-causal_conv1d_update.md) | occupied | 264/32 | 7 | KernelX | 8.6696x |
| 44 | [chain_speculative_sampling](tasks/batch-4/44-chain_speculative_sampling.md) | occupied | 220/35 | 1 | c2flow | 77.9541x |
| 45 | [chunk_scaled_dot_kkt](tasks/batch-4/45-chunk_scaled_dot_kkt.md) | occupied | 354/40 | 6 | EvokeAgent | 23.4187x |
| 46 | [chunked_embedding_lora_a](tasks/batch-4/46-chunked_embedding_lora_a.md) | occupied | 324/32 | 7 | EvokeAgent | 28.9367x |
| 47 | [chunked_sgmv_expand](tasks/batch-4/47-chunked_sgmv_expand.md) | occupied | 183/27 | 4 | EvokeAgent | 55.2154x |
| 48 | [chunked_sgmv_shrink](tasks/batch-4/48-chunked_sgmv_shrink.md) | occupied | 260/32 | 5 | RSI | 32.1779x |
| 49 | [ernie45_rope_fused](tasks/batch-4/49-ernie45_rope_fused.md) | occupied | 116/23 | 9 | c2flow | 21.1873x |
| 50 | [extend_attention](tasks/batch-4/50-extend_attention.md) | occupied | 350/31 | 3 | RSI | 15.2667x |
| 51 | [fla_layernorm_gated](tasks/batch-4/51-fla_layernorm_gated.md) | occupied | 133/15 | 12 | SoulCoder | 60.7192x |
| 52 | [fused_dual_residual_rmsnorm](tasks/batch-4/52-fused_dual_residual_rmsnorm.md) | occupied | 171/31 | 3 | RSI | 6.9649x |
| 53 | [fused_gdn_gating](tasks/batch-4/53-fused_gdn_gating.md) | occupied | 115/19 | 15 | Nectar | 288.4262x |
| 54 | [fused_norm_rope_stacked](tasks/batch-4/54-fused_norm_rope_stacked.md) | occupied | 229/25 | 3 | EvokeAgent | 11.6580x |
| 55 | [hc_head](tasks/batch-4/55-hc_head.md) | occupied | 122/33 | 5 | EvokeAgent | 7.2824x |
| 56 | [l2norm](tasks/batch-4/56-l2norm.md) | occupied | 85/17 | 15 | sikadeer | 72.3068x |
| 57 | [log_scaling_tau](tasks/batch-4/57-log_scaling_tau.md) | occupied | 249/23 | 15 | c2flow | 4.1384x |
| 58 | [w8a8_block_int8_matmul](tasks/batch-4/58-w8a8_block_int8_matmul.md) | occupied | 200/24 | 6 | EvokeAgent | 658.8503x |

## 第 5 批

| 题号 | 算子 | 状态 | 提交/队伍 | 达标队伍 | 当前第一 | 平均加速比 |
| ---: | --- | --- | ---: | ---: | --- | ---: |
| 59 | [build_trtllm_mha_page_table](tasks/batch-5/59-build_trtllm_mha_page_table.md) | reviewing | 236/34 | 15 | c2flow | 39.0024x |
| 60 | [clamp_position](tasks/batch-5/60-clamp_position.md) | reviewing | 272/42 | 12 | EvokeAgent | 4.0556x |
| 61 | [compute_src2dst](tasks/batch-5/61-compute_src2dst.md) | reviewing | 165/29 | 9 | EvokeAgent | 3.1958x |
| 62 | [concat_mla_k](tasks/batch-5/62-concat_mla_k.md) | reviewing | 339/39 | 13 | c2flow | 2.4417x |
| 63 | [create_flashinfer_kv_indices](tasks/batch-5/63-create_flashinfer_kv_indices.md) | reviewing | 282/29 | 19 | c2flow | 416.4744x |
| 64 | [deepep_permute](tasks/batch-5/64-deepep_permute.md) | reviewing | 358/27 | 17 | EvokeAgent | 34.3461x |
| 65 | [deepep_post_reorder](tasks/batch-5/65-deepep_post_reorder.md) | reviewing | 447/49 | 16 | EvokeAgent | 78.7369x |
| 66 | [dsv3_fused_a_gemm](tasks/batch-5/66-dsv3_fused_a_gemm.md) | reviewing | 132/26 | 13 | c2flow | 5.7273x |
| 67 | [fill_padded_rows](tasks/batch-5/67-fill_padded_rows.md) | reviewing | 231/31 | 16 | 金狐狸 | 12.4635x |
| 68 | [fused_eh_norm](tasks/batch-5/68-fused_eh_norm.md) | reviewing | 207/28 | 13 | c2flow | 18.2831x |
| 69 | [fused_moe_dispatch_index](tasks/batch-5/69-fused_moe_dispatch_index.md) | reviewing | 187/28 | 6 | EvokeAgent | 85.0167x |
| 70 | [gate_topk](tasks/batch-5/70-gate_topk.md) | reviewing | 122/23 | 6 | 金狐狸 | 18.8800x |
| 71 | [gelu_tanh_and_mul](tasks/batch-5/71-gelu_tanh_and_mul.md) | reviewing | 245/28 | 17 | EvokeAgent | 7.9675x |
| 72 | [group_norm_silu](tasks/batch-5/72-group_norm_silu.md) | reviewing | 360/43 | 9 | c2flow | 4.7808x |
| 73 | [residual_gate_add](tasks/batch-5/73-residual_gate_add.md) | reviewing | 153/17 | 13 | Nectar | 6.4178x |
| 74 | [seqlens_expand](tasks/batch-5/74-seqlens_expand.md) | reviewing | 189/20 | 12 | EvokeAgent | 28.6603x |
| 75 | [sigmoid_gate_mul](tasks/batch-5/75-sigmoid_gate_mul.md) | reviewing | 183/28 | 16 | c2flow | 5.9694x |

## 第 6 批

| 题号 | 算子 | 状态 | 提交/队伍 | 达标队伍 | 当前第一 | 平均加速比 |
| ---: | --- | --- | ---: | ---: | --- | ---: |
| 76 | [add3](tasks/batch-6/76-add3.md) | competing | 178/29 | 23 | Sweetdeath | 1.6197x |
| 77 | [compute_position](tasks/batch-6/77-compute_position.md) | competing | 137/23 | 11 | HAiWORLD | 1348.5673x |
| 78 | [concat_and_cast_mha_k](tasks/batch-6/78-concat_and_cast_mha_k.md) | competing | 138/22 | 14 | c2flow | 1909.9204x |
| 79 | [create_flashmla_kv_indices](tasks/batch-6/79-create_flashmla_kv_indices.md) | competing | 148/23 | 18 | c2flow | 343.5152x |
| 80 | [fixup_zero_kv](tasks/batch-6/80-fixup_zero_kv.md) | competing | 95/21 | 12 | OpeGoodn | 485.0419x |
| 81 | [fused_gate_sigmoid_mul_add](tasks/batch-6/81-fused_gate_sigmoid_mul_add.md) | competing | 133/19 | 16 | sitraliqui | 5.1453x |
| 82 | [hash_topk](tasks/batch-6/82-hash_topk.md) | competing | 56/15 | 8 | c2flow | 9.1347x |
| 83 | [indexed_scale_shift](tasks/batch-6/83-indexed_scale_shift.md) | competing | 71/16 | 4 | c2flow | 143.3986x |
| 84 | [moe_align_block_size](tasks/batch-6/84-moe_align_block_size.md) | competing | 90/15 | 6 | c2flow | 871.1429x |
| 85 | [moe_align_single_token](tasks/batch-6/85-moe_align_single_token.md) | competing | 65/14 | 8 | 金狐狸 | 32.8863x |
| 86 | [moe_topk_sum](tasks/batch-6/86-moe_topk_sum.md) | competing | 33/14 | 11 | cgzhou | 3.7285x |
| 87 | [pack_topk_ids](tasks/batch-6/87-pack_topk_ids.md) | competing | 63/17 | 13 | 金狐狸 | 3.3744x |
| 88 | [post_reorder_cutlass](tasks/batch-6/88-post_reorder_cutlass.md) | competing | 30/7 | 3 | c2flow | 28.3469x |
| 89 | [relu2](tasks/batch-6/89-relu2.md) | competing | 4/3 | 3 | c2flow | 2.7279x |
| 90 | [sigmoid_gate_mul_broadcast](tasks/batch-6/90-sigmoid_gate_mul_broadcast.md) | competing | 11/5 | 4 | c2flow | 2.9891x |
| 91 | [tiny_k_gemm](tasks/batch-6/91-tiny_k_gemm.md) | competing | 28/9 | 3 | cgzhou | 1.9215x |
| 92 | [unpad_draft_extend_output](tasks/batch-6/92-unpad_draft_extend_output.md) | competing | 7/5 | 3 | c2flow | 220.8796x |

## 第 7 批

| 题号 | 算子 | 状态 | 提交/队伍 | 达标队伍 | 当前第一 | 平均加速比 |
| ---: | --- | --- | ---: | ---: | --- | ---: |
