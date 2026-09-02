# 第二届 FlagOS 算子赛题索引

> 来源：[比赛页](https://flagos.io/race-detail-season2?id=782kzq4m)；同步时间：`2026-09-02T11:21:57+08:00`。
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
| 8 | [apply_token_bitmask](tasks/batch-2/08-apply_token_bitmask.md) | reviewing | 188/26 | 18 | hbmu9306 | 709.4368x |
| 9 | [bmm_chunk](tasks/batch-2/09-bmm_chunk.md) | reviewing | 262/25 | 13 | RSI | 4.0070x |
| 10 | [chunk_cumsum](tasks/batch-2/10-chunk_cumsum.md) | reviewing | 379/31 | 4 | c2flow | 5.5907x |
| 11 | [chunk_local_cumsum_vector](tasks/batch-2/11-chunk_local_cumsum_vector.md) | reviewing | 179/26 | 5 | EvokeAgent | 2.1942x |
| 12 | [chunk_state](tasks/batch-2/12-chunk_state.md) | reviewing | 170/26 | 12 | RSI | 17.7777x |
| 13 | [chunk_state_varlen](tasks/batch-2/13-chunk_state_varlen.md) | reviewing | 128/21 | 6 | c2flow | 707.0045x |
| 14 | [context_attention](tasks/batch-2/14-context_attention.md) | reviewing | 185/32 | 1 | EvokeAgent | 3.7924x |
| 15 | [decode_attention](tasks/batch-2/15-decode_attention.md) | reviewing | 95/18 | 3 | RSI | 103.3379x |
| 16 | [decode_grouped_attention](tasks/batch-2/16-decode_grouped_attention.md) | reviewing | 92/23 | 2 | RSI | 303.0163x |
| 17 | [embedding_lora_a](tasks/batch-2/17-embedding_lora_a.md) | reviewing | 186/22 | 8 | RSI | 25.5830x |
| 18 | [fused_recurrent_gdn](tasks/batch-2/18-fused_recurrent_gdn.md) | pending_challenge | 137/30 | 0 | - | - |
| 19 | [fused_rmsnorm](tasks/batch-2/19-fused_rmsnorm.md) | reviewing | 114/23 | 18 | torpedo | 703.1127x |
| 20 | [mamba_layernorm_gated](tasks/batch-2/20-mamba_layernorm_gated.md) | reviewing | 127/15 | 8 | RSI | 7.4590x |
| 21 | [moe_sum_reduce](tasks/batch-2/21-moe_sum_reduce.md) | reviewing | 195/22 | 11 | HAiWORLD | 3.8270x |
| 22 | [qkv_lora_b](tasks/batch-2/22-qkv_lora_b.md) | reviewing | 74/11 | 1 | c2flow | 181.7155x |
| 23 | [sgemm_lora_b](tasks/batch-2/23-sgemm_lora_b.md) | reviewing | 137/17 | 5 | RSI | 40.5614x |
| 24 | [softcap_out](tasks/batch-2/24-softcap_out.md) | reviewing | 175/20 | 14 | MakeYUNAGreatAgain | 58.5631x |

## 第 3 批

| 题号 | 算子 | 状态 | 提交/队伍 | 达标队伍 | 当前第一 | 平均加速比 |
| ---: | --- | --- | ---: | ---: | --- | ---: |
| 25 | [draft_topk1](tasks/batch-3/25-draft_topk1.md) | competing | 192/24 | 8 | c2flow | 2.2311x |
| 26 | [fused_moe_router_cudacore](tasks/batch-3/26-fused_moe_router_cudacore.md) | competing | 208/30 | 8 | c2flow | 1.9465x |
| 27 | [fused_moe_router_tensorcore](tasks/batch-3/27-fused_moe_router_tensorcore.md) | competing | 122/27 | 6 | c2flow | 1.7541x |
| 28 | [gate_up_lora_b](tasks/batch-3/28-gate_up_lora_b.md) | competing | 127/22 | 4 | RSI | 48.1316x |
| 29 | [gelu_and_mul](tasks/batch-3/29-gelu_and_mul.md) | competing | 234/29 | 16 | Nectar | 4.5662x |
| 30 | [interleaved_rope](tasks/batch-3/30-interleaved_rope.md) | competing | 171/24 | 14 | HAiWORLD | 39.0282x |
| 31 | [moe_fused_gate](tasks/batch-3/31-moe_fused_gate.md) | competing | 139/27 | 7 | RSI | 59.1920x |
| 32 | [moe_fused_mul_sum](tasks/batch-3/32-moe_fused_mul_sum.md) | competing | 88/15 | 9 | YY-L | 23.9013x |
| 33 | [per_token_group_quant_int8](tasks/batch-3/33-per_token_group_quant_int8.md) | competing | 118/20 | 10 | wwwwww | 7.7529x |
| 34 | [per_token_quant_int8](tasks/batch-3/34-per_token_quant_int8.md) | competing | 126/19 | 12 | EvokeAgent | 7.2513x |
| 35 | [rotary_embedding](tasks/batch-3/35-rotary_embedding.md) | competing | 104/22 | 13 | c2flow | 12.7588x |
| 36 | [selective_state_update](tasks/batch-3/36-selective_state_update.md) | competing | 193/23 | 3 | EvokeAgent | 9.5367x |
| 37 | [sgemm_lora_a](tasks/batch-3/37-sgemm_lora_a.md) | competing | 103/22 | 6 | EvokeAgent | 42.1385x |
| 38 | [sigmoid_gate_topk_renorm](tasks/batch-3/38-sigmoid_gate_topk_renorm.md) | competing | 143/26 | 5 | sitraliqui | 9.0482x |
| 39 | [silu_and_mul_masked](tasks/batch-3/39-silu_and_mul_masked.md) | competing | 132/21 | 11 | EvokeAgent | 24.3040x |
| 40 | [softcap_inplace_logits](tasks/batch-3/40-softcap_inplace_logits.md) | competing | 174/24 | 15 | AutokenLab | 2.1556x |
| 41 | [state_passing](tasks/batch-3/41-state_passing.md) | competing | 175/27 | 6 | wwwwww | 7.8095x |
