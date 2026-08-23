# 第二届 FlagOS 算子赛题索引

> 来源：[比赛页](https://flagos.io/race-detail-season2?id=782kzq4m)；同步时间：`2026-08-24T02:17:26+08:00`。
> 状态和榜单会变化，运行 `python tools/sync_flagos_season2_docs.py` 更新。

## 第 1 批

| 题号 | 算子 | 状态 | 提交/队伍 | 达标队伍 | 当前第一 | 平均加速比 |
| ---: | --- | --- | ---: | ---: | --- | ---: |
| 1 | [causal_conv1d_fn](tasks/batch-1/01-causal_conv1d_fn.md) | reviewing | 114/21 | 1 | EvokeAgent | 13.6331x |
| 2 | [chunk_local_cumsum_scalar](tasks/batch-1/02-chunk_local_cumsum_scalar.md) | reviewing | 111/18 | 6 | xuanzhengdu | 2.3866x |
| 3 | [fused_moe_gemm](tasks/batch-1/03-fused_moe_gemm.md) | reviewing | 104/21 | 1 | c2flow | 13.2849x |
| 4 | [merge_state](tasks/batch-1/04-merge_state.md) | reviewing | 41/13 | 8 | xuanzhengdu | 7.3011x |
| 5 | [mrope_fused](tasks/batch-1/05-mrope_fused.md) | reviewing | 48/13 | 2 | c2flow | 24.9050x |
| 6 | [per_group_transpose](tasks/batch-1/06-per_group_transpose.md) | reviewing | 33/14 | 8 | c2flow | 631.8800x |
| 7 | [silu_and_mul](tasks/batch-1/07-silu_and_mul.md) | reviewing | 75/18 | 12 | EvokeAgent | 4.8633x |

## 第 2 批

| 题号 | 算子 | 状态 | 提交/队伍 | 达标队伍 | 当前第一 | 平均加速比 |
| ---: | --- | --- | ---: | ---: | --- | ---: |
| 8 | [apply_token_bitmask](tasks/batch-2/08-apply_token_bitmask.md) | competing | 104/17 | 11 | Skyspeed | 11.9299x |
| 9 | [bmm_chunk](tasks/batch-2/09-bmm_chunk.md) | competing | 90/14 | 5 | hbmu9306 | 2.8196x |
| 10 | [chunk_cumsum](tasks/batch-2/10-chunk_cumsum.md) | competing | 218/20 | 3 | c2flow | 5.0001x |
| 11 | [chunk_local_cumsum_vector](tasks/batch-2/11-chunk_local_cumsum_vector.md) | competing | 120/16 | 3 | EvokeAgent | 2.1208x |
| 12 | [chunk_state](tasks/batch-2/12-chunk_state.md) | competing | 103/14 | 9 | c2flow | 12.4844x |
| 13 | [chunk_state_varlen](tasks/batch-2/13-chunk_state_varlen.md) | competing | 65/11 | 2 | MakeYUNAGreatAgain | 477.7849x |
| 14 | [context_attention](tasks/batch-2/14-context_attention.md) | competing | 97/19 | 1 | EvokeAgent | 3.7924x |
| 15 | [decode_attention](tasks/batch-2/15-decode_attention.md) | competing | 41/10 | 1 | c2flow | 78.2958x |
| 16 | [decode_grouped_attention](tasks/batch-2/16-decode_grouped_attention.md) | competing | 41/9 | 1 | c2flow | 76.4536x |
| 17 | [embedding_lora_a](tasks/batch-2/17-embedding_lora_a.md) | competing | 42/6 | 2 | EvokeAgent | 23.0558x |
| 18 | [fused_recurrent_gdn](tasks/batch-2/18-fused_recurrent_gdn.md) | pending_challenge | 49/13 | 0 | - | - |
| 19 | [fused_rmsnorm](tasks/batch-2/19-fused_rmsnorm.md) | competing | 23/7 | 6 | MakeYUNAGreatAgain | 5.5523x |
| 20 | [mamba_layernorm_gated](tasks/batch-2/20-mamba_layernorm_gated.md) | competing | 31/7 | 4 | autokernel | 6.1401x |
| 21 | [moe_sum_reduce](tasks/batch-2/21-moe_sum_reduce.md) | competing | 49/10 | 6 | EvokeAgent | 3.3856x |
| 22 | [qkv_lora_b](tasks/batch-2/22-qkv_lora_b.md) | competing | 29/5 | 1 | c2flow | 181.7155x |
| 23 | [sgemm_lora_b](tasks/batch-2/23-sgemm_lora_b.md) | competing | 62/9 | 2 | EvokeAgent | 28.6790x |
| 24 | [softcap_out](tasks/batch-2/24-softcap_out.md) | competing | 44/12 | 8 | autokernel | 2.4291x |
