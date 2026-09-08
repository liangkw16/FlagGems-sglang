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
| 42 | act_and_mul | valid | 8/8(e6,3.25835x,排名10) | e6 3.25835x | no | e7 stride融合已验证3.149675未超E6(计分布局以连续为主);stride轴关闭,保留E6,无新结构证据不开轴 | 2026-09-08 | [act_and_mul](act_and_mul.md) |
| 43 | causal_conv1d_update | valid | 8/8(e13,6.545875x最佳);e14 6.5204375x非最佳;e16候选就绪 | e13 6.545875x | no | E16三vendor字节瘦身+视图返回,回执/ZIP/manifest全绿,进入平台preflight | 2026-09-08 | [causal_conv1d_update](causal_conv1d_update.md) |
| 44 | chain_speculative_sampling | invalid | s0八芯predicts失败;本轮修复OOB但half精确采样仍失败,未重投 | s0 -x | no | 诊断已复现predicts[13]失配;release因expected_failure被拒,目标scan/舍入仍未修复,禁止带缺口提交 | 2026-09-08 | [chain_speculative_sampling](chain_speculative_sampling.md) |
| 45 | chunk_scaled_dot_kkt | invalid_threshold | 8/8正确(e15),昆仑0.063x<0.1;平均7.000125不计有效排名 | - -x | no | 新row/head epilogue代理正确但未提速;候选暂不晋级,昆仑仍需>=0.1目标证据 | 2026-09-08 | [chunk_scaled_dot_kkt](chunk_scaled_dot_kkt.md) |
| 46 | chunked_embedding_lora_a | valid | 8/8(e3,14.1051875x) | e3 14.1051875x | no | 华为token tile8及rank>128修复代理通过/ZIP验签;等待华为正确性和长短段性能复验 | 2026-09-08 | [chunked_embedding_lora_a](chunked_embedding_lora_a.md) |
| 47 | chunked_sgmv_expand | valid | E11/11031八芯valid,21.6584375x;历史E5 best25.0048125x | e5 25.0048125x | no | E11目标正确性通过但未晋级,保留E5守榜;不重投同字节,需新增目标性能证据再开轴 | 2026-09-08 | [chunked_sgmv_expand](chunked_sgmv_expand.md) |
| 48 | chunked_sgmv_shrink | valid | e8(11170)=7/8已通过,昆仑待回调;最佳仍e7(10808,4.7489375x) | e7 4.7489375x | no | 仅查11170昆仑回调,不得重复提交;总分与Top1未确定 | 2026-09-08 | [chunked_sgmv_shrink](chunked_sgmv_shrink.md) |
| 49 | ernie45_rope_fused | valid | 8/8(e3,8.58253125x) | e3 8.58253125x | no | head16候选代理正确/ZIP验签;需昆仑live buffer和完整wrapper收益后晋级 | 2026-09-08 | [ernie45_rope_fused](ernie45_rope_fused.md) |
| 50 | extend_attention | candidate-wip | 6/8(e4,昆仑conclusive封轴;华为数值不可修;燧原0.013x<门槛) | - -x | no | 逐阶段诊断已实现但插桩有观察效应;需要目标原失败重放,不再将两种失败形态称数值永久不可修 | 2026-09-08 | [extend_attention](extend_attention.md) |
| 51 | fla_layernorm_gated | valid | e9/11124昆仑真机correctness失败invalid;team best保持E7 5.816325x | e7 5.816325x | no | e9行块特化证伪(天数-20%/昆仑真机失败);行块轴关闭,保留E7,需c2flow结构情报 | 2026-09-08 | [fla_layernorm_gated](fla_layernorm_gated.md) |
| 52 | fused_dual_residual_rmsnorm | invalid | e6/10743六芯通过,昆仑失败,燧原pending;已无有效分可能 | - -x | no | 阶段探针和大形状代理验证已完成;昆仑4元素问题未复现,仍需目标原输入首分歧,不重投 | 2026-09-08 | [fused_dual_residual_rmsnorm](fused_dual_residual_rmsnorm.md) |
| 53 | fused_gdn_gating | valid | 8/8(e5,11044,2.954625x新team best) | e5 2.954625x | no | 距榜首仍远(288x);多行结构已兑现,剩余轴=弱芯(昆仑0.72/华为1.47/燧原1.60)新结构证据 | 2026-09-08 | [fused_gdn_gating](fused_gdn_gating.md) |
| 54 | fused_norm_rope_stacked | invalid_correctness | e3/11046 5/8,燧原PassManager/昆仑24.7%失配/华为大case三结构三连败,轴关闭 | - -x | yes | 三种结构(fused/row/three-kernel)均败于同三芯;重开需目标芯输入或他队公开PR,不再盲投 | 2026-09-08 | [fused_norm_rope_stacked](fused_norm_rope_stacked.md) |
| 55 | hc_head | invalid_correctness | 7/8(e1,10668已终态;昆仑compile_worker Aborted,归因未定) | - -x | no | 独立进程和逐核取证工具已实现;需要昆仑同worker/runtime重放,崩溃归因未定,停止盲投 | 2026-09-08 | [hc_head](hc_head.md) |
| 56 | l2norm | valid | 8/8(e3,11062,3.20691667x新team best) | e3 3.20691667x | no | 昆仑行块vendor兑现0.578→1.08;榜首72x未破译,弱芯华为1.54/燧原1.21需新证据,短行轴已尽 | 2026-09-08 | [l2norm](l2norm.md) |
| 57 | log_scaling_tau | valid | 8/8(e2,10747,2.36478125x首次有效) | e11 2.50278125x | no | PTX已证实16B访存,本轮向量化轴停止;保留E11,无新生产候选或ZIP | 2026-09-08 | [log_scaling_tau](log_scaling_tau.md) |
| 58 | w8a8_block_int8_matmul | valid | 8/8(e5,11143,250.64599167x新team best) | e5 250.64599167x | no | E6分组边界与FP32契约修复release通过;大矩阵代理回退约20%,未晋级不提交 | 2026-09-08 | [w8a8_block_int8_matmul](w8a8_block_int8_matmul.md) |

缺 CURRENT 块（未计入索引）：apply_token_bitmask.md、bmm_chunk.md、chunk_cumsum.md、chunk_local_cumsum_vector.md、chunk_state.md、chunk_state_varlen.md、context_attention.md、decode_attention.md、decode_grouped_attention.md、embedding_lora_a.md、fused_recurrent_gdn.md、fused_rmsnorm.md、mamba_layernorm_gated.md、moe_sum_reduce.md、qkv_lora_b.md、sgemm_lora_b.md、softcap_out.md
