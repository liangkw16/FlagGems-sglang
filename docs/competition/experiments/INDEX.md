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
| 43 | causal_conv1d_update | valid | 8/8(e13,6.545875x,排名3;昆仑首次0.407x) | e13 6.545875x | no | 保持有效解;后续只按全芯均值收益排序,勿重试旧昆仑gather/GEMM轴 | 2026-09-07 | [causal_conv1d_update](causal_conv1d_update.md) |
| 44 | chain_speculative_sampling | invalid | s0八芯predicts失败;本轮修复OOB但half精确采样仍失败,未重投 | s0 -x | no | 对齐各后端reference扫描/舍入;保留已知half失败,禁止带缺口提交 | 2026-09-07 | [chain_speculative_sampling](chain_speculative_sampling.md) |
| 45 | chunk_scaled_dot_kkt | invalid_threshold | 8/8正确(e15),昆仑0.063x<0.1;平均7.000125不计有效排名 | - -x | no | constexpr除数平台无收益;停止该轴;保留契约修复,需新的epilogue结构证据 | 2026-09-07 | [chunk_scaled_dot_kkt](chunk_scaled_dot_kkt.md) |
| 46 | chunked_embedding_lora_a | valid | 8/8(e3,14.1051875x) | e3 14.1051875x | no | e6天数预路由水位带内无法归因,轴关闭;维持E3守榜(14.105x);后续按逐芯晋级纪律择机 | 2026-09-06 | [chunked_embedding_lora_a](chunked_embedding_lora_a.md) |
| 47 | chunked_sgmv_expand | valid | 8/8(e5,25.0048125x,榜首) | e5 25.0048125x | no | 守榜首(25.00x vs 前榜首23.33x);燧原0.178x/昆仑3.72x为余量轴;stale意图经用户授权归档重提(移至archived/) | 2026-09-04 | [chunked_sgmv_expand](chunked_sgmv_expand.md) |
| 48 | chunked_sgmv_shrink | valid | 8/8(e6,4.7198125x);e4=7/8(燧原评测机忙超时,同字节vendor) | e6 4.7198125x | no | e4七芯+1.5%未过15%门,保留e6;自适应BLOCK_S轴关闭;冲分回到预注册vendor轴(燧原dot模板/天数dtype/昆仑BLOCK) | 2026-09-07 | [chunked_sgmv_shrink](chunked_sgmv_shrink.md) |
| 49 | ernie45_rope_fused | valid | 8/8(e3,8.58253125x) | e3 8.58253125x | no | e3昆仑0.664x击穿uni_sram墙,8/8 VALID(第9个,is_team_best);榜单:达标4队(含我队),榜首c2flow 15.6064x(差45%);可选轴:昆仑BLOCK_HEADS与A芯21.1x | 2026-09-07 | [ernie45_rope_fused](ernie45_rope_fused.md) |
| 50 | extend_attention | candidate-wip | 6/8(e4,昆仑conclusive封轴;华为数值不可修;燧原0.013x<门槛) | - -x | no | 华为深层数值问题(online+two-pass均败);5/8已是208发1队过线题的好成绩;冲分优先 | 2026-09-05 | [extend_attention](extend_attention.md) |
| 51 | fla_layernorm_gated | valid | 8/8(e1,5.390025x) | e1 5.390025x | no | e3华为constexpr-D仅+3.7%证伪关闭;昆仑0.97已修;剩沐曦rsqrt;燧原病态在评 | 2026-09-06 | [fla_layernorm_gated](fla_layernorm_gated.md) |
| 52 | fused_dual_residual_rmsnorm | invalid | 6/8(e5,五种数学形态同指纹边界失配;已分诊) | - -x | yes | 封存6/8;仅sqrt_rn+div_rn或torch归约树复刻等全新结构证据可重开 | 2026-09-06 | [fused_dual_residual_rmsnorm](fused_dual_residual_rmsnorm.md) |
| 53 | fused_gdn_gating | valid | 8/8(e4,2.1267x) | e4 2.1267x | no | e4沐曦2.7402x(+104%,过门,is_team_best);榜单:达标10队,榜首EvokeAgent 4.3306x(差51%);后续轴:generic二代[ROWS_TILE,H] | 2026-09-07 | [fused_gdn_gating](fused_gdn_gating.md) |
| 54 | fused_norm_rope_stacked | pending | 5/8(e2) | e1 0x | no | 当日额度耗尽;e2=5/8(行式vendor未修复三芯);代理fuzz仅覆盖已测输入;固定源码复现并定位首个分歧 | 2026-09-06 | [fused_norm_rope_stacked](fused_norm_rope_stacked.md) |
| 55 | hc_head | invalid_correctness | 7/8(e1,昆仑=评测器崩溃族第2次;七芯总分+40%) | - 0x | no | 昆仑对该题reference确定性崩溃(s0/e1同指纹,同日昆仑评过T48/T49/T53/T56);等平台窗口同字节重试;燧原0.30x仍弱 | 2026-09-07 | [hc_head](hc_head.md) |
| 56 | l2norm | valid | 8/8(e1,3.10772917x) | e1 3.10772917x | no | e1修复版8/8复验通过(3.1077x,s0同水位,is_team_best);榜单:达标11队,榜首Warmhearted 4.2591x(差27%);后续可试短行num_warps=1/2与多行tile | 2026-09-07 | [l2norm](l2norm.md) |
| 57 | log_scaling_tau | pending | e1提交10704;7/8通过,燧原0.54675x已修复;昆仑waiting_callback | - -x | no | 仅查10704昆仑回调,不得重复上传;最终有效性待八芯终态 | 2026-09-07 | [log_scaling_tau](log_scaling_tau.md) |
| 58 | w8a8_block_int8_matmul | valid | 8/8(e1,118.15512x) | e1 118.15512x | no | 冲分轴:燧原4.48x/B7.68x;榜首209.3x差43%;skill沉淀昆仑向量整除崩溃教训 | 2026-09-06 | [w8a8_block_int8_matmul](w8a8_block_int8_matmul.md) |

缺 CURRENT 块（未计入索引）：apply_token_bitmask.md、bmm_chunk.md、chunk_cumsum.md、chunk_local_cumsum_vector.md、chunk_state.md、chunk_state_varlen.md、context_attention.md、decode_attention.md、decode_grouped_attention.md、embedding_lora_a.md、fused_recurrent_gdn.md、fused_rmsnorm.md、mamba_layernorm_gated.md、moe_sum_reduce.md、qkv_lora_b.md、sgemm_lora_b.md、softcap_out.md
