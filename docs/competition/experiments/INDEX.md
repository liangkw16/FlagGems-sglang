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
| 42 | act_and_mul | valid | 8/8(e6,3.25835x,排名10) | e6 3.25835x | no | 榜首已431.4843;暂停旧tile/M1轴,仅整体带宽结构有新证据再开 | 2026-09-07 | [act_and_mul](act_and_mul.md) |
| 43 | causal_conv1d_update | valid | 8/8(e14,6.5204375x非最佳);e13最佳6.545875x | e13 6.545875x | no | E15燧原PassManager编译失败收券(华为+41%保留在树);team best仍e13 6.545875;重开需燧原单变量形态拆分证据 | 2026-09-07 | [causal_conv1d_update](causal_conv1d_update.md) |
| 44 | chain_speculative_sampling | invalid | s0八芯predicts失败;本轮修复OOB但half精确采样仍失败,未重投 | s0 -x | no | 对齐各后端reference扫描/舍入;保留已知half失败,禁止带缺口提交 | 2026-09-07 | [chain_speculative_sampling](chain_speculative_sampling.md) |
| 45 | chunk_scaled_dot_kkt | invalid_threshold | 8/8正确(e15),昆仑0.063x<0.1;平均7.000125不计有效排名 | - -x | no | E16昆仑correctness失败(复合谓词mask/early-return错译定位),树已回滚e15字节;本轴关闭,重开需绕开masked谓词的新epilogue结构证据 | 2026-09-07 | [chunk_scaled_dot_kkt](chunk_scaled_dot_kkt.md) |
| 46 | chunked_embedding_lora_a | valid | 8/8(e3,14.1051875x) | e3 14.1051875x | no | e6天数预路由水位带内无法归因,轴关闭;维持E3守榜(14.105x);后续按逐芯晋级纪律择机 | 2026-09-06 | [chunked_embedding_lora_a](chunked_embedding_lora_a.md) |
| 47 | chunked_sgmv_expand | valid | 8/8(e5,25.0048125x,榜首) | e5 25.0048125x | no | 守榜首(25.00x vs 前榜首23.33x);燧原0.178x/昆仑3.72x为余量轴;stale意图经用户授权归档重提(移至archived/) | 2026-09-04 | [chunked_sgmv_expand](chunked_sgmv_expand.md) |
| 48 | chunked_sgmv_shrink | valid | 8/8(e6,4.7198125x);e4=7/8(燧原评测机忙超时,同字节vendor) | e6 4.7198125x | no | e4七芯+1.5%未过15%门,保留e6;自适应BLOCK_S轴关闭;冲分回到预注册vendor轴(燧原dot模板/天数dtype/昆仑BLOCK) | 2026-09-07 | [chunked_sgmv_shrink](chunked_sgmv_shrink.md) |
| 49 | ernie45_rope_fused | valid | 8/8(e3,8.58253125x) | e3 8.58253125x | no | e3昆仑0.664x击穿uni_sram墙,8/8 VALID(第9个,is_team_best);榜单:达标4队(含我队),榜首c2flow 15.6064x(差45%);可选轴:昆仑BLOCK_HEADS与A芯21.1x | 2026-09-07 | [ernie45_rope_fused](ernie45_rope_fused.md) |
| 50 | extend_attention | candidate-wip | 6/8(e4,昆仑conclusive封轴;华为数值不可修;燧原0.013x<门槛) | - -x | no | 华为深层数值问题(online+two-pass均败);5/8已是208发1队过线题的好成绩;冲分优先 | 2026-09-05 | [extend_attention](extend_attention.md) |
| 51 | fla_layernorm_gated | valid | 8/8(e4,5.195325x非最佳);e2最佳5.3939x | e2 5.3939x | no | 保留e2最佳5.3939;多行tile平台回退,关闭本轴;需新目标芯结构证据 | 2026-09-07 | [fla_layernorm_gated](fla_layernorm_gated.md) |
| 52 | fused_dual_residual_rmsnorm | invalid | e6/10743六芯通过,昆仑失败,燧原pending;已无有效分可能 | - -x | no | e6昆仑同4元素失配,本轴关闭;只等10743燧原回调,禁止重投;需目标归约/首分歧证据 | 2026-09-07 | [fused_dual_residual_rmsnorm](fused_dual_residual_rmsnorm.md) |
| 53 | fused_gdn_gating | valid | 8/8(e4,2.1267x,排名10) | e4 2.1267x | no | 榜首已288.426175;旧4.3306对标过期,暂停小比例调参 | 2026-09-07 | [fused_gdn_gating](fused_gdn_gating.md) |
| 54 | fused_norm_rope_stacked | invalid_correctness | 5/8(e2,10414已终态) | - -x | no | 三芯失败均已终态;仅固定源码目标复现定位首分歧后重开,非额度用尽 | 2026-09-07 | [fused_norm_rope_stacked](fused_norm_rope_stacked.md) |
| 55 | hc_head | invalid_correctness | 7/8(e1,10668已终态;昆仑compile_worker Aborted,归因未定) | - -x | no | E2(e2-afbe602)昆仑三平铺kernel vendor已过release门禁待单次平台裁决;假设=深嵌套generic在昆仑编译超时,预注册门=昆仑跑出结果且七芯维持 | 2026-09-07 | [hc_head](hc_head.md) |
| 56 | l2norm | valid | 8/8(e1,3.10772917x,排名8) | e1 3.10772917x | no | 榜首已72.30680208;守有效分,多行/短行需有整体收益证据再晋级 | 2026-09-07 | [l2norm](l2norm.md) |
| 57 | log_scaling_tau | valid | 8/8(e2,10747,2.36478125x首次有效) | e3 2.401375x | no | E3终态valid 2.401375新TB(+1.55%,华为+23.6%);距榜首2.81646875仍-14.8%,高分芯结构未破,收轴待新证据 | 2026-09-07 | [log_scaling_tau](log_scaling_tau.md) |
| 58 | w8a8_block_int8_matmul | valid | 8/8(e1,10412,122.66158333x,排名3) | e1 122.66158333x | no | 榜首562.41590833;弱两芯翻倍不足追榜,须高分芯GEMM结构收益;保持int8先castFP32契约 | 2026-09-07 | [w8a8_block_int8_matmul](w8a8_block_int8_matmul.md) |

缺 CURRENT 块（未计入索引）：apply_token_bitmask.md、bmm_chunk.md、chunk_cumsum.md、chunk_local_cumsum_vector.md、chunk_state.md、chunk_state_varlen.md、context_attention.md、decode_attention.md、decode_grouped_attention.md、embedding_lora_a.md、fused_recurrent_gdn.md、fused_rmsnorm.md、mamba_layernorm_gated.md、moe_sum_reduce.md、qkv_lora_b.md、sgemm_lora_b.md、softcap_out.md
