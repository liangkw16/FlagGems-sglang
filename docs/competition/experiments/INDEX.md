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
| 42 | act_and_mul | valid | 8/8(e9w/12643,3.260075x新team best,排名13/25) | e9w 3.260075x | no | 批4收盘;10发额度用尽(30/30).e9w/e9x两发水位重掷燧原1.685/1.679(常规水位,未撞慢窗);e9w avg3.260075微破TB;排名13/25,上邻+0.0498.燧原结构轴全类封死 | 2026-09-10 | [act_and_mul](act_and_mul.md) |
| 43 | causal_conv1d_update | valid | e19r/11237八芯valid,6.560375x新team best(重掷第1次破线) | e21 6.6005625x | no | E19重掷第1次即破线(6.5604);同字节还剩≤1次,榜首7.90需结构面,守TB为主 | 2026-09-08 | [causal_conv1d_update](causal_conv1d_update.md) |
| 44 | chain_speculative_sampling | invalid | s0八芯predicts失败;本轮修复OOB但half精确采样仍失败,未重投 | s0 -x | no | 诊断已复现predicts[13]失配;release因expected_failure被拒,目标scan/舍入仍未修复,禁止带缺口提交 | 2026-09-08 | [chain_speculative_sampling](chain_speculative_sampling.md) |
| 45 | chunk_scaled_dot_kkt | invalid_threshold | 8/8正确(e15),昆仑0.063x<0.1;平均7.000125不计有效排名 | - -x | no | 新row/head epilogue代理正确但未提速;候选暂不晋级,昆仑仍需>=0.1目标证据 | 2026-09-08 | [chunk_scaled_dot_kkt](chunk_scaled_dot_kkt.md) |
| 46 | chunked_embedding_lora_a | valid | 8/8(e3,14.1051875x);e7/11229 7/8华为aclnnCat内部错误 | e3 14.1051875x | no | E8终态7/8(燧原PassManager,昆仑0.64兑现/华为int32未过门);去燧原包avg~13.9<TB,轴封存 | 2026-09-10 | [chunked_embedding_lora_a](chunked_embedding_lora_a.md) |
| 47 | chunked_sgmv_expand | valid | E11/11031八芯valid,21.6584375x;历史E5 best25.0048125x | e12 25.36275x | no | E11目标正确性通过但未晋级,保留E5守榜;不重投同字节,需新增目标性能证据再开轴 | 2026-09-08 | [chunked_sgmv_expand](chunked_sgmv_expand.md) |
| 48 | chunked_sgmv_shrink | valid | 8/8(e6,4.7198125x);e4=7/8(燧原评测机忙超时,同字节vendor) | e12r 5.289875x | no | e12收盘5.207(rank~5);燧原0.828(adapter分组+55%但未到门);结构面未定位,水位重掷可选 | 2026-09-10 | [chunked_sgmv_shrink](chunked_sgmv_shrink.md) |
| 49 | ernie45_rope_fused | valid | e4/11245八芯valid,8.71115625x新team best | e7r2 10.35265625x | no | 批4收盘;额度用尽.e9w(12648)9.68269燧原0.560,水位未抬;TB保持e7r2 10.35266;排名8/9,上邻+0.227 | 2026-09-10 | [ernie45_rope_fused](ernie45_rope_fused.md) |
| 50 | extend_attention | candidate-wip | 6/8(e4,昆仑conclusive封轴;华为数值不可修;燧原0.013x<门槛) | - -x | no | 逐阶段诊断已实现但插桩有观察效应;需要目标原失败重放,不再将两种失败形态称数值永久不可修 | 2026-09-08 | [extend_attention](extend_attention.md) |
| 51 | fla_layernorm_gated | valid | e10/11204八芯valid,60.7192x新team best,实时rank1 | e10 60.7192x | no | 已登顶(燧原442.8极端水位+华为子块+10%同发兑现);守榜为主,华为子块结构知识可迁移 | 2026-09-08 | [fla_layernorm_gated](fla_layernorm_gated.md) |
| 52 | fused_dual_residual_rmsnorm | invalid | e6/10743六芯通过,昆仑失败,燧原pending;已无有效分可能 | - -x | no | 阶段探针和大形状代理验证已完成;昆仑4元素问题未复现,仍需目标原输入首分歧,不重投 | 2026-09-08 | [fused_dual_residual_rmsnorm](fused_dual_residual_rmsnorm.md) |
| 53 | fused_gdn_gating | valid | 8/8(e9x/12658,3.044825x新team best,排名14/15) | e9x 3.044825x | no | 批4收盘;额度用尽.e9w 2.95675/e9x 3.044825(末发破TB);燧原1.593/1.601常规水位未撞慢窗;排名14/15,上邻+0.2658 | 2026-09-10 | [fused_gdn_gating](fused_gdn_gating.md) |
| 54 | fused_norm_rope_stacked | invalid_correctness | e3/11046 5/8,燧原PassManager/昆仑24.7%失配/华为大case三结构三连败,轴关闭 | - -x | yes | 三种结构(fused/row/three-kernel)均败于同三芯;重开需目标芯输入或他队公开PR,不再盲投 | 2026-09-08 | [fused_norm_rope_stacked](fused_norm_rope_stacked.md) |
| 55 | hc_head | invalid_correctness | 7/8(e1,10668已终态;昆仑compile_worker Aborted,归因未定) | - -x | no | 独立进程和逐核取证工具已实现;需要昆仑同worker/runtime重放,崩溃归因未定,停止盲投 | 2026-09-08 | [hc_head](hc_head.md) |
| 56 | l2norm | valid | 8/8(e3,11062,3.20691667x新team best) | e3 3.20691667x | no | 批4收盘;额度用尽.e9w 3.16114/e9x(12659)3.17715;燧原1.2015/1.20675两次同水位,未撞慢窗;TB保持e3 3.20692;排名11/14,上邻仅+0.035 | 2026-09-10 | [l2norm](l2norm.md) |
| 57 | log_scaling_tau | valid | 8/8(e2,10747,2.36478125x首次有效) | e11 2.50278125x | no | 批4收盘;额度用尽.e9w(12649)2.47272燧原0.546;TB保持e11 2.50278;排名9/15,上邻仅+0.0396 | 2026-09-10 | [log_scaling_tau](log_scaling_tau.md) |
| 58 | w8a8_block_int8_matmul | valid | e6r/11210八芯valid,258.04890833x team best(排名3);e7/11228 valid 253.07(B已修复) | e9r2 266.20655x | no | e7字节(e6r组级+amd逐块B)为最优组合,均值差=华为窗口三连下行;水位回常态时以e7字节重掷(新ZIP身份,≤2次) | 2026-09-08 | [w8a8_block_int8_matmul](w8a8_block_int8_matmul.md) |
| 59 | build_trtllm_mha_page_table | valid | completed(14705,e9,8/8,23.406x;行打包华为4.29证伪;TB e4r 24.1284x) | e4r 24.1284375x | no | e9 行打包华为4.29(-51%)证伪,华为轴全关待新证据;TB e4r 24.128守 | 2026-09-14 | [build_trtllm_mha_page_table](build_trtllm_mha_page_table.md) |
| 60 | clamp_position | invalid_correctness | completed(13821,e8,7/8;燧原轴八轮终封) | - | no | T60 燧原轴终封;重启需 torch-gcu view 缺陷外部证据或逐元素 torch 拆装合规确认 | 2026-09-13 | [clamp_position](clamp_position.md) |
| 61 | compute_src2dst | valid | completed(14519,e9,8/8,2.099675x 新TB) | e9 2.099675x | no | e9 新TB 2.0997（e7字节恢复+zeros→empty，燧原4.46）；残余轴=muxi 2.5x 无证据；守榜 | 2026-09-14 | [compute_src2dst](compute_src2dst.md) |
| 62 | concat_mla_k | valid | completed(14686,e7,8/8,1.2201x 新TB) | e7 1.2201x | no | e7 海光BH4 2.545(+25%,差门0.055)新TB 1.2201;BH轴停;剩余四芯缺口按v3队列 | 2026-09-13 | [concat_mla_k](concat_mla_k.md) |
| 63 | create_flashinfer_kv_indices | valid | completed(14590,e10,8/8,167.39x;TB e8 199.695x) | e8 199.69521875x | no | e10 flat-grid燧原回17.5≈历史水位,无结构增益不追发;TB e8 199.70守;6队≥37x真因未破 | 2026-09-14 | [create_flashinfer_kv_indices](create_flashinfer_kv_indices.md) |
| 64 | deepep_permute | valid | submitted(13375,e2=C1,7/8已判;仅昆仑回调;燧原+5.5%未达1.3x门) | e1 6.7478x | no | e2（燧原 grid 封顶 24）已发射；预注册燧原 ≥1.3x（目标轴 2.53→6.68 次优，值均值 +0.371） | 2026-09-12 | [deepep_permute](deepep_permute.md) |
| 65 | deepep_post_reorder | invalid_correctness | completed(13820,e4,7/8;昆仑间歇崩溃,BLOCK=64未获裁决) | - | no | 昆仑=间歇窗口+uni_sram 双条件;新 ZIP 重掷待健康窗 | 2026-09-12 | [deepep_post_reorder](deepep_post_reorder.md) |
| 66 | dsv3_fused_a_gemm | valid | completed(13371,e1,8/8,3.09395x team best) | e1 3.09395x | no | e1 valid 3.094（+13%，A+70%/沐曦+31%）；对榜首 4.12 差 1.0，燧原+2.1 为最大单芯缺口（tiny-GEMM 向量核方向已注册） | 2026-09-12 | [dsv3_fused_a_gemm](dsv3_fused_a_gemm.md) |
| 67 | fill_padded_rows | valid | completed(13386,e3,8/8,4.23825x;三宽芯门全负,轴关闭;TB e2 4.2969x) | e2 4.2969x | no | e3 列分块轴关闭(唯沐曦+24%);TB e2 4.2969 守榜(#8,榜首 8.32);宽 shape 结构待新证据 | 2026-09-12 | [fill_padded_rows](fill_padded_rows.md) |
| 68 | fused_eh_norm | valid | completed(14589,e6,8/8,7.042x;TB e4 7.089x) | e4 7.08945833x | no | e6 split-row证伪(燧原0.87,饥饿假说假);TB e4 7.089守;燧原缺因未破,主攻降级 | 2026-09-14 | [fused_eh_norm](fused_eh_norm.md) |
| 69 | fused_moe_dispatch_index | invalid_correctness | completed(13809,e7,7/8;昆仑七连崩,轴终封仅剩工单) | - | no | T69 昆仑轴终封;仅剩平台工单;七芯部分和 357.5 永久待命 | 2026-09-12 | [fused_moe_dispatch_index](fused_moe_dispatch_index.md) |
| 70 | gate_topk | invalid_correctness | completed(14556,e3,7/8;昆仑编译过但3630s挂死@0%util,转工单) | - | no | e1 终态 5/8：三失败芯均 exec 0ms 崩溃族（12:20 同窗），kunlun vendor 被选中但未执行；下一步=新 ZIP 真实改动重评（燧原/华为可补迭代选择 vendor）或用户授权的同字节重掷 | 2026-09-12 | [gate_topk](gate_topk.md) |
| 71 | gelu_tanh_and_mul | valid | completed(14573,e4,8/8,2.7109x 新TB) | e4 2.7109x | no | e4 no-loop昆仑0.304复制链失败(门0.5未过,链收口);TB 2.711守;昆仑真因/燧原8192待证 | 2026-09-14 | [gelu_tanh_and_mul](gelu_tanh_and_mul.md) |
| 72 | group_norm_silu | valid | completed(14726,e6,8/8,2.6627x 新TB;华为移植无增益) | e6 2.66266667x | no | e6 padding修复+华为移植无增益(轴停)新TB 2.663;燧原/华为缺因未破,差榜首1.28 | 2026-09-14 | [group_norm_silu](group_norm_silu.md) |
| 73 | residual_gate_add | valid | completed(14379,e4,8/8,4.027x;TB e3 4.0308x) | e3 4.03084375x | no | e4 fusion-off 无回归（契约修复零代价）；TB e3 守；榜首金狐狸 4.388 差 0.357 需多芯叠加 | 2026-09-13 | [residual_gate_add](residual_gate_add.md) |
| 74 | seqlens_expand | valid | completed(14535,e3,8/8,19.3101x 新TB) | e3 19.3101x | no | e3 前缀入kernel新TB 19.31（+81%,launch假说兑现）;残余轴=flat-grid并行度;差榜首7.05 | 2026-09-14 | [seqlens_expand](seqlens_expand.md) |
| 75 | sigmoid_gate_mul | valid | completed(14598,e4,8/8,2.739325x 新TB) | e4 2.739325x | no | e5 i32无增益(2.149≈2.156),i64税假设关;TB e4 2.739守;燧原缺因待他队情报 | 2026-09-14 | [sigmoid_gate_mul](sigmoid_gate_mul.md) |

缺 CURRENT 块（未计入索引）：apply_token_bitmask.md、bmm_chunk.md、chunk_cumsum.md、chunk_local_cumsum_vector.md、chunk_state.md、chunk_state_varlen.md、context_attention.md、decode_attention.md、decode_grouped_attention.md、embedding_lora_a.md、fused_recurrent_gdn.md、fused_rmsnorm.md、mamba_layernorm_gated.md、moe_sum_reduce.md、qkv_lora_b.md、sgemm_lora_b.md、softcap_out.md
