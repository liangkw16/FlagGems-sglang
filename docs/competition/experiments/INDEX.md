# 实验状态索引（GENERATED）

> 由 `tools/gen_experiment_index.py` 从各账本顶部 ` ```current ` 块生成，
> 不要手改本文件；状态更新只改账本 CURRENT 块，然后重跑脚本。

| Task | 算子 | 有效性 | 平台 | 团队最佳 | 封存 | 下一步 | 更新 | 账本 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 25 | draft_topk1 | invalid | 6/8 | e6c | yes | 需外部证据(他人通过样例/平台澄清)才可重启 | 2026-08-31 | [draft_topk1](draft_topk1.md) |
| 26 | fused_moe_router_cudacore | invalid | 6/8;本轮契约修复9/9 NVIDIA代理通过,未提交 | e5-e8(等价) | yes | padding remask修复9/9 NVIDIA代理通过，未提交；华为case7与昆仑旧风险仍待独立目标证据 | 2026-09-16 | [fused_moe_router_cudacore](fused_moe_router_cudacore.md) |
| 27 | fused_moe_router_tensorcore | valid | 8/8(e9,1.039975x);本轮契约修复12/12 NVIDIA代理通过,未提交 | e9 1.039975x | yes | padding remask修复12/12 NVIDIA代理通过，未提交；保留历史e9成绩及封存状态，目标芯待验证 | 2026-09-16 | [fused_moe_router_tensorcore](fused_moe_router_tensorcore.md) |
| 28 | gate_up_lora_b | valid | 8/8(e14,14.98025x) | e14 14.98025x | yes | 采样两连 TB(e13 14.4435/e14 14.98025);封存,明日 1-2 发守榜采样 | 2026-09-02 | [gate_up_lora_b](gate_up_lora_b.md) |
| 29 | gelu_and_mul | valid | 8/8(e9,2.805042x) | e9 2.805042x | yes | exact-erf 官方实现与 minimax 数值边界复核;预期收益远不足66.68%榜差,封存 | 2026-09-02 | [gelu_and_mul](gelu_and_mul.md) |
| 30 | interleaved_rope | valid | 8/8(s1,25.9236875x) | s1 25.9236875x | yes | 实时榜首37.7641;一读一写下界已达,MCP/官方实现复核无可信46.17%路径 | 2026-09-02 | [interleaved_rope](interleaved_rope.md) |
| 31 | moe_fused_gate | invalid_correctness | E9 sub8270 7/8;Kunlun 1833723ms 同指纹(第16例);本轮契约修复7/7 NVIDIA代理通过,未提交 | e7(=e6字节载体) 七芯~7.73x | yes | 契约修复7/7 NVIDIA代理通过但未提交；昆仑平台封存及原重启条件保留，不重投旧候选 | 2026-09-16 | [moe_fused_gate](moe_fused_gate.md) |
| 32 | moe_fused_mul_sum | valid | 8/8;本轮契约修复10/10 NVIDIA代理通过,未提交 | S0 4.4829x | yes | top_k=0契约修复10/10 NVIDIA代理通过，未提交；保留S0平台TB和既有性能轴结论 | 2026-09-16 | [moe_fused_mul_sum](moe_fused_mul_sum.md) |
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
| 58 | w8a8_block_int8_matmul | valid | 最新e9r3/sub12441八芯valid 261.50x;TB e9r2/sub12426 266.20655x;本轮契约修复8/8 NVIDIA代理通过,未提交 | e9r2 266.20655x | no | 非2幂量化组修复8/8 NVIDIA代理通过，未提交；保留e9r2 TB，旧重掷轴已关闭，目标芯未验证 | 2026-09-16 | [w8a8_block_int8_matmul](w8a8_block_int8_matmul.md) |
| 59 | build_trtllm_mha_page_table | valid | completed(14705,e9,8/8,23.406x;行打包华为4.29证伪;TB e4r 24.1284x) | e4r 24.1284375x | no | e9 行打包华为4.29(-51%)证伪,华为轴全关待新证据;TB e4r 24.128守 | 2026-09-14 | [build_trtllm_mha_page_table](build_trtllm_mha_page_table.md) |
| 60 | clamp_position | invalid_correctness | completed(13821,e8,7/8;燧原轴八轮终封) | - | no | 新发现torch-gcu逻辑int64物理窄化证据；先核目标输入保真/布局和实际版本，不截断契约、不重投 | 2026-09-16 | [clamp_position](clamp_position.md) |
| 61 | compute_src2dst | valid | completed(15835,e11,8/8,2.138475x新TB;Metax门未过) | e11 2.138475x | no | 保留E11团队最佳；Ascend cap/tail候选仅隔离留档，代理未提速；等待Ascend同源运行证据 | 2026-09-16 | [compute_src2dst](compute_src2dst.md) |
| 62 | concat_mla_k | valid | completed(14847,e10,8/8,1.1908x;燧原BH4反降-47%;TB e9 1.2458x) | e9 1.24575x | no | e10 燧原BH4 0.15(-47%)跨芯平移证伪;TB e9 1.246守;收官仅守榜 | 2026-09-14 | [concat_mla_k](concat_mla_k.md) |
| 63 | create_flashinfer_kv_indices | valid | completed(e12,8/8,190.674x<TB;沐曦60.8负向,metax2048轴证伪) | e8 199.69521875x | no | metax splits2048证伪(沐曦98.5→60.8大负);TB e8 199.70守;下一轴=燧原BLOCK1024/warps1、华为warps8或ps0 constexpr(单变量) | 2026-09-15 | [create_flashinfer_kv_indices](create_flashinfer_kv_indices.md) |
| 64 | deepep_permute | valid | completed(15868,e6,8/8,6.94085x;TB e5 7.17105x) | e5 7.17105x | no | e6均值回退且Enflame4.7192未达8.0门，停止空task轴；TB仍e5，不重投 | 2026-09-16 | [deepep_permute](deepep_permute.md) |
| 65 | deepep_post_reorder | valid | completed(15913,e12,7/8,invalid_correctness;昆仑collection设备错误;历史TB e10 26.917125x) | e10 26.917125x | no | E12终态7/8，昆仑收集测试失败；保留权重契约修复，需目标执行新证据后再迭代，不重发15913 | 2026-09-16 | [deepep_post_reorder](deepep_post_reorder.md) |
| 66 | dsv3_fused_a_gemm | valid | completed(15861,e2,8/8,3.09855x新TB) | e2 3.09855x | no | e2仅+0.15%，未达3.40再投入门；停止窄N split4轴，不重投 | 2026-09-16 | [dsv3_fused_a_gemm](dsv3_fused_a_gemm.md) |
| 67 | fill_padded_rows | valid | completed(e4,8/8,4.1965x;固定块轴零增益,重编译假说在本题证伪) | e2 4.2969x | no | 固定BLOCK轴关闭;下一轴=num_warps/过特化(PR扫描模式,华为1.9vs17.5缺口形态未破);TB e2 4.2969 守 | 2026-09-15 | [fill_padded_rows](fill_padded_rows.md) |
| 68 | fused_eh_norm | valid | completed(15864,e8,8/8,7.10215x新TB) | e8 7.10215x | no | 保留E8团队最佳；Ascend多行候选隔离留档，代理未提速；等待目标芯同源计时与IR | 2026-09-16 | [fused_eh_norm](fused_eh_norm.md) |
| 69 | fused_moe_dispatch_index | valid | completed(e13-pair/sub16291,8/8,50.9577x；TB仍E10,51.3375x,排名5) | e10-generic-init 51.3375x | no | E13八芯有效但50.9577未超E10，代理收益未兑现；保留E10，不重投同候选，需目标逐case/编译证据再迭代 | 2026-09-16 | [fused_moe_dispatch_index](fused_moe_dispatch_index.md) |
| 70 | gate_topk | invalid_correctness | completed(14849,e3r,7/8;昆仑exec 3633491ms挂死后判失败,与e3同指纹) | - | no | generic零值/宽索引/NaN key修复已8/8代理通过；昆仑平台失败仍未解，不建立ZIP或提交intent | 2026-09-16 | [gate_topk](gate_topk.md) |
| 71 | gelu_tanh_and_mul | valid | completed(e8/sub15474,8/8,2.712425x实际TB) | e8 2.712425x | no | 以实时TB E8为基线开发Enflame短行分组；旧width/warps/no-loop轴不重试，未验证E9 resolver不进入提交 | 2026-09-16 | [gelu_tanh_and_mul](gelu_tanh_and_mul.md) |
| 72 | group_norm_silu | valid | completed(15841,e9,8/8,2.48429167x;TB e6 2.66266667x) | e6 2.66266667x | no | 大group分块两轮control未过门，停止本轮；正确性修复8/8已入库未提交，保留e6 TB与旧uncertain | 2026-09-16 | [group_norm_silu](group_norm_silu.md) |
| 73 | residual_gate_add | valid | evaluating(e12-group4/sub16400；TB仍E6 4.0993125x) | e6 4.0993125x | no | E12已单次提交16400，等待燧原及八芯结果；其他题并行，不重发相同候选 | 2026-09-16 | [residual_gate_add](residual_gate_add.md) |
| 74 | seqlens_expand | valid | evaluating(e14-flat/sub16386,昆仑编译过但case8数值失败；TB仍E4 20.177775) | e4 20.177775x | no | E14昆仑出现新数值指纹，暂停group轴等目标IR/最小复现；剩余回调只读跟踪，推进T73/T63/T67 | 2026-09-16 | [seqlens_expand](seqlens_expand.md) |
| 75 | sigmoid_gate_mul | valid | completed(e9,8/8,2.8967x;华为warps8无效关闭) | e9 2.89671667x | no | Ascend direct候选代理1.01924x未达1.05且fp16稳定回退，不提交；TB保持e9 | 2026-09-16 | [sigmoid_gate_mul](sigmoid_gate_mul.md) |

缺 CURRENT 块（未计入索引）：apply_token_bitmask.md、bmm_chunk.md、chunk_cumsum.md、chunk_local_cumsum_vector.md、chunk_state.md、chunk_state_varlen.md、context_attention.md、decode_attention.md、decode_grouped_attention.md、embedding_lora_a.md、fused_recurrent_gdn.md、fused_rmsnorm.md、mamba_layernorm_gated.md、moe_sum_reduce.md、qkv_lora_b.md、sgemm_lora_b.md、softcap_out.md
