# 第四批结构尝试闭环（2026-09-09）

榜单与终态刷新时间：2026-09-09T12:50:10.837847+08:00。本轮正式提交6次，5次八芯有效、1次无效；T47和T48刷新团队最佳。当前仍为1项Top1（T51），没有新增Top1；账号当日已用24/30次，剩余6次。

两题均保留 `cea2a0c10878b39c251a36857d97311d1ab4cd73` 的同adapter段合并实现。T47由25.36275升到25.5965（+0.92%，第3）；T48由4.7489375升到4.8135625（+1.36%，第5）。两弱芯合计分别提高31.52%和43.58%；整体均值还包含未改芯的波动。

|题目|候选|提交号|八芯结果|平均分|决策|
|---|---|---:|---|---:|---|
|T47|e15 设备路由向量GEMV|11764|invalid_correctness，7/8正确|—|关闭该轴|
|T47|e16 紧凑段tile调度|11769|valid，8/8正确|24.897|关闭该轴|
|T48|e8 FP32窄输出Split-K|11767|valid，8/8正确|4.460625|关闭该轴|
|T47|e17 重复adapter段合并|11771|valid，8/8正确|25.5965|保留：当前团队最佳|
|T48|e9 重复adapter段合并|11775|valid，8/8正确|4.8135625|保留：当前团队最佳|
|T47|e18 合并后燧原原生半精度dot|11776|valid，8/8正确|25.4674375|关闭该轴|

有效结构是按相同weight index合并多个段，减少gather、常规GEMM和scatter次数。沿用已通过弱卡评测的GEMM，不在矩阵内增加设备路由。新增重复/穿插adapter、int32索引、空段哨兵、三dtype回归；T47 release16方法、T48 release8方法通过。

T47设备路由向量方案：昆仑出现编译错误，燧原0.01低于门槛；紧凑调度和原生半精度dot虽正确，均未达到晋级门。T48窄输出Split-K虽有FP32代理收益，整题未提分。T58组并行与单launch FP32 IEEE两次screening各7方法通过，但完整调用只有基线的0.22–0.69倍、0.20–0.90倍，已归档并恢复源码，未上传。

每次提交均绑定源码commit、测试SHA、release回执和不可变ZIP；上传后远端SHA256逐一匹配。每个候选只上传、提交一次。最终源码与两题最佳release的源码、测试、runner和依赖逐文件SHA一致。代码和账本已推送至 `fork/codex/batch4-structural-20260909`；本地主分支未改。

本轮也刷新了[GitHub PR列表](https://github.com/flagos-ai/FlagGems-sglang/pulls)：59个PR元数据；相对09-08旧报告新增#57/#58/#59。检查了[#58 qkv_lora_b](https://github.com/flagos-ai/FlagGems-sglang/pull/58)的规整昆仑GEMM，其无mask读需要额外补齐K尾，未直接移植；[#59 fused_rmsnorm](https://github.com/flagos-ai/FlagGems-sglang/pull/59)是单行归约实现，未发现能直接改变本轮三题结构的代码。完整返回保存在本地structural-20260909产物目录。

以下是最新榜单时点数据，未达有效门槛的题目记为“—”。

|题|算子|我方最佳|我方排名|榜首分数|榜首队伍|
|---:|---|---:|---:|---:|---|
|42|act_and_mul|3.25835|13|431.4843|gonzhanshishenmeganjue|
|43|causal_conv1d_update|6.6005625|4|8.47775|EvokeAgent|
|44|chain_speculative_sampling|—|—|77.954125|c2flow|
|45|chunk_scaled_dot_kkt|—|—|19.7779375|EvokeAgent|
|46|chunked_embedding_lora_a|14.1051875|6|28.0405625|EvokeAgent|
|47|chunked_sgmv_expand|25.5965|3|42.9844375|c2flow|
|48|chunked_sgmv_shrink|4.8135625|5|26.944|c2flow|
|49|ernie45_rope_fused|8.9578125|8|21.18734375|c2flow|
|50|extend_attention|—|—|10.72328125|c2flow|
|51|fla_layernorm_gated|60.7192|1|60.7192|SoulCoder|
|52|fused_dual_residual_rmsnorm|—|—|5.63800833|c2flow|
|53|fused_gdn_gating|3.03205|13|288.426175|Nectar|
|54|fused_norm_rope_stacked|—|—|10.80896875|EvokeAgent|
|55|hc_head|—|—|6.49028125|c2flow|
|56|l2norm|3.20691667|10|72.30680208|sikadeer|
|57|log_scaling_tau|2.50278125|7|3.27190625|EvokeAgent|
|58|w8a8_block_int8_matmul|261.84538333|4|616.79514167|EvokeAgent|

[完整逐芯结果和榜单JSON](data/batch4-structural-results-20260909.json) · [T47账本](experiments/chunked_sgmv_expand.md) · [T48账本](experiments/chunked_sgmv_shrink.md) · [T58账本](experiments/w8a8_block_int8_matmul.md)
