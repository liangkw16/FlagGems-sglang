# 41 题上游实现与八芯优化打法

> 调研日期：2026-08-31。
>
> 本文逐题核对 vLLM、SGLang、FlashInfer、FLA、Mamba 等一手源码，并将
> 上游生产实现拆成“可迁移算法”和“硬件专用调度”。这里的“最佳”表示当前
> 生产实现或最佳已知参考，不表示未经同机、同 shape benchmark 的绝对最快。
> 当前候选、成绩和封存状态以
> [实验状态索引](experiments/INDEX.md) 为准，本文不作为动态成绩真相源。

## 固定上游版本

- [SGLang `0674be7`](https://github.com/sgl-project/sglang/tree/0674be736ceb138a2f4982c6d612754d2b319807)
- [vLLM `399247c`](https://github.com/vllm-project/vllm/tree/399247cc8877f60f02f3aa859c61c3330a59bfbb)
- [FlashInfer `2cc51dc`](https://github.com/flashinfer-ai/flashinfer/tree/2cc51dcf67ee71aade7074c64e84f13b7b7b117b)
- [FLA `35dceae`](https://github.com/fla-org/flash-linear-attention/tree/35dceaee5408e69a555fec34cb215c93c375dabe)
- [Mamba `e9594ce`](https://github.com/state-spaces/mamba/tree/e9594ce1c732d97440f0332fdc43170a2294dbfa)

CUDA 的 TMA、PDL、`cp.async`、`.ca`、SM-count 和固定 warp=32 只作为调度
思想参考。比赛 generic 只能迁移 online softmax、split-KV、GQA KV 复用、
expert/token 分组、状态驻留和 epilogue fusion 等算法层。

## 当前行动优先级

1. T27/T28/T31/T36/T38/T41 已有七芯候选；平台或工单出现明确信号后，
   复用封存候选做一次验证，不继续盲改 generic。
2. T33 是当前唯一开放且已有正向结构证据的题。E8 tile16 已达 8/8、
   `5.4430x`；下一轴只做离线 tile32 扫描和燧原路由，不扩大搜索面。
3. T29/T34 仅保留小行多-row、长行整-row 的静态分桶假设；没有代理门收益
   不发射。
4. T32 的 row-owner、T39 的有效行压缩只作为未来结构研究。T32 宽瓦片和
   zero-weight skip、T39 普通 BLOCK/grid 轴已经平台证伪。
5. T30/T35/T40 已在代理或平台可见轴上收盘，不再做普通参数扫描。

## Batch 1

| Task | 上游生产参考 | 可迁移优化方向 |
| ---: | --- | --- |
| 1 `causal_conv1d_fn` | [SGLang causal-conv Triton](https://github.com/sgl-project/sglang/blob/0674be736ceb138a2f4982c6d612754d2b319807/python/sglang/kernels/ops/mamba/causal_conv1d_triton.py) | `width` constexpr 展开；token×channel 二维 tile，权重驻留并融合 bias+SiLU。按 `query_start_loc` 隔离变长序列，删除 CUDA `.ca` 等专用提示。 |
| 2 `chunk_local_cumsum_scalar` | [SGLang/FLA cumsum](https://github.com/sgl-project/sglang/blob/0674be736ceb138a2f4982c6d612754d2b319807/python/sglang/kernels/ops/attention/fla/cumsum.py) | 短 chunk 一个 program 完整 scan；reverse 用 `total-prefix+x`。低质量 lowering 芯片改软件 scan，长 chunk 才用局部 scan+块前缀。 |
| 3 `fused_moe_gemm` | [SGLang fused MoE](https://github.com/sgl-project/sglang/blob/0674be736ceb138a2f4982c6d612754d2b319807/python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py) | 大 token 数按 expert 分组后做 grouped GEMM，routing weight 融入 epilogue；小 token 数直接 token×top-k tiled GEMM，避免排序成本。FP32 dot 显式 IEEE。 |
| 4 `merge_state` | [vLLM Triton](https://github.com/vllm-project/vllm/blob/399247cc8877f60f02f3aa859c61c3330a59bfbb/vllm/v1/attention/ops/triton_merge_attn_states.py)、[FlashInfer cascade](https://github.com/flashinfer-ai/flashinfer/blob/2cc51dcf67ee71aade7074c64e84f13b7b7b117b/flashinfer/triton/kernels/cascade.py) | 每 token/head 一个 program，D 向量化；FP32 stable max-exp-sum 合并，显式处理空状态和 Inf，避免中间权重张量。 |
| 5 `mrope_fused` | [SGLang rotary/mRoPE](https://github.com/sgl-project/sglang/blob/0674be736ceb138a2f4982c6d612754d2b319807/python/sglang/kernels/ops/attention/rotary_triton.py) | Q/K 同 kernel；同 token 多 head 复用 cos/sin，T/H/W section 编译期化。大 head 数分 tile，防寄存器溢出。 |
| 6 `per_group_transpose` | [SGLang `_per_group_transpose`](https://github.com/sgl-project/sglang/blob/0674be736ceb138a2f4982c6d612754d2b319807/python/sglang/kernels/ops/quantization/fp8_kernel.py) | expert×M-tile×K-tile 二维搬运，连续读取、合并写入，offset 用 int32；空/极小 group 单独 fast path，保持 bit-exact。 |
| 7 `silu_and_mul` | [SGLang elementwise](https://github.com/sgl-project/sglang/blob/0674be736ceb138a2f4982c6d612754d2b319807/python/sglang/kernels/ops/elementwise/elementwise.py) | 两半输入成对加载，FP32 SiLU 后直接写；小 hidden 多-row，大 hidden 256/512/1024 静态分桶。 |

## Batch 2

| Task | 上游生产参考 | 可迁移优化方向 |
| ---: | --- | --- |
| 8 `apply_token_bitmask` | [SGLang bitmask](https://github.com/sgl-project/sglang/blob/0674be736ceb138a2f4982c6d612754d2b319807/python/sglang/kernels/ops/grammar/bitmask_ops.py) | 一个 packed word 只加载一次并展开 32 bits，融合 copy+mask。赛题是 out-of-place；word-layout 只按已验证 vendor 使用。 |
| 9 `bmm_chunk` | [SGLang SSD BMM](https://github.com/sgl-project/sglang/blob/0674be736ceb138a2f4982c6d612754d2b319807/python/sglang/kernels/ops/mamba/triton_ops/ssd_bmm.py) | 16/32 M/N/K tiled dot，输出 FP32；赛题 `causal` 无数值语义，不能照抄上游 causal mask。精度按芯分派，generic 禁 TF32。 |
| 10 `chunk_cumsum` | [SGLang SSD cumsum](https://github.com/sgl-project/sglang/blob/0674be736ceb138a2f4982c6d612754d2b319807/python/sglang/kernels/ops/mamba/triton_ops/ssd_chunk_state.py) | bias、softplus、clamp、`dt*A`、scan 和双输出保持融合。当前问题在 scan lowering；下一步是分层 scan，不是继续扫 BLOCK/warp。 |
| 11 `chunk_local_cumsum_vector` | [SGLang/FLA triangular-dot](https://github.com/sgl-project/sglang/blob/0674be736ceb138a2f4982c6d612754d2b319807/python/sglang/kernels/ops/attention/fla/cumsum.py) | 小 chunk/坏 dot 芯片走软件或 `tl.cumsum`；固定长度、dot 友好芯片才走下三角矩阵乘。 |
| 12 `chunk_state` | [SGLang Mamba](https://github.com/sgl-project/sglang/blob/0674be736ceb138a2f4982c6d612754d2b319807/python/sglang/kernels/ops/mamba/triton_ops/ssd_chunk_state.py) | 衰减构造与 16×16 外积/GEMM 融合，同 group heads 复用 B。天数 split-fp16、昆仑 fp32-ieee，必须 vendor 分派。 |
| 13 `chunk_state_varlen` | [SGLang chunk-state](https://github.com/sgl-project/sglang/blob/0674be736ceb138a2f4982c6d612754d2b319807/python/sglang/kernels/ops/mamba/triton_ops/ssd_chunk_state.py) | 每序列只算最后有效 chunk，完整 chunk 走无 mask fast path。赛题 `chunk_states` 仅决定输出 dtype，不能加入上游累积项。 |
| 14 `context_attention` | [SGLang prefill](https://github.com/sgl-project/sglang/blob/0674be736ceb138a2f4982c6d612754d2b319807/python/sglang/kernels/ops/attention/prefill_attention.py)、[FlashInfer prefill](https://github.com/flashinfer-ai/flashinfer/blob/2cc51dcf67ee71aade7074c64e84f13b7b7b117b/flashinfer/prefill.py) | FlashAttention online softmax和因果 tile 剪枝；32/64 tile、stage=1 作为低共享内存八芯基线。 |
| 15 `decode_attention` | [SGLang decode](https://github.com/sgl-project/sglang/blob/0674be736ceb138a2f4982c6d612754d2b319807/python/sglang/kernels/ops/attention/decode_attention.py)、[FlashInfer decode](https://github.com/flashinfer-ai/flashinfer/blob/2cc51dcf67ee71aade7074c64e84f13b7b7b117b/flashinfer/decode.py) | 短 KV 单阶段；长 KV split-KV 产生 partial O/LSE，再用 T4 公式合并。split 数按 KV 长度和并行度分桶。 |
| 16 `decode_grouped_attention` | [SGLang grouped decode](https://github.com/sgl-project/sglang/blob/0674be736ceb138a2f4982c6d612754d2b319807/python/sglang/kernels/ops/attention/decode_attention.py) | 一个 KV head 同时处理多个 Q heads，K/V 只加载一次；group 大时拆 Q-head tile 控制寄存器。 |
| 17 `embedding_lora_a` | [SGLang embedding LoRA-A](https://github.com/sgl-project/sglang/blob/0674be736ceb138a2f4982c6d612754d2b319807/python/sglang/kernels/ops/gemm/embedding_lora_a.py) | token×rank tile，连续 segment 多 token 合并；adapter=-1、rank=0、extra vocab 用 constexpr/masked fast path。 |
| 18 `fused_recurrent_gdn` | [FLA fused recurrent](https://github.com/fla-org/flash-linear-attention/blob/35dceaee5408e69a555fec34cb215c93c375dabe/fla/ops/gated_delta_rule/fused_recurrent.py) | 状态 `[V-tile,K]` 留 FP32 寄存器并沿 T 串行；减小 V tile 控制 spill。长 T 才考虑严格等价的 chunk/WY。 |
| 19 `fused_rmsnorm` | [SGLang RMSNorm](https://github.com/sgl-project/sglang/blob/0674be736ceb138a2f4982c6d612754d2b319807/python/sglang/kernels/ops/elementwise/elementwise.py) | 中小 hidden 一行一 program、FP32 reduce；小 hidden 多行 pack，超大 hidden 才两阶段。 |
| 20 `mamba_layernorm_gated` | [SGLang/FLA](https://github.com/sgl-project/sglang/blob/0674be736ceb138a2f4982c6d612754d2b319807/python/sglang/kernels/ops/attention/fla/layernorm_gated.py) | RMS/LN、group、gate-before/after constexpr 剪枝；x/z 一次加载，统计、affine、SiLU gate 同 epilogue 完成。 |
| 21 `moe_sum_reduce` | [SGLang MoE reduce](https://github.com/sgl-project/sglang/blob/0674be736ceb138a2f4982c6d612754d2b319807/python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py) | top-k 静态展开、FP32 累加并融合 scale；hidden 256–1024 分桶。昆仑保留已验证 BLOCK1024 vendor。 |
| 22 `qkv_lora_b` | [SGLang](https://github.com/sgl-project/sglang/blob/0674be736ceb138a2f4982c6d612754d2b319807/python/sglang/kernels/ops/gemm/qkv_lora_b.py)、[vLLM Punica expand](https://github.com/vllm-project/vllm/blob/399247cc8877f60f02f3aa859c61c3330a59bfbb/vllm/lora/ops/triton_ops/lora_expand_op.py) | Q/K/V 三 slice 同 launch，复用 rank 输入和 adapter metadata，base-add 融入 epilogue。rank<16 用 FMA，规整 rank 才 padded dot。 |
| 23 `sgemm_lora_b` | [SGLang](https://github.com/sgl-project/sglang/blob/0674be736ceb138a2f4982c6d612754d2b319807/python/sglang/kernels/ops/gemm/sgemm_lora_b.py)、[vLLM expand](https://github.com/vllm-project/vllm/blob/399247cc8877f60f02f3aa859c61c3330a59bfbb/vllm/lora/ops/triton_ops/lora_expand_op.py) | 小 segment/BGMV 和大 segment/SGMV 两套路径；同 adapter token 连续处理，base output 只读写一次。 |
| 24 `softcap_out` | [SGLang softcap](https://github.com/sgl-project/sglang/blob/0674be736ceb138a2f4982c6d612754d2b319807/python/sglang/kernels/ops/activation/softcap.py) | flat 单遍，输出固定 FP32。不要照抄可能溢出的 exp 比值；用稳定 tanh 或 `2*sigmoid(2x)-1`。 |

## Batch 3

| Task | 上游生产参考 | 可迁移优化方向与当前结论 |
| ---: | --- | --- |
| 25 `draft_topk1` | [SGLang hierarchical topk1](https://github.com/sgl-project/sglang/blob/0674be736ceb138a2f4982c6d612754d2b319807/python/sglang/kernels/ops/speculative/topk1.py) | 小 V 单 pass、大 V partial+finalize，融合 position/draft 写回，明确 lowest-id tie。当前 6/8；普通 kernel 轴已结束。 |
| 26 `fused_moe_router_cudacore` | [SGLang router](https://github.com/sgl-project/sglang/blob/0674be736ceb138a2f4982c6d612754d2b319807/python/sglang/kernels/ops/moe/router.py) | 小 E×H 整行 FMA；大形状拆 H/E，稳定 softcap+softmax，top-k 用 deterministic iterative argmax。当前 6/8。 |
| 27 `fused_moe_router_tensorcore` | [SGLang tensorcore router](https://github.com/sgl-project/sglang/blob/0674be736ceb138a2f4982c6d612754d2b319807/python/sglang/kernels/ops/moe/router.py) | 大 batch/专家数走 dot，小 shape 走 FMA；每芯独立决定路径并显式 IEEE。当前个账为 7/8，仅差昆仑。 |
| 28 `gate_up_lora_b` | [SGLang](https://github.com/sgl-project/sglang/blob/0674be736ceb138a2f4982c6d612754d2b319807/python/sglang/kernels/ops/gemm/gate_up_lora_b.py)、[vLLM expand](https://github.com/vllm-project/vllm/blob/399247cc8877f60f02f3aa859c61c3330a59bfbb/vllm/lora/ops/triton_ops/lora_expand_op.py) | gate/up 合成输出轴，rank 输入和元数据只加载一次。七芯均值约 18.5x；等昆仑平台恢复后复用封存包。 |
| 29 `gelu_and_mul` | [SGLang exact-erf](https://github.com/sgl-project/sglang/blob/0674be736ceb138a2f4982c6d612754d2b319807/python/sglang/kernels/ops/elementwise/elementwise.py) | 必须 exact erf，不能默认换 tanh 近似。燧原宽 tile 已兑现；只保留按 N 分桶的多-row/整-row 假设。 |
| 30 `interleaved_rope` | [SGLang 完全同构实现](https://github.com/sgl-project/sglang/blob/0674be736ceb138a2f4982c6d612754d2b319807/python/sglang/kernels/ops/attention/mrope.py) | 三流 bit-exact gather。2D、row-strip、三元组三种重写均负收益；保持 flat 1D，收盘。 |
| 31 `moe_fused_gate` | [SGLang unified gate](https://github.com/sgl-project/sglang/blob/0674be736ceb138a2f4982c6d612754d2b319807/python/sglang/kernels/ops/moe/moe_fused_gate.py) | 小专家数 row-pack；大专家数 streaming/bitonic top-k。CUDA radix 只作算法参考，generic 显式 tie。当前 7/8，昆仑平台阻断。 |
| 32 `moe_fused_mul_sum` | [vLLM 同题 kernel](https://github.com/vllm-project/vllm/blob/399247cc8877f60f02f3aa859c61c3330a59bfbb/vllm/model_executor/layers/fused_moe/moe_fused_mul_sum.py)、[SGLang](https://github.com/sgl-project/sglang/blob/0674be736ceb138a2f4982c6d612754d2b319807/python/sglang/kernels/ops/moe/moe_fused_mul_sum.py) | vLLM row-owner 与 SGLang token×hidden tile 是未来仅存结构参考；宽瓦片、BLOCK、direct、zero-weight skip 已证伪，当前收盘。 |
| 33 `per_token_group_quant_int8` | [SGLang INT8](https://github.com/sgl-project/sglang/blob/0674be736ceb138a2f4982c6d612754d2b319807/python/sglang/kernels/ops/quantization/int8_kernel.py) | 必须 `tl.math.div_rn` 后向零截断。E8 tile16 已达 5.4430x；下一轴仅离线 tile32/燧原路由，禁两阶段。 |
| 34 `per_token_quant_int8` | [SGLang INT8](https://github.com/sgl-project/sglang/blob/0674be736ceb138a2f4982c6d612754d2b319807/python/sglang/kernels/ops/quantization/int8_kernel.py) | 上游专用路径是 round，但比赛可执行 reference 实际复用 T33、语义是 trunc。保持整行单 pass；华为/昆仑两阶段已证伪。 |
| 35 `rotary_embedding` | [SGLang rotary](https://github.com/sgl-project/sglang/blob/0674be736ceb138a2f4982c6d612754d2b319807/python/sglang/kernels/ops/attention/rotary_triton.py)、[FlashInfer RoPE](https://github.com/flashinfer-ai/flashinfer/blob/2cc51dcf67ee71aade7074c64e84f13b7b7b117b/flashinfer/rope.py) | token×head×pair tile，cos/sin 跨 heads 复用。四头 tile 已兑现；Ascend 保持 1D，燧原增宽到 16 已证伪，收盘。 |
| 36 `selective_state_update` | [Mamba Triton](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/ops/triton/selective_state_update.py)、[FlashInfer](https://github.com/flashinfer-ai/flashinfer/blob/2cc51dcf67ee71aade7074c64e84f13b7b7b117b/flashinfer/mamba/selective_state_update.py) | 每 batch/head/dim tile 持有 dstate，融合 dt/exp/B/C/D/z 并只写 state 一次。真实平台 A 是 `[H,P,N]`；当前 7/8，仅昆仑平台阻断。 |
| 37 `sgemm_lora_a` | [SGLang](https://github.com/sgl-project/sglang/blob/0674be736ceb138a2f4982c6d612754d2b319807/python/sglang/kernels/ops/gemm/sgemm_lora_a.py)、[vLLM shrink](https://github.com/vllm-project/vllm/blob/399247cc8877f60f02f3aa859c61c3330a59bfbb/vllm/lora/ops/triton_ops/lora_shrink_op.py) | 大 K、小 rank 可 split-K，小 token 数保持单 CTA。当前燧原行错位、昆仑崩溃；先正确性，后性能。 |
| 38 `sigmoid_gate_topk_renorm` | [SGLang streaming top-k](https://github.com/sgl-project/sglang/blob/0674be736ceb138a2f4982c6d612754d2b319807/python/sglang/kernels/ops/moe/sigmoid_gate_topk_renorm.py) | N 小 iterative argmax；N 大 streaming top-k。selection 用加 bias score，输出重取原 sigmoid；generic 避免依赖 uint64 packed-key。当前 7/8。 |
| 39 `silu_and_mul_masked` | [SGLang EP MoE](https://github.com/sgl-project/sglang/blob/0674be736ceb138a2f4982c6d612754d2b319807/python/sglang/kernels/ops/moe/ep_moe_kernels.py) | 在 load 前跳过无效 token，不能先越界读再 mask。若重开，只研究有效行压缩或 host-resolved grid；普通 BLOCK/grid 轴已结束。 |
| 40 `softcap_inplace_logits` | [SGLang inplace softcap](https://github.com/sgl-project/sglang/blob/0674be736ceb138a2f4982c6d612754d2b319807/python/sglang/kernels/ops/activation/softcap.py) | 单遍、原地、同 dtype、稳定 tanh/sigmoid。BLOCK、Ascend direct/native tanh 均证伪，已贴 I/O 下界，收盘。 |
| 41 `state_passing` | [SGLang](https://github.com/sgl-project/sglang/blob/0674be736ceb138a2f4982c6d612754d2b319807/python/sglang/kernels/ops/mamba/triton_ops/ssd_state_passing.py)、[Mamba](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/ops/triton/ssd_state_passing.py) | batch/head/dim tile 保持 running state，串行遍历 chunks，写 pre-update state 和 FP32 final state。当前七芯候选封存。 |

## 八芯通用落地规则

1. Generic 从 `num_warps<=4`、`num_stages=1` 起步，不假定 warp=32；
   物理 grid 超 65535 时使用 capped grid-stride。
2. 热路径优先 int32 索引；精确 int64 输出和超大地址另行处理，不让 int64
   进入每元素地址计算。
3. FP32 `tl.dot` 显式 IEEE。天数可能需要 split-fp16 四点积，昆仑又可能
   要求 fp32，必须 vendor 分派。
4. top-k/argmax 明确 lowest-id tie，不假定 `tl.argmax`、`tl.sort`、
   `tl.topk` 在各后端的 tie 和 lowering 一致。
5. 每芯 `0.1x` 是发布门槛。先修唯一弱芯，再追均值；冻结已验证芯的源码字节，
   不把多个猜测绑进一次平台提交。

## 已知契约陷阱

- T8 上游是 inplace+可选 indices，赛题是 out-of-place、无 indices。
- T10 上游 Mamba 的两个输出顺序与赛题相反。
- T13 的 `chunk_states` 在赛题中只决定输出 dtype，不能参与累积。
- T24 输出固定 FP32；T40 则是 inplace 且保持输入 dtype。
- T27 状态应以个账 7/8 为准，早期汇总中的 6/8 已过期。
- T34 题面文字写 round，但可执行 reference 复用 T33，实际向零截断。
- T35 虽传 `interleaved=False`，可执行 reference 仍按相邻偶/奇维配对。
- T36 平台实证 `A=[nheads,dim,dstate]`，题面二维描述已过期。
