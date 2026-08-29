# 第三批按题上游打法速查(2026-08-29 调研定稿)

来源:用户侧对 vLLM/SGLang/fla-core 权威实现的调研 + 本仓 batch-1/2/3
平台实证。与 kernel-skills 快照(见 skillhub-tools.md)互为补充。

| 题目 | 权威上游 | 核心技巧 | 本仓状态/实证 |
| --- | --- | --- | --- |
| 25 draft_topk1 | vLLM MTP、SGLang | 行级 argmax 用 (value,int32_idx) 打包归约;禁 tl.sort/argmax(issue #6635);gather 融合同 kernel | 燧原五种结构硬不支持,已终态 6/8 |
| 26/27 fused_moe_router ×2、31 moe_fused_gate | sglang #26771 统一 Triton router、sgl-kernel-npu | 整行 experts 一个 block,softmax→topk→renorm 全寄存器;topk=迭代 argmax+mask;tensorcore 版 topk 转小 GEMM + tl.dot | 华为 reduce kernel 数值问题待他人证据;T31 未打 |
| 28 gate_up_lora_b | vLLM punica | [r,2*intermediate] skinny GEMM;小 r 时 dot tile≥16 需 padding 或 FMA | 七芯 18.5x,昆仑平台崩溃,工单路径 |
| 29 gelu_and_mul | FlagGems baseline | 1D 展平跑满带宽;大 BLOCK + 宽向量;超越函数按芯换多项式 | **已终态 2.7229x**:A&S erf + 逐块除法 + 按芯 vendor 分治 |
| 30 interleaved_rope、35 rotary_embedding | vLLM mrope、flashinfer | 偶奇对布局用 stride=2 strided load;cos/sin fp32;小 shape launch 主导 | T30 已终态 25.835x(平铺局部最优);T35 待打 |
| 32 moe_fused_mul_sum | vLLM TopKWeightAndReduce | top_k 维整个进一个 program,寄存器加权累加,单次写;权重标量广播 | 待打 |
| 33/34 per_token(_group)_quant_int8 | vLLM int8_utils | 单 pass amax→scale→int8 全寄存器;rounding 模式须与参考逐位一致 | T33 3.5385x(**div_rn 根因:普通 `/` 是近似除**);T34 待打 |
| 36 selective_state_update | fla-core | 一 program 一 (batch,head);state 常驻寄存器;时间步外循环,零全局 state 读写 | 未打 |

## 跨芯通用(上游调研 ∩ 本仓平台实证)

1. 非 NVIDIA 后端对 **int64 索引惩罚重**:offset 全 int32(T30 c1 平台
   前车之鉴;T29 E5 int64 曾拖累)。
2. grid 优先 1D capped + grid-stride(昇腾/昆仑 2D 总数 ≤65535)。
3. BLOCK 取 2 的幂;BLOCK 是昆仑唯一有效轴,燧原偏好大 tile + stages≥2
   (仅 dot),华为 reduce tile 有 UB 上限。
4. `tl.dot` tile 约束各芯不同,需 FMA fallback;fp16 操作数 dot 在
   昆仑数值失败、天数 fp32 操作数静默不可执行(互为镜像)。
5. Triton 普通 `/` 是近似除;与 torch 逐位一致必须 `tl.math.div_rn`
   (T33 根因)。libdevice 超越函数(erf)在昆仑崩溃,A&S 多项式替代。
6. 保守语法可移植性最好;triton-ascend 官方迁移指南:
   github.com/triton-lang/triton-ascend/docs/zh/migration_guide。
