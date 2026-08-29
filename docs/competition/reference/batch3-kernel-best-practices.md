# 第三批算子最佳实践调研（2026-08-29 联网沉淀）

> 来源：两轮联网调研（batch-3 逐题最佳实践 + kernel 优化 skill/方法论盘点）。
> 本页是结论摘要与检索入口；逐题细节以各算子实验账本为准。
> 配套引入的 skill 全文见 [kernel-skills/](kernel-skills/README.md)。

## 逐题 playbook（题号 → 权威上游 + 已验证技巧）

### T25 draft_topk1（MTP top-1 argmax + gather）

- 上游：vLLM speculative decoding（MTP/EAGLE）、SGLang DeepSeek-V3 MTP。
- 技巧：行级 `(value, int32_idx)` 打包归约，禁用 `tl.sort`；`tl.argmax`
  有正确性 issue（triton#6635），需自带 value+index 联合归约与测试；
  gather 融进同一 kernel；小算子 launch 开销占比高。
- 昇腾参考：sgl-kernel-npu 的 Top-K kernel。

### T26/T27 fused_moe_router（cudacore / tensorcore）与 T31 moe_fused_gate

- 上游：sglang #26771 统一 Triton router（ungrouped+grouped，H100/B200
  快于 AOT）；HF 博客 SGLang MoE align&sort 剖析；vLLM fused_moe.py
  与 Fused MoE Modular Kernel 设计。
- 技巧：整行 experts（128–512）装进一个 block，softmax→topk→renorm
  全寄存器单次读写；topk 用迭代 argmax+mask（k≤8），不做全局排序；
  tensorcore 版把 topk 转小 GEMM / DeepSeek 式 topk_group + bias +
  renormalize 走 `tl.dot`；num_experts 非 2 幂时 mask/padding；
  topk_ids 输出 int32。

### T28 gate_up_lora_b（LoRA B 侧 skinny GEMM）

- 上游：vLLM punica_gpu wrapper（lora_a/lora_b Triton GEMM）、Punica
  论文（SGMV/BGMV：BGMV 利 decode、SGMV 利 prefill）、vLLM #2893。
- 技巧：gate/up 视作 `[r, 2*intermediate]` 大 N GEMM 一次 launch；K=r
  很小，`tl.dot` tile≥16 约束需 padding 或 outer-product/FMA 路径；
  单 LoRA 场景免 SGMV 间接寻址；A tile 常驻沿 N 流式（PyTorch MoE
  locality 博客）。

### T29 gelu_and_mul

- 上游：FlagGems 自带实现即 baseline。
- 技巧：1D 展平跑满带宽；大 BLOCK（1024–4096）+ `num_warps` 4–8 +
  宽向量 load/store（`tl.multiple_of`）；constexpr 特化 tanh/erf；
  前半/后半错开 hidden/2 偏移一次读两段。

### T30 interleaved_rope / T35 rotary_embedding

- 上游：vLLM mrope（interleaved 处理）、flashinfer in-place rope。
- 技巧：GPT-J interleaved（相邻奇偶对）用 stride=2 load 或 reshape
  重排；GPT-Neo half-split 一次加载 `tl.where` 组装避免双读；cos/sin
  可 kernel 内由 position 现算省一次全局读，或整块 load 复用；in-place
  更新省一次写；部分旋转 dim 用 mask 且不进热路径。

### T32 moe_fused_mul_sum

- 上游：vLLM moe_sum / TopKWeightAndReduce（mul+reduce 融合 epilogue）。
- 技巧：top_k 维整个放进一个 program，权重标量广播，寄存器内加权
  累加，单次读 inputs 单次写 output；竞争只考本算子时重点是访存合并
  与向量化。（本仓库 S0 定稿即此形态 + flat 1D capped grid + int32。）

### T33/T34 per_token(_group)_quant_int8

- 上游：vLLM int8_utils（per_token_group_quant Triton）、vLLM #24185
  重构、Red Hat DeepSeek-R1 优化文。
- 技巧：单 pass——load 一组 → `tl.max(abs)` amax → scale → 写 int8 +
  scale 全寄存器；DeepSeek group=128 恰为 2 幂；scale 布局行主序连续
  写；**rounding 模式必须与 reference 一致**（libdevice round 或
  floor(x+0.5)），容差是 bit-exact 风险点；group 边界 mask。

### T36 selective_state_update（Mamba 状态递推）

- 上游：fla-org/flash-linear-attention（权威 Triton 实现）、mamba-ssm
  （语义参考）、TFLA 论文 arXiv:2503.14376。
- 技巧：一个 program 负责一个 (batch, head) 完整 dstate 维，state
  常驻寄存器，时间步外循环——state 零全局读写；A 逐通道 decay 用
  `tl.exp`；注意 softplus 与 dt_bias 数值路径；state 最内维连续；
  B⊗V 可 `tl.dot` 或 FMA 视精度；重点减少每步冗余 load 并把 nheads
  撑满 SM。

## 跨芯（非 NVIDIA Triton 后端）适配清单

对各厂商路线的固定证据：triton-ascend（官方迁移指南）、Cambricon
triton-linalg（MLIR/Linalg）、摩尔线程 MUSA+FlagTree+TLE。对写 generic
kernel 的直接影响：

1. **grid/分核**：优先 1D/2D 简单 grid；block 数与核数对齐；华为 grid
   上限 65535（本仓库 T21/T32 已两次验证 flat 1D capped grid 必要性）。
2. **对齐**：BLOCK 与 dtype 宽度取 2 幂；指针 32/64B 对齐；避免奇数
   stride（triton-ascend 迁移指南明示）。
3. **`tl.dot` 约束**：各后端 tile 最小尺寸/对齐不同（常 M/N/K≥16），
   非对齐场景必须有 FMA fallback。
4. **特性滞后**：新 intrinsic、reduce 组合、TMA 可能不可用；基础
   load/store/arith 语法可移植性最好。
5. **int64 索引惩罚**：国产后端更重；offset 一律 kernel 内 int32
   计算（大 shape 用块级除法 + 局部 int32，参考 T32 S0 重写）。
6. 原子操作、warp 语义差异见 Triton SPIR-V 后端实践文（共性坑）。

## Kernel 优化 skill / 方法论盘点（持续跟踪入口）

| 名称 | 形式 | 状态 |
| --- | --- | --- |
| [tensormux/kernel-skills](https://github.com/tensormux/kernel-skills) | ~35 个 SKILL.md，MIT，CUDA/Triton/量化/跨芯 | 已精选 10 个落盘 [kernel-skills/](kernel-skills/README.md) |
| [ForceInjection/cuda-code-skill](https://github.com/ForceInjection/cuda-code-skill) | NVIDIA 文档离线 RAG + ncu 分析 + bench 修复 | 未引入（NVIDIA 向） |
| [nvidia/skills](https://github.com/nvidia/skills) | 官方 TileGym（CuTe DSL） | 未引入（NVIDIA 专用） |
| [KernelBench](https://github.com/ScalingIntelligence/KernelBench) | 最佳实践 in-context + 硬件提示 + 先验证后测速 | 方法论已体现在本仓库 screening/release 两段式 |
| meta-pytorch/KernelAgent | 硬件信号闭环多 agent | 方法论参考 |
| flagos-ai/KernelGenBench、awesome-LLM-driven-kernel-generation | 自家基准与跟踪列表 | 持续跟踪 |

### 来源清单（调研原文）

- sglang: [#26771 统一 router](https://github.com/sgl-project/sglang/issues/26771) /
  [MoE align&sort 剖析](https://huggingface.co/blog/yiakwy-xpu-team/efficient-moe-align-sort-design-for-sglang) /
  [sgl-kernel-npu](https://github.com/sgl-project/sgl-kernel-npu)
- vLLM: [fused_moe.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py) /
  [Fused MoE Modular Kernel](https://docs.vllm.ai/en/latest/design/fused_moe_modular_kernel/) /
  [int8_utils](https://docs.vllm.ai/en/v0.28.0/api/vllm/model_executor/layers/quantization/utils/int8_utils/) /
  [#24185](https://github.com/vllm-project/vllm/issues/24185) /
  [punica_gpu](https://docs.vllm.ai/en/stable/api/vllm/lora/punica_wrapper/punica_gpu/) /
  [Punica 论文](https://arxiv.org/pdf/2310.18547)
- fla: [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) /
  [TFLA](https://arxiv.org/html/2503.14376v3)
- Triton 生态: [triton#6635 argmax 精度](https://github.com/triton-lang/triton/issues/6635) /
  [triton-ascend 迁移指南](https://github.com/triton-lang/triton-ascend/blob/main/docs/zh/migration_guide/migrate_from_gpu.md) /
  [triton-linalg](https://github.com/Cambricon/triton-linalg/issues/39)
- 其他: [PyTorch MoE locality](https://pytorch.org/blog/accelerating-moe-model/) /
  [Red Hat DeepSeek-R1 vLLM 优化](https://developers.redhat.com/articles/2025/03/19/how-we-optimized-vllm-deepseek-r1) /
  [中国多芯 AI 栈综述](https://leonliao.substack.com/p/inside-chinas-multi-chip-ai-stack)
