# 外部 Kernel 优化 Skills（参考文档）

来源：[tensormux/kernel-skills](https://github.com/tensormux/kernel-skills)（MIT，见 `LICENSE`），
2026-08-29 引入，浅克隆 commit 为 main@HEAD。仅作为第三批算子开发的最佳实践
参考资料，不注册为 SkillHub skill；具体取舍仍服从
`.agents/skills/flagos-operator-race/SKILL.md` 的契约锁定与门禁流程。

## 精选清单与批次对应

| 目录 | 内容 | 对应赛题 |
|---|---|---|
| `triton-optimize-triton-block-parameters` | BLOCK_M/N/K、num_warps/stages、autotune 取舍 | 全部 Triton 题 |
| `triton-write-triton-gemm-kernel` | Triton GEMM 模板与 tile/流水线 | T28 gate_up_lora_b、router tensorcore |
| `inference-write-triton-rope-kernel` | RoPE kernel（interleave/half-split、cos/sin 复用） | T30 interleaved_rope、T35 rotary_embedding |
| `inference-write-triton-silu-mul-kernel` | 激活+乘法融合的带宽优化 | T29 gelu_and_mul |
| `quantization-write-int8-quantized-kernel` | 逐 token/group 量化（amax、scale、rounding） | T33/T34 per_token(_group)_quant_int8 |
| `patterns-fuse-elementwise-ops` | elementwise 融合通用模式 | T32 moe_fused_mul_sum、T29 |
| `patterns-choose-tile-size-and-work-partitioning` | tile 与工作划分通用方法论 | 全部 |
| `patterns-write-numerically-stable-kernel` | 数值稳定性（softmax/exp 归约、容差） | router/gate 类 |
| `patterns-write-kernel-test-plan` | kernel 测试计划模板 | tests/ |
| `portability-write-backend-agnostic-kernel-plan` | 跨后端（NVIDIA/AMD/国产）可移植写法 | 八芯 generic 首版 |

## 未引入但可回查的来源

- 完整列表见上游 `skills/`（cuda/* 9 个、inference/* 10 个等）。
- [ForceInjection/cuda-code-skill](https://github.com/ForceInjection/cuda-code-skill)：NVIDIA 文档离线 RAG + ncu 报告分析。
- [nvidia/skills](https://github.com/nvidia/skills)：TileGym（CuTe DSL），NVIDIA 专用。
