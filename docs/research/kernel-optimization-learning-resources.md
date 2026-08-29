# Kernel 优化学习资源索引

> 调研日期：2026-08-29。联网检索整理的 GPU kernel 优化技巧、最佳实践、工具链与学习材料。
> 面向 FlagOS 算子赛场景（Triton/CUDA、attention/MoE/RoPE/量化类算子）做了优先级标注。

## 推荐学习路径（TL;DR）

1. **建立概念**：LLM Inference Handbook 的 Kernel Optimization 章节 + CUDA Best Practices Guide 通读。
2. **动手练 kernel**：BBuf/how-to-optim-algorithm-in-cuda（算子类型与比赛最对口）+ Triton 官方 tutorials。
3. **看讲座补深度**：GPU MODE 系列（Lecture 1 profile、Lecture 14 Triton、Lecture 44 NVIDIA Profiling）。
4. **工具落地**：nsys 定位热点 kernel → ncu 做硬件计数器分析，应用到远端八芯评测前的瓶颈定位。
5. **前沿打法**：KernelBench / CUDA Agent 的 agent 策略，可借鉴到 kernelgen-flagos 闭环。

---

## 1. 权威指南 / 最佳实践

| 资源 | 说明 |
| --- | --- |
| [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) | NVIDIA 官方，内存/并行执行/指令级优化权威参考，持续更新 |
| [LLM Inference Handbook — Kernel Optimization](https://handbook.modular.com/kernel-optimization/) | Modular 出品，从计算/带宽/片上资源角度系统讲 kernel 优化，概念入门首选 |
| [Optimizing Triton Kernels (AMD ROCm)](https://rocm.docs.amd.com/en/docs-6.1.1/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html) | ROCm 官方 Triton 优化步骤，思路与 CUDA/HIP 优化类似 |
| [Mastering CUDA Kernel Development (Medium)](https://medium.com/@omkarpast/mastering-cuda-kernel-development-a-comprehensive-guide-1f3032666b94) | 线程配置、occupancy、合并访存等实操要点 |
| [Optimizing CUDA Kernels Checklist](https://www.rimikawrites.com/cuda-3-your-checklist-for-optimizing-cuda-kernels/) | 结构化优化清单，适合评审自查 |

## 2. 教程 / 代码仓库

- **[BBuf/how-to-optim-algorithm-in-cuda](https://github.com/BBuf/how-to-optim-algorithm-in-cuda)** ⭐ 与比赛最对口：中文社区最全的 CUDA 算子优化合集，含 CUTLASS/CuTe 笔记、Triton 示例、PTX ISA、LLM 训推优化文章（attention/MoE/RoPE/量化全覆盖）。
- **[rkinas/triton-resources](https://github.com/rkinas/triton-resources)**：Triton 从简单 kernel 到高级应用的阶梯式学习路径。
- [Triton 官方 tutorials](https://triton-lang.org) + [PyTorch Blog: Triton Kernel Compilation Stages](https://pytorch.org/blog/triton-kernel-compilation-stages/)（理解编译流程对调优有帮助）。
- [Fast LLM Inference From Scratch (Andrew Chan)](https://andrewkchan.dev/posts/yalm.html)：C++/CUDA 从零构建 LLM 推理引擎，端到端实践。
- [vLLM 深入与自定义 kernel 开发](https://martinuke0.github.io/posts/2026-03-21-optimizing-llm-inference-a-deep-dive-into-vllm-and-custom-kernel-development/)：vLLM 原理 + 自定义 attention kernel 开发与 benchmark。
- [Deep Dive: CUDA Optimization for LLM & GenAI Inference](https://medium.com/@dwivediavivek/deep-dive-cuda-optimization-for-llm-genai-inference-2e39b5046595)：persistent kernel fusion、减少 launch 延迟。
- [NVIDIA: TensorRT-LLM 推理优化](https://developer.nvidia.com/blog/optimizing-inference-on-llms-with-tensorrt-llm-now-publicly-available/)。
- 社区问答：[Stack Overflow: CUDA optimization techniques](https://stackoverflow.com/questions/3090493/cuda-optimization-techniques)、[Reddit r/CUDA](https://www.reddit.com/r/CUDA/)。

## 3. 视频讲座

- **[GPU MODE YouTube 频道](https://www.youtube.com/@GPUMODE)** ⭐ 社区质量最高的免费 GPU 编程系列：
  - [Lecture 1: 在 PyTorch 中 profile CUDA kernel](https://christianjmills.com/posts/cuda-mode-notes/lecture-001/)（[笔记](https://christianjmills.com/posts/cuda-mode-notes/lecture-001/)）
  - [Lecture 14: Triton 实战指南](https://christianjmills.com/posts/cuda-mode-notes/lecture-014/)
  - [Lecture 44: NVIDIA Profiling](https://www.youtube.com/watch?v=F_BazucyCMw)（[笔记](https://www.josherich.me/podcast/gpu-mode/lecture-44-nvidia-profiling)）
- NVIDIA 官方：[Nsight Compute 入门](https://www.youtube.com/watch?v=Iuy_RAvguBM)、[SOL (Speed of Light) 分析](https://www.youtube.com/watch?v=uHN5fpfu8As)、[Nsight 视频合集](https://developer.nvidia.com/nsight-compute-videos)。
- [GTC24: Introduction to CUDA Programming and Performance](https://www.nvidia.com/en-us/on-demand/session/gtc24-s62191/)。
- [DevConf.US 2025: GPU Programming with Triton Kernels](https://www.youtube.com/watch?v=sv4soasZK7U)。

## 4. Profiling 工作流（可直接落地）

各来源一致推荐的流程：

1. **`nsys`（Nsight Systems）先看系统级 timeline**，找出占主导的热点 kernel（含 launch 开销、内存拷贝、gap）。
2. **`ncu`（Nsight Compute）针对该 kernel 做硬件计数器分析**，例如 `ncu -k <kernel> --launch-count N`，重点看 SOL（Speed of Light）、occupancy、内存吞吐、warp stall 原因。

参考：[Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)、[NERSC NVIDIA Profiling Tools](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)。

> 对 FlagOS 赛题的应用：远端评测前先 nsys 定位 batch 内最慢算子，ncu 判断是带宽瓶颈还是 occupancy/调度瓶颈，再决定 vectorization / tile 重排 / persistent kernel 等优化方向。

## 5. AI 辅助优化：Skill / MCP

- **[cfregly/gpu-perf-tune](https://github.com/cfregly/gpu-perf-tune)**：GPU 推理 profiling + 优化 Claude Code skill，内置 MCP server（profile-and-optimize 插件）。
- [HuggingFace: Custom Kernels for All（Codex/Claude agent skill）](https://huggingface.co/blog/custom-cuda-kernels-agent-skills)：~550 token 结构化指导 + 参考脚本 + GPU 优化指南。
- [CUDA Optimization skill (MCP Market)](https://mcpmarket.com/tools/skills/cuda-performance-optimizer)、[CUDA GPU Computing skill](https://mcpmarket.com/tools/skills/cuda-gpu-computing)。
- [kernel-tileir-optimization](https://mcpservers.org/agent-skills/nvidia/kernel-tileir-optimization)：面向 Blackwell TileIR 后端优化已有 Triton kernel。

> Skill 与 MCP 的区别（[Reddit 讨论](https://www.reddit.com/r/mcp/comments/1pn2eyj/claude_skills_and_mcp/)）：MCP 连接外部系统，skill 是可复用的提示词工作流。可借鉴其结构化指导写法完善本仓库的 kernelgen-flagos skill。

## 6. LLM 驱动 kernel 生成（前沿方向）

- [KernelBench (Stanford ScalingIntelligence)](https://github.com/ScalingIntelligence/KernelBench) + [官方博客](https://scalingintelligence.stanford.edu/blogs/kernelbench/)：250 个 PyTorch 算子任务，评估 LLM 写正确且高效的 CUDA kernel。
- [CUDA Agent](https://cuda-agent.github.io/)（[HF 论文页](https://huggingface.co/papers/2602.24286)）：大规模 agent RL 训练写 CUDA，KernelBench SOTA（对 torch.compile 100%/100%/92% faster）。
- [Towards Automated Kernel Generation in the Era of LLMs（综述, arXiv 2601.15727）](https://arxiv.org/html/2601.15727v3)。
- [NeurIPS 2025 Tutorial: Agents for Kernel Generation](https://neurips.cc/virtual/2025/128792)：2.5 小时教程，覆盖 CUDA/HIP/Triton 的 agent 生成方法。
- [GTC26: LLM-Generated CUDA Kernels — Are We There Yet?](https://www.nvidia.com/en-us/on-demand/session/gtc26-s81653/)：分析 LLM 产出的 kernel "能跑但不快" 的差距来源。
- [Making LLMs Optimize Multi-Scenario CUDA Kernels (arXiv)](https://arxiv.org/html/2603.07169v1)、[Profiling-Guided Automated Triton Optimization (arXiv)](https://arxiv.org/html/2512.09196v1)。
- [awesome-LLM-driven-kernel-generation](https://github.com/flagos-ai/awesome-LLM-driven-kernel-generation)：项目聚合页。
- [Simon Guo: Towards Automated GPU Kernel Generation](https://simonguo.tech/blog/2025-10-automated-gpu-kernels.html)：KernelBench 作者视角。

> 对比赛的启示：GTC26 的结论（LLM kernel 功能正确但性能不足）与我们实验账本中的观察一致；CUDA Agent 的 RL + profiling 反馈循环、以及 profiling-guided 自动调优论文，可作为 kernelgen-flagos 闭环（生成 → 远端评测 → 账本反馈）的改进参考。
