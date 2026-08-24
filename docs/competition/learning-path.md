<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

# FlagOS 第二届算子挑战赛：快速学习路径

> 面向第二批 17 题，整理时间：2026-08-24。目标不是系统学完 GPU 编译器，
> 而是尽快具备“读懂题面 → 写出可移植 Triton kernel → 正确计时 → 生成合规
> ZIP → 根据八芯结果做单变量优化”的能力。

## 先说结论

参赛前真正需要掌握的只有四层：

1. **PyTorch 张量语义**：shape、stride、contiguous/view、dtype 转换、广播、
   数值容差和可信 reference。
2. **Triton 基础**：blocked-program 模型、grid、`program_id`、offset/mask、
   load/store、reduction、`tl.dot`、JIT 参数和基准测试。
3. **题型知识**：只学习所选算子需要的 pointwise、reduction、scan、GEMM、
   attention、Mamba/FLA 或 LoRA 中的一支。
4. **竞赛工程**：严格接口、三 dtype 正确性、尾块、同步计时、八芯可移植性、
   ZIP 文件名和实验留痕。

不需要先学完整 CUDA C++、LLVM/MLIR pass、模型训练、分布式通信或八家厂商
的原生 SDK。只有通用 Triton 已正确、平台结果指出某一芯片瓶颈后，才进入
对应厂商资料。

## 一、必修：按这个顺序学

### 0. 先把比赛边界读准（30 分钟）

- [比赛规则、评分和 ZIP 提交规范](README.md)：正确性优先、每款芯片至少
  `0.1x`、禁止失败后回退 PyTorch、文件 basename 和提交额度。
- [第二批题目索引](task-index.md)：17 题的题面入口；实现前必须再读所选题的
  完整接口、输出 dtype 和容差。
- [第二批快速策略](strategy-batch2.md)：上游映射、已知语义陷阱和推荐顺序。
- [官方仓库贡献指南](../CONTRIBUTING.md)：generic/vendor/arch 三层布局、
  `__all__`、测试、benchmark 和 CI 约定。对应固定上游版本为
  [`3946b9a`](https://github.com/flagos-ai/FlagGems-sglang/blob/3946b9a6e489dce76c37a722c3846c2bba95afca/docs/CONTRIBUTING.md)。

完成标准：能用一句话说清“函数签名、每个输入的 shape/dtype/layout、FP32
累加点、输出 dtype、边界输入、容差和 ZIP 文件名”。没有这张契约表，不写
kernel。

### 1. PyTorch 张量与数值语义（1–2 小时）

重点不是学习模型 API，而是学会把 reference 精确翻译成地址计算：

- [Tensor Views](https://docs.pytorch.org/docs/stable/tensor_view.html)：理解
  view、stride、contiguous 与非连续输入；`reshape`/`permute` 后不能默认线性
  地址仍连续。
- [Broadcasting semantics](https://docs.pytorch.org/docs/stable/notes/broadcasting.html)：
  标量、head/group 维和 bias 的广播规则。
- [Numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)：
  浮点运算不具备数学实数的结合律，不同设备/实现可产生小差异；因此要按题面
  dtype 和容差验证，而不是逐位比较。
- [`torch.testing.assert_close`](https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close)：
  学会显式设置 `atol`/`rtol`，并单独检查 shape、dtype、NaN/Inf 和输入未被
  修改。

最小练习：任选一道题，把 PyTorch reference 拆成“索引变换、读取、FP32
计算、写回”四列，并列出长度 `0/1/BLOCK-1/BLOCK/BLOCK+1` 的预期行为。

### 2. Triton 编程模型与最小语法（3–5 小时）

Triton 的核心抽象是 blocked program，而不是手写每个 CUDA thread；官方
[编程模型介绍](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html)
解释了这种分块 SPMD 模型。按顺序只做以下官方教程，固定源码均来自 Triton
commit `dff2f7d0`：

1. [Vector Addition](https://github.com/triton-lang/triton/blob/dff2f7d03532e9ca0598c728c60c204ae7555fc9/python/tutorials/01-vector-add.py)：
   `program_id`、`arange`、mask、load/store、grid；对应 Task 8/24 的全部骨架。
2. [Fused Softmax](https://github.com/triton-lang/triton/blob/dff2f7d03532e9ca0598c728c60c204ae7555fc9/python/tutorials/02-fused-softmax.py)：
   行级 reduction、数值稳定、融合和资源约束；对应 norm/attention。
3. [Matrix Multiplication](https://github.com/triton-lang/triton/blob/dff2f7d03532e9ca0598c728c60c204ae7555fc9/python/tutorials/03-matrix-multiplication.py)：
   二维 tiling、`tl.dot`、K 循环、layout 和 autotune；对应 Task 9/12/22/23。
4. [Layer Normalization](https://github.com/triton-lang/triton/blob/dff2f7d03532e9ca0598c728c60c204ae7555fc9/python/tutorials/05-layer-norm.py)：
   FP32 reduction、hidden-size block 和寄存器压力；对应 Task 19/20。
5. 只有选择注意力题时再读
   [Fused Attention](https://github.com/triton-lang/triton/blob/dff2f7d03532e9ca0598c728c60c204ae7555fc9/python/tutorials/06-fused-attention.py)。
6. 只有选择分段 LoRA/GEMM 时再读
   [Group GEMM](https://github.com/triton-lang/triton/blob/dff2f7d03532e9ca0598c728c60c204ae7555fc9/python/tutorials/08-grouped-gemm.py)。

同时熟悉这些官方入口：

- [Triton language API](https://triton-lang.org/main/python-api/triton.language.html)：
  不凭记忆猜 `tl.*` 的参数和 dtype 行为。
- [Debugging Triton](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html)：
  `static_assert`、`device_assert` 和 `TRITON_INTERPRET=1`；官方文档同时列出
  interpreter 对 BF16 和间接访存的限制。
- [`triton.testing.do_bench`](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html)：
  使用框架提供的 warmup、重复和 quantile 计时，不用一次 `time.time()`。

完成标准：能独立写一个 flat 1D masked kernel，并解释每个指针 offset、tail
mask、FP32 cast、grid 和 `BLOCK_SIZE` 的作用。

### 3. 性能分析的最小知识（2 小时）

只学会判断瓶颈，不背硬件参数：

- **内存**：连续/合并访存、重复 global load、输出写流量、对齐和数据复用。
- **计算**：特殊函数、reduction、`tl.dot`，以及计算量相对访存量的比例。
- **并行度**：grid 是否覆盖设备、单 program 工作量、寄存器/片上存储是否限制
  活跃 program；更大的 BLOCK 并不自动更快。
- **测量**：先 warmup/JIT，再同步计时；固定输入和 shape，轮换候选顺序，一次
  只改一个变量，正确性失败的候选不计性能。

背景材料只需扫读：

- [NVIDIA CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)：
  coalescing、occupancy、register/shared-memory 权衡；这些概念可帮助解释本地
  NVIDIA 结果，但不能直接当作其他芯片参数。
- [AMD HIP Performance Guidelines](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html)：
  官方建议的流程是 baseline/profile → 判断 compute/memory bound → 定向优化
  → 复测；文档也说明 AMD 平台可能存在不同 wave 宽度。
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)：
  只有普通 benchmark 无法解释瓶颈时，再看 launch、memory、compute 和指令
  指标。

## 二、按题选修：只学主攻题所在的一行

| 题型 | 第二批题目 | 需要补的知识 | 第一方材料 |
| --- | --- | --- | --- |
| Pointwise / bitmask | 8、24 | flat indexing、mask、位运算、FP32 特殊函数、数值稳定 | [Task 8](tasks/batch-2/08-apply_token_bitmask.md)、[Task 24](tasks/batch-2/24-softcap_out.md)、[SGLang softcap 固定源码](https://github.com/sgl-project/sglang/blob/8014d9d062c3cc5d393596ecdf2f7009191965df/python/sglang/kernels/ops/activation/softcap.py#L30-L68)、[FlagGems 稳定 tanh fallback](https://github.com/flagos-ai/FlagGems/blob/ed2508bcb5a03000e9774734201d840ba362cd11/src/flag_gems/utils/triton_lang_helper.py#L72-L74) |
| Reduction / norm | 19、20、21 | 行级 FP32 reduction、Welford/mean/rsqrt、top-k reduction、hidden size 与寄存器 | [Task 19](tasks/batch-2/19-fused_rmsnorm.md)、[Task 20](tasks/batch-2/20-mamba_layernorm_gated.md)、[Task 21](tasks/batch-2/21-moe_sum_reduce.md)、[SGLang RMSNorm](https://github.com/sgl-project/sglang/blob/8014d9d062c3cc5d393596ecdf2f7009191965df/python/sglang/kernels/ops/elementwise/elementwise.py#L139-L188)、[SGLang MoE reduction](https://github.com/sgl-project/sglang/blob/8014d9d062c3cc5d393596ecdf2f7009191965df/python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py#L1163-L1249) |
| Scan / SSM / recurrence | 10、11、12、13、18 | prefix scan、chunk/packed-varlen 索引、指数衰减、状态递推、跨时间依赖 | [Task 10](tasks/batch-2/10-chunk_cumsum.md)、[Task 11](tasks/batch-2/11-chunk_local_cumsum_vector.md)、[Task 12](tasks/batch-2/12-chunk_state.md)、[Task 13](tasks/batch-2/13-chunk_state_varlen.md)、[Task 18](tasks/batch-2/18-fused_recurrent_gdn.md)、[SGLang FLA cumsum](https://github.com/sgl-project/sglang/blob/8014d9d062c3cc5d393596ecdf2f7009191965df/python/sglang/kernels/ops/attention/fla/cumsum.py#L81-L248) |
| Segmented gather | 17 | segment 边界、adapter routing、embedding gather、permutation、空 segment | [Task 17](tasks/batch-2/17-embedding_lora_a.md)、[SGLang embedding LoRA](https://github.com/sgl-project/sglang/blob/8014d9d062c3cc5d393596ecdf2f7009191965df/python/sglang/kernels/ops/gemm/embedding_lora_a.py) |
| Dense / grouped GEMM | 9、12、22、23 | `tl.dot`、M/N/K tiling、FP32 accumulator、ragged segment、permutation、base-output 累加 | [Task 9](tasks/batch-2/09-bmm_chunk.md)、[Task 12](tasks/batch-2/12-chunk_state.md)、[Task 22](tasks/batch-2/22-qkv_lora_b.md)、[Task 23](tasks/batch-2/23-sgemm_lora_b.md)、[SGLang QKV LoRA](https://github.com/sgl-project/sglang/blob/8014d9d062c3cc5d393596ecdf2f7009191965df/python/sglang/kernels/ops/gemm/qkv_lora_b.py) |
| Attention / paged KV | 14、15、16 | online softmax、causal/length mask、packed sequence、paged KV addressing、MHA/GQA head mapping | [Task 14](tasks/batch-2/14-context_attention.md)、[Task 15](tasks/batch-2/15-decode_attention.md)、[Task 16](tasks/batch-2/16-decode_grouped_attention.md)、[SGLang prefill attention](https://github.com/sgl-project/sglang/blob/8014d9d062c3cc5d393596ecdf2f7009191965df/python/sglang/kernels/ops/attention/prefill_attention.py)、[SGLang decode attention](https://github.com/sgl-project/sglang/blob/8014d9d062c3cc5d393596ecdf2f7009191965df/python/sglang/kernels/ops/attention/decode_attention.py) |

阅读上游源码时只提取四样：公开 wrapper 的参数语义、地址公式、kernel 数学、
测试 shape。不要整文件复制；赛题 reference 可能删参数、换返回顺序或改变输出
dtype，[第二批策略表](strategy-batch2.md#17-题上游映射与语义陷阱)已经列出
这些差异。

## 三、八芯兼容：需要知道什么

### 通用原则

1. **generic 先行**：首版只用 Triton core API、显式 mask、保守 block 和后端
   默认 launch 参数；不要预建八份文件。
2. **不写 GPU 专属假设**：不要把 warp 固定为 32，也不要假设所有后端支持
   相同的 `num_warps`、`num_stages`、libdevice、cache hint、shared memory 或
   最大 grid。
3. **FP32 语义显式化**：输入 cast、accumulator、特殊函数和输出 dtype 都按
   题面写清；fast math 必须重新跑极值和近零数据。
4. **平台结果才是证据**：比赛未公开具体型号、驱动、Triton 版本和隐藏 shape；
   本地 RTX 5070 Ti 只能筛选 NVIDIA 候选，不能代表八芯。
5. **vendor 后置**：只有某芯片 generic 编译失败、正确性失败、低于 `0.1x` 或
   明显拖累平均分时，才读它的后端并增加 `<operator>_<vendor>.py`。

### 每类芯片的最短材料入口

下表把“比赛实际可用的编译器证据”和“厂商背景资料”分开。FlagTree 链接固定
在 commit `c1ea8285`；厂商门户会更新，只用于理解术语，不能据此猜比赛 worker
型号。

| 比赛芯片 | 先看什么 | 只有平台暴露问题后再看 |
| --- | --- | --- |
| 天数智芯 / `_iluvatar` | [FlagTree CoreX backend](https://github.com/flagos-ai/FlagTree/blob/c1ea8285a06e97afad9dd2644bc71f2efca072f4/third_party/iluvatar/backend/compiler.py#L117-L209)：target、warp/wave 和编译选项如何落地 | [天数智芯官方开发者社区](https://developer.iluvatar.com/)的 SDK/工具文档 |
| 沐曦 / `_metax` | [FlagTree MetaX backend](https://github.com/flagos-ai/FlagTree/blob/c1ea8285a06e97afad9dd2644bc71f2efca072f4/third_party/metax/backend/compiler.py#L115-L167) | [沐曦官方 mcTriton 用户指南](https://developer.metax-tech.com/doc/214)和[文档中心](https://developer.metax-tech.com/doc?primary_category=%E7%BC%96%E7%A8%8B%E5%8F%82%E8%80%83) |
| 燧原 / `_enflame` | [FlagTree Enflame backend](https://github.com/flagos-ai/FlagTree/blob/c1ea8285a06e97afad9dd2644bc71f2efca072f4/third_party/enflame/backend/compiler.py#L101-L108) | [燧原官方 torch-gcu 固定源码](https://github.com/EnflameTechnology/torch-gcu/tree/f17a922ab48d82b4458b6c8c4c2dd8dc7a3fba5e)，用于确认 GCU device/runtime/profiler 术语 |
| 海光 / `_hygon` | [FlagTree HCU backend](https://github.com/flagos-ai/FlagTree/blob/c1ea8285a06e97afad9dd2644bc71f2efca072f4/third_party/hcu/backend/compiler_hcu.py) | [光合开发者社区 DTK](https://developer.sourcefind.cn/dtk)的 DCU 性能分析、调试和优化入口 |
| 昆仑芯 / `_kunlunxin` | [FlagTree XPU backend](https://github.com/flagos-ai/FlagTree/blob/c1ea8285a06e97afad9dd2644bc71f2efca072f4/third_party/xpu/backend/compiler.py#L96-L184)：先确认 GPU 风格 launch 参数是否有意义 | [昆仑芯官方技术页](https://www.kunlunxin.com/%E6%A0%B8%E5%BF%83%E6%8A%80%E6%9C%AF)用于理解 XPU/SDK 层次；具体比赛行为仍以 backend 和平台为准 |
| 华为 / `_ascend` | [Triton-Ascend Vector Operator 指南（固定 commit）](https://github.com/Ascend/triton-ascend/blob/865691e2e9b656bc58008170207b4108d92e8dd1/docs/en/programming_guide/vector_operator.md)：Vector Core、grid 和 program 内循环 | [Triton-Ascend 官方文档](https://triton-ascend.readthedocs.io/zh-cn/latest/)的迁移、调试调优、autotune 与典型算子 |
| 国际通用芯片 NVIDIA 路径 / `_nvidia` | [Triton NVIDIA tutorial 固定源码](https://github.com/triton-lang/triton/tree/dff2f7d03532e9ca0598c728c60c204ae7555fc9/python/tutorials)；本地 5070 Ti 仅作代理 | [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)和[Nsight Compute](https://docs.nvidia.com/nsight-compute/) |
| 国际通用芯片 AMD 路径 / `_amd` | [Triton AMD backend 固定源码](https://github.com/triton-lang/triton/blob/dff2f7d03532e9ca0598c728c60c204ae7555fc9/third_party/amd/backend/compiler.py#L68-L110) | [AMD HIP Performance Guidelines](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html) |

比赛页面没有公开国际通用 A/B 与 NVIDIA/AMD 的一一对应关系；这里只描述 ZIP
允许的两个 vendor 路径，不能据此给 A/B 命名。后缀规则见[比赛规范](README.md#4-zip-提交规范)。

FlagGems 是最值得搜索的跨芯参考库：官方项目目标是用 backend-neutral Triton
kernel 覆盖多种硬件，并在需要时提供 backend 特化；可先读固定 commit
[`ed2508b`](https://github.com/flagos-ai/FlagGems/tree/ed2508bcb5a03000e9774734201d840ba362cd11/src/flag_gems)
中的同类 generic op，再比较
[pointwise 后端配置](https://github.com/flagos-ai/FlagGems/blob/ed2508bcb5a03000e9774734201d840ba362cd11/src/flag_gems/utils/codegen_config_utils.py#L107-L172)。

## 四、最短实践顺序

### 路线 A：现在主攻 Task 24 `softcap_out`

1. **30 分钟**：读[题面](tasks/batch-2/24-softcap_out.md)，写 PyTorch
   reference 检查三 dtype、空张量、tail、极值、近零值、输出 FP32。
2. **45 分钟**：照 Vector Addition 的骨架写一个 flat 1D generic kernel；先不
   autotune、不写 vendor 文件。
3. **45 分钟**：在远程 NVIDIA 上跑正确性；失败先分为 layout/grid/dtype/
   formula 四类，不先调 BLOCK。
4. **30 分钟**：warmup 后比较 wrapper-inclusive latency；只测试少量
   `BLOCK_SIZE`，保存 p50/分位数，不把单次最快值当结论。
5. **15 分钟**：生成只含 `softcap_out.py` 的 ZIP，检查语法、manifest、大小和
   SHA-256。
6. **平台 S0 后**：八芯均正确且 `>=0.1x` 才进入性能优化；按逐芯结果最多选
   三个 vendor，每次只改一个轴。完整门槛见
   [Task 24 跨芯设计](../superpowers/specs/2026-08-23-softcap-out-cross-chip-optimization-design.md)。

### 路线 B：完成 Task 24 后扩题

按学习复用而不是题号扩展：

1. `softcap_out`：pointwise、mask、FP32 特殊函数。
2. `moe_sum_reduce`：在同一骨架上增加小 reduction。
3. `fused_rmsnorm`：增加一行一 program、平方和与 `rsqrt`。
4. 再二选一：
   - 熟悉 LoRA 分段与 GEMM：`embedding_lora_a` → `sgemm_lora_b` →
     `qkv_lora_b`；
   - 熟悉序列/SSM：`chunk_local_cumsum_vector` → `chunk_cumsum` →
     `chunk_state`。
5. 最后才做 attention 或长 recurrence；它们同时放大地址、数值、片上存储和
   跨芯 launch 差异。

## 五、每个候选都要留下的检查

### 正确性

- 函数名、参数顺序、默认值和返回数量与题面一致。
- shape、device、输出 dtype 正确，输入未被意外原地修改。
- FP16/BF16/FP32、空输入、最小输入、非整 BLOCK tail、最大公开 shape。
- 零、正负极值、NaN/Inf（题面表达式能产生时）和固定随机种子。
- reference 使用题面 PyTorch 表达式或固定第一方上游，不使用自己的 Triton
  实现作为 reference。

### 性能

- 首次 JIT 不计入；warmup 后多次计时并进行设备同步。
- 比较相同输入、相同输出分配边界；记录 p50 和抖动，不只记最佳值。
- 一次只改 BLOCK、grid、warps、数学 lowering 或数据布局中的一个。
- 性能结果绑定 git commit、源码/ZIP SHA-256、环境版本和逐芯片结果。

### 提交

- generic basename 与算子名完全一致；vendor 文件只用平台允许的后缀。
- ZIP 不超过 10 MB，除忽略项外只含 `.py`，UTF-8，可独立导入/编译。
- 实际路径运行 Triton/Triton-TLE kernel；没有设备分支、异常 fallback 或纯
  PyTorch 核心计算。
- 上传前保留回退包，默认建议预留两次最终回归额度；每次可能扣额度的点击都按
  [完整平台门禁](../../.agents/skills/flagos-operator-race/references/platform-workflow.md)
  核对实时提交 tuple；全部门禁通过后执行一次性自动提交，无需逐次确认。

## 六、可以暂时不学的内容

- CUDA C++ 语法、PTX/AMDGPU ISA：只有 profiler 已证明编译产物是瓶颈时再学。
- Triton-TLE 与厂商私有 intrinsic：generic 无法达到正确性/最低性能门槛，且
  官方 backend 资料给出明确路径后再引入。
- 全量 autotune：比赛每天提交额度有限，隐藏八芯环境也无法由本地搜索替代。
- 完整 SGLang server：本赛题以算子接口和隐藏 harness 评分；先能解释上游
  kernel 的调用语义即可。
- 八套本地 SDK：当前没有对应硬件时，阅读安装手册不会替代平台实测。

这条学习路线的退出条件很简单：能够独立完成一个 generic kernel 的契约表、
正确性矩阵、可靠 benchmark 和合规 ZIP。达到后应进入候选开发和平台只读预检，
完整门禁通过后自动使用一次额度获取八芯反馈，而不是继续读完所有材料。
