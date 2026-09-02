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

# 八芯片公开资料调研

> 调研时间：2026-08-24；平台实证更新至 2026-09-02。公开资料来源为厂商官网、
> 开发者社区和第三方报道。
> 用途：理解 `race-overview.json` 中八芯的架构差异，指导 generic Triton kernel
> 的可移植性决策和 vendor 特化方向。比赛平台未公开各芯具体型号、驱动和
> Triton 版本；本文所有规格均**不能**当作评测 worker 的事实，平台逐芯结果仍是
> 唯一证据。信源间参数冲突处已标注，采用时以厂商官网为准。

八芯构成：天数智芯、沐曦、燧原、海光、昆仑芯、华为昇腾六个实名厂商，加
两个匿名的国际通用芯片（`card_a`/`card_b`）。ZIP 提交规范只开放
`_nvidia` 与 `_amd` 两条 vendor 路径，A/B 与 NVIDIA/AMD 的一一对应关系
未公开，不得猜测。

## 1. 主力产品规格对比

| 厂商 / vendor 后缀 | 代表产品 | FP16/BF16 峰值 | 显存 / 带宽 | 软件栈 | Triton 路径 |
| --- | --- | --- | --- | --- | --- |
| 天数智芯 `_iluvatar` | 天垓150（BI-V150），上一代天垓100 | 192 TFLOPS（天垓100 为 147） | 64GB HBM2e，约 1.6TB/s（另说 896GB/s，口径不一） | 天数智算 / CoreX，类 CUDA 生态 | FlagTree iluvatar backend |
| 沐曦 `_metax` | 曦云 C500（2025 年有 C600） | 未完整公开（IPO 材料称单卡算力超 NVIDIA H20） | 64GB HBM2e；MetaXLink 支持 2/4 卡互连 | MACA / MXMACA | mcTriton（Triton 2.1.0 + MXMACA 后端） |
| 燧原 `_enflame` | 云燧 i20（邃思 DTU 2.0）；第三代 S60 为 6nm | TF32 128 TFLOPS，INT8 256 TOPS，FP32 32 TFLOPS | 16GB HBM2e，819GB/s（发布时推理卡最大带宽） | TopsRider（TopsRuntime / TopsAten / ECCL）+ torch-gcu | FlagTree enflame backend |
| 海光 `_hygon` | 深算二号 K100 / K100 AI | K100 AI 约 192 TFLOPS，INT8 392 TOPS | 64GB，896GB/s（AI 版约 1.2TB/s HBM2e） | DTK（DCU ToolKit），封装兼容 ROCm 生态 | ROCm 路径可复用 |
| 昆仑芯 `_kunlunxin` | P800（自研 XPU 架构，XRE runtime） | 345 TFLOPS | 64GB HBM2e（较新说法 128GB HBM3e），1.5~1.8TB/s 口径不一 | XRE + XCCL，深度绑定飞桨，兼容 CUDA 风格 | FlagTree xpu backend / Triton-TLE |
| 华为昇腾 `_ascend` | 910B；910C 为双 910B 合封 | 910B 约 320 TFLOPS / INT8 640 TOPS；910C 约 780~800 | 910B 64GB HBM2e 约 400GB/s；910C 96GB 约 3.2TB/s（口径不一） | CANN / Ascend C | triton-ascend（官方开放 AscendNPU IR） |

匿名国际通用芯片：NVIDIA 路径对应官方 CUDA/Triton 后端；AMD 路径对应
HIP/ROCm，wave64 语义（见 FlagTree AMD backend 固定源码）。

## 2. 各厂商要点

### 天数智芯（Iluvatar CoreX）

- 全自研通用 GPU 架构，自研指令集覆盖标量、矢量、张量运算，兼容 CUDA 生态
  和主流框架。
- 天垓100（BI-V100，2021）：7nm、240 亿晶体管、2.5D CoWoS，FP32 37
  TFLOPS，FP16 147 TFLOPS，32GB HBM2，PCIe Gen4 x16。
- 天垓150（BI-V150）：FP32 48 TFLOPS，FP16/BF16 192 TFLOPS，INT8 384
  TOPS，64GB HBM2e，350W，定位对标 A100。
- 推理卡智铠100（MR-V100）：32GB HBM2e，FP16 96 TFLOPS，INT8 最高
  384 TOPS。

### 沐曦（MetaX）

- 自研 GPGPU IP（曦云 C500 为 7nm；C600 基于 XCORE 1.5，144GB HBM3e）。
- C500：64GB HBM2e 全链路 ECC，MetaXLink 2/4 卡互连，最小 1% 颗粒度软切分
  虚拟化，350W。
- 软件栈 MACA / MXMACA；Triton 分支为 mcTriton，在 Triton 基础上增加
  MXMACA 后端（当前发布基于 Triton 2.1.0）。

### 燧原（Enflame）

- 芯片为邃思 DTU 系列（DTU 2.0 为 12nm；第三代 S60 为 6nm 工艺）。
- 云燧 i20（2021.12 发布）：FP32 32 TFLOPS，TF32 128 TFLOPS，INT8 256
  TOPS，16GB HBM2e 819GB/s，PCIe 4.0，150W，推理定位。
- 软件栈 TopsRider：驱动 + TopsRuntime + TopsAten 算子库 + ECCL 集合通信；
  PyTorch 经 PrivateUse1 的 torch-gcu 接入，支持 torch.compile/Inductor、
  AMP、Profiler、GCU Graph 与一键 CUDA 迁移。
- **数值坑**：torch-gcu 文档明确 GCU 不原生支持 F64/I64，会隐式降为 32 位
  （可用 `TORCH_GCU_ENABLE_INT64_AND_UINT64` 尝试开启）。算子题索引计算
  常用 int64，燧原路径需留意。

### 海光（Hygon DCU）

- 深算系列 GPGPU，"类 CUDA"通用并行架构。深算一号（2021）约为 A100 的
  40%+；深算二号 K100（2023）峰值约 100 TFLOPS、FP64 24.5 TFLOPS、64GB、
  896GB/s；K100 AI 版 FP16 192 TFLOPS、INT8 392 TOPS。
- DTK（DCU ToolKit）封装兼容 ROCm 生态组件，对标 CUDA 软件栈，算子覆盖
  99%+，支持 PyTorch/TF/飞桨，CUDA 迁移成本低。
- 对算子赛的推论：`_hygon` 与 AMD 路径同源（ROCm 系），wave64 等经验可
  相互迁移；做 vendor 特化时优先检查与 `_amd` 的行为一致性。

### 昆仑芯（Kunlunxin XPU）

- 自研第三代 XPU 架构（XPU-R），P800 为大模型训推主力卡。
- P800 公开口径：FP16/BF16 345 TFLOPS（约为 H20 的 2.3 倍、A100 的 1.1
  倍），64GB HBM2e（另有 96GB HBM3 / 128GB HBM3e 的新说法），带宽
  1.5~1.8TB/s（另有 4TB/s 说法），TDP 400W 上下，7nm。
- 软件栈 XRE（XPU Runtime Environment）+ XCCL，与飞桨深度协同；支持
  XPU/vXPU 算力切分；天池 256/512 超节点提供卡间全互联。
- 对算子赛：XPU 风格 launch/grid 参数与 GPU 不完全对应，FlagTree xpu
  backend 需确认 GPU 风格参数是否实际生效；本仓库 task21 `moe_sum_reduce`
  的 kunlun grid axes 修复即为此差异的实例。

### 华为昇腾（Ascend）

- 达芬奇架构 NPU，与 GPU 差异最大：AI Core = 3D Cube 矩阵单元（16×16×16，
  单周期 4096 MAC）+ Vector 向量单元 + Scalar 标量单元，配 L0A/L0B/L0C
  多级片上存储。
- 910B（Atlas 300T A2）：FP16 约 320 TFLOPS，INT8 640 TOPS，64GB HBM2e，
  带宽约 400GB/s；910C 双 die 合封后 FP16 约 780~800 TFLOPS、96GB HBM2e。
- 软件栈 CANN / Ascend C；官方 triton-ascend 已开放 AscendNPU IR 全面支持
  Triton，昇腾亲和优化宣称提升算子性能 20%+。
- 对算子赛：Triton 算子在昇腾由 Vector Core 执行，grid/program 内循环语义
  与 GPU SIMT 不同，无 32 线程 warp 概念；`num_warps` 等 launch 参数语义
  需以 Triton-Ascend 文档为准（learning-path 已固定 Vector Operator 指南
  链接）。

## 3. 统一 Triton 生态现状

- FlagTree：FlagOS 多芯片统一编译器，一个仓库接入 18 个 AI 芯片后端，截至
  2026 年初覆盖 12 家厂商近 20 款芯片（含昇腾、沐曦、海光、摩尔线程、寒武纪
  等非 GPU 架构）。
- Triton-TLE（Triton Language Extension）：FlagTree v0.5 语言扩展，三层
  DSL（TLE-Lite 面向算法工程师等），首批支持华为昇腾、清微智能、ARM AIPU。
- FlagGems 算子库已用 Triton 覆盖海光/沐曦/天数/昆仑芯/昇腾/摩尔等芯片；
  Qwen3.8 与 DeepSeek V4 的 Day0 多芯适配均经此路线完成。
- 比赛"generic Triton + 按需 vendor 特化"的要求与该生态设计一致。

## 4. 对跨芯 kernel 的直接影响

本节结论已由本地缓存的 backend 源码逐条核对，见
[`data/vendor-backends/`](data/vendor-backends/README.md)（固定 commit +
SHA-256 manifest）。以下引用的行号对应该缓存。

### 4.1 backend 源码可证的编译期事实

**warp 语义：默认值没有一个是 32，且昆仑根本没有 warp**

| 芯片 | `warp_size` | 证据 | launch 实际 block 线程 |
| --- | --- | --- | --- |
| 天数 `_iluvatar` | `64` | `iluvatar/compiler.py:122`、`driver.py:731` | `64*num_warps`（`driver.py:320` 硬编码） |
| 沐曦 `_metax` | `64` | `metax/compiler.py:120` | `64*num_warps`（`driver.py:267` 硬编码） |
| 海光 `_hygon` | `32 if gfx_major>=10 else 64` | `hygon/compiler_hcu.py:110` | `{warp_size}*num_warps`（`driver.py:536`） |
| 燧原 `_enflame` | `1`；gcu500 为 `128` | `enflame/compiler.py:474,535` | SIMT，gcu500 下 1 warp = 128 threads |
| 昆仑 `_kunlunxin` | `1`，注释 `we don't have warp` | `kunlunxin/driver.py:969` | 不适用 |
| NVIDIA `_nvidia` | `32` | `nvidia/compiler.py:113` | `32*num_warps` |
| AMD `_amd` | `32 if gfx_major>=10 else 64` | `amd/compiler.py:107` | wave64 由 `__oclc_wavefrontsize64` 控制 |
| 华为 `_ascend` | 无 warp 概念（Vector Core） | triton-ascend 指南 | 不适用 |

结论：generic kernel 绝不能假设 warp=32，也不能假设 `num_warps` 的语义在各芯
一致。

**`num_warps` 的有效性与上限差异极大**

- 昆仑：`num_warps: int = -1`，注释写明 *"invalid value, just to keep
  num_warps function signature"*；`__post_init__` 把传入的 `num_warps`、
  `num_ctas`、`num_stages` 全部记为 `invalid_params`
  （`kunlunxin/compiler.py:140,172`）。**在昆仑上调这三个参数是无效操作**，
  这解释了为什么 Task 21 昆仑最终生效的是 `BLOCK` 而不是 warps。
- 燧原：**编译期硬 assert**，不是软性建议。`make_ttir` 中 gcu400/gcu410
  `num_warps > 4` 直接 `assert False`，gcu300/gcu500 上限 8
  （`enflame/compiler.py:102-108`）。
- 昆仑真正可调的是 `cluster_num=12`（arch 3）、`core_num=64`、
  `buffer_size_limit=512`（`TRITONXPU_BUFFER_SIZE`），以及一批
  `TRITONXPU_*` 开关（`kunlunxin/compiler.py:105-129`）。

**昆仑物理并行度固定，与 grid 无关**

`kunlunxin/driver.py:41-53` 的 `get_xpu_spec` 对 arch 3 返回
`(12, 64)`，launch 时 `nclusters`/`ncores` 是**编译进 launcher 的常量**
（`driver.py:703-710`）；`gridX/Y/Z` 由 `LoopGrid` pass 在设备侧
`launch.cpp` 注入（`driver.py:554-556`）。因此昆仑上"加大 grid 提高并行度"
的 GPU 直觉不成立。

另注：`TTXPU_F_INTERLEAVE` 仅在 `metadata["grid"] == (12, 1, 1)` 时才开启
（`kunlunxin/compiler.py:255`）——grid 恰为 cluster 数时才走 interleave 优化。

**`tl.dot` 默认精度：天数与沐曦默认降 tf32**

| 芯片 | `default_dot_input_precision` |
| --- | --- |
| 天数 | `tf32`（`iluvatar/compiler.py:137`） |
| 沐曦 | `tf32`（`metax/compiler.py:130`） |
| 昆仑 | `tf32`（`kunlunxin/compiler.py:135`） |
| 燧原 / 海光 / AMD | `ieee` |

对 FP32 严容差的 GEMM 类题（Task 09/12/22/23），天数/沐曦/昆仑会**默认**用
tf32 输入精度，是隐藏的数值风险；需要 `ieee` 时必须在 `tl.dot` 显式指定
`input_precision`。注意昆仑的允许集合只有 `("ieee", "tf32")`，没有
`tf32x3`。

**片上存储上限差异**

燧原有显式 `OutOfResources` 判据（`enflame/compiler.py:303-323`）：
`max_dsm` gcu400/410 为 `896 KiB`、gcu500 为 `312 KiB`；gcu300 的
`max_shared = 8 MiB * num_warps`——**随 `num_warps` 线性变化**，因此在燧原上
减 warps 会同时压缩可用 shared。

**燧原 int64**：`enable_i64: bool = False` 默认关闭
（`enflame/compiler.py:491`），与 torch-gcu 文档的 F64/I64 降位说明一致。
索引计算走 int64 的算子在燧原路径有静默风险。

### 4.2 backend 源码不能证明的（必须花额度买）

**`grid` 展平上限 65535 在全部 8 份源码中都不存在。**
`grep -rn 65535` 只命中海光 driver 的无关 `gridX*gridY*gridZ == 0` 判断。
它属于 runtime/驱动层约束，只能由平台报错反推：

| 芯片 | 平台报错形态 | 本仓库证据 |
| --- | --- | --- |
| 昆仑 | `uni_sram PassManager::run` 编译期失败 | Task 21 S0c/S1/S2 |
| 华为 | `Invalid_Argument(EE1003) coreDim=114688` 启动失败 | Task 20 E2、Task 21 S0c |
| 燧原 | `grid.x` 超上限 | Task 24 S0（256512）、Task 08 S0c（2433024） |

这也解释了为什么昆仑假设矩阵需要 S0c→S1→S2→S3 四次提交才收敛：**结构约束
可静态读，规模约束只能实测。**

### 4.3 其余跨芯影响

1. **带宽量级差 4 倍以上**（819GB/s ~ 3.2TB/s）：bandwidth-bound 算子
   （`moe_sum_reduce`、`fused_rmsnorm` 等）各芯最优 BLOCK/向量化策略可能
   完全不同，逐芯结果才是调参依据。
2. **架构同源分组**：海光（ROCm 系）≈ AMD 路径；天数/沐曦为类 CUDA GPGPU
   （但 wave64 + 默认 tf32）；昆仑虽宣称兼容 CUDA 风格，实际 launch 模型
   最特殊（无 warp、固定 cluster/core）；昇腾是唯一矩阵 Cube + Vector 的
   NPU，需要最多的单独验证。
3. **华为官方建议与本仓库实测一致**：triton-ascend Vector Operator 指南原文
   为"关键不是创建尽可能多的 grid program，而是让 launch 接近物理 Vector
   Core 数"，并直接给出 `range(pid, num_blocks, num_core)` 写法。本仓库
   Task 08/20/21 三次验证成功的 capped grid-stride 即该官方范式。

### 4.4 昆仑正式平台成功样本与选型规则

本节只统计正式平台终态；MCP 生成、MCP HTTP 502、CUDA/NVIDIA 代理和重复载体
均不算昆仑实机证据。

| 任务 / 阶段 | source commit | submission | 昆仑 | 全题 | 已验证结构 |
| --- | --- | ---: | ---: | ---: | --- |
| [T28 E11](experiments/gate_up_lora_b.md) | `b40e5aa` | 7959 | **4.4045x** | 8/8 | framework route/materialize → 逐段规则 GEMM → inverse restore；主 GEMM 无 permutation、metadata 或间接行访问 |
| [T37 E5](experiments/sgemm_lora_a.md) | `6ce280b` | 7992 | **3.484x** | 8/8 | 先物化 routed rows/weights，每个非空 segment 一次规则 GEMM，再 inverse restore |
| [T24 S5](experiments/softcap_out.md) | `76551bc` | 5210 | `0.97591667x` | 8/8 | BLOCK4096 direct pointwise，4 warps/stages1，使用 XPU 官方 `tl_extra_shim.tanh` |
| [T40 E3](experiments/softcap_inplace_logits.md) | `248be7f` | 6612 | `0.9445x` | 8/8 | 去动态 grid-loop，连续且 grid≤65535 时使用无循环 direct kernel |
| [T19 S0](experiments/fused_rmsnorm.md) | `3fac516` | 页面未展示 ID | `0.94x` | 8/8 | 一行一个 RMSNorm program，`next_pow2(hidden)`，无 autotune/vendor 复杂度 |
| [T24 S2e](experiments/softcap_out.md) | `1a5ea26` | 4657 | `0.8637x` | 8/8 | 1D direct pointwise，BLOCK4096、4 warps、stages1，无动态 grid-loop |
| [T27 E8](experiments/fused_moe_router_tensorcore.md) | `140a632` | 7571 | `0.6756x` | 8/8 | split-K/partials/persistent 改为 32×32×64 stages1 direct GEMM；第二核做 softmax/top-2 |
| [T34 S0](experiments/per_token_quant_int8.md) | `0159b26` | 6344 | `0.607x` | 8/8 | 一行一 program 完成整行 amax、scale、round、clamp 和 int8 写回 |
| [T29 E2](experiments/gelu_and_mul.md) | `01e8113` | 5840 | `0.488x` | 8/8 | flat BLOCK2048 单 pass；用 A&S 纯算术近似替换会触发 worker 崩溃的 `tl.math.erf` |
| [T12 E2d](experiments/chunk_state.md) | `3d31481` | 4332 | `0.251x` | 8/8 | 规则 FP32 IEEE `tl.dot`，32×32、K32/64、4 warps、stages1 |
| [T39 E2](experiments/silu_and_mul_masked.md) | `f879895` | 6588 | `0.241x` | 8/8 | 删除 metadata gating，合法写满 padding；flat-full output-element grid-stride、BLOCK1024、int32 索引 |
| [T21 S3](experiments/moe_sum_reduce.md) | `1ca7dd2` | 4291 | `0.1754x` | 8/8 | 保留 2D program 语义但去 div/mod/循环；BLOCK256→1024，将总 program 数压到 65535 内 |

由这些样本得到的实现顺序：

1. 不规则路由、permutation 或 scatter 先在 framework/host 侧物化；核心 kernel
   只做规则连续计算，最后 inverse restore。T28/T37 是优先模板。
2. 简单算子优先单用途 direct kernel：无动态 device loop、受控 grid、少量固定
   launch 参数、stages1。不要为“统一结构”保留昆仑不需要的 split-K、persistent
   loop 或大中间态。
3. `tl.dot` 本身不是禁区；T12/T27/T28/T37 证明规则、静态、连续且显式
   FP32 IEEE 的 dot 可以通过。先消除间接访问和动态 metadata，再讨论换 FMA。
4. 对已隔离的不支持 lowering 直接换表达式；T29 已证明 `tl.math.erf` 是内容
   触发器，A&S 近似可恢复编译和正确性。
5. vendor 特化保持单变量并冻结其他七芯；BLOCK、grid 和拆阶段经验不得机械跨题
   迁移。T34 两趟列分块曾在昆仑数值失败，T19 multi-row 也不优于单行 S0。

负面边界：T38 E4/sub8160 与 E5/sub8170 都越过 1830 秒编译墙并在约 9–11 秒
执行完，但昆仑 9/9 correctness case 仍出现 80%–100% 错误及未初始化样式大值。
删除 dynamic `rank` 后错误形态不变，因此“host 多次发射同一 selector、跨 launch
反复修改 workspace”不是已验证模板；该轴已封存，不得把它包装成简单两阶段成功
经验。KernelGen MCP 仍可用于生成/优化假设，但 HTTP 502 只表示 verifier/worker
不可用，既不证明候选失败，也不能记作目标芯通过。

## 5. 主要来源

- 天数智芯：[天垓100 官方页](https://www.iluvatar.com/productDetails?fullCode=cpjs-yj-xlxl-tg100)、
  [天垓150 官方页](https://www.iluvatar.com/productDetails?fullCode=cpjs-yj-xlxl-tg150)、
  [模力方舟天垓150](https://moark.com/docs/compute/clusters_gpu/iluvatar/iluvatar_BI-V150_gpu)
- 沐曦：[曦云 C500 官方](https://www.metax-tech.com/prod.html?cid=107&id=21)、
  [mcTriton 用户指南](https://developer.metax-tech.com/api/client/document/preview/551/C500_mcTritonUserGuide_CN.html)、
  [IPO 问询报道](https://zhuanlan.zhihu.com/p/1945913942527448174)
- 燧原：[云燧 i20 发布](https://oceanpine.com/news/20220118451.html)、
  [21 经济报道](http://www.21jingji.com/article/20211207/herald/3ca406c23d4c0e7a8e30ab4d602c2b29.html)、
  [torch-gcu 固定源码](https://github.com/EnflameTechnology/torch-gcu/tree/f17a922ab48d82b4458b6c8c4c2dd8dc7a3fba5e)
- 海光：[K100 AI 参数](https://mirrorfrog.com/docs/cards/others/hygon-dcu-k100)、
  [DTK 介绍](https://zhuanlan.zhihu.com/p/705584420)、
  [国产 GPU 对比](https://www.eet-china.com/mp/a404728.html)
- 昆仑芯：[P800 解析](https://blog.csdn.net/Rong_Toa/article/details/151322568)、
  [P800 深度解析](https://mirrorfrog.com/en/blog/kunlun-p800-performance/)、
  [超节点发布](https://www.kunlunxin.com/news/4541.html)、
  [腾讯云 XRE 部署](https://cloud.tencent.com/document/product/1397/127039)、
  [2025 国产 AI 卡参数汇总](https://zhuanlan.zhihu.com/p/1977778068450997471)
- 昇腾：[官网](https://e.huawei.com/cn/products/computing/ascend)、
  [芯片综述](https://zhuanlan.zhihu.com/p/1913660152676094004)、
  [达芬奇架构](https://www.hiascend.com/developer/blog/details/0243195984483656011)、
  [开放 AscendNPU IR 支持 Triton](https://www.hiascend.com/developer/techArticles/20250529-1)
- FlagOS 生态：[FlagTree 架构拆解](https://zhuanlan.zhihu.com/p/2071199903762610097)、
  [FlagOS 2.0 发布](https://digital.gmw.cn/2026-03/30/content_38680080.htm)、
  [Triton-TLE 百科](https://baike.baidu.com/item/Triton-TLE/67554837)
