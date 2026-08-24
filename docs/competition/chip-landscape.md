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

> 调研时间：2026-08-24，来源为公开网络资料（厂商官网、开发者社区、第三方报道）。
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

1. **warp/wave 语义不统一**：NVIDIA 32；AMD/海光 wave64；昇腾无 warp 概念
   （Vector Core）；昆仑芯 launch 参数另有语义。generic kernel 不硬编码
   `num_warps` 的硬件假设。
2. **int64 支持**：燧原不原生支持 F64/I64，索引计算有隐式降位风险。
3. **带宽量级差 4 倍以上**（819GB/s ~ 3.2TB/s）：bandwidth-bound 算子
   （`moe_sum_reduce`、`fused_rmsnorm` 等）各芯最优 BLOCK/向量化策略可能
   完全不同，逐芯结果才是调参依据。
4. **架构同源分组**：海光（ROCm 系）≈ AMD 路径；天数/昆仑芯/沐曦为类 CUDA
   GPGPU；昇腾是唯一矩阵 Cube + Vector 的 NPU，需要最多的单独验证。

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
