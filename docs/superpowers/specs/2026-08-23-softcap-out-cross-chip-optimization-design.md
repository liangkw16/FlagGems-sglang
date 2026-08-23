# Task 24 `softcap_out` 跨芯优化设计

状态：已确认设计，等待书面规格复核
日期：2026-08-23
范围：FlagOS 第二届算子挑战赛第二批 Task 24

## 1. 决策摘要

采用“一个可移植通用实现 + 仅对有平台证据的瓶颈芯片增加 vendor
实现”的路线。

第一阶段只实现通用 `softcap_out.py`，固定稳定 FP32 数学路径和
`BLOCK_SIZE=256`，launch 参数使用各后端默认值，先获得八类芯片的正确性
和性能基线。基线通过后，按 S0 与公开最好成绩的差距选择最多三个 vendor
进入下一轮。当前公开数据提示沐曦、昆仑芯和华为优先级较高，但 S0 实测
可以覆盖这一先验，不预先排除其他芯片。

不自动上传比赛平台。每个待上传 ZIP 必须先完成本地检查，再由用户确认。

## 2. 已知约束

- 计算语义为 `tanh(fp32(x) / softcap_const) * softcap_const`。
- 输入 dtype 为 FP16、BF16 或 FP32；计算和输出必须为 FP32。
- FP32 容差为 `atol=1e-4, rtol=1e-4`；BF16 为 `1.5e-2`；FP16
  为 `1e-2`。
- 支持天数智芯、沐曦、燧原、海光、昆仑芯、华为及国际通用 A/B。
- 赛方未公开具体芯片型号、驱动版本、Triton 版本和隐藏 shape。
- 题面没有声明输入一定连续，也没有声明 `softcap_const` 的 Python/tensor
  表示形式或正负取值域；实现不得把这些假设当作题面事实。
- 通用 A/B 与 AMD/NVIDIA 的展示映射未公开；只能使用平台支持的
  `_amd.py`、`_nvidia.py` 文件路由，不能根据 A/B 名称推断厂商。
- 核心计算必须实际运行 Triton 或 Triton-TLE。不得使用设备判断、
  `try/except` 或执行失败后回退 PyTorch。
- ZIP 必须包含精确 basename `softcap_out.py`，可选 vendor 文件只能为
  `softcap_out_<vendor>.py`；ZIP 不超过 10 MB，除忽略项外只含 `.py`。
- 每队每天默认 15 次提交，两次提交至少间隔 120 秒；至少保留两次最终
  回归额度。
- 本地 GPU 主机实际为 RTX 5070 Ti 16 GB。它只能验证 NVIDIA 路径和
  筛选候选，不能代表八芯平台，也不能确定匿名 A/B 的映射。

题面和提交规则分别见：

- [`docs/competition/tasks/batch-2/24-softcap_out.md`](../../competition/tasks/batch-2/24-softcap_out.md)
- [`docs/competition/README.md`](../../competition/README.md)

## 3. 目标与成功标准

### 3.1 首要目标

1. 通用实现通过三种输入 dtype、数值边界和非整块尾部的正确性验证。
2. 第一次平台基线在全部八类芯片上正确，并且每类芯片加速比均不低于
   `0.1x`。
3. 所有后续实验都能关联到代码 commit、ZIP 哈希、单一变量和逐芯片结果。
4. 最终实现可直接进入官方仓库结构，不维护一份与 PR 实现分叉的比赛代码。

### 3.2 优化目标与可执行门槛

- S0 后，若某芯片未通过正确性/`0.1x` 门槛，或其得分比 2026-08-23
  公开最好成绩低至少 10%，它才有资格创建 vendor 候选。按潜在平均分贡献
  排序，每轮最多选三个。
- 新候选必须通过全部正确性检查。相对上一冠军提升至少 5% 时保留；提升
  低于 3% 时舍弃；处于 3%–5% 时只重复一次，重复仍达到 3% 才保留。
- fast math 候选除满足题面容差外，最大误差还必须不超过对应容差的一半。
- 同一优化轴连续两个候选均被舍弃时，停止该轴。

### 3.3 非目标

- 不为八类芯片预先创建八份实现。
- 不在提交代码中加入运行时 autotune、设备探测或动态 fallback。
- 不为实验记录建立数据库、服务或定制仪表盘。
- 不为本题引入 TLE、共享内存、矩阵流水或额外依赖。
- 不自动操作网页上传和最终比赛提交。

## 4. 实现边界与文件布局

通用实现的唯一代码源为：

```text
src/flaggems_sglang/ops/softcap_out.py
```

该模块保持自包含，只依赖 PyTorch、Triton 和 Triton language，带仓库要求的
Apache 2.0 header，并导出：

```text
__all__ = ["softcap_out"]
```

因此它既符合仓库现有算子注册规则，也能直接以 basename
`softcap_out.py` 放入比赛 ZIP，不再维护单独的 `submissions/` 代码副本。

最小验证文件为：

```text
tests/test_softcap_out.py
```

它只覆盖本题的关键数值和尾块风险，不引入新的测试框架或公共抽象。

经过平台验证后，必要的 vendor 实现才进入既有目录：

```text
src/flaggems_sglang/runtime/backend/_metax/ops/softcap_out.py
src/flaggems_sglang/runtime/backend/_kunlunxin/ops/softcap_out.py
src/flaggems_sglang/runtime/backend/_ascend/ops/softcap_out.py
src/flaggems_sglang/runtime/backend/_enflame/ops/softcap_out.py
```

仓库路径到 ZIP basename 的导出映射为：

| 仓库 vendor | ZIP basename |
| --- | --- |
| `_metax/ops/softcap_out.py` | `softcap_out_metax.py` |
| `_kunlunxin/ops/softcap_out.py` | `softcap_out_kunlunxin.py` |
| `_ascend/ops/softcap_out.py` | `softcap_out_ascend.py` |
| `_enflame/ops/softcap_out.py` | `softcap_out_enflame.py` |

没有平台收益证据的文件不创建。每个保留的 vendor 模块必须：

- 带 Apache 2.0 header；
- 定义与 generic 完全相同的 `softcap_out(x, softcap_const)`；
- 定义 `__all__ = ["softcap_out"]`，供现有 registrar 覆盖 generic；
- 是可独立打包的自包含文件，不依赖 ZIP 中不存在的仓库包路径或运行时设备
  分发逻辑。

测试源码同样带 Apache 2.0 header。

实验结果使用一个 Markdown 表记录：

```text
docs/competition/experiments/softcap_out.md
```

每行只记录时间、commit、ZIP SHA-256、每个 vendor 的唯一改动、逐芯片
正确性/加速比和保留或回退决策。

## 5. 通用实现设计

### 5.1 Wrapper

`softcap_out(x, softcap_const)`：

1. 连续输入直接使用；非连续输入先通过 `contiguous()` 物化，再执行同一个
   Triton kernel。该分支只处理布局，不替代核心计算。
2. 分配与原始 `x` 同 shape、同 device 的连续 FP32 输出，不修改输入。
3. 以物化后输入的 `numel()` 将任意逻辑 shape 展平为连续一维元素。
4. 空张量直接返回空的 FP32 输出，避免启动零 grid。
5. 非空张量启动单个 Triton kernel，并返回原 shape 输出。

`softcap_const` 接受 Python 实数或单元素 tensor。tensor 输入必须恰有一个
元素，由 wrapper 规范化为 Python 数值；其他 tensor shape 明确报错。cap 作为
运行时标量传入 kernel 并转换为 FP32，不使用 `tl.constexpr`，也不限制正负或
零值，从而保持参考表达式的取值域。

wrapper 不检查设备厂商，不捕获 kernel 异常，也不提供 PyTorch fallback。

### 5.2 Kernel

通用 kernel 采用标准一维 masked pointwise 形式：

1. `offset = program_id * BLOCK_SIZE + arange(0, BLOCK_SIZE)`。
2. 使用同一个 `offset < numel` mask 完成 load 和 store。
3. 输入显式转换为 FP32。
4. `softcap_const` 转换为 FP32 运行时标量。
5. 使用稳定公式：

   ```text
   z = fp32(x) / softcap_const
   y = softcap_const * (2 / (1 + exp(-2 * z)) - 1)
   ```

6. FP32 结果直接写入 FP32 输出。

首版固定：

```text
BLOCK_SIZE = 256
grid = ceil(numel / BLOCK_SIZE)
```

不显式传入 `num_warps`、`num_stages` 或 `num_ctas`，让 GPU、XPU 和 NPU
后端分别使用其合法默认值。kernel 不使用
libdevice、cache modifier、alignment hint、shared memory、TLE 或 autotune。

稳定公式采用 FlagGems 已有实现，避免 SGLang 正指数比值在大正输入上的
`inf/inf`。近零区域可能存在 `2*sigmoid-1` 的消减误差，因此它是正确性
测试的重点。

## 6. 固定源码事实与实验假设

赛方未公开 worker 型号和软件版本。下表把固定源码能证明的事实与需要平台
验证的假设分开；任何配置都不是比赛硬件参数结论。

| 芯片 | 固定源码事实 | 待平台验证的第一假设 |
| --- | --- | --- |
| 天数 | FlagTree `c1ea8285` 的 CoreX target 为 wave64；FlagGems `ed2508bc` pointwise policy 使用最大 tile 1024 | 若 S0 入选，先只改 BLOCK 256→512 |
| 沐曦 | FlagTree `c1ea8285` 的 MACA target 为 wave64；FlagGems `ed2508bc` policy 为 tile≤2048、最多 8 warps，而目标仓库当前 policy 上限为 16，说明版本口径不一致 | 若 S0 入选，先只改 BLOCK 256→1024，不把 policy 当硬件上限 |
| 燧原 | FlagGems `ed2508bc` policy 使用最多 12 CTA、grid-stride、4 warps；这只是软件策略，不是硬件 grid 上限 | 固定 BLOCK，仅比较 full-grid 与 12 CTA loop |
| 海光 | FlagTree `c1ea8285` 的 HCU gfx9 路径为 wave64，FP32 exp lowering 包含 exp2 路径 | 若 S0 入选，先只改 BLOCK 256→512 |
| 昆仑芯 | FlagTree `c1ea8285` 明确没有 GPU warp，GPU 式 warps/stages 不是调优轴；同一固定源码暴露 FP32 direct tanh | 固定 BLOCK，仅比较 core exp 与 direct tanh |
| 华为 | Triton-Ascend `865691e2` 指南要求 Vector pointwise 的 grid 接近物理 Vector Core 数，并让 program 内部循环 tiles | 固定 BLOCK，仅比较 full-grid 与物理核 grid-stride |
| NVIDIA | Triton `dff2f7d0` target 使用 warp32；本地 5070 Ti 只作代理 | 若 S0 入选，先只改 BLOCK，再单独改 warps |
| AMD | Triton `dff2f7d0` 会按 target 选择 wave32 或 wave64 | 若 S0 入选，只比较不依赖固定 wave 宽度的配置 |

通用实现不吸收任何厂商私有选项。vendor 文件若编译失败或收益不足，可直接
从 ZIP 中删除，由通用文件覆盖该芯片。

## 7. 平台实验流程

平台按文件名为各厂商独立分发实现。因此一次提交可以并行测试多个芯片，
但每个芯片相对其上一冠军只能改变一个变量。

### S0：通用基线

- ZIP 只包含 `softcap_out.py`。
- 稳定 core-exp 公式，BLOCK 256，后端默认 launch 参数，普通一维 grid。
- 目标是八芯正确且全部达到 `0.1x`。

若 S0 存在正确性失败，停止性能实验，先按 dtype、极值和尾块分类定位。

### S0 后的 vendor 选择门

从 S0 结果计算每个芯片相对 2026-08-23 公开最好成绩的差距。只有满足
3.2 节门槛的芯片进入 S1，按对八芯平均分的潜在贡献排序，每轮最多三个。
公开榜单提示沐曦、昆仑芯、华为可能优先，但真实 S0 结果拥有最终决定权。

### S1：识别架构瓶颈

以下是芯片入选时的首个单变量实验，不代表它们必然全部执行：

| Vendor | 相对该芯片上一冠军的唯一改变 |
| --- | --- |
| 沐曦 | BLOCK 256 改为 1024，其余不变 |
| 昆仑芯 | 稳定 exp 改为 XPU direct tanh，其余不变 |
| 华为 | full-grid 改为物理核数 grid + 内层循环，BLOCK 保持 256 |
| 燧原 | full-grid 改为最多 12 CTA + grid-stride，BLOCK 保持 256 |
| 天数/海光/NVIDIA/AMD | BLOCK 256 改为 512，其余不变 |

### S2：细化几何

- 沐曦：先保持默认 warps，只比较 BLOCK 1024 与 2048；若 2048 获胜，再保持
  BLOCK 2048，单独比较默认 warps 与 8 warps。
- 昆仑芯：冠军数学路径下比较 BLOCK 256 与 512。
- 华为：冠军 persistent 路径下比较 BLOCK 256 与 512。
- 燧原：冠军 grid 下比较 BLOCK 2048 与 4096。

### S3：数学 lowering

只在各芯片冠军几何上比较：

1. 稳定 `tl.exp`。
2. 显式 `exp2`。
3. 该 vendor 的 direct `tanh`。

`tl.sigmoid` 与显式 exp 若编译产物等价，不作为独立实验。fast tanh/exp
只有在常规路径已验证、完整数值集通过，且最大误差不超过题面容差的一半时
才可成为 vendor 候选；它永不进入通用实现。

### Final：回归与提交候选

1. 合并每个芯片最后一个正确冠军。
2. 对 ZIP 文件名、内容、大小、Python 语法和 SHA-256 做本地检查。
3. 保留至少两次平台额度：一次完整八芯回归，一次必要回退。
4. 将待上传 ZIP 路径、哈希和预期改动展示给用户。
5. 只有用户明确确认后才上传。

## 8. 正确性验证

### 8.1 数据类型和输出

- 输入分别覆盖 FP16、BF16 和 FP32。
- 输出必须始终为 FP32，shape 与输入完全一致。
- 输入必须保持不变。
- 至少各覆盖一个连续输入和一个非连续二维输入。
- `softcap_const` 分别覆盖 Python 数值、CPU 单元素 tensor 和设备单元素
  tensor；多元素 tensor 必须明确报错。

### 8.2 Shape 与尾块

最小长度集合：

```text
0, 1, 17, 63, 64, 65, 127, 128, 129,
255, 256, 257, 511, 512, 513, 1023, 1024, 1025
```

另覆盖二维 logits 风格 shape 和至少一个大 numel 性能 shape。所有非整块
长度必须验证 tail mask。

### 8.3 数值点

按归一化值 `z=x/cap` 覆盖：

```text
0, ±2^-24, ±2^-16, ±2^-10, ±0.125,
±1, ±4, ±10, ±44, ±50, ±88, ±100
```

cap 至少覆盖 `-30, -1, 0, 0.5, 1, 5, 30, 100`。cap 为零时直接与
PyTorch 表达式逐元素比较 NaN/Inf/有限值分类，不用普通相对误差掩盖差异。
另加入固定随机种子的正态和均匀输入。
若 PyTorch 参考在目标 dtype/device 上定义 NaN/Inf 行为，则记录 NaN mismatch；
性能候选不因 NaN/Inf 测试扩大正式输入域。

每个候选必须满足题面容差，且目标是把最大误差控制在对应容差约一半以内，
给不同后端 lowering 留出余量。

## 9. 性能验证

本地 5070 Ti 测试矩阵使用 FP16、BF16、FP32，numel 至少覆盖：

```text
4K, 64K, 1M, 16M
```

本地筛选统一使用 wrapper-inclusive 计时，即包含 FP32 输出分配、非空检查和
kernel launch，不包含首次 JIT 编译。每个 dtype/numel 组合先完成一次编译，
再执行五组 `triton.testing.do_bench`；每组使用 `warmup=25ms`、`rep=100ms`、
`quantiles=[0.2, 0.5, 0.8]`，由该工具处理设备同步和相同的缓存清理策略。
候选顺序在每组间轮换，记录五个 p50 的 median 及 p20/p80 范围。

本地候选只有在主要尺寸的聚合 median 提升至少 5%、任一主要尺寸回退不超过
3%，且所有 dtype 正确时才晋级平台。只在单一尺寸获胜的配置不晋级。

本地 NVIDIA 阶段要求检查 TTIR/TTGIR 和 PTX/SASS 中的 exp2/tanh 路径、
寄存器和 spill。其他 vendor 的 IR/汇编检查不是当前实施前置条件；只有后续
获得对应 runner、编译产物或平台诊断下载能力时才执行：

- 海光/AMD：AMDGPU 汇编、VGPR/SGPR 和 OCML/helper 调用。
- 华为：TTAdapter/Linalg、Vector Core grid、UB 使用和 scalar/vector 比例。
- 昆仑芯：TTXIR 的 vectorize、mask、loop-grid 和 EW table。
- 燧原：实际 arch、vector length、warp clamp 和 GCU lowering。

本地性能只用于筛选；最终性能结论以比赛平台逐芯片结果为准。

## 10. 失败分类与回退

| 现象 | 处理 |
| --- | --- |
| 八芯同类正确性失败 | 检查 FP32 cast/output、cap、grid、mask 和公式；不做 vendor 特化 |
| 仅非整块长度失败 | 修复公共 tail mask；不增加厂商分支 |
| 仅极值或 FP32 失败 | 比较常规 direct tanh；暂停 BLOCK 调优 |
| 单个 vendor 编译失败 | 删除私有 API/选项，回退该 vendor 上一正确冠军或通用实现 |
| vendor 配置低于 `0.1x` | 立即回退上一正确版本，先恢复有效排名资格 |
| 收益低于 3% | 舍弃候选；同轴连续两个候选如此则停止该轴 |
| 收益为 3%–5% | 仅重复一次，重复仍达到 3% 才保留 |
| 收益至少 5% | 正确性全部通过后保留 |
| 本地 NVIDIA 收益未在平台复现 | 舍弃候选，不据此推断 A/B |

任何回退都通过选择一个已验证源文件完成，不在运行时捕获错误或调用
PyTorch 参考算子。

## 11. 分阶段交付与实施计划边界

本规格描述完整决策协议，但下一份实施计划只覆盖可以在当前资源下连续完成的
Phase A：

1. 实现通用 `src/flaggems_sglang/ops/softcap_out.py`。
2. 添加一个最小正确性测试文件。
3. 在远程 RTX 5070 Ti 完成语法、正确性、边界和本地性能验证。
4. 生成并检查 S0 ZIP，写入实验 ledger，等待人工上传确认。

S0 平台结果是外部门槛。收到结果后进入 Phase B：按选择门为每个入选 vendor
分别创建一个小实施计划和单变量候选；没有 S0 数据时不提前实现 vendor 文件。

Phase C 是人工上传、最终八芯回归和额度治理，不属于代码实施。若方案入选评审，
最终官方 PR 另行规划，并继续使用同一代码源，不从比赛 ZIP 反向复制代码。

## 12. 参考依据

- [FlagGems 稳定 tanh fallback](https://github.com/flagos-ai/FlagGems/blob/ed2508bcb5a03000e9774734201d840ba362cd11/src/flag_gems/utils/triton_lang_helper.py#L72-L74)
- [FlagGems 各厂商 pointwise 配置](https://github.com/flagos-ai/FlagGems/blob/ed2508bcb5a03000e9774734201d840ba362cd11/src/flag_gems/utils/codegen_config_utils.py#L107-L172)
- [目标仓库 pointwise 配置](https://github.com/flagos-ai/FlagGems-sglang/blob/3946b9a6e489dce76c37a722c3846c2bba95afca/src/flaggems_sglang/utils/codegen_config_utils.py#L82-L131)
- [Triton `sigmoid` 实现](https://github.com/triton-lang/triton/blob/dff2f7d03532e9ca0598c728c60c204ae7555fc9/python/triton/language/standard.py#L46-L50)
- [SGLang softcap 上游实现](https://github.com/sgl-project/sglang/blob/8014d9d062c3cc5d393596ecdf2f7009191965df/python/sglang/kernels/ops/activation/softcap.py#L30-L68)
- [天数 CoreX backend](https://github.com/flagos-ai/FlagTree/blob/c1ea8285a06e97afad9dd2644bc71f2efca072f4/third_party/iluvatar/backend/compiler.py#L117-L209)
- [沐曦 backend](https://github.com/flagos-ai/FlagTree/blob/c1ea8285a06e97afad9dd2644bc71f2efca072f4/third_party/metax/backend/compiler.py#L115-L167)
- [燧原 backend](https://github.com/flagos-ai/FlagTree/blob/c1ea8285a06e97afad9dd2644bc71f2efca072f4/third_party/enflame/backend/compiler.py#L101-L108)
- [海光 HCU backend](https://github.com/flagos-ai/FlagTree/blob/c1ea8285a06e97afad9dd2644bc71f2efca072f4/third_party/hcu/backend/compiler_hcu.py)
- [昆仑芯 XPU backend](https://github.com/flagos-ai/FlagTree/blob/c1ea8285a06e97afad9dd2644bc71f2efca072f4/third_party/xpu/backend/compiler.py#L96-L184)
- [昇腾 Vector Operator 开发指南](https://github.com/Ascend/triton-ascend/blob/865691e2e9b656bc58008170207b4108d92e8dd1/docs/en/programming_guide/vector_operator.md)
- [Triton AMD backend](https://github.com/triton-lang/triton/blob/dff2f7d03532e9ca0598c728c60c204ae7555fc9/third_party/amd/backend/compiler.py#L68-L110)
