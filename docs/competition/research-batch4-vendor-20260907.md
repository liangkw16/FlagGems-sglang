# 第四批逐芯调研与下一轮尝试方案（2026-09-07）

本记录汇总本轮 SGLang、vLLM、FlagGems、FlagTree、FLA、厂商资料及已提交 PR 的调研，并与第四批 T42–T58 的历史实验对照。本文是**研究结论和候选计划**，没有新增 GPU 验证或平台提交；所有“下一步”均为目标运行时待验证假设，不改变各题实验账本的有效性、已关闭实验轴或提交状态。

## 依据与排序口径

- 本地状态基准：`dbe05d514c70b29d5ccfe85b3608956b7ac32116`；逐题以对应账本最新结果和 `team_best_speedup` 为准。个别 `platform` 行仍描述更早候选，不据此覆盖后续已落账结果。
- 公开榜单：2026-09-07 22:56:35 +08:00 获取的[第四批公开 API](https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks?batch_no=4)，见[榜单快照](data/batch4-leaderboard-20260907-225635.json)。存档仅给原始响应补末尾换行，文件 SHA256 `c5c9e0c861307764eb9f579b8d56c5ae605b4a7d1eaae201026ba50b5bb596f1`；原响应 SHA256 `3ba27c1c0e397eb74daa294edf8da31cc99535aaddc710ed3a53588505d2bc64`。这是时点记录，不是持续更新的榜单；匿名响应中的 `my_best_speedup=null` 不代表本队没有成绩。
- [排名规则](README.md)：八芯均正确且每芯加速比至少 0.1，才以八芯加速比的算术平均排名。未完成、正确性失败或低于门槛的结果不能作为有效均分。
- 排序是结合已有有效分、距榜首差距、剩余结构空间和修复难度的定性判断，不是统计概率。T47 第一表示守住既有第一的把握；T52 属于修复成功后才成立的条件机会。T51 账本记载当日额度已用尽，方案先作为储备；本文未刷新个人额度。

| 优先序 | 任务 / 算子 | 本队有效最好成绩 | 快照榜首 | 第一项尝试 |
| ---: | --- | ---: | ---: | --- |
| 1 | T47 chunked_sgmv_expand | 25.0048125，第一 | 25.0048125 | 守 E5；先审计多轮 K 步长，再考虑昆仑低秩外积 |
| 2 | T51 fla_layernorm_gated | 5.816325 | 6.668225 | 华为 gate 延迟加载，验证 UB 与活跃向量是否下降 |
| 3 | T43 causal_conv1d_update | 6.545875 | 7.90325 | 多 token 更新用寄存器滚动窗口复用权重和状态 |
| 4 | T57 log_scaling_tau | 2.50278125 | 2.81646875 | 先查生成代码；仅窄访存路径尝试 16B 向量化 |
| 5 | T52 fused_dual_residual_rmsnorm | 无，E6 六芯通过 | 5.425125 | 目标芯逐阶段找首个数值分歧 |
| 6 | T58 w8a8_block_int8_matmul | 122.66158333 | 576.316175 | 沐曦 GEMM 的 basic / cpasync 流水线对照 |
| 7 | T46 chunked_embedding_lora_a | 14.1051875 | 24.413875 | 华为按 segment × token tile 复用路由元数据 |
| 8 | T49 ernie45_rope_fused | 8.58253125 | 17.0628125 | 昆仑 pair 小 program 中调整 head 分组 |
| 9 | T48 chunked_sgmv_shrink | 4.7489375 | 23.743625 | 沐曦流水线；收益不足再研究燧原批量规则 GEMM |
| 10 | T45 chunk_scaled_dot_kkt | 无，昆仑 0.063 未达标 | 17.957125 | 昆仑固定 row/head 的标量地址 epilogue |
| 11 | T54 fused_norm_rope_stacked | 无，E2 五芯通过 | 10.22346875 | 拆开 norm / RoPE，保留 FP32 中间值定位失败 |
| 12 | T55 hc_head | 无，E2 七芯通过 | 6.3633125 | 昆仑候选与 reference 分离执行，定位崩溃阶段 |
| 13 | T50 extend_attention | 无，正确性和性能均有缺口 | 10.72328125 | 华为逐阶段数值诊断；再判断是否重开分段结构 |
| 14 | T44 chain_speculative_sampling | 无，八芯 predicts 失败 | 77.954125 | 对齐各后端 reference 的扫描和舍入次序 |
| 15 | T56 l2norm | 3.10772917 | 72.30680208 | 短 D 多行 tile，先保持数学表达式不变 |
| 16 | T42 act_and_mul | 3.25835 | 431.4843 | 查真实拷贝、带宽和短行利用率，有证据才重开 |
| 17 | T53 fused_gdn_gating | 2.1267 | 288.426175 | 多行共享 head 参数，保持已通过的 softplus 配方 |

## 芯片资料：本轮确认了什么

FlagTree 的五家后端统一固定在 `59052f82e7cc100444b5e941b3d3ed842f723e37`。这些是公开源码中的编译模型和分支条件，**不能据此认定比赛 worker 的卡型、SDK、后端版本或实际资源大小**；匿名 A/B 也不映射到猜测的芯片型号。

| 厂商 | 读到的具体证据 | 对下一轮方案的影响 |
| --- | --- | --- |
| 昆仑芯 | [XPU compiler][hw-xpu] 的 `num_warps/num_ctas` 是兼容签名占位；非 SDNN 路径将 `num_stages` 标为无效参数。存在 core tiling、buffer、vectorization 控制项 | 不把 warps/stages 扫描作为必然有效的调优。T45/T49 先改 program 数据形状和标量地址；编译开关须先确认目标版本及生成代码变化 |
| 燧原 | [GCU compiler][hw-enflame] 对 gcu300/400/410/500 使用不同流程；400/410 的 warps 上限为 4，500 的 `warp_size=128`；默认字段与架构覆盖值不同 | T43/T48/T50/T54 的 PassManager、间接寻址、向量化问题必须定位到架构和 pass，不能概括为硬件永久不支持。无需按 NVIDIA warp32 模型盲调 |
| 沐曦 | [MACA compiler][hw-metax] 中 `warp_size=64`，并有 `pipeline` 等选项。[官方 mcTriton 指南][metax-guide] 区分 basic、basic-prefetch、cpasync，预取可能增加寄存器占用 | T58/T48 优先做同数学路径的流水线对照，检查寄存器、shared memory 和实际 load；先核对运行时支持的语法，不能把不同 Triton 版本参数照搬 |
| 天数 | [Corex compiler][hw-iluvatar] 中 `warp_size=64`，并有架构相关编译流程与 FTZ 设置；[官方开发者门户](https://developer.iluvatar.com/)提供 SDK/工具入口 | T43/T51/T56 的行映射、归约和活跃寄存器预算按目标后端检查；代理 GPU 的 warps 最优值不直接外推 |
| 海光 | [HCU compiler_hcu.py][hw-hcu] 的精度默认仅 IEEE，但 `parse_options` 在 `gfx942` 分支加入 TF32；wave 大小随架构变化，并有 `waves_per_eu` 等控制项。[DTK 官方入口](https://developer.sourcefind.cn/dtk)用于版本核对 | **修正前轮“海光只能 IEEE”的过度概括**：既不能把默认值当所有分支能力，也不能照搬 AMD 的 TF32/TF32x3。T58 先查实际加载模块、arch 和最终 options，再决定精度实验 |
| 华为 | 固定提交 `865691e2e9b656bc58008170207b4108d92e8dd1` 的[开发指南][hw-ascend]和[profiling 指南][ascend-prof]建议按执行单元核数分工，检查 UB、MTE、Scalar 与流水停顿 | T51 先判断活跃中间量与 UB，T56 判断短行分核；T50/T54 分阶段定位。文档示例中的 192KB UB、48 Vector 核不是比赛实机参数 |

另已检索 NVIDIA/AMD 相关资料：[CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)可支撑带宽、占用率和访存分析；本轮 AMD 部分官方页面读取受限，不把未读内容当已验证依据。跨厂可复用的是实验方法，具体指令能力以运行时证据为准。

特别保留三条边界：公开源码出现一个参数，不等于当前版本会生效；某种间接 dot 写法失败，不等于芯片没有对应计算能力；小误差位于舍入边界，不等于换成“更高精度”一定匹配 reference。

## 上游与 PR 复用边界

本轮源码锚点：SGLang `b5c9b68f03c0552c94b717e200e7c2e6166c6741`；vLLM `70584f69b1bf866224604029fae038864a368019`；FlagGems `6e9839ecfb2d427a00a9dd39827ef52ea5cf264e`；FLA `9d981ffef3b361ba931102b633ae2a9fd91ca6c3`。下列 PR 状态为本轮查询时点，后续可能改变。

| PR / 状态 / 本轮 head | 可借鉴内容 | 不可直接外推的部分 |
| --- | --- | --- |
| [FlagGems-sglang #33](https://github.com/flagos-ai/FlagGems-sglang/pull/33)，merged；`3a938960b8167f10d82b5dc106aef79073d6fe01` | SiLU 融合，T42 的逐元素基线 | 单一 SiLU 不能覆盖全部 activation、stride 与 dtype 契约 |
| [#34](https://github.com/flagos-ai/FlagGems-sglang/pull/34)，merged；`e7f91a5f6c813d499275b3f3a6288e1b3b5dddc9` | causal_conv1d_fn 的燧原地址规范化 | T43 是有状态 update；E15 已出现编译失败，不能称迁移成功 |
| [#40](https://github.com/flagos-ai/FlagGems-sglang/pull/40)，merged；`a983226a8c23220620324ff8640d529231c0b489` | 昆仑 token/head-group/pair 小 program；T49 E3 已有本地迁移正证据 | ERNIE 的位置布局与普通 mRoPE 不同 |
| [#47](https://github.com/flagos-ai/FlagGems-sglang/pull/47)，open；`a316cd06e881490d0051c88ceb4ae62ec8a04264` | 昆仑 attention 的 QK/exp/sum/norm/PV 拆分 | 不是 T50 prefix/GQA 的已通过方案；中间显存可能平方增长 |
| [#49](https://github.com/flagos-ai/FlagGems-sglang/pull/49)，open；`0b0c9a20e2bae3334fd7bcd2b6a8ec6f5e46efda` | 昆仑实际调用的 FP32 串行外积，BM16/BN64 可作起点 | 燧原虽定义串行核，wrapper 实际启动 dot；不能说两芯串行外积均已证实 |
| [#50](https://github.com/flagos-ai/FlagGems-sglang/pull/50)，open；`987e6070a6b9435bf8cf19298468291e08277552` | grouped norm 的任务分工 | hidden4096/group128 专用分支不等于 T51 的普通 LN/RMS；通用多行方案本地已有回退结果 |
| [#51](https://github.com/flagos-ai/FlagGems-sglang/pull/51)，open；`27c8507d2584eee25c66ac0d22a2cd2b7f333166` | Ascend embedding LoRA 的常驻循环与元数据复用 | 固定 shape 对 rank/end 的假设不能带入 T46 动态段契约 |
| [FLA #1044](https://github.com/fla-org/flash-linear-attention/pull/1044)，merged；`3cde9602d30ff8252bc1b946973a09862b22a0dc` | Ascend 按物理核循环，weight/bias 外提，前向行 tile | 文中 2.5x 属于 backward，不是 T51 forward 的收益承诺；没有证明大 D 分列一定快 |
| [FLA #1002](https://github.com/fla-org/flash-linear-attention/pull/1002)，merged；`47e4c1ffc6cea10b4e6e9d512867545e78eb250c` | Ascend L2/Norm 实现，T56 行分工参考 | 仍需核对目标运行时与高维 stride |
| [vLLM #50278](https://github.com/vllm-project/vllm/pull/50278)，open；`093c3e146b6f853466a942e2c05765456bd575ea` | LoRA split-K 的吞吐与确定性取舍 | 原子归约的顺序不可视为确定；T48 若 split-K 应单独验证固定顺序归约 |
| [vLLM #47152](https://github.com/vllm-project/vllm/pull/47152)，closed，未 merged；`fc6b181e92ba07e08ac160717baf583ae35b1e53` | descriptor/分块寻址的组织思路 | Intel XPU 不是昆仑 XPU；原生 INT8 dot 不能搬进要求先转 FP32 的 T58 |

## 逐任务实验卡（按上述优先序）

### 1. T47 chunked_sgmv_expand：守住 E5，先清除多轮 K 的干扰项

- **功能与已有证据**：按 LoRA adapter/segment 把低秩结果展开并加到 base；[E5 账本](experiments/chunked_sgmv_expand.md)为八芯第一。SGLang 的 [expand][sg-expand] 和 PR #49 提供路由、规则 GEMM、外积参考。
- **先做什么**：先审计现有昆仑/燧原 GEMM 的多轮 K 地址递增。当前 `ops/chunked_sgmv_expand.py` 两 vendor 均以 `stride_bk` 初始化 K 地址，却用 `BLOCK_K * stride_bn` 递增；单轮路径不会执行第二次错误读取。现有 wrapper 以覆盖 rank 的 BLOCK_K 保持单轮，因此这是重开多轮方案前的静态疑点，不能继续把所有多轮错误归因于编译器。
- **随后唯一性能候选**：昆仑低秩 FP32 串行外积，合并 scaling/base 后处理；保持 E5 作为比较基准。先核对真实 wrapper 的 gather、host 同步和 scatter 占比，再决定外积是否有足够收益空间。
- **验证与止损**：覆盖 R 大于实验 BLOCK_K、非 2 次幂 rank、长 segment、多 slice、permutation、非零 base。题目非零 rank 用满 stored rank，不能搬入上游按 cur_rank 截断或提前低精度舍入的语义。只有完整 wrapper 稳定提升才考虑替换；守榜不因 kernel 局部提速冒险。

### 2. T51 fla_layernorm_gated：先缩短 gate 的活跃区间

- **功能与已有证据**：LayerNorm/RMSNorm 后带门控；[E7](experiments/fla_layernorm_gated.md)已达 5.816325，距榜首所需均分增幅约 14.6%。constexpr-D、逐芯配方已有成功经验，通用多行与沐曦 rsqrt 不再作为新轴。
- **第一候选**：以 E7 为底，仅在华为把 gate 加载后移到均值/方差归约之后，保持数学顺序，比较生成 IR、UB 占用和 Scalar/MTE 时间。源码顺序变化必须落实到生成代码才算形成实验。
- **条件分支**：若确有 UB/调度瓶颈，再借 [FLA norm][fla-norm] / PR #1044 做小 D 行 tile 2/4/8 与物理核循环；大 D 分列仅在资源不足或搬运问题有证据时尝试。保留海光已通过配方及其他强芯路径。
- **验证与止损**：覆盖 LN/RMS、gate 激活、weight/bias、小 batch/大 D；LN 不改成 `E[x²]-E[x]²`。IR 无变化或收益落在噪声内即停。账本记载当日额度为 0，当前只储备下一轮验证方案。

### 3. T43 causal_conv1d_update：从重复加载改为滚动状态

- **功能与已有证据**：短卷积逐 token 更新历史状态并输出；[E13](experiments/causal_conv1d_update.md)是有效最好成绩。E15 华为有约 41% 单芯改善记录，但燧原编译失败，组合不是有效新成绩。
- **第一候选**：参照 [SGLang update][sg-conv]，只改多 token 分支：width 2/3/4 的权重预加载，历史列放入寄存器滚动窗口，每个新 token 读一次；保留单 token 路径、加法顺序及最后 state 写回语义。
- **保留什么**：以 E13 燧原通过形态为底。华为 E15 的独立改动可另做单变量复验，不能与新滚动窗口同时混入一发。PR #34 的地址形态仅供燧原最小复现分析。
- **验证与止损**：同时校验输出和原地 state；覆盖序列 1、多步、长于卷积窗口、padding slot、state indices 和非连续 stride。寄存器增加造成占用率下降、仅单 case 更快或目标芯仍 PassManager 失败则停该候选。

### 4. T57 log_scaling_tau：先证明还有可改的访存

- **功能与已有证据**：读取已给定 tau 对行缩放，不能额外重算 log；[E11](experiments/log_scaling_tau.md)为 2.50278125。逐行 program、constexpr/i32、兼容 hook 的启动路径已做，燧原/昆仑约 0.55 的启动层瓶颈已记录。
- **第一步**：检查强芯生成代码是否已有宽 load/store。只有仍为窄访存时，才借 [SGLang row-scale][sg-scale-cpp] 的 BF16×8（16B）组织方式写 Triton 对齐分支；C++ 代码只借组织思想。
- **验证**：保留 FP32 tau 乘法后再 cast，覆盖对齐、尾部、不同 dtype 与 stride；测完整调用，包括 launch。不能缓存结果、返回输入别名或绕开正常调用语义。
- **止损**：如果已经是等效宽访存，本轮到代码证据为止；不再重复 BLOCK 扫描，也不把 12.5% 均分差距当作仍有 12.5% 可实现空间。

### 5. T52 fused_dual_residual_rmsnorm：先定位第一个舍入分歧

- **功能与已有证据**：两级 residual/RMSNorm 串联，中间 dtype 会影响第二级；[E6](experiments/fused_dual_residual_rmsnorm.md)昆仑仍 4/33,554,432 元素失配，RN 代理实验没有解决目标芯，燧原在账本时点仍 pending。此前 reciprocal/div/rsqrt/RN 试探不能重复包装。
- **第一步**：在同一目标 runtime 和失败输入上逐阶段比对：第一段平方和、rms、cast 前结果、cast 后中间值、第二段 residual、第二次归约和最终输出。诊断中间张量仅用于本地复现，候选仍遵守正式接口。
- **修复方向由证据决定**：若首差在归约，核对 reference 实际派发与归约顺序；若首差在中间 cast，保留题面指定的舍入位置。更精确的公式不保证更接近该 reference。
- **验证与止损**：复现 case18 及已知失败坐标后再修，扩展相同数值边界；无首分歧证据不重投。条件机会较高的原因是六芯已通过加速比合计 39.75153334；若保持它们不变，另两芯正确且均达标、合计超过 3.64946666，才可能超过快照榜首 5.425125。这是算术条件，不是有效成绩或修复成功预测。

### 6. T58 w8a8_block_int8_matmul：先用沐曦流水线检验大 GEMM 空间

- **功能与已有证据**：分组量化 scale 的矩阵乘；[E1](experiments/w8a8_block_int8_matmul.md)八芯 122.66158333。榜首约为本队 4.70 倍，单独把两个低分芯翻倍不足追平，需高分芯也有结构改善。
- **第一候选**：沐曦保持 M/N/K tile、IEEE 算法和 scale 次序，仅比较现有 basic 与目标版本支持的 cpasync；若异步路径不可用或资源不合适，再独立评估 basic-prefetch。核对实际生成的搬运、寄存器和 shared memory。
- **后续条件轴**：参考 [SGLang INT8 kernel][sg-int8] 的分组组织，在一个 scale 组内完成 FP32 累加后统一缩放，对照当前 64 子块路径。精度选项另开实验；海光须检查实际 arch/parse_options，不能硬塞 TF32x3。
- **验证与止损**：int8 输入必须先 cast FP32，禁止改为原生 INT8 dot。保持题面 FP32 scales，覆盖 K/N 尾部、group 边界、负数与消减；INT8 可被某种格式精确表示不等于累加和最终舍入已经正确。流水线无真实变化或 spill 抵消收益即停。

### 7. T46 chunked_embedding_lora_a：复用段级路由

- **功能与已有证据**：token/adapter 路由后查表产生低秩输出；[E3](experiments/chunked_embedding_lora_a.md)为 14.1051875，E6 天数预路由增益未能归因，原轴关闭。
- **第一候选**：只针对仍逐 token 查段的华为路径，试 segment × token tile（8/16 起步），同 tile 共享 adapter/rank/边界；若入口已实现同类复用，则先检查生成代码而非再造路由。参照 [SGLang embedding LoRA][sg-embed] 的元数据组织及 PR #51 的核内循环。
- **验证**：长段必须分多 tile，使用本题 `bs`/seg_indptr 契约；覆盖空段、零 rank、rank 尾部、重复 adapter、permutation 和未覆盖输出的置零。生产系统短段假设及固定 shape 跳过边界的做法不能移入。
- **止损**：元数据复用若被空 tile/padding 或 UB 成本抵消即停；不把旧天数预路由再换参数重开。

### 8. T49 ernie45_rope_fused：扩大 head 复用，保留小 pair 形态

- **功能与已有证据**：ERNIE45 特定位置布局的融合 RoPE；[E3](experiments/ernie45_rope_fused.md)已借 PR #40 的 pair 小 program 让昆仑达到 0.66425，八芯有效。
- **第一候选**：在昆仑保持一个旋转 pair 的程序结构，单独试 `BLOCK_HEADS=4/8/16`，让更多 head 共享一次位置和 cos/sin 加载；用实际 live buffer/编译报告约束上限，不扫描占位 warps 参数。
- **验证**：以 [vLLM ERNIE RoPE][vllm-ernie] 辅助核对 H/W 交替和剩余 T 部分，覆盖不同 Q/K head 数、非旋转尾部及 stride。若再合并相邻 pair，作为独立候选。
- **止损**：重新触发片上空间问题或生成程序变差即退回 E3；昆仑改善对全题均分的贡献仍需除以八，不能把单芯翻倍当作全题翻倍。

### 9. T48 chunked_sgmv_shrink：规则 GEMM 的流水与启动次数分开处理

- **功能与已有证据**：分段 LoRA 降维；[E7](experiments/chunked_sgmv_shrink.md)为 4.7489375，燧原约 0.528，原生低精度 dot 轴收益不足，逐段 wrapper 开销仍突出。
- **第一候选**：沐曦规则 GEMM 保持路由、tile 和运算次序，只做 basic/cpasync 流水线对照，避免同时修改 split-K。参考 [SGLang shrink][sg-shrink]、[vLLM shrink][vllm-shrink] 的 tile 分工。
- **结构储备**：若启动次数确为燧原主瓶颈，研究同 adapter 段批量 pack → 规则 batched GEMM → scatter，让 weight base 由 grid 决定，避开已失败的加载 ID 后间接 dot 形态。先计算 padding 和拷贝代价；这不是“间接 dot 硬件不支持”的结论。强芯仅在占用率不足时研究 FP32 scratch + 固定顺序 split-K。
- **验证与止损**：计时包含 pack、元数据、清零和 scatter；覆盖段长 63/64/65、256/257 等跨 tile 情况。本题无 lora_ranks，不能照搬其他题的 rank 跳过语义。无法摊薄启动或额外访存更重就停。

### 10. T45 chunk_scaled_dot_kkt：先让昆仑跨过有效门槛

- **功能与已有证据**：分块 K·Kᵀ 加缩放/门控/因果结构；[E15](experiments/chunk_scaled_dot_kkt.md)八芯正确，但昆仑 0.063，7.000125 仅是无效候选的诊断均值。GQA Gram 共享已做，E16 复合谓词/early-return 失败已回滚。
- **第一候选**：Gram 阶段不动，仅重排昆仑 epilogue：一个 program 固定 row 和 head，lane 沿列推进，减少向量 head 除模及重复 beta/g_m 加载；用简单尾 mask，完整写出应为零的上三角。可参考 [FlagGems KKT][gems-kkt] 的数学结构。
- **验证**：先对照 E15 做全部正确性，尤其尾 chunk、head 分组与上三角清零，再测完整 wrapper。core tiling/vectorization 开关只有生成代码提供证据后才单变量加入。
- **止损**：至少要从 0.063 提到 0.1（约 59% 单芯增益）才有效；0.12 可作工程余量目标。若没有足够结构收益，不再围绕已失败的复合 mask 小改。

### 11. T54 fused_norm_rope_stacked：用阶段边界定位，而非继续堆融合

- **功能与已有证据**：norm、RoPE、stack/copy 的组合；[E2](experiments/fused_norm_rope_stacked.md)五芯通过，燧原编译失败、昆仑数值失败、华为大规模 case 失败。
- **第一步**：目标芯逐阶段导出 norm 的平方和/inv、FP32 归一化 K、旋转结果和 V copy，找首个错误阶段。先用独立 FP32 norm buffer + T49 类 pair RoPE + V copy 的诊断形态，确定后再形成二/三 kernel 的正式候选。
- **验证**：K 保持 FP32 至旋转后才按题面 cast；非旋转尾部仍应保留对应 norm 结果，V 按契约复制。必须覆盖华为约 67M 元素 case、H=2/3/5/7、D=96、部分 rotary 和全部 mask 轴。
- **止损**：拆分仍在同一位置失配则转向 reference/后端取证；多 kernel 先证明正确性，不能预称更快。没有首分歧，不重投当前失败形态。

### 12. T55 hc_head：先判清昆仑崩溃属于哪里

- **功能与已有证据**：HC 通道相关的归约/门控和输出合成；[E2](experiments/hc_head.md)七芯通过，昆仑三核拆分已经做过，崩溃指纹由 Aborted 演化到验证执行段 Segfault，尚不能单凭阶段名称归因。
- **第一步**：用相同 worker/runtime 分离执行候选和 reference，记录首个崩溃栈；再逐一隔离现有三阶段，确认是编译、候选执行、reference 还是验证器。已有三核拆分不再作为“新方案”。
- **条件优化**：通过后再按 vendor 比较 S0/E1 的核外循环收益，检查海光活跃寄存器与 waves；组合前逐芯复测。[FlagGems hc_head][gems-hc] 只作计算组织参考，其 HC=2/4 或框架 fallback 不能覆盖本题 HC=8 的要求。
- **止损**：相同崩溃且无新增栈/阶段信息即停，不靠反复上传探测。正确性失败与平台/reference 故障必须保留不同归因。

### 13. T50 extend_attention：三个卡点分别设证据门

- **功能与已有证据**：已有 KV prefix 与新增 token 的 attention，含 GQA/因果边界；[当前账本](experiments/extend_attention.md)中华为、昆仑仍有正确性缺口，燧原仅 0.013。online 和 two-pass 都失败，只能关闭已试形态，不能证明数值问题永久不可解。
- **第一步**：华为固定失败输入，依次比较 QK、row max、exp、sum、PV，核对 dtype 与实际 dot lowering，找首差。参照 [SGLang extend attention][sg-attn] 的 prefix/extend 划分，而非只核对标准 self-attention。
- **条件结构**：昆仑从 PR #47 的规则 QK/softmax/PV 分段思想出发，但采用有界 query tile，先核算 scratch；燧原若间接 KV 寻址是主因，再研究按 KV head 打包复用。两者均需新的目标编译/性能证据才重开。
- **止损**：覆盖空 prefix、变长、GQA、因果边界和长序列；禁止无界 N² 中间显存。燧原需约 7.7 倍才能仅过 0.1，微调 tile 不足以支撑重投理由。

### 14. T44 chain_speculative_sampling：扫描语义优先于吞吐

- **功能与已有证据**：链式 speculative 接受/拒绝与末 token 采样，整数 predicts 必须严格正确；[账本](experiments/chain_speculative_sampling.md)记录八芯失败，9 月 7 日已修全接受路径 OOB，half 精确采样仍有缺口。
- **第一步**：固定各目标 torch/runtime，定位 reference 实际采用的归约和 scan 路径，再构造归一化/CDF 舍入边界的最小用例。对照 [SGLang rejection sampling][sg-reject] 的控制流，但不假设各后端、不同扫描维度都使用同一种归约树。
- **验证**：保留每一步必要的低精度舍入、严格 `>` 判定、零 residual、draft 含 NaN 的既定语义；覆盖全接受/拒绝位置、V=128k/152k 和 tau 紧贴 CDF 的边界。
- **止损**：任何已知 predicts 失败都阻止晋级；不要以容差正确的浮点 CDF 代替严格正确的采样结果，也不把已修 OOB 再列为未完成修复。

### 15. T56 l2norm：限定短行多行 tile 的收益范围

- **功能与已有证据**：末维平方和归一化；[E1](experiments/l2norm.md)已经修复高维 stride，3.10772917 对榜首 72.30680208 的差距约 23.3 倍。
- **第一候选**：参考 [vLLM L2Norm][vllm-l2] 的多行组织，只对短 D 试行 tile 8/16/32，先保持 sqrt/除法表达式不变，检查天数/沐曦 64-lane 编译布局的利用率；不同时改变数学配方。
- **条件轴与验证**：华为若调度过多，再研究物理核循环；覆盖高维非连续 stride、短 D 尾部、零向量和正常长 D。先确认计分 case 中短行占比足够。
- **止损**：短行改善无法带来整体均分变化就守 E1；不因通用实现有多行 tile 就承诺跨越 23 倍差距。

### 16. T42 act_and_mul：先看有无多余数据搬运

- **功能与已有证据**：门控激活与另一半输入相乘；[E6](experiments/act_and_mul.md)八芯 3.25835，榜首约为本队 132 倍，旧 tile/M1 轴关闭。
- **第一步**：逐 case 分离必要读写、额外 contiguous 拷贝和 kernel/launch 时间；若确认存在可消除拷贝，再以 stride-aware 融合替代；若短行欠填充，才尝试多行打包。参考 [SGLang elementwise][sg-element] 及 PR #33。
- **验证**：保持各 activation、gate/up 的 cast 与乘法顺序，覆盖非连续 stride、尾块、极值及全部题面 dtype。数学上更稳定的替代公式仍要按历史精度失败经验验证。
- **止损**：已达到合理读写下界且没有新的调度证据，就不新开性能候选；不能用几百分点局部收益推断具备冲第一条件。

### 17. T53 fused_gdn_gating：只研究行间参数复用

- **功能与已有证据**：GDN 门控中的 softplus、指数参数与逐 head 广播；[E4](experiments/fused_gdn_gating.md)有效 2.1267，榜首约为本队 136 倍，旧配方/小比例调参暂停。
- **第一候选**：在仍逐行重复加载参数的路径，试 rows tile 2/4/8 × heads，在一次调用内共享 A_log、bias 等参数以及可复用的 `exp(A_log)`；参考 [SGLang GDN gating][sg-gdn]。如果当前生成代码已等效复用，结束该轴。
- **验证**：保持已通过的 softplus/log1p/exp 顺序、beta/threshold 和 cast 点，覆盖非连续布局与广播；不缓存跨调用参数值或输出，输入变化必须即时反映。
- **止损**：同一数学路径下看不到寄存器/访存改善就不晋级；稳定公式重写已有数值风险，不再作为无证据的冲分手段。

## 执行与晋级纪律

1. 每题从账本最好有效版本或最近全正确版本出发；研究文档不直接变成新的提交轮次。先检查相关实验轴是否已做过，新增证据明确后才重开。
2. 每个候选只改变一个可归因的因素：数学顺序、路由结构、tile、流水线分开比较。保留源码 hash、目标设备/arch、torch/Triton/SDK 版本、编译选项、逐 case 正确性及完整 wrapper 耗时。
3. GPU 验证使用目标运行时；代理 GPU 只作筛选。远端验证按项目要求后台执行。先做失败 case 和必要边界，再做正式契约集，不能用一个随机种子或单 case 的改善替代完整验证。
4. 性能候选先确认改动超过测量噪声；可把目标芯稳定提升 10% 或八芯均分提升 2% 作为投入筛选线，**这不是比赛规则，也不是收益预测**。小于此线但有明确整体价值的候选，应写清成本和贡献再决定。
5. 无效题先过正确性及每芯 0.1；有效题计算真实八芯算术均分，不能只报弱芯百分比或代理峰值。任何低于门槛、已知失败、未结束回调都不能提前记为有效。
6. 优先投入 T51/T43 的具体结构验证，T58 做有依据的 GEMM 探索；T47 低成本守榜。T52/T55/T50/T44 先取证，T57/T56/T42/T53 必须先证明还有结构空间。后续是否提交依既有授权、门禁和实时额度执行，本记录本身不触发提交。

## 固定源码链接

[hw-xpu]: https://github.com/flagos-ai/FlagTree/blob/59052f82e7cc100444b5e941b3d3ed842f723e37/third_party/xpu/backend/compiler.py
[hw-enflame]: https://github.com/flagos-ai/FlagTree/blob/59052f82e7cc100444b5e941b3d3ed842f723e37/third_party/enflame/backend/compiler.py
[hw-metax]: https://github.com/flagos-ai/FlagTree/blob/59052f82e7cc100444b5e941b3d3ed842f723e37/third_party/metax/backend/compiler.py
[hw-iluvatar]: https://github.com/flagos-ai/FlagTree/blob/59052f82e7cc100444b5e941b3d3ed842f723e37/third_party/iluvatar/backend/compiler.py
[hw-hcu]: https://github.com/flagos-ai/FlagTree/blob/59052f82e7cc100444b5e941b3d3ed842f723e37/third_party/hcu/backend/compiler_hcu.py
[metax-guide]: https://developer.metax-tech.com/doc/214
[hw-ascend]: https://github.com/Ascend/triton-ascend/blob/865691e2e9b656bc58008170207b4108d92e8dd1/docs/zh/programming_guide/index.md
[ascend-prof]: https://github.com/Ascend/triton-ascend/blob/865691e2e9b656bc58008170207b4108d92e8dd1/docs/zh/debug_guide/profiling.md
[sg-expand]: https://github.com/sgl-project/sglang/blob/b5c9b68f03c0552c94b717e200e7c2e6166c6741/python/sglang/kernels/ops/gemm/chunked_sgmv_expand.py
[sg-conv]: https://github.com/sgl-project/sglang/blob/b5c9b68f03c0552c94b717e200e7c2e6166c6741/python/sglang/kernels/ops/mamba/causal_conv1d_triton.py
[sg-scale-cpp]: https://github.com/sgl-project/sglang/blob/b5c9b68f03c0552c94b717e200e7c2e6166c6741/python/sglang/kernels/jit/csrc/inkling/inkling_row_scale.cuh
[sg-int8]: https://github.com/sgl-project/sglang/blob/b5c9b68f03c0552c94b717e200e7c2e6166c6741/python/sglang/kernels/ops/quantization/int8_kernel.py
[sg-embed]: https://github.com/sgl-project/sglang/blob/b5c9b68f03c0552c94b717e200e7c2e6166c6741/python/sglang/kernels/ops/gemm/chunked_embedding_lora_a.py
[sg-shrink]: https://github.com/sgl-project/sglang/blob/b5c9b68f03c0552c94b717e200e7c2e6166c6741/python/sglang/kernels/ops/gemm/chunked_sgmv_shrink.py
[sg-attn]: https://github.com/sgl-project/sglang/blob/b5c9b68f03c0552c94b717e200e7c2e6166c6741/python/sglang/kernels/ops/attention/extend_attention.py
[sg-reject]: https://github.com/sgl-project/sglang/blob/b5c9b68f03c0552c94b717e200e7c2e6166c6741/python/sglang/kernels/ops/speculative/reject_sampling.py
[sg-element]: https://github.com/sgl-project/sglang/blob/b5c9b68f03c0552c94b717e200e7c2e6166c6741/python/sglang/kernels/ops/elementwise/elementwise.py
[sg-gdn]: https://github.com/sgl-project/sglang/blob/b5c9b68f03c0552c94b717e200e7c2e6166c6741/python/sglang/kernels/ops/attention/fla/fused_gdn_gating.py
[vllm-ernie]: https://github.com/vllm-project/vllm/blob/70584f69b1bf866224604029fae038864a368019/vllm/model_executor/layers/rotary_embedding/ernie45_vl_rope.py
[vllm-shrink]: https://github.com/vllm-project/vllm/blob/70584f69b1bf866224604029fae038864a368019/vllm/lora/ops/triton_ops/lora_shrink_op.py
[vllm-l2]: https://github.com/vllm-project/vllm/blob/70584f69b1bf866224604029fae038864a368019/vllm/third_party/flash_linear_attention/ops/l2norm.py
[gems-kkt]: https://github.com/flagos-ai/FlagGems/blob/6e9839ecfb2d427a00a9dd39827ef52ea5cf264e/src/flag_gems/fused/FLA/chunk_scaled_dot_kkt.py
[gems-hc]: https://github.com/flagos-ai/FlagGems/blob/6e9839ecfb2d427a00a9dd39827ef52ea5cf264e/src/flag_gems/fused/mhc/hc_head_fused_kernel.py
[fla-norm]: https://github.com/fla-org/flash-linear-attention/blob/9d981ffef3b361ba931102b633ae2a9fd91ca6c3/fla/modules/layernorm_gated.py
