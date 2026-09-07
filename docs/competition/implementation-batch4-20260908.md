# 第四批方案实现与提交前验证（2026-09-08）

本轮完成 **11 题的候选源码改动、5 题的诊断工具落地和 T57 的生成指令检查**。17 题均运行了绑定 Git 字节的 release 验证：**14 题通过当前 NVIDIA/代理范围门禁，112 个测试方法、1,015 次实际 kernel 调用；T44、T48、T58 被门禁拦截**。这不是八芯通过，也不改变任何历史平台成绩。

9 份候选 ZIP 已生成并验签，仅作为可复核产物；**未执行平台 preflight、上传或正式提交**。其中部分候选没有性能收益证据，另一些仍需要目标芯片验证，不能统一标成“可以冲榜提交”。逐题结论见下表，完整提交、文件、日志和 ZIP 哈希在[证据清单](data/batch4-implementation-20260908.json)。

## 实现和验证范围

- 调研依据：[09-07 逐芯方案](research-batch4-vendor-20260907.md)。比较基线为本轮改动前的 `2acbdc054bedd0d72b9342efdeea86e7da393e32`，不把它称作每题历史 team-best 的统一源码版本。
- 源码提交：`8ba31a102f4ef0430c08f12c4622b27430915071`（T43/T47/T51）；`26a95766b179d263916e9483dfc8d2343c40406a`（第二组 8 题）；`c73f6c3f83ec38d5a2c40cfdef996e64e50ecd67`（T53/T56 小规模分流、T54 测试、诊断和配对计时工具）。已推送 fork 的 `research/season2-batch2`。
- 唯一实际执行设备：NVIDIA GeForce RTX 5070 Ti，Python 3.12.13、torch 2.13.0+cu130、Triton 3.7.1、CUDA 13.0。GPU 工作均在已授权 `gpu` 远端后台、串行、有超时限制地运行。
- 本轮复查 KernelGen 当前 schema：没有可直接绑定现有 Triton 源码的执行入口。未把生成/调优服务的结果冒充固定源码验证；没有独立可用的华为、昆仑、沐曦等目标 runtime。所有未执行及目标芯未验证源码逐项保存在回执。
- 静态检查：Black、isort、flake8、py_compile、`git diff --check` 通过。release 回执经 `verify_receipt` 按 Git blob、选定源码、完整必需测试、实际 kernel 执行、环境及日志哈希复核。
- ZIP 由对应 source commit 生成；构建前 manifest、构建结果、`--verify-existing` 的成员 SHA256、source commit 和整个 ZIP SHA256 完全一致。没有为失败题生成本轮 ZIP。

下表“测试/调用”指测试方法数 / 实际 kernel 调用数；代理 vendor 执行不等于该芯片验证。顺序沿用调研时的优先级，不拿代理加速比重新估计八芯排名。

| 任务 | 本轮实现或诊断 | 测试/调用 | 判定和下一步 |
| --- | --- | ---: | --- |
| T47 chunked_sgmv_expand | 昆仑/燧原多轮 K 的 B 指针由 `stride_bn` 修为 `stride_bk`；BK 上限 128；新增直接多轮与 rank 127/128/129/511/512/513 回归 | 12/105，通过 | 修复已证实的寻址错误。ZIP 已验签；目标芯复验后再考虑低秩外积，保留 E5 成绩 |
| T51 fla_layernorm_gated | 华为 gate load 移到归约和 affine 后，数学顺序不变 | 7/147，通过 | NVIDIA PTX 加载顺序确有变化，shared memory 未降，计时约 0.994–1.002x。保留候选，等待华为 UB/IR 和耗时证据 |
| T43 causal_conv1d_update | width 2/3/4、多 token 使用权重预加载和寄存器滚动历史；补长序列、stride、输入不变检查 | 11/96，通过 | 配对约 0.974–1.000x，没有稳定收益，暂不晋级。单 token 与最终返回状态语义保留 |
| T57 log_scaling_tau | 导出现有 FP32/BF16 PTX；未新增生产分支 | 5/19，通过 | 两种 dtype 均已有 `ld/st.global.v4.b32`（16B）；结束本轮宽访存轴，保留 E11 |
| T52 fused_dual_residual_rmsnorm | 原有 RN 候选保持；实现归约、RMS、cast、mid、out 的选定位置打印及独立 reference 重放 | 6/20，通过代理 | 4096×8192 BF16 输入代理容差通过、插桩无最终输出变化；仍未复现昆仑平台 4 元素错误，禁止据此宣布修复 |
| T58 w8a8_block_int8_matmul | 新增沐曦同数学路径 `pipeline="cpasync"` 候选；修正测试原先误用 int64，覆盖真正 int8 和 K/N 尾部 | 6/23，失败门禁 | NVIDIA 不识别 `pipeline`，13 条 error 均在该选项；需要沐曦实际参数解析/编译证据，未打 ZIP |
| T46 chunked_embedding_lora_a | 华为 segment 内 token tile=8，共享路由；空段先判空；rank 分 tile，补齐 >128 的输出 | 12/49，通过 | 代理长段约 2.70x、短段约 0.98x；需华为性能和编译验证，ZIP 已验签 |
| T49 ernie45_rope_fused | 昆仑保留单 pair 结构，head tile 8→16；补 head 边界、不同 Q/K head 和尾部 | 6/204，通过 | 代理 256 token/32 heads 约 1.15x，小案例约持平；需昆仑资源和计时证据 |
| T48 chunked_sgmv_shrink | 新增沐曦 cpasync 候选；补段长 63/64/65/256/257、三种 dtype | 6/18，失败门禁 | NVIDIA 不识别 `pipeline`，7 条 error；保留失败记录，未打 ZIP，不把参数删除后运行算流水线验证 |
| T45 chunk_scaled_dot_kkt | 昆仑 Gram 不动；epilogue 固定 row/head，beta 与 g_m 标量复用，列向量简单尾 mask | 10/47，通过 | 代理约 0.96x，无提速证据。还不能证明昆仑从 0.063 跨过 0.1；不晋级 |
| T54 fused_norm_rope_stacked | 导出 norm FP32、旋转和 V；修正测试 cache 宽度为 rotary_dim，增加 67M 输入 stress 和必需测试清单 | 8/27，通过代理 | 华为源码在大规模代理重放容差通过，V bitwise 相等，插桩不改最终输出。真实失败输入/runtime 缺失，未形成新的生产修复 |
| T55 hc_head | 原有三核保留；候选/reference 独立进程；逐核完成日志与 sumsq/mix/output 文件 | 2/22，通过代理 | 昆仑三阶段源码在代理完成，HC=8 诊断无崩溃；不能归因或关闭目标 Segfault，需要相同 worker/runtime 重放 |
| T50 extend_attention | 华为 QK、max、exp、sum、PV 诊断副本；候选/reference/trace 分进程、同输入 | 4/42，通过代理 | 插桩改变 818/6144 个最终 FP32 元素的位值；该 trace 有观察效应，不能当目标首分歧证据，保留诊断而不改公式重投 |
| T44 chain_speculative_sampling | 固定 BF16 输入、输出/阶段日志；严格整数比较，保留既有失败测试 | 9/110，失败门禁 | 复现 predicts[13] 不同，accept_index/accept_token_num 相同；release 因 expected_failure 被拒，未打 ZIP |
| T56 l2norm | 多行 tile=8 仅用于 D≤128 且 rows>4096；小规模沿用旧核；补分流边界 | 8/128，通过 | 最终 65536×64/128 约 4.97x/3.56x；小案例约持平，形成有代理性能证据的候选 |
| T42 act_and_mul | 移除强制 contiguous，融合核读取真实行/列 stride，保持激活/cast 顺序 | 14/84，通过 | 非连续输入约 1.34–1.62x，连续案例约 0.96–0.97x；价值取决于计分布局分布，不能推算整体倍数 |
| T53 fused_gdn_gating | 多行共享 head 参数/exp；非连续参数按 stride；总元素<16384 且参数连续时沿用旧核 | 7/25，通过 | 最终 128×128/256×128/4096×128 约 1.19x/2.27x/33.51x；小案例约 0.98x，已消除初版约18%的回退 |

## 关键回归和测量证据

**T47 的失败先于修复。** 旧 `2acbdc0` 配新测试时，两 vendor 的直接 rank=65、BLOCK_K=32 测试各有 51/51 元素错误，最大绝对误差 65.6481；大 rank 还分别请求 262144/131072 字节 shared memory，超过代理上限 101376。修复后的提交字节完整通过。因而不能再把所有多轮 K 错误归为后端编译器问题。

**性能计时包含 wrapper。** [配对脚本](../../tools/benchmark_batch4_candidates.py)先编译、检查输出，再交替执行 6 组 AB/BA；每次测量运行 30 次完整 wrapper，前后同步，以 wall clock 记录，保留每组原始微秒值及配对比值。包含分配、contiguous、路由和全部 launch；没有把 JIT 编译时间计入加速。配对计时使用 BF16 输入（T57 指令检查覆盖 FP32/BF16），不是全形状、全 dtype 的性能基准。所有厂商源码的这些数字都来自 NVIDIA，只能用于筛选。

首轮 `perf2` 检查 8 个改动题以及 T57 指令；`perf3` 扩大短行归约负载并复验小规模保护；`perf4` 定位 T53 crossover；`perf5` 使用最终 `c73f6c3` 字节复验 T53/T56。早期 `perf1` 因编译 metadata 的 GPUTarget JSON 序列化失败，没有用于性能结论。最后两题的结果以 `perf5` 为准。

T51 的 NVIDIA 指令可见 gate 加载后移，但这不证明华为 UB 已下降。T43 的新源码没有在本轮完整调用计时中兑现收益；T45 也没有达到推进目标芯提交的性能证据。三者已形成可审查候选，暂不晋级。

## 五题诊断如何重放

[诊断工具](../../tools/diagnose_batch4.py)复用现有测试 reference 和输入工厂，将输入保存为 CPU 张量文件，记录 SHA256；`reference`、`candidate`、`trace` 分进程运行。候选阶段记录 kernel 的编译/启动开始及同步完成，启用 faulthandler；HC 直接保存已有 scratch，其他题在独立源码副本中插入选定坐标的 FP32 位值打印。提交源码不含这些打印或 reference fallback。

```bash
python tools/diagnose_batch4.py fused_dual_residual_rmsnorm make input.pt --large
python tools/diagnose_batch4.py fused_dual_residual_rmsnorm reference input.pt --out reference
python tools/diagnose_batch4.py fused_dual_residual_rmsnorm candidate input.pt --source src/flaggems_sglang/runtime/backend/_kunlunxin/ops/fused_dual_residual_rmsnorm.py --out candidate
python tools/diagnose_batch4.py fused_dual_residual_rmsnorm trace input.pt --source src/flaggems_sglang/runtime/backend/_kunlunxin/ops/fused_dual_residual_rmsnorm.py --row 3969 --col 4831 --out trace
```

这些 GPU 命令按项目要求放在授权远端的限时后台队列运行。目标环境可用 `--device` 和 `--import-runtime` 指定实际已安装后端；本轮只实际测试了 CUDA。输入生成仍复用 CUDA 测试工厂，目标重放读取同一保存文件。工具不假设该工厂的随机输入就是平台原始失败输入。

- T52 大诊断：4096×8192，最终 out 与 reference 有 254 个 bitwise 差异、mid 有 75 个，**均在题面容差内**；插桩与原候选最终输出 bitwise 相同。坐标 (3969,4831) 仅用于探针演练，没有把代理随机输入伪称平台 case18。
- T54 大诊断：输入 2048×8×4096，共 67,108,864 元素，是同量级 stress；不是已拿到的华为失败 case。K 有 427 个 bitwise 差异但容差通过，V 完全相同；插桩不改变最终输出。
- T55：HC=8、hidden=2048，三个原有 kernel 均有完成记录和阶段输出，reference 独立完成；目标崩溃归因仍未解决。
- T50：代理完整测试通过，但插桩改变结果位值，必须先处理观察效应或在目标调试器中取证；不从当前 trace 推出归约、exp 或 PV 是平台根因。
- T44：BF16 seed=1 的 predicts[13] 仍严格失败，其他两项整数输出一致，插桩未改变结果。保留 xfail 的历史标记，release 门禁明确拒绝 expected_failure；没有放宽到浮点容差。

本轮已实现可重放的取证手段。T52/T54/T55/T50 仍缺目标运行时的原失败重放，因此没有编造“根因已修复”的生产改动；T44 的扫描顺序修复也尚未完成。

## 产物与后续入口

原始回执、完整日志、PTX/IR、配对结果、探针日志和 manifest 在：
`artifacts/competition/batch4-implementation-20260907/`。目录沿用启动时日期，内容包含跨日结果。大型诊断输入、中间张量和输出保存在已授权远端本轮临时目录 `/tmp/flagos-batch4.Mb6c6k/diagnostics{1,2}/`；本地保存它们的身份、摘要及日志，未把大型张量加入 Git。该临时目录不是长期存储保证。

[结构化证据清单](data/batch4-implementation-20260908.json)列出 17 题回执身份、测试文件/源码哈希、已执行/未执行路径、原始配对耗时和 9 个 ZIP 的完整清单。对应逐题账本的 `team_best_*`、`validity` 和既有平台结果保持原值，仅更新本轮候选和下一步。

下一步优先补 T47 的目标正确性回归，以及 T46/T49/T51 的目标编译和资源证据；T42/T56/T53 已有可复核的 NVIDIA 性能改善，进入其他适用芯片与计分形状复验。T58/T48 必须先拿到实际支持 pipeline 的沐曦运行时。T43/T45 未过性能筛选；T52/T54/T55/T50/T44 继续按具体失败证据推进，不用代理通过替代平台修复。

校正前一份研究文档的 T43 验证措辞：本题接口没有 `state indices` 或 `padding slot`，返回新状态且输入 `conv_state` 必须不变；本轮测试以这份实际题面契约执行。
