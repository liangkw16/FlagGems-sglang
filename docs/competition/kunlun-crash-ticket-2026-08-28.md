# 昆仑芯 1830 秒崩溃：根因研究与彻底治理方案

**更新日期**：2026-09-01

**队伍**：SoulCoder（team_id 2223）

**范围**：FlagOS 第二季竞赛中昆仑芯验证出现的 `Fatal Python error`、
`subproc_pool.py::_recv_msg`、空 `failed_cases` 和约 1830 秒超时；本文替代
2026-08-28 版“reference 编译崩溃”初步判断。

## 结论先行

这不是一个靠继续扫描 `BLOCK_SIZE`、`num_warps` 或换一种 top-k 写法就能
彻底解决的问题，而是两个故障叠加：

1. **已经确认的控制面失败模式**：异步编译路径未能交付结果，也没有及时
   fail-fast；父进程长期停在 `_recv_msg`，最终约 1830 秒终止，与平台给出的
   1800 秒验证超时一致。具体由哪个进程终止、退出 signal 是什么仍待平台确认。
   `_recv_msg` 是结果接收点，不是实际编译崩溃点。上游已知的 FD 泄漏、
   sidecar liveness 和结果投递缺陷与该指纹高度相似，但在拿到比赛 vendor torch
   SHA 和 child 证据前，不能把具体机制写成已确认根因。
2. **尚待平台证据确认的第一故障**：某些候选结构确实能触发昆仑 XPU
   compiler/runtime 缺陷；候选包括原生编译 pass 崩溃、fork 后继承了不安全的
   runtime 状态、Triton/FlagTree/XMLIR/native SO 混装、缓存污染、OOM 或设备
   reset。现有日志没有 child exit signal、完整 `Current thread`、core 和最后
   compiler pass，因此不能把其中任一项直接宣布为唯一根因。

彻底解决必须同时做两件事：**先让任何 child 故障在数秒内显式失败并留下
完整证据，再针对最小 IR 修复昆仑后端**。只做前者会得到“快速报错但仍不能
编译”；只做后者仍会让下一种 native crash 卡满 30 分钟。

## 旧结论为什么必须修正

早期工单把失败归因于 reference 内的 `topk/argsort/matmul/einsum`，并推断
“与选手代码无关”。后续同 reference、只改候选结构的成对实验已经直接反证：

| 成对证据 | 触发版本 | 结构改写后 | 结论 |
| --- | --- | --- | --- |
| T15 `decode_attention` | sub 5053，1833.6s Aborted | sub 5170，17.868s 完成，仅数值 mismatch | 候选结构能解除同族崩溃 |
| T16 `decode_grouped_attention` | sub 4499，1830s Aborted | sub 5186，昆仑正确通过 | reference 不是必崩条件 |
| T29 `gelu_and_mul` | 5733/5810，含 `tl.math.erf` 时崩 | 5840 仅换 A&S 逼近即通过，后续持续通过 | `erf` 是一个已隔离的内容触发器 |
| T40 `softcap_inplace_logits` | 动态 grid-loop 已能通过，0.2457x | 去循环 direct 提至 0.9445x | 某些结构是性能病理，不是统一 crash 根因 |

同时，也不能把问题反过来简化成“全是候选代码”：

- T31 同一可执行结构/AST（两次载体只改了注释）先返回“服务线程卡死”，后返回
  1830s compile-worker crash，说明 worker/服务状态会改变表现；同题其他队伍又
  能通过，说明平台状态与候选触发共同参与。
- T28 已试 3D dot、host-resolved dot、无 dot FMA、`num_stages=1`、int32
  等多种路径仍失败。这排除了这些单变量作为**必要条件**，但不能概括成“所有
  结构都已排除”。
- T38 sub 7518 是单 kernel/单 launch，无 dot、无显式 `erf/log/sqrt`
  libdevice 调用，其他七芯全部通过，昆仑仍在 1,833,762ms 后 Segfault。这推翻
  了“只有重型多 launch 才崩”，但源码仍含 `tl.sigmoid`，不能写成“完全无超越
  函数”。
- T41 去掉外层动态 grid loop 后仍为 1,833,502ms Segfault，且 reference 不含
  topk/argsort/matmul/einsum，直接推翻旧工单的 reference 6/6 因果论。
- T36 曾返回明确的 `uni_sram PassManager::run failed`，证明至少存在内容相关的
  XPU 编译 pass 问题；后续结构变化后出现服务卡死或 child crash，并不等价于
  “已编译通过”。

详细提交证据见实验账本：
[T15](experiments/decode_attention.md)、
[T16](experiments/decode_grouped_attention.md)、
[T28](experiments/gate_up_lora_b.md)、
[T29](experiments/gelu_and_mul.md)、
[T31](experiments/moe_fused_gate.md)、
[T36](experiments/selective_state_update.md)、
[T38](experiments/sigmoid_gate_topk_renorm.md)、
[T40](experiments/softcap_inplace_logits.md)、
[T41](experiments/state_passing.md)。

## `_recv_msg` 到底说明了什么

Inductor 的进程关系大致如下：

```text
validator
└── torch parent
    └── compile sidecar（Popen）
        └── compiler workers（旧实现通常 fork）
            └── Triton / FlagTree XPU compiler / native tools
```

在 PyTorch 的
[`SubprocPool`](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/_inductor/compile_worker/subproc_pool.py)
中，`_recv_msg` 只是在结果管道上读取带长度前缀的消息。普通 Python 编译异常会
被序列化送回；SIGSEGV/SIGABRT、进程被 OOM kill、native deadlock 或结果投递
失败则可能绕开这条正常路径。

官方 Triton [issue #8693](https://github.com/triton-lang/triton/issues/8693)
给出了同类反例：普通 reader thread 停在 `subproc_pool.py:53::_recv_msg`，真正的
`Current thread` 却崩在 `code_generator.py -> ast_to_ttir`。所以比赛日志只截
`_recv_msg`，等于只拍到了等快递的人，没有拍到包裹在哪里丢了。

更关键的是，PyTorch 2026-07 的
[PR #189290](https://github.com/pytorch/pytorch/pull/189290) 精确修复了一个与本次
1830 秒指纹高度相似的控制面缺陷：fork worker 和 parent 错误持有 sidecar→parent
结果管道的写端，sidecar 中途死亡后 read 端收不到 EOF，pending futures 永远不
完成，直到外部 stuck-job killer 介入。官方同时明确指出
`TORCHINDUCTOR_COMPILE_THREADS=1` 会绕过该 subprocess pool。

因此可以分层定性：

| 判断 | 置信度 | 边界 |
| --- | --- | --- |
| `_recv_msg` 不是已定位的编译 pass | 高 | 它是父侧收包点 |
| 1830s 表明编译控制面没有交付结果并到达验证超时 | 高 | 不能据此判断 child 是 crash 还是 hang，也不知道最终 terminator/signal |
| 当前评测控制面没有对未交付结果有效 fail-fast | 高 | 这是 1830s 现象，不等于已经知道 child 第一故障 |
| 上游 FD/liveness/failure-delivery 缺陷是当前具体机制 | 中高 | 指纹高度相似，仍需 vendor torch SHA 和多轮 A/B |
| 昆仑 native compiler/ABI 是最初故障 | 中高 | 有内容触发和官方历史缺陷支持，但缺 core/backtrace |
| fork-unsafe runtime 是最初故障 | 中 | 有相邻后端官方复现，昆仑仍需串行 A/B |
| 某一个固定 pass 或某一种语法是全部任务的统一根因 | 低 | 现有跨任务证据不支持 |

## 上游与昆仑生态给出的关键线索

### 1. PyTorch 已经补齐这类 worker 故障的控制面修复

建议向比赛 vendor torch **移植行为与回归测试**，不要在未知补丁栈上机械
cherry-pick：

- [7dca256](https://github.com/pytorch/pytorch/commit/7dca2564e58ce958df497172cdcbfe0b6d5d7f28)：
  修复 fork FD 泄漏、监控 sidecar 生死、sidecar 死亡时 fail pending futures，
  并以 SIGTERM→有界等待→SIGKILL 收敛卡死 worker。
- [f4da608](https://github.com/pytorch/pytorch/commit/f4da608cb3fc0fe43010c7a90b735d05e00ba4f3)：
  周期性上报慢编译任务状态。
- [9abc546](https://github.com/pytorch/pytorch/commit/9abc5460749ef85e489d960cb5facefc8cc1eb7c)：
  上报 `queued/running/querying_cache/compiling`、阶段耗时和 worker PID。
- [e77e5d5](https://github.com/pytorch/pytorch/commit/e77e5d57b455deece295bb28d79f4da480f68450)：
  普通异常形成 per-job failure；当异常本身也无法序列化、frame 截断或结果回调
  失效时用 `JOB_ERROR` 兜底，并限制 `BrokenProcessPool` 重建次数。
- [fc69434](https://github.com/pytorch/pytorch/commit/fc69434ab0af58a207d86cc02187bb3c1c9dcc0c)：
  为仍然存活但编译不返回的 worker 增加有界等待，支持
  `TORCHINDUCTOR_COMPILE_WORKER_WAIT_TIMEOUT`，补足 sidecar death watchdog
  覆盖不到的 live hang。

PyTorch 官方 [issue #148651](https://github.com/pytorch/pytorch/issues/148651)
说明“已经启动线程的 manager 再 fork”本身会制造难复现故障；
[issue #184643](https://github.com/pytorch/pytorch/issues/184643) 则记录了
`pre_fork_setup()` 初始化 accelerator driver 后，fork worker 继承无效 runtime
状态的具体案例。这两项对昆仑是强假设，不是直接定案。

### 2. FlagTree 自己已有相邻架构的 async worker 事故

FlagTree [issue #1031](https://github.com/flagos-ai/FlagTree/issues/1031)
在 PPU 后端复现了 Inductor worker fork 后重新初始化 runtime 的问题：默认异步
worker 有失败，设置 `TORCHINDUCTOR_COMPILE_THREADS=1` 后 15/15 通过。
后续 [commit 4a773ff](https://github.com/flagos-ai/FlagTree/commit/4a773ffd38189b30ac77ceeca042d270d05803bf)
让 hint manager 直接读取 backend marker，不再为了查询 hint 探测 active device；
修复同时涉及 XPU 副本。比赛镜像必须确认是否包含该提交。

### 3. XPU compiler 是会发生 native fault 的长链，而不是纯 Python 翻译器

FlagTree XPU
[`compiler.py`](https://github.com/flagos-ai/FlagTree/blob/main/third_party/xpu/backend/compiler.py)
依次经过 TTIR→TTXIR→LLIR→ELF→XPUBIN，并运行 offset、local-memory、dtype、
tiling、mask、vectorization、control-flow、loop-grid 等多组 pass。源码还明确记录过：
LLVM target triple 未正确设置会让 target lookup 返回空指针，并在创建 target machine
时 segfault。这个历史事实只证明“native compiler 可以直接把 worker 打死”，不证明
当前事故就是同一个旧 bug。

主线在事故前后仍有高相关修复：

- [3821582](https://github.com/flagos-ai/FlagTree/commit/3821582d91213086e7a5e7d77823525ce398da55)
  修复会把可回绕的离散 offset 错误优化成越界 local-memory 访问、最终导致硬件
  exception 的问题；对 top-k/间接索引族值得优先核验。
- [a7d8892](https://github.com/flagos-ai/FlagTree/commit/a7d8892b2e5331e452636325ba4eadbcc3dbe132)
  修复 SDNN tensor arguments 错位、把错误 buffer base 交给 kernel 并产生 illegal
  memory access 的问题；对分段 GEMM/SDNN 路径值得核验。

FlagTree 的官方
[Triton 3.6 XPU 验证报告](https://github.com/flagos-ai/FlagTree/blob/main/third_party/xpu/docs/triton-3.6-validation.md)
也不是“全覆盖”：公开矩阵为 127/150 pass、20 fail、3 timeout，并列出
vectorization/codegen、masked memory、atomic、verifier/timeout 等未解决项。

### 4. top-k/sort 是公开能力红旗，但不是所有 1830s 的统一解释

FlagGems 当前昆仑后端把 `sort`/`topk` 列入
[`CUSTOMIZED_UNUSED_OPS`](https://github.com/flagos-ai/FlagGems/blob/master/src/flag_gems/runtime/backend/_kunlunxin/__init__.py)，
并在
[`op_black_list.yaml`](https://github.com/flagos-ai/FlagGems/blob/master/src/flag_gems/runtime/backend/_kunlunxin/op_black_list.yaml)
中标记 `grouped_topk`、`topk_softmax` 等兼容问题。这说明 T25/T31/T38 家族继续只扫
通用 Triton 参数的成功概率较低，应优先使用赛制允许的 Kunlun-specific Triton/TLE
`.py` 实现或推动后端修复；但 T28/T36/T41 并非同一 top-k 能力问题，不能用黑名单
解释整个崩溃族。

百度官方
[`vLLM-Kunlun`](https://github.com/baidu/vLLM-Kunlun/blob/main/vllm_kunlun/platforms/kunlun.py)
把 compilation backend 设为 eager，并大量依赖定制算子；其
[安装文档](https://github.com/baidu/vLLM-Kunlun/blob/main/docs/source/installation.md)
要求匹配版本的定制 PyTorch、patch 和多个 vendor op 包。这是另一个强信号：生产
昆仑栈依赖整套版本和能力路由，不能从“vLLM 能跑”推导“任意 Triton kernel 能跑”。
这些二进制定制 op 只作生态证据；竞赛 ZIP 仅允许 UTF-8 `.py`，除非规则明确白名单，
不得打包或调用 vLLM/私有二进制 op，也不得用 PyTorch fallback 绕过赛题。

### 5. 不能单独升级一个 wheel

FlagGems 的
[`backends.yaml`](https://github.com/flagos-ai/FlagGems/blob/master/src/flag_gems/backends.yaml)
记录的公开昆仑测试组合仍是 Triton 3.0/FlagTree XPU 3.0 及一整套绑定版本；FlagTree
主线则已升级 XPU 到 Triton 3.6，并明确建议使用推荐镜像以避免兼容问题。
`torch`、FlagTree/Triton、XMLIR、torch plugin、runtime 和 native SO 必须作为一个
经过 CI 的整体升级。混用两套栈或只替换 compiler wheel，可能比旧版更不稳定。

## 一次就能最大化信息量的判别实验

平台应取一个**不可变失败候选**（首选 T38/sub 7518）和一个**已知通过控制**
（T29/sub 5840、T30/sub 5735 或 T40/sub 6612），固定 ZIP SHA、镜像和节点，使用
全新缓存目录运行。B/C 需要在同节点、同镜像、固定 SHA 下交错重复至少 5 轮，
不能用一次偶然结果定因：

| 顺序 | 模式 | 目的 |
| --- | --- | --- |
| A | known-good add + 已通过 control，默认 async | 验证节点和基础编译链 |
| B | 失败候选，默认 async | 重现比赛指纹 |
| C | 同一失败候选，`TORCHINDUCTOR_COMPILE_THREADS=1` | 绕过 SubprocPool，切开 worker 生命周期与 compiler 缺陷 |
| D | 同一候选，串行 compile-only，逐阶段落盘 | 定位最后成功的 IR/pass |
| E | 已生成 binary 只 launch 一个公开 shape | 切开编译与 runtime/driver |
| F | 冷缓存和暖缓存各一次 | 识别 stale cache/ABI 问题 |

结果解释：

| 观察 | 结论 |
| --- | --- |
| 多轮 B 稳定失败、C 稳定通过 | 强烈提示进程拓扑/并发参与；昆仑可先固定串行编译，再继续区分 fork、IPC 与 bootstrap |
| B/C 均在同一 pass 崩 | XPU compiler/lowering 是主因；对该 IR 做 pass bisection |
| compile-only 通过、E 崩 | codegen/runtime/driver 或硬件 exception |
| 只在暖缓存失败 | cache key 或镜像/ABI 污染 |
| add/control 也失败 | 节点、镜像、runtime 或设备不健康，立即 quarantine |
| 仅某一 worker 失败 | pod/driver/设备污染，不是通用算子限制 |

`spawn` 只作为补充 A/B：如果修过 FD 生命周期的版本中 default fork 失败而 spawn
通过，支持 fork-unsafe runtime；不要在旧 vendor torch 上盲切 spawn，PyTorch 历史上
也出现过 spawn 专属的 FD/`BrokenProcessPool` 问题。

## 彻底治理方案

### P0：当天止血

1. 昆仑验证临时设置 `TORCHINDUCTOR_COMPILE_THREADS=1`，编译串行、执行与计时
   逻辑不变。若竞赛分数不包含首次编译，这个兼容模式不损失 kernel benchmark；
   若包含，应单独记录 compile overhead，而不是恢复危险的 fork 池。
2. `empty failed_cases` 且有明确 child/sidecar 非零退出、signal、服务线程故障、
   平台 watchdog 或节点异常证据时，归类为 `infrastructure_error`，不得记成
   correctness failure 或消耗提交额度；单独一个 timeout 不足以免责。只允许在另一
   个通过 preflight 的 fresh worker 免费重跑一次，`sending/uncertain` 状态不得
   自动重试。
3. 每个 worker 接单前用 fresh process 依次做：设备查询、真实 allocation/copy/compute、
   编译并 launch trivial add、运行一个比赛已通过 control。任一步失败即隔离节点。
4. 不再盲目延长 1800 秒 timeout；确定性 compiler crash 延长时间只会降低吞吐。

### P1：修复编译控制面

把 PyTorch 当前主线的行为回补到比赛 vendor torch：

- parent/sidecar/fork worker 正确关闭管道 FD；
- 周期轮询 sidecar liveness，sidecar 死亡后立即 fail 所有 pending futures；worker
  死亡通过 Future/`BrokenProcessPool` 路径传播；若要逐 worker 轮询，需作为平台
  扩展明确实现；
- 普通编译异常形成 per-job failure；frame 截断、错误也无法序列化或结果回调失效时
  由 `JOB_ERROR` 兜底；
- `BrokenProcessPool` 最多重建三次，确定性同 kernel crash 不无限重试；
- 用 `TORCHINDUCTOR_COMPILE_WORKER_WAIT_TIMEOUT` 有界终止仍存活但不返回的编译；
- SIGTERM 后有界等待，再 SIGKILL；shutdown 自身也有 timeout；
- 保存 sidecar/worker/vendor child 的 return code、signal、stderr tail 和 core path；
- 报告 `queued/querying_cache/compiling` 及最后进展时间。

这一步的验收不是“不再看到错误”，而是任何 child 被 SIGSEGV/SIGABRT/SIGKILL 后，
父任务在数秒内得到带 process role、PID、signal 和最后 phase 的明确失败。

### P2：锁定并净化整套镜像

每次评测必须落盘：

- image digest、OS/kernel/CPU、XPU 型号、driver、firmware、XRE/XCCL；
- Python、`torch.version.git_version`、Triton/FlagTree/FlagGems/XMLIR/plugin commit；
- `import triton` 的真实路径、`FLAGTREE_BACKEND`/`TRITON_BACKEND`、device id；
- `libtriton.so`、`libxpujitc.so`、`liblaunch_shared.so`、XMLIR/LLVM 等关键 SO 的
  SHA256 和 `ldd`；
- 父进程、sidecar、每个 worker 的环境快照；
- compiler cache hit/miss、cache path 和 cache key。

缓存 namespace 至少包含 image/compiler/runtime digest 和 XPU arch；镜像、driver、
FlagTree 或任一 native SO 更新后必须换 namespace，不能复用旧 persistent cache。
升级时选择完整的、已在 XPU CI 认证的 bundle，禁止只换 Triton/FlagTree wheel。

### P3：找到第一个 native fault 并修 XPU 后端

串行模式仍崩时，按 FlagTree pipeline 逐段运行并持久化最后成功产物：

```text
TTIR -> TTXIR -> LLIR -> ELF -> XPUBIN -> one launch -> full cases
```

先核对比赛 vendor 版本是否支持这些变量。当前上游 Triton/FlagTree 的串行编译与 IR
落盘变量如下；旧 XPU 3.0 wheel 不保证全部支持：

```text
TORCH_COMPILE_DEBUG=1
PRINT_TRITON_FUNC=1
TRITON_ALWAYS_COMPILE=1
TRITON_REPRODUCER_PATH=<persistent-file>
MLIR_ENABLE_DUMP=<kernel-name>
MLIR_DUMP_PATH=<persistent-file>
MLIR_ENABLE_TIMING=1
LLVM_IR_ENABLE_DUMP=1
LLVM_ENABLE_TIMING=1
TRITON_KERNEL_DUMP=1
TRITON_DUMP_DIR=<persistent-dir>
```

完成 2026 主线 worker 补丁回移后，async 复现还可启用：

```text
TORCHINDUCTOR_WORKER_SUPPRESS_LOGGING=0
TORCHINDUCTOR_WORKER_LOGPATH=<persistent-dir>/worker-<rank-or-pid>.log
TORCHINDUCTOR_COMPILE_WORKER_WATCHDOG_INTERVAL=10
TORCHINDUCTOR_COMPILE_WORKER_WAIT_TIMEOUT=<bounded-seconds>
TORCH_TRACE=<persistent-dir>
```

`TORCHINDUCTOR_COMPILE_THREADS=1` 已绕过 SubprocPool，因此上面的 `WORKER_*` 和
watchdog 变量对串行复现没有作用；`WORKER_LOGPATH` 还必须按 rank/PID 唯一命名，
否则以写模式打开时会互相覆盖。这些 2026 主线配置不存在于
[PyTorch v2.5.1 config](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/_inductor/config.py)
时，单纯设置环境变量不会产生能力，必须先回补实现并用实际 vendor SHA 验证。

Triton 官方
[调试说明](https://github.com/triton-lang/triton/blob/main/README.md#tips-for-hacking)
支持把失败前 IR 落盘；官方 [issue #5122](https://github.com/triton-lang/triton/issues/5122)
展示了将最小 TTGIR 交给 `triton-opt` 单 pass 重现并 bisect 到具体 LLVM 更新的标准
路径。XPU 侧应以相同方式逐个关闭/运行 offset analysis、vectorization、control-flow、
loop-grid 等 pass，只用于定位；不能把全局禁优化当最终修复。

拿到最小 IR 后，vendor 修复必须包含：触发 IR、修复前 native signal/backtrace、修复
commit，以及 cold/warm cache、serial/async 的回归测试。native compiler 工具最好再
隔离到可监督的外部进程，任何 native fault 都只能杀掉单次 compile job。

### P4：按能力边界修候选与路由

平台完成可观测性后，候选侧才值得做定向规避：

- `erf` 家族沿用 T29 的 A&S 逼近，直到 XPU lowering 有正式回归；
- top-k/sort 家族优先使用题面允许的 Kunlun-specific Triton/TLE `.py` 实现；若无
  合法实现，先用标量/分阶段安全基线形成最小复现，不得打包二进制 vendor op 或用
  PyTorch fallback，也不再无目标地扫 generic block 参数；
- masked/间接索引必须使用 safe address，再用 mask 选择值，重点回归 offset wrap；
- 动态 grid-stride loop 只在证明收益后保留；T40 已显示 direct 映射既更快又更简单；
- SDNN/分段 GEMM 核验参数 ABI 和 buffer base，不能只看 Triton IR 正确。

但这些只能绕过已知触发器，无法修复 worker FD、stale ABI 或 native child 失联，不能
作为“彻底解决”的替代品。

### P5：建立昆仑专属回归门禁

最小语料应包含：

- 通过控制：T30/5735、T29/5840、T33/6333、T40/6612；
- 成对触发/反证：T29 `5733 -> 5840`、T15 `5053 -> 5170`、T16
  `4499 -> 5186`；
- 持续失败：T38/7518、T41/6675 与 6693、T28/6194；
- 编译错误分类：T36/6897 的显式 `uni_sram PassManager` 与 T36/7135 的 child
  crash。

每个样本至少覆盖 cold/warm cache、serial/async、compile-only/one-launch，并保留
kernel source SHA 和 specialization。版本升级门禁中应连续多进程运行，而不是只测
一次 happy path。

## 平台工单必须索取的字段

没有以下字段，就无法再从“待收包线程”追到第一故障：

- `process_role`（validator/torch parent/sidecar/compiler worker/vendor child）、
  PID/PPID、job ID、compile ID；
- kernel 名、源码 SHA、shape/dtype、grid、`num_warps/num_stages`；
- submit/start/last-phase/end 的单调时钟；
- sidecar、worker 和 vendor compiler 的 return code；负数转换后的 signal；
- 完整 Fatal dump 中的 `Current thread`、native backtrace、stderr 末 200 行、core；
- cgroup `memory.events`、OOM kill、kernel/driver reset/XPU hardware exception；
- 上述完整版本、SO 哈希、worker/pod 和 cache 信息。

## 验收标准

“问题彻底解决”应同时满足：

1. sidecar、compiler worker 或 vendor child 任意死亡，任务在 10 秒量级显式结束，
   不再出现 1830 秒假死；
2. 错误能指出死亡角色、signal、最后 compiler phase 和 kernel SHA；
3. `infrastructure_error` 与用户 correctness/performance failure 分账，前者不扣额度；
4. T38/7518、T41/6693、T28/6194 等原失败样本在 fresh process 下 cold/warm cache
   各连续 20/20 完成 compile+launch；
5. serial 和 async 均通过；若暂时只允许 serial，必须明确记录为兼容模式并保留后续
   async 修复门禁；
6. 至少跨两个通过 preflight 的 worker 完成全八芯验证；
7. 每个已修 native bug 都进入 FlagTree/XPU CI，后续升级不能回归为 timeout。

## 当前建议的执行顺序

1. **先做一个 worker 上的单变量实验**：clean cache + 已知通过 control +
   T38/7518，默认 async 与 `TORCHINDUCTOR_COMPILE_THREADS=1` 至少交错重复 5 轮。
2. **并行回补 PyTorch worker fail-fast**：这不依赖第一故障究竟是什么，且能永久消灭
   1830 秒无证据等待。
3. 串行仍崩时，按 TTIR→TTXIR→LLIR→ELF→XPUBIN→launch 切层，抓 core 和
   最小 IR。
4. 核对并整体升级已认证的 XPU 软件栈，至少确认 `4a773ff`、`3821582`、
   `a7d8892` 及 target-triple 修复是否存在。
5. 最后才根据最小 IR 修 top-k/offset/loop-grid/SDNN 等具体后端 pass，并跑完整回归
   门禁。

最关键的一点是：**不要再把 30 分钟的 `_recv_msg` 栈当成编译器根因，也不要再用
30 分钟全量提交替代一次 1–5 分钟的串行、分阶段、可观测复现。**
