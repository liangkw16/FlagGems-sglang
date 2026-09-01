# Task 41 `state_passing` 实验记录

```current
task: 41
operator: state_passing
batch: 3
validity: invalid
platform: E5 ready;baseline E3 7/8(8/8 terminal)
team_best_stage: E3
team_best_commit: 248693b
blockers: E5燧原i32与昆仑host-step均待目标芯裁决;KernelGen昆仑验证器HTTP502
sealed: no
next: E5单次提交;两目标芯均>=0.1且8/8才转正
updated: 2026-09-02
```

## S0：KernelGen generic 基线

状态：已唯一一次提交；7 芯被同一 BF16 initial dtype 断言拦截，昆仑在
compile worker 崩溃；0/8，通过状态为 `invalid_correctness`，S0 不得重试。

### 契约与生成

- 签名为 `state_passing(states, dA_cumsum, initial_states=None)`；输入布局分别为
  `[B,C,H,D]`、`[B,H,C,L]` 和可选 FP32 `[B,H,D]`。每块先把更新前的
  FP32 running state 写到 `out[:, c]`，再执行
  `cur = cur * exp(dA_cumsum[:, :, c, -1]) + states[:, c].float()`；输出
  `out` 保持 states dtype，`final_states` 始终为 FP32。
- 通过已配置的 `kernelgen-server.generate_kernel` 生成首版 raw Triton。
  MCP 服务端验证 harness 报 `NameError: torch is not defined`，`total_tests=0`，
  因此不把该报告作为正确性或性能证据。用户选择 B 后，按仓库约定在远端 GPU
  完整复验。
- 保留 MCP 的单 kernel 顺序递推，只做仓库适配：固定 `BLOCK=256/4 warps/1
  stage`、optional initial state 改为 constexpr、地址 stride 在乘加前升为
  int64、直接写 states dtype 输出，并复用现有 capped 1D grid + kernel 内
  grid-stride。后者修复 MCP 原始产物在 logical tiles 超过 65,535 时漏写尾部
  tile 的确定性缺陷；递推公式未手改。
- `nchunks=0` 返回空 out 和 initial 的独立 FP32 clone/FP32 zeros；`B/H/D=0`
  在 wrapper 短路，避免零 grid。无 fallback、设备分支、try/except、autotune 或
  vendor 文件。

### 构建身份

| 项目 | 值 |
| --- | --- |
| source / verification commit | `91f3fd86ce9d626b3c642c44b2c4d8dc1268b7ec` |
| generic SHA-256 | `f65abb5c20eea703e0560868520c5936ef93bc57679492a8922eeaf68e753a25` |
| test SHA-256 | `e1d76eede747e3bb50f3a34db1eadc708f4c73c4ea506732af785099a7a2083a` |
| benchmark SHA-256 | `d5dec07565eaab4c7a59ddb60f0966ae8cb6e58e1457e586f1ddd684c3ee9f12` |
| canonical ZIP | `artifacts/competition/state_passing/s0-91f3fd8/state_passing.zip` |
| ZIP size / SHA-256 | `6494` bytes / `005a5f5b52c6b8e24f7623c6a9d1554f9639c64877686956265f19af024222ea` |
| ZIP members | 顶层 `state_passing.py`，唯一普通 UTF-8 `.py` 成员 |

规范打包器的 dry-run、create、`--verify-existing` 给出相同 canonical ZIP
SHA-256；`unzip -t/-Z1`、10 MB 上限、basename、成员集合及成员与 source commit
逐字节核对均通过。旧格式化前的 `s0-ca9b984` 产物已废弃，不得 preflight 或上传。

### 验证证据

- screening：`gpu-et:/tmp/flagos-state-passing-screen2.ClrNOB`，base commit
  `ca9b9848601e511db2f57bea208854d466be990a`，从工作树明确传入上述三份最终字节；
  runner SHA-256
  `1716e76c11733fb011c958124aa6bd2af7b1e0a90a93000d114336cdb3ccb35f`，
  manifest SHA-256
  `fb85db4def46e16f6724f2185df6dbfae604cad592300dc284a97e245b0fb39b`，
  日志 SHA-256
  `82150e454d4e413282aca822839ee527f4a9cd2f80d392a5e6b7b03736daea72`。
- release：`gpu-et:/tmp/flagos-state-passing-release2.m12IUn`，三份文件全部从
  verification commit 的 Git 对象生成；前后哈希与 screening 完全一致。
  runner/manifest 同上，release 日志 SHA-256
  `e9dfa9de301762ce788b551177fcfc41c78948a847faef4258997cc6a31e5d59`。
- 两轮均使用 Python 3.12.13、PyTorch 2.13.0+cu130、Triton 3.7.1、CUDA
  13.0、NVIDIA driver 610.57.04、RTX 5070 Ti。启动时 GPU 无竞争计算进程；
  `py_compile`、Black 26.5.1、isort 8.0.1、flake8 7.1.2、末尾 SHA-256
  复核和 unittest **4/4** 全过。
- 回归覆盖 FP16/BF16/FP32、initial None/FP32、三类非连续末维 stride、
  `C != H`、dim=257 尾块、只读取 dA 最后一 lane、pre-update 快照、
  `nchunks=0`、输入不可变，以及 `131072 > 65535` logical tiles 的完整
  grid-stride 覆盖。

release 的 wrapper-inclusive 五轮交替 AB/BA p50 代理结果：

| `(B,C,H,D,L)` / dtype | reference ms | candidate ms | speedup |
| --- | ---: | ---: | ---: |
| `(1,1,3,257,17)` / FP16 | `0.014361` | `0.004323` | `3.246494x` |
| `(4,16,16,256,16)` / FP32 | `0.088955` | `0.013042` | `6.793776x` |
| `(2,8,64,8192,256)` / BF16 | `0.232786` | `0.064886` | `3.575274x` |
| `(2,64,8,4096,64)` / FP16 | `0.825440` | `0.042400` | `19.534728x` |

四组中位加速平均 `8.287568x`，最差 `3.246494x`，最好 `19.534728x`；
完整 paired speedup 样本依次为
`[3.245074,3.240570,3.452174,3.246494,3.354879]`、
`[6.784156,6.827726,6.701846,6.887595,6.793776]`、
`[3.606574,3.575274,3.559324,3.529822,3.597469]`、
`[19.578217,19.534728,19.656105,19.503743,19.467547]`。
该结果仅为 NVIDIA 代理，不外推八芯速度。

### 提交预注册

2026-08-30 08:29:42 CST 平台只读状态：race `782kzq4m`、season 2、账号
`15600308080`、团队 `SoulCoder`、Task 41/`s2t1op041`、batch 3、
`competing/submitting`、`can_submit=true`、action=`challenge_operator`；本队
尚无该题提交，额度 `14/30`，最小提交间隔 120 秒已满足。08:17 题目快照显示
38 次提交、12 队，2 队过门槛，榜首 c2flow 为 `5.4805625x`、8/8。

S0 只允许上述 ZIP 上传和正式提交各一次，八芯均只选择 generic。基础门为
8/8 correctness 且每芯 `>=0.1x`；冲榜门为平均严格高于实时榜首
`5.4805625x`。若单芯编译或正确性失败，只为该芯新增一个自包含 vendor；若
8/8 valid 但未登顶，先按逐芯差距选择唯一一个 BLOCK/grid/control-flow 轴，
冻结其余已通过字节，不重复上传 S0。

### S0 平台记录（sub 6674，2026-08-30 08:33:56 CST）

- 实时 preflight tuple 全匹配，额度 `14/30`；唯一一次上传和正式提交成功后
  额度为 `13/30`。八芯都选择 `state_passing.py`，intent 状态为
  `submitted`；`file_url_sha256` 为
  `329f3cc34caefcabd74a685f293791d0a6e9db90a81b494515377645e7cd36eb`。
- 远端对象存储 hostname 未配置为可信值，匿名回读状态为 `unavailable`；这不
  改变已提交事实，也不得据此重试。
- 天数、沐曦、燧原、海光、华为、国际 A/B 共 7 芯均在 hidden
  case 1 进入 wrapper 后触发同一行
  `assert initial_states.dtype == torch.float32`。平台实际输入的 states 与
  initial_states 都是 BF16，dA_cumsum 为 FP32；失败发生在 Triton launch 前，
  没有数值差异证据。昆仑在 1830 秒后以空 `failed_cases` 崩于
  `torch/_inductor/compile_worker/subproc_pool.py::_recv_msg`。截至 09:41，
  S0 已 8/8 终态、0 芯通过；两类失败相互独立，S0 不得重试。

## E1：接受低精度 initial state

状态：已唯一一次提交；7/8 芯正确性通过，昆仑 compile worker 崩溃；
`invalid_correctness`，E1 不得重试。

E1 是一次 targeted wrapper 修复：只删除 S0 的 initial dtype 断言；shape 断言、
kernel、BLOCK/grid/warps、递推、输出 dtype 和全部地址逻辑逐字节语义不变。测试把
BF16 case 的非连续 initial 改为 BF16，同时保留 FP32 initial case。kernel 原本就会
在 `tl.load` 后转 FP32，完全匹配题面 reference 的 `initial_states.float().clone()`。

| 项目 | 值 |
| --- | --- |
| source / verification commit | `64b1eadd0fac3429db09725634b898f9a8ff1d9a` |
| generic SHA-256 | `af0e622458f7ab89fadc45273c60f209bad5477fc8374de5e996d8193447c034` |
| test SHA-256 | `e7acd4e2cc60f16b0df1d916215fee0e3631fc1aa71e727b6ebc6e8f9f7a5eac` |
| benchmark SHA-256 | `d5dec07565eaab4c7a59ddb60f0966ae8cb6e58e1457e586f1ddd684c3ee9f12`（=S0） |
| canonical ZIP | `artifacts/competition/state_passing/e1-64b1ead/state_passing.zip` |
| ZIP size / SHA-256 | `6441` bytes / `052a0e1240977adeb79616d843b7ed771781fd89aea4766b4b4a6629d3365fbf` |
| ZIP members | 顶层 `state_passing.py`，唯一普通 UTF-8 `.py` 成员 |

- screening：`gpu-et:/tmp/flagos-state-passing-screen3.zD9gFs`，base commit
  `7304abccdfaf52122cf92e9f4adc81baadc91692`；runner SHA-256
  `1716e76c11733fb011c958124aa6bd2af7b1e0a90a93000d114336cdb3ccb35f`，
  manifest SHA-256
  `404f29fbe38bed3af7984525b04b997d7b584238df11d7b13c18c7fae4ec0b3b`，
  日志 SHA-256
  `37816ede7b26119751390bd073d2e2515b7fdfa2368e7f95dfb15d08cbe0df6c`。
- release：`gpu-et:/tmp/flagos-state-passing-e1-release.MTG4vX`，三份文件全部
  从 verification commit Git 对象生成，哈希与 screening 一致；release 日志
  SHA-256
  `2f7fde19b208c89c5db8a9f72c106a068b3bd53ea43f834e2480a4b1e5aca14b`。
- 两轮的 py_compile、Black、isort、flake8、末尾哈希复核和 unittest 4/4
  全过；BF16 non-contiguous initial 现已实际执行 kernel 并匹配 FP32 reference。
  release 四组 wrapper-inclusive 中位加速为 `3.263597x / 6.780337x /
  3.586362x / 19.461507x`，平均 `8.272951x`，与 S0 代理一致。
- 规范打包器 dry-run/create/`--verify-existing`、`unzip -t`、唯一成员、UTF-8、
  10 MB 和 commit 逐字节门禁全过。E1 只允许该 ZIP 上传和正式提交各一次；
  基础门为 8/8 correctness 且每芯 `>=0.1x`，冲榜门仍为平均严格高于实时榜首
  `5.4805625x`。若出现数值 mismatch，按 KernelGen Category B 回到 MCP
  重新生成，不手改递推。

### E1 平台进度（sub 6675，2026-08-30 08:37:41 CST）

- 实时 preflight tuple 全匹配，额度 `13/30`；唯一一次上传和正式提交成功后
  额度为 `12/30`。八芯均选择 `state_passing.py`，intent 状态为
  `submitted`；`file_url_sha256` 为
  `37274576c31d3853034a82856f1433913f03e8e6cf33cd0f013710a4108b5d22`。
- 远端对象存储 hostname 未配置为可信值，匿名回读状态为 `unavailable`；已提交
  事实不受影响，E1 不得重试。
- 2026-08-30 09:41:40 CST 只读终态：8/8 芯完成，7 芯通过正确性；昆仑
  1830 秒后崩于 Inductor compile worker，`failed_cases=[]`，没有数值或用户
  kernel traceback。逐芯结果如下：

| 芯片 | speedup | 门槛 | 文件 |
| --- | ---: | --- | --- |
| 天数 | `7.6365x` | 通过 | generic |
| 沐曦 | `4.3290x` | 通过 | generic |
| 燧原 | `0.0605x` | **低于 0.1x** | generic |
| 海光 | `10.1765x` | 通过 | generic |
| 昆仑 | - | compile-worker segmentation fault | generic |
| 华为 | `1.0905x` | 通过 | generic |
| 国际 A | `8.4275x` | 通过 | generic |
| 国际 B | `4.5500x` | 通过 | generic |

E1 最终为 7/8、`invalid_correctness`，平台不计算平均或排名；7 个速度的简单平均
`5.181571x` 仅作排障观察。E1 已证明递推和低精度 initial 修复对 7 芯正确；已知
失败轴是昆仑编译崩溃，另有燧原 `0.0605x < 0.1x` 性能门槛失败。

## E2：Enflame physical grid cap 12

状态：commit-bound release 与 canonical ZIP 门禁通过；因 E1 昆仑失败，已按
预注册停止，未 preflight、未上传、未提交。产物保留用于后续组合，不得单独提交。

E2 冻结 E1 generic 字节，只新增自包含 `_enflame`。该 vendor 与 generic 的
完整源码 diff 只有 `_MAX_GRID = 65535` 改为 `12`；BLOCK256、4 warps、1 stage、
1D grid-stride、FP32 递推、地址、wrapper 和全部 dtype 语义不变。T24 同芯单变量
A/B 曾证明 full-grid 65535 相对 cap12 使燧原下降 `24.49%`，T40 的 cap12 vendor
四轮稳定在 `2.3212–2.3312x`，因此先验证这条目标芯已有正证据的最小轴。

按 KernelGen 低性能流程调用 `optimize_kernel` 一次。服务返回的候选同时引入 2D
grid、`num_stages=2`、两个 eviction hint 和地址 hoist，不能隔离单变量，且目标芯
没有对应正证据；在写文件前按 usability gate 拒绝，未覆盖任何已验证源码。

| 项目 | 值 |
| --- | --- |
| source / verification commit | `2c2bb3a9227ad039bf27c5b927e96ae1f7926cef` |
| generic SHA-256 | `af0e622458f7ab89fadc45273c60f209bad5477fc8374de5e996d8193447c034`（=E1） |
| Enflame SHA-256 | `1e24ec0b2e3d92ee8a9f9d4f8a7a8f6451a31f9e29b953d01d9663dd7b92b218` |
| test SHA-256 | `a8c08b8d1da26e24629a50cef9ce041c1659384b3415965a69ac525cff6ffc8f` |
| benchmark SHA-256 | `d5dec07565eaab4c7a59ddb60f0966ae8cb6e58e1457e586f1ddd684c3ee9f12`（=E1） |
| canonical ZIP | `artifacts/competition/state_passing/e2-2c2bb3a/state_passing.zip` |
| ZIP size / SHA-256 | `12873` bytes / `416c1852df7673dc8bc2f2ae8f6aef6c2308d93f9df16fcda6e2a05303bcc3bc` |
| ZIP members | `state_passing.py`、`state_passing_enflame.py` |

### 验证证据

- screening：`gpu-et:/tmp/flagos-state-passing-e2-screen.CHrXKu`，base commit
  `2f3bc84`，四份候选字节显式传入；runner SHA-256
  `2c42fb8b966432c7c46deb69aeb32ac13ee58bb7b01e5e9919fa1a4466872690`，
  manifest SHA-256
  `881ba7a3dd5415198fa3c65b29a002c97f7bd01f7c2ed96537e102f1f1aaa597`，
  日志 SHA-256
  `9124007c94ac6961ef07965c678158fa40665343dae4911310e3b30ae5754311`。
- release：`gpu-et:/tmp/flagos-state-passing-e2-release.OqoiC4`，四份文件全部从
  verification commit 的 Git 对象生成，哈希与 screening 完全一致；runner 与
  manifest 同上，release 日志 SHA-256
  `9f42898485d12e2d7b3acb413d6e89bc4e97e40bb60e1040eb50477c1cfd8a2e`。
- 两轮都通过 py_compile、Black、isort、flake8、末尾哈希复核和 unittest
  **5/5**。新增 `(13,1,1,1)` case 实际覆盖 `13 > cap12` 时的第二轮
  grid-stride；原 `131072 > 65535` generic 完整覆盖回归保留。
- 环境仍为 Python 3.12.13、PyTorch 2.13.0+cu130、Triton 3.7.1、CUDA
  13.0、RTX 5070 Ti。release 的冻结 generic 四组 wrapper-inclusive 中位加速为
  `3.241357x / 6.875155x / 3.572753x / 19.867773x`，平均 `8.389259x`。
  NVIDIA 结果只证明语法、数值和 generic 未漂移，不外推 cap12 在 GCU 的收益。
- 规范打包器 dry-run/create/`--verify-existing` 均给出同一 ZIP SHA-256；
  `unzip -t/-Z1`、两成员 basename、UTF-8、10 MB 上限及成员与 commit 逐字节
  核对全过。

### 提交预注册

E2 原计划只允许上述 ZIP 上传和正式提交各一次，平台仅为燧原选择
`state_passing_enflame.py`。E1 昆仑已终态失败，因此前置门不满足；E2 的 upload、
submit 计数均为 0，永久停止其独立提交。Enflame 字节仍由 commit `2c2bb3a` 和
canonical ZIP `416c1852...3bcc3bc` 固定，只有昆仑独立轴通过后才可原样组合。

E2 基础门为 8/8 correctness 且每芯 `>=0.1x`；单轴晋级门为燧原严格高于 E1
`0.0605x` 且达到 `0.1x`；冲榜门为平均严格高于实时榜首。09:41 平台仍
`can_submit=true`，实时额度为 `11/30`；该额度变化不归因于从未提交的 E2。
若 cap12 不升，永久停止 Enflame grid-cap 轴，不在同轮追加 BLOCK、warps、math
或 MCP 的多变量候选。

## E3：Kunlun host-segmented direct tile ownership

状态：Kunlun-only 单变量候选已唯一一次提交；7/8 芯正确性通过，昆仑复现
compile-worker crash；`invalid_correctness`，E3 不得重试，direct 轴停止。

E1 generic 在 kernel 内把 capped grid 的每个 program 再放入 logical-tile
grid-stride 外循环，外层 loop 与 `nchunks` 顺序递推形成两层 runtime loop。昆仑
backend 自身还会注入 LoopGrid pass；E1 的 1830 秒空 case compile-worker crash、
T40 同芯去动态 grid-loop 后 `0.2457x -> 0.9445x`，以及现有 Kunlun
`decode_attention` 的 host 65,535 分段模式，共同支持只测试 tile ownership。

按 `kernelgen-flagos` 流程把 E1 崩溃作为 `check_result` 调用
`kernelgen-server.optimize_kernel`。首个结果引入第二 kernel 和 helper；第二个结果
保留无用 `total_tiles` 参数并加入两个 eviction hint，均在写文件前被 usability
gate 拒绝。第三个结果收敛为一个 kernel：`tile_id = tile_start + program_id(0)`，
wrapper 每次最多启动 65,535 tiles；只按 Black 26.5.1 合并一行换行后落盘。
`nchunks` 递推、BLOCK256、4 warps、1 stage、数学、地址、dtype 和空输入行为均冻结。

E3 以 E1 平台字节为基线，仅新增 `_kunlunxin`，不携带从未上平台验证的 Enflame
E2。当前分支删除 Enflame vendor 只为保证规范打包器不会夹带第三成员；其字节仍由
E2 commit 与 canonical ZIP 完整保存。

| 项目 | 值 |
| --- | --- |
| source / verification commit | `248693bfe210408dfefb0dbfbf687f195b797825` |
| generic SHA-256 | `af0e622458f7ab89fadc45273c60f209bad5477fc8374de5e996d8193447c034`（=E1） |
| Kunlun SHA-256 | `755f5b47029c7878e358db746757bfeb954e1ad3e81880ac7de7b14e4ccc4075` |
| test SHA-256 | `756f60731938baae6b0f02a91a68aaacecf3bd8e561143672e57782f4e1a5747` |
| benchmark SHA-256 | `d5dec07565eaab4c7a59ddb60f0966ae8cb6e58e1457e586f1ddd684c3ee9f12`（=E1） |
| canonical ZIP | `artifacts/competition/state_passing/e3-248693b/state_passing.zip` |
| ZIP size / SHA-256 | `12838` bytes / `a8fa080d5d80dd92d679dd89162e9ec8a990681d4163768098ba7b896e36c4e2` |
| ZIP members | `state_passing.py`、`state_passing_kunlunxin.py` |

### 验证证据

- screening：`gpu-et:/tmp/flagos-state-passing-e3-screen2.htJBk6`，base commit
  `632dd5e`，四份工作树候选字节显式传入；runner SHA-256
  `39296e4023dad47cccacf1e1c83cc6e4d9ff9e35964a0a5a8ba27a9f249021b1`，
  manifest SHA-256
  `a4a3f3936908fdbc3521fdc7e5ec2116f189a952065f97c84d60b1f1d86e319f`，
  日志 SHA-256
  `7c8152af4ac3f8121de83e6614372482ad4227a560c48b42faed6cac9c652193`。
- release：`gpu-et:/tmp/flagos-state-passing-e3-release.fN18o0`，四份文件全部由
  verification commit 的 Git 对象生成，哈希与 screening 完全一致；runner 与
  manifest 同上，release 日志 SHA-256
  `e0ff68d6d6e244b9f3755a1d6219fd85e4eb7b86b333c84e53b6012c86321de5`。
- 两轮均通过 py_compile、Black 26.5.1、末尾 SHA-256 复核和 unittest **4/4**。
  三 dtype/非连续 stride/BF16 initial/dim tail 同时执行 generic 与 Kunlun；
  `131072 > 65535` logical tiles 同时验证 generic grid-stride 与 Kunlun 三次 host
  launch 全覆盖。NVIDIA 只能证明 JIT、数值与索引覆盖，不外推 XPU crash 修复。
- release 冻结 generic 的四组 wrapper-inclusive 中位加速为
  `3.295136x / 6.892384x / 3.592256x / 19.296081x`，平均 `8.268964x`；
  benchmark 字节与 E1 完全一致。
- 规范打包器 dry-run/create/`--verify-existing` 均给出同一 ZIP SHA-256；
  `unzip -t/-Z1`、两成员 basename、UTF-8、10 MB 上限及成员与 commit 逐字节
  核对全过。

### 提交预注册

E3 是昆仑独立诊断候选，只允许上述 ZIP 上传和正式提交各一次；预期仅昆仑选择
`state_passing_kunlunxin.py`，其他 7 芯继续选择冻结 generic。09:41 平台 tuple 为
race `782kzq4m`、season 2、Task 41/`s2t1op041`、batch 3、
`competing/submitting`、`can_submit=true`，额度 `11/30`，120 秒间隔已满足。

E3 的单轴晋级门为昆仑完成正确性且 `>=0.1x`。已知 Enflame 仍用 generic，故 E3
本身预期仍因其 `0.0605x` 低于门槛而无效；本次额度只用于隔离昆仑 crash，不据此
重测其他 7 芯。若昆仑通过，下一候选把 E2 Enflame 字节与 E3 Kunlun 字节原样组合；
若仍复现约 1830 秒、空 `failed_cases` compile-worker crash，永久停止 direct 轴，
不盲扫 BLOCK、warps 或数学。

### E3 平台进度（sub 6693，2026-08-30 09:48:53 CST）

- 实时 preflight 的 race/season/account/team/batch/Task/tid/operator、source commit、
  stage、ZIP 绝对路径、SHA-256、两成员、提交窗口、120 秒间隔和 `can_submit=true`
  全匹配，额度为 `11/30`。只执行返回的一次性 confirm 命令；提交后额度 `10/30`，
  intent 状态为 `submitted`，E3 不得重试。
- `file_url_sha256` 为
  `7621487d9adacbfe8aca36df9b239226a7f87b38d9e779cffc458e732552c781`。
  远端对象存储 hostname 未配置为可信值，匿名回读为 `unavailable`；这不改变已提交
  事实。昆仑选择 `state_passing_kunlunxin.py`，其他 7 芯均选择 generic，完全符合
  预注册。
- 09:50:27 CST，7 个非昆仑芯片全部完成并通过正确性；昆仑 validation
  `b80ee32ab974` 为 `waiting_callback`。当前逐芯结果：

| 芯片 | speedup | 门槛 | 文件 |
| --- | ---: | --- | --- |
| 天数 | `7.7945x` | 通过 | generic |
| 沐曦 | `4.3090x` | 通过 | generic |
| 燧原 | `0.0610x` | **低于 0.1x，符合已知基线** | generic |
| 海光 | `10.1800x` | 通过 | generic |
| 昆仑 | - | compile-worker segmentation fault | Kunlun vendor |
| 华为 | `1.0295x` | 通过 | generic |
| 国际 A | `8.4260x` | 通过 | generic |
| 国际 B | `4.5855x` | 通过 | generic |

7 个已返回速度的简单平均为 `5.197929x`，仅作排障观察。10:23:08 CST 只读终态：
昆仑在 `1833502 ms` 后再次以空 `failed_cases` 崩于
`torch/_inductor/compile_worker/subproc_pool.py::_recv_msg`，与 E1 的错误类型、
阶段和约 1830 秒时长完全一致；host-segmented direct 未改变故障。

E3 最终为 8/8 terminal、7/8 passed、`invalid_correctness`，平台不计算平均或排名；
额度剩余 `10/30`。按预注册永久停止 direct tile-ownership 轴，不盲扫 BLOCK、warps
或数学。当时不构建提交候选；后续只把 E2 Enflame 与 E3 Kunlun 原样组合成健康窗口
载体，不把它记作 direct 轴修复或平台进展。

## E4：三 vendor 健康窗口载体收口（2026-09-01）

E4 不再改 kernel。generic 保持 E1 平台已证字节，Enflame 恢复 E2 的精确字节，
Kunlun 保持已被 E3 平台证伪的 direct-tile 字节。`fc6dd4f` 曾为 carrier 加过五行
说明注释；本轮仅删除这些注释，使 Enflame SHA-256 精确回到 E2 的
`1e24ec0b2e3d92ee8a9f9d4f8a7a8f6451a31f9e29b953d01d9663dd7b92b218`，并把现有
unittest 从 generic+Kunlun 扩为 generic+Enflame+Kunlun 矩阵，补 B/H/D 为零。

| 项目 | 值 |
| --- | --- |
| source / verification commit | `eb5adcb73a9d4963be0a3662edc4b4d2949fe098` |
| generic SHA-256 | `af0e622458f7ab89fadc45273c60f209bad5477fc8374de5e996d8193447c034`（=E1） |
| Enflame SHA-256 | `1e24ec0b2e3d92ee8a9f9d4f8a7a8f6451a31f9e29b953d01d9663dd7b92b218`（=E2） |
| Kunlun SHA-256 | `755f5b47029c7878e358db746757bfeb954e1ad3e81880ac7de7b14e4ccc4075`（=E3） |
| test SHA-256 | `0bd11f2626cabd57121e9f706a1e7ed62e64f3e90d1d0c64632e2ff5493cc454` |
| benchmark SHA-256 | `d5dec07565eaab4c7a59ddb60f0966ae8cb6e58e1457e586f1ddd684c3ee9f12`（=E1） |

- screening 为
  `gpu-et:/tmp/flagos-state-passing-e4-screen-final.OGBprF`，显式传入待提交字节；
  log SHA-256 为
  `b166c3acc253ef7dce5bb606ae4615e4dabe303699b30de4fba0cd301f51135d`。
- exact Git-object release 为
  `gpu-et:/tmp/flagos-state-passing-e4-release.T2Couk`；runner/manifest/release log
  SHA-256 分别为 `e5e61138540fcdf9b3bcc528ab8d96bb1ded5a1cc30f2120aa60db17eb7f49d1` /
  `06e9fb61bb8627823e084d3437919a9cbfeb9155865172953bdb5fe74e1571dd` /
  `8f378c1c842414aa393b676da1b8c8617136ca37eaa7416b9dcb8fa5004a9ec3`。
- 两轮均通过 py_compile、格式/静态检查、逐文件 SHA-256 复核和 unittest
  **5/5**。三 vendor 共同覆盖 dtype/非连续 stride、pre-update、只读 dA 最后一
  lane、空 C、B/H/D=0 和 `131072 > 65535` logical tiles。release 的冻结 generic
  四组 NVIDIA 代理 speedup 为 `3.286442x / 6.833279x / 3.563873x /
  19.538204x`，平均 `8.305449x`；不外推 GCU/XPU。

### KernelGen 与 Kunlun 服务边界

- 实际调用 `generate_kernel(device=sunrise)`；request SHA-256 为
  `a360ff697be146b17a629a4d912e5b4c8249149557d928f0eacdcb0d057001fe`，生成
  Torch/Triton/test/benchmark 成功，但所有用例都在创建 `gcu` tensor 时因当前
  PyTorch 未注册该 device 而失败，最终 `total_tests=passed_tests=failed_tests=0`。
  派生审计摘要 SHA-256 为
  `b53b42f7ef9ae9898466bbe797b11da3f87d92c8f73f8138925ccce1fef051fa`；按门禁
  拒绝零测试结果，不构成燧原正确性或性能证据，生成代码也未落库。
- 用户补充的最小 `x+y` Kunlun 对照耗时 293 秒：MCP 入口正常、
  `mcp_isError=false`，Triton/测试/benchmark 均生成，但 Kunlun verifier 三次均
  HTTP 502。该对照与算子复杂度无关，因而把 T28/T37/T41 的同类 502 定性为
  `generate_kernel -> Kunlun verifier/worker` 服务基线故障或无健康 worker；它不
  能证明候选通过，但足以停止本地脚本和 kernel 结构上的盲改。

E4 没有重建 canonical ZIP、没有 preflight、上传或提交。平台事实仍是 E3 的
7/8 correctness、Enflame generic `0.0610x < 0.1x`、Kunlun compile-worker crash；
本地榜单快照为 147 submissions / 25 teams / 6 个达标队，榜首 EvokeAgent
`7.6239375x`。六个稳定过门槛芯片合计 `36.3245x`；若要严格超过榜首总和
`60.9915x`，Enflame+Kunlun 还需合计 `>24.667x`，所以 E4 只是 validity-first
载体，不是可宣称登顶的候选。

## E5：Enflame i32 地址 + Kunlun chunk-major 单步递推（2026-09-02）

状态：source/verification commit、exact Git-object release 与 canonical ZIP
门禁通过；已预注册为一次性 validity-first 候选，尚未上传或提交。

E5 冻结 E1 generic 的平台已证字节，只改两个被阻塞芯片：

- Enflame 以 E2 cap12 为基线，唯一新变量是删除 kernel 开头 18 个显式
  `tl.int64` stride cast；BLOCK256、grid cap12、两层循环、4 warps、1 stage、
  地址公式与数值语义保持不变。固定 GCU backend 的 `enable_i64=False` 与
  T13/T17/T22 同芯 i32 正证据支持该最小改动，但从 E1 的 `0.0610x` 到门槛仍需
  约 64% 提升，目标芯结果未知。
- Kunlun 不复用 E3 已崩溃 SHA `755f5b47...4075`。wrapper 先把 states 与
  dA-last 物化为 chunk-major contiguous 布局，再按 Python shape 顺序逐 chunk、
  每 65,535 rows 分段发射。单步 Triton kernel 只接收 states/dA/current/out 四个
  指针和 `DIM/BLOCK` constexpr，2D grid 直接映射 row 与 D tile；kernel 内无
  runtime chunk/grid loop、stride 参数、显式 i64、hint 或 XPU 私有 flag。每步先写
  pre-update snapshot，再以 FP32 执行 `cur * exp(dA) + state` 并原位更新 current；
  BLOCK256、4 warps、1 stage 不变。布局恢复由 wrapper 完成，核心递推每个 chunk
  都实际运行 Triton。
- 现有三 vendor 数值矩阵保留；capped-grid 回归改为
  `(B,C,H,D)=(257,2,256,1)`，同时跨过 `B*H=65792 > 65535` 与第二个 chunk，
  可捕获 row 分段和 running-state 错位。

### KernelGen 结构门

已通过配置的 `kernelgen-server.optimize_kernel` 运行五轮，全部
`mcp_isError=false`。首轮给出 host chunk 单步方向，但带未使用的 `nchunks` /
`IS_LAST_CHUNK` 与 eviction hints；第二轮收敛到可执行单步骨架；第三轮偏离为重算
前一 chunk，按 usability gate 拒绝；第四轮精确生成 Enflame 去 i64 版本；第五轮
生成四指针 chunk-major Kunlun 版本。落库仅删除第五轮冗余 hint/comment/contiguous
调用并按仓库格式收口，不改变其结构和数学。`generate_kernel -> Kunlun verifier`
仍沿用最小 `x+y` 三次 HTTP 502 的服务健康结论，故本轮 MCP 只作为结构生成/审查，
不记为目标芯验证证据。

### 构建身份

| 项目 | 值 |
| --- | --- |
| source / verification commit | `d980b8db2238085d1014e4643ba40a349546cbb0` |
| generic SHA-256 | `af0e622458f7ab89fadc45273c60f209bad5477fc8374de5e996d8193447c034`（=E1） |
| Enflame SHA-256 | `e265a12fab195c8c94a54d4c3b1bf6324f2127841cf7db00da6ce74790b0e448` |
| Kunlun SHA-256 | `a263d37733867e56f7ce004f8fa2e2a4a30088651e08bc0a877515e63ee0d889` |
| test SHA-256 | `e8eb6f67d27e6215bb8db2450e17114482f242c87f3b84a80b00ae69f628d2b7` |
| benchmark SHA-256 | `d5dec07565eaab4c7a59ddb60f0966ae8cb6e58e1457e586f1ddd684c3ee9f12`（=E1） |
| canonical ZIP | `artifacts/competition/state_passing/e5-d980b8d/state_passing.zip` |
| ZIP size / SHA-256 | `15816` bytes / `0e6bb8e512514b3f07716ed13d9045925a95d7209a034569838158f40576f492` |
| ZIP members | `state_passing.py`、`state_passing_enflame.py`、`state_passing_kunlunxin.py` |

打包器 dry-run/create/`--verify-existing` 三次身份一致；`unzip -t/-l`、唯一顶层
UTF-8 `.py` 成员、10 MB 上限和三个成员与 source commit 逐字节核对全部通过。

### 代理验证

- screening：`gpu-et:/tmp/flagos-state-passing-e5-screen.TdCtLM`，显式传入候选；
  py_compile、isort/flake8、逐文件哈希和 unittest **5/5** 通过。随后 Enflame 仅
  发生 Black 格式化字节变化，因此 screening 只作探索证据，不为最终 ZIP 背书。
- release：`gpu-et:/tmp/flagos-state-passing-e5-release.kxIkkD`，源码、测试、
  benchmark 全部从 commit `d980b8d` 的 Git 对象生成。runner / manifest /
  release log SHA-256 分别为
  `e5e61138540fcdf9b3bcc528ab8d96bb1ded5a1cc30f2120aa60db17eb7f49d1` /
  `2ca37b6fdb9ed65260ef322cc92c1907e81b22d5686bf8ab84d5ce1f766668af` /
  `1797f500367132666d0f628e8ab248be642a218c478fde5f19760b9e10f67c09`。
  Python 3.12.13、PyTorch 2.13.0+cu130、Triton 3.7.1、CUDA 13.0、RTX
  5070 Ti；改动文件 Black、全文件 py_compile/isort/flake8、末尾哈希复核和
  unittest **5/5** 全过；冻结 generic 保持 E1 原字节。
- release generic 四组 wrapper-inclusive 中位 speedup 为
  `3.362903x / 6.836346x / 3.565019x / 19.545939x`，平均 `8.327552x`。
  exact Kunlun vendor 四组为 `2.145164x / 2.979971x / 1.733660x /
  2.523657x`，平均 `2.345613x`，log SHA-256
  `c264819b8731d68b08b7ceaabb7c4437c243d4a5789536fd27f353e5508810e2`。
  exact Enflame vendor 四组为 `3.304710x / 2.127230x / 0.213619x /
  1.693844x`，平均 `1.834851x`，log SHA-256
  `b26f28743c4ba057a8f92ba0d0d43b0d1feb7ae8b442192e39780aa34ca5575d`。
  NVIDIA 只证明实际字节可 JIT、数值正确和代理性能过 `0.1x`，不外推 GCU/XPU。

### 单次提交预注册

2026-09-02 00:07:32 CST 实时状态：race `782kzq4m`、账号 `15600308080`、
团队 `SoulCoder`、batch 3、Task 41/`s2t1op041`、`competing/submitting`、
`can_submit=true`，额度 `30/30`，最小间隔 120 秒已满足。00:20 左右题目详情为
166 submissions / 27 teams / 6 个达标队，榜首 `wwwwww` 为 `7.8095x`；本队
尚无有效分数或排名。

E5 只允许上述 ZIP 上传和正式提交各一次；预期 Enflame 选择新 i32 vendor、Kunlun
选择新四指针 vendor，其余六芯继续选择冻结 generic。基础/晋级门为 8/8 correctness
且每芯 `>=0.1x`，两目标芯必须同时过门才转正。若 Kunlun 再现约 1830 秒且空
`failed_cases` 的 compile-worker crash，永久封存 chunk-major host-step 轴，不做
注释载体或同字节重投；若 Enflame 正确但 `<0.1x`，永久封存 cap12+i32 轴；数值
mismatch 只按明确 case 判断是否允许一次根因修复，不扫 BLOCK/flag/hint。

六个稳定芯片 E3 合计 `36.3245x`。严格超过当前榜首所需八芯总和 `62.476x`，因此
Enflame+Kunlun 需合计 `>26.1515x`；本发定位为恢复有效性而非宣称登顶。
