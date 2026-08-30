# Task 41 `state_passing` 实验记录

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

状态：Kunlun-only 单变量候选已唯一一次提交；7 个非昆仑芯片均通过正确性，
昆仑等待回调；E3 不得重试。

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
| 昆仑 | - | 等待回调 | Kunlun vendor |
| 华为 | `1.0295x` | 通过 | generic |
| 国际 A | `8.4260x` | 通过 | generic |
| 国际 B | `4.5855x` | 通过 | generic |

7 个已返回速度的简单平均为 `5.197929x`，仅作排障观察；平台 validity 与平均仍待
昆仑终态。heartbeat 已绑定本次 file URL hash；等待期间不改候选、不发起新提交。
