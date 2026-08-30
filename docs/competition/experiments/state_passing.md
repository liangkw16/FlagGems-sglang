# Task 41 `state_passing` 实验记录

## S0：KernelGen generic 基线

状态：commit-bound release 与 canonical ZIP 门禁通过，待一次性平台提交。

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
