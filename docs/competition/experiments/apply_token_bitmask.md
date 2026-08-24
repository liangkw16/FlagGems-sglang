# Task 08 `apply_token_bitmask` 实验记录

## S0：generic baseline

当前状态：E2 已提交平台并通过 8/8；平均 4.686925x，第 12/13 名

验证时间：2026-08-24 01:22–01:28 CST

源码 commit：`3fac516`

### 契约

| 项目 | 值 |
| --- | --- |
| Task / batch | 08 / 第二批 |
| 公开接口 | `apply_token_bitmask(logits, bitmask)` |
| `logits` | `[B, V]`，float16 / bfloat16 / float32；按真实二维 stride 读取 |
| `bitmask` | `[B, ceil(V/32)]`，int32；按真实二维 stride 读取 |
| 计算 | 第 `v` 位为 0 时输出 `-inf`，否则保留原 logit |
| 输出 | 与 `logits` 同 shape、同 dtype，out-of-place；不修改两个输入 |
| 支持芯片 | 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用 A/B，共 8 款 |
| 截止时间 | 2026-08-27 19:59:59 |
| 赛题门槛 | `speedup_threshold=0.1` |

题面来源为本地
`docs/competition/tasks/batch-2/08-apply_token_bitmask.md`。2026-08-24
01:22 CST 重新读取公开 API 时为 103 次提交、17 支队伍、11 支达到门槛，
榜首 8/8、平均 11.929875x；动态值仅用于当时决策。

固定上游为 SGLang commit
`8014d9d062c3cc5d393596ecdf2f7009191965df` 的
`python/sglang/kernels/ops/grammar/bitmask_ops.py`。上游实现是原地写、有可选
indices，并依赖设备 SM 数；本题只复用 int32 bit 位布局，不复制其接口、设备
策略或连续 stride 假设。

### 唯一候选配置

- 单个 generic Triton kernel；BLOCK 256、4 warps、1 stage，无 autotune。
- 一维 grid 中每个 program 处理一行的一个 token 块；logits、bitmask、输出
  都显式使用真实二维 stride。
- `token < V` 同时保护 logits、bitmask 和输出，覆盖 `V` 非 32/256 倍数。
- 位判定严格使用 `(packed >> (token % 32)) & 1`。int32 算术右移后再取最低
  位与 Torch reference 一致，包括负数的 bit31。
- `torch.empty_like` 只负责分配 out-of-place 输出；非空输入的计算路径必经
  Triton。无 try/except、设备判断、Torch fallback、vendor 文件或私有 API。

### 验证证据

| 项目 | 值 |
| --- | --- |
| 源文件 SHA-256 | `5da3d966936c919cd4b0fab2c32ecc66526eb375c3cdc20a2e3f2f37cddb697c` |
| 测试 SHA-256 | `b4e386f542f0015c0bfed3ed308586bd9eae2311ce030ae4cf9d78ea20a3ec65` |
| ZIP | `artifacts/competition/apply_token_bitmask/s0-3fac516/apply_token_bitmask.zip` |
| ZIP SHA-256 | `394d287484e04c62eba5deea0c3f698787b1bd053ee7803598a7e9c98567a4b7` |
| ZIP manifest | 顶层 `apply_token_bitmask.py`，2458 bytes |
| 远端证据目录 | `gpu:/tmp/flagos-batch2.SQaIX2` |
| 远端环境 | RTX 5070 Ti 16 GB；PyTorch 2.13.0+cu130；Triton 3.7.1；CUDA 13.0 |

- 本地 `py_compile` 与 `git diff --check` 通过。
- 远端 unittest 4/4 通过，源码与测试的本地/远端 SHA-256 完全一致。
- Black 79、isort 和 flake8 通过。
- 回归覆盖 float16、bfloat16、float32，空 B/V，多 batch、多 256-token 块，
  `V=33/35/513`，非连续 logits/bitmask，负 int32 的 bit31，输入不变性和
  out-of-place 输出。
- wrapper-inclusive 代理 benchmark 覆盖 `B=8,V=32768` 与
  `B=16,V=131072` 的三 dtype；相对 Torch reference 为 `3.333x–5.154x`。

### 已知风险与下一步

- RTX 5070 Ti 只能证明 NVIDIA 代理路径，不能证明其余七款芯片。
- 公开题面没有 shape/benchmark 矩阵；当前物理 grid 为
  `B * ceil(V / 256)`。若平台大用例超过某芯片 grid.x 上限，只为失败芯片加
  固定上限的 grid-stride vendor 实现，不预先复制八份代码。
- 八芯正确性成立后，第一个单变量候选是每次只加载一个 int32 word 并展开
  32 bit，减少重复 bitmask load。
- ZIP 由 commit `3fac516` 的算子目录直接通过 `git archive` 生成；`unzip -t`、
  UTF-8、单一 `.py`、10 MB、basename 和 ZIP 内源码哈希门禁均通过。
- 本节记录发布门禁完成时的状态，当时尚未上传；实际首投与额度变化见 S0c。

## S0 发布复核与 E1 word-expansion 负实验

状态：边界回归已扩展；E1 正确但无端到端收益，已撤回；S0 源码与 ZIP 不变

复核时间：2026-08-24 03:06–03:09 CST

| 项目 | 值 |
| --- | --- |
| source commit | `3fac516a8d64c88b183801668a7857d969a05e37` |
| verification commit | `1197a410b1cbdaa6ab138c37b2e13225f6e0b195` |
| S0 源码 SHA-256 | `5da3d966936c919cd4b0fab2c32ecc66526eb375c3cdc20a2e3f2f37cddb697c` |
| 当前测试 SHA-256 | `78cfb2fb10c97e54d70877178391a181d44edad0565fd2fa8f12cecd73ebb967` |
| S0 ZIP SHA-256 | `394d287484e04c62eba5deea0c3f698787b1bd053ee7803598a7e9c98567a4b7`，`verified-existing-legacy` |
| 规范 ZIP SHA-256 | `f4068dd290bb16821d75eb669485b5607bf8cd3a8f4b1807af2e67aa23d41a21`，已生成 S0c |
| E1 临时源码 SHA-256 | `6d74aafacc53922890ae5e3041231eb1860851a59de3467c75d00206a4ae044e` |
| 远端证据目录 | `gpu:/tmp/flagos-task08.rF1D2s`，mode 0700 |
| baseline 门禁 | PID `71769`；03:07:08 CST；`baseline-release-gates.log`；SHA-256 `aaf6f11860fb7824bbfb74cfbb15c0c2401c7a338262458cb63b1846c686ea41` |
| E1 门禁 | PID `71871`；03:07:44 CST；`candidate-gates.log`；SHA-256 `7a0edad1f3b31137b720c167e1e6fdf6a40815b7750ffa79aaf7f5ca34eb5888` |
| A/B | PID `71982`；03:08:52 CST；`ab.log`；SHA-256 `7dcca056416316684c20f33394150b527a83412a2e4bc73e40a93598bcab24ef` |
| 平台结果 | 未提交；逐芯结果、均值、排名和实时额度均为 N/A |

新增第五个 unittest 方法，覆盖三 dtype × `V=31/32/33/255/256/257`、bit31
负 int32、输入不变性和 out-of-place 语义。原有非连续 stride、空维度和多 block
回归保留；S0 与 E1 均通过 py_compile、Black 79、isort、flake8 和 5/5 unittest。
远端 S0/E1/测试哈希与上表完全一致。E1 另在三 dtype 下完整校验
`(B,V)=(1,32000)/(8,32768)/(16,131072)`，全部与 reference 精确相等。

E1 只改变 packed mask 表示：每个 program 仍覆盖 256 token，grid、4 warps、
1 stage、stride、tail mask 和 wrapper 不变；把 token 组织成 `[8,32]`，加载 8 个
int32 word 后沿 32 bit 广播。编译证据确认预期变化：TTIR bitmask load 从
`tensor<256xi32>` 缩到 `tensor<8xi32>`，选定 PTX 的 bitmask `ld.global` 从 2 条
降到 1 条。unit 产生的编译变体中，S0 为 14–18 registers，E1 为 14–22；两者
均为 stack/shared/local 0，说明部分布局还增加了寄存器压力。

wrapper-inclusive 五组轮换 A/B，组内 `warmup=25, rep=100`：

| dtype | `(B,V)` | S0 ms | E1 ms | S0 / E1 | reference / E1 |
| --- | --- | ---: | ---: | ---: | ---: |
| FP16 | `(1,32000)` | 0.004293 | 0.004263 | 1.0072x | 4.5401x |
| FP16 | `(8,32768)` | 0.006461 | 0.006459 | 1.0002x | 3.8797x |
| FP16 | `(16,131072)` | 0.015761 | 0.015770 | 0.9994x | 4.7999x |
| BF16 | `(1,32000)` | 0.004302 | 0.004316 | 0.9967x | 4.5507x |
| BF16 | `(8,32768)` | 0.006413 | 0.006435 | 0.9965x | 3.8798x |
| BF16 | `(16,131072)` | 0.015788 | 0.015776 | 1.0008x | 4.7985x |
| FP32 | `(1,32000)` | 0.004322 | 0.004358 | 0.9917x | 4.5421x |
| FP32 | `(8,32768)` | 0.006432 | 0.006477 | 0.9930x | 4.0341x |
| FP32 | `(16,131072)` | 0.026628 | 0.026659 | 0.9989x | 3.5530x |

九点 S0/E1（E1 speedup）几何平均为 `0.998267x`，最差回归 `0.833%`。它没有达到预设的
`>=1.05x` 晋级线，而且部分变体寄存器增加，因此不提交 E1，也不生成新 ZIP。
工作树源码已恢复到 S0 SHA-256；扩大后的测试继续保留。下一次源码迭代等待 S0
八芯结果或新的固定来源，不在 NVIDIA 上继续微调同一路径。

## S0c：canonical 首投包与平台结果

状态：已提交；`invalid_correctness`，6/8，平均值与排名 N/A

生成时间：2026-08-24 17:20 CST

不改 kernel，复用 source commit
`3fac516a8d64c88b183801668a7857d969a05e37` 与 verification commit
`1197a410b1cbdaa6ab138c37b2e13225f6e0b195` 的 5/5 release 证据。当前源码与测试
SHA-256 仍分别为
`5da3d966936c919cd4b0fab2c32ecc66526eb375c3cdc20a2e3f2f37cddb697c`、
`78cfb2fb10c97e54d70877178391a181d44edad0565fd2fa8f12cecd73ebb967`，因此不重复消耗
远端 GPU 做同字节验证。

canonical ZIP 为
`artifacts/competition/apply_token_bitmask/s0c-3fac516/apply_token_bitmask.zip`，
2600 bytes，SHA-256
`f4068dd290bb16821d75eb669485b5607bf8cd3a8f4b1807af2e67aa23d41a21`；唯一成员
`apply_token_bitmask.py` 为 2458 bytes，成员 SHA-256 与 source commit 完全一致。
确定性打包器 dry-run、生成后 `verified-existing` 和 `unzip -t` 均通过。旧 S0 legacy
ZIP 保持原字节不覆盖，平台候选只使用本节 S0c。

2026-08-24 17:26:16 CST 在账号 `15600308080`、团队 `SoulCoder` 下提交一次，
submission ID `4170`、daily seq `5`。平台远端 ZIP 为 2600 bytes，下载后
SHA-256 与本地 `f4068dd290bb16821d75eb669485b5607bf8cd3a8f4b1807af2e67aa23d41a21`
一致；`file_url_sha256` 为
`a8318bb48e09ac02022de7f9e774f83fcadf3a626894dfb3d3b1265f55103b14`。
实时额度由 `11/15` 变为 `10/15`。

| 芯片 | 结果 | speedup | 选中文件 |
| --- | --- | ---: | --- |
| 天数 | 通过 | 9.8134x | `apply_token_bitmask.py` |
| 沐曦 | 通过 | 4.9598x | `apply_token_bitmask.py` |
| 燧原 | 失败 | N/A | `apply_token_bitmask.py` |
| 海光 | 通过 | 4.8052x | `apply_token_bitmask.py` |
| 昆仑芯 | 通过 | 0.6448x | `apply_token_bitmask.py` |
| 华为 | 失败 | N/A | `apply_token_bitmask.py` |
| 国际通用 A | 通过 | 6.7162x | `apply_token_bitmask.py` |
| 国际通用 B | 通过 | 6.1658x | `apply_token_bitmask.py` |

燧原 correctness case 6/7 分别要求 `grid.x=304128/2433024`，超过硬件上限
65535。华为 case 6 同样因 `coreDim=304128` 超过 65535 导致 kernel launch 失败；
输出比较错误只是异步 launch 错误的外层表现。其余六芯全部通过且均高于 0.1x，
因此下一轮保持 generic 字节不变，只为这两芯增加 capped grid-stride vendor。

## S1：燧原/华为 capped grid-stride 修复

状态：已提交；`valid`，8/8，平均 4.3870x，第 12/13 名

源码与验证 commit：`c33d45fb47e6dce7a72d440b3f4eca4dfe486d6a`

generic 源码逐字节保持 S0 不变。只增加两个自包含 vendor 文件：燧原使用
`grid=(min(total_blocks, 12),)`，华为使用
`grid=(min(total_blocks, 48),)`；kernel 以 `tl.num_programs(0)` 为步长遍历逻辑
block，BLOCK 256、4 warps、1 stage、stride、tail mask 和计算公式均不变。固定
backend 证据来自 Triton-Ascend commit
`865691e2e9b656bc58008170207b4108d92e8dd1` 与 FlagGems commit
`ed2508bcb5a03000e9774734201d840ba362cd11`：华为二维 grid 会展平为总
`coreDim`，因此不使用二维拆分；12/48 也与 Task 24 已恢复两芯通过的平台模式一致。

回归新增 `(B,V)=(256,65537)`，共 65792 个逻辑 block，能让旧的一 program/block
实现越过 65535，并对两个 vendor 与 Torch reference 做 FP16 精确比较。screening
和从 commit Git 对象生成的 release 均通过 py_compile、Black、isort、flake8 与
6/6 unittest；最大代理 shape 即 `(256,65537)`。screening 的三份源码与测试
SHA-256 也和晋级后的 `c33d45f` Git objects 逐项一致。

| 项目 | 值 |
| --- | --- |
| generic SHA-256 | `5da3d966936c919cd4b0fab2c32ecc66526eb375c3cdc20a2e3f2f37cddb697c` |
| 华为 vendor SHA-256 | `7c4daf2fa5774dcbf0c9891b787c77d18d4d0766ae94292ed38347172ead3fd8` |
| 燧原 vendor SHA-256 | `69bfed8aeb81e36402406d2f5d5727cfd924cd589f0a2f1e856f4fadffe260bc` |
| 测试 SHA-256 | `981f91d58cc0e04914244f0c4d00cf6086c990e378ea97fb487311b9651b90f1` |
| screening | `gpu:/tmp/flagos-apply-token-bitmask.SqYnGH`；PID/PGID `98490`；`screening.log` SHA-256 `2b03ec9cb0d1f28b210593420e7abfacf146a840e198cce1ddfc6fa4d4dc6da8`；6/6，0.803s |
| release | `gpu:/tmp/flagos-apply-token-bitmask-release.o2UHwd`；PID/PGID `99223`；`release.log` SHA-256 `2afad9a11806e2626f9a9c4c71104b7c5c2f5d5ea7c870665ce9c39700f95e93`；6/6，0.551s，`RELEASE_OK` |
| 代理环境 | RTX 5070 Ti 16 GB；driver 610.57.04；Python 3.12.13；PyTorch 2.13.0+cu130；Triton 3.7.1；CUDA 13.0 |
| 编译资源 | 2026-08-24 18:05:41 CST；同一 release 目录 `resource.log`，mode 0600、700 bytes、SHA-256 `a5b4d75f515a629fc88abf443ec53ded307e5a58f805442bfca1f4386d5c3e41`；两个 vendor 均 26 registers、0 spill、0 shared、0 global scratch |
| ZIP | `artifacts/competition/apply_token_bitmask/s1-c33d45f/apply_token_bitmask.zip`，8382 bytes |
| ZIP SHA-256 | `c6304895495de8aa601ba61acff6b18f0e3f3cc45158ec1b0aa90856927a60c6` |

ZIP 由确定性打包器从上述 commit 生成并以 `verified-existing`、`unzip -t` 复验。
成员为 `apply_token_bitmask.py` 2458 bytes、`apply_token_bitmask_ascend.py` 2756
bytes、`apply_token_bitmask_enflame.py` 2756 bytes，成员哈希与上表源码完全一致。
NVIDIA release 只能证明代理编译和数值，不能证明两款目标 runtime；平台晋级门禁为
8/8、两芯都选中对应 vendor 且各自不低于 0.1x。若燧原恢复正确但性能偏低，下一轮
才单变量尝试 BLOCK 256 → 4096。

2026-08-24 19:14:49 CST 在同一账号和团队下提交一次，submission ID `4207`、
daily seq `6`。平台远端 ZIP 为 8382 bytes，下载后 SHA-256 与本地
`c6304895495de8aa601ba61acff6b18f0e3f3cc45158ec1b0aa90856927a60c6` 一致；
`file_url_sha256` 为
`5d39b0a2c98a4b27f59ab754a266c3c61710a8eb6c56b59cd7c13aba3749eb32`。
实时额度由 `10/15` 变为 `9/15`。

| 芯片 | 结果 | speedup | 选中文件 |
| --- | --- | ---: | --- |
| 天数 | 通过 | 10.2536x | `apply_token_bitmask.py` |
| 沐曦 | 通过 | 4.7580x | `apply_token_bitmask.py` |
| 燧原 | 通过 | 0.4292x | `apply_token_bitmask_enflame.py` |
| 海光 | 通过 | 4.9440x | `apply_token_bitmask.py` |
| 昆仑芯 | 通过 | 0.7674x | `apply_token_bitmask.py` |
| 华为 | 通过 | 0.9710x | `apply_token_bitmask_ascend.py` |
| 国际通用 A | 通过 | 6.7830x | `apply_token_bitmask.py` |
| 国际通用 B | 通过 | 6.1898x | `apply_token_bitmask.py` |

两份 vendor 均被正确选中，原 grid/coreDim 超限消失，八芯全部高于 0.1x。官方
Task 08 leaderboard 在 2026-08-24 19:23:43 CST 显示 `SoulCoder` 第 12/13 名，
本次也是团队首次有效及截至当时的最佳提交。当时的下一条单变量假设只改燧原
BLOCK 256 → 4096，
保持 grid 12 和其余文件不变；先做代理验证，不沿用本次已消耗的提交确认。

## E2：燧原 BLOCK 4096

状态：平台有效，8/8，团队当前最佳

源码与验证 commit：`86fca8738c850a9d1b83ff3a9ace06d71cc9f6cf`

E2 只把燧原 vendor 的 `block_size` 从 256 改为 4096；grid 上限仍为 12，generic、
华为 vendor、公式和 launch 其余参数逐字节不变。固定 FlagGems commit
`ed2508bcb5a03000e9774734201d840ba362cd11` 的 Enflame pointwise policy 允许
4096 tile / grid 12 / 4 warps；Task 24 同型代理结果支持该方向，但都不是本题真实
GCU 性能保证。

新增 Enflame 三 dtype 回归使用 `V=12*4096+17`，覆盖第二轮 grid-stride 和 17-token
tail。最终 screening 与从 commit Git 对象生成的 release 均通过 py_compile、Black、
isort、flake8 和 7/7 unittest；screening 的源码与测试 SHA-256 和 Git objects 完全
一致。第一次 screening 因临时目录没有仓库 `pyproject.toml`、Black 使用默认行宽而在
测试前停止；修正为显式仓库行宽 79 后通过，未因此改源码。

| 项目 | 值 |
| --- | --- |
| generic SHA-256 | `5da3d966936c919cd4b0fab2c32ecc66526eb375c3cdc20a2e3f2f37cddb697c` |
| 华为 vendor SHA-256 | `7c4daf2fa5774dcbf0c9891b787c77d18d4d0766ae94292ed38347172ead3fd8` |
| 燧原 vendor SHA-256 | `0aa6cd79e37408623eeded1d123b23b114b973f63c9b526f1fe3e1a56cb7b380` |
| 测试 SHA-256 | `876fe5cbf46cbf50ecde5f72c94d9d2646e14970245b4fbdf1f91dcd57dee510` |
| screening | `gpu:/tmp/flagos-apply-token-bitmask-e2.98iRfU`；600s wall limit；19:30:15 首次 PID/PGID `100046`、脚本 SHA-256 `425f410e0f10a149b0c5b7d1b3922221acde6aa10db5f0f71a8ee626a6631697`，配置失败日志 SHA-256 `9db83b32433cba6b4b976212b3551a2a94496c6eac1ad6de532dc39618c9521c`；19:31:40 r2 PID/PGID `100121`、6/6、0.634s，日志 SHA-256 `38fc345b9dca230a88895d5096f208713eeed1c660277401403fa5e4f64b800d`；新增回归后 19:36:41 r3 PID/PGID `100424`、脚本 SHA-256 `2dab4b84bc3c5a665bb472e5fd201fe83aabb389a39ceaf2a968d20a9774a49d`，7/7、0.804s，日志 SHA-256 `18c336f7a862d1ce97434cfdeb23442c031872c467c460ab0e80345b94f8e929` |
| screening A/B | 19:34:16；900s wall limit；PID/PGID `100277`；脚本 SHA-256 `946f0ad12ff7de7463d7dc96a30005501d53338da7035e2ac3f9a21571328b12`；`benchmark.log` SHA-256 `d771a93a75b280dcc217011bf7fe5b635f5bd3507222908c8f4c44a400f75b88`；affected 几何平均 4.790304x |
| release | `gpu:/tmp/flagos-apply-token-bitmask-e2-release.HMJ520`；19:38:38；600s wall limit；PID/PGID `100638`；脚本 SHA-256 `399d0290a6510174e730d40bbea2a28990932c489defe8f052cacc8b0dc74abc`；`release.log` SHA-256 `b3cdd82b6c195df5a1a15bf295e58db13cce276a538ceb444307659eff642fab`；7/7，0.547s，`RELEASE_OK` |
| A/B | 19:38:58；900s wall limit；release PID/PGID `100742`；脚本 SHA-256 `946f0ad12ff7de7463d7dc96a30005501d53338da7035e2ac3f9a21571328b12`；`benchmark.log` SHA-256 `e9c3ae8187b8ca467a9c45d159a4e804cf4379218d3268cdcc437ae6902b6dbd` |
| 代理环境 | RTX 5070 Ti 16 GB；driver 610.57.04；Python 3.12.13；PyTorch 2.13.0+cu130；Triton 3.7.1；CUDA 13.0 |
| 编译资源 | 本次 A/B 变体中 S1 为 20–23 registers、E2 为 48–76；两者均 0 spill、0 shared、0 global scratch，4 warps、1 stage |
| ZIP | `artifacts/competition/apply_token_bitmask/e2-86fca87/apply_token_bitmask.zip`，8383 bytes |
| ZIP 成员 | `apply_token_bitmask.py` 2458 bytes；`apply_token_bitmask_ascend.py` 2756 bytes；`apply_token_bitmask_enflame.py` 2757 bytes |
| ZIP SHA-256 | `88d2e8387ac2e7de785cf1574ad9c762df54c0baa79e4ada67fad7252987c1dc` |

wrapper-inclusive A/B 在无竞争 workload 时按 AB/BA 交替五组，每组
`warmup=25, rep=100`。control `(B,V)=(12,256)` 的三 dtype 均为 `1.000000x`；
affected 代理 shape `(512,152064)` 的 BLOCK256 `total_blocks=304128`，与平台 S0c
失败值相同，但不据此声称它就是 hidden case。候选输出均与 S1 精确一致：

| dtype | 五组 S1/E2 speedup | 中位数 |
| --- | --- | ---: |
| FP16 | 4.703450 / 4.704456 / 4.704211 / 4.697376 / 4.703442 | 4.703450x |
| BF16 | 4.704274 / 4.703711 / 4.703703 / 4.702181 / 4.703957 | 4.703711x |
| FP32 | 4.963269 / 4.961438 / 4.963846 / 4.968179 / 4.961059 | 4.963269x |

affected 三 dtype 几何平均为 `4.788597x`，超过 `>=1.05x` 晋级线，control 也高于
`>=0.97x` 门禁。规范 ZIP 的三个成员与 release 前 manifest、commit 源码和上述哈希
完全一致，并通过 `verified-existing` 与 `unzip -t`。平台门禁为继续 8/8、燧原选中
vendor 且从 S1 的 0.4292x 提升到至少 0.4507x；否则保留 S1。

2026-08-24 19:53:06 CST 在账号 `15600308080`、团队 `SoulCoder` 下提交一次，
submission ID `4215`、daily seq `7`，实时额度由 `9/15` 变为 `8/15`。提交命令
未设置可信对象存储 hostname，因此内置远端验签为 `unavailable`；随后从已核实的
`flagos.ks3-cn-beijing.ksyuncs.com` 无认证下载同一对象，得到 8383 bytes，SHA-256
与确认值 `88d2e8387ac2e7de785cf1574ad9c762df54c0baa79e4ada67fad7252987c1dc`
一致。`file_url_sha256` 为
`60844196c1021250756fd87df74561490bdbb4049f5a9f073e92d4713f9c8472`。

| 芯片 | 结果 | speedup | 选中文件 |
| --- | --- | ---: | --- |
| 天数 | 通过 | 10.0010x | `apply_token_bitmask.py` |
| 沐曦 | 通过 | 4.7746x | `apply_token_bitmask.py` |
| 燧原 | 通过 | 2.8510x | `apply_token_bitmask_enflame.py` |
| 海光 | 通过 | 4.9626x | `apply_token_bitmask.py` |
| 昆仑芯 | 通过 | 0.7686x | `apply_token_bitmask.py` |
| 华为 | 通过 | 0.9646x | `apply_token_bitmask_ascend.py` |
| 国际通用 A | 通过 | 6.9462x | `apply_token_bitmask.py` |
| 国际通用 B | 通过 | 6.2268x | `apply_token_bitmask.py` |

E2 平均 `4.686925x`，较 S1 提升 `0.299925x`（6.8367%）；燧原由 `0.4292x`
升至 `2.8510x`，为原来的 6.6426 倍，单变量门禁通过。官方 Task 08 leaderboard
在 2026-08-24 19:56:01 CST 显示 `SoulCoder` 第 12/13 名，本次为团队当前最佳。
保留 E2，不再继续同一燧原 BLOCK 假设；后续额度优先用于其他算子。
