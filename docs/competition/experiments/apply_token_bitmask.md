# Task 08 `apply_token_bitmask` 实验记录

## S0：generic baseline

状态：S0 已打包并通过本地门禁；等待当次上传确认

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
- 尚未上传，也未消耗提交额度；必须由用户针对上述 ZIP 路径和 SHA-256
  当次确认后才能上传。

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
| 规范 ZIP SHA-256 | `f4068dd290bb16821d75eb669485b5607bf8cd3a8f4b1807af2e67aa23d41a21`，仅内存生成 |
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
