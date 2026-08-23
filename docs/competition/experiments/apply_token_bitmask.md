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
