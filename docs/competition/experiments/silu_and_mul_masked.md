# Task 39 `silu_and_mul_masked` 实验记录

## 契约

- 签名：`reference(input, masked_m)`。
- `input` 为 BF16 `[E,T,H]`，`masked_m` 为整数 `[E]`，输出 BF16
  `[E,T,H//2]`。
- 仅计算每个专家前 `masked_m[e]` 行的
  `gate.float() * sigmoid(gate.float()) * up.float()`；填充行不验值。
- BF16 容差 `atol=rtol=1.5e-2`；支持八芯，核心计算必须走 Triton。

## S0：KernelGen 基线

状态：release 通过，canonical ZIP 已验签，待实时 preflight。

- 2026-08-30 通过 `kernelgen-server.generate_kernel` 生成两轮：首轮报告
  `41.188x` 但 65535 grid cap 后无步进，静态否决；第二轮补齐步进但嵌套
  `@triton.jit` 且代理仅 `0.123x`，静态否决。
- 随后调用 `optimize_kernel`，产物加入 autotune 并再次丢失 grid cap，静态
  否决。按 KernelGen 重试门禁停止继续生成。
- S0 仅保留第二轮的单 pass 算法，复用 T29 已八芯验证的固定
  `BLOCK_COL=1024`、capped 1D grid + kernel 内 grid-stride 骨架；无
  autotune、vendor 分支、fallback 或 host `.item()`。
- 输出用 `torch.empty`，因为填充行不参与正确性判定；有效行完整写入。
- 回归覆盖 int32/int64 mask、0/1/T-1/T、1024 列尾块、三维非连续输入、
  空维、NaN/Inf/极值、输入与 mask 不变性，以及
  `total_blocks > 2 * 65535` 的三轮折叠路径。

### Screening 身份与正确性

| 项目 | 值 |
| --- | --- |
| base commit | `14d5133` |
| source SHA-256 | `bdafd313c6bb841a3334eca33e7bd1637c110d5edbf2c0180c00b127820c9cad` |
| test SHA-256 | `7037aac3c9f8410542b08acd65d70482d6f531855a49505ebabbb4c690ea5c85` |
| 远端目录 | `gpu:/tmp/flagos-silu-and-mul-masked.NeO5UW`，mode 0700 |
| screening 日志 SHA-256 | `3411ebf555fb25c3c5a2c086ae90aab848e6bad4135cadf70e2027d277133733` |

远端 RTX 5070 Ti 16 GB、Python 3.12.13、PyTorch 2.13.0+cu130、
Triton 3.7.1：py_compile、Black 79、isort 80、flake8 全过，unittest
**5/5** 通过，远端复验 source/test 哈希不变。

特殊值审查曾提出把负半轴改成稳定 sigmoid；代理反例实证题面原样的
`torch.sigmoid(float32)` 对 `-92/-90` 返回 0，半指数改写反而分别输出
约 `-3.46/-25` 而不匹配 oracle，故回退到与题面参考一致的
`gate / (1 + exp(-gate))`。诊断目录
`gpu:/tmp/flagos-silu-numeric.fV6zJu`，日志 SHA-256
`a5fc99a10e9b835c2fc1bd03979e309c89f8a1959fd807d14e452461e429b6ae`。

### NVIDIA 代理性能

目录 `gpu:/tmp/flagos-silu-and-mul-masked.u0iZ1U`；benchmark 脚本
SHA-256 `854c4df1ad3764c1367ab9ab744c86307cfb163fe4c6244f6a78c81d465e76b5`，
完整 AB/BA 原始样本日志 SHA-256
`00a1efcf08231deba2cc9afae11009db2b040fae7bebe881c95f02486c388a96`。
运行前 GPU 无竞争进程；wrapper-inclusive、5 轮交替、p50：

| E×T×half | BLOCK 512 | BLOCK 1024 | BLOCK 2048 | BLOCK 4096 |
| --- | ---: | ---: | ---: | ---: |
| 8×64×512 | 25.21x | **26.19x** | 25.74x | 25.73x |
| 64×128×2048 | 43.14x | 61.13x | **82.00x** | 49.50x |
| 256×64×3584 | 32.14x | **32.27x** | 31.99x | 31.81x |

最大 input 224 MiB；S0 选择 BLOCK 1024：两组最优、最大 shape 与 2048
持平，且同骨架已有 T29 八芯实证。2048 只作为平台逐芯结果后的单变量候选。
NVIDIA 结果仅为代理证据，不外推八芯。

### Release 与不可变 ZIP

| 项目 | 值 |
| --- | --- |
| source/verification commit | `bd5bf8b040b934797a7686bddef06b0093dc3481` |
| release 目录 | `gpu:/tmp/flagos-silu-release.K5JLHZ`，mode 0700 |
| release 日志 SHA-256 | `fd37e5cc7d057a68f73d2b44a5c8418f86e22c5402134d45e9f70d3e7e0d85f0` |
| ZIP | `artifacts/competition/silu_and_mul_masked/s0-bd5bf8b/silu_and_mul_masked.zip` |
| ZIP SHA-256 | `cc9da72e2ad6c551aeba2eac74dbe2d7882d3f489b285601223b926f2f9815e0` |
| ZIP 内容 | 顶层 `silu_and_mul_masked.py`，2708 bytes；ZIP 2850 bytes |

release 文件全部由 Git 对象生成；source/test 与 screening 哈希一致。lint、
unittest 5/5、末尾哈希复核全过。BLOCK 1024 的 release speedup p50 为
25.88x / 61.31x / 32.28x，与 screening 一致。打包器 create 后
`--verify-existing` 通过，actual 与 canonical ZIP SHA-256 一致，单成员、
普通 UTF-8 Python 文件、远低于 10 MB。

### 已知边界

- 题面参考隐含 `H` 为偶数；奇数宽度时参考自身 gate/up 维度不匹配。
- grid-stride 采用与既有八芯实现相同的 int32 tile 归纳变量；仅在接近
  `2^31` tiles 或单行 `H >= 2^31` 的非常规不可分配 shape 存在理论索引
  溢出，不为该题合理 MoE shape 增加全 int64 成本。

## 平台记录

尚未提交。preflight tuple：season 2、race `782kzq4m`、account
`15600308080`、team `SoulCoder`、batch 3、task 39、tid `s2t1op039`、
operator `silu_and_mul_masked`、stage `s0`、commit
`bd5bf8b040b934797a7686bddef06b0093dc3481`、member
`silu_and_mul_masked.py`、ZIP SHA-256
`cc9da72e2ad6c551aeba2eac74dbe2d7882d3f489b285601223b926f2f9815e0`。
实时门禁全部匹配即按项目授权执行一次性 submit。
