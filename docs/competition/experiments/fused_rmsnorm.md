# Task 19 `fused_rmsnorm` 实验记录

## 契约

- 接口：`fused_rmsnorm(x, weight, eps)`。
- 公式：`x * rsqrt(mean(x * x, dim=-1) + eps) * weight`。
- `x`、平方和、均值、`rsqrt` 和权重乘法均按 FP32 计算，输出转换回
  `x.dtype`；输入保持不变。
- 题面支持 FP16、BF16、FP32，容差分别为 `1e-2`、`1.5e-2`、`1e-4`。
- 支持八类芯片；最低加速比为 0.1x。核心路径必须使用 Triton/TLE，禁止
  设备判断、异常 fallback 和纯 PyTorch 实现。

固定参考：SGLang `8014d9d` 的 `elementwise.py` 第 139–188 行，以及
FlagGems `ed2508b` 的 `_fused_rms_norm.py` 第 31–67 行。SGLang 的设备分支、
autotune 和最高 32 warps 未进入 generic 首版。

## S0：generic single-row baseline

状态：S0 已打包并通过本地门禁；等待当次上传确认

验证时间：2026-08-24 01:23–01:28 CST

源码 commit：`3fac516`

### 构建身份

| 项目 | 值 |
| --- | --- |
| 源文件 | `src/flaggems_sglang/ops/fused_rmsnorm.py` |
| 源文件 SHA-256 | `02bed1a5cb28b583c343892569d9e25d1ef3d888e124fdd066d1155a0b964997` |
| 测试文件 | `tests/test_fused_rmsnorm.py` |
| 测试 SHA-256 | `e77dd1a17f6df51b310eb145fec621a49af5e8dcf7f0dbca97f2a8032b608f91` |
| ZIP | `artifacts/competition/fused_rmsnorm/s0-3fac516/fused_rmsnorm.zip` |
| ZIP SHA-256 | `93780caf704341737ddfe5925cfacdcd7115ccefc2f38edf3c7ff006716d1820` |
| ZIP manifest | 顶层 `fused_rmsnorm.py`，2337 bytes |

### 唯一候选配置

- 一行一个 Triton program，`BLOCK_SIZE=next_power_of_2(hidden_size)`，
  8 warps、1 stage。
- 显式使用输入行/列 stride 和权重 stride；输出为连续、同 shape、同 dtype
  tensor。高维非连续输入在 wrapper 中 reshape，必要时由 PyTorch 生成布局副本，
  核心归一化仍只由 Triton kernel 完成。
- 尾部完整 mask；空输入直接返回空输出。
- 无 autotune、vendor 文件、设备判断、异常捕获或 PyTorch 计算 fallback。

### 验证

- `py_compile`、AST 解析、Black 79、isort、flake8 和 `git diff --check`
  已通过。
- 公开接口测试先于实现落盘；覆盖 FP16/BF16/FP32、hidden 513/8193、
  非连续 `x/weight`、输入不变性、输出 shape/dtype 和空输入。
- 本机没有 PyTorch/Triton/GPU，不能执行数值测试。远端 `gpu` 使用 RTX 5070 Ti
  16 GB、driver 610.57.04、Python 3.12.13、PyTorch 2.13.0+cu130、
  Triton 3.7.1、CUDA 13.0、compute capability 12.0。
- 远端源码与测试 SHA-256 和本地一致。执行 `tests/test_fused_rmsnorm.py -v`
  为 2/2 unittest 方法通过；循环内覆盖 6 个 dtype/hidden 组合，运行 1.291 秒。
- wrapper-inclusive 代理 benchmark 覆盖 `(rows, hidden)=(128,4096)、
  (32,8193)、(512,1024)` 的三 dtype；相对 Torch reference 为
  `1.830x–4.720x`。

### 已知风险

- NVIDIA 代理只验证语法、数值与候选性能，不能证明其余七类后端正确或达标。
- 大 hidden 会把整行放入一个 power-of-two block；8193 已纳入回归，但更大隐藏维
  可能出现寄存器/本地内存压力，应先看隐藏 harness 或平台结果再决定分块两阶段方案。
- 当前只验证题面常见的一维 weight（元素数等于 hidden）；没有为未公开的广播形状
  增加推测性分支。
- ZIP 由 commit `3fac516` 直接生成；`unzip -t`、UTF-8、单一 `.py`、10 MB、
  basename 和 ZIP 内源码哈希门禁均通过。没有平台提交授权，也未消耗额度。
