# Task 20 `mamba_layernorm_gated` 实验记录

## 契约

- 接口：`mamba_layernorm_gated(x, weight, bias, eps, z=None,
  group_size=None, norm_before_gate=True, is_rms_norm=True)`。
- 输入与输出为 `[M, N]`；`group_size=None` 等价于 N，否则 N 必须被
  `group_size` 整除。
- `norm_before_gate=False` 时先计算 `x * z * sigmoid(z)` 再归一化；为 True
  时先完成归一化及 affine，再乘 `z * sigmoid(z)`。
- 支持 RMSNorm 与 LayerNorm；输入、均值/方差、`rsqrt`、weight、可选 bias 和
  gate 全部按 FP32 计算，输出转换回 `x.dtype`，所有输入保持不变。
- 题面支持 FP16、BF16、FP32，容差分别为 `1e-2`、`1.5e-2`、`1e-4`；
  支持八类芯片，最低加速比为 0.1x。

固定参考为 SGLang `8014d9d` 的
`python/sglang/kernels/ops/attention/fla/layernorm_gated.py`。S0 只保留
1-pass forward 语义，删除 backward、SM count、PDL、NPU/CPU/XPU 分支、
autograd wrapper 和设备上下文。

## S0：generic row-group baseline

状态：S0 已打包并通过本地门禁；等待当次上传确认

验证时间：2026-08-24 01:28–01:37 CST

源码 commit：`f431ba4`

### 构建身份

| 项目 | 值 |
| --- | --- |
| 源文件 | `src/flaggems_sglang/ops/mamba_layernorm_gated.py` |
| 源文件 SHA-256 | `99c66cd49ad9e8f37ea8b087d63fd93fd3fab9c7602361b360a37e4dfaa6ca73` |
| 测试文件 | `tests/test_mamba_layernorm_gated.py` |
| 测试 SHA-256 | `1d66c36617728fd8b46b4ad2bdee0e53e75ca3a067652ab869e3f928723712a6` |
| ZIP | `artifacts/competition/mamba_layernorm_gated/s0-f431ba4/mamba_layernorm_gated.zip` |
| ZIP SHA-256 | `0bf5d8f26c6e3b3b827e2541bc58c058dc6b6fec05efe7bcff127492dfaedf76` |
| ZIP manifest | 顶层 `mamba_layernorm_gated.py`，4394 bytes |

### 唯一候选配置

- 每个 `(row, group)` 一个 Triton program；
  `BLOCK_SIZE=next_power_of_2(group_size)`。
- BLOCK 小于 2048 使用 4 warps，否则使用 8 warps；固定 `num_stages=1`。
- 显式使用 `x/z` 行列 stride 与 `weight/bias` stride，尾部完整 mask，输出为
  连续同 shape、同 dtype tensor；空输入直接返回。
- None 指针以已有合法 tensor 作为占位，实际读取由 constexpr 分支完全移除。
- 无 backward、autotune、vendor、设备判断、异常捕获或 PyTorch 计算 fallback。

### 正确性与静态检查

- 本地 `py_compile`、AST 解析和 79 字符行宽通过。
- 公开接口测试先于实现落盘。远端 RTX 5070 Ti 16 GB、driver 610.57.04、
  Python 3.12.13、PyTorch 2.13.0+cu130、Triton 3.7.1、CUDA 13.0、
  compute capability 12.0。
- 远端源码/测试 SHA-256 与本地一致；Black 79、isort、flake8 均通过。
- `tests/test_mamba_layernorm_gated.py -v` 为 2/2 unittest 方法通过，运行
  1.074 秒。内部 12 个组合覆盖三 dtype、RMS/LN、整维/分组、bias None/有、
  z None/有、门控前/后、非连续 `x/z/weight/bias`、输入不变和 hidden=259
  的非 power-of-two tail；另覆盖空输入。

### NVIDIA 代理性能

wrapper-inclusive；shape `[64, 4096]`；每个候选先做正确性检查和 JIT，再用
`triton.testing.do_bench(warmup=25, rep=100, quantiles=[0.5])`。

| dtype | 分支 | S0 p50 (ms) | Torch p50 (ms) | speedup |
| --- | --- | ---: | ---: | ---: |
| FP16 | RMS、整维、无 gate/bias | 0.006144 | 0.024672 | 4.016x |
| FP16 | LN、group 512、bias、后 gate | 0.006144 | 0.038048 | 6.193x |
| BF16 | RMS、整维、无 gate/bias | 0.006144 | 0.024672 | 4.016x |
| BF16 | LN、group 512、bias、后 gate | 0.006144 | 0.038912 | 6.333x |
| FP32 | RMS、整维、无 gate/bias | 0.006144 | 0.018528 | 3.016x |
| FP32 | LN、group 512、bias、后 gate | 0.008192 | 0.028768 | 3.512x |

六个编译产物均为 1 stage、4 或 8 warps、16 或 32 bytes shared memory、
0 global scratch；PTX 未出现 `ld.local`/`st.local`。最小代理 speedup 为 3.016x。

### 已知风险

- NVIDIA 代理不能证明其余七类后端正确或达到门槛。
- 单 program 保存整个 group；超过常见 8K group 后可能出现寄存器或本地内存
  压力。没有为未公开的大 group 推测两阶段实现。
- 二维 grid 的 y 轴等于 group 数；公开题面没有 shape 上界，极端小 group/大 N
  仍需隐藏 harness 或平台验证。
- ZIP 由 commit `f431ba4` 直接生成；`unzip -t`、UTF-8、单一 `.py`、10 MB、
  basename 和 ZIP 内源码哈希门禁均通过。没有平台提交授权，也未消耗额度。
