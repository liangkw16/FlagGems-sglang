# Task 12 `chunk_state` 实验记录

## 契约

- 接口：`chunk_state(B, x, dt, dA_cumsum)`；参数 `B` 是状态投影张量，不是
  batch size。
- `B` 为 `[batch,seqlen,ngroups,dstate]`，`x` 为
  `[batch,seqlen,nheads,headdim]`；`dt/dA_cumsum` 为
  `[batch,nheads,nchunks,chunk_size]`。
- head `h` 使用 group `h // (nheads/ngroups)`；S0 要求
  `seqlen == nchunks * chunk_size`，与题面 reference 的 reshape 一致。
- `scale = exp(dA_last - dA_cumsum) * dt`，在 chunk 时间维累加
  `x * B * scale`；decay、scale、乘法累加和输出均为 FP32。
- 输出固定为 `[batch,nchunks,nheads,headdim,dstate]` FP32；输入不变；
  正确性容差 `atol=rtol=3e-2`。
- 支持八类芯片，最低加速比 0.1x；核心路径必须为 Triton/TLE，无 fallback。

固定参考为 Mamba v2.2.4
`mamba_ssm/ops/triton/ssd_chunk_state.py`。S0 只复用 forward 索引与 GQA 映射，
删除 autotune、backward、seq_idx、device context 和 autograd wrapper。

## S0：fixed 32x32x32 generic baseline

状态：远端 NVIDIA 正确性、代理性能和不可变 ZIP 门禁通过；未提交平台
验证时间：2026-08-24 01:33–01:37 CST
源码 commit：`b05bfeb`

### 构建身份

| 项目 | 值 |
| --- | --- |
| 源文件 | `src/flaggems_sglang/ops/chunk_state.py` |
| 源文件 SHA-256 | `1c38f2f8cd2f81f60a69c3d138e3ccbadfa8feffbca82df32c1979bc4285ca00` |
| 测试文件 | `tests/test_chunk_state.py` |
| 测试 SHA-256 | `00812ec619eb1ea81036ee98a6cff9aa3d856eb687345f506e4ac90234454aae` |
| ZIP | `artifacts/competition/chunk_state/s0-b05bfeb/chunk_state.zip` |
| ZIP SHA-256 | `c689def894513d211ae96a1085d9e937a6b2da6dbc40e3db4aa5e9c9cb0a9686` |
| 远端证据目录 | `gpu:/tmp/flagos-batch2.SQaIX2`，mode 0700 |

### 唯一候选配置

- `BLOCK_M/BLOCK_N/BLOCK_K = 32/32/32`，4 warps，1 stage；无 autotune。
- grid 为 `(ceil(headdim/32)*ceil(dstate/32), batch*nchunks, nheads)`。
- `B/x/dt/dA_cumsum` 的全部真实 stride 和输出五维 stride 都显式传入。
- `x`、`B`、decay 和 scale 转为 FP32；`tl.dot` 使用
  `input_precision="ieee"` 禁用 TF32，accumulator 与 store 都为 FP32。
- 所有 M/N/K tail 均 mask；无 vendor、设备判断、异常捕获或 PyTorch 计算
  fallback。

### 正确性与静态检查

- 本地 `py_compile`、AST 解析和 79 字符行宽通过。
- 公开接口测试先于实现落盘。远端 RTX 5070 Ti 16 GB、driver 610.57.04、
  Python 3.12.13、PyTorch 2.13.0+cu130、Triton 3.7.1、CUDA 13.0、
  compute capability 12.0。
- 远端源码/测试 SHA-256 和本地一致；Black 79、isort、flake8 均通过。
- `tests/test_chunk_state.py -v` 为 2/2 unittest 方法通过，运行 1.207 秒。
  内部覆盖 FP16/BF16/FP32、GQA ratio 3、四输入非连续 stride、chunk/headdim/
  dstate 为 17/19/23 的非 power-of-two tail、输入不变性和空 batch。

### NVIDIA 代理性能

wrapper-inclusive；每个候选先做正确性检查和 JIT，再用
`triton.testing.do_bench(warmup=25, rep=100, quantiles=[0.5])`。

| dtype | case `(B,L,H,P,G,N)` | S0 p50 (ms) | Torch p50 (ms) | speedup |
| --- | --- | ---: | ---: | ---: |
| FP16 | typical `(2,256,8,64,2,64)` | 0.010208 | 0.035904 | 3.517x |
| FP16 | tail `(1,68,6,33,2,37)` | 0.006144 | 0.026624 | 4.333x |
| BF16 | typical `(2,256,8,64,2,64)` | 0.010208 | 0.036768 | 3.602x |
| BF16 | tail `(1,68,6,33,2,37)` | 0.006144 | 0.027072 | 4.406x |
| FP32 | typical `(2,256,8,64,2,64)` | 0.010208 | 0.026656 | 2.611x |
| FP32 | tail `(1,68,6,33,2,37)` | 0.006144 | 0.016416 | 2.672x |

六个编译产物均为 4 warps、1 stage、8192 bytes shared memory、0 global
scratch；PTX 未出现 `tf32`、`ld.local` 或 `st.local`。最小代理 speedup 为
2.611x。

ZIP 由 commit `b05bfeb` 的算子子树直接生成，仅含顶层 UTF-8
`chunk_state.py`。`unzip -t`、10 MB、成员名和逐字节 SHA-256 门禁均通过。

### 已知风险

- NVIDIA 代理不能证明其余七类后端正确或达到门槛。
- FP32 IEEE dot 优先保证语义，可能比上游将 scaled B 降回输入 dtype 的 tensor
  core 路径慢；先用平台结果判断是否需要受控的混合精度优化。
- 32 固定 tile 避免跨芯 autotune，但非常小或很大的维度未必最优。
- 题面 reference 不能 reshape partial final chunk，因此 S0 不推测 padding 语义。
- 尚未获得针对上述 ZIP 与实时额度的当次确认；未提交平台，也未消耗额度。
