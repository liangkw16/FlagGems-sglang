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

## E1 拒绝与 E2 长 chunk K tile 候选

状态：E1 未过线；E2 已通过本地代理门禁并生成不可变 ZIP；
候选就绪，未提交平台

验证时间：2026-08-24 03:57–04:08 CST

### 构建身份

| 项目 | 值 |
| --- | --- |
| source / verification commit | `67350fa9bc365d7b26b2c5215f1cd716f244fbc2` |
| E2 源码 SHA-256 | `c50cda381c48712e108e34578c9805e74422b6b7b81be9b6dd6b2972d3753c47` |
| E1 临时源码 SHA-256 | `221a0a00c976e71bf1531971f793b34515297f62c95d329673f2251b8bb219b1` |
| 最终测试 SHA-256 | `06a717b62a1f2c6498e2755ac4849838199283a82a49e95c586576f8caeba645` |
| E2 ZIP | `artifacts/competition/chunk_state/e2-67350fa/chunk_state.zip` |
| E2 ZIP SHA-256 | `35f11803055ccc0a7e6bff71c974ad3671032c1cec35d2a556367789206de9e3` |
| ZIP manifest | 顶层 `chunk_state.py`，7137 bytes，成员 SHA 与 E2 源码相同 |
| S0 回滚 ZIP | `s0-b05bfeb`，SHA-256 `c689def894513d211ae96a1085d9e937a6b2da6dbc40e3db4aa5e9c9cb0a9686` |
| 平台结果 | 未提交；逐芯结果、均值、排名和实时额度均为 N/A |

打包器从 source commit 生成 7263-byte 规范 ZIP，二次检查为
`verified-existing`。`unzip -t/-l`、UTF-8、单一顶层 `.py`、10 MB、
basename、成员源码哈希和 ZIP 哈希门禁全部通过。

### 固定来源与单变量

Mamba v2.2.4 的固定 commit `95d8aba8a8c75aedcaa6143713b11e745e7cd0d9`
在 9 个 forward autotune 配置中只有 1 个 K64，且绑定
M/N=128/256、8 warps、3 stages；其余 8 个均为 K32。上游还会把
scaled B 降回输入 dtype 再做默认 dot，与本题 FP32 operands、IEEE dot
和 FP32 输出不同，所以上游只证明 K64 值得实验，不能直接复制。

E1 只把所有 shape 的 `BLOCK_K` 从 32 改为 64。E2 拒绝全局改动，
仅在 `chunk_size >= 256` 时使用 K64；其余 shape 继续使用 S0 K32。
M/N=32、4 warps、1 stage、grid、FP32 计算、IEEE dot、stride 与 tail mask
均不变。该分支只选择 Triton constexpr tile，不是设备分支或 Torch fallback。

### 正确性与发布证据

远端环境为 RTX 5070 Ti 16 GB、driver 610.57.04、compute capability
12.0、Python 3.12.13、PyTorch 2.13.0+cu130、Triton 3.7.1、CUDA 13.0。

最终远端目录为 `gpu:/tmp/flagos-chunk-state-release.KbHjup`，mode 0700。
源码和测试逐字节与 source / verification commit 相同；`py_compile`、
Black 79、isort、flake8 和 3/3 unittest 通过。永久回归覆盖三 dtype ×
`chunk_size=1/31/32/33/63/64/65/255/256/257`，包含 FP32 `dt/dA`、
GQA、M/N tail；原有四输入非连续 stride、输入不变性和空 batch 继续保留。

发布 A/B 额外对 reference 验证 K=127/128/129/255/256/257/
511/512/513、M/N=`1x1/31x33/32x32/33x35/64x64/65x63`、GQA ratio
1/2/3、同 dtype 和 FP32 scale、四输入非连续、单 head 低并行和典型并行，
全部通过 `atol=rtol=3e-2`。

| 证据 | PID / 时间 | 脚本 SHA-256 | 日志 SHA-256 |
| --- | --- | --- | --- |
| S0 扩展门禁 | `74632` / 03:57:19 | 仓库 unittest | `696c3a79040d6cb902b1e55734b51b6286d4f1888b671610567460eb092681c3` |
| E1 门禁 | `74757` / 03:58:52 | 仓库 unittest | `d0dbe162781c975f7f4015dda6d5d332a80ad06de9341d851a0dfe5624e4ac4f` |
| E1 A/B | `74987` / 04:01:28 | `6a8bae18ceb3d09a32bc6bd859b524c18e0659b4d5dd3658e5057e898990d8db` | `f36953194a8e8f0c682e90fad047ab96d41d66f0c5cb33fb94866921206e7f59` |
| E2 发布门禁 | `75477` / 04:06:47 | 仓库 unittest | `5d643a05200a167ae1ada07a6a5e40154dc3d2a25dac51bad2694af0da6d18b1` |
| E2 发布 A/B | `75578` / 04:07:14 | `5caec0d63a5d4f3ee24fa862fb6142d0e489f588cd6eac670533f36ebc9f7c04` | `e1058235cdf428d915731233cb5d5bac234b6a2a02e00e607f0c1ceed4168c61` |

### E1 拒绝证据

E1 在 9 个点上做五组轮换、wrapper-inclusive A/B，每组
`warmup=25, rep=100`。所有正确性通过，但总体几何平均只有
`0.984555x`，K>32 为 `1.021308x`，K<=32 control 为 `0.866025x`；
FP16/BF16/FP32 分别为 `0.922163/1.001959/1.139490x`。K31 稳定慢
25%，K65 慢 16.9%；只有长 K 出现有效收益，如 FP32 K256 为
`1.298438x`。E1 shared 从 8192 增到 16384 bytes，寄存器上限从
96 增到 128；虽然仍为 0 spill/scratch/local，但性能未过线，因此拒绝。

### E2 NVIDIA 代理性能

发布结果为 wrapper-inclusive 五组轮换 A/B，每组
`warmup=25, rep=100`：

| dtype / shape | S0 p50 (ms) | E2 p50 (ms) | S0 / E2 |
| --- | ---: | ---: | ---: |
| FP16 K256、M64/N128、典型并行 | 0.030720 | 0.030592 | 1.004184x |
| BF16 K256、M64/N64、典型并行 | 0.020480 | 0.018432 | 1.111111x |
| FP32 K256、M64/N64、典型并行 | 0.026592 | 0.020480 | 1.298438x |
| FP16 K257、M64/N64、典型并行 | 0.022528 | 0.022528 | 1.000000x |
| BF16 K512、M64/N64、典型并行 | 0.034848 | 0.032768 | 1.063477x |
| FP32 K513、M64/N64、典型并行 | 0.045056 | 0.036864 | 1.222222x |
| FP16 K256、M33/N35、单 head 低并行 | 0.014336 | 0.012224 | 1.172775x |

| 指标 | 结果 |
| --- | ---: |
| 全 14 点几何平均 | `1.058224x` |
| 7 个 K>=256 受影响点几何平均 | `1.119837x` |
| 7 个 K<256 control 几何平均 | `1.000000x` |
| FP16 / BF16 / FP32 受影响点 | `1.056029/1.087033/1.259754x` |
| 受影响点最差 | `1.000000x`，无稳定回退 |

15 个 S0 编译变体均为 8192 bytes shared；20 个 E2 变体根据
shape 为 8192 或 16384 bytes shared。两者寄存器上限均为 96，全部
0 spill、0 global scratch，PTX 无 local load/store 和 TF32。

E2 超过预设的全套 `>=1.05x`、每 dtype `>=1.02x`、稳定回退
`<=2%`和 16 KiB/0-spill 门禁，因此作为新候选。剩余风险是其余七类
芯片的 K64 lowering、16 KiB 片上存储和隐藏 shape 分布尚无实证；
固定后端声明未发现 32x32x64/4-warps/1-stage 的 tile 禁令，但最终
正确性与性能仍必须由平台逐芯证明。上传前须重新读取实时额度，
并针对 Task 12、E2 ZIP 的绝对路径和完整 SHA-256 取得当次确认；
本记录不构成上传授权。
