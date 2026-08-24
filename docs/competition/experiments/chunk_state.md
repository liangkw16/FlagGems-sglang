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

## E3：short-K `BLOCK_M=64` 拒绝

状态：严格 screening 拒绝，源码和测试已逐字节恢复 E2；未生成新 ZIP，
未提交平台

验证时间：2026-08-24 10:15–10:19 CST

### 假设与身份

E3 只在 `headdim >= 64 and chunk_size < 256` 时把 `BLOCK_M` 从 32
改为 64，其他 tile、warps、stages、计算和 grid 规则不变。affected 固定为
6 个 short-K/M>=64 shape × 3 dtype，control 固定为 M63 与 K256 × 3
dtype；晋级前预注册总体 `>=1.05x`、每 dtype `>=1.02x`、最差点
`>=0.98x`、最差单轮 `>=0.95x`、control 点 `[0.98,1.02]`，并要求
`<=128` registers、`<=16 KiB` shared、0 spill/scratch/local、无 TF32。

| 项目 | 值 |
| --- | --- |
| baseline commit | `67350fa9bc365d7b26b2c5215f1cd716f244fbc2` |
| baseline 源码 SHA-256 | `c50cda381c48712e108e34578c9805e74422b6b7b81be9b6dd6b2972d3753c47` |
| E3 临时源码 SHA-256 | `5d6778e717f247528f3678f0b40dcf21f8f3061ace8513095b17d8cf461da1d8` |
| E3 临时测试 SHA-256 | `8e35df823b7e6ae7d7cad2e0850bc03d1625e8e2da003b1f1dc03a00126c0d95` |
| harness / gates SHA-256 | `037b5ef95ad0bfbaa69f7287332e5a08351d741989813820b7c5ef3cc5042475` / `4e721269b169e9eda83bbf144c37229468b93af73124687cf146e63435ed08df` |
| raw JSON / gates log SHA-256 | `5855dc73ca3c17818d4728e74fb3ba0bc75ca5a601d211e6f53d36ae49164548` / `459a05153458d3d73889e93455fa6aa0c132ba40f35750c22f3377e87f7825ec` |
| static/unit log SHA-256 | `5d343776167b28e63884688befaa78c5a89fa3d1b1e0a2bbccff08b5276e3017` |
| 远端证据 | `gpu:/tmp/flagos-chunk-state-e3.dgS8ng`，mode 0700，PID/PGID `94177` |

远端 RTX 5070 Ti 环境与 E2 相同。静态门禁和 3/3 unittest 通过；screening
对 54 个 case 做 reference 正确性，全部通过。性能使用 batch 20、
`warmup=25, rep=100`，六轮严格 `AB/BA` 各三轮，并以逐轮 paired
speedup 中位数裁决。

### 拒绝证据

| 指标 | 结果 | 门槛 |
| --- | ---: | ---: |
| affected 18 点几何平均 | `1.120652x` | `>=1.05x` |
| FP16 / BF16 / FP32 | `1.135781/1.120736/1.105642x` | 各 `>=1.02x` |
| affected 最差点 | `0.959101x` | `>=0.98x` |
| affected 最差单轮 | `0.958203x` | `>=0.95x` |
| control 6 点几何平均 | `0.999767x` | `[0.98,1.02]` |
| control 点范围 | `0.998567–1.000166x` | `[0.98,1.02]` |

最差点为 BF16 `a1_small=(batch=1,nchunks=2,chunk_size=31,nheads=4,
headdim=65,ngroups=2,dstate=35)`：E2/E3 中位数分别为
`0.0024576/0.0025624 ms`，六轮 speedup 为
`0.958802/0.959400/0.960025/0.960025/0.958203/0.958802x`。

资源门禁也独立失败：FP32 `a1_small` 和 `a3_tail` 的 E3 编译产物均从
E2 的 64 registers、8192-byte shared、0 spill 变为 80 registers、
12288-byte shared、2 spills；虽仍为 0 scratch/local、4 warps、1 stage、
无 TF32，但违反 0-spill 门槛。aggregate 收益不能覆盖稳定的小 shape 回退和
新增 spill，因此不做事后 shape 缩窗，不晋级 E3，继续保留 E2 作为 Task 12
唯一候选。

## E2a：预防性 Ascend capped grid vendor（首投候选）

状态：release 门禁通过，候选就绪，等待 preflight 与提交

背景：Task 08/20/21/24 的平台记录证明 Ascend 把全部 grid 维展平为总
`coreDim` 且上限 65535；本 generic 为 3D grid
`(tiles, batch*nchunks, nheads)`，总 program 数在较大隐藏 shape（如
seqlen 16384、chunk 64、batch 2、nheads 32、H64/N128 时约 131072）会超限。
为避免 Task 21 式 6/8→7/8 的额度消耗，首投前预防性加入华为 vendor。

E2a 的 generic 与 E2 逐字节相同；新增 `_ascend/ops/chunk_state.py` 采用
Task 20 E3 平台验证三次的 capped grid-stride 模式：一维物理 grid
`min(total_programs, 4096)`，program 内以 `tl.num_programs(0)` 跨步遍历
逻辑 id，并按 `head → batch·chunk → tile` 分解还原三元组；kernel 数学、
BLOCK 32/32/64|32、4 warps、1 stage、stride 与 `tl.dot` 路径逐行保持 E2。
新增回归 `test_ascend_capped_grid_covers_multi_iteration_scale`：
`(2,128,64,8,64,64)` 的总 program 数 8192 > 4096，覆盖每 program 两轮
grid-stride；`(2,3,17,6,19,23)` 覆盖非 2 次幂尾块；均与 reference 按
`3e-2` 容差比对。既有四个回归不变。

screening 目录 `gpu:/tmp/flagos-chunk-state-asc.n46PpQ`（mode 0700）。第一
次 PID/PGID `104493`（23:30:51，wall 900s，脚本 SHA-256
`70ca1d86c85ead3db06dd2d54ba8e652307d802d82d7696bbff92454bee71da7`）因测试
新增方法的 Black 折行失败停止；本地仅修折行（测试 SHA-256 由
`f2c85270601a390d42cc83181626117d4edb338be28f0bb7fd1edc6082c8a53d` 变为
`b44848d183139069253aa082fefa225d9ffefe484e86a8ea1dbf3cf138dc3f48`）后以
PID/PGID `104636`（23:32:46）重跑通过：py_compile、Black 79、isort、
flake8 与 4/4 unittest（1.242s），`screening.log` SHA-256
`a0b778ff2e7a7f981bae4a25bdaa2cb8ed387f533343de73addf037a767b1c7b`。环境
RTX 5070 Ti 16 GB、driver 610.57.04、Python 3.12.13、PyTorch 2.13.0+cu130、
Triton 3.7.1、CUDA 13.0。

source/verification commit 均为
`9816257680bbd1f59716359993c3b1327786ac7d`；generic blob 仍为
`c50cda381c48712e108e34578c9805e74422b6b7b81be9b6dd6b2972d3753c47`（与 E2
一致），Ascend vendor blob
`7d3f772f586ab54f9764200697aabdfe0e7c9c224aa14f113d12cd3d0f7cc637`，测试
`b44848d183139069253aa082fefa225d9ffefe484e86a8ea1dbf3cf138dc3f48`。release
目录 `gpu:/tmp/flagos-chunk-state-asc-release.KkyZYv`（mode 0700）从该
commit 的 Git 对象建立，PID/PGID `104814`（23:34:37，wall 600s）；静态
门禁与 4/4 unittest（0.571s）通过并输出 `RELEASE_OK`，`release.log`
SHA-256
`9851c2e56e7e72ccb5a2ed3a3cc612144e66db08c0d8c67d055c2370cd4779dc`。

canonical ZIP 为
`artifacts/competition/chunk_state/e2a-9816257/chunk_state.zip`，15365
bytes，SHA-256
`f09ffdcaa945c5781220a67bed96732b33c00d0c7dfafc39f7c829d1a64cf506`；成员
`chunk_state.py`、`chunk_state_ascend.py`，`unzip -t` 通过。平台门禁：8/8
通过且每芯 ≥0.1x；华为在超限 shape 选中 vendor，其余七芯 generic。昆仑 dot
kernel 走 XPU SDNN 路径，grid 行为未知，若失败则按 Task 21 的 BLOCK 放大
模式修复；燧原 grid.x（tiles ≤64）安全。

### E2a 平台首投：7/8

2026-08-24 23:36:30 CST 提交，submission ID `4298`、当日序号 `14`，额度由
`17/30` 变为 `16/30`；远端验签 `verified`，`file_url_sha256` 为
`c0d1c6da5d6c50321ed5b66d3e9dd74215bc0740b01e506db940d92d49d17780`。
23:40:47 CST 终态 `completed` / `invalid_correctness`，7/8：沐曦 2.7285x、
燧原 0.1340x、海光 4.4840x、昆仑 0.2510x（generic，XPU SDNN 路径正确处理
dot 与 3D grid）、华为 0.2720x（`chunk_state_ascend.py`，预防性 vendor 按
预期被选中并通过）、国际 A 3.8365x、国际 B 2.9190x。天数智芯 correctness
case 0–4 失败：单 case 489/512 元素不匹配，最大绝对差 6.40、最大相对差
449.87，远超 TF32 级噪声，属结构性数值错误而非精度损失。

根因假设：generic 的 `tl.dot(x, B, input_precision="ieee")` 在天数后端被
错误编译。依据：固定 FlagGems `ed2508b` 的全部 FLA kernel（含
`chunk_delta_h.py`）均使用不带 `input_precision` 的裸 `tl.dot` 并在含
`_iluvatar` 的全部后端运行；其 `_iluvatar` heuristics 无任何 dot 精度
workaround。天数无 grid 展平限制（Task 21 天数以总 program 114688 通过），
故 vendor 只做单变量：去掉 `input_precision="ieee"`。

## E2b：天数 plain-dot vendor

状态：release 门禁通过，候选就绪，等待 preflight 与提交

E2b 新增 `_iluvatar/ops/chunk_state.py`：kernel 与 generic 逐字节相同，仅
`tl.dot(x, B, input_precision="ieee")` 改为 `tl.dot(x, B)`；generic、华为
vendor 与 grid 不变。新增回归
`test_iluvatar_plain_dot_chunk_boundary_precision`（fp32/fp16 × chunk
`63/64/255/256/257`，代理上 NVIDIA 对 fp32 dot 默认 tf32，验证 3e-2 容差仍
满足）并把天数 vendor 纳入多迭代规模回归。共 5/5 unittest。

screening 目录 `gpu:/tmp/flagos-chunk-state-ilv.WZ9o1L`（mode 0700），
PID/PGID `105015`（23:43:56，wall 900s，脚本 SHA-256
`33b991989d954003c302da6c9af9a867a9c2f19a0544c9ab2cedaf25cb442ee1`）：静态
门禁与 5/5 unittest（1.695s）通过，`screening.log` SHA-256
`592ca3064954a7edc85377c72163451e26139218ba2ccd678e0c8e945296bb4a`。
天数 vendor blob
`add042f318103aa58c89d5eac7a856b0f2c3132527f99c364b12c80c65571d19`，测试
`8f70bf5e232ef4782251bb89a0de6f6f15d6d9857f96c360debd70ff79959d08`。
release 目录 `gpu:/tmp/flagos-chunk-state-ilv-release.syDsTG`（mode 0700），
source/verification commit `b5ff6524acb1c175e79c44ea866c50f201e7d686`，
PID/PGID `105212`（23:46:11，wall 600s）；`RELEASE_OK`，`release.log`
SHA-256
`8d9ab87f5ddf30b952952784332150dedd2fb5f56eadb6527508e37de45a32a1`。

canonical ZIP 为
`artifacts/competition/chunk_state/e2b-b5ff652/chunk_state.zip`，成员
`chunk_state.py`、`chunk_state_ascend.py`、`chunk_state_iluvatar.py`，
SHA-256
`80db396c711dc482470c37374f49a3c505793daab730e455a0a2c5ad33ab9948`，
`unzip -t` 通过。平台门禁：8/8 通过且每芯 ≥0.1x；天数选中 plain-dot
vendor，华为继续选中 ascend vendor。

### E2b 平台结果：仍 7/8，天数错误指纹不变

2026-08-24 23:48:05 CST 提交，submission ID `4311`、当日序号 `15`，额度
`16/30` 变 `15/30`，远端验签 `verified`，`file_url_sha256` 为
`26d529d12a3b2a226c7f358fa2ca0ca8d3972d618221d895c5d41c76a4b41dd7`。
23:52:04 CST 终态 `completed` / `invalid_correctness`，7/8；天数选中
`chunk_state_iluvatar.py` 但 case 0–4 仍失败，逐 case 的不匹配计数、最大
绝对/相对差与出错索引与 E2a 完全一致（489/512、6.40、449.87 等），证明
`input_precision` 不是根因，存在确定性结构错误。相对差高达 2.8e7 说明读到
的是不相关内存而非精度损失；case 5–7（推断为连续输入）通过、0–4（推断为
非连续输入）全挂，且 Task 21 天数以显式 stride 的一维 kernel 通过——新
假设：天数后端对二维 stride load 的显式 stride 处理有缺陷。固定
community `0e8023d` 的多芯笔记亦表明跨芯 dot 惯例为低精度操作数 + fp32
累加，与本假设正交。

## E2c：天数 contiguous 输入 vendor

E2c 在天数 vendor 的 wrapper 中对 B/x/dt/dA_cumsum 增加 `.contiguous()`
（已连续时为零拷贝），kernel 与 E2b 逐字节相同；generic 与华为 vendor 不
变。新增回归 `test_iluvatar_vendor_strided_inputs`（三 dtype 非连续
B/x/dt/dA）。screening `gpu:/tmp/flagos-chunk-state-ilv2.3siJcY`，PID/PGID
`105414`（23:54:54，wall 900s，脚本同 E2b，SHA-256
`33b991989d954003c302da6c9af9a867a9c2f19a0544c9ab2cedaf25cb442ee1`），
6/6 unittest，`screening.log` SHA-256
`2fffb693726de937165b3de57214c25a6f502b3df50a863d7edf167c74a24ada`。
天数 vendor blob
`3c15e6b9e677aa408f1b2dc9d27a97089d52c914140b67f5c439f672b68b9c5c`，测试
`c5831b866b973b272b4ee4ccd7525946c6f94fd5f5224b2fd0389350824be6e5`。release
`gpu:/tmp/flagos-chunk-state-ilv2-release.0Hpyr4`，source/verification
commit `c18553db285f163f9027991e4e745485779eb807`，PID/PGID `105603`
（23:57:11，wall 600s），`RELEASE_OK`，`release.log` SHA-256
`43a4baa47298aa8b2a17362f2360cf7c37732220351a6aa72d9b68b68cc259d5`。
canonical ZIP
`artifacts/competition/chunk_state/e2c-c18553d/chunk_state.zip`，SHA-256
`ff1a27fd142d3f8ec92a09da7bc9c625463e0d62e771e8f99116d5b1fe9b06ad`，成员
generic + `_ascend` + `_iluvatar`，`unzip -t` 通过。若本轮天数仍失败，
记录负结果后停止 Task 12 迭代，额度转队列下一任务。
