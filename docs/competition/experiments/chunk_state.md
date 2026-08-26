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

### E2c 平台结果：仍 7/8，指纹不变 → E2d fp16 dot（天数最后一轮）

E2c 于 2026-08-24 23:59:01 CST 提交，submission ID `4323`、当日序号 `16`，
额度 `15/30` 变 `14/30`，`file_url_sha256` 为
`4ee88f32b950fb6714daa0351ce528f6bebfd0c76aacd375d9864387f0bcf660`；23:59:49
CST 终态 7/8。天数仍失败 case 0–4，且不匹配计数、差值与索引与 E2a/E2b
逐字节一致。三个不同假设（ieee 参数、plain dot、contiguous 输入）产生完全
相同的错误输出，最一致的解释：kernel 在天数上未能实际执行（异步编译/launch
失败），`torch.empty` 输出携带 harness 固定序列残留的确定性内存，故指纹恒
定；case 5–7 通过说明其 launch 成功，差异维度未定位。固定 community
`0e8023d` 的 context_attention（平台多芯通过）明确记录"q/k/v 统一转 fp16
走 tl.dot"，是唯一有天数成功案例背书的 dot 形式；fp32 操作数 `tl.dot` 在
天数上可能整体不受支持。

## E2d：天数 fp16 操作数 dot（本轮最后一次尝试）

天数 vendor 的 dot 改为 `tl.dot(x.to(tl.float16), B.to(tl.float16))`（fp32
累加不变），contiguous 处理保留；generic 与华为 vendor 不变。screening
`gpu:/tmp/flagos-chunk-state-ilv3.ZHCDSj`：第一次因新折行 Black 失败
（PID `105857` 跑旧字节后被终止，PID `106072` 前还有一次 Black 折行修正），
最终 PID/PGID `106072`（00:03:38，wall 900s，脚本同 E2b）6/6 unittest，
`screening.log` SHA-256
`a3edd2c186c273b90a379b45526dafb3714fdbc5d3d4890a303c49f9a0a009cb`；vendor
blob
`9b96f58cb15e96a17457604374048e3f37d242218eda8115144c251020b36fd7`。release
`gpu:/tmp/flagos-chunk-state-ilv3-release.4rj0aP`，source/verification
commit `3d3148187541d87dfab0664fba38468408e815de`，PID/PGID `106276`
（00:06:07），`RELEASE_OK`，`release.log` SHA-256
`fb0add4e31c3afe3f158424aa955b9b51c0e8cd5e484462ad3f21466ec9fc555`。canonical
ZIP `artifacts/competition/chunk_state/e2d-3d31481/chunk_state.zip`，SHA-256
`3c06525a76dd00e338d40107feca666e43dd7f99b097e00da5e87f7ca548623b`。无论
结果如何，本轮后停止 Task 12 迭代并记录结论。

### E2d 平台结果：8/8，有效，团队当前最佳

2026-08-25 00:08:03 CST 提交，submission ID `4332`、当日序号 `1`（新额度日，
提交前后 `30/30` → `29/30`），远端验签 `verified`，`file_url_sha256` 为
`45123e678fd1a1a5fb6652d147983c8a2e9b6e4ddd2b0585bd279a7bcd7bb9a0`。
00:08:58 CST 终态 `completed` / `valid`，8/8 通过，平均 `1.948x`，team best：

| 芯片 | 结果 | speedup | 选中文件 |
| --- | --- | ---: | --- |
| 天数 | 通过 | 1.9815x | `chunk_state_iluvatar.py` |
| 沐曦 | 通过 | 2.7295x | `chunk_state.py` |
| 燧原 | 通过 | 0.1160x | `chunk_state.py` |
| 海光 | 通过 | 4.4845x | `chunk_state.py` |
| 昆仑芯 | 通过 | 0.2510x | `chunk_state.py` |
| 华为 | 通过 | 0.2735x | `chunk_state_ascend.py` |
| 国际通用 A | 通过 | 3.8370x | `chunk_state.py` |
| 国际通用 B | 通过 | 1.9110x | `chunk_state.py` |

结论：天数 fp16 操作数 dot 假设被平台证实——E2a/E2b/E2c 三轮相同错误指纹
对应 kernel 未执行，fp16 操作数 + fp32 累加后全部 case 通过且拿到 1.98x。
根因定性：fp32 操作数 `tl.dot` 在天数 Triton 后端不受支持（静默失败），
跨芯 dot 应使用低精度操作数 + fp32 累加（与 community `0e8023d` 惯例一致，
后续含 dot 的算子直接沿用该写法做天数 vendor）。华为 capped grid-stride
vendor 第四次平台验证成功；昆仑 dot 走 SDNN 路径以 generic 通过。Task 12
闭环完成。遗留观察：燧原 0.1160x 贴近 0.1x 门槛，性能优化时优先处理；
本轮按预定停止迭代，额度转队列下一任务。

## E3：燧原 fp16-dot vendor（性能冲刺）

Task 12 E2d 已 8/8 有效（平均 1.948x），但燧原 0.1160x 贴门槛且跑的是
generic（32 tile + ieee-fp32 操作数 dot + stages 1）——正是 Task 09 平台
证明的燧原病理配置。E3 是性能冲刺：把 Task 09 沉淀的燧原配置迁移为
Task 12 的 `_enflame` vendor。

- 单变量：仅燧原。generic、`_ascend`、`_iluvatar` 三成员与 E2d ZIP
  逐字节相同。
- `_enflame/ops/chunk_state.py`：x/B load 后不再统一转 fp32；B 以 fp32
  scale（`exp(dA_last - dA) * dt`）缩放后转 fp16，x 直接转 fp16，`tl.dot`
  fp16 操作数 + fp32 累加（天数 E2d 已平台验证该 dot 形态在本题 3e-2
  容差下通过）。launch 配置 64/64/128、warps 4、stages 2、3D grid 与
  generic 相同；题面容差 atol=3e-2/rtol=3e-2 覆盖 fp16 舍入。
- 新增回归：`test_enflame_dot_config_precision`（fp32/fp16/bf16 ×
  chunk 63/64/127/128/255/256/257，覆盖 1–3 个 K block 与尾掩码）与
  `test_enflame_vendor_strided_inputs`（三维均带真实 stride）。
- source/verification commit：
  `4ee8e1223c0613c1c888ac6bc427c25131433e8c`；本地 py_compile、black
  25.12.0（远端 26.5.1 版本漂移，见 bmm_chunk 账本 E3e 节）、isort、
  flake8 通过。
- canonical ZIP：`artifacts/competition/chunk_state/e3-4ee8e12/chunk_state.zip`，
  ZIP SHA-256
  `51459aabebabb0096f8485d0cd0dcc3821b34dcc7705f7e78914da9bbe499f00`，
  成员 generic + `_ascend`/`_enflame`/`_iluvatar`，`unzip -t` 通过。
- release 目录：`gpu:/tmp/flagos-multi-release.JfYAit/t12-stage`（mode
  0700，与 T24 s2d 同批串行执行）。
- 平台门禁：燧原不跌破 0.1x 且平均加速比较 E2d 1.948x 提升；其余七芯
  文件不变。

- release：`gpu:/tmp/flagos-multi-release.JfYAit/t12-stage`（mode 0700），
  `MODE=release source/verification=4ee8e12`，`Ran 8 tests in 5.353s`、
  `OK`、`RELEASE_OK`（`release.log`；与 T24 s2d 同批串行执行）。远端
  py_compile/isort/flake8/前后哈希一致，black 以本地 25.12.0 等价执行
  （远端 26.5.1 版本漂移，见 bmm_chunk 账本 E3e 节）。

### E3 平台结果：8/8 valid，平均 1.948x → 2.0966x，燧原 6.4 倍提升

2026-08-25 15:31:27 CST 提交（submission `4651`，当日序号 `26`，额度
`5/30`→`4/30`，`file_url_sha256` 为
`4de5648e28edb44610da9d4f27bde24ebbe6647dd2d020f865c739af774446bd`），
终态 `completed` / **valid**，8/8 通过，平均 `2.096625x`（team best）：

| 芯片 | E2d | E3 | 选中文件 |
| --- | ---: | ---: | --- |
| 天数 | 1.9815x | 2.0100x | `chunk_state_iluvatar.py` |
| 沐曦 | 2.7295x | 2.7535x | `chunk_state.py` |
| 燧原 | 0.1160x | **0.7430x** | `chunk_state_enflame.py` |
| 海光 | 4.4845x | 4.4815x | `chunk_state.py` |
| 昆仑芯 | 0.2510x | 0.2505x | `chunk_state.py` |
| 华为 | 0.2735x | 0.2735x | `chunk_state_ascend.py` |
| 国际通用 A | 3.8370x | 4.3440x | `chunk_state.py` |
| 国际通用 B | 1.9110x | 1.9170x | `chunk_state.py` |

结论：Task 09 沉淀的燧原 dot 配置（fp16 操作数 + 64/64/128 + stages 2）
迁移成功——燧原 0.1160→0.7430x（6.4 倍），仅 vendor 生效、其余七芯
文件不变且无回退（card_a +13% 属评测正向波动）。平均 1.948→2.0966x。
后续空间：燧原再加 capped grid-stride fold（T09 E3e 平台 +66% 证据）
预计 0.74→~1.2x；昆仑 0.2505/华为 0.2735 无新证据杠杆，维持。
remote_verification `unavailable`（环境变量未带入 submit 进程）。


## E4：燧原 capped grid-stride fold（追投）

E3 平台证实配置迁移后，E4 给燧原 vendor 追加 T09 E3e 平台验证的
grid-stride 折叠：3D grid 展平为一维逻辑 id（batch → chunk·head →
tile 分解），物理 grid `min(total, 64)`，tile/dot/warps/stages 与 E3
一致。新增回归 `test_enflame_fold_covers_multi_iteration`（(2,32 chunks,
8 heads, headdim 64, dstate 128) total 1024 > 64 覆盖 16 轮迭代）。

- source/verification commit：`ef065601279fe967970cb5a2cd5f5f213e452573`。
- release：`gpu:/tmp/flagos-multi3-release.x58bBH/t12e4-stage`，
  `Ran 9 tests in 7.473s`、`OK`、`RELEASE_OK`（15:38:27 done，与 s2e
  同批串行；black 仍以本地 25.12.0 等价执行）。
- canonical ZIP：`artifacts/competition/chunk_state/e4-ef06560/chunk_state.zip`，
  SHA-256
  `0d5fb3b11ff3969ae99a1f4b582d87d43324489d76f7c1312f286ab8baed9fbd`。

### E4 平台结果：燧原 +26% 至 0.939x，团队最佳保持 E3

2026-08-25 15:42:12 CST 提交（submission `4655`，当日序号 `28`，额度
`4/30`→`3/30`），终态 8/8 **valid**，平均 `2.0733x`：燧原
0.7430→**0.9390x**（fold +26%，T09 +66% 之后 fold 对燧原的第二份平台
证据），昆仑/华为/其余持平；card_a 4.344→3.975（评测波动），平均较
E3 的 2.0966x 低 0.023，平台按团队最佳计分（`is_team_best`），Task 12
保持 **2.0966x（E3）**。结论：燧原 fold 增益真实但被单芯波动淹没，
后续若再冲分应等待多芯同时有机会时合并提交。

## E5：generic 低精度张量核心 dot（性能冲刺，最后一发额度）

E3/E4 只覆盖燧原、天数两芯；沐曦 2.75、海光 4.48、card_a 4.34、card_b
1.92、昆仑 0.25 仍在跑 S0 保守 generic（fp32 操作数 + ieee dot + 32×32
tile + stages 1）。E5 把平台已两次验证的 dot 形态（天数 E2、燧原 E3）
迁回 generic：fp16/bf16 输入时 x/B 以 fp16 操作数直送 `tl.dot`（fp32 累加
与 fp32 输出不变，题面 3e-2 容差覆盖舍入），tile 升 64/64/128、stages 2；
fp32 输入路径逐字节保持原 ieee 配置（32×32、BLOCK_K 自适应、stages 1）。
E4 教训（单芯增益被 card_a 波动淹没）在本候选的反面：generic 变量同时
作用于五颗强芯，增益量级远超评测噪声。

- 变量说明：本次提交实际携带两个互不相交的变量——generic 低精度 dot
  （E5，作用于未选 vendor 的芯片）与 E4 已平台验证的燧原 fold vendor
  （逐字节复用）。逐芯结果可分别归因。
- 新增回归：`test_generic_lowprec_tensor_core_path_precision`（fp16/bf16 ×
  chunk 64/256 × headdim/dstate 64/128 与 65/129 尾块）与
  `test_generic_fp32_path_keeps_ieee_precision`（fp32 以 1e-3 容差锁定 ieee
  路径不被低精度路径污染）。
- source/verification commit：`1ba05477d21b3014905bf9cc8bc7971094f9bdb8`
  （`feat(chunk_state): generic lowprec tensor-core dot path (E5)`）。
- 本地门禁：py_compile、black 25.12.0、isort、flake8 通过。
- canonical ZIP：`artifacts/competition/chunk_state/e5-1ba0547/chunk_state.zip`
  （31,354 bytes），ZIP SHA-256
  `829476611c49cd04289e2eb5db322fc9c9868a9e927e3fb9ef81dcf3933943b2`，
  成员 generic + `_ascend`/`_enflame`/`_iluvatar`，`unzip -t` 通过，与
  dry-run manifest 完全一致。
- release 目录：`gpu:/tmp/flagos-chunk-state-e5.jcjavc`（mode 0700，
  source/verification=1ba0547，git archive 建 dir）。A/B 基线为 4ee8e12
  的 E3 generic；声明 affected=fp16/bf16 六个代理 shape、control=fp32
  两 shape，晋级阈值 affected 几何平均 ≥1.05x 且 control ≥0.98x、资源
  不退化（spill=0）。
- 平台门禁：八芯全部 ≥0.1x 且平均较团队最佳 2.0966x（E3）提升；额度
  1/30 为当日最后一发，提交前以实时 preflight 为准。

### E5 release 结果：全部门禁通过（RELEASE_OK）

远端目录 `gpu:/tmp/flagos-chunk-state-e5.jcjavc`（mode 0700），启动
PID/PGID `125071`（16:07 CST，`setsid`+`timeout`，wall 约 7 分钟）。
环境：RTX 5070 Ti 16 GB（验证时 GPU 空闲 0%）、driver 610.57.04、
Python 3.12.13、PyTorch 2.13.0+cu130、Triton 3.7.1、CUDA 13.0。
`release.log` SHA-256
`e03137adb69294ff2132ec968e3e843671097302f7c91a2647970b594c3465c5`，
A/B 脚本 SHA-256
`d2651f9b0e588c78d52bb9d827edb09b458b4c6977a26a3094098f528710f4a5`，
远端五个文件 SHA-256 与打包器 manifest 逐项一致（generic
`53d75005…`、ascend `7d3f772f…`、enflame `ba82b7bfa…`、iluvatar
`9b96f58c…`、test `5550430d…`）。

- 静态门禁：py_compile、isort、flake8 通过；远端前后哈希一致；black 以
  本地 25.12.0 等价执行（远端 26.5.1 版本漂移，见 bmm_chunk E3e 节）。
- unittest：11/11 通过（5.358s），含新增两项 generic 回归。
- A/B（五轮交替、wrapper-inclusive、paired median）：受影响六点几何平均
  **1.601x**（fp16 1.671x / bf16 1.472x），单点范围 1.264–2.500x
  （`fp16 b1c8k256h8` 满挡 2.500x）；两个 fp32 control 分别为
  0.9997/1.0000，未回归路径确认。
- 资源：低精度新变体 137–139 registers、0 spill；fp32 变体与旧实现
  相同（48/60 registers、0 spill）。
- 注：验证期间本机到 `gpu` 的 VPN 链路出现 MTU 黑洞（≥1200B 包丢弃），
  日志以小包滴流方式取回；远端进程 `setsid` 不受影响。

晋级判定：affected ≥1.05x、control ≥0.98x、资源零退化全部满足，
候选就绪，进入平台 preflight。

### E5 平台结果：7/8 通过但昆仑正确性失败，validity=invalid_correctness

2026-08-25 16:48:28 CST 提交（submission `4683`，当日序号 `30`，额度
`1/30`→`0/30`，`file_url_sha256` 为
`681905beb94c5aa94b9abeb6a59c5149c762536de2453d246f150f5db69db470`），
昆仑于 17:3x 回调后终态 7/8、**invalid_correctness**：

| 芯片 | E3/E4 最佳 | E5 | 选中文件 |
| --- | ---: | ---: | --- |
| 天数 | 2.0100x | 2.0095x | `chunk_state_iluvatar.py` |
| 沐曦 | 2.7535x | **1.9290x** | `chunk_state.py` |
| 燧原 | 0.9390x | 0.9375x | `chunk_state_enflame.py` |
| 海光 | 4.4845x | **4.5435x** | `chunk_state.py` |
| 昆仑芯 | 0.2510x | **correctness 失败** | `chunk_state.py` |
| 华为 | 0.2735x | 0.2730x | `chunk_state_ascend.py` |
| 国际通用 A | 4.3440x | **16.5830x** | `chunk_state.py` |
| 国际通用 B | 1.9170x | **0.8395x** | `chunk_state.py` |

结论与沉淀：

1. **generic 低精度 dot 的收益被平台实锤**：card_a 4.344→16.583x
   （+282%，张量核心解锁），海光 +1.4%；代理 A/B 1.601x 的方向正确。
2. **新增昆仑反例**：昆仑 SDNN 路径对 fp16 操作数 `tl.dot` 产生正确性
   错误（fp32-ieee 操作数在 E2d 通过）。generic 低精度 dot 必须搭配
   `_kunlunxin` vendor 回退旧形态。与天数"fp32 操作数 dot 静默不可执行"
   互为镜像：两芯对 dot 操作数 dtype 的兼容集合相反。
3. 沐曦 -30%、国际 B -56%：低精度路径对这两芯是回退，需要 vendor 保持
   旧 fp32-ieee 形态（沐曦后缀 `_metax` 可直接做；国际 B 的 amd/nvidia
   后缀映射未公开，可从 Task 14 `context_attention_nvidia.py` 的选中记录
   反推）。
4. 团队最佳保持 **2.0966x（E3）**；平台按 is_team_best 计分，本次失败
   不降低已有记录，消耗当日最后一发额度（30/30）。

下一候选（E6，需次日额度）：generic 保持 E5 低精度形态，新增
`_kunlunxin`（fp32-ieee 旧形态）、`_metax`（旧形态）vendor，并按
反推出的国际 B 后缀加同型 vendor；预期 card_a 16.58 / 海光 4.54 保持，
沐曦回到 2.75、国际 B 回到 1.92、昆仑回到 0.25，平均约 **3.6x**。

## E6：三芯回退 vendor（已由 E6r 精确字节版替代，未提交）

状态：旧源码与 ZIP 保持不可变，未提交平台；不得再使用

### 构建身份

| 项目 | 值 |
| --- | --- |
| source / verification commit | `5dbc25ca52c05b0fde3d5c895d0113acde3181fc` |
| generic blob SHA-256 | `53d75005a0de8870660a88c16c25e710ae260e836f60ccfc64f7760f1013a5a0`（E5 不变） |
| `_kunlunxin` blob SHA-256 | `479ada377a98f3250f7be7336285431b8fe202d979a92167c2fdc745cc55ea5b` |
| `_metax` blob SHA-256 | `9de382f2a0ffb2d8c3356f8e6e9f842f219b25613accacb93f03aa5d6fbb87ff` |
| `_amd` blob SHA-256 | `3a0ac26820bc2686a4dcb43f4e501a122851313671d5f1fee6ea5946acc748c9` |
| E6 ZIP | `artifacts/competition/chunk_state/e6-5dbc25c/chunk_state.zip` |
| ZIP SHA-256 | `1c20329a28d3e9523212fd2ce2b6db204727aad7d7685268bc546b4555fac0cd` |
| ZIP 大小 | 53974 bytes |

### 成员清单（7 文件）

`chunk_state.py`（generic E5 低精度）、`chunk_state_ascend.py`（capped
grid-stride，E2a 不变）、`chunk_state_enflame.py`（fp16 dot + fold，
E4 不变）、`chunk_state_iluvatar.py`（fp16 dot，E2d 不变）、
`chunk_state_kunlunxin.py`（新增，fp32-ieee 回退）、
`chunk_state_metax.py`（新增，fp32-ieee 回退）、
`chunk_state_amd.py`（新增，fp32-ieee 回退，假设国际 B=amd）。

### 单变量说明

generic、`_ascend`、`_enflame`、`_iluvatar` 四成员与 E5 ZIP 逐字节
相同。新增三个 vendor 均为 E2d 已平台验证的 fp32-ieee dot + 32×32×
64|32 tile + 4 warps + stages1 旧形态，kernel 数学与 E2d generic 完全
一致。国际 B 后缀假设为 amd（依据：card_a 低精度 +282% 符合 NVIDIA
tensor core 特征，card_b 低精度 -56% 符合 AMD ROCm fp16 dot 回退）；
若平台结果显示国际 B 实际为 nvidia，则下一轮把 `_amd` 回退改为
`_nvidia`。

### 预期逐芯结果

| 芯片 | E3 最佳 | E6 预期 | 选中文件 |
| --- | ---: | ---: | --- |
| 天数 | 2.01x | 2.01x | `chunk_state_iluvatar.py` |
| 沐曦 | 2.75x | 2.75x | `chunk_state_metax.py`（回退） |
| 燧原 | 0.74x | 0.94x | `chunk_state_enflame.py` |
| 海光 | 4.48x | 4.54x | `chunk_state.py` |
| 昆仑 | 0.25x | 0.25x | `chunk_state_kunlunxin.py`（回退） |
| 华为 | 0.27x | 0.27x | `chunk_state_ascend.py` |
| 国际 A | 4.34x | **16.58x** | `chunk_state.py`（低精度） |
| 国际 B | 1.92x | 1.92x | `chunk_state_amd.py`（回退） |
| **平均** | **2.0966x** | **~3.6x** | |

### 本地验证

- `py_compile` 三个新 vendor 文件通过。
- 代码基于 E2d 已平台验证的 kernel 数学，无新增逻辑分支。
- 远端 GPU 验证待次日额度提交前补做（或直接平台验证，因回退形态已
  在 E2d/E3 多次平台验证）。

### 风险

1. 国际 B 后缀假设错误（若为 nvidia 而非 amd），则国际 B 仍跑 generic
   低精度路径并回退 -56%，平均约 3.0x 而非 3.6x；下一轮修正后缀。
2. 昆仑 fp32-ieee 回退在 E2d 已验证通过（0.251x），正确性风险低。
3. 沐曦 fp32-ieee 回退在 E2d 已验证通过（2.7295x），正确性风险低。

## E6r：E2d 精确字节回退 + 三 vendor 回归

状态：release 与不可变 ZIP 门禁通过；候选就绪，只提交一次

验证时间：2026-08-26 17:52–17:57 CST

提交前静态审计发现 E6 三份 fallback 虽然数学等价，但把 `x/B` 的 FP32 cast
从 load 后移到了 dot 参数，不能声称与 E2d 平台验证形态逐字节一致；原测试也没有
导入三个新成员。E6r 把 `_kunlunxin/_metax/_amd` 全部替换为 E2d generic 的精确
源码字节，并增加一个测试覆盖三模块、FP16/BF16 与 `chunk_size=64/256` 两条 K tile
路径。generic、Ascend、Enflame、Iluvatar 均保持 E5/E4/E2d 已提交字节不变。

### Release 与构建身份

- source / verification commit：`cfbef9c9f96ae9f26d67f43d22f0e20d57d9c30b`。
- 三个 fallback 的 SHA-256 均为 E2d generic
  `c50cda381c48712e108e34578c9805e74422b6b7b81be9b6dd6b2972d3753c47`；测试
  SHA-256 为 `cc5c27e243723b58936b5e5c166769d803fbe2785b99da7e460fa6b7a053fd6b`。
- fresh Git-object release：`gpu:/tmp/flagos-task12-e6-release2.xJG6ZU`；新增专项
  1/1、全套 12/12 unittest 通过。4 个 fallback dot 变体最大 94
  registers/thread、1,024 bytes shared，stack/local 均为 0。
- release log SHA-256
  `00bbd779c3fac4e21c0f5c2d31fb251bb32ee3650e9dbe7284162ad02044983f`；release
  script SHA-256
  `a35b9d3c1bcd7e1fae8fa6db8df7e8c2835363926aa24c437b8fd1dd6c65fb2f`。
- canonical ZIP：
  `artifacts/competition/chunk_state/e6-cfbef9c/chunk_state.zip`，53,117 bytes，
  SHA-256
  `09e1f66998074ffecdae1fc88259c7ea403c941f56cbe03977e221ef500e729c`；7 个成员为
  generic + amd/ascend/enflame/iluvatar/kunlunxin/metax，`unzip -t` 和第二次
  `--verify-existing` 均通过。

Task 18 submission `5087` 已实证国际 A 选择 `_nvidia`；因此 `_amd` 对国际 B
是当前最小后缀推断，不得添加 `_nvidia` fallback 覆盖国际 A 的 `16.583x` generic
路径。即使 `_amd` 未命中，按 E5 实测值回填其余回退，预计平均约 `3.524x`；若命中
并恢复国际 B，则约 `3.658x`。平台门禁仍为 8/8 correctness、每芯 `>=0.1x`；无论
结果如何 E6r 只提交一次，再根据实际 selected file 决定是否开发新的 Huawei 候选。

### E6r 平台终态：8/8 valid，团队最佳 `3.828875x`

2026-08-26 17:59:17 CST 执行唯一一次正式提交（submission `5107`，当日序号
`12`，额度 `19/30`→`18/30`）。平台对象存储回读 53,117 bytes，SHA-256 与本地
规范 ZIP 完全一致；本次 `file_url_sha256` 为
`0c147ad8cdf01b48aa4bd4ccff2fb985e4668054b70e53ea7e5f43758d1e4c63`。

终态为 `completed` / **valid** / `is_team_best=true`，八芯全部 correctness 通过：

| 芯片 | E6r | 选中文件 |
| --- | ---: | --- |
| 天数 | `2.0100x` | `chunk_state_iluvatar.py` |
| 沐曦 | `2.7425x` | `chunk_state_metax.py` |
| 燧原 | `0.9420x` | `chunk_state_enflame.py` |
| 海光 | `4.5405x` | `chunk_state.py` |
| 昆仑 | `0.2505x` | `chunk_state_kunlunxin.py` |
| 华为 | `0.3290x` | `chunk_state_ascend.py` |
| 国际 A | `17.9020x` | `chunk_state.py` |
| 国际 B | `1.9145x` | `chunk_state_amd.py` |
| **平均** | **`3.828875x`** | |

E6r 相比旧团队最佳 E3 `2.096625x` 提升 `1.73225x` 绝对均值、约 **82.62%**。
三份回退全部被目标芯精确选中：昆仑恢复 correctness，沐曦/国际 B 恢复旧性能；
generic 低精度路径在国际 A 由 E5 的 `16.583x` 进一步测得 `17.902x`。本候选按
预注册规则停止，不再重投。下一步只评估官方 FlagGems Ascend 同构 state kernel
能否单独提升当前最低但已过门槛的华为 `0.329x`，不得改动其余七个已验证成员。

## E7：Ascend 低精度 Cube dot（候选就绪，只提交一次）

状态：Git-object release 与规范 ZIP 门禁通过；等待实时 preflight

E6r 已把 generic 与六个非华为 vendor 的选择和性能全部实证。本轮只改变
`chunk_state_ascend.py`：FP16/BF16 且 `x/B` dtype 相同、`headdim` 与
`dstate` 属于 `{64,128}`、`chunk_size` 属于 `{64,128,256}` 时，采用
64×64×64 的低精度 `tl.dot`，FP32 scale、accumulator 和输出保持不变；其余
shape 继续走 E6r 的 32×32 IEEE 路径。一维 `min(total,4096)` grid-stride、
真实 strides、GQA 和所有索引均保持原结构。

### 固定官方依据与淘汰过程

- FlagGems `a7620cc191a0b42e040194622c5758b22a7a25dc` 的 Ascend FLA state
  kernel 证明低精度操作数、FP32 state accumulator 与 4-warps dot lowering
  可用；只借计算形态，不复制其双 accumulator、二维 grid、连续布局或递归语义：
  <https://github.com/flagos-ai/FlagGems/blob/a7620cc191a0b42e040194622c5758b22a7a25dc/src/flag_gems/runtime/backend/_ascend/fla/chunk_delta_h.py#L73-L83>。
- 同一固定 commit 的 Ascend BF16 matmul 明确包含 64×64×64 配置与 FP32
  accumulator + dot 主循环：
  <https://github.com/flagos-ai/FlagGems/blob/a7620cc191a0b42e040194622c5758b22a7a25dc/src/flag_gems/runtime/backend/_ascend/ops/matmul_bf16.py#L37-L42>、
  <https://github.com/flagos-ai/FlagGems/blob/a7620cc191a0b42e040194622c5758b22a7a25dc/src/flag_gems/runtime/backend/_ascend/ops/matmul_bf16.py#L111-L121>。
- 首版 128×64、stages2（源码 SHA-256 `1ceb5749…`）虽 13/13 与代理
  GM `1.371x`，但为 254 registers，拒绝；64×64、stages2
  (`47c71197…`) 为 177 registers，仍拒绝。最终只再减 stages 2→1，
  不把失败形态带入 release。

### Source、screening 与 release

- source / verification commit：
  `294990c2c0139c5be75e378ca8c872bb2645f607`。
- Ascend 源码 SHA-256：
  `d1f44f3bad114c79dd8e649118b24e6d336f610544521e0acc93d48137ae63d0`；
  测试 SHA-256：
  `22886b682896d945d2eff0c3ab0bb6b41e28c8882ce4cf4159ffb8c6313d4475`。
  新增一个测试方法覆盖 FP16/BF16、chunk 64/128/256、M/N 64/128、GQA
  与 FP32 输出。
- exact-byte screening：
  `gpu:/tmp/flagos-task12-e7-final-screening.6O1uCT`；13/13 unittest；7 个
  affected 点含 `total_programs=4128>4096` 的低精度 grid-stride case，
  GM `1.414147x`、最差 `1.253906x`。快路径为 121–123 registers、16 KiB
  shared、0 spill/local/forbidden IR。两个 control 为 `1.0000/1.022364x`；
  后者只是正向超出原对称噪声窗，未改路径没有回退。screening log / harness
  SHA-256 分别为 `8536cf8e6b6141e436046e72ccb39449e57d0be2d9178ffd63175f4384b384f3` /
  `a8c2b7e7bc0d6e9fdea5944d3b1ccbbd83d847b0ba30c0f5e0ed38b0b8cf4058`。
- fresh Git-object release：`gpu:/tmp/flagos-task12-e7-release.cMf4Ug`
  （mode 0700）；13/13 unittest，affected GM `1.412083x`、最差
  `1.250000x`，123 registers、16 KiB shared、0 spill/local，control 无
  回退，`SCREENING_RELEASE_GATE_OK`。release log / A/B harness / run script
  SHA-256 分别为
  `dfeba23b608fb1cbcaefd725fb58a96c7f8285099da38e92dc89033d10041b81`、
  `8d18944fb31a6841f36d097d615fb5dca9d54409428f25f7ae8d1f13544de8d5`、
  `61256fddf5c1c8e49f08f7d8b7ecf1873789d571414de26ffb743088fa764b2e`。

### 不可变 ZIP 与平台止损

canonical ZIP 为
`artifacts/competition/chunk_state/e7-294990c/chunk_state.zip`，53,755 bytes，
SHA-256
`583b55a2518091cd707ff6dbf1080a10bd1fe2fd690da2885d0bfd48daae04a8`。
7 个成员中只有 `chunk_state_ascend.py` 相对 E6r 改变；generic 与
amd/enflame/iluvatar/kunlunxin/metax 六个成员逐字节冻结。`unzip -t`、
二次 `--verify-existing` 和 Git blob 对照均通过。

本候选只提交一次。华为 correctness 失败、低于 0.1x、未选中 Ascend 文件，
或 valid 但不高于 E6r 的 `0.329x`，均立即停止 Task 12 并保留 E6r；不做
tile/stages 追投。新均值公式为
`3.828875 + (Huawei_new - 0.329) / 8`；达到 4.0x 需要华为至少
`1.698x`（当前的 5.16 倍）。

### E7 平台提交：7/8 已通过，华为等待回调

2026-08-26 19:42:04 CST 执行唯一一次正式提交（submission `5124`，当日
序号 `13`，额度 `18/30`→`17/30`）。上传后对象存储回读 53,755 bytes，
SHA-256 与本地规范 ZIP 完全一致；本次 `file_url_sha256` 为
`eac0b58b07b97f39e9969a1c29ce62c1b0a646028ac0fc1a9e9650e2b468ce3a`。

19:42:39 CST 状态为 `evaluating`：天数 `1.9810x`、沐曦 `2.7315x`、
燧原 `0.9390x`、海光 `4.5410x`、昆仑 `0.2510x`、国际 A `17.8175x`、
国际 B `1.9180x` 均 correctness 通过且选中文件与 E6r 一致；华为已选中
`chunk_state_ascend.py`，状态 `waiting_callback`，尚无 speedup。提交 intent
已是 `submitted`，不得重试；等待同一 submission 终态后再写回均值。

### E7 平台终态：8/8 valid，华为 6.44 倍，团队最佳 `4.0371875x`

2026-08-26 19:48:20 CST 只读重查确认 submission `5124` 已终态
`completed` / **valid** / `is_team_best=true`，八芯全部 correctness 通过：

| 芯片 | E7 | 选中文件 |
| --- | ---: | --- |
| 天数 | `1.9810x` | `chunk_state_iluvatar.py` |
| 沐曦 | `2.7315x` | `chunk_state_metax.py` |
| 燧原 | `0.9390x` | `chunk_state_enflame.py` |
| 海光 | `4.5410x` | `chunk_state.py` |
| 昆仑 | `0.2510x` | `chunk_state_kunlunxin.py` |
| 华为 | **`2.1185x`** | `chunk_state_ascend.py` |
| 国际 A | `17.8175x` | `chunk_state.py` |
| 国际 B | `1.9180x` | `chunk_state_amd.py` |
| **平均** | **`4.0371875x`** | |

华为由 E6r 的 `0.3290x` 提升至 `2.1185x`，为 **6.44 倍**；全题平均由
`3.828875x` 提升 `0.2083125x` 至 `4.0371875x`（约 +5.44%）。结果证明
FlagGems 官方 Ascend 低精度 dot / Cube 配置可迁移；64×64×64、stages1
比筛选淘汰的 128×64 和 stages2 更符合本题资源边界。Task 12 达到预注册
成功条件，永久停止，不再追投；额度保持 `17/30`，转 Task 09 官方 GCU300
grid cap 方案。
