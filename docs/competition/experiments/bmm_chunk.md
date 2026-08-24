# Task 09 `bmm_chunk` 实验记录

## S0：generic baseline

状态：本地静态检查、NVIDIA 代理验证和不可变 ZIP 门禁通过；
E1 `BLOCK_N=64` 和 E2 `K>=256 → BLOCK_K=64` 均未过预设门禁，保留 S0；未提交平台
验证时间：2026-08-24 CST

### 契约

- 接口：`bmm_chunk(a, b, chunk_size, causal=False)`；`causal` 当前保留但
  不参与计算，`True/False` 必须产生相同完整矩阵。
- `a`、`b` shape 均为 `[B, T, G, K]`。
- 将 T 切成 `nchunks = T / chunk_size` 后，对每个 batch/chunk/group 计算
  `a_chunk @ b_chunk.T`。
- 输出是 out-of-place 的 `[B, nchunks, G, chunk_size, chunk_size]`，固定
  FP32；三种输入 dtype 均先转 FP32 再乘加。
- 题面虽然用 `ceil(T / chunk_size)`，紧接着的无 padding `reshape` 只有
  `T % chunk_size == 0` 才成立；S0 对不整除输入明确报 `ValueError`。
- 容差：FP32 `1e-4/1e-4`，BF16 `1.5e-2/1.5e-2`，
  FP16 `1e-2/1e-2`（atol/rtol）。
- 支持八芯：天数、沐曦、燧原、海光、昆仑芯、华为、国际通用 A/B。
- 缓存目录显示提交窗口为 2026-08-20 20:00 至 2026-08-27 19:59:59，
  最低加速比 `0.1x`；提交前以平台页面为准。

### 固定参考

- Mamba 固定 commit
  [`95d8aba8/ssd_bmm.py`](https://github.com/state-spaces/mamba/blob/95d8aba8a8c75aedcaa6143713b11e745e7cd0d9/mamba_ssm/ops/triton/ssd_bmm.py)：
  完整读取了 forward/backward、autotune、seq_idx、causal 和 wrapper。
- 赛题只保留四维 grouped forward；删除 3D、seq_idx、backward、contiguous
  copy、CUDA device context 和九档 NVIDIA autotune。
- 上游默认输出可跟随输入 dtype，且 FP32 dot 未固定 precision；赛题要求输出
  FP32，因此 S0 load 后显式转 FP32，并用 IEEE dot 禁用 TF32。

### 构建身份

| 项目 | 值 |
| --- | --- |
| 源文件 | `src/flaggems_sglang/ops/bmm_chunk.py` |
| 源文件 SHA-256 | `b533a3d59f883b01716297f603e344ad6d9399f011223a97ee5705309ed35843` |
| S0 原始测试 SHA-256 | `1f4d1bdc848afcb64c50d614c873784088a415821a7d30bbaec303a7558e06fd` |
| 当前测试 SHA-256 | `29a6d3dfe792e5887e730b7c3135e8f6cb1ffc12c943e89fa3a706ebf0e123fd` |
| 源码 commit | `b05bfeb` |
| 验证 commit | `3ff39721ed95222fff2f1c6304b71a53ff1bce1b` |
| ZIP | `artifacts/competition/bmm_chunk/s0-b05bfeb/bmm_chunk.zip` |
| ZIP SHA-256 | `058b016c309c0affa5ecbbcb125de415a6565be93e2b76a9535473021169c4e3` |
| S0 原始远端目录 | `gpu:/tmp/flagos-batch2.SQaIX2`，mode 0700 |
| 最终发布门禁目录 | `gpu:/tmp/flagos-bmm-chunk-release.gj9TrN`，mode 0700 |
| 最终发布门禁日志 SHA-256 | `4087a0f1448067affeeed3841928d2650e380950e9d78e95eec64d7a84737050` |
| 平台 | 未提交；未经用户当次确认不得上传 |

### 唯一候选配置

- 固定 `BLOCK_M=32`、`BLOCK_N=32`、`BLOCK_K=32`、4 warps、1 stage；
  无 autotune 或 vendor 参数。
- grid 为 `(M/N tiles, B, nchunks*G)`；每个 program 计算一个输出 tile。
- a/b 四维 strides 与 output 五维 strides 全部参与 64-bit 地址计算；不做
  `contiguous()` copy。
- K 和 chunk_size 均使用完整 tail mask，支持非 2 次幂。
- `tl.dot(a_fp32, b_fp32, input_precision="ieee")` 显式禁 TF32，accumulator
  与输出均为 FP32。
- 无 PyTorch 核心计算、fallback、设备判断、autotune 或 vendor 文件。

### 正确性与静态检查

本地 Python `py_compile` 通过。远端 RTX 5070 Ti 16 GB 环境：Python
3.12.13、PyTorch 2.13.0+cu130、Triton 3.7.1、CUDA 13.0、compute
capability 12.0。

最新远端公开接口 unittest：5/5 通过，覆盖：

- FP16、BF16、FP32 输入与固定 FP32 输出；
- `K=13`、`chunk_size=7` 等非 2 次幂 tail；
- causal `True/False` 与完整 reference 同结果；
- 四维均带真实 stride 的非连续 a/b；
- a/b 使用互不相同的四维 stride，不能误用另一输入的 stride；
- `chunk_size=31/32/33` 会实际进入第二个 M/N tile，`K=31/32/33`
  会实际进入第二个 K block；
- 空序列输出和不整除 T 的显式契约；
- a/b shape、dtype、数值保持不变。

FP32 随机用例在 `1e-4` 容差通过，也验证了 IEEE dot 路径。Black 79、
isort、flake8 均通过。最终门禁使用 source commit `b05bfeb` 和
verification commit `3ff3972` 的逐字节内容；上述结果只证明 NVIDIA 代理路径。

### NVIDIA 代理性能

wrapper-inclusive；每项先验证正确性，再用
`triton.testing.do_bench(warmup=20, rep=50)`。

| dtype | `[B,T,G,K]` | CS | S0 (ms) | reference (ms) | speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| FP16 | `[1,128,1,64]` | 64 | 0.006903 | 0.012637 | 1.831x |
| BF16 | `[2,256,4,64]` | 64 | 0.009569 | 0.019537 | 2.042x |
| FP32 | `[1,128,2,31]` | 32 | 0.006243 | 0.010426 | 1.670x |
| FP16 | `[2,126,3,47]` | 63 | 0.007811 | 0.017180 | 2.199x |

ZIP 由 commit `b05bfeb` 的算子子树直接生成，仅含顶层 UTF-8
`bmm_chunk.py`。`unzip -t`、10 MB、成员名和逐字节 SHA-256 门禁均通过。

## 单变量优化审计

新证据目录为 `gpu:/tmp/flagos-bmm-chunk.36ba06`，mode 0700。环境为
RTX 5070 Ti 16 GB、driver 610.57.04、Python 3.12.13、PyTorch
2.13.0+cu130、Triton 3.7.1 和 CUDA 13.0。两个候选均先通过
5/5 unittest 和额外 reference 正确性，再进行五组轮换、wrapper-inclusive
`warmup=25, rep=100` A/B；正确性不等于性能晋升。

| 证据 | SHA-256 |
| --- | --- |
| 扩展 S0 门禁日志 | `a7731117792acf34f4d54d89c6f9346a17fb7ef4c1deee5aa47d98ccbd0af0ed` |
| E1 源码 / 测试 | `1e67038c3fbd1e13be82ebd5303b75b76141c4bab5ad15d88d2c9e7f25ed78fd` / `98ef96364148b71854f94d622220df77dc993e1843d87bc9fa9a7814673a35b3` |
| E1 门禁 / A/B 脚本 / A/B 日志 | `10a1f23e81ed622f148742c99a89896f3869c6eac3a553b41b5b3021d34bfea4` / `2d4bc8c8b77ca8804410b0e4f3cc9afd58e9567f7b8e0b639469b83ed5210166` / `ef0232b7cbbf0d3fb4a63a4ec3e94bc15b323f7c1d4003c647dd3d200efe354f` |
| E2 源码 / 测试 | `5d5d9cf6435550f6c5c0721e92d9b13511fb942dbf2829d98f7ac9e184eff2c5` / `91a4cc993b3faec8d5390d3347c6bbaf6feed9ef3327ee82d20fab74f79d0b68` |
| E2 门禁 / A/B 脚本 / A/B 日志 | `2894bf982346cba60e74cdd7879e532cd4079ecfa32326ed07385a501e356dde` / `ce074f43aef62797e54a7f0dab92d008a9152a2b633a430e3bcba792435ad309` / `2b0ce0b4a4ce1edbcd37d10b292a9c792cbc78e94cc20f692d06f56752ce4507` |

| 运行 | PID | 启动时间（CST） | 日志文件 |
| --- | ---: | --- | --- |
| 扩展 S0 门禁 | `76316` | `2026-08-24 04:32:40` | `baseline-validation.log` |
| E1 门禁 | `76451` | `2026-08-24 04:33:43` | `candidate-validation.log` |
| E1 A/B | `76590` | `2026-08-24 04:35:14` | `bmm-ab.log` |
| E2 门禁 | `76769` | `2026-08-24 04:38:13` | `e2-validation.log` |
| E2 A/B | `76875` | `2026-08-24 04:39:12` | `bmm-k64-ab.log` |
| S0 最终发布门禁 | `77021` | `2026-08-24 04:41:49` | `gpu:/tmp/flagos-bmm-chunk-release.gj9TrN/release-gates.log` |

### E1：`chunk_size % 64 == 0 → BLOCK_N=64`，拒绝

E1 只放大连续输出维 N tile，M/K、4 warps、1 stage、grid 和 IEEE FP32
归约不变。固定 Mamba 的 9 档 forward 配置中包含 `32x64x32`，
但上游还同时使用 2 warps/5 stages 和低精度 dot，只能支持实验，不能直接复制。

| 指标 | 结果 |
| --- | ---: |
| 全 12 点几何平均 | `0.924849x` |
| 9 个受影响点 | `0.901075x` |
| 3 个 S0 control | `1.000000x` |
| FP16 / BF16 / FP32 受影响点 | `0.841682/0.870137/0.998957x` |
| 最差受影响点 | `0.750000x` |

完整逐点 p50：

| dtype | `[B,T,G,K]` | CS | S0 (ms) | E1 (ms) | S0/E1 |
| --- | --- | ---: | ---: | ---: | ---: |
| FP16 | `[1,64,1,64]` | 64 | 0.006144 | 0.008192 | 0.750000x |
| FP16 | `[1,256,2,47]` | 128 | 0.006176 | 0.008192 | 0.753906x |
| FP16 | `[2,1024,4,128]` | 256 | 0.118784 | 0.112640 | 1.054545x |
| BF16 | `[1,128,1,64]` | 128 | 0.006144 | 0.008192 | 0.750000x |
| BF16 | `[1,384,2,65]` | 192 | 0.014336 | 0.016384 | 0.875000x |
| BF16 | `[2,256,4,33]` | 64 | 0.008224 | 0.008192 | 1.003906x |
| FP32 | `[1,64,1,31]` | 64 | 0.006144 | 0.006144 | 1.000000x |
| FP32 | `[1,256,4,64]` | 128 | 0.010208 | 0.010240 | 0.996875x |
| FP32 | `[2,512,4,129]` | 256 | 0.077824 | 0.077824 | 1.000000x |
| FP16 control | `[1,128,2,64]` | 32 | 0.006144 | 0.006144 | 1.000000x |
| BF16 control | `[1,126,2,65]` | 63 | 0.008192 | 0.008192 | 1.000000x |
| FP32 control | `[1,130,2,31]` | 65 | 0.006144 | 0.006144 | 1.000000x |

仅 FP16 `CS=256,K=128` 达到 `1.054545x`；低并行 FP16/BF16 稳定慢
12.5–25%。S0 最高 114 registers、8 KiB shared、0 spill；E1 达 154 registers、
12 KiB shared 并出现 2–4 spills，因此同时失败性能和资源门禁。
资源检查覆盖 13 个 S0 编译变体（均 8 KiB）和 14 个 E1 变体
（4 个 8 KiB control、10 个 12 KiB 受影响变体）。

### E2：`K >= 256 → BLOCK_K=64`，拒绝

E2 保持 M/N、grid、4 warps、1 stage 不变，只对长 K 减少循环次数。
三 dtype 的 K=255/256/257/511/512/513、M/N tail 和互异 stride 全部在
题面容差内通过。

| 指标 | 结果 |
| --- | ---: |
| 全 9 点几何平均 | `1.051676x` |
| 6 个 K>=256 受影响点 | `1.078507x` |
| 3 个 K255 control | `1.000000x` |
| FP16 / BF16 / FP32 受影响点 | `1.005505/1.112697/1.121263x` |
| 最差受影响点 | `1.002232x` |

完整逐点 p50：

| dtype | `[B,T,G,K]` | CS | S0 (ms) | E2 (ms) | S0/E2 |
| --- | --- | ---: | ---: | ---: | ---: |
| FP16 control | `[1,128,2,255]` | 64 | 0.016384 | 0.016384 | 1.000000x |
| BF16 control | `[1,128,2,255]` | 64 | 0.016384 | 0.016384 | 1.000000x |
| FP32 control | `[1,128,2,255]` | 64 | 0.016384 | 0.016384 | 1.000000x |
| FP16 | `[1,64,1,256]` | 64 | 0.014368 | 0.014336 | 1.002232x |
| BF16 | `[1,128,2,256]` | 64 | 0.016384 | 0.014336 | 1.142857x |
| FP32 | `[2,256,4,256]` | 64 | 0.024576 | 0.022528 | 1.090909x |
| FP16 | `[1,66,2,257]` | 33 | 0.016528 | 0.016384 | 1.008789x |
| BF16 | `[1,128,2,512]` | 64 | 0.026624 | 0.024576 | 1.083333x |
| FP32 | `[1,128,2,513]` | 64 | 0.030720 | 0.026656 | 1.152461x |

E2 的 K64 变体使用 16 KiB shared、0 spill/scratch/local，PTX 无 TF32，
但寄存器最高 144，超过预设 128 门槛；FP16 `1.005505x` 也低于每 dtype
`1.02x` 门槛。因此不用事后细分阈值追求过拟合，拒绝 E2。两个候选都没有
commit 或生成新 ZIP，当前仍是 S0 候选就绪、未提交。
E2 资源检查覆盖 14 个 S0 编译变体（均 8 KiB）和 20 个候选变体
（6 个 8 KiB control、14 个 16 KiB 受影响变体）。

### 已知风险与下一步

- 平台未公开 correctness/benchmark shape；代理 shape 不能证明八芯性能。
- `input_precision="ieee"` 是标准 Triton API，但尚未由八种编译器全部验证。
- 3D grid、runtime K loop 和固定 32 tile 可能需要针对单芯编译/性能反馈调整；
  在没有平台 shape/profile 新证据前不再追加本地 tile 试探。
- 若首次平台仅单芯失败，保持 generic 与已通过芯片不变，只做最小 vendor
  override；下一门禁是用户针对上述 ZIP 路径、哈希和实时额度作当次确认。

## E3：低精度输入直送 `tl.dot`（晋升）

状态：源码、测试、release 代理验证和不可变 ZIP 门禁通过；未提交平台

验证时间：2026-08-24 07:48–07:52 CST

### 单变量

E3 保持 `32x32x32` tile、grid、4 warps、1 stage、FP32 accumulator 和输出不变。
当 a/b 同为 FP16 或 BF16 时，load 后直接送入 `tl.dot`；两个低精度数的乘积可由
FP32 accumulator 精确承载，差异只剩允许容差内的归约顺序。FP32 或 mixed-dtype
继续先转 FP32，并显式使用 IEEE input precision。该路径与固定 Mamba forward
的输入 dtype 用法一致，不增加 vendor API 或设备分支。

源码 commit 为 `a5afc1902d1eeb5b6c42657110f4dfed31d90b18`；源码 SHA-256
`e3bd27c94affcf9987f75751e3a23fadc6eb4b9056ce26fe3e013dbd7380b7b7`，
测试 SHA-256
`6e9f0c318d62213b7f9460984244c1a357c6304f111f842511be27601b382836`。
新增 mixed FP16/BF16 用例，明确锁住原 FP32 control 路径。

### Release 代理验证

release 目录 `gpu:/tmp/flagos-bmm-chunk-release-e3.F5TK11`，mode 0700；source
和 verification commit 均为 `a5afc1902d1eeb5b6c42657110f4dfed31d90b18`。
RTX 5070 Ti 16 GB；driver 610.57.04；Python 3.12.13；PyTorch
2.13.0+cu130；Triton 3.7.1；CUDA 13.0。

- py_compile、Black 79、isort、flake8、逐文件哈希和 unittest 6/6 通过。
- 31 个随机/边界/非连续 case 同时对照 S0、E3 和 reference 通过。五轮交替
  A/B，`warmup=25, rep=100`，按每轮配对 speedup 中位数统计：8 个 FP16/BF16
  affected 几何平均 `1.4306x`，FP16/BF16 分别为 `1.4308/1.4304x`，
  范围 `1.0000–4.1475x`；4 个 FP32 controls 全部 `1.0000x`。
- 另做 23 个 `chunk_size/K=31/32/33/63/64/65/127/128/129/257`、
  FP16/BF16 pair-cancellation、动态范围与 mixed-dtype case，最大绝对误差
  `3.815e-5`，抵消和 mixed controls 的误差为 0。
- S0/E3 各 13 个编译变体，最高均为 114 registers/thread、8 KiB shared；
  spill、global scratch、local load/store 和 TF32 PTX 标记均为 0。

release gates、A/B、扩展正确性、provenance 和 A/B harness 的 SHA-256 依次为
`b0e4055ede75a871a2325b1b64a6147607e08e2eb8cdf403375b0d413afad884`、
`6009c15b6efa953315e722247b0cbeafe2be486f68053f8c730279accc6c820f`、
`73a7f1bf507d89a95834a412cbefbe646568ddbca6dbb9f15ddcc31c22976f68`、
`57ff2e4ee58b14dd399b227353dcc868972f7fb94f6d7ada35f621312be4ddfc`、
`1375e537579b22ce0dd767298f2003ecd17ba5f00f3eea72c45e9d64d8f99836`。

### 产物

- ZIP：`artifacts/competition/bmm_chunk/e3-a5afc19/bmm_chunk.zip`
- ZIP SHA-256：
  `d8577b2ee314cad758f756d47794685240448d2654baf6b685a7e53fac415b95`
- 大小 / 成员：5,323 bytes；顶层 `bmm_chunk.py` 5,201 bytes。

确定性构建、`--verify-existing`、`unzip -t`、UTF-8、basename、10 MB 和逐字节
来源门禁均通过。标准低精度 `tl.dot` 在其余七芯仍需平台证明；未打开浏览器、未读取
实时额度、未提交平台，旧确认不授权此 ZIP。

## E3a：预防性 Ascend capped grid + 天数 split-fp16 vendor（首投候选）

状态：release 门禁通过，候选就绪，等待 preflight 与提交

背景：E3 generic 为 3D grid `(tiles, batch, nchunks*ngroups)`，总数在大 shape
可超 65535（Ascend 展平限制，Task 12 已平台证实）；其 fp32 输入路径使用
fp32 操作数 + `input_precision="ieee"` 的 `tl.dot`，按 Task 12 E2a–E2d 的
平台证据（fp32 操作数 dot 在天数静默失败，fp16 操作数可执行）需要天数
vendor。题面 fp32 容差 1e-4，直接降 fp16（rel ~1e-3）不满足，故 fp32 路径
采用 split-fp16 三点积仿真（a = a_hi + a_lo，dot(a_hi,b_hi)+dot(a_hi,b_lo)
+dot(a_lo,b_hi)，有效精度 ~2^-22）。

- `_ascend/ops/bmm_chunk.py`：Task 12 平台验证的 capped grid-stride 模式，
  一维物理 grid `min(total, 4096)`，逻辑 id 按 batch → chunk·group → tile
  分解；kernel 数学、BLOCK 32/32/32、ieee dot 与 causal wrapper 逐行保持
  generic。
- `_iluvatar/ops/bmm_chunk.py`：仅改 dot 块——fp16/bf16 输入路径
  `USE_INPUT_DTYPE` 保持原样走裸 `tl.dot`；fp32 路径 split-fp16 三点积，
  累加仍为 fp32。grid 与 generic 相同（天数无展平限制）。

新增回归：`test_ascend_capped_grid_multi_iteration`（(2,128chunks,8组,
chunk64) 总数 8192 > 4096，覆盖两轮 grid-stride，fp32 1e-4 / fp16 1e-2）与
`test_iluvatar_split_fp16_precision`（chunk/k 组合 `64/64、33/65、257/64`
× 三 dtype，fp32 按 1e-4 验证）。共 8/8 unittest。

screening 目录 `gpu:/tmp/flagos-bmm-chunk-vend.tTsu4Q`（mode 0700）：首跑
PID `106470`/`106549` 因脚本 FILES 行漏列 iluvatar 文件与测试一处超长行
先后停止（脚本修正后 SHA-256
`31ba42b832644b6b7f107dc694289a13610a492abf7e52390cac97057e0ab076`）；最终
PID/PGID `106699`（00:16:55，wall 900s）静态门禁与 8/8 unittest（1.438s）
通过，`screening.log` SHA-256
`43745ba1d038837042833bc9a41baacf1d425429cf630e78024295c5a07f63f1`。环境
RTX 5070 Ti 16 GB、Python 3.12.13、PyTorch 2.13.0+cu130、Triton 3.7.1、
CUDA 13.0。Ascend vendor blob
`a2fa97a9cf813f36629f64469f62dd6469c788113f571a0b86322985966cc682`，天数
vendor blob
`3d547d5b0bd6b68bac11c6faadccb6f0e87cb5d4616256e1f897619a30e4c3e1`，测试
`3d18cf686e3902d6b8166c8fc9574836f4ca8f4f2ead3fce3204a416b9f5aaf1`。
release 目录 `gpu:/tmp/flagos-bmm-chunk-vend-release.sAPBpV`，
source/verification commit `4fb53d61e83ff06e6b3d5e81bb5634ae3a351828`，
PID/PGID `106903`（00:19:43，wall 600s），`RELEASE_OK`，`release.log`
SHA-256
`b16e3416900d5fedf86dbd045b0f7d5b504d62841bf4243f2260c961a87da7a8`。

canonical ZIP
`artifacts/competition/bmm_chunk/e3a-4fb53d6/bmm_chunk.zip`，16962 bytes，
SHA-256
`3de1f1379434e1e7e65fd74168354485da92e003cc061b445f1bc2ccce924f4f`，成员
`bmm_chunk.py`、`bmm_chunk_ascend.py`、`bmm_chunk_iluvatar.py`，
`unzip -t` 通过。平台门禁：8/8 通过且每芯 ≥0.1x；天数与华为各自选中
vendor，其余六芯 generic。
