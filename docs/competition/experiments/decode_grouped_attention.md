# Task 16 `decode_grouped_attention` 实验记录

## S0：generic GQA baseline

状态：S0 已打包并通过本地门禁；等待当次上传确认

验证时间：2026-08-24 01:28–01:37 CST

源码 commit：`f431ba4`

### 契约

| 项目 | 值 |
| --- | --- |
| 公开接口 | `decode_grouped_attention(q, k_buffer, v_buffer, kv_indptr, kv_indices, sm_scale)` |
| GQA 约束 | `H_KV < H_Q` 且 `H_Q % H_KV == 0` |
| 头映射 | `kv_head = query_head // (H_Q // H_KV)` |
| 输入 | q `[B,H_Q,D]`；K `[P,H_KV,D]`；V `[P,H_KV,D_v]`；CSR page 索引 |
| 计算 | `softmax(q @ K.T * sm_scale) @ V`，logits/softmax/累加均 FP32 |
| 输出 | `[B,H_Q,D_v]`，FP32，out-of-place；输入不变 |
| 支持芯片 | 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用 A/B，共 8 款 |
| 截止 / 门槛 | 2026-08-27 19:59:59；`speedup_threshold=0.1` |

固定来源与 Task 15 相同：题面、SGLang `8014d9d` production 实现和
community `0e8023d:whd3/decode_attention.py`。S0 仅保留在线 softmax 与正确
GQA 头映射；无 reference/demo、Torch 计算、连续化、host/设备识别、fallback、
split 临时张量或私有 API。

2026-08-24 01:31 CST 公开 API 状态：41 次提交、9 支队伍、1 支达到门槛；
榜首 c2flow 为 8/8、76.45365x。动态值仅用于当时决策。

### 唯一候选

- 与 Task 15 相同的自包含单 kernel 保守策略；每个 Task ZIP 不依赖另一文件。
- 每个 `(batch, query_head)` 一个 program；通过整数除法映射到 KV head。
- 所有输入按真实 stride 读取；int32/int64 CSR 均可；输出 FP32。
- `BLOCK_D/BLOCK_DV` 下一 2 次幂，sequence tile 受 8192-element tile 预算
  限制；4 warps、1 stage、FP32 online softmax、完整变长 mask。

### 验证

| 项目 | 值 |
| --- | --- |
| 源文件 SHA-256 | `d14bde1140cec961690d940da8b3d8e8c89a45e78a466bdeb88e6eb5be0cd7c9` |
| 测试 SHA-256 | `8e7a714a79a3487d4c4ebb4fc4abbebad92fc12f1afbdd5beda750344019f43b` |
| ZIP | `artifacts/competition/decode_grouped_attention/s0-f431ba4/decode_grouped_attention.zip` |
| ZIP SHA-256 | `4ed5e04d8453e100a38feff3d8986801fab9a13c4d77481e070a3260855136ef` |
| ZIP manifest | 顶层 `decode_grouped_attention.py`，5054 bytes |
| 远端证据目录 | `gpu:/tmp/flagos-batch2.SQaIX2` |
| 远端环境 | RTX 5070 Ti 16 GB；PyTorch 2.13.0+cu130；Triton 3.7.1；CUDA 13.0 |

- 共享 unittest 5/5 通过；Task 16 覆盖 BF16/FP32/FP16，`H_Q/H_KV` 为
  `8/2`、`4/1`，真实非连续 Q/K/V，int32 CSR，变长 `3/70`，
  `D=40,D_v=24`，以及 `D=64,D_v=257` 和空 batch。
- `py_compile`、`git diff --check`、Black 79、isort、flake8 均通过；本地与
  远端源码/测试哈希一致。
- wrapper-inclusive FP16 benchmark：`B=8,H_Q=32,H_KV=8,D=128,D_v=64,
  L=256`，S0 `0.137529 ms`，题面 reference `0.726847 ms`，代理 speedup
  `5.285x`。

### 风险与下一步

- NVIDIA 代理不能证明其余七款芯片；平台前不增加 vendor 文件。
- 公开 reference 对单个 `L=0` 序列未定义；只承诺空 batch。CSR tensor 必须
  与 Q/K/V 位于同一设备。
- GQA 当前每个 Q head 重复读取共享 KV head。平台正确后，第一性能候选是单个
  program 合并同组 Q heads；长序列再评估 split-KV，不同时改变两项。
- ZIP 由 commit `f431ba4` 直接生成；`unzip -t`、UTF-8、单一 `.py`、10 MB、
  basename 和 ZIP 内源码哈希门禁均通过。尚未上传或消耗额度；上述 ZIP 需要
  用户当次确认。

## E1：同组 query heads 复用 KV tiles

状态：受控 grouped fast path 已通过提交字节发布门禁并生成不可变 ZIP；未提交平台

验证时间：2026-08-24 07:15–07:31 CST

源码 commit：`bc729bd18a039bc183d7bb2aa6b069869ff08007`

### 设计

E1 保留 S0 kernel 作为完整 fallback，只为 tensor-core 友好的 GQA shape 增加
一个 `BLOCK_H=16` kernel：每个 `(batch,kv_head)` program 一次载入 K/V，并为
同组 query heads 共同计算 online softmax，消除 S0 对共享 KV head 的重复读取。

fast path 同时满足以下条件才启用：

- `4 <= H_Q/H_KV <= 16`，且 `batch * H_KV >= 64`；
- Q/K/V 同 dtype，且为 FP16 或 BF16；
- D/Dv 为 16–128 的完整 2 次幂 tile；
- Q/K/V 最内维连续。

其余 FP32、group 2/3 或大于 16、低并行、大/尾维、混合 dtype 和非连续最内维
全部逐字节走原 S0 Triton kernel。QK 使用输入 dtype dot 并 FP32 累加；online
maximum、denominator、probability、`P @ V` 和输出均为 FP32，其中 `P @ V`
显式 IEEE。没有设备识别、Torch 计算 fallback、autotune 或临时张量。

### 收敛证据

宽门控原型暴露了三个可复现风险，均在发布版本中从结构上排除：

- FP32 group2 点仅 `0.8308x`，且 FP32 grouped 变体达到 255 registers/thread、
  40-byte stack，因此最终 FP32 走 S0。
- 非连续最内维 grouped 变体达到 149–163 registers/thread，因此只对连续最内维
  启用。
- `D=33,Dv=65` 尾维变体达到 200 registers/thread，因此只覆盖完整幂次 tile。

最初将 FP32 probability 降到输入 dtype 做 `P @ V` 虽快，但存在大值相消误差；
发布版本改回 FP32 IEEE dot。构造两 token 概率接近 0.5、V 为 `+1000/-1000`
的反例后，FP16/BF16 最大误差仅 `2.56e-5`。

### 发布验证

| 项目 | 值 |
| --- | --- |
| 源文件 SHA-256 | `f1885afedc06f059a27b9bd66554ad49a44f096cf11f6f4c10816fea3c769ee7` |
| 测试 SHA-256 | `cb3207348e1d1efce1f38b47ed59f28ebb5b54ac0aaf67b1a9ffa302335516b0` |
| ZIP | `artifacts/competition/decode_grouped_attention/e1-bc729bd/decode_grouped_attention.zip` |
| ZIP SHA-256 | `088a9ebfcae10a608528e5614a684997753cd8693ac13f49496383ced4ca80c0` |
| ZIP 大小 / 成员 | 10,020 bytes；顶层 `decode_grouped_attention.py` 9,868 bytes |
| 远端发布目录 | `gpu:/tmp/flagos-decode-gqa-release.rZei2o`，mode 0700 |
| 远端环境 | RTX 5070 Ti 16 GB；PyTorch 2.13.0+cu130；Triton 3.7.1；CUDA 13.0 |

- source commit 导出的发布字节 unittest 6/6 通过。新增直接 fast-path 回归覆盖
  group 4/8/16、FP16/BF16 和连续 D/Dv=64；原有三 dtype、非连续 stride、
  int32/int64 CSR、尾维、大 Dv、变长和空 batch 全部保留。
- 扩大筛选另覆盖 group 4/8/16、D/Dv `15/17`、`33/65`、`128/128`，长度
  `31/33/65/257`，scale `-0.25/0/4`、重复 page、int32/int64 和混合 dtype；
  最大绝对误差 `0.002424`，均通过题面容差。
- Black 79、isort、flake8、`py_compile`、`git diff --check` 和独立只读审查
  通过；发布目录 source/test 与 commit SHA-256 一致。
- 发布 grouped 变体为 72–80 registers/thread、12,352-byte shared、0 stack、
  0 local、0 scratch；PTX 保留 QK MMA，FP32 `P @ V` 使用 IEEE 路径。

五轮交替、wrapper-inclusive 配对结果：

| dtype / shape | S0 ms | E1 ms | paired E1/S0 |
| --- | ---: | ---: | ---: |
| FP16 `B8,HQ32,HK8,D128,Dv64,L256,g4` | 0.0392 | 0.0293 | 1.3373x |
| FP16 `B16,HQ32,HK4,D64,Dv128,L512,g8` | 0.0871 | 0.0472 | 1.8453x |
| BF16 `B8,HQ32,HK8,D128,Dv64,L256,g4` | 0.0391 | 0.0295 | 1.3261x |
| BF16 `B16,HQ32,HK4,D64,Dv128,L512,g8` | 0.0871 | 0.0473 | 1.8416x |

受影响点几何均值 `1.5668x`、最差 `1.3261x`。七个 fallback controls 覆盖
FP32 group4/8、group2、低 grid、大 Dv、尾维和 group32，中位 speedup 范围
`0.9994–1.0033x`，几何均值 `1.0009x`。

确定性打包器从 commit 生成单一 generic 成员；重复验签得到相同 canonical ZIP
SHA-256，`unzip -t`、UTF-8、10 MB 和逐字节来源门禁均通过。E1 仍只有 NVIDIA
代理证据，各 vendor 对 mixed-precision QK dot 和 16-head layout 的 lowering
必须由平台证明。上传前需重新读取实时额度，并取得用户针对 Task 16、上述绝对
ZIP 路径和完整 SHA-256 的当次确认。

## E1a：天数 fp16-dot vendor（首投候选，≤2 次预算）

状态：release 门禁通过，候选就绪

E1 generic 的两个 `tl.dot(..., input_precision="ieee")` 为 fp32 操作数，按
Task 12/22/23 平台证据在天数上不可执行。题面容差 `atol=3e-2, rtol=1e-2`
宽松，`_iluvatar` vendor 直接把两处 dot 操作数降为 fp16（累加/softmax 路径
保持 fp32），无需 split 仿真。grid 为一维小规模（bs×heads），华为/昆仑/
燧原均不加 vendor。测试把天数 vendor 纳入 grouped 回归循环。screening
`gpu:/tmp/flagos-dga-vend.8QQN3z`（先后补传 decode_attention.py、
_nvidia vendor 两个测试依赖；含一次 Black 回拷），最终 PID/PGID `115351`
（03:11:11，wall 900s），7/7 unittest（0.910s），`screening.log` SHA-256
`c07da3861cc827f0dbcff4c624a65e15adc5b7791895bd1e4f2450d09660063a`。
天数 vendor blob
`96ae034897773a3d3066ccd62b40bb1095b875067a9b41346d0cc61a926b7ae8`，测试
`63309bcc9149b7d5c694ddd1c9fe0cbb68fd01aabc192636f276f94fd552b7e3`。
release `gpu:/tmp/flagos-dga-release.*`，source/verification commit
`9801c56bdfa8be3854f991e339783177a29abade`，`RELEASE_OK`，`release.log`
SHA-256
`20df68d617a46dc6511f4b8da28c0b381f1cdadfde28381d9c6507cb21dd520b`。
canonical ZIP
`artifacts/competition/decode_grouped_attention/e1a-9801c56/decode_grouped_attention.zip`，
SHA-256
`c8dd889f7820f52e73bfc2ea1c88c007b2a969c3811e0970b260f911e25a5b2b`，
成员 generic + iluvatar，`unzip -t` 通过。

### E1a 平台结果与 Task 16 停止（保留 1 次额度）

E1a 于 03:15:04 CST 提交（submission `4499`，当日序号 `18`，额度区间
`14/30`→`13/30`，`file_url_sha256` 为
`e2452ab1c7d6a20e78392fb0eb46323040548a26b98c94df66eee60a5cd00948`）。
五芯通过（天数 vendor 生效）；燧原为评测超时 + `Segmentation fault`、
昆仑为评测超时 + `Aborted`（评测器崩溃族）、华为 case 7 数值失败且指纹
与 Task 15 华为同型（整行重复，Ascend flash 型 kernel 边界 bug）。三芯
失败互独立且无单变量公共解，按"大把握才提交"原则不再使用第 2 次额度，
Task 16 记 5/8 停止；未用额度转 Task 14。
