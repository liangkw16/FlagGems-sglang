# Task 15 `decode_attention` 实验记录

## S0：generic MHA baseline

状态：S0 已打包并通过本地门禁；等待当次上传确认

验证时间：2026-08-24 01:28–01:37 CST

源码 commit：`f431ba4`

### 契约

| 项目 | 值 |
| --- | --- |
| 公开接口 | `decode_attention(q, k_buffer, v_buffer, kv_indptr, kv_indices, sm_scale)` |
| MHA 约束 | `H_Q == H_KV` |
| 输入 | q `[B,H_Q,D]`；K `[P,H_KV,D]`；V `[P,H_KV,D_v]`；CSR page 索引 |
| 计算 | `softmax(q @ K.T * sm_scale) @ V`，logits/softmax/累加均 FP32 |
| 输出 | `[B,H_Q,D_v]`，FP32，out-of-place；输入不变 |
| 支持芯片 | 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用 A/B，共 8 款 |
| 截止 / 门槛 | 2026-08-27 19:59:59；`speedup_threshold=0.1` |

固定来源：本地 Task 15 题面、SGLang
`8014d9d062c3cc5d393596ecdf2f7009191965df` 的 production decode attention，
以及 community commit `0e8023d` 的 `whd3/decode_attention.py`。S0 只保留在线
softmax 思路，删除 reference/demo、Torch 计算、`.contiguous()`、host/设备识别、
设备迁移、fallback、split 临时张量和设备 SM 策略。

2026-08-24 01:31 CST 公开 API 状态：41 次提交、10 支队伍、1 支达到门槛；
榜首 c2flow 为 8/8、78.2958x。动态值仅用于当时决策。

### 唯一候选

- 单 kernel、每个 `(batch, query_head)` 一个 program；4 warps、1 stage。
- Q/K/V、CSR indptr/indices 和输出全部使用真实 stride；page ID 转 int64 后参与
  地址计算。
- `BLOCK_D`、`BLOCK_DV` 为实际维度的下一 2 次幂。sequence tile 为
  `max(8, min(32, 8192 / max(BLOCK_D, BLOCK_DV)))`，限制常见大 `D_v` 的
  二维 gather tile。
- 跨 sequence tile 使用 FP32 online softmax；完整 mask 覆盖任意非整块长度。
- 仅 `torch.empty` 分配 FP32 输出；非空路径核心计算全在 Triton。

### 验证

| 项目 | 值 |
| --- | --- |
| 源文件 SHA-256 | `886332facce98b6fa3ab783de064e322ddbcabb3a66f9b17bad70914fc3212aa` |
| 测试 SHA-256 | `8e7a714a79a3487d4c4ebb4fc4abbebad92fc12f1afbdd5beda750344019f43b` |
| ZIP | `artifacts/competition/decode_attention/s0-f431ba4/decode_attention.zip` |
| ZIP SHA-256 | `850cf12333241a450b342edbd2e108dca5841ddfb4f576129df45d863e5123b9` |
| ZIP manifest | 顶层 `decode_attention.py`，5016 bytes |
| 远端证据目录 | `gpu:/tmp/flagos-batch2.SQaIX2` |
| 远端环境 | RTX 5070 Ti 16 GB；PyTorch 2.13.0+cu130；Triton 3.7.1；CUDA 13.0 |

- 共享 unittest 5/5 通过；Task 15 覆盖 FP16/FP32，真实非连续 strides，
  int64 非连续 CSR，变长 `1/35/65`，`D=33,D_v=17`，以及
  `D=64,D_v=257` 和空 batch。
- `py_compile`、`git diff --check`、Black 79、isort、flake8 均通过；本地与
  远端源码/测试哈希一致。
- wrapper-inclusive FP16 benchmark：`B=8,H=16,D=D_v=64,L=128`，S0
  `0.038249 ms`，题面 reference `0.620144 ms`，代理 speedup `16.213x`。

### 风险与下一步

- NVIDIA 代理不能证明其余七款芯片；平台前不增加 vendor 文件。
- 公开 reference 对单个 `L=0` 的序列会在 empty `amax` 处报错；S0 覆盖其
  可定义的空 batch，不另行发明空序列数值。CSR tensor 需与输入位于同一设备。
- 单 program 顺序扫描长序列。若平台正确但长序列性能不足，下一单变量候选是
  split-KV；若大 `D/D_v` 编译超资源，再按失败芯片做 128-wide dim chunk。
- ZIP 由 commit `f431ba4` 直接生成；`unzip -t`、UTF-8、单一 `.py`、10 MB、
  basename 和 ZIP 内源码哈希门禁均通过。尚未上传或消耗额度；上述 ZIP 需要
  用户当次确认。

## E1：sequence tile 上限 32 → 64（否决）

状态：未晋升；源码已恢复 S0，未生成新 ZIP，未提交平台

验证时间：2026-08-24 06:42–06:45 CST

### 假设与单变量

S0 在 `BLOCK_D/BLOCK_DV <= 128` 时仍把 sequence tile 限为 32，在线 softmax
对 L=128 需循环 4 次。固定 community 来源存在同类 64/128 tile 的可编译锚点，
因此 E1 只把 host 公式中的上限 `32` 改为 `64`；kernel、grid、4 warps、数学、
mask、stride 和 FP32 累加均保持不变。`D=129` 与 `D_v=257` controls 的 tile
仍分别为 32 和 16。

### 结果与停止门禁

- 证据目录 `gpu:/tmp/flagos-decode-attention-e1.EC7hh6`，mode 0700；静态/共享
  unittest PID/PGID `82952`，5/5 通过。候选源码 SHA-256 为
  `28852592ad7dc14e4645bbe7ca3c11ade8cb5259382d883a1a7b365a625c899e`。
- 正确性覆盖三 dtype、非连续 Q/K/V、strided int32/int64 CSR，以及
  `D=33,D_v=17,L=31/32/33/63/64/65`；含性能规格共 66/66 次
  base/candidate module-case 检查通过。
- 五轮交替 AB/BA，`warmup=25, rep=100`，每次 wrapper 批量 20 次；每点以
  5 个配对 speedup 的中位数计。9 个 affected 点几何均值 `1.3201x`，
  FP16/BF16/FP32 分别为 `1.3373x`、`1.2880x`、`1.3355x`，范围
  `1.1976–1.4488x`。6 个 controls 几何均值 `1.0000x`，范围
  `0.9999–1.0001x`。
- 性能门禁全部通过，但资源门禁失败：base/candidate 最大均为 128
  registers/thread、2,048 bytes shared，base 18 个变体 spill 总数为 0；
  candidate 18 个变体中 4 个出现 2–4 spills，合计 12。映射探针确认连续
  `D=D_v=128` 的 BF16 变体已有 2 spills，其余出现在非连续边界特化；两者都在
  公开契约内。global scratch 与 local load/store 为 0。
- corrected A/B harness SHA-256 为
  `40725faa5110729a0fc3232d33c2e8d50fdc8cbc2b129557bb3345922281098f`；
  `screening.log`、`ab-corrected.log`、`resources.log` SHA-256 分别为
  `baf53bfe3dfd2dbfaadf37fc2e279bae1c6acf717e2e2bb8ee42babfc15e0e34`、
  `868b5ce86d9e481dcd2e10da7b58f961a70a53c65d8415591bd4b0f6361f6322`、
  `d73af848a3744758c6638fc2c500ae31a381314ebddc9ae865354a171bc60bdf`。

验证环境：NVIDIA GeForce RTX 5070 Ti 16 GB，driver 610.57.04，Python 3.12.13，
PyTorch 2.13.0+cu130，Triton 3.7.1，CUDA 13.0。新增 spill 在其余后端可能放大为
寄存器溢出或编译资源失败，因此严格按预设“无新 spill”停止：E1 不晋升，不继续
用 dtype/stride 特判过拟合代理机，当前不可变候选仍为 S0 `f431ba4`。

未打开浏览器、未读取或消耗平台额度。后续若有真实平台 shape/resource 反馈，再
考虑 split-KV 或 vendor 限定配置；旧的上传确认不授权任何新产物。

## E2：NVIDIA 长度门控 tile64（晋升）

状态：源码、测试、release 代理验证和不可变 ZIP 门禁通过；未提交平台

验证时间：2026-08-24 07:34–07:48 CST

### 变更与边界

E2 保持 generic S0 字节不变，只新增自包含 NVIDIA vendor。E1 已证明 tile64
对 `L>=33` 有收益但会让 `L<=32` 变慢并在四个编译特化中产生 12 spills；E2
用可在 host 直接读取、不会触发 device 同步的 `kv_indices.numel()` 选择 tile：

- 总 page 数不超过 `batch_size * 32` 时保持 S0 tile32/4 warps；否则用 tile64。
- tile64 默认仍是 4 warps；int64 CSR 或 BF16 `D=D_v=128` 使用 8 warps，
  消除 E1 已定位的寄存器溢出。
- kernel 数学、在线 FP32 softmax、真实 stride、mask 和输出均与 S0 相同；
  其余七芯继续使用 generic S0。

源码 commit 为 `59cb094e8e5662aadabce3bfc55a23c9e8a97e76`。generic SHA-256
仍为 `886332facce98b6fa3ab783de064e322ddbcabb3a66f9b17bad70914fc3212aa`，
NVIDIA vendor 为
`0c52bfe006212dd7dab9ab88229d4ca7bdec158bbd95a3434a0a642521ca2a07`，
测试为 `1776bf99e046b67d3bc1724eb09c44acd4b383898208567f877b3e995558f474`。

### Release 代理验证

固定 release 目录 `gpu:/tmp/flagos-decode-attention-release.6fs4hq`，mode 0700；
source 和 verification commit 均为 `59cb094e8e5662aadabce3bfc55a23c9e8a97e76`。
RTX 5070 Ti 16 GB 环境与 E1 相同。

- Black 79、isort、flake8、py_compile、逐文件 SHA-256 和共享 unittest
  7/7 通过；新增测试实际执行 tile32、tile64/4-warps、tile64/8-warps、
  int32/int64 CSR 和 BF16 资源敏感路径。
- 主矩阵共 66 次 base/vendor reference 检查通过。五轮交替 A/B，
  `warmup=25, rep=100`，每次 wrapper 批量 20 次：9 个长序列 affected
  几何平均 `1.3285x`，FP16/BF16/FP32 分别为
  `1.3370/1.3131/1.3354x`，最差 `1.1974x`；6 个 controls 几何平均
  `0.99997x`，范围 `0.99976–1.00010x`。
- 补充 `L=1/31/32/33/63/64/65` 共 84 次 correctness 检查通过；21 个
  dtype/length 点整体几何平均 `1.1045x`，最差 `1.0000x`。`L<=32`
  保持 S0，`L>=33` 获得收益。
- 24 个 NVIDIA vendor 编译变体最高 128 registers/thread、4,096 bytes
  shared；spill、global scratch、local load/store 均为 0。18 个 S0 变体
  同样无 spill/scratch/local。

release gates、主 A/B、短序列 A/B、provenance 和 A/B harness 的 SHA-256
依次为
`faf7691fffc5129aee13f1959651a40c2a49a32e669812cde2a6e0a6dfaccd44`、
`04efae18c3d1a4e5ecf1896ef982ceba0d84af2184fb434ee382de400cc9728b`、
`4e9a929b3bac5f2c37bfacef375e3cac2c810c37de1b6b2d391056364e8a9d30`、
`c64d29360d6d39dacd12b09edb0cd4375b30e2376eb1acd78a6cba2d3e430492`、
`40725faa5110729a0fc3232d33c2e8d50fdc8cbc2b129557bb3345922281098f`。

### 产物

- ZIP：`artifacts/competition/decode_attention/e2-59cb094/decode_attention.zip`
- ZIP SHA-256：
  `0170fd15d5da5e0bd268fa1c5d12c7e9ee36e5cb5af50625a33da26e6ef4da62`
- 大小 / 成员：10,637 bytes；`decode_attention.py` 5,016 bytes，
  `decode_attention_nvidia.py` 5,357 bytes。

确定性构建与 `--verify-existing`、`unzip -t`、UTF-8、basename、10 MB 和
逐字节来源门禁均通过。NVIDIA 结果只为代理证据；未打开浏览器、未读取实时额度、
未提交平台，旧确认不授权此 ZIP。
