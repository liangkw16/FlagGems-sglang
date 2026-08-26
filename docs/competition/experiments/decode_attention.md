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

### E2/E2a 平台结果与 Task 15 停止（2 次预算用尽）

E2 于 03:00:45 CST 提交（submission `4496`，当日序号 `17`）：华为 case 8
数值失败（输出呈整行重复的确定性错误指纹，疑似 Ascend 对该 flash 型
kernel 特定 shape 的 load/broadcast lowering bug）；其余在评。E2a（华为
2D-grid vendor，免 div/mod；commit `5add38cd68e26ac52a7e6d8e2ab6f541b0cfd512`，
submission `4502`，03:27:01，额度区间 `13/30`→`12/30`）华为仍以同一
case 8 同指纹失败，昆仑为评测超时崩溃（`Fatal Python error: Aborted`，
与 Task 16/23 昆仑评测器崩溃同族）。最终 6–7/8（invalid）。Task 15 两次
预算用尽，停止；燧原/国际系芯片路径未被两轮证伪，天数/沐曦/海光正常。

2026-08-26 实时 API 复核修正上述早期观察：两轮终态实际均为 5/8。E2 的
燧原在验证阶段 1830s 后 segmentation fault，E2a 又遇服务线程卡死自动恢复；
昆仑两轮均 1830s 后 `Fatal Python error: Aborted`；华为两轮均为 case 8
确定性重复行数值错误。最近 E2a 五颗通过芯分数为天数 14.7238x、沐曦
34.5286x、海光 81.8842x、国际 A 98.6508x、国际 B 84.9164x，合计
314.7038x。

## E3-i32：三失败芯统一 page-routing int32（一次性重开）

状态：submission `5053` 已终态 5/8，正确性无效；Task 15 永久停止。

### 新证据与单变量

Task 17 submission `5048` 已平台实证：删除 Enflame kernel 的 i64 IR 后，燧原
从连续三轮 `PassManager` 全 case 失败恢复为正确且 0.3885x。GitHub 一手源码又
显示 FlagGems [Kunlun paged attention](https://github.com/flagos-ai/FlagGems/blob/d1c970e0c9ccb3c26d9fc8de906a7e21a64cc0a1/src/flag_gems/runtime/backend/_kunlunxin/ops/flash_kernel.py#L1162-L1170)
把 page table load 固定为 int32；FlagTree
[#922](https://github.com/flagos-ai/FlagTree/issues/922) 记录 XPU attention
runtime loop 的 pointer-state pass 失败。旧 Task 15 generic/Ascend kernel 则让
`kv_indptr` 的 dtype 贯穿动态 loop，并把每个 page ID 强制 cast 为 int64。

因此按用户批准的高倍数冲刺破例重开一次，但只改变一个结构变量：新增
Enflame/Kunlun vendor，并在三颗失败芯的 wrapper 把 int64
`kv_indptr/kv_indices` 有界转换为 int32；kernel 的 `start/end/pages` 全部保持
int32。Ascend 保留既有 2D grid，Enflame/Kunlun 保留 generic 1D grid；数学、
BLOCK、warps、stages、在线 softmax 与 Q/K/V stride 全部冻结。generic、NVIDIA
vendor 逐字节不变，因此五颗已过芯不受影响。竞赛 page/cache 元数据与官方实现均
在 int32 范围；题面未明写大于 `2^31-1` 的理论边界作为已知风险，不加会拒绝
隐藏 case 的 host assert。

若三颗失败芯只达到最低 0.1x，而五颗冻结芯保持 E2a 分数，平均也约为
`39.3755x`。E3-i32 只正式提交一次；任一芯仍失败或低于 0.1x 即永久停止
Task 15，不追加 dtype/tile/warps/stages 碰运气。

### 验证与产物

source/verification commit 为
`49d07a9d0a35183278dfbcb4232b326816652451`。generic/NVIDIA/Ascend/Enflame/
Kunlun/test SHA-256 分别为
`886332facce98b6fa3ab783de064e322ddbcabb3a66f9b17bad70914fc3212aa`、
`0c52bfe006212dd7dab9ab88229d4ca7bdec158bbd95a3434a0a642521ca2a07`、
`0a1b0fbc51d8d5b1464e3e7a35ca78dfa444ed6f3250d0d38c152b45ab7b0fc7`、
`bb7befaff0688cebb247cad7ec18769e324f397b053ef009e46a9bfd811884c0`、
`bb7befaff0688cebb247cad7ec18769e324f397b053ef009e46a9bfd811884c0`、
`ca76378eb93e3e702d1015fdf5e4edc1ef77e6f8a259b0aa2fc50c547d35b1ab`。
三份失败芯 vendor 均无 `tl.int64`/`to(tl.int64)` 命中。回归覆盖非连续
int64 CSR 的转换路径、非连续 int32 CSR 的无转换路径、长度 `1/35/65`、
非连续 Q/K/V、输入不变和 reference 容差。

最终 screening 位于
`gpu:/tmp/flagos-decode-attention-i32-screening.wPNIGQ`，base commit
`68b4e90`，最终 PID/PGID `135624`；`replay.sh` SHA-256 为
`01ff475a2ae59aa044f1176cf8fa6758e7bb939f612aa1f014121835e33e0c51`。
8/8 unittest 通过（0.867s），尾行为 `SCREENING_OK`，最终
`screening-r2.log` SHA-256
`d5fc0d47aa6107a7d2bdf0c3fbfe199e3d8969e7e8c841baba34a0c896c30a56`；
输入前后 manifest SHA-256 均为
`6467fd12903118168acdb560dbbce03e323e37900b063fa20ddb924dcc3df5a0`。

同目录 wrapper-inclusive NVIDIA 代理基准 PID/PGID `135374`；脚本/日志
SHA-256 分别为
`228a8ca878d6dda321baef4abf4dbb9fd02725975ae9c4dacdadd4d1bd73d3f1`、
`900f0f3cec5982ea9755fee64972fa9e588e0acfcaddaf85a2e073b045a6b390`，
尾行为 `BENCHMARK_OK`。三个代表 shape 的候选延迟为
0.01438/0.03214/0.04946ms，相对 generic 为 1.452/1.212/1.475x，相对
reference 为 16.29/19.25/6.45x；峰值显存增量 2,048/37,376/20,992B。
该结果只排除代理门槛与资源灾难，不外推三款目标芯。

Git-object release 位于
`gpu:/tmp/flagos-decode-attention-i32-release.CuOydK`，PID/PGID `135933`；
`replay.sh` SHA-256
`d7766bc8abe8a3c726fd0d3cb0771787ee19745246db00dcb1c5297b4802c7e2`。
8/8 unittest 通过（0.638s），尾行为 `RELEASE_OK`，`release.log` SHA-256
为 `b531106bf8b270c2f61119b9c21775c5b7e42aeb36f01ab6a73a5be6cfef2949`；
release 前后 manifest 与最终 screening 完全相同。

canonical ZIP 为
`artifacts/competition/decode_attention/e3-i32-49d07a9/decode_attention.zip`，
26,924B，SHA-256
`ba8b013b8e21c4f2edd34808c6dae93764ee0a8795b9801831955714534e98d7`；
成员为 generic + ascend/enflame/kunlunxin/nvidia。`--verify-existing`、
`unzip -t` 与成员白名单全部通过。

### E3-i32 平台终态与停止

E3-i32 于 2026-08-26 15:23:10 CST 严格按一次性 preflight 命令提交一次，
submission `5053`，当日序号 `5`，额度 `26/30`→`25/30`；平台回读的远端
ZIP 与上述不可变产物逐字节一致。终态为 `completed / invalid_correctness`，5/8：

| 芯片 | 结果 | speedup / 失败指纹 |
| --- | --- | --- |
| 天数 | 通过 | `15.3936x` |
| 沐曦 | 通过 | `36.3022x` |
| 燧原 | 失败 | case 8，`grid.x Required 131072 > 65535` |
| 海光 | 通过 | `81.6698x` |
| 昆仑 | 失败 | 约 1833.6s，执行超时；子进程 `Fatal Python error: Aborted`，停在 compile worker |
| 华为 | 失败 | case 8，仍为确定性重复行数值错误 |
| 国际 A | 通过 | `91.2698x` |
| 国际 B | 通过 | `83.4538x` |

平台回传后曾在本地准备但**未发布、未打包、未提交**一个 launch-cap 诊断候选
（commit `ee13599`）：Ascend 按 32,768、Enflame 按 65,535 分片，并用显式
program 起始偏移保持逐 program 数学不变；远端 10/10 回归通过。它能直接解释并
规避 Enflame 的 launcher 上限，也可能规避 Ascend 的旧 32,774-block 报错，
但无法改变昆仑连续多轮独立的 1830s 编译器崩溃。按 E3 预先写定的“一芯仍失败
即永久停止”门禁，Task 15 到此停止，不为未闭环的两芯修复再消耗额度。

## E4：三失败芯结构恢复（最终一次官方证据重开）

状态：Git-object release 与不可变 ZIP 门禁通过，待实时 preflight。

E3 后新增的固定一手证据同时覆盖了三个独立失败指纹，构成结构性重开依据，而非继续
调 BLOCK/warps/stages：

- Enflame 直接采用 `ee13599` 已准备的 65,535 host 分片；平台 case 8 的精确错误
  就是 `grid.x Required 131072 > 65535`。
- `libtriton_jit@acd8b52` 会把 Ascend 真实 launch block 数钳到物理 Vector Core
  数，而原 grid 仍作为 system arg 传入 kernel；这解释了 E2/E3 case 8 的重复行。
  E4 改成一次只启动 `num_vectorcore` 个真实 worker，并用
  `tl.num_programs(0)` 在 kernel 内遍历全部 `B*H` logical program。core 数通过
  FlagGems-Experimental 同源 Ascend cumsum 已使用的 driver property 接口读取。
- 固定的 `FlagGems-Experimental@c73617a` Kunlun sparse attention 使用逐 key
  标量循环、`tl.sum(q*key)`、FP32 online softmax 与 2 warps。E4 迁移该 lowering，
  再把 value 维固定按 64 分片；`D_v=257` 时每个 program 只保留 64-lane FP32
  accumulator，避免旧二维动态 tile 在 compile worker 卡死。

五颗已通过芯继续使用逐字节冻结的 generic/NVIDIA，SHA-256 仍为
`886332facce98b6fa3ab783de064e322ddbcabb3a66f9b17bad70914fc3212aa`、
`0c52bfe006212dd7dab9ab88229d4ca7bdec158bbd95a3434a0a642521ca2a07`。
source/verification commit 为
`96a0dfef9a8ec5ab04516dea4d68934a856c92ce`；Ascend/Enflame/Kunlun/test
SHA-256 分别为
`ea88124951442445b8c63ef06b500a30372925e6fd51818ffc13ed647b0fcd6a`、
`74b5532257f70084444d978874dce98d8d1f95f8e0af7fe185eaac9e5f5271ee`、
`f36ecbf8a75eaba4021d069385c38f3b8dadf088ff824179a2dce0afe5d871a5`、
`85e65c742cf861dd15e1bcf089007dcbe8e7e08579617758fa8afd84f931ff88`。

最终 screening 位于
`gpu:/tmp/flagos-decode-attention-e4-screening-r2.mVVzZl`，PID/PGID
`156980`；11/11 unittest 在 0.643s 内通过，尾行为 `SCREENING_OK`。
`replay.sh`、`screening.log`、输入 tar 和前后 manifest SHA-256 分别为
`1225b3defe4972be1e263121987a0ffd3e5a3adac723e861a4fdd155a9d9c795`、
`efa6d2e3772350c0d2125e29f641b5cece59ee6b712e8df2039c7e735d7593f9`、
`198d24935ee299ba636aa7db4d85207f91b39b6e309ce646acae3fa92751af3f`、
`bafd7d2f23325321096823b3daa3af8fbbb04895e48cb67d916a6a859d350d91`。

同目录 benchmark PID/PGID `157115`；脚本/日志 SHA-256 为
`25019e009beaf5eb9316b1651c12c5fd3306f97d7106f5dbf8e6e7461d388017`、
`be2aaba491a4028dc111752ddbdaf525a403365e47ef4acd978478f70148665d`。
四个代表 shape 经六轮交替测得 Kunlun 相对 reference 为
`4.217/5.285/1.424/7.057x`，均超过预设 `0.2x` 门槛；相对 generic 为
`0.373/0.334/0.328/0.492x`。峰值显存增量为
2,048/37,376/20,992/26,112B；Triton metadata 均为 0 global scratch，PTX
无 local load/store。Ascend 与 Enflame 的 `B*H=131072,L=1` 实跑均精确正确，
最大绝对误差 0。NVIDIA 结果只排除语法、数值和资源灾难，不证明目标芯运行时。

Git-object release 位于
`gpu:/tmp/flagos-decode-attention-e4-release.0LKvLj`，PID/PGID `157279`；
11/11 unittest 在 0.646s 内通过，尾行为 `RELEASE_OK`。`replay.sh`、
`release.log`、Git archive 与前后 manifest SHA-256 分别为
`46586c2693416ffa7c0d33b782371c1b4d38a9cb9012fc68bd3a9e281e21f3c7`、
`ceffa6d28c10e409bec0de7fa0ccb9f512f1e2201acc699c743a3d954061abdc`、
`9f07e89dcf724aa3179e2eb824c1e7ffce65ea91242d2f096c53c33ade8cde1a`、
`bafd7d2f23325321096823b3daa3af8fbbb04895e48cb67d916a6a859d350d91`。

canonical ZIP 为
`artifacts/competition/decode_attention/e4-96a0dfe/decode_attention.zip`，
28,233B，SHA-256
`51ec3d98ca1da7e33bd1aaee93c398399855dfb7f60a6b4b8e73a3e6f0f9ca7a`；
成员为 generic + ascend/enflame/kunlunxin/nvidia，规范构建、
`--verify-existing`、`unzip -t` 与成员白名单全部通过。2026-08-26
22:46:58 CST 只读平台状态为 `competing/can_submit`，额度 10/30。

E4 最终只提交一次；五颗冻结芯按 E3 分数、三个失败芯仅按最低 `0.1x` 计算时，
有效均值下界约 `38.54865x`。任一目标芯仍失败或低于门槛即停止 Task 15，不追加
普通配置微调。
