# Task 38 `sigmoid_gate_topk_renorm` 实验记录

```current
task: 38
operator: sigmoid_gate_topk_renorm
batch: 3
validity: invalid_correctness
platform: E4 sub 8160 7/8;Kunlun快速执行但9/9数值失败
team_best_stage: S0
team_best_commit: 311570f
blockers: E3 exact tuple禁止重试;E4 exact tuple禁止重试
sealed: no
next: 仅评估移除dynamic rank写槽的新E5;无强证据则封存
updated: 2026-09-02
```

状态:S0 候选就绪

## 契约锁定

- 签名:`reference(logits, k, n_shared_experts, route_scale, global_scale, bias)`
- `logits [T, N+S]`(末 S 列共享);`bias [N]`;`global_scale [1]` fp32
- sel=sigmoid(routed)+bias → topk(argsort 降序);routed_vals 取
  **原始 logit**(非 sigmoid);probs=sigmoid(cat(routed_vals,shared));
  归一化 ×route_scale×global_scale;输出
  `(routed_weights [T,k] 输入dtype, indices [T,k] int32, shared_weights [T,S])`
- indices 精确匹配;容差 fp32 1e-4 / bf16 1.5e-2 / fp16 1e-2;八芯

## S0(2026-08-31,commit `311570f`)

- T31 机器(迭代 argmax+min-index 平局)**双确定性选轮**:第一轮
  indices + sigmoid 概率总和,第二轮复现选点写归一化权重——规避
  同 program store→load 可见性风险;共享专家恒活跃参与归一化;
  1D capped grid。
- screening(gpu:/tmp/t38.z8kOOd,字节与 blob 一致):unittest 3/3
  (3 dtype × 7 形状含 S=0/S>T 边界);bench 4/5 shape 1.42–2.61x,
  65536×64 小 N 档 0.536x(两轮选点在大 T 小 N 下偏慢,过门槛,
  平台 mix 待证)。
- 风险:topk 族昆仑前科(T25/26/27/31 皆崩);fp32 无 dot、无
  libdevice 超越函数(sigmoid=1/(1+exp))是本次差异点。

### S0 提交记录(2026-08-31 12:2x CST)

preflight 全过(tid `s2t1op038`,额度 4/30 消耗 1 → 3/30);单次
confirm 提交,评测入队,逐芯结果待回填。

### S0 平台终态(sub 6930,2026-08-31 14:0x CST)

**7/8**——七芯全过(卡_B 1.854/沐曦 1.580/华为 1.255/海光 1.215/
card_a 1.146/天数 0.819/燧原 0.421),**昆仑 1830s 验证段
Segmentation fault**(崩溃族第 15 例,与 Aborted 同族不同信号量)。

- topk 族昆仑结论第三次复现(T25/26/27/31/38);本 kernel 无 dot、
  无 libdevice 超越函数仍崩——"最干净形态"也未能幸免,题内止损
  (1 发即停,候选 `311570f` 封存可复用)。
- 第三批 17 题全部处置完毕:8 valid + 3 题 7/8 昆仑墙封存 +
  2 题语义黑盒止损 + 4 题早前止损。额度 3/30。

## E2:global_scale 保持设备侧(2026-09-01 06:4x CST)

状态:结构改写候选已完成单次平台评测；七芯性能机制兑现，但昆仑再现
同族崩溃，按 stop gate 重新封存，不得自动重载。

### 假设与单变量

- S0 wrapper 对题面 GPU `global_scale[1]` 做 host scalar extraction,
  每次调用产生 GPU→CPU 同步;T39 已测同类同步税约 25us,足以淹没
  本题短 top-k kernel,也会给 torch.compile/Inductor 增加 host graph break;
- E2 只把 tensor `global_scale` 作为指针传入 Triton,每个 program 一次
  `tl.load`;双确定性选轮、sigmoid/归一化数学、TOPK/BLOCK/grid、输出和
  Python scalar 兼容路径全部冻结。源码不含 `.item()`/`.tolist()`/`.cpu()`;
- KernelGen `optimize_kernel` 在完整负结果约束下生成同一指针方案;Huawei
  `specialize_kernel` 预检保持该 scalar pointer load 不变,其余未经实机
  验证的 NPU 改写全部拒绝,不进入候选。

### screening 与性能门

- base `2761fd8`,候选/测试 SHA-256 分别为
  `e4840878c98e2d2a061a60a5438b069f70af2868c0d10322963e38b823a82e03` /
  `33b565aed5255236706764c69b25f39e32ae5f5a6a42a28e9b8b0f9db10d9c9f`;
  `gpu:/tmp/flagos-t38-nosync.7FRzZI`,RTX 5070 Ti,PyTorch
  `2.13.0+cu130`,Triton `3.7.1`;py_compile/Black/isort/flake8、
  unittest **3/3** 全过(三 dtype×七 shape、indices 精确、tensor/scalar
  scale 输出逐位一致);screen_final2.log SHA-256
  `64a31528691b14e109fece4432db7a350c773041b1f9b16fd54b22316de1d115`;
- 五轮交替 wrapper-inclusive AB/BA,八形状 base/candidate 时间比分别
  `4.780/3.288/3.281/1.191/1.619/3.274/1.012/1.593`,几何均值
  **`2.188x`**,最差大 T/小 N 仍不回退;资源最大同为 122 registers、
  0 spill/0 scratch,candidate 某变体 107→105 registers;
- 代理晋级门(geomean ≥1.5x、任一 shape 不回退 >5%、0 新 spill)全部
  通过。benchmark script SHA-256
  `f7026a40c3a2fb5305360ab80bb8f6f18ee2f74561fec7597e1c48f42f354c98`。

### 构建身份与平台预注册门

| 项目 | 值 |
| --- | --- |
| source / verification commit | `b610b7a1ef6d494130121495b455d3b2ab9c09e8` |
| release | `gpu:/tmp/flagos-t38e2-rel.o1sM3e`,Git 对象导出,哈希前后一致,unittest 3/3,`RELEASE_OK`;日志 SHA-256 `8cfe4e5e8525000f1a17707b5279afd3d0f986589c4b2a9af2d24dfe1ced695b` |
| canonical ZIP | `artifacts/competition/sigmoid_gate_topk_renorm/e2-b610b7a/sigmoid_gate_topk_renorm.zip`,5255 bytes,SHA-256 `5dc0316b2f10bc07342405506a70a86f0e5a772ac497f16430eac83e86685c80` |
| ZIP 成员 | `sigmoid_gate_topk_renorm.py`,SHA-256 `e4840878c98e2d2a061a60a5438b069f70af2868c0d10322963e38b823a82e03`;`--verify-existing`/`unzip -t` 通过 |

- 基础门:8/8 valid、每芯 ≥0.1x;Kunlun pointer scalar load 编译/执行是
  唯一新增跨芯风险面。
- 机制门:此前通过的七芯中至少 5 芯严格高于 S0,且无芯回退 >10%;代理
  表明收益随短 shape 占比变化,不强行外推 2.188x 到八芯。
- 登顶门:平均严格高于实时榜首 Fields `7.60535x`。
- stop gate:昆仑再现 1830s/Segfault/compile-worker 同族指纹,或任一既过芯
  因 E2 失败,立即恢复封存,只留结构算法重写/工单路径,不做同字节或载体重投。

### E2 平台提交与终态(sub 7518,2026-09-01 06:44–07:15 CST)

- preflight 全过后单次 confirm，daily_seq 4，额度 27→**26/30**；远端
  对象回读与 canonical ZIP SHA-256
  `5dc0316b2f10bc07342405506a70a86f0e5a772ac497f16430eac83e86685c80`
  一致；该 submission/ZIP 不得重试；
- 终态 **7/8、invalid_correctness**。七个已过芯全部继续通过，七芯均值
  `1.1842x → 4.9415x`，合计 **4.17 倍**，去同步机制跨芯兑现：

| 芯片 | S0 | E2 | 倍率 |
| --- | ---: | ---: | ---: |
| 天数 | 0.8192 | **6.1470** | 7.50x |
| 沐曦 | 1.5804 | **2.9262** | 1.85x |
| 燧原 | 0.4208 | **0.8620** | 2.05x |
| 海光 | 1.2148 | **7.7342** | 6.37x |
| 华为 | 1.2546 | **2.7362** | 2.18x |
| 国际 A | 1.1462 | **5.9922** | 5.23x |
| 国际 B | 1.8536 | **8.1926** | 4.42x |

- **昆仑 1,833,762ms 验证段超时，子进程先退出且结果未送达，Fatal
  Python `Segmentation fault`；栈仍在
  `torch/_inductor/compile_worker/subproc_pool.py::_recv_msg`。** 与 S0
  1830s 崩溃是同一指纹，证明 device scalar pointer 改写没有改变 topk
  族在该平台的 runtime/compiler 墙；
- stop gate 已触发：不重投同包、不做注释/载体重载、不再消耗额度。
  E2 作为七芯机制证据保留；Task 38 只在昆仑平台 runtime 修复，或出现
  不含当前 topk lowering 的全新算法结构时重开。

## E3:Kunlun 选点/归一化两阶段(2026-09-02 08:4x–09:0x CST)

状态:新源码结构已通过 NVIDIA 代理 correctness、性能和资源门；KernelGen
Kunlun verifier 为基础设施 502，目标芯待 commit-bound release 后单次平台验证。

### 重开证据与单变量

- E2 之后的 T27 E8 已证明 Kunlun 上 direct 两阶段 router 可将同族
  `1830s` compile-worker timeout 降至 `12604ms`，并以 `0.6756x` 正确通过。
  T38 E2 的单 kernel 为规避同 program store→load 可见性，完整静态 TOPK
  选点执行两遍；隐藏 `k` 最大为 16。该后续跨题实证满足 E2 的“全新算法结构”
  重开条件。
- generic E2 SHA-256 继续冻结为
  `e4840878c98e2d2a061a60a5438b069f70af2868c0d10322963e38b823a82e03`，
  七个已过芯不改。只新增 `_kunlunxin` vendor：stage1 每行只做一次迭代
  top-k 并写 `int32` ids；stage2 跨 launch 读取 ids、gather 原始 routed
  logits，与 shared logits 的 sigmoid 一起做 FP32 联合归一化和最终写回。
- 非 2 次幂 `k=5/6` 用 `BLOCK_K=next_power_of_2(k)` + mask；`T>65535`
  用 host `row_start` 分段，不使用 device persistent loop。固定
  `num_warps=4,num_stages=1`，不含 Torch compute fallback、host scalar
  extraction、atomics、`tl.sort` 或输入修改。

### KernelGen MCP

- `optimize_kernel` 第一版因非 2 次幂 `tl.arange(0, TOPK)` 与 `T>65535`
  漏行在写盘前被 usability gate 拒绝；第二版按 `BLOCK_K` mask + host chunk
  修复后进入 screening。flake8 随后只检出一个未使用局部变量，MCP fix
  返回逐字节等价的单行删除，未手改算法。
- 独立 `generate_kernel(device="kunlun")` 以 T27 E8 和本候选为参考，生成了
  同一两阶段结构及 20 shapes × 3 dtype 测试矩阵；但目标验证
  `attempt 1/2/3` 均返回 **HTTP 502**，终态
  `verify_result.passed=false`。这与 T28 最小 `x+y` 对照相同，只能判定
  `generate_kernel → Kunlun verifier/worker` 不健康，不能作为候选失败或通过。

### Screening 与性能门

- base `7040b0e5ed178a1150446f14fa88031a6af92cd9`；目录
  `gpu-et:/tmp/flagos-sigmoid-gate-topk-renorm.KuByqj`，mode `0700`；
  correctness PID/PGID `264777`，benchmark PID/PGID `264912`。环境 RTX
  5070 Ti、PyTorch `2.13.0+cu130`、Triton `3.7.1`、CUDA `13.0`。
- vendor/test/helper SHA-256 分别为
  `1be1c7f7fbe6cbcfcfe57632a0c9fad592ac625f763bbbf080699a093a24ca87` /
  `883446df2662c53200fe8c18f723e64d077d14e6d93e1146f4c9653f4fcc3263` /
  `cdc5fe3e4cb5a85976f0a3414cd194bb53c79f6f2830be01f685f996b97ca0d7`。
  py_compile、Black、isort、flake8 与 generic+vendor unittest **4/4 PASS**，
  覆盖三 dtype、七组 shape、`k=1/4/5/6/8/16`、`N=33/512`、`S=0`、
  empty 和 `T=70000` fold；screening log SHA-256
  `bdd6f0adec10032b0006775ff401f1028bdbf22ec06398719b8e912d4bd90614`。
- 修正旧 helper 的 AB/BA 记账后，五轮 wrapper-inclusive 八 shape 候选相对
  reference 为 `4.3206/4.9466/5.0028/4.9786/6.7337/5.9686/0.8637/3.2410x`；
  每点均高于 `0.1x` 代理门。最大资源为 48 registers、0 spill、0 scratch、
  `num_stages=1`；generic control 最大 122 registers。benchmark log SHA-256
  `d5200b5829be1425d61b61472a59ac499758e2a2daac62963230f0038ed0d882`。

### 平台预注册门

- 基础门:八芯全部正确、每芯 `>=0.1x`；唯一受影响芯为 Kunlun，七芯继续选择
  字节冻结的 E2 generic。机制门:Kunlun 必须返回真实 validation id，且不再出现
  `1830s` compile-worker/Segfault/空 `failed_cases` 指纹。
- 晋级门:Kunlun `>=0.1x` 即恢复有效成绩；七芯合计按 E2 为 `34.5904x`，故
  最低有效投影平均约 `4.3363x`。本轮目标是新增有效排名，不把代理速度外推为登顶。
- stop gate:任一数值失败、同族 1830s 崩溃或 Kunlun `<0.1x` 即关闭本两阶段
  候选，不做同字节、注释载体或参数重投；只有平台给出新的源码级根因证据才重开。

### Commit-bound release 与不可变 ZIP

- source commit `2750268c469eff85cddfb435049a97ffc6fa3eeb`；独立 release
  `gpu-et:/tmp/flagos-sigmoid-gate-topk-renorm-release.T3Tudh`，mode `0700`，
  PID/PGID `265067`。从该 commit 的 Git objects 导出 generic/vendor/test/helper，
  静态门、unittest **4/4**、相同八 shape 性能/资源矩阵和前后 SHA 全过；release
  log SHA-256
  `1c44db4b354d17d97716961eff18ad6548ad2e85d97e6841bc8037307b9280d6`。
- canonical ZIP
  `artifacts/competition/sigmoid_gate_topk_renorm/e3-2750268/sigmoid_gate_topk_renorm.zip`，
  11887 bytes，SHA-256
  `c23e9dc368f48c34cd699c8956e6cbf9c24c797e38109ec4c2df536b5c334a0c`；
  dry-run/created/`--verify-existing` 一致，`unzip -t` 通过。成员仅 generic
  `e4840878...2e03` 与 `_kunlunxin` `1be1c7f7...a87`。

### Preflight 与上传失败(2026-09-02 09:07–09:08 CST)

- verification commit `98631f1869283bf6833f1fbb7330633507210b7a`，test SHA-256
  `883446df2662c53200fe8c18f723e64d077d14e6d93e1146f4c9653f4fcc3263`。
  实时 preflight 精确匹配账号 `15600308080`、团队 `SoulCoder`、T38/
  `s2t1op038`、source/verification commit、stage `e3`、两成员、ZIP/test/
  release log SHA、提交窗口和最小间隔；额度 `27/30`，intent nonce
  `b047b062d900445fa6a8a4337f35b1aa` 为 `prepared`。
- 按授权执行一次性 confirm 时，平台上传端点立即返回 **HTTP 503**；CLI 在任何
  可定位 submission/file URL 前将 intent 置为 `uncertain`。按幂等纪律未重试、
  未改 intent、未换载体。
- `2026-09-02T09:07:57+08:00` 与 `09:08:57+08:00` 两次只读 status 均显示
  额度仍为 `27/30`、`latest_submission_at=2026-09-02T01:17:22+08:00`，T38
  最新仍是旧 submission `7518`/daily seq 4；没有 E3 新记录。因此 E3 尚未进入
  八芯评测，候选门禁本身已通过，但发送状态未消解前不得自动 preflight/submit。

## E4:物化 selector + host-stepped 单次选点(2026-09-02 11:1x CST)

状态:新算法、新 commit、新 ZIP SHA 的独立候选已完成 commit-bound release；E3
的 `uncertain` tuple 继续冻结，绝不重试。

### 历史证据与结构

- 昆仑实机成功样本 T27/T28/T37/T39/T40 的共同点是把 persistent、复杂循环、
  间接读写拆成规则的小 kernel；T38 E3 仍把 `TOPK<=16` 完整展开在一个 kernel，
  与 T31 E8 一样继续命中 1830s compile-worker 崩溃。
- E4 只改昆仑路径：先物化 chunk-local FP32 selector；Python 按 rank 重复发射同一个
  `_select_one_kernel`，每次 kernel 仅含一次 max/min 并把已选 lane 写为 `-inf`；最后
  单独 gather 原始 logits、联合 shared experts 归一化。三个 kernel 均固定
  `4 warps/1 stage`，`row_start/rank` 禁止特化，无 kernel 内循环、Torch compute、
  host sync、persistent grid-stride、atomics 或输入修改。
- 公开有效 shape 与既有平台证据均为 `1<=k<=16`；`k=0,S>0` 仍沿用 generic/E3 的
  既有早退语义，不作为本轮新增变量。NVIDIA `torch.topk` 对精确平局的次序不稳定；
  诊断中 E3/E4 对同一平局逐位一致，候选没有改变既有最低 index tie 规则。

### Screening 与 release

- screening `gpu-et:/tmp/flagos-sigmoid-gate-topk-renorm.IorLFZ`；base
  `49b8c08c4bbd116c7a30edd97117a55e572b2f15`；vendor/test/helper SHA-256 为
  `5f2c6dec4d782859aefd47145a1a4c70f8c4c1804444fa95b85082512f29996e` /
  `883446df2662c53200fe8c18f723e64d077d14e6d93e1146f4c9653f4fcc3263` /
  `5fd7de19ccad6cc0294dac09e300199ac317292f29eb320c6fa7bfcfe00df5f1`；
  unittest **4/4 PASS**，E3/E4 indices 全 shape 逐位一致；E4 八 shape 相对 reference
  `3.5810/3.0445/3.3230/3.5976/3.8432/3.6694/0.7286/1.5986x`，最大
  23 registers、0 spill、0 scratch。日志 SHA-256
  `aa3cf922a0836e8b79a03ee341dc86ea5f24cdd78fedea798c0001cd38a2f8b3`。
- source/verification commit
  `901c880507070beff781d746c1f95815ad7c3916`；release
  `gpu-et:/tmp/flagos-sigmoid-gate-topk-renorm-release.ud5RSX` 从 Git objects 导出，
  unittest **4/4 PASS**，八 shape/资源与 screening 一致；release log SHA-256
  `35a2e7d8e59781097fb05a67d7e0cd6623bf978836fd7c7b6b2374709ea8df27`。
- canonical ZIP
  `artifacts/competition/sigmoid_gate_topk_renorm/e4-901c880/sigmoid_gate_topk_renorm.zip`，
  12603 bytes，SHA-256
  `d3e8bd9baf9ebfb2d5cd76484bc5584266e7bec14b9bfe59184a6d3bfcbde5f4`；成员仅
  generic `e4840878...2e03` 与 Kunlun vendor `5f2c6dec...996e`，dry-run、created、
  `--verify-existing` 与 `unzip -t` 全部一致。

### 平台门

- 基础门:8/8 correctness、每芯 `>=0.1x`；七个已过芯仍使用冻结 generic，唯一
  受影响芯为 Kunlun。最低有效投影仍约 `(34.5904+0.1)/8=4.3363x`。
- 机制门:Kunlun 必须返回真实 validation id，且不得再出现 1830s compile-worker
  crash/空 `failed_cases`。stop gate:任一数值失败、Kunlun `<0.1x` 或同族 1830s
  指纹即封存 E4，不做同包、参数或载体重投。

### 一次性平台提交(sub 8160,2026-09-02 11:27 CST)

- 实时 preflight 核对账号 `15600308080`、团队 `SoulCoder`、T38/
  `s2t1op038`、source/verification commit、stage `e4`、两成员、ZIP/test/release
  log SHA、窗口与 120s 最小间隔全部通过；额度 `26/30`，intent nonce
  `b5e730586de2d354922dd80a9018a06a` 为 `prepared`。
- 仅执行预检返回的一次性 submit 命令；submission `8160`、daily seq `5`，八芯
  入队。该 intent/ZIP 已为 `submitted`，不得重试。
- CLI 因未设置受信 remote host 将上传对象验签标为 `unavailable`；随后无认证、
  不跟随重定向地回读公开对象到 `/tmp/flagos-t38-e4-remote.JDyCGm`，精确得到
  12603 bytes 与 ZIP SHA-256
  `d3e8bd9baf9ebfb2d5cd76484bc5584266e7bec14b9bfe59184a6d3bfcbde5f4`，
  `unzip -t` 通过。当前只读等待逐芯终态。

### E4 平台终态(sub 8160,2026-09-02 11:29 CST)

- 终态 **7/8、invalid_correctness**。七个 generic 芯全部通过：天数
  `6.2856x`、沐曦 `2.9332x`、燧原 `0.8738x`、海光 `7.1736x`、华为
  `2.8448x`、国际 A `6.2540x`、国际 B `8.1724x`，合计 `34.5374x`。
- Kunlun validation id `aafbe7184104`，仅 **10634ms** 即完成编译和执行，彻底
  避开 E2/E3 的 1830s compile-worker crash；但 **9/9 cases 数值失败**，routed
  weights 约 80–94% 元素错误，多处出现 `1e28–1e38` 未初始化量级。
- 失败比例与 `k` 个槽仅一个被有效写入高度吻合；CUDA 上 E3/E4 indices 逐位一致，
  公式和 selector workspace 主路径已通过。当前最窄根因是 Kunlun 对
  `do_not_specialize=["rank"]` 动态标量参与输出地址的错误 lowering：连续 rank
  launch 可能反复写同一列，随后 normalize 从未写槽读取垃圾 index。跨 launch
  workspace 可见性是次要备选根因。
- stop gate 已执行：E4 intent/ZIP 永不重试。只有移除动态 rank 地址、让每次 launch
  直接接收已偏移的 `indices[:, rank]` view，并保持其余 E4 字节结构不变，才构成可评估
  的 E5 单变量；若无 MCP/静态/代理证据支持，或 E5 仍数值失败/超时/低于 `0.1x`，
  立即永久封存 T38 host-stepped 轴。
