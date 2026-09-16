# Task 69 `fused_moe_dispatch_index` 实验记录

```current
task: 69
operator: fused_moe_dispatch_index
batch: 5
validity: valid
platform: completed(e12-relaxed/sub16139,8/8,51.072975x；TB仍E10,51.3375x,排名5)
candidate_stage: e13-pair-ready
team_best_stage: e10-generic-init
team_best_commit: a01fb6344cfa9d9f92a88cd8d47d3d9db3d2ff1b
team_best_speedup: 51.3375
sealed: no
next: E13两路聚合已通过E10同分布代理筛选及11方法四源码release，ZIP就绪未提交；目标芯仍未验证
updated: 2026-09-16
```

## 契约与范围

- 完整题面：[Task 69](../tasks/batch-5/69-fused_moe_dispatch_index.md)（2026-09-11 新增）。
- 接口 `fused_moe_dispatch_index(topk_ids, num_local_experts, m_max)`；
  返回 `(masked_m, src2dst)`：atomic cursor 按 expert 分桶，
  `src2dst[i] = e*m_max + offset`；`masked_m[e]` 精确比较，
  `src2dst` 按桶内 multiset 比较（桶内顺序来自 atomics）。
- `-1` padding 跳过（reference 对应写 0）；int32；核心计算 Triton；八芯 0.1x。

## 实现（S0）

- 上游：SGLang 8014d9d `fused_moe_dispatch_index_triton_kernel`。本仓 S0
  简化为双 `torch.zeros` 初始化（masked_m 与 src2dst 的 padding 位语义
  都与 reference 对齐）+ 单 kernel masked `tl.atomic_add`；
  舍弃上游 single-block `ZERO_INIT + debug_barrier` 分支（弱芯 barrier
  语义风险 > 省 1 次 memset 的收益；reference 本身含 `.tolist()` CPU
  往返，多一次 memset 不改变量级）。
- grid-stride（cap 65535）× BLOCK=256。

## 不可变身份

- source / verification commit：`b4727f1`。
- source SHA-256：`de7fa1488ed40608b95ed23894942883f89a0ea472268b86fda59bac224e4d4d`。
- test SHA-256：`aac8b512245e0c3817b9a01e5379ba8577cd126950e1f3579f038ccc362712d3`。
- ZIP：`artifacts/competition/fused_moe_dispatch_index/s0-b4727f1/fused_moe_dispatch_index.zip`。
- ZIP SHA-256：`4c133b19523f74a3ac46ab19c26d7412566071790d03e0870f10cacaa20ed20a`。

## 验证状态

- py_compile、格式与 lint 通过（本地）。
- 测试：3 方法 / 常规与 25% padding、形状族（E=1/256、T*K 跨单块与
  grid-stride 边界 8193×8）、空输入；`masked_m` 精确、`src2dst` 全局
  排序 multiset（合法 dst 全局唯一，桶错位必被检出）、padding 位精确 0。
- **远端 GPU 不可达（2026-09-11 晚）**：release 回执待补，
  `target-runtime-unverified`。

## 风险

- **`tl.atomic_add` 跨芯证据缺口**：本仓已交付算子无 atomic 内核先例；
  FlagTree #1019 仅确认昆仑不支持 `tl.atomic_cas`，未涉及 atomic_add。
  昆仑 XMLIR / 燧原 GCU 的 masked atomic_add 需平台首轮反馈；若失败，
  备选是双 kernel 无原子方案（Triton 直方图计数 + 前缀和重写），代价是
  复杂度与两次启动。
- `expert_safe * m_max + offset` 为 int32：按题面约束
  （每专家计数 < m_max、E×m_max 受显存约束）不越界。

## 优化方向（按把握）

1. S0 直投，首轮重点读昆仑/燧原编译结果。
2. 若 atomic 芯失败：E1 无原子双 kernel（直方图 + 独占前缀和 + scatter），
   仅覆盖失败芯做 vendor 文件。

## 2026-09-12 平台提交（submission 13303，daily_seq 8）

- 回执（b4727f1，含 6 次 launch 的 atomic 路径）：
  `batch5-ext6-validate-20260912/fused_moe_dispatch_index/`
  （SHA-256 `c788579079acc54e32b8195bc66dd266d48ce0b8da557d23f4e7f8161a00828d`）。
- **燧原失败**：`RuntimeError: Pipeline run failed: PassManager execution
  failed`——masked `tl.atomic_add` 在 GCU300 无法合法化（编译层，与
  FlagTree#1019 确认的 atomic_cas 缺口同族）。
- **华为失败**：exec 582678ms 后 speedup 0.0，错误栈落在平台 reference
  文件 `flaggems_reference/fused_moe_dispatch_index.py:23` 与 torch_npu
  utils——atomic 形态拖垮 NPU 运行时后 reference 崩溃的嫌疑最大
  （c2flow 8/8 通过证明 reference 本身可在华为运行）。
- **昆仑失败（崩溃族）**：`服务线程卡死自动恢复，请重新提交`，
  exec 0ms——昆仑评测器平台侧故障，非内核裁决，按崩溃族协议不计代码
  止损、重掷需用户当次明示授权。
- 已过 5 芯读数（atomic 路径跑通处非常快，reference 的 CPU tolist 循环
  极慢）：海光 **119.737** / A 71.1100 / 天数 62.0734 / B 54.3222 /
  沐曦 36.1954。
- 结论：atomic 路线在燧原（编译）/华为（reference 侧崩溃）双双落定失败；
  下一候选为**无原子三段式 vendor**（Triton 直方图计数 → 独占前缀和 →
  确定性 scatter），覆盖燧原/华为（可试带昆仑，若跑通则一并绕开崩溃族
  重掷问题），generic 保留五芯已证路径。

## 2026-09-12 E1 平台结果（submission 13332，daily_seq 11）

- 无原子三段式 vendor（`_enflame/_ascend/_kunlunxin`，commit 4cc7092，
  ZIP `e1-4cc7092`，SHA-256
  `788cc34d60c011f4f0a49b22d3cd54f378beea52de35adf935bbbe5c1ad8394a`；
  回执 `batch5-e1-validate-20260912/fused_moe_dispatch_index/`，SHA-256
  `3d8a2ee6791c9be95d21acd46e8950585b44cb6f5d9d7d80a412a0bef7e63513`，
  generic 6 + 各 vendor 18 launch 全绿）。
- **燧原仍失败**：`PassManager execution failed`，vendor 被选中——无原子
  版仍有第二个 GCU300 编译毒点（候选嫌疑：kernel1/kernel3 的
  `[64,64]` 2D 比较/归约 lowering）。
- **华为首次真实执行**（exec 72306ms，vendor 选中）但 1/32 元素错，
  最大绝对差 1 / 相对 0.5 ⇒ rank 或 masked_m 的 off-by-one。代理复现
  失败：280 组小形状（含单块、E∈{1..100}、跨 tile 边界）× 4 源全部
  与 reference 一致 ⇒ **昇腾特有 lowering 缺陷**。
- 昆仑回调未返回。已判 5 芯通过。
- 下一版预注册：纯 1D 算子变体——kernel1 改标量专家循环
  （`for j in static_range(E_TILE): tl.sum(e == e0+j)`），kernel3 以
  1D `tl.cumsum`（昇腾已证可编译仅慢）替代 pairwise 名次；燧原同步
  受益于去 2D 形态。

## 2026-09-12 E2 候选就绪（纯 1D 原语收窄，待提交）

- 对 E1 两处失败的针对性收窄（预注册方向的修正版）：**不用 `tl.cumsum`**
  （retrospective 实证昆仑/燧原 cumsum 家族 lowering 毒点/瓶颈，共享
  vendor 不能带）；全部 2D 广播归约拆除：
  - kernel1：标量专家循环 `for j in tl.static_range(E_TILE)` + 1D
    compare/sum，标量线性 store——e1 的 `[64,64]` 2D 比较归约消除。
  - kernel3：逐 lane j 循环；1D 归约取本块内名次
    （`sum((e == e_j) & (idx < j))`）；独占基址用**标量**
    `load-value → .to(int64)` 寻址 gather（T64 deepep_permute 燧原 2.53x
    已证形态）；标量线性 store + `if e_j >= 0` 运行时分支（T64 同款）。
    e1 的 `[64,64]` pairwise 名次（华为 off-by-one 头号嫌疑）与
    向量 gather-by-loaded-index（燧原未证形态）一并消除。
  - kernel2 前缀和不变（1D 向量、线性访存，e1 五芯通过路径）。
- source commit：`2866bda3963ebf5b833ff6eb4b3f6b345d1062c1`（generic 与
  测试字节不变；三 vendor 逐字节相同）。
- ZIP：`artifacts/competition/fused_moe_dispatch_index/e2-2866bda/fused_moe_dispatch_index.zip`，
  18513 bytes，4 成员（generic `de7fa148…` + ascend/enflame/kunlunxin
  各 `df0f2fb2c256f96269ae99c92ba0518c3fe11bc4a8936bf349bda1eee4adf336`）。
- ZIP SHA-256：`8f1ef36c592fe1f4066e289e1eed70aab54fec35819fb436b76988176f24b685`。
- release 回执（v2，绑定 2866bda，proxy-vendor×3）：
  `artifacts/competition/batch5-e2-validate-20260912/fused_moe_dispatch_index/verification.json`，
  SHA-256 `6477a48e0ea2b146d34a071bba0d6193ac1ae92e0ece52198f51be959cf967f9`；
  完整日志 SHA-256 `209116819feb1d24134538dd3ac9d3e0f78505a62f465d03ab9c6de62ea3488f`。
  3 方法 0 失败/错误/skip/xfail；generic 6 + 每 vendor 18 次真实 launch；
  非空 shape 覆盖 [1,1]/[5,4]/[64,16]/[129,8]/[8193,8]。
- 昇腾/燧原/昆仑目标 runtime 仍 target-runtime-unverified（NVIDIA 代理
  仅证数学与 JIT）；裁决权在平台。新 ZIP 身份在全芯全新评测，昆仑崩溃族
  窗口自然重掷。

## 2026-09-12 E2 平台提交（submission 13359）

- 上传与正式 POST 各一次，无自动重试；state submitted，评测排队中。
- file_url SHA-256：`ab1996b9da873e72f9e02966f0f8a24eb0ff100ccb8d4d50a5f2b698370c14c0`。
- 观察时额度：17/30（发后）。等待八芯逐芯回调。

## 2026-09-12 E3：标量每专家前缀扫描（候选就绪后提交）

- E2 平台判决补充：燧原仍 PassManager（vendor 被选中，exec 8926ms）；
  五芯通过（天数 63.29 / 沐曦 **41.14（较 e1 的 36.20 +14%）** / 海光
  120.60 / A 71.54 / B 54.59）；华为/昆仑回调未返回时已入 e3 开发。
- 根因收敛：e1/e2 唯一共有结构 = kernel2 的**循环携带张量扫描**（手写
  串行 cumsum）——retrospective T14/T18/T21 实证该族在燧原/昆仑触发
  Pipeline 失败；e2 新加的整型 `tl.where` 地址钳位亦无燧原通过先例
  （clamp_position E1 证据）。
- E3 改动：①kernel2 改**每专家一 program 的纯标量扫描**（标量 load/
  store + 标量累加，无循环携带张量、无 masked 标量 load）；②kernel1
  去尾部分支——counts/prefix 补齐到 `E_PAD=ceil(E/64)*64` 列，内层
  static_range 无条件写全部列（padding 列恒 0）；③去掉 e2 的整型
  where 钳位（deepep_permute 已证裸 masked 向量 load 形态）。
  kernel3 保持 e2 形态（逐 lane 1D 名次 + T64 式标量 gather + 线性
  标量 store + 运行时分支）。
- source commit：`a7aa3d0097883214b8a94fb278f6ae1c7da50005`。
- ZIP：`artifacts/competition/fused_moe_dispatch_index/e3-a7aa3d0/fused_moe_dispatch_index.zip`，
  SHA-256 `1c5e434fac1e284ea769c52d44cbd63f10eef28dfd1db408a00e3d5c4075caf6`；
  4 成员：generic `de7fa148…` + 三 vendor `41a94adc…`（逐字节相同）。
- release 回执（v2，绑定 a7aa3d0，proxy-vendor×3）：
  `artifacts/competition/batch5-e3-validate-20260912/fused_moe_dispatch_index/verification.json`，
  SHA-256 `39de777a3c5f4aee19d1bd1ccb9a4c2d054e5af53d95d2016db6a19090196a3f`；
  日志 SHA-256 `0e31e0111cc047e96c7947b949a0961800042f7f1c074179fb4768f6de24dc59`。

## 2026-09-12 E3 平台提交（submission 13362，daily_seq 15）

- 上传与正式 POST 各一次；state submitted，13:3x 入队。
- file_url SHA-256：`2066e049…`（完整值见 status 快照）。额度：发后 15/30。
- 裁决点：燧原 PassManager 是否解除（标量扫描假设）；华为 off-by-one；
  昆仑窗口重掷。

## 2026-09-12 E3 回调（进行中）：燧原破局

- **燧原 PASS（0.6256x，> 0.1 门槛）**——三轮定位（e1 2D 形态 → e2 去 2D
  仍挂 → e3 标量化 kernel2）收敛到根因：**循环携带张量扫描（手写串行
  cumsum）= GCU300 PassManager 编译毒点**，与 retrospective T14/T18/T21
  的 cumsum 家族实证一致；标量累加器解除。该结论可迁移：任何燧原 vendor
  不得在运行时循环里携带张量累加器。
- 已判 6/8 全过：天数 61.7366 / 沐曦 41.0154 / **燧原 0.6256** /
  海光 120.6028 / A 71.5154 / B 54.9678；昆仑/华为回调中。

## 2026-09-12 E3 回调（续）：华为通过，仅剩昆仑

- **华为 PASS（3.6952x，exec 1568925ms）**——e3 `_ascend` vendor 首次
  真跑通过；e1 的 off-by-one 未复现（kernel3 换 j 循环 1D 名次后消失）。
  标量逐 lane 形态在 NPU 上偏慢（26 分钟执行）但 reference 更慢，远超
  0.1 门槛；性能轴留后续（向量化 rank / 更大 BLOCK）。
- 已判 7/8 全过：天数 61.7366 / 沐曦 41.0154 / 燧原 0.6256 / 海光
  120.6028 / 华为 3.6952 / A 71.5154 / B 54.9678（部分和 354.16）。
  仅昆仑 waiting_callback（vendor 已选中未执行，今日崩溃族窗口）。
- 昆仑过线即 8/8：K≥0.1 即 valid；均值 =(354.16+K)/8，追平 c2flow
  52.48 需 K>65.7（大概率 #2，仍是本题第二支有效队伍）。

## 2026-09-12 E3 平台终态：7/8（昆仑=崩溃族，未获裁决）

- 昆仑终态 `completed + passed=false + exec 0ms`，错误原文「服务线程
  卡死自动恢复，请重新提交」——vendor 被选中但从未执行，非内核裁决，
  按崩溃族协议不计代码止损。七芯读数已记（部分和 354.16）。
- 判决：e3 invalid_correctness（缺一芯不排名）。昆仑 vendor 的编译/数值
  至今零执行记录。
- 下一发 e4：BLOCK 64→32（kernel1/kernel3 j 循环长度减半，缓解华为
  26 分钟长跑的串行开销）——真实改动的全新评测自然重掷昆仑窗口。

## 2026-09-12 E4：BLOCK 32 重掷（候选就绪后提交）

- source commit：`fc765d8b74ada1448fd5d1ec0e0a95b13fbe0177`（三 vendor
  同字节；generic/测试不变）。
- ZIP：`artifacts/competition/fused_moe_dispatch_index/e4-fc765d8/fused_moe_dispatch_index.zip`，
  SHA-256 `1443b56b87d94022b87a99f98c5d1ec96cd6e985be6f96d4b0cfbdc0346016da`；
  4 成员（generic `de7fa148…` + 三 vendor `f5a7eb67…`）。
- release 回执（v2，绑定 fc765d8，proxy-vendor×3）：
  `batch5-t69e4-validate-20260912/fused_moe_dispatch_index/verification.json`，
  SHA-256 `d5ffd449f830594251fea14bf177db75a26810b612ad264e5ec68e680ba88dcc`；
  日志 SHA-256 `17931bb31304498d748d2ab30a233b8e30de8b3a1c6852f7ef477c4f866240c7`；
  3 方法 0 失败，generic 6 + 每 vendor 18 launch。
- 预期：七芯读数与 e3 同级（华为 kernel3 串行减半或提速）；昆仑窗口
  重掷，K≥0.1 即 8/8 valid（部分和基准 354.16）。

## 2026-09-12 E4 平台提交（submission 13381）

- 上传与正式 POST 各一次；state submitted。额度：发后 8/30。
- 裁决点：昆仑是否获得真实执行（K≥0.1 即 8/8 valid）；华为 kernel3
  提速幅度。

## 2026-09-12 E4 回调（进行中）：七芯全过，燧原 +16%

- 已判 7/8 全过：天数 62.9682 / 沐曦 41.3952 / **燧原 0.7242（e3 0.6256
  → +16%，BLOCK=32 兑现）** / 海光 111.7322 / 华为 3.6156（exec 26min →
  12.5min，j 循环减半生效）/ A 71.0090 / B 54.3634。部分和 356.25。
- 仅昆仑 waiting_callback（本日第三次撞该芯崩溃族窗口）。

## 2026-09-12 E4 平台终态：7/8（昆仑崩溃族第四次）

- 昆仑终态 exec 0ms 服务线程卡死（13362/13375/13381 三连 + T70 同窗）；
  同期 T67 两发昆仑正常判决——按提交闪断而非全天停摆。七芯读数已记
  （部分和 356.25，燧原 0.7242 +16% 为 e3/e4 结构性增益）。
- 后续：明日健康窗口新 ZIP 再掷（下一真实改动候选=kernel3 rank 向量化
  或 kernel2 每专家多 block 并行）；无变化不再同字节重掷。

## 2026-09-12 E5：runtime range 替换 static_range 展开（候选就绪后提交）

- 假设：燧原/华为 exec（19-46s / 12.5min）远超数据量数个量级 ⇒
  **JIT 展开体积是读数瓶颈**（kernel1 的 64 宽 static_range 嵌运行时
  tile 循环 + kernel3 的 32 lane 展开产生巨大 IR）。
- E5：两处 static_range 改 runtime range，语义逐字节等价；全新 ZIP
  身份第五次重掷昆仑窗口。
- source commit：`1aff023f2d320a2062dedc289a2d7fcfbd450b7f`。
- ZIP：`e5-1aff023`，SHA-256 `772f9f75c3a7da96c6ba290a9951c603d5c03f17142506f7d198a862bae39905`；
  4 成员（generic 不变，三 vendor `7abe15c4…` 相同）。
- release 回执：`batch5-t69e5-validate-20260912/fused_moe_dispatch_index/verification.json`，
  SHA-256 `5108e5a4a6181e9b845f79ecf1c82aa51cbb758092c555f3f7e768aabb312dd4`；
  日志 `c8a1f2cfc60bb041d5663263203a10d5eec685feb66ed67ffbde8d7d9df040ac`。
- 预注册：燧原/华为 exec 显著缩短（≥2x）且读数 ≥e4 水位；昆仑过线即
  8/8（部分和基准 356.25）。

## 2026-09-12 E5 平台提交（submission 13406）

- 上传与正式 POST 各一次；state submitted。额度：发后 4/30（留收盘余量）。

## 2026-09-12 E5 平台终态：7/8（昆仑崩溃族第五连崩）

- 昆仑 exec 0ms 服务线程卡死——本题连续第五次，同期其他题昆仑正常
  判决；工单路径（健康 worker rerun，不耗额度）升级为首选。
- 七芯读数与 e4 持平：天数 63.37 / 沐曦 41.81 / 燧原 0.6772 / 海光
  107.96 / 华为 3.7880（exec 12.5→9.5min，JIT 瘦身部分有效但非读数
  主因，假设降级）/ A 66.06 / B 53.83。部分和 357.48。

## 2026-09-13 E6：昆仑 kwargs 重掷（候选就绪后提交）

- 五连崩后首个有据动作：`_kunlunxin` 副本加
  `isCloseOffsetAnalysis=True, isCloseUnrollControl=True` launch kwargs
  （FlagGems fused_moe 生产同款；FlagTree #1147/#1053 的误判 pass 恰在
  此二 pass）。燧原/华为 vendor 字节与 e5 一致；kwargs 为 XPU 专属，
  代理不可执行——release 以 enflame/ascend 代理 + 昆仑 target-only 记录。
- source commit：`3bd378afbe9e90fcae6843534c52ab798de16421`；ZIP `e6-3bd378a`，
  SHA-256 `a53fc93bd6e9946e5d97e5fd70dc4030c95cfc92ccea7b39537f21b3b9319be1`。
- release 回执：`batch5-t69e6-validate-20260913/fused_moe_dispatch_index/verification.json`，
  SHA-256 `9063a912ade241842f46c956e7bafa55f1235835974b835f29b0f24357f07d4d`；
  3 方法 0 失败（generic 6 + enflame 18 + ascend 18 launch）。
- submission 13769（09-13 08:0x）；裁决点=kwargs 是否解除昆仑崩溃。

## 2026-09-13 E6 平台终态：7/8（kwargs 未救昆仑，第六连崩）

- 昆仑终态 exec 0ms 服务线程卡死（此前 ~1h 的"执行中"为假象）。
  kwargs 假设证伪。**六连崩后 kernel 侧路径全部试尽**：仅剩平台
  工单（健康 worker rerun）。七芯部分和 357.5 保持待命。

## 2026-09-13 E7：num_warps=1/num_stages=1 重掷（已发射）

- 六连崩为间歇性（同窗 T62/T74 昆仑正常判）→ E7 在 kwargs 之上加钉
  FlagGems 昆仑生产 launch 形态，全新 ZIP 身份重掷。
- source commit：`267a1abfc4dcdbdc91c989680620e1de5d19a6d4`；ZIP `e7-267a1ab`，
  SHA-256 `1e0428efe161faacb30d28e76a3d62343ef0ca5709548a193a4067c8e5dfbc82`；
  release 回执前缀 `685f0412`；3 方法 0 失败。

## 2026-09-13 E7 平台终态：7/8（昆仑第七连崩）

- exec 0ms——kwargs+num_warps/stages 双钉无效。七连崩后 T69 昆仑轴
  终封,仅剩工单。

## 2026-09-15 17:40 E8 重开发射：昆仑标量重写（T65 解锁先例解除终封）

- **重开依据（新源码级结构证据）**：T65 e6 当日证实"masked 向量载入 +
  比较掩码 + tl.sum 归约提取标量"在本后端触发 make_llir SIGABRT；
  T69 昆仑 vendor 的 `_dispatch_counts`（hits）与 `_dispatch_ranks`
  （rank）正是该形态。原七连崩终封是在误判为窗口问题的前提下做出。
- commit `1244021d`：kernel1/3 全标量重写（标量 load、runtime 分支、
  标量 load-modify-store/分支计数；kernel2 已是纯标量不动）；
  counts 改 `torch.zeros` 初始化（标量版只写命中格）；去掉
  isCloseOffsetAnalysis/isCloseUnrollControl kwargs 恢复默认 pass。
- ZIP `e8-1244021`，SHA-256
  `5f891bd571af61b3e4b345b388257bbe4c084ab7c7963758d80eb38f2c26d5d4`；
  release 回执 `artifacts/competition/batch5-verify-20260915/t69e8/`
  （exit 0，generic+昆仑 vendor 真实 launch，0 skip）。
- 已提交（submission 待回填）。门：**昆仑产生有效判决且 ≥0.1**；
  七芯部分和 357.5 待命，若昆仑 ~0.1-0.5 量级均值约 44+。
- 若仍崩：按停损规则不再盲试，需要 assert 文本（T65 式完整堆栈）。

## 2026-09-15 18:10 E8→E9 平台终态：8/8 VALID 43.6439x —— 昆仑解锁（今日第二题）

- **E8（1244021d）**：标量重写后昆仑**首次编译并真实执行**（8 连崩
  破局），但数值失配——`masked_m` 7/32 差 ±1。根因：kernel1 的分支内
  全局 load-modify-store（`store(load()+1)`）在昆仑后端错误执行/丢增量
  （NVIDIA 语义正确，我方 release 全过）。
- **E9（e079237f）**：kernel1 改每 expert 一 program 的寄存器累加 +
  每 (block, expert) 单次 store（kernel2 同构、T65 e6 已证形态），
  全 kernel 无内存 RMW。
- **终态 8/8**：天数 61.1946 / 沐曦 40.7292 / 燧原 0.7558 /
  海光 121.343 / **昆仑 0.1016（贴门过线）** / 华为 3.828 /
  A 67.548 / B 53.6512；均值 **43.6439**，首次有效上榜。
- 经验沉淀（昆仑后端新事实）：①向量+tl.sum 提取 → make_llir SIGABRT
  （T65）；②分支内全局 RMW → 丢增量（本例）；③安全形态=标量读 +
  分支计数 + 寄存器累加 + 单次 store。

## 2026-09-16 E10 generic 初始化融合：8/8有效，51.3375x新TB、排名5

- 16:33:53 只读榜单确认 E9 仍为有效 TB：**43.643925，第 6**；榜首
  EvokeAgent 80.917125。天数、海光分别贡献均分差 18.612125、7.37465。
  快照 `artifacts/competition/contract-fixes-20260916/platform-t69.json`
  的路径与哈希已绑定筛选计划；这些分数不能反推隐藏 shape 或设备耗时。
- 最小变量：generic 的 `src2dst=torch.zeros` 改 `empty`，现有 atomic 核
  对全部 inbounds slot 写 `where(valid,dst,0)`。计数清零、atomic mask、
  grid-stride 和三份 vendor 字节不变，没有增加单 CTA 初始化或跨程序同步。
  基线来自 E9 `e079237f16094894b0c2b3ba013f73e92d09aae5`，generic 的
  Git 源、ZIP 成员与已执行 release 哈希三方一致。
- 筛选目录：`artifacts/competition/t69-generic-init-screening-20260916/`。
  baseline 原 3 方法、candidate 6 方法全部通过，0 失败/错误/skip；保留原测试，
  新增 per-expert ownership、padding、非连续输入、强制 grid=1/2、poison 输出
  与前后哨兵。`correctness.json` SHA-256：
  `965c641c9f5c7b8c094e552f7e237b8eddc951b94e0b3976d3de206797ba47b4`。
- 两个 probe 的真实 CUDA profiler 事件均确认 **Fill 核 2→1，总 GPU 核 3→2**；
  双方仍各执行一个 Triton 路由核。根任务已核对 TTGIR/PTX：输出 store 覆盖 padding，
  atomic mask 不变，无新增 barrier；寄存器 34→31，0 spill，shared 均为 2048 B。
  `probe.json` SHA `b7bf9ec3fcfcbb5ee5009841091d0dc0482f776ed1b963503a36f073aa137625`；
  `ir-decision.json` SHA `dda6601b31ad163262b4b229dadfd3d07d8b16bf0bdce6143c7a8d40224fb383`。
- **84 主桶＋4 空输入 controls，5 轮 AB/BA，共 440 对原始样本**完整结束。
  主桶中位数比值算术平均 **1.0974858566**、GM **1.0961482729**；逐轮算术均值
  1.0835832910–1.1122761541，最差主桶 1.0203289748，无回退、无 control 漂移，
  所有资源记录 0 spill，达到预注册代理筛选门。`benchmark.json` SHA：
  `f488a1841bd376a6234d82dce8e950805dbea5aae329168b5c0b84ae00be94d8`；
  `raw-samples.csv` SHA：`b8caf6e8754909198fda009e5fd586d5f2545a3a3aaab251ae38ff786f22810d`。
  这是 RTX 5070 Ti / Torch 2.13.0+cu130 / Triton 3.7.1 的包装器整体证据，
  **不代表天数、沐曦、海光、A、B 五条 generic 目标芯路径的收益或平台均分**。
- 晋级 source commit：`a01fb6344cfa9d9f92a88cd8d47d3d9db3d2ff1b`。
  generic 与筛选候选逐字节一致，SHA：
  `b53af3d40707430d846f76ae37e27df3215c71776a8c6542f71dd0888b5845eb`；
  正式测试改接 `load_operator_modules`，SHA：
  `f9d35cffc42a2533a9ecaa6882c6b22a2c2d64f2ead0168b60f67f64a4f5fc7b`，
  不把旧筛选测试回执冒充正式四源回归。ZIP 已构建为
  `artifacts/competition/fused_moe_dispatch_index/e10-generic-init-a01fb63/fused_moe_dispatch_index.zip`，
  SHA **`80c2cbdfea2e02bd47bd72dad815762af1a635c8e51adae0268ff13732bcbdc8`**；
  四成员为 generic、ascend、enflame、kunlunxin，后三者保持 E9 字节。
- 正式 release：source / verification 均为上述 `a01fb634`，**6/6、0失败/错误/skip**；
  generic / Ascend / Enflame / Kunlun 各78次入口，包装器内实际kernel分别76/228/228/228。
  回执 `artifacts/competition/t69-generic-init-release-20260916/verification.json`，SHA
  `9055f7fa366e8899cd327db20bbc660e0cd8a6ce903cc7f1cc17d2eef1ba4896`；日志 SHA
  `88fdc2e83ba76b8fbbc5d0660857f90cf91f4874da6861ee3fc048855b58446e`。
  已逐项对照 Git source/test/依赖与实际日志；裸核 poison 回归直接执行，未计入上述包装器 launch 数。
  远端 `/tmp/flagos-t69-init-release.onj1HZ`、PID393468，外层930秒/内层900秒，EXIT0；
  NVIDIA代理不替代目标芯验证。前后无其他compute进程。
- 16:50:09 **单次上传和提交成功，submission 16056、daily_seq=10、state=submitted**。
  preflight现场额度21/30，提交后20/30。nonce绑定
  `48db81a7ec2d7c79a14ec9724a6b8466`，file URL SHA
  `82948c01336d69290d37141b7b5edf6dea347203201516935c39a77c580a41e0`。
  提交工具的附加远端验签最初缺少可信host配置；随后仅对已返回的官方对象存储URL做只读GET，
  17793字节与ZIP SHA完整一致（`remote-zip-verification.json`）；没有重传或重提。
- **16:52:11 终态：8/8、valid、is_team_best=true，均分51.3375**，比E9
  43.643925增加 **17.6281%**；实时榜单排名6→5，榜首仍80.917125。
  天数76.9982、沐曦45.7358、燧原0.7568、海光134.7284、昆仑0.1016、
  华为3.8250、A80.7688、B67.7854。五条generic路径均上涨；三份vendor字节
  未变，回调保持有效。主核融合输出初始化的收益已在本次平台评测兑现。
  `platform-final.json` SHA
  `7d4bc3f6bc595015bc15829308c51d1b1b6a9ba00271922d54297125190f6665`；
  `leaderboard-final.json` SHA
  `1939fe66de6e1b3a8a416567a1ab040938c51cab2bf81efbeb684d8d5b69e9cd`。
  两文件均位于本轮release目录；新TB为E10，下一候选必须据此重新选择基线。
- 提交后独立审查发现旧vendor的expert grid截断：题面未限制E上界，
  E=65536、ids=[[65535]]时旧每expert单program路径没有expert grid-stride。
  该缺口不在已执行回归覆盖中，不能用本次八芯评测通过推断该未覆盖域正确；正在另立修复与旧版失败回归。
  原E10候选状态只读跟踪，不以此重试同一次提交。该缺口随后由下述E11修复。

## 2026-09-16 local-bucket 独立筛选：仅昆仑基线达到机制门

- 独立目录 `artifacts/competition/t69-local-bucket-screening-20260916/` 的
  6 桶×5 轮 NVIDIA 代理结果：相对 E9 Kunlun vendor 的 GM **4.1885537003**，
  相对 Enflame vendor 的 GM **1.4544346206**。按事先登记的 GM≥2 机制门，
  仅前者支持继续 IR/目标芯探查；不能称为昆仑或燧原硬件实测收益，也不外推平台分数。
- `benchmark-kunlunxin.json` SHA：
  `55dea1dc2e8a77a95d3c428d1b3dcc19b443cb577a4b8100138c2cde750cbace`；
  `benchmark-enflame.json` SHA：
  `aec9cee97b8cfe6c4e4b99ef609908cd13f65d1ecb4ad4ee777668e17ee3d748`。
  完整 IR 审查已完成，目标验证待定；**未修改正式 vendor、未做该候选 release、未提交平台**，
  与 E10 generic 初始化候选分别记录。

- 补充完整 IR 已完成：5 个输入、15 次实际候选 launch（含 E257），60 份 asm
  全部验签，0 spill。histogram 实际为 **shared atomic，无 global atomic**；
  i64 sort 为比较交换网络；maxscan 为局部扫描，prefix 只有 scalar i32 累加。
  `full-ir-review.json` SHA
  `9748af4976eb0159833cb6bc13228fadd7bbe6eafbae32c38c5fc0ccb0d41ce1`。
  T70 历史只证明昆仑缺 `tl.topk`，不能据此认定 `tl.sort` 不支持；
  共享原子同步、i64 sort、scan/gather 组合及大 E 资源在目标芯仍未知，保持候选隔离。

## 2026-09-16 E11：修复 vendor 专家 grid 截断，正式回归完成

- 根因：Ascend/Enflame 的 prefix、Kunlun 的 counts/prefix 只覆盖
  `min(E,65535)` 个专家。E=65536、`ids=[[65535]]` 时末专家计数漏写；
  这是公开契约缺口，不能由 E10 八芯有效推断正确。
- 改动只有三份 vendor 和回归测试；新增 constexpr `EXPERT_TILES=ceil(E/grid)`，
  以 `pid + tile*grid` 唯一覆盖全部专家，末 tile 防越界。普通 0<E≤65535 的
  tiles=1；E=0 的 grid=1、tiles=0，不解引用空 counts/prefix。generic 保留 E10 字节。
  source / verification commit：`521b0656ff113d2f25cee4130c29fcf81baea6d5`。
- 新增高专家编号、E=0/空输入/全 padding、强制 grid=1/2 的 poisoned counts/prefix；
  旧 6 方法 AST 全部保留，现 9 项 REQUIRED。旧源 `a01fb634` 配新公开入口
  `test_expert_grid_boundaries` 实测 **9 个 counts 失败子例**，没有用私有签名错误冒充 RED。
  RED 日志 `artifacts/competition/t69-expert-grid-20260916/red.log` SHA `04a54de874ac7c582895f1ac63c74ac960e1ccb5999ff34ff7c7edfafbc43686`。
- 独立静态审查无新增阻断，`expert-grid-review.json` SHA
  `449fd1e0f05448120d1643bf22fae48ce0667cb01fb374743c59d8ec0adbe6de`。
- 远端 `/tmp/flagos-t69-expert-grid.7cHdYa`、PID393640，输入 28 文件启动前验签；
  RED/IR/release 顺序后台执行，总 timeout1540 秒，结束 **EXIT0**，前后 GPU 无其他计算进程。
  release **9/9，0失败/错误/skip**，四源各94次入口；包装器内实际核
  generic84 / Ascend252 / Enflame252 / Kunlun252；裸核 poison 回归另行真实执行。
  回执 `artifacts/competition/t69-expert-grid-20260916/release/verification.json` SHA
  `b0887c6e08a59a2e914e718d0f4bbcea98593c9b8d6e14e994a75f8807fcfa64`；完整日志 SHA
  `08ecfe72e6784244704436ee01e293d896beb724a0ed2234bcc9eb8b3906e5ee`。已复核 Git blob、测试、依赖、回执和日志。
- 普通域 IR：E32/E64 × n1032/n8192 × Enflame/Kunlun，共24对核、48次实际launch；
  Ascend与Enflame源字节相同。20对去调试元数据后 TTGIR/PTX 完全一致；
  其余4对仅昆仑 counts 新增未使用形参、参数编号平移及常量声明顺序变化，
  无新增 GPU 循环、分支或读写，寄存器/0spill/shared全保持。新外层 tile loop
  在普通域消除；这不是目标芯性能保证，昆仑旧0.1016贴门风险仍需平台裁决。
  `normal-ir/comparison.json` SHA `ab3db075e49449e23ce8185a212ec797b49c576aaa8037df26e492fc48930e6d`。
- 正式 ZIP：`artifacts/competition/fused_moe_dispatch_index/e11-expert-grid-521b065/fused_moe_dispatch_index.zip`，
  18935字节，SHA `f58e1dfd107fb57696dea225130f87bd725ec6c1ec8724ab834d99d79464d5be`，
  四成员 generic/ascend/enflame/kunlunxin。测试 SHA
  `e37b9554930e30ac8e34bf3f3d8dfc5e820109603a0de18aa3b2d7dd7fd9c21c`。
  NVIDIA代理通过；目标runtime尚未验证，按既有授权执行实时preflight及单次提交。

- 第二位审查者独立核对28冻结文件、8份Git绑定源、192份原始/归一化IR及48份diff；
  完整代码对应后结论相同，`normal-ir-review.json` SHA
  `f652658345c0261ba04d75ebd5fa9926f0b79e8d542d9765dfb1e507bcfd248d`。
  NVIDIA四个普通域shape未观察到新增GPU工作；新增参数的主机封装成本未单独计时。
- **17:08:25 单次上传及正式提交成功**：submission16063、daily_seq11，
  nonce `cfa195eb2ab3db145165fd4c8032a6c0`；现场额度20/30→19/30。
  返回对象存储ZIP已只读下载验签，18935字节与上述SHA完全一致；状态submitted，未重试。
  file URL SHA `d6f838d0b6de80d380ba23a6f5f7978f44d21db43bef3d14531f0c996e7b26ad`。
- **17:09:52 终态8/8、valid，均分49.699075，is_team_best=false**。
  天数74.2028、沐曦40.4208、燧原0.7950、海光134.5978、昆仑0.1036、
  华为3.7254、A77.3418、B66.4054。三份改动vendor均通过；均值比E10低，
  不记为性能提升，不重投同候选。generic字节未变但平台读数有变化，
  此次两次平台分数不能用于量化大E修复的因果性能影响。
  大E公开回归在NVIDIA执行通过；平台没有暴露shape，八芯有效不能证明其隐藏测试覆盖大E。
  `platform-final.json` SHA `5c9e1b05d58d9cc5b73f00da022d218f60d0cc96e54d0c2db14486ace30b1a95`；
  `leaderboard-final.json` SHA `b7145d1db919315f7797e11673de506312421b3ff460e7d30f0c8c77c7263bdc`。
  这两份文件均在 `artifacts/competition/t69-expert-grid-20260916/`。
- 2026-09-16T17:10:46.144535+08:00 实时榜单仍为 **E10 51.3375、第5**；榜首80.917125，
  差29.579625。天数贡献16.636675、海光5.701475，二者合占均分差75.5187%；
  后续突破优先研究这两条generic路径，不能把昆仑代理4.19倍直接当冲Top1依据。
  平台保留历史E10为TB；**后续实现从已修复的E11源码出发**，不为保留旧TB撤回正确性修复。

历史解释订正：上文旧实验中的`exec 0ms`不能单独证明平台故障或kernel未执行，
`execution_time_ms`也不能当作纯kernel/JIT耗时；旧的循环形式泛化只作当时假设保留，
不得覆盖后续已验证的标量循环事实。

## 2026-09-16 E12：generic 原子票号改 relaxed，代理筛选与正式回归通过

- 本轮重新读取T69实时逐芯榜，仍为E10 **51.3375、第5**，榜首80.917125；
  剩余额度19/30。优先generic天数/海光，不撤回E11三份vendor大E修复。
- 新访问[Triton atomic_add文档](https://triton-lang.org/main/python-api/generated/triton.language.atomic_add.html)
  确认默认acq_rel，gpu为默认作用域。当前counter仅分配唯一票号，不用于发布其他内存；
  输出每个输入slot独占写，外部消费在kernel完成后。因此只把generic原子改为
  `sem="relaxed"`，scope、地址、mask、常量加1、BLOCK256和wrapper全部保留。
  独立依赖审查未发现既往同轴候选；NVIDIA旧PTX确实保留acq_rel，非源码空改动。
- 以E11正确代码（ledger HEAD559df772）作基线，generic源SHA
  `b53af3d40707430d846f76ae37e27df3215c71776a8c6542f71dd0888b5845eb`。
  筛选目录 `artifacts/competition/t69-atomic-aggregation-20260916/relaxed/`。
  baseline/candidate均10方法通过（generic-only）；其中vendor内部测试在此未选择vendor，
  不将其空循环计为vendor验证。新补 `m_max=0/1` 专家区间重叠和可表示int32末端，
  原9方法完整保留，不假定不同专家的dst全局唯一。
- 两个实际probe的TTGIR/PTX只有acq_rel→relaxed；仍1个Fill+1个Triton kernel，
  保留8个既有layout barrier。寄存器31→32，shared2048B，0spill。
  probe之后、计时之前root独立IR决定已写入 `ir-decision.json`。
- **84主桶+4空输入control，5轮AB/BA，440对原始样本**：算术均值1.0702407957，
  GM1.0688247595；五轮均值1.071486/1.070535/1.073349/1.066980/1.069718；
  最差桶0.991818，未出现<0.95回退桶、control漂移或spill，通过原数字筛选门
  （均值≥1.03、每轮≥1.01）。uniform/hot/pad分组均值1.07960/1.04582/1.08530。
  这是RTX5070Ti/Triton3.7.1/Torch2.13.0+cu130代理的wrapper整体证据，不推算目标芯成绩。
  benchmark SHA `c921d134c8c771ad76d1da68e15908730bba46bb3569c4f031b3bec606ffc247`；raw-samples SHA
  `f73b73b0ef9e6b4dea26473cb3e99ae5489db910aba89a248e360e4cf59b660c`。筛选远端 `/tmp/flagos-t69-relaxed.POIhkW`，
  precheck PID394268/390s、benchmark PID394368/930s，均EXIT0，前后无其他GPU计算进程。
- **计划文本勘误**：复用的冻结plan仍残留旧初始化实验的“3/6方法、zeros2→1/总核3→2、
  TB43.643925”说明；这些说明不成立，原文件保留。methods数组、源绑定和执行断言实际为
  10/10、双方1Fill+1Triton；计时前IR决定也如实记录。数字性能门未改变，
  正式新TB只能与当前51.3375比较，不使用旧值。独立审查与另存勘误保留该问题。
- 正式source/verification commit `0af3c9708a1eef1592ac45773de64c1aa65e9569`，generic逐字节同筛选候选，SHA
  `92132171969ca265a8c8e01d09e5128e72b5bc4e8be7a62ed96189810b78f6af`。
  全部四源 **10/10，0失败/错误/skip**；各100次入口，包装器kernel
  generic90 / 三vendor各270；裸核poison单独实际执行。
  回执 `artifacts/competition/t69-atomic-aggregation-20260916/relaxed-release/verification.json` SHA
  `3bca3f761ff620d657322c15d27b0fe0cbcb5d4ca5d7f8abfc4833a1a6dcd23e`；完整日志SHA
  `379ef4d68445276b5251cf415577d204bbc6e93e6106c1c26f4a5c0b1e0975fa`。Git源码、测试/依赖和日志均已验签。
  `/tmp/flagos-t69-relaxed-release.ZHXuPc`、PID394426，先release后sorted预检，
  外层1340s、release900s，EXIT0。
- 不可变ZIP `/Users/bytedance/ccc/flagos/artifacts/competition/fused_moe_dispatch_index/e12-relaxed-0af3c97/fused_moe_dispatch_index.zip`，18972字节，SHA
  `7525cfebcdd0311d318b553054e1af82d6fb0e85fdd4390fdc30a248f34d1593`，四成员generic/ascend/enflame/kunlunxin，
  dry-run与实际构建一致。目标runtime未验证，按授权执行实时preflight后单次提交。

- 独立筛选复核 `relaxed/screen-review.json` SHA
  `e3b2b9ecb6d49904ac7cbf12bdbb0708befd99515f8e32cf1387e70fffc25815`；
  `relaxed/plan-errata.json` SHA
  `04690826c9d905acc6cd21f2b7910a9b9fb8683413e20cd9cead7fabb3240bc7`。
  回执允许进入exact-commit四源release；上面的正式release已实际完成并验签。

- **19:37:26 单次上传/提交成功，submission16139、daily_seq12**；nonce
  `5e4c5e2791b3899fdee3ab28071eda39`，额度19/30→18/30。服务器ZIP下载验签一致，
  18972字节与已绑定SHA匹配，没有重传/重提。file URL SHA
  `e2103404094a7a44e30eeda5cc2afc45ba49942d82f91a0ed60b27afdf25bef5`。
- **19:39:06 平台8/8、valid，均分51.072975，is_team_best=false**；
  天数77.6478/沐曦41.2886/燧原0.7408/海光135.823/昆仑0.1032/
  华为3.6496/A82.2514/B67.0794。榜单仍E10 51.3375、第5。
  代理+7.02%不能直接换算平台收益；本次未创新TB，不重投此候选。
  `relaxed-release/platform-final.json` SHA `af1ac4979b6274a1bb6df3338e1716e8e11292417ac2f9cb2fe1c2be762965a9`；
  `relaxed-release/leaderboard-final.json` SHA `2a61964fddccb68cca19c01e149923b2b9c6a1c865850d8e59feff527fd4170d`。

## 2026-09-16 单核排序聚合：数字短名单门通过，分布性回退后暂不晋级

- 与E12独立，从E11默认acq_rel generic出发；i64打包expert+原lane排序，前向max/
  反向min两scan求段边界，仅段首atomic预约整个段，再gather起点并scatter回原slot。
  保留单路由核/计数zeros、BLOCK256与grid-stride，未引入global scratch或vendor改动。
  候选 `sorted/candidate.py` SHA `1629a375ede139af6d4331c3421e6908ae32d9080636c818fea151a6380fc8d1`。
  CPU模型不替代GPU；实际generic双方10/10通过，含forced grid/poison、高E、空和padding、
  小m_max重叠、int32末端与stride，vendor内部方法不算该筛选覆盖。
- 新取回16份IR与profiler验签；段首mask、segment_count、masked返回清零后gather、
  原slot store都保持。外层只有scalar induction，无tensor iter_args。
  PTX barrier **8→33**、shuffle0→144、register31→96，shared2048B、0spill。
  按冻结输入推导（非硬件计数）small-pad请求171→18；large-uniform E256每tile全不同，
  请求65544→65544，付出排序却不减少atomic。
  `sorted/ir-review.json` SHA `537a3d0ae9fa62c1b6f8b3c8095ad18f415f7c1fb78777ca1d639771f84c4e88`。
- 复用plan的陈旧说明另存 `sorted/plan-supplement.json`，在**计时之前**明确实际
  10/10、1Fill+1Triton、当前TB51.3375及资源可行性上限；数值短名单门与88桶不变，
  不覆盖原冻结文件。资源上限在已看IR后、计时前声明，不伪称probe前注册。
  supplement SHA `44753413ae3cd7d780ea5647fd798ab945bfc82d70a528d60a09190063405f43`。
- 84主桶+4control、5轮440对完成：算术均值 **1.1023310446**，GM **0.9935606129**；
  五轮均值1.100563–1.103773，确实达到数字短名单门。但**64/84桶速度比低于0.95**（69/84低于1），
  最差0.780519；uniform/hot/pad组均值0.99788/1.37162/0.93749。
  热点大N最好3.8855倍拉高均值；无control漂移、无spill，不能删去回退桶重新算收益。
  缺乏隐藏分布与目标收益证据，暂不将此无条件排序替换泛化到generic；不是因单桶回退自动否决，
  也不是声称数字门失败。没有该方案release、ZIP或平台提交。
  benchmark SHA `09d58d2b43d7e337e1cb4d65a71fbb55748d445cfcfefef3a4ba80c733c97b35`；raw SHA `7b94582554b9549d9de1e7219ea88d36ac352fc875d0416a0e1e189d2ce60a6f`。
  决策 SHA `34f7feb15cde203f1e793d5132294e4d9036a2d55659b2259e489d05ac05b61c`。远端 `/tmp/flagos-t69-relaxed-release.ZHXuPc/sorted`，
  PID394579、930s上限、EXIT0，前后无其他GPU计算进程。

## 2026-09-16 单warp布局：同步点减半，实际性能未过门

- 从已四源release/平台有效的E12 `0af3c9708a1eef1592ac45773de64c1aa65e9569`出发，
  唯一改动为generic wrapper `num_warps=1`；kernel/BLOCK256/relaxed原子/三vendor不变。
  历史generic未测试该轴，过去warps1只涉及Kunlun vendor。
  `warp1/candidate.py` SHA `fe52497b1fe5d046e6b877d089243b538ab0878998c60d78ad0eb198bb566b52`；
  `warp1/plan.json` SHA `c6bf39be300be5bf25c1963b28b3303b3a923ae4cc91301918411cdc6ef77c3d`，本次重新构建完整计划，无旧初始化说明混入。
- generic双方10/10，原方法保留且poison/grid1/2明确分别跑warp1与4；该实验未晋级，
  扩展测试保留在screening产物。16份实际IR/profiler独立验签，reqntid128→32，
  四次layout转换仍在、barrier8→4、shuffle仍0；register32→62、shared2048B、0spill。
  静态atomic2→8是每线程处理量变化，128×2=32×8，未减少动态原子请求。
  candidate部分IR调试路径为baseline.py：两者kernel本体完全相同，forced baseline warp1
  先编译触发编译缓存复用；实际public caller与kernel block32共同核实候选配置，不是结果缓存。
  IR审查 SHA `2b806b879e07e0325bd6ef7db95052da250cfbdadb3d3255e6a719380d4290bd`。
- 84主桶+4control、5轮440对全部完成。算术均值 **0.9668489988**、GM0.9664182139，
  五轮0.965222–0.968846；最差0.908831，26/84低于0.95；control无漂移、全部0spill。
  uniform/hot/pad分组均值0.96967/0.96576/0.96512，**数字晋级门失败**。
  不把barrier减少等同速度提升，也不单独归因到寄存器或线程数；未做该候选release/ZIP/提交。
  benchmark SHA `102cf4a4f564676a192089d54d2e765395d49de1ef765d478c70f1dc088d76af`；raw SHA `e61d119124d700cba5500f4d5d7eae125bb05c2c56f5ca1919d06e47f9dac6d4`；
  决策 SHA `66ffc09b392ec59c87158cc72d848e16b0aa97b383edeff14baf52208536d830`。
  `/tmp/flagos-t69-warp1.dvpSMz`：precheck PID394642/390s，benchmark PID394737/930s，
  均EXIT0，前后无其他GPU计算进程。原子聚合、relaxed、warp1分别计时，未拼接三份收益。

三组各440对原始样本经独立复算，哈希、逐桶/逐轮统计与control全部吻合。
复核产物 `artifacts/competition/t69-atomic-aggregation-20260916/screening-closeout-review.json`，
SHA `14fb499d07b98e7676eb3e95a4ecacd9190a013181604e37a556259e36a0a591`。

本轮闭环：三条候选完成固定矩阵筛选，只有relaxed进入正式提交16139（8/8、51.072975）；
新增源码与重叠capacity/int32边界测试已入库。历史榜单最佳仍E10 51.3375、第5，剩18次额度。
全块排序和单warp没有晋级，源码保留E12；下一步需要更低局部通信成本的聚合结构证据，
不再沿本次两个失败配置重复提交，也不据此断言天数/海光榜差不可突破。

## 2026-09-16 晚间 E13：两路寄存器聚合，开发验证完成、未提交

- 新结构：每个256-route块采用两条128-route条带；同expert一次预留两个ticket，不同expert各自预留。无sort/scan/gather/shared histogram，保留负id写0、m_max0/1重叠桶所有权、非连续输入和专家大grid回归。wrapper及三个vendor源码均未改。
- 预注册35个非周期主桶+6个旧分布control+2个empty，固定IID、token内distinct top-k、20%/80%混合热点、相同频数shuffle；五轮AB/BA，主GM≥1.03、每轮≥1.01、各非热点组≥0.99、零spill。旧周期uniform不参与主GM。完整计划和原始样本在 `artifacts/competition/t69-pair-20260916/` 与 `artifacts/competition/t69-pair-tb-20260916/`。
- 对E12：主GM **1.0494666**，各轮1.042307–1.051643，IID1.016320/distinct1.020983/shuffle1.019717，无主桶<0.95。对团队最佳E10同一43桶复核：主GM **1.12869445**、算术均值1.13189760，各轮1.125933–1.132949，IID1.096447/distinct1.106652/shuffle1.117097，无主桶<0.95、零spill，两轮均过门。仅NVIDIA代理，不能推断八芯均分涨12.87%。
- 实际IR：3个probe中E12/E10分别32/31寄存器、2048B共享、8个barrier；候选21寄存器、0共享、0barrier、0spill。ticket增量、掩码返回值与双条带store逐项审查；格式化前后candidate PTX指令相同（排除源码位置/debug）。两轮baseline/candidate均各10方法通过；新pair边界已接入正式必测清单。
- source/verification commit `ea20af241b572d6669b3628683156ba1355cfb43`；源码SHA `cc61f445d11aab86d964ae426429b971c1ef18a0ecb7d43bd55806ecb8f368b1`，正式测试SHA `2f56bfdef27dcb3acf4394030202d874ebf244eedf434e1ca09fecce1a07dad7`。完整release **11/11**，0失败/错误/skip/xfail；generic真实launch138，Ascend/Enflame/Kunlun各414，均为RTX5070Ti数学代理。非NVIDIA目标runtime仍未验证，没有把代理折算成四芯通过。
- 发布回执来自独立远端 `/tmp/flagos-t69-pair-release.SwWjWo`，后台PID396185，运行脚本及日志在 `artifacts/competition/t69-pair-release-20260916/`；Python3.12.13/Torch2.13.0+cu130/Triton3.7.1/driver610.57.04。所有stage与计时串行，无并行GPU干扰。
- ZIP `artifacts/competition/fused_moe_dispatch_index/e13-pair-ea20af2/fused_moe_dispatch_index.zip`，19759 bytes，SHA-256 `478a1d3cb0651788524c94123e5fd412dcc2c876464721fb0f01ff8090547002`，generic+ascend+enflame+kunlunxin四成员。dry-run/final manifest、Git对象、全部成员、回执及相邻日志均验签。
- `verification.json` SHA-256 `b429cf2987d89932a1579f180584ff01813f9ef130a817493659421a6b32f1cd`。
- `verification.log` SHA-256 `fa51ed80f0ae28d098997bbb6d67a4c445e25c165e5a688b3ef9b307204f7844`。
- `release-audit.json` SHA-256 `a565cb2c6c21e9f7baf625ecc878cb0548eca2c0bf814dbedb7df5711191f0e2`。
- 当前完成开发、验证和打包；**未运行平台preflight、上传或提交**，平台TB与额度未因此改变。后续平台须按当时实时门禁执行，不能将本地候选就绪记成已上榜。

## 2026-09-16 22:21 E13 平台实验预注册

- 用户继续既有闭环，按现有持续授权提交；仅本候选一次上传和一次正式提交。实时赛题为competing/can_submit，账号全局剩18/30，截止2026-09-17 19:59:59。22:20–22:21只读逐芯快照 `docs/competition/data/pair-grouped-leaderboard-20260916-before.json`，SHA-256 `f7441d24290931d2782ecfb3a5c080f01117a62c3f4f6a33d9efca36c79e764b`；我方51.3375、第5，榜首80.917125。
- 正式晋级门：八芯正确、每芯≥0.1，且八芯算术均值>团队最佳E10的51.3375。假设为两路聚合降低generic五芯局部通信成本，三个vendor冻结；代理1.12869445仅支持方向，不直接外推平台均分。
- 风险：昆仑冻结vendor在E11/E12只有0.1036/0.1032，接近0.1门；本次generic改动不改善它。目标runtime证据由本次平台补齐。若均分未过门，保留E10，不重投相同ZIP；若失败，先读raw_result/selected_file定位，不用注释载体重掷。
