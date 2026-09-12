# Task 69 `fused_moe_dispatch_index` 实验记录

```current
task: 69
operator: fused_moe_dispatch_index
batch: 5
validity: invalid_correctness
platform: submitted(13406,e5,评测中;e4=13381 7/8 昆仑第4崩)
candidate_stage: e5
team_best_stage: -
sealed: no
next: 明日新 ZIP 再掷昆仑(候选改动=rank向量化/kernel2并行);七芯已过部分和356.25
updated: 2026-09-12
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
