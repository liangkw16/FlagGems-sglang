# Task 70 `gate_topk` 实验记录

```current
task: 70
operator: gate_topk
batch: 5
validity: invalid_correctness
platform: completed(14556,e3,7/8;昆仑编译过但3630s挂死@0%util,转工单)
candidate_stage: e3
team_best_stage: -
sealed: no
next: e1 终态 5/8：三失败芯均 exec 0ms 崩溃族（12:20 同窗），kunlun vendor 被选中但未执行；下一步=新 ZIP 真实改动重评（燧原/华为可补迭代选择 vendor）或用户授权的同字节重掷
updated: 2026-09-12
```

## 契约与范围

- 完整题面：[Task 70](../tasks/batch-5/70-gate_topk.md)（2026-09-11 新增）。
- 接口 `gate_topk(x, k)`：小 k（≤32）流式 top-k，`values [M,k]` 同 dtype、
  `indices [M,k]` int32，降序、tie 取较小列号（题面即按此精确比较 indices）；
  x 2D 连续、numel ≤ 2^31；per-dtype tolerance 比较 values、indices 精确。
- 核心计算 Triton；八芯 0.1x。

## 实现（S0）

- 上游：SGLang 8014d9d `kernels/ops/moe/gate_topk.py` 的
  `_streaming_topk_kernel` 近逐行移植：(value,index) 打包可排序 key
  （fp 位翻转 + 低 16 位放 `N_PAD - col` 实现较小列号 tie-break），
  BLOCK_N=32 列块流式 + 寄存器态 top-k（`tl.topk`/`tl.bitonic_merge`/
  `tl.maximum`/`tl.sort`）。改动仅两处：签名收敛为竞赛双返回值；外层
  M 块 grid-stride（cap 65535）防超大 M 越 grid。
- dtype→key 位宽：fp16/bf16 → uint32 key；fp32 → uint64 key
  （仅在 kernel 体内，签名无 i64，不触燧原签名级 i64 毒点，但体内
  i64 ALU 在 GCU300 的编译仍是未证风险）。

## 不可变身份

- source / verification commit：`b4727f1`。
- source SHA-256：`e9a6eccb70db1ba7a1739aafe3dcaff737fb936e8f681dfd43cd4a0d19c02f71`。
- test SHA-256：`0c677de477b761dbe05b6a23e9c1e359aa322df7ab41b55404923381d5b74588`。
- ZIP：`artifacts/competition/gate_topk/s0-b4727f1/gate_topk.zip`。
- ZIP SHA-256：`5f68c43821ad3631dcda072b178a50809e4960870ac6b0085285820edb7d2e6f`。

## 验证状态

- py_compile、格式与 lint 通过（本地）。
- 测试：5 方法 / dtype（fp16/bf16/fp32）× k∈{1,3,8,32}、列数
  {1,8,31,32,33,64,96,1024}（N 尾块与 num_iters=0/多块）、行数
  {1,32,33,8193}（grid 边界）、**非循环 tie 判据**（全 fill 行 +
  双 1.5 列，硬编码期望 indices=[0..k-1]）、NaN/±Inf 行（equal_nan，
  key 翻转语义下 NaN 视为最大、与 torch.topk 一致）。
- **远端 GPU 不可达（2026-09-11 晚）**：release 回执待补，
  `target-runtime-unverified`。

## 风险

- `tl.topk/tl.sort/tl.bitonic_merge` 是较新 Triton API：代理 Triton
  3.7.1 有；厂商 fork（FlagTree/燧原/沐曦等）版本未证——这是六题中
  最大跨芯风险。若某芯编译失败，备选：
  1. vendor 变体改为手工 bitonic 网络（纯 tl.compare/xor 交换），
     规避 tl.topk/tl.sort 内建；
  2. 极端兜底：k 次迭代 max+掩码提取（O(N·k)，k≤32 可接受）。
- fp32 → uint64 key 的体内 i64 位移在 GCU300「int64 hw limitation」下
  未证；若燧原失败优先给 fp32 路径做 vendor（双 32-bit 键或分离比较）。
- torch.topk 自身 tie 顺序在 CUDA 上与题面 smaller-index 语义的完全
  一致性由非循环 tie 判据独立保障。

## 优化方向（按把握）

1. S0 直投，首轮读全芯编译结果（本题主要不确定性在编译面）。
2. E1：编译失败芯的 vendor 变体（手工 bitonic / 迭代提取）。
3. 性能轴后置：BLOCK_SIZE_M=32 × 多行共享列扫描已在 S0 内；如需再提，
   参考 T38（sigmoid_gate_topk_renorm）历史经验。

## 2026-09-12 平台提交（submission 13305，daily_seq 10）

- 源 5573ffc（截断转换修复）；测试 oracle 修正 commit 436be25
  （实测本机 torch.topk 在 CUDA 上违反题面 tie 规则约半数，改用 stable
  argsort 推导规格索引）；ZIP `s0-5573ffc`，SHA-256
  `1a99b9c8e55d7d4b33e50d60e2127e980e7cab9d8ca0333826397444c30c8f54`；
  回执 `batch5-ext6-validate-20260912/gate_topk/`
  （SHA-256 `2a717f129c8b78efc1b218479ff8d650c8d12673eeb72c18fee53167bfff36bb`，
  5 方法 0 失败、29 launch）。
- **昆仑失败**：`AttributeError: module 'triton.language' has no attribute
  'topk'`——XMLIR 的 Triton fork 版本落后，无 tl.topk（版本缺口，非数值）。
- 已过 5 芯（indices 精确比较全过 ⇒ 平台 reference 遵循题面 tie 规则）：
  天数 3.4901 / 沐曦 2.1890 / 海光 3.6598 / A 1.8043 / B 2.9476。
  燧原/华为终态=崩溃族（`服务线程卡死自动恢复，请重新提交`，exec 0ms）
  ——未获内核裁决；最终 5/8，唯一真实失败是昆仑版本缺口。
- 结论：下一候选为**去 tl.topk/tl.sort/tl.bitonic_merge 的昆仑 vendor**
  （手工 bitonic 交换网络或 k 次迭代 max+掩码提取），燧原/华为回调后
  定 vendor 覆盖范围。

## 2026-09-12 E1 平台提交（submission 13335，daily_seq 12）

- 昆仑 vendor（commit 4cc7092，live-mask 迭代提取 + NaN-first 最小列号，
  T27 昆仑已证 `tl.max + tl.min(tl.where)` 同形态；Codex 审查修正了
  原始草稿的 -inf 重复选中与 NaN 永不选中两个缺陷）。ZIP `e1-4cc7092`，
  SHA-256 `16d5ebafeea8a3f9ff7f6edeae75925c9221e0fd8d442aa6a0b905b72de3996b`；
  回执 `batch5-e1-validate-20260912/gate_topk/`，SHA-256
  `bfcf9d502ff4e5592bb6f90ce63d97ec82d5d0c0bfde8cfbd34e1f3346e3c672`，
  generic 29 + kunlun vendor 29 launch 全绿。
- 已判 5 芯通过；昆仑（本题唯一真实失败芯）、燧原/华为（崩溃族未裁决）
  回调未返回，收齐后补记。

## 2026-09-12 E1 回调终态（12:20 落定，13:5x 记账）：5/8，三芯均崩溃族

- 燧原/昆仑/华为全部 `completed + passed=false + exec 0ms`（12:20:03-04
  同窗落定）——服务线程卡死家族，内核未获裁决。昆仑 vendor
  （`gate_topk_kunlunxin.py`）被选中但同样 exec 0ms，live-mask 迭代提取
  的编译/数值仍未被平台检验。
- 已过 5 芯读数：天数 3.5076 / 沐曦 2.2318 / 海光 3.4154 / A 1.7945 /
  B 2.9266（对照 s0：3.49/2.19/3.66/1.80/2.95，窗口持平）。
- 后续路径：①同字节重掷三芯 = 崩溃族协议，需用户当次明示授权；②新 ZIP
  真实改动（如燧原/华为补迭代选择 vendor，或 kunlun vendor 微调）走
  全新评测。今日三芯崩溃族窗口频发（T65/T69 各芯亦有），明日窗口优先。

## 2026-09-14 E2 候选就绪：燧原/华为迭代选择 vendor，新 ZIP 重掷（待发射）

- 依据：e1 三失败芯均为 exec 0ms 崩溃族（同窗 12:20），generic 的
  `tl.topk/tl.bitonic_merge/tl.sort` 在燧原/华为栈无平台实证；昆仑
  迭代 vendor（tl.max+tl.where 逐列、live-lane mask）当时被选中但未
  获执行。E2 = 复用昆仑 vendor 字节 + 新增燧原/华为 vendor（同迭代
  形态）成真实改动新 ZIP，触发崩溃族重评。
- 新 vendor 设计（两芯同构）：逐行迭代 K 次 `tl.max + 两级无分支
  rank 选择`——NaN 候选恒优先于非 NaN 候选（修复点：纯 +inf 键会与
  真实 +inf 输入撞键，代理 test_special_values 18/72 失配后修正），
  层内最小列 tie-break，store 重读原始比特保 NaN 载荷；i32 寻址
  全程；grid-stride 行覆盖（coreDim≤65535）；不钉 num_warps。
- 代理证据：`--proxy-vendor enflame/ascend` 让两 vendor 在 NVIDIA
  代理跑完整数值矩阵（29 calls 各、0F0E0S），数值正确性已核，仅
  目标芯性能/编译未验（target-runtime-unverified）。
- source / verification commit：`ca0a5b1e…`；ZIP `e2-ca0a5b1`，
  4 members，SHA-256
  `9e20219838322cf6b6698a890bd722ab69d550a2505a21a61a830a997a61e48e`。
- release 回执 `batch5-submit-20260914/gate_topk/verification.json`
  SHA-256 `e93092497a6facba8fff7a9bf4273987bb537f557a7645616017ef2d59bd630a`。
- 预注册晋级门：**8/8（燧原/华为/昆仑任一过线即改善 5/8 现状）**；
  同指纹 exec 0ms 复现则按崩溃族协议收口（工单+健康窗重掷，≤2 次）。

## 2026-09-14 E2 平台终态：7/8（submission 14544）——燧原/华为首次过线

- **燧原 vendor 被选中、编译通过、passed 0.3539；华为 vendor passed
  0.3804**——两芯从 exec 0ms 崩溃族变为真实判决，`tl.topk` 流式形态
  在两栈不可用的推断成立。
- 昆仑：kunlun vendor 被选中，**验证执行阶段 4s 后 SIGABRT（编译器
  内部断言）**——与 e1 的"服务线程卡死 0ms"不同指纹，是本 vendor
  字节首次真实编译判决：旧形态的 runtime 标量分支（`if tl.sum(...)>0`
  包两路 tl.min）+ i64 行寻址疑似踩编译毒点。
- 七芯读数：天数 3.5447 / 沐曦 2.2379 / 燧原 0.3539 / 海光 3.7065 /
  华为 0.3804 / A 1.7959 / B 2.9283（部分和 17.948）；validity 仍
  invalid_correctness（昆仑缺），未上榜不变。
- E3 方向：把昆仑 vendor 的 kernel 体换成**与燧原 vendor 完全相同的
  无分支两级选择形态**（真实 GCU 已跑通该体）+ 代理 --proxy-vendor
  kunlunxin 补数值证据；同指纹绝不重掷，本改动是新字节+结构理由。
- 额度：发后 22/30。

## 2026-09-14 E3 候选就绪：昆仑 vendor 换 GCU 已证形态（待发射）

- 载体 = 昆仑 vendor kernel 体与燧原 e2 字节**完全一致**（import 后
  逐字节 diff 为空）：无分支两级选择、i32 寻址、grid-stride、不钉
  warps；弃 runtime 标量分支包双路 tl.min 的旧体与 i64 行寻址。
- 代理证据升级：`--proxy-vendor enflame/ascend/kunlunxin` 三 vendor
  各 29 calls、0F0E0S——昆仑 vendor 数值在 CUDA 代理全矩阵通过。
- source / verification commit：`7bfeef13…`；ZIP `e3-7bfeef1`，
  SHA-256 `477de9f22d0c7d66a435b0942449aa685fe71a985893f41ab28e251a5687ea09`。
- release 回执 `batch5-submit-20260914/gate_topk_e3/verification.json`
  SHA-256 `7574df54e274ba123fe266b6a8d06df7136418e53d2183bd2328c10696e91f8a`。
- 预注册晋级门：**8/8（仅昆仑待裁，七芯 e2 已过）**；昆仑再 SIGABRT
  同类即按崩溃族收口（工单+健康窗，不再改字节）。

## 2026-09-14 E3 平台状态：评测中（submission 14556，七芯已过、昆仑待判）

- 七芯终态读数：天数 3.4957 / 沐曦 2.1971 / 燧原 0.3638 / 海光 3.5482 /
  华为 0.3603 / A 1.825 / B 2.9635——与 e2 同水位，替换体的数值正确性
  在真实双芯再证。
- 昆仑 kunlun vendor 被选中，waiting_callback 已超 15 分钟未判
  （e2 的 SIGABRT 在 4s 内即判；本次长时间未崩是好信号，也可能在
  编译队列）。终态落地后回填本节与 CURRENT。

## 2026-09-14 E3 平台终态：7/8（submission 14556）——昆仑编译过、执行挂死，转工单

- 昆仑 kunlun vendor（无分支两级体）**编译通过**（e2 的 SIGABRT 消失，
  结构改造有效），但**验证执行阶段 3630s/3600s 超时**：子进程 R 状态、
  设备利用率 0%——平台自归因"疑似硬件/驱动卡死，可能不是用户代码
  问题"。与 e1 的 0ms 卡死、e2 的 4s 断言构成三种不同指纹，昆仑轴
  kernel 侧证据链完备（数值已在 CUDA 代理 29 calls 全过）。
- 处置：按崩溃族协议同指纹不重掷；三种指纹+代理证据打包提平台工单，
  等健康窗口以 e3 字节重评（≤2 次）。七芯读数：天数 3.4957 /
  沐曦 2.1971 / 燧原 0.3638 / 海光 3.5482 / 华为 0.3603 / A 1.825 /
  B 2.9635。validity 仍 invalid_correctness，未上榜不变。
