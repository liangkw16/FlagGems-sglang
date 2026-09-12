# Task 70 `gate_topk` 实验记录

```current
task: 70
operator: gate_topk
batch: 5
validity: invalid_correctness
platform: completed(13335,e1,5/8;燧原/昆仑/华为=exec0ms崩溃族未裁决)
candidate_stage: e1
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
