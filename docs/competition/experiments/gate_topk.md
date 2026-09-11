# Task 70 `gate_topk` 实验记录

```current
task: 70
operator: gate_topk
batch: 5
validity: candidate-wip
platform: not-submitted
candidate_stage: s0
team_best_stage: -
sealed: no
next: 远端 GPU 恢复后补 release 回执（重点：tl.topk/tl.sort/tl.bitonic_merge 在代理 Triton 3.7.1 的编译与数值，tie-break 与 NaN 语义）；回执齐全进入发射队列（把握序第 6）
updated: 2026-09-11
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
