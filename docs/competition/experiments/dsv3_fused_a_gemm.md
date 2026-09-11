# Task 66 `dsv3_fused_a_gemm` 实验记录

```current
task: 66
operator: dsv3_fused_a_gemm
batch: 5
validity: candidate-wip
platform: not-submitted
candidate_stage: s0
team_best_stage: -
sealed: no
next: 远端 GPU 恢复后补 release 回执 + 代表形状测速（M≤16 skinny GEMM，权重读取为带宽瓶颈）；回执齐全进入发射队列（把握序第 5）
updated: 2026-09-11
```

## 契约与范围

- 完整题面：[Task 66](../tasks/batch-5/66-dsv3_fused_a_gemm.md)（2026-09-11 新增）。
- 接口 `dsv3_fused_a_gemm(mat_a, mat_b)`；`out = (a.float() @ b.float()).to(bf16)`；
  `mat_a [M,hd_in]` bf16 行主序，M ∈ [1,16]；`mat_b [hd_in,hd_out]` bf16
  **列主序**（等价 row-major `[hd_out,hd_in]` 的 `.t()`）；hd_in %256==0、
  hd_out %16==0；标准 per-dtype tolerance。
- 核心计算 Triton；八芯 0.1x。

## 实现（S0）

- 上游：SGLang 8014d9d `kernels/ops/gemm/dsv3_fused_a_gemm.py`（SM90 CUDA
  JIT skinny-M kernel）→ Triton：`tl.dot` GEMM，BLOCK_M=16（M≤16 恒满行
  掩码）、BLOCK_N=64、BLOCK_K=128（hd_in %256 保证无 K 尾块）；N 维
  grid-stride（cap 65535）。B tile 沿 K 连续（列主序下天然 coalesced）。
  bf16 输入 fp32 累加 = 参考的 fp32 语义（bf16 乘积在 fp32 中精确）；
  fp32 输入走 `input_precision="ieee"` 编译期分支（规格 dtype 为 bf16，
  fp32 属代理额外覆盖）。

## 不可变身份

- source / verification commit：`b4727f1`。
- source SHA-256：`3ca25d8cd5b5bc244d85bb90c3f238de4f0e11f3b956a4109a407293a0c966a7`。
- test SHA-256：`dce2d38c165431aec2602bc49672cc19a9fb971eaa137059685d29c1656f8533`。
- ZIP：`artifacts/competition/dsv3_fused_a_gemm/s0-b4727f1/dsv3_fused_a_gemm.zip`。
- ZIP SHA-256：`c1d84511cdefb73f14b618e6aacdb740d8f85fe049ea30edce326155919519a5`。

## 验证状态

- py_compile、格式与 lint 通过（本地）。
- 测试：4 方法 / dtype（bf16/fp16/fp32-ieee）、M∈{1,2,15,16}、
  hd_in∈{256..2048}、hd_out∈{16,64,256,2112}（2112 覆盖 N 尾块）、
  行距 stride mat_a、NaN/±Inf/-0 行（fp32 ieee，equal_nan）。
- **远端 GPU 不可达（2026-09-11 晚）**：release 回执与测速待补，
  `target-runtime-unverified`。

## 风险

- 性能：M≤16 时总 traffic ≈ hd_in×hd_out×2B（权重一次读），带宽受限；
  S0 的 `cdiv(N,64)` 程序数（hd_out=2048 → 32）在大卡上欠占用。
  E1 主轴是 **split-K**（K 维切分 + fp32 部分和归并）或 BLOCK_N=32 提
  程序数；弱芯（燧原 24 SIP）则相反，可能需要大 BLOCK_N 少程序。
- `tl.dot` BLOCK_M=16 为下限形态，各芯 tl.dot 均有本仓 GEMM 先例
  （T09/T23/T28/T58 均 8/8），编译风险低。

## 优化方向（按把握）

1. S0 直投（正确性把握高；性能首轮看逐芯）。
2. E1：按平台逐芯读数决定 split-K（大卡）或 BLOCK_N 放大（弱芯），
   单变量各一发。
