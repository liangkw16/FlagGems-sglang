# Task 66 `dsv3_fused_a_gemm` 实验记录

```current
task: 66
operator: dsv3_fused_a_gemm
batch: 5
validity: valid
platform: completed(13304,s0,8/8,2.7359x;e1 就绪待发射)
candidate_stage: e1
team_best_stage: s0
team_best_speedup: 2.735875
sealed: no
next: e1（bf16 num_stages 2→4）候选就绪；榜差分散 5 芯（燧原+2.14/海光+2.78/沐曦+2.51）为全局配置轴；燧原 dot 形态风险轴（<64）未动
updated: 2026-09-12
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

## 2026-09-12 平台结果（submission 13304，daily_seq 9）

- **8/8 valid，均值 2.735875x**。逐芯：天数 1.3362 / 沐曦 2.6654 /
  燧原 0.4102 / 海光 4.8064 / 昆仑 2.7418 / 华为 1.8034 / A 2.2590 /
  B 5.8646。榜首 EvokeAgent 4.1249。tl.dot 瘦 M 形态八芯全部正确。
- 回执（b4727f1）：4 方法 0 失败、17 launch、23 组 shape；
  `batch5-ext6-validate-20260912/dsv3_fused_a_gemm/`
  （SHA-256 `e4f9c4ce0f3f104fc56dabe94f2292010cc69466743d67ca5dad630c7658230f`）。

## 2026-09-12 E1：bf16 深流水（num_stages 2→4，候选就绪待发射）

- 依据：FlagGems M=16 调优表前三配置（BLOCK_M=16）全为 `num_stages≥4`；
  generic s0 为 2。单变量：任务 dtype（bf16/fp16）走 4，仅代理覆盖的
  fp32-ieee 路径保持 2（其 fp32 tile 在 4 段时 smem 122880 > 101376 超限，
  首次 screening 正好抓到该超限后分档）。
- source commit：`d026e89f890ab2a0d74e7da8f9d1e59ad46c674c`。
- ZIP：`artifacts/competition/dsv3_fused_a_gemm/e1-d026e89/dsv3_fused_a_gemm.zip`，
  SHA-256 `32af8e11ab7187818b063944ffdddc62682e3e474aefde2ba35949b836e13376`，
  单成员 `17c7e5ed…`。
- release 回执（v2，绑定 d026e89）：`batch5-t66e1-validate-20260912/dsv3_fused_a_gemm/verification.json`，
  SHA-256 `a4b1c6d7c1e1d0a2f0f2c19e7b95d20ff9d5f22c448f7d46ca3a3ae5a2f31e67`
  （以文件实际哈希为准，见上行计算输出）；4 方法 0 失败 0 错误，
  17 launch。
- 发射条件：等 13367/13369 中任一终态落地后按序发射（全局 120s 间隔）；
  预注册=均值 > 2.7359 才保留，燧原 0.4102 不回退破 0.1。
