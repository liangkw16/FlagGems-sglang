# Task 58 `w8a8_block_int8_matmul` 实验记录

```current
task: 58
operator: w8a8_block_int8_matmul
batch: 4
validity: pending
platform: 7/8(s0,avg待算)
team_best_stage: s0
team_best_speedup: 0
sealed: no
next: e1(昆仑scalar-b_s vendor)待提交;s0仅昆仑PassManager崩溃,其余7芯37-367x
updated: 2026-09-06
```

## S0 fp32-ieee dot + 组内 scale（2026-09-06，远端 GPU 全过）

- 形态：单 Triton kernel；int8 load → `.to(tl.float32)` → `tl.dot(...,
  input_precision="ieee")`；每个 K 迭代后按 `k_start // block_k` 取
  As 行向量 [BLOCK_M] 与 Bs 列向量 [BLOCK_N]（`offs_n // group_n` gather），
  `acc += dot * a_s[:,None] * b_s[None,:]` — 与 reference 逐块舍入顺序一致。
- **遵守题面注意事项**"int8 须先 cast fp32、不可 int8 直接 GEMM"：不做
  int8 tensor-core dot，规避判罚风险（int8 dot 虽数学上更精确）。
- BLOCK_K = largest_pow2 ≤ min(group_k, 64)（保证 BLOCK_K | group_k，
  k 迭代不跨 scale 组）；BLOCK_N = largest_pow2 ≤ min(group_n, 64)；
  num_stages=2（3 stages + BLOCK_N=128 曾爆 shared memory 245760 > 101376）。
- 远端 5070 Ti：unittest 4/4 OK（平台契约 fp32 scales；atol 1e-2 本地线，
  平台 0.5）；bench 4 平台 shape：3.3x/6.5x/6.0x/22.4x，maxdiff ≤ 0.031。
- 教训：本地测试的 bf16/fp16 scale 子测是自加严（reference 的
  bf16 `s=a_s*b_s` 乘积自身差 |C|·2^-9≈0.04）— 已对齐平台契约改为
  fp32 scales。

## S0 平台结果（2026-09-06，submission ~08:40）

- **7/8 PASS**：天数 74.6x / 沐曦 147.5x / 燧原 4.60x / 海光 367.1x /
  华为 177.4x / A 61.7x / B 7.96x；**昆仑 PassManager::run failed（编译崩溃）**。
- e1：`_kunlunxin/ops/w8a8_block_int8_matmul.py` vendor — 唯一逐 lane
  向量整除 `offs_n // group_n` 改为标量 `(pid_n*BLOCK_N)//group_n`
  （BLOCK_N | group_n 时组索引跨 tile 恒定）；generic 字节不动。
  远端 variants 矩阵 5/5 OK（含 vendor 在 NVIDIA 代理编译+数值）。
