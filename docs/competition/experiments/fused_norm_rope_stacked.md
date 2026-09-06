# Task 54 `fused_norm_rope_stacked` 实验记录

```current
task: 54
operator: fused_norm_rope_stacked
batch: 4
validity: pending
platform: 0/0
team_best_stage: s0
team_best_speedup: 0
sealed: no
next: s0 待提交;榜首 EvokeAgent 10.22x(8/8)仅2队过线,高价值;提交后按逐芯结果迭代
updated: 2026-09-06
```

## S0 单遍融合 kernel（2026-09-06，远端 GPU 全过）

- 形态：3D grid `(T, L, cdiv(H, 4))`；每 program 处理
  `[HEADS_TILE, BLOCK_D]` 的 K+V 切片。
  - K：fp32 load → per-head `sum(k*k)/D` → `1/sqrt(var+eps_l)` →
    `* w[l,:]`；tail（d ≥ rotary_dim）直接存归一化值；
  - RoPE：k1/k2 对偶 load（[0,r) 与 [r,2r)），各自乘 inv_rms 和
    w_r1/w_r2 后 cos/sin 旋转，避免寄存器 tile 内 gather；
  - V：纯转置拷贝（不进 fp32，与 reference 一致）。
- reference 是 float→pow→mean→rsqrt→两次 slice 写→clone→permute→
  contiguous 的几十遍数据流 → 单遍 kernel 有大空间。
- 远端 5070 Ti：unittest 5/5 OK（fp32 1e-4 / bf16 1.5e-2 / fp16 1e-2，
  含 rotary=D、D=96 非 2 幂、T=257 奇数、T=1、H=16）。
