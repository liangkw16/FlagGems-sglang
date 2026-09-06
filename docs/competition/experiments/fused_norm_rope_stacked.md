# Task 54 `fused_norm_rope_stacked` 实验记录

```current
task: 54
operator: fused_norm_rope_stacked
batch: 4
validity: pending
platform: 0/8(s0)
team_best_stage: e1
team_best_speedup: 0
sealed: no
next: e1(h_mask修复)待提交;s0全芯49%错=H非4倍数时OOB写坏下一token
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

## S0 平台结果 + e1 修复（2026-09-06，submission 08:43）

- 0/8：49-99% mismatch，abs diff 7-21。根因：`[HEADS_TILE=4, BLOCK_D]`
  tile 的 mask 只有 d 轴（`d_mask[None,:]` 广播），H 非 Heads_TILE 整数倍
  时多出的 head 行未屏蔽——K 装载读到同 (t,l) 的 V 区、store 越界写进
  下一个 token 的输出槽。平台 case0 是 H=2（本地测试 H∈{4,8,16} 全是
  4 的倍数，未覆盖）。
- e1：所有 load/store 加 `h_mask[:,None]`（K/V tile、rope 对偶、tail）；
  测试补 H∈{2,3,5,7}×T×L 用例。远端 6/6 OK。
- 教训沉淀：**tile 的每一根轴都必须进 mask，"shape 恰好整除"不能当不变量**；
  测试矩阵要覆盖非整除 tile 的轴值。
