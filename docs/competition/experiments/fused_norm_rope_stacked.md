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
next: 当日额度耗尽;e2=5/8(行式vendor未修复三芯);远端fuzz 9504组合零失配=非逻辑bug
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

## E1 平台结果（5/8）+ E2 行式 vendor（2026-09-06）

- e1：天数 14.93x / 沐曦 4.74x / 海光 16.69x / A 13.06x / B 7.21x 过；
  燧原 PassManager 编译崩；昆仑 1-3% 元素大偏差（2D tile + axis=1
  reduction lowering 嫌疑）；华为仅最大 case（67M 元素）mismatch、
  且错误信息构造本身在 NPU 上崩（只在大张量打印时）。
- e2：燧原/昆仑/华为三 vendor 统一用"一 program 一 (t,l,h) 行"的
  1D [BLOCK_D] 结构（T51 已验证形态：tl.rsqrt、无 2D tile、无 3D grid、
  无 int64 cast）；已过 5 芯继续 generic。远端 variants 矩阵 7/7 OK。

## E2 平台结果（5/8，行式 vendor 未修复）+ 差分 fuzz 证据

- e2：同三芯败（燧原 PassManager 崩、昆仑 case0 反而 48.8%、华为大 case）。
  vendor 均被平台正确选中（selected_file 确认）。**行式结构证伪**：
  两种数学等价结构在同一芯上失败模式不同（e1 case0 4/256 → e2 125/256）
  → 非确定性特征（降级/调度），非逻辑错误。
- 远端 5070 Ti 差分 fuzz：9504 组合（T∈{1..33}×L×H∈{1..8}×D∈
  {32..256}×rd 组合×3 dtype×非连续 kv×bf16/fp16 cache×int32/int64 pos×
  [1,L,1,D] weight×float eps）—— **generic+row 两实现零失配**
  （仅 fuzz 自身 reference 对 python-float eps 崩，平台 eps 是张量）。
- 次日方向：(a) 燧原 PassManager 崩在两种结构都崩 → 怀疑 wrapper 的
  torch.full/isinstance 分支外的公共构造（如 eps reshape 或 contiguous
  链）不成立，更可能是 kernel 公共算子（tl.rsqrt/1÷sqrt 或 rope 对偶
  load）→ 试替换超越函数形态；(b) 昆仑/华为非确定性 → 试
  torch.zeros 替 empty、单 stream、或在 rope 对偶 load 改单 load +
  寄存器内 shift（tl.where 构造 partner 索引）消除双读；(c) 用一发
  "探针"提交（输出=inv_rms 广播）直接读出中间量差异定位。
