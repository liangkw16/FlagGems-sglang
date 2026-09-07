# Task 54 `fused_norm_rope_stacked` 实验记录

```current
task: 54
operator: fused_norm_rope_stacked
batch: 4
validity: invalid_correctness
platform: 5/8(e2,10414已终态)
team_best_stage: -
team_best_speedup: -
sealed: no
next: 阶段探针/67M代理stress通过;真实三芯失败仍待目标原输入复现,无新生产修复或ZIP
updated: 2026-09-08
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
  vendor 均被平台正确选中（selected_file 确认）。**该版本仍失败**：
  两种数学等价结构在同一芯上失败模式不同（e1 case0 4/256 → e2 125/256）
  → 只能证明两个版本行为不同；尚不能排除逻辑错误或确定性 lowering 问题。
- 远端 5070 Ti 差分 fuzz：9504 组合（T∈{1..33}×L×H∈{1..8}×D∈
  {32..256}×rd 组合×3 dtype×非连续 kv×bf16/fp16 cache×int32/int64 pos×
  [1,L,1,D] weight×float eps）—— **generic+row 两实现零失配**
  （仅 fuzz 自身 reference 对 python-float eps 崩，平台 eps 是张量）。
- 次日方向：(a) 燧原 PassManager 崩在两种结构都崩 → 怀疑 wrapper 的
  torch.full/isinstance 分支外的公共构造（如 eps reshape 或 contiguous
  链）不成立，更可能是 kernel 公共算子（tl.rsqrt/1÷sqrt 或 rope 对偶
  load）→ 试替换超越函数形态；(b) 昆仑/华为先固定源码与输入重复运行，再评估
  torch.zeros 替 empty、单 stream、或在 rope 对偶 load 改单 load +
  寄存器内 shift（tl.where 构造 partner 索引）消除双读；(c) 在独立 debug kernel 中比较 inv_rms、归一化值和 RoPE；
  不经自动发布门禁提交故意改变输出契约的探针。

## 2026-09-07 只读盘点校正

最新 E2/sub10414 已终态 invalid_correctness，天数/沐曦/海光/A/B五芯通过；燧原/昆仑/华为失败，无有效均值。旧pending与当日额度耗尽措辞过期；本轮未提交。
- 查询证据 `/Users/bytedance/ccc/flagos/artifacts/competition/batch4-top1-20260907/tasks-now.json` SHA256 `fc73368c3d98b228b0c7815d6ec1e9042a58af8d953337fff58990daec1474fc`。
- 查询证据 `/Users/bytedance/ccc/flagos/artifacts/competition/batch4-top1-20260907/54-submissions-now.json` SHA256 `d5e160132b38dc9eb7f53cf8671f861a4223a7a89bcd8538da3d0fa20b6efd05`。

## 2026-09-08 推荐方案实现与提交前验证（未提交平台）

新增norm/rotary/V取证与准确cache宽度测试，67M输入stress进入必需集。华为源码大形状代理K容差通过、V bitwise相等，插桩不改最终输出；未拿到实际平台失败输入，未声称生产修复。

- source `c73f6c3f83ec38d5a2c40cfdef996e64e50ecd67`；verification `c73f6c3f83ec38d5a2c40cfdef996e64e50ecd67`。8 个测试方法、27 次实际 kernel 调用；选定 NVIDIA/代理范围门禁通过。
- 回执 `artifacts/competition/batch4-implementation-20260907/t54-release1/verification.json`，SHA256 `bfb27398beaad6298dc0515a2f59f703ec99ff446dc1b478c36ea3e15fbd432e`；日志 SHA256 `11242e63d9605f5f53f8c6d22efd24424383215ed18d6c2ee2a24aee8d7f7b10`。
- 环境、逐源码执行范围、原始配对数据和未完成条件见[本轮报告](../implementation-batch4-20260908.md)及[证据清单](../data/batch4-implementation-20260908.json)。本轮不更新历史有效分，未做平台 preflight、上传或正式提交。
