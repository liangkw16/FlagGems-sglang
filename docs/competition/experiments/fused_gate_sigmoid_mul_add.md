# Task 81 `fused_gate_sigmoid_mul_add` 实验记录

```current
task: 81
operator: fused_gate_sigmoid_mul_add
batch: 6
validity: valid
platform: completed(17372,e6,8/8,3.94720833x新TB;燧原vendor静态化+26%过门)
candidate_stage: e6
team_best_stage: e6
sealed: no
next: 燧原静态化过门(1.56→1.96,+26%);昆仑轴判负关闭(0.733<门0.88,1D store近memory-bound与T80同理);静态机制三连兑现(TB 3.07→3.90→3.95);剩余=T81-B双链(代理前置)/T78维度特化(IR前置)/燧原vs c2flow 4.4仍2.2x
updated: 2026-09-18
```

## 契约与实现（S0）

- 题面：[Task 81](../tasks/batch-6/81-fused_gate_sigmoid_mul_add.md)。
  `gate = Σ(hidden*gate_weight,-1)` fp32；`out = final + sigmoid(gate)*shared`
  （fp32 乘加）→ final dtype；标准容差。上游 d4ad368 FUSE_GATE 路径对照。
- 实现：单 kernel 两阶段（阶段 1 行点积 fp32——hidden 只读一次；阶段 2
  final+shared FMA store），一 program 行粒度 grid-stride（≤2048），
  BLOCK_H=1024 + 尾 mask；内维连续断言、行 stride 透传。
- 测试：dtype 矩阵 × 形状（1×1 到 2049×7168、BLOCK_H 尾 1535）、门控饱和
  （±200）与 gate=0 行、fp32 抵消行、NaN 传播、行间 stride。
  本地容差覆盖归约重排噪声（抵消点相对差放大），远严于平台标准。

## 不可变身份

- source commit：`370923bfb83a8b7e681051acb2a3b1585dde12fc`（五题同批）。
- verification commit：`70a4a4d58fe727496001e029deac9ff3aa6fe93d`（含测试侧修复轮）。
- source SHA-256：`b8053e124d7309d80e108ac9ae69a6fc3dfd3060b0c8c1aac1bf606ac4ab853c`。
- test SHA-256：`7c0bae7f2c82eb87f89dcbeec844549f8c4cdd26eaa12fe724fe3cc5ebe20227`。
- ZIP：`artifacts/competition/fused_gate_sigmoid_mul_add/s0-370923b/fused_gate_sigmoid_mul_add.zip`，SHA-256
  `5327412cb23be84e62a287f4b3c9c7e0fe3dcca892c18193a66b9d44ad32c0bf`（单成员 `fused_gate_sigmoid_mul_add.py`，generic-only）。
- release 回执：`artifacts/competition/b6-s0-release-20260917/fused_gate_sigmoid_mul_add/verification.json`
  （5 测试 0 失败，22 次非 warmup kernel launch，
  NVIDIA RTX 5070 Ti / torch 2.13.0+cu130 / triton 3.7.1），
  SHA-256 `75e8667cd4aed70549dfce35a5827f1b5554142225743d695ddc809ff49d48ab`；日志 SHA-256 `48cf0b58e7ac5b0c5dec8361875dd6341cce5db3775e996a292a1f297c464401`。

## 靶子与下一步

- 榜首 c2flow 4.4852x（领先次席 +1.95%）；e 轴：华为 persistent（上行假设，方向性）→ 燧原（3.89 vs 次优 2.20）→ 多行宽度对照。
- 验证三轮教训已固化在测试侧：numel%16 契约、token 边界与 kv_lens 解耦、
  归约噪声容差、行内连续 stride 构造。
- 八芯目标 `target-runtime-unverified`（NVIDIA 代理证据），裁决权在平台。

## 2026-09-18 S0 平台首回执（submission 17213，observed_at 01:0x +08）

- 状态：8/8 valid, ~#7/8；均值 3.002075。
- 逐芯：天数 5.3907 / 沐曦 2.8475 / 燧原 1.0784 / 海光 3.5045 / 昆仑 0.7309 / 华为 2.2013 / A 4.405 / B 3.8584。
- s0 未达 3.5 预注册门。逐行两阶段在带宽芯全面落后（海光 -53%、燧原 -72%），结构轴优先于 vendor 轴。

## 2026-09-18 E1 平台终态：多行 2D tile 证伪，8/8 但均值 2.0064 < TB 保 s0

- 结构（`0420600f`）：generic B_ROWS=4 × BLOCK_H=512 2D tile + `_kunlunxin`
  vendor 保留 s0 1D 逐行字节（XPU packing 预防）。release 双路径 22+22
  launch 全过；ZIP `e1-0420600` SHA-256
  `c57d31b6b11ae7492132f3c5389020e136de18b67e49e6bc07594819c222b7e8`。
- submission **17271** completed/valid，8/8，均值 **2.0064x < TB s0 3.0021**
  （保 s0）。逐芯（vs s0）：天数 5.39→2.96（-45%）/ 海光 3.50→1.78
  （-49%）/ 沐曦 2.85→1.68（-41%）/ A 4.41→2.96 / B 3.86→2.77——带宽芯
  全线大幅回退；昆仑 0.7298（vendor=s0 字节，读数稳定 ✓）/ 华为 2.18
  （持平）。
- 判读：**多行复用在本题为负结构**。T53 的多行收益来自 gate_weight 重读
  摊销（其权重按 program 全量重载）；本题 gate_weight 仅 [hidden]（~10KB，
  L2 常驻），摊销收益趋零，而 program 数÷4 直接砍掉内存级并行。与 T53
  经验边界互补：**多行适用性=权重重读成本÷并发损失，逐题验证**。
- 新假设（未投）：两 kernel 分裂——K1 批量 GEMV 点积（tensor-core/分块
  归约）+ K2 纯流 FMA（全 grid elementwise）；触发条件=代理 benchmark 先
  证 K1+K2 wrapper 总耗时 < 单 kernel 两阶段。燧原轴另有 BLOCK 阶梯假设
  （单趟全宽 dot，GCU 偏好）待同门验证。

## 2026-09-18 E2 平台终态：双 kernel 分裂证伪，8/8 但均值 2.56 < TB 保 s0

- 结构（`11e5693e`）：K1 纯行点积（grid=rows）+ K2 纯流 FMA（sigmoid 在
  K2 内），每相满格并行；`_enflame` vendor 保 s0 单 kernel 字节（GCU 双
  launch 开销风险），`_kunlunxin` vendor 不变。代理 AB 基准平手
  （0.97-1.02，三形状五轮交替）。release 三路径 44+22+22 launch 全过；
  ZIP `e2-11e5693`。
- submission **17311** completed/valid，8/8，均值 **2.55998333x < TB s0
  3.0021**（保）。逐芯（vs s0）：天数 5.39→4.71 / 沐曦 2.85→2.46 /
  海光 3.50→3.12 / A 4.41→3.79 / B 3.86→3.36 / **华为 2.20→1.22（-44%）**；
  燧原 1.074（vendor 字节稳定 ✓）/ 昆仑 0.732（稳定 ✓）。
- 判读：**并发假设在此形态亦证伪**——额外 launch + gate 中间量的成本
  超过相位并行收益；Ascend 双 launch 惩罚与 GCU 同型（T80 昆仑 2D、
  T81 华为双launch 互证）。s0 单 kernel 两阶段在 {s0, 多行, 双kernel}
  三结构中最优。榜首 5.1（c2flow，燧原 4.4）结构未破译，重开需新证据
  （上游 PR/他队泄露/逐芯分布）。
- 基准期发现并修复：重写时丢失 sigmoid 的 bug 被代理基准 correctness
  交叉校验拦下（K2 曾直接乘原始点积）。

## 2026-09-18 E3/E4 平台终态：full-row 燧原 +42%，组合 TB 3.07044167

- E3（17322，`7e80f5b3`）：generic 与燧原 vendor 同改整行 tile
  （BLOCK_H=min(8192,next_pow2(hdim))）。**燧原 1.078→1.5696（+46%）**、
  B +3%/A +5.5%；但沐曦 2.85→1.96（-31%）、海光 3.50→2.18（-38%）——
  5120/7168 宽度下 8192 tile 的掩码 lane 浪费 37-60%。均值 2.755 < TB。
- E4（17326，`54d3cd8b`）：generic 回 s0 1024 循环，燧原 vendor 保留宽
  tile——组合判决：**均值 3.07044167 新 TB**（+2.3%）。逐芯：天数 5.55 /
  沐曦 2.88 / 燧原 **1.5344（+42% vs s0）** / 海光 3.43 / 昆仑 0.729（vendor
  稳定）/ 华为 2.23 / A 4.28 / B 3.93。
- 判读：**同结构按 chip 分化第二次实证**（融合 vs 分离于 T80 之后）——
  GCU 偏好单趟宽 tile（T63 阶梯同源），沐曦/海光偏好紧凑 1024 循环。
  四结构档案：s0+燧原宽 vendor（现 TB）> s0 > e1 多行 > e2 双kernel。

## 2026-09-18 E5 平台终态：HDIM constexpr 静态展开，TB 3.90445（+27.3%）

- 结构（`292fdca3`，重规划轴 A）：`hdim` 移出 do_not_specialize 改
  `HDIM: tl.constexpr` + `tl.static_range`——hidden 循环全展开、整除时尾
  掩码折叠。代理门通过（受影响形状 +7.6~112%，控制形状回退 ≤0.7%）。
- submission **17355** completed/valid，8/8，均值 **3.90445 新 TB**
  （vs 3.0704，+27.3%）。逐芯（vs e4 TB）：**海光 3.43→6.08（+77%）/
  沐曦 2.88→4.13（+43%）/ A 4.28→5.29（+24%）/ 天数 5.55→6.75（+22%）/
  B 3.93→4.48（+14%）**；华为 2.22 持平；燧原 1.56（vendor 保持）；
  昆仑 0.728（vendor 稳定）。
- 判读：**控制流开销是 T81 带宽芯缺口的第二主因**（仅次于结构本身），
  平台兑现远超代理（+27% vs 代理中位 ~10%——平台编译器对 runtime 循环
  的成本高于 NVIDIA）。同机制待迁移：T80 generic 的 hv 循环、T81 昆仑
  vendor 的 runtime 循环。

## 2026-09-18 E6 平台终态：vendor 静态化第三连兑现，TB 3.94720833

- 结构（`991a2878`，重规划轴 1+2 一发覆盖）：燧原/昆仑两 vendor 同步
  HDIM constexpr + static_range（generic e5 字节不变）；测试矩阵补
  8191/8192/8193（燧原 cap 两侧）。release 三路径全过；ZIP `e6-991a287`。
- submission **17372** completed/valid，8/8，均值 **3.94720833 新 TB**
  （+1.1%）。逐芯判决：**燧原 1.556→1.956（+26%，≥1.80 门过 ✓）**；
  昆仑 0.728→0.733（+0.6%，<0.88 门——**昆仑轴判负关闭**：1D store 循环
  近 memory-bound，与 T80 HV 中性同理）；A 5.29→5.43；其余窗口持平。
- 静态机制三连：TB 3.07→3.90（e5 generic）→3.95（e6 vendor）。燧原对
  c2flow 4.4 的差距 2.8x→2.2x。
