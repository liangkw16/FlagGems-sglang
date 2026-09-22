# Task 81 `fused_gate_sigmoid_mul_add` 实验记录

```current
task: 81
operator: fused_gate_sigmoid_mul_add
batch: 6
validity: valid
platform: completed(17611,e11,8/8,4.291x<TB;保e10;燧原单波消循环中性)
candidate_stage: e12
team_best_stage: e10
sealed: no
next: e12已上膛(五元组齐:ZIP d7b06976,回执8测试0失败6路径x35launch);09-23额度刷新后preflight→submit;回执后燧原GCU几何移植候选为下一发(文件互不重叠)
updated: 2026-09-22
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

## 2026-09-18 GitHub 情报：PR #26856 上游同构 kernel 与 e3 判读重释

- `sgl-project/sglang` PR #26856（open，Qwen3.5Opt 系列 2/N，2026-05-31）
  提交了与本题同名的 `fused_gate_sigmoid_mul_add` Triton kernel：**单行/
  program、HDIM constexpr、全行单 tile（next_pow2 无上限）、双相位**——
  与我方 e5 同构；关键差异是 **num_warps 公式钉位**：
  `max(min(next_pow2(cdiv(hdim,256)), 32 or HIP 16), 4)`（5120/7168 → 32）。
- **e3 判读重释**：e3 全行 8192 tile 在沐曦/海光 -31/-38% 当时归因
  "掩码 lane 浪费"；上游同 tile 配 32 warps 可跑——真实根因更可能是
  **宽 tile × 默认 4 warps 线程不足**。e7 候选=generic 全行单 tile +
  上游 warps 公式（燧原 vendor 不钉、昆仑 vendor 不动）。
- 关联：#26727（系列 1/N，共享专家门融合）、#36176（CUDA warp 向量化
  拷贝基建，T78 背景参考，Triton 不可移植）。竞赛上游 flagos-ai 无第 6
  批 PR（最新停在 batch3）。

## 2026-09-18 E7/E8/E9 平台终态：上游 warps 形态三连击，TB 4.306075（+9.1%）

- E7（17374，`3d840395`）：PR #26856 上游形态（全行无上限 tile +
  `warps=max(min(next_pow2(cdiv(hdim,256)),32),4)`）。**6/8**：天数
  6.63→7.75（+17%）/ A +7% / B +10% 兑现；**沐曦/海光 OutOfResources**
  （线程上限 512/1024，海光 warpsize 64）；华为 -11%。
- E8（17375，`9cedb6ca`）：组合修正——metax/hygon vendor 16-warp 上限、
  ascend vendor 回 e6 字节。**7/8**：海光 6.00→**6.74（+12% 恢复且新
  高）**、华为回 2.24、天数 7.77 保持；**沐曦仍挂**（warpsize 64 →
  16 warps 仍要 1024 线程 > 512 上限，实际上限 8 warps）。
- E9（17378，`403a852a`）：metax vendor 回 e6 已证字节（4.13）。
  **8/8，均值 4.306075 新 TB**。逐芯：天数 7.80 / 沐曦 4.11 / 燧原
  1.99 / 海光 6.76 / 昆仑 0.73 / 华为 2.31 / A 5.91 / B 4.83。
- 判读：**e3 的"tile 宽度掩码浪费"判读被修正为 warps 饥饿**——上游
  PR 的 warps 公式在 CUDA 类芯兑现 +17%（天数），按芯线程上限
  （海光 16-warp、沐曦 8-warp 装不下→回退）拆 vendor 是正确组合。
  A 轴已与 c2flow 打平（5.91 vs 5.9）。
- 新增跨芯硬事实：沐曦线程上限 512@warpsize64、海光 1024@warpsize64
  ——上游 warps 公式在国产芯需按 `threads_limit/warpsize` 换算封顶。

## 2026-09-18 E10 平台终态：沐曦 8-warp 全行 tile，TB 4.34518333（+0.9%）

- 结构（`c66dd3e2`，单变量）：metax vendor 改 8-warp 上限的全行 tile
  （线程上限 512@ws64 内的最大 warps）。release 六路径全过。
- submission **17390** completed/valid，8/8，均值 **4.34518333 新 TB**
  （vs 4.3061，+0.9%）。逐芯：沐曦 4.108→4.179（+1.7%，编译通过但收益
  有限——与海光 16-warp 的 +12% 分化，宽 tile 收益按芯分化再添一例）；
  天数 7.90 / A 5.98 / B 4.93 / 华为 2.36 小幅上漂（窗口）。
- 排名：**超 GuanghuLab（4.32）升 #4**；距 #3 HAiWORLD（4.3993）差
  0.055，距真实靶 c2flow 5.10 差 17.6%。

## 2026-09-18 T81-B 双累加链：代理筛失败，不发射

- 候选：双独立 fp32 累加链交替处理相邻块（重规划轴 B）。代理对照
  e7 单链（三形状七轮交替）：ratio 1.001-1.024 **平手**——静态展开后
  累加依赖链不是 NVIDIA 代理上的瓶颈。门（≥2 形状 ≥10%）未过，
  按纪律不消耗平台额度。轴关闭（无新证据不重开）。

## 2026-09-18 晚 E11 候选就绪（提交前验证完成，待额度刷新首发）

- 结构：燧原单波消外层循环（round-5 轴2）。预注册门：燧原≥2.4或耗时-15%;均值>4.3452换TB。
- source/verification commit `8ec9e828c7e4f05e0131feda09853ae77f31b5ce`；release 回执
  `artifacts/competition/b6-e11-ready-20260918/fused_gate_sigmoid_mul_add/verification.json`（6成员,6路径x31launch,5测试）。
- ZIP `artifacts/competition/fused_gate_sigmoid_mul_add/e11-*/fused_gate_sigmoid_mul_add.zip`：
  SHA-256 `13bc12c9420198347e7c7931c77e08f9d3a5b8ec466f84abb8b1965b218d56b8`；test SHA-256 `0cc7c7af52c87ca0e3b6ec7e74c8b71f3d582ce3967204e6551da0e0f242efc2`；回执 SHA-256 `5f8691620c853108b43bc45d7c5eb787d2e49fd4e1a3990f46e3c8a297b4e675`。
- 发射参数齐备（commit/zip/sha/test/receipt 五元组已核对），午夜额度
  刷新后按第五轮排序直接 preflight→submit。

## 2026-09-22 E12 候选开发完成：上游 #26856 单波 launch 结构（generic/metax/hygon/kunlun 四路径），待远端回执

- 结构（第 1 轮，四路径同构单变量=launch 结构）：grid=(rows,) 每行一
  program（去 min(rows,2048) cap 与 grid-stride 行循环）、行偏移
  pid*HDIM constexpr、寻址 int32；generic warps 公式按 #26856 对 HIP 族
  （card_b 走 generic）封 16、CUDA 类仍 32；metax 8 / hygon 16 上限与
  昆仑 BLOCK_H=1024（XPU packing 防御）均不动。守门：rows>65535、
  rows*hdim≥2^31 或行 stride≠hdim 回 multi-wave 已证字节（e10 形态
  kernel 原样保留为回退路径）。
- 假设与证据：上游 PR #26856 实读 patch（单行/program、pid*hidden_dim、
  HIP 16 warps 封顶）；16:37 快照前 12 队沐曦/海光/card_b 一致高我方
  而 card_a 持平的形态指纹；e5 +27% 控制开销先证；昆仑 0.73 vs c2flow
  0.988。四芯天花板算术（s2t1op081 快照）：muxi 4.179→4.883(+0.70)/
  haiguang 6.712→7.700(+0.99)/kunlun 0.731→0.988(+0.26)/card_b
  4.93→6.268(+1.34)，全对齐 +3.29 单芯和 ÷8=+0.41 均值；
  expectedAvgGain 0.2 为天花板×约半成功率取整档，非全对齐承诺。
  generic 改动波及 tianshu/card_a（无 vendor 覆盖），场榜 card_a
  5.97-6.03 持平支持中性预期，-5% 回滚门兜底。
- 预注册门：均值 >4.3452 换 TB 且 muxi≥4.6 与 haiguang≥7.3 至少一芯
  兑现；任一带宽芯 -5% 判负回滚 e10 字节。
- 测试：沿用全矩阵（现有形状全部走单波路径），新增
  test_single_wave_grid_routing（65535/2^31 路由边界，纯 host 断言——
  2^31 档无法在 16GB 代理上真实分配）、test_multiwave_fallback_above_
  grid_limit（rows=65536 真实走 multi-wave 回退，bf16+fp32）、
  test_single_wave_multi_wave_parity（同字节稠密/行间隙两分支逐位一致，
  覆盖含燧原/华为全部 7 模块）；三项均列入 RELEASE_REQUIRED_TESTS。
  py_compile 五文件通过；_single_wave_grid 路由算术自四文件实源提取
  执行、边界断言全过（本机无 torch/triton，kernel 数值回归待远端
  release 回执）。
- 额度：09-22 已用 30/30，09-23 刷新后发射；燧原 GCU 几何移植候选
  （天花板 +0.31）与本候选文件互不重叠，本候选回执后即为下一发。

## 2026-09-22 E12 上膛完成：release 回执绿 + 不可变 ZIP，五元组齐备待发射

- release 回执（NVIDIA 代理 RTX 5070 Ti / torch 2.13.0+cu130 / triton 3.7.1，
  source=verification commit `fa77dbce34702f56f0759db09fc8d45c3f5e6021`）：
  `artifacts/competition/day5prep-20260921/fused_gate_sigmoid_mul_add-wf/verification.json`
  SHA-256 `bd159074f62f51a32d9e6b28bf7f4533b9ba8d46804bf96270a5b8e4b32c343a`；日志
  SHA-256 `58113196b19ad3c1ccf264d4c814276c319b42ed5f3ac63311ff9abf532b8986`。
  8 测试 0 失败 0 skip 0 xfail，RELEASE_REQUIRED_TESTS 8/8 进入 suite 且
  通过（expected==passed 核对）；6 适用源（generic+ascend+enflame+hygon+
  kunlunxin+metax）各 35 次非 warmup kernel launch，unexecuted 空；5 vendor
  路径照例 target-runtime-unverified（NVIDIA 代理证据，裁决权在平台）。
- 五元组（发射 preflight 依据）：
  - commit（source=verification）：`fa77dbce34702f56f0759db09fc8d45c3f5e6021`；
  - ZIP：`artifacts/competition/fused_gate_sigmoid_mul_add/e12-fa77dbc/fused_gate_sigmoid_mul_add.zip`
    （32849B），SHA-256
    `d7b06976465c7346c98ce9fe8b4ea013ad3bf11e7e924e4efc342671201719be`
    （≠ e11 `13bc12c9…`，新 ZIP 字节，平台去重键 zip_sha256 不冲突）；
  - test SHA-256：`ccde599a8406b398364fa4fe67f33193a75612388046c1294ebc5493c5f77ccc`；
  - 回执：上述 verification.json（SHA-256 `bd159074…c343a`）；
  - 预注册门：均值 >4.3452 换 TB 且 muxi≥4.6 与 haiguang≥7.3 至少一芯
    兑现；任一带宽芯 -5% 判负回滚 e10 字节。
- ZIP 成员逐项（zipfile 实际枚举 = 打包器 manifest = git `fa77dbce` 源字节
  三方核对一致，unzip -t 无错，无夹带成员）：
  - `fused_gate_sigmoid_mul_add.py`（generic，6680B，SHA-256
    `7320436d39a08799cc8ca6f44c721ab585750b2ea88deea818f6c09e74710830`）；
  - `fused_gate_sigmoid_mul_add_ascend.py`（3446B，
    `aa48c034172dcdc122d1703918f70894a28d4388ac5c223d134503680e482c29`）；
  - `fused_gate_sigmoid_mul_add_enflame.py`（3930B，
    `638dab223d791a8ec3ff8579bc897c0394bd8b06f757970a633b2b2799751204`）；
  - `fused_gate_sigmoid_mul_add_hygon.py`（6258B，
    `30bab80e7a45e3606b560e7c2f3061797a44bc3f494cb01b9756a38001f72f8b`）；
  - `fused_gate_sigmoid_mul_add_kunlunxin.py`（5771B，
    `90949b70209008dab131bb300ca6dbcfe6526d1c463c605de8ce9ca7ce4b041c`）；
  - `fused_gate_sigmoid_mul_add_metax.py`（5864B，
    `8645794657aa4f03dc7e223fd97f51ba36b18d8402321280d2c6124cbe2d290d`）。
  全部 UTF-8 `.py`、generic/vendor basename 精确合规、无目录前缀或垃圾文件。
