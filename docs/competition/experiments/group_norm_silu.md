# Task 72 `group_norm_silu` 实验记录

```current
task: 72
operator: group_norm_silu
batch: 5
validity: candidate-wip
platform: completed(13818,e4,7/8;昆仑uni_sram@64,四档全超,kernel复杂度问题)
candidate_stage: e4
team_best_stage: -
sealed: no
next: 昆仑 tile 降档轴封;重开=FlagGems 昆仑 softmax 实际形态研究或工单
updated: 2026-09-13
```

## 契约与实现（S0）

- 完整题面：[Task 72](../tasks/batch-5/72-group_norm_silu.md)（2026-09-12 晚新增五题之一）。
- division-free [C,S] tiles; scalar-carried stats; three-pass centered variance。
- 核心计算 Triton，无 fallback；八芯 0.1x；截止 2026-09-17 19:59:59。

## 不可变身份

- source / verification commit：`e6b450fd4beb001255079ade929d3b2de51796d6`（五题同批提交）。
- source SHA-256：`690aa6bd15ed593fa76f8b412457aa9d4a1cec4d18d8ae7f0519b213f50bc229`。
- test SHA-256：`80f5c46cc34626a26ce0f6bba354420daee33ae9cb511d8e2bb0da1ba34645fa`。
- ZIP：`artifacts/competition/group_norm_silu/s0-e6b450f/group_norm_silu.zip`，SHA-256 `9d6d79bfd0c1dc2a2bc5eb72cacb7d8c5f950aaca46c9d71ec9243e22fdff3a2`（单成员 `group_norm_silu.py`）。
- release 回执：`artifacts/competition/batch5-new5-validate-20260913/group_norm_silu/verification.json`，
  SHA-256 `563d574916eaef42da2edfb405e5472a01d705064052736e93cd9843d4221ec0`；日志 SHA-256 `4e85ba3445e569bf0584ce2e0be7a76c9f56e90a4b3e58eb8ebe487958d5ca1d`。

## 验证状态

- screening 多轮门禁拦下并修复的缺陷已记录于提交说明；
  最终 release：0 失败/错误/skip，非空 kernel launch，NVIDIA 代理范围。
- 所有八芯目标 `target-runtime-unverified`；裁决权在平台。

## 2026-09-13 E1：昆仑小 tile vendor（候选就绪后提交）

- S0 判决：七芯过（含燧原/华为），**昆仑 `OutOfResources: uni_sram`**
  ——[C,S] tile 超该栈 SRAM 预算（真实执行 7.5s 后报错，非崩溃族）。
- E1：`_kunlunxin` vendor 同 kernel，tile 上限 8192→2048 lane
  （BLOCK_S 下限 32，FlagGems 昆仑小 tile 注记）；generic 字节不动。
- source commit：`8b539234fc8d130dc8698ce68b57982e2000ad46`；ZIP `e1-8b53923`，
  SHA-256 `356e1012819dfc0bb4d553c569bb4d58300ef6d00bae76f594c09e3d8d25af09`。
- release 回执：`batch5-t72e1-validate-20260913/group_norm_silu/verification.json`，
  SHA-256 `1e430b40e9d733dfd25f3501d7d6878448929aedd98ca2c897ae762d4bedbc10`；
  4 方法 0 失败。
- submission 13774；裁决点=昆仑 uni_sram 解除。

## 2026-09-13 E1 平台终态：7/8（昆仑仍 uni_sram）

- 2048-lane cap 后昆仑仍 `OutOfResources: uni_sram`（exec 8611ms 真实
  执行）——2D [C,S] tile 形态本身超预算。E2 改 1D 形态（仅空间维
  lane，channel 维标量循环 + num_warps=1）。

## 2026-09-13 E2：纯 1D 昆仑 vendor（候选就绪后提交）

- 2048-lane cap 仍 uni_sram ⇒ 任何 2D tile 都超预算。E2 改纯 1D：统计
  循环扫全组（total 平铺，首轮 release 抓到只扫首 channel 的 bug 已修），
  归一化段 channel 标量循环 + 空间维 lane，num_warps=1。
- source commit：`0bce5d0b9ad5c7edca057e703614130c919cadb6`；ZIP `e2-0bce5d0`，
  SHA-256 `86e87f38d31d2048d050e6e7dc37fb4831ec8c779f9b57a345ab31eaf6a5dd25`；
  release 回执 SHA-256 `d5d1476c47f0ba3f1d3455e7d1d0d6a3656233c6fea030e7bfba676068efb087`。
- submission 13778；裁决点=昆仑 uni_sram 是否解除。

## 2026-09-13 E2 平台终态：7/8（昆仑崩溃族）

- 昆仑 exec 0ms 服务线程卡死——1D 嵌套循环形态与 T69 同族触发
  评测器崩溃（对照：s0/e1 的 2D 形态是真实 uni_sram 执行错，非崩溃）。
- 下一假设（E3，未开发）：2D 形态 + 更小 tile（512 lane）+ num_warps=1；
  或 1D 形态去嵌套（channel 展平进 spatial 一维）。

## 2026-09-13 E3：512-lane 2D tile + num_warps=1（已发射）

- E2 的 1D 形态触发间歇崩溃 → E3 回 2D 形态但 tile 压到 512 lane +
  num_warps=1（介于两个已知 uni_sram 失败点 8192/2048 之间）。
- source commit：`c6ac0ca43675efef3d339f77c0386b11753b1ced`；ZIP `e3-c6ac0ca`，
  SHA-256 `21a5ced869a6226f47aac06ca8b220135f36e448b19dc0f10143e757af81d4b3`；
  release 回执 `batch5-unlock4-20260913/group_norm_silu/verification.json`
  SHA-256 前缀 `c30415d8`；4 方法 0 失败。

## 2026-09-13 E3 平台终态：7/8（昆仑 uni_sram，512 lane 仍超）

- 昆仑 exec 10524ms 真实执行后 uni_sram——**8192/2048/512 三档全超**，
  该栈 uni_sram 预算极小。E4 方向：BLOCK_S=64（FlagGems 注记 ≤32
  miscompile,64 为下限上方）或查 FlagGems 昆仑 softmax 的实际 BLOCK。

## 2026-09-13 E4：BLOCK_S=64（已发射）

- 512 仍超 uni_sram ⇒ E4 直接降到 64 lane（FlagGems ≤32 miscompile
  下限上方）。
- source commit：`ef6270e3229672dfdabd3399d793fab4c35b9b85`；ZIP `e4-ef6270e`，
  SHA-256 `8d42e71140c0a3c39e6689ed7bba447452a00edc9daac7515ae3545ce359a162`；
  release 回执 SHA-256 `d9f42f277fb181ea7c91f316a116ed1747a9e0726f72ddb9d57743833e5c5737`。

## 2026-09-13 E4 平台终态：7/8（昆仑 uni_sram@64——64 lane 也超）

- **BLOCK_S=64 仍 uni_sram**（exec 7559ms 真实执行）——8192/2048/512/64
  四档全超。该栈 group_norm 类的 uni_sram 预算不是 tile 宽度问题,
  是**kernel 复杂度本身**（三遍循环+多个常驻向量）。T65 e4 BLOCK=64
  撞间歇崩溃（exec 0ms）同窗。
- 处置：T72 昆仑轴暂停 tile 降档(已到下限);重开条件=FlagGems 昆仑
  softmax/group_norm 的实际可行形态研究,或工单。
