# Task 72 `group_norm_silu` 实验记录

```current
task: 72
operator: group_norm_silu
batch: 5
validity: valid
platform: completed(15841,e9,8/8,2.48429167x;TB e6 2.66266667x)
candidate_stage: e9-completed
team_best_stage: e6
team_best_speedup: 2.66266667
sealed: no
next: e9华为1.1215<2.0门，小group驻留轴关闭；保留e6 TB，旧uncertain不重试
updated: 2026-09-16
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

## 2026-09-14 E5 候选就绪：昆仑 vendor 采用 master native_group_norm 骨架（待发射）

- 重新归因（FlagTree #1126 + FlagGems #6166 调研）：`uni_sram` 是
  make_ttxir 任意 PassManager 失败的统一包装，"四档 tile 全超"不成立；
  E2 的 1D 形态当时撞的是崩溃族而非本路径。
- 载体 = 弃 [C,S] 2D tile，照 master `_kunlunxin/ops/native_group_norm.py`
  骨架重写：扁平 1D 归约（[BLOCK_HW] 向量累加器 + 末尾单次 tl.sum，
  BLOCK_HW=min(next_pow2(spatial),1024)）+ GROUP_SIZE constexpr 逐
  channel 静态展开（标量 W/B + 连续 BLOCK_HW 块，无 idx//spatial
  gather）。两处刻意偏离 master：方差保持三遍中心化（大均值回归钉死）；
  silu 融入 normalize，fp32 全程、store 时才转输出 dtype。grid=
  (N*group,) 与 generic/master 同形。eps do_not_specialize。
- source / verification commit：`49a61251…`；ZIP `e5-49a6125`，
  SHA-256 `bc02ad082f3a09bd3841fe04046a7d82854fc6a49ddef8f4391049fb0b629cae`。
- release 回执 `batch5-submit-20260914/group_norm_silu/verification.json`
  SHA-256 `e1be532eb0cb0bd2d69fb8ac82b0b9f5866694b09b637678ae72659767f64601`
  （4 tests 0F0E0S 含 large_mean_small_var，generic 18 次真实 launch；
  昆仑 vendor target-runtime-unverified，裁决在平台）。
- 预注册晋级门：**昆仑通过（≥0.1 即 8/8）且七芯无回归**；判据=平台
  逐芯读数，崩退则回 S0 字节守七芯。

## 2026-09-14 E5 平台终态：8/8 VALID 2.5686x——昆仑解锁，任务首次有效（submission 14528）

- **预注册门全过**：昆仑 `group_norm_silu_kunlunxin.py` 被选中、编译
  通过、**passed speedup 0.502**（四轮 uni_sram 全败后首次过线）——
  master 骨架（扁平 1D 归约 + constexpr channel 循环 + BLOCK_HW=1024）
  证实可行，"uni_sram 预算"确为误导性包装（#1126 路径）。
- 逐芯：天数 4.1687 / 沐曦 2.4638 / 燧原 0.4392 / 海光 4.6098 /
  **昆仑 0.502** / 华为 1.163 / A 3.4537 / B 3.7487 → 均值 2.5686，
  任务从 invalid_correctness 变 valid（榜首 c2flow 3.024，差 0.46）。
- 七芯 generic 与 S0 同字节，读数均为窗口水位；燧原 0.439 与华为
  1.163 偏低是窗口与形态叠加，后续按逐芯榜单再定 vendor 轴。
- 额度：发后 24/30。

## 2026-09-14 E6 候选就绪：昆仑 padding 方差修复 + 华为 1D 骨架移植（Codex 激进菜单 B，待发射）

- **Codex 激进审查抓到 TB 载体的静态可证缺陷**：e5 昆仑 vendor 第二遍
  `sq_acc += d*d` 对 masked lane（x=0 → d=-mean）累进 mean²——
  num_elements 非整除 BLOCK_HW 时方差被高估。平台 benchmark shape
  整除未触发；本发修复（`tl.where(m, d*d, 0)` 算术掩码）。
- 新增 _ascend vendor：同一 1D 扁平骨架移植到 Ascend 规则集
  （BLOCK_HW≤1024、扁平 1D 满足 #1610 归约轴对 lane 轴、算术掩码、
  fp32 silu、eps do_not_specialize），华为现读 1.16 vs c2flow 4.37。
- source / verification commit：`ea4bfef4…`；ZIP `e6-ea4bfef`，
  SHA-256 `617a38e0275173110618ad27343fdab99efa2ea8cd1294a48fde435f24545ac2`。
- release 回执 `batch5-submit-20260914/group_norm_silu_e6/verification.json`
  SHA-256 `ef6aea03ae2087d078779b5f3cbf7af0810ace68391640b54e8945c5d47ca8d5`
  （--proxy-vendor ascend/kunlunxin：两 vendor 各 18 launch 0F0E0S，
  修复后字节全矩阵通过）。
- 预注册晋级门：昆仑保持 ≥0.1（修复不应回归）；**华为 ≥2.0 为继续
  研发门**（升位目标 ≥4.8）；均值不低于 2.56 TB 水位减窗口余量。

## 2026-09-14 E6 平台终态：8/8 VALID 2.6627x 新TB（submission 14726）

- **华为 ascend vendor 被选中、passed 1.1958**（generic 1.163，+3% 噪声）
  ——门（≥2.0）未过，1D 骨架移植在华为无增益，按 Codex 规则停止
  （无新瓶颈证据不再发）。c2flow 的 4.37 形态仍未破译。
- **昆仑修复载体 0.491**（e5 的 0.502，噪声内）——padding 方差修复
  正确性中性确认，TB 载体 hardened。
- 其余芯窗口上行（muxi 2.96/2.46 +20%、海光 4.74/4.61、A 3.65/3.45、
  B 3.74/3.75；天数 4.15/4.17、燧原 0.376/0.439）→ 均值 2.6627
  > TB e5 2.5686，新 TB = e6。#4/6，榜首 Evoke 3.9446，差 1.28。

## 2026-09-14 E7 候选就绪：昆仑 BLOCK_HW 1024→2048（新骨架首档加宽，待发射）

- 1D 扁平骨架（e5 解锁形态）的首个宽度档；代理 kunlunxin vendor
  36 launch 0F0E0S。门：**昆仑 ≥ 0.65**；未过回 1024 守。
- source / verification commit：`6c68935b…`；ZIP `e7-6c68935`，
  SHA-256 `6423c7a6ffb4a0b813acc4f8558c55933012f0ef65f65129fe1fbebc93a0d5be`。

## 2026-09-14 E7 平台终态：8/8 VALID 2.5626x < TB（submission 14845）

- **昆仑 vendor 0.525**（e6 的 0.491 → +7%）——2048 档微正但未过门
  （≥0.65）；其余芯窗口回落（沐曦 2.44/2.96、海光 4.48/4.74），
  均值 2.5626 < TB e6 2.6627。宽度轴在该 skeleton 上边际小，回 1024
  守 TB；不再加宽。

## 2026-09-14 E8 候选就绪：昆仑 BLOCK_HW 2048→4096（今日最后一发，待发射）

- 昆仑宽度轴今日全正（T73 4096 +23%、T75 大涨、T72 2048 +7%）；
  4096 为本 skeleton 未试档。代理 kunlunxin 全矩阵 0F0E0S。
  门：**昆仑 ≥ 0.65**；未过回 1024 守。
- source / verification commit：`a9b8444…`；ZIP `e8-a9b8444`，
  SHA-256 `4448588ea36b75403d4ea9f4f4ce484887746444a56b88db394e469965254677`。
- release 回执 `batch5-submit-20260914-finale2/group_norm_silu-t72e8/verification.json`
  SHA-256 `6fc9d812b9014479f36c945d8f3daa53bfc0e80b5b87e4522d1933e5fca21aeb`。

## 2026-09-14 E8 平台终态：8/8 VALID 2.5400x < TB（submission 14859）——4096 档回落

- **昆仑 vendor 0.5452**（2048 的 0.525 → +4%，但 1024 的 0.491 →
  +11% 峰值在中间）——宽度轴在本 skeleton 呈抛物线，4096 过头；
  未过门（≥0.65）。均值 2.5400 < TB e6 2.6627。
- TB 载体最终=e6（BLOCK_HW 1024+padding 修复+ascend 移植字节）；
  昆仑宽度轴收官：1024（0.491）→2048（0.525）→4096（0.545），
  边际递减，明日若再试唯一合理档=2048 与均值门联评。

## 2026-09-16 00:06 E7 uncertain：submit 阶段客户端异常，POST 未达

- _ascend vendor num_warps=8 探针（ce3d59c）验证通过、preflight 过、
  submit 阶段 platform_cli 异常退出，intent 停在 uncertain；
  task 72 平台无今日记录、额度未扣（used 仅计他题）。处置待明早：
  注释载体新 ZIP 重发（同 T75 e7r 模式），或放弃低优先探针。

## 2026-09-16 E9：小 group 一次读取，寄存器内中心化方差

- 原因：旧Ascend形态三遍全局读；小group可一次读取后计算均值、中心化方差、affine和SiLU。复用FlagGems固定 `a7620cc191a0b42e040194622c5758b22a7a25dc` `src/flag_gems/ops/groupnorm.py:31-86` 成熟驻留结构，修正非2幂group channel mask。
- 单变量比较基线=e6 `ea4bfef4`。本次恢复HEAD中昆仑4096→TB1024、去掉旧uncertain Ascend warps8漂移，其他路径回到TB；新变化仅Ascend padded group≤2048走驻留。大组保留原centered三遍，绝不改为E[x²]-mean²。
- 上一节0.491→0.525→0.545是单调递增、边际递减，不是“抛物线/回落”；未过0.65门才是封轴依据。旧warps8 uncertain intent保留不动，E9是有实际结构差的新候选。
- source commit `26fe22d9a1d0d22454723f34bbbdef9f97ea4a02`，测试补2047/2048/2049、非2幂group_channels、大均值100；Black25.12/isort/flake8/py_compile通过。
- screening完整5/5、generic/ascend各34launch、0失败/skip；18桶×5 AB/BA，12affected中11桶≥1.15x。Cg20约1.97–2.13x，256elements约1.18–1.20x，Cg3约1.18–1.23x，2048 fp16/bf16约1.29–1.37x；2048 fp32约1.039x。6 controls 0.9936–1.0027，全部0spill，最多54regs/16B共享内存。
- 原始screening `artifacts/competition/t72e9-screening-20260916/`，benchmark SHA-256 `77b0b1fef1edbbc24d3f52b9008cee1d8c11bae10244da117e2853a4f16d5e9f`；NVIDIA代理不代表Ascend性能。
- 平台预注册：8/8且每芯≥0.1，华为≥2.0及均值>2.66266667才继续优化；华为8.414833是他队已观测目标，单芯达到它仍不足夺第一。未达2.0则关本小group轴，不重复投。

### E9 不可变发布证据

- source / verification commit `26fe22d9a1d0d22454723f34bbbdef9f97ea4a02`；ledger commit 为本节所属提交。测试SHA-256 `6e1eacafff5875a630b6cc0325085b2398532cf58e5d6852cfdcc647ffe288aa`。
- exact release `artifacts/competition/t72e9-release-20260916/verification.json` SHA-256 `621cab2a951251992700ee458f6fbe8d2dadd4e3873754b37ad47d12669e129d`；日志SHA-256 `0b263914e4bcbe14f893bc705a93eefaacc6cacf9d0005740e4dde038f00457e`。5/5、generic/ascend/kunlunxin各34次kernel launch、无fail/error/skip；2047/2048/2049、Cg3和大均值均实际执行。
- ZIP `/Users/bytedance/ccc/flagos/artifacts/competition/group_norm_silu/e9-26fe22d/group_norm_silu.zip`，14301 bytes，SHA-256 `ae8277fded564b3b1b7a2f42ca7e455d1e9856a5f205ead5cda65022aea6445e`。
- 成员 `group_norm_silu.py` SHA-256 `690aa6bd15ed593fa76f8b412457aa9d4a1cec4d18d8ae7f0519b213f50bc229`。
- 成员 `group_norm_silu_ascend.py` SHA-256 `85756844d064365b8d9b9135f13dcc2eeba58a1182d8220ee841685e36e4e377`。
- 成员 `group_norm_silu_kunlunxin.py` SHA-256 `1e69e32091b08fe221304a3ab768f04a9af7e7a70e69cd62b3562edf35739672`。
- 两vendor在NVIDIA代理执行，target-runtime-unverified；平台验证补齐，不与代理证据混用。

### E9 平台终态：8/8 valid，目标收益未迁移

- 2026-09-16 11:11:52单次提交，nonce `b15b70ed1b3a808dba141019d5234708`，submission **15841**；上传远端回读14301 bytes及SHA-256与immutable ZIP一致。未重试旧uncertain。
- 11:12:52状态：8/8 valid，均值 **2.48429167x** < TB E6 **2.66266667x**。华为选中 `group_norm_silu_ascend.py`，**1.1215 < 2.0**，预注册继续门失败；代理小group收益未迁移，关闭这一轴，不复投。
- 逐芯：天数3.85983333、沐曦2.45116667、燧原0.43833333、海光4.235、昆仑0.4915、华为1.1215、A3.50983333、B3.76716667。冻结路径的读数变化不能归因为新Ascend代码。
- 原始preflight/submit/status：`artifacts/competition/top1-20260916/t72-e9-{preflight,submit,status}.json`。实时额度26/30，平台最佳仍E6；需要新的目标芯瓶颈证据才重开。
