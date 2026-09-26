# Task 101 `post_reorder_deepgemm` 实验记录

```current
task: 101
operator: post_reorder_deepgemm
batch: 7
validity: valid
platform: e1(21509)valid avg11.63934286;海光20.61/card_a10.79/天数12.12/沐曦6.34/昆仑11.45/华为12.39/card_b7.78;快照rank4(OpeGoodn13.24于09-26T07:34反超,night-fire时11.50疑r1已过时);T65-E10同构阶梯全面兑现
candidate_stage: e2
team_best_stage: e1
team_best_speedup: 见platform行
sealed: no
next: E2(华为_ascend两段式)已上膛待发射;门:数值败或华为<11.77回退删vendor/>=15保留/>=18判轴;均值>11.63934286换TB
updated: 2026-09-26
```

## 不可变身份（s0，2026-09-25 第二批释放夜）

- 四题同批提交（21439/21440/21442/21443），review 5 项 P2 全修后门禁全绿。
- release 回执：`artifacts/competition/b7b-s0-release-20260925/post_reorder_deepgemm/verification.json`。
- ZIP：`artifacts/competition/post_reorder_deepgemm/s0-*/post_reorder_deepgemm.zip`（T102 绑定 2cad7b0d，其余 f5b7add4）。

## E1 launch 结构重排 + BLOCK 阶梯（2026-09-26 上膛，armed-unfired）

- **假设**：s0 的 `next` 行指明 E1 轴=天数#4/沐曦#4/A,B#5 的 gather 带宽
  （BLOCK/warps 微调）。gather 端改为平台已验证的同构结构：T65
  deepep_post_reorder E10（read +67.16% mean：tianshu x1.80 / muxi x1.49 /
  haiguang x1.96 / card_a x1.86 / card_b x1.54）——hidden 块从每 program
  串行 static_range 移到 `grid.y` 维（`gy=min(cdiv(hdim,2048),255)`、
  `gx=min(rows,65535//gy)` 保证 grid 乘积 ≤65535），BLOCK 1024→2048、
  num_warps=8，hidden 走 `tl.range` 双轴跨步。算术形态逐位保持 s0：TOPK
  static 展开、标量槽位读、clamp+keep 门、fp32 累加、scale 折进 store、
  stride 全参数化。昆仑跑**冻结 s0 vendor**（宽 BLOCK 在昆仑是编译期
  SIGABRT 前科，见 deepep_post_reorder 账本）。
- **实现**（source commit `0f9cf744d239b302e029ee53ae653986b5dabe2d`）：
  generic `src/flaggems_sglang/ops/post_reorder_deepgemm.py` 重排 launch +
  BLOCK 阶梯；新增 `runtime/backend/_kunlunxin/ops/post_reorder_deepgemm.py`
  = s0 generic 字节（成员 sha256 `c14b806e…` 与 s0 回执 generic 哈希相同，
  冻结佐证）；回归新增 `test_hidden_program_boundaries`（2047/2048/2049/
  4095/4096/4097/7168/255*2048-1/+1）、`test_token_program_boundaries`
  （65535/65536/65537 过 grid-stride 行环）、`test_grid_product_boundary`
  （rows=32768×hidden=2049 尾行仅 grid-stride 可达）与 CPU 元数据
  `PostReorderDeepgemmGridTest.test_grid_product_cap`（捕获 launch 维度、
  乘积 ≤65535、generic 断言 (32767,2)/(257,255)/(257,255)），8 项全部列入
  RELEASE_REQUIRED_TESTS。
- **代理证据（NVIDIA RTX 5070 Ti / torch 2.13.0+cu130 / triton 3.7.1）**：
  release v2 回执（source=verification commit `0f9cf744`，mode=release，
  `--proxy-vendor kunlunxin`）：**8/8 RELEASE_REQUIRED 全过**，0 fail/0
  error/0 skip，70 case（含 generic+kunlunxin 双模块 subTest）；入口调用
  generic 23 / kunlunxin 23，实际 kernel launch generic 20 / kunlunxin 20
  （23-20=3 为 grid 捕获用例的 mock 拦截，不计 launch）；非空张量 shape
  122 条，E1 新边界 [1,2047]/[1,2048]/[1,2049]/[1,7168]/[1,255*2048±1] 与
  65535+ token 形状均真实 JIT 执行。回执
  `artifacts/competition/day5prep-20260921/E1-hidden-grid-BLOCK2048-wf/verification.json`
  / `verification.log`（log sha256
  `889f500572a4eacc9ae364db94a8e0f5b42bc7752a7e0f7b1eb6a92fea790547`，
  远端 `Ran 8 tests in 225.865s OK`，exit 0）。kunlunxin vendor 在
  target_unverified_sources 中——代理通过不构成昆仑芯背书，昆仑正确性由
  s0 平台 11.426 逐芯通过 + 字节冻结保证。
- **五元组（上膛身份）**：
  - source_commit：`0f9cf744d239b302e029ee53ae653986b5dabe2d`
  - verification_commit：`0f9cf744d239b302e029ee53ae653986b5dabe2d`（=本回执）
  - ledger_commit：本 commit（E1 上膛记账）
  - ZIP：`artifacts/competition/post_reorder_deepgemm/e1-0f9cf74/post_reorder_deepgemm.zip`
    （7480 字节，zip_sha256 =
    `67f83c2ff57ba2b0f29563091c9779b9de15683f0c48689425b700cb1eb6831a` =
    canonical_zip_sha256；≠ s0 `515f1337…` 且成员 1→2，平台去重键成立）
  - stage：`e1`（CURRENT candidate_stage s0→e1 递增）
  - ZIP 成员名单（与 `zipfile.namelist()` 实际核对一致，无夹带；`unzip -t`
    无错；均为 UTF-8 `.py`，无测试/缓存/目录前缀/macOS 垃圾；成员字节 =
    git blob @0f9cf744 = 远端实际执行字节）：
    | 成员 | 字节 | sha256 |
    |---|---|---|
    | `post_reorder_deepgemm.py` | 3954 | `b405f3cf8c8d25f67208372ea1a5099631f785290bf91c5a50738af2ad889827` |
    | `post_reorder_deepgemm_kunlunxin.py` | 3236 | `c14b806eb825cc11db35f71d1a59128ce89790e60614061025260e2bc579d879` |
- **预注册各芯预期签名**（s0 基线 = climb-loop s2t1op101 快照 sub 21443，
  队均值 9.97257143；T65-E10 为同构机制锚）：
  | 芯 | s0 | 机制预期 | 兑现/回退判读 |
  |---|---|---|---|
  | 天数 tianshu | 10.9936 | ↑ 主靶（T65-E10 x1.80 同构） | ≥+5% 判轴兑现 |
  | 沐曦 muxi | 5.6906 | ↑ 主靶（T65-E10 x1.49 同构） | ≥+5% 判轴兑现 |
  | card_a | 7.2636 | ↑ 主靶（T65-E10 x1.86 同构） | ≥+5% 判轴兑现 |
  | card_b | 7.2182 | ↑ 主靶（T65-E10 x1.54 同构） | ≥+5% 判轴兑现 |
  | 昆仑 kunlunxin | 11.426 | 中性（冻结 s0 字节，无新风险面） | 噪声带内即过，-5% 即异常排查 |
  | 华为 huawei | 12.2534 | 中性~↑（无芯专属证据） | 噪声带内即过 |
  | 海光 haiguang | 14.9626 | 中性~↑（T65-E10 x1.96 同构，非主靶） | 噪声带内即过 |
  - **晋级门（预注册）**：任一芯 -5% 回退触发回滚（generic 回 s0 字节，
    昆仑 vendor 本就冻结不动）；平台均值 >11.10205714（榜首 c2flow）换
    TB；四主靶至少两芯 ≥+5% 才判 E1 轴兑现并沿同轴继续 E2（否则回到
    BLOCK/warps 微调或换轴）。本节之前无该轴任何平台读数；门与签名在
    平台提交前落账，读数后不得改判据。

## E2 _ascend vendor 华为两段式 hidden 循环（2026-09-26 上膛，armed-unfired）

- **假设**：E1 平台读数后华为是我队最大的固定点缺口——climb-loop
  s2t1op101 快照我队 huawei 12.3874（r4）vs 榜首 OpeGoodn 22.567（r1）
  -45.1%。T93 E3 已在平台验证同构两段式结构（full hidden 块
  no-mask/no-other load/store，仅 tail 块保 masked load without other +
  same-mask store，uniform per-iteration scalar branch `h0+BLOCK<=hdim`，
  热路径零 int32 Vector CMP、零 MTE2 prefill），华为 +34% 兑现（night-fire
  `7ee1f7a6` 落账）。反证披露：T92 E23（华芯同构尝试 -28.8%，差异面=
  该题 row 形态与本题 gather 形态不同）；T65 持续 -19.3% 已证伪不采用。
- **实现**（vendor-only，generic 与 _kunlunxin 字节冻结不动）：candidate
  commit `fc047598`，新增
  `src/flaggems_sglang/runtime/backend/_ascend/ops/post_reorder_deepgemm.py`
  ——两段式 hidden `tl.range` 自 T93 E3 模板迁移；其余字节（TOPK static
  展开、标量槽位读、clamp+keep 门、fp32 累加、scale 折 store、stride 全
  参数化、grid.y 重排、BLOCK=2048/warps=8）全部自 E1 保留。回归新增
  `test_two_segment_hidden_semantics`（full+tail 混合落同一 grid-stride
  program，含 `255*2048+2049`/`+4097` 强制第二次 tl.range 遍、padded
  slot 双臂），RELEASE_REQUIRED_TESTS 8→9。
- **代理证据（NVIDIA RTX 5070 Ti / torch 2.13.0+cu130 / triton 3.7.1 /
  python 3.12.13）**：release v2 回执（source=verification commit
  `87d0e627d74c8ebbc8c2a65eeda136093fbedc44`，mode=release，
  `--proxy-vendor ascend --proxy-vendor kunlunxin`）：**9/9
  RELEASE_REQUIRED 全过**（expected=passed=9），0 fail/0 error/0 skip，
  110 case；入口调用 generic/_ascend/_kunlunxin 各 27，实际 kernel
  launch 各 24（27-24=3 为 grid 捕获 mock 拦截，不计 launch）；非空
  张量 shape 225 条；远端 `Ran 9 tests in 159.539s OK`，exit 0（log
  sha256
  `bba1c0aec0496427f0e31139a589a888612c28aa10fff3a4568bd10426585406`）。
  回执
  `artifacts/competition/day5prep-20260921/post_reorder_deepgemm-wf/verification.json`
  / `verification.log`。_ascend 与 _kunlunxin 均在
  target_unverified_sources——NVIDIA 代理不构成华为/昆仑背书：华为正确性
  由 E1 平台 12.3874 逐芯通过 + T93 E3 同构模板平台已验证佐证；昆仑由
  s0 平台通过 + 字节冻结（成员 sha256 与 e1 ZIP 相同）保证。
- **五元组（上膛身份）**：
  - source_commit：`87d0e627d74c8ebbc8c2a65eeda136093fbedc44`
  - verification_commit：`87d0e627d74c8ebbc8c2a65eeda136093fbedc44`（=本回执）
  - ledger_commit：本 commit（E2 上膛记账）
  - ZIP：`artifacts/competition/post_reorder_deepgemm/e2-87d0e62/post_reorder_deepgemm.zip`
    （15824 字节，zip_sha256 =
    `c53082a0f29ee8744911b7758c993772274bbdafd814395da1384f468554251f` =
    canonical_zip_sha256；≠ e1 `67f83c2f…` 且成员 2→3，平台去重键
    zip_sha256 成立）
  - stage：`e2`（CURRENT candidate_stage e1→e2 递增）
  - ZIP 成员名单（与 `zipfile.namelist()` 实际核对一致，无夹带；`unzip -t`
    无错；均为 UTF-8 `.py`，无测试/缓存/目录前缀/macOS 垃圾；成员字节 =
    git blob @87d0e627 = 远端实际执行字节）：
    | 成员 | 字节 | sha256 |
    |---|---|---|
    | `post_reorder_deepgemm.py` | 3954 | `b405f3cf8c8d25f67208372ea1a5099631f785290bf91c5a50738af2ad889827`（=E1 字节，vendor-only 变更佐证） |
    | `post_reorder_deepgemm_ascend.py` | 8206 | `607a80b46ec5511a2f93596f958c099e0c299ef2251a01c0f897df699adcd0bf`（E2 新增，=回执执行字节） |
    | `post_reorder_deepgemm_kunlunxin.py` | 3236 | `c14b806eb825cc11db35f71d1a59128ce89790e60614061025260e2bc579d879`（=e1/s0 冻结字节） |
- **预注册各芯预期签名**（e1 平台基线 = climb-loop s2t1op101 快照
  is_mine 行，sub 21509，avg 11.63934286；E2 相对 e1 的唯一读数字节差异
  = 华为芯从 generic(E1) 切读 _ascend vendor，其余 6 芯字节不变）：
  | 芯 | e1 平台 | 机制预期 | 兑现/回退判读 |
  |---|---|---|---|
  | 华为 huawei | 12.3874 | ↑ 唯一主靶（T93 E3 同构 +34% 平台已验证；OpeGoodn 22.567 差 -45.1%） | ≥15 保留 / ≥18 判轴确认 |
  | 天数 tianshu | 12.1168 | 中性（读 generic E1 字节不变） | 噪声带内即过，-5% 异常排查 |
  | 沐曦 muxi | 6.342 | 中性（同上） | 同上 |
  | 海光 haiguang | 20.6092 | 中性（同上） | 同上 |
  | 昆仑 kunlunxin | 11.453 | 中性（冻结 s0 字节） | 同上 |
  | card_a | 10.7918 | 中性（同上） | 同上 |
  | card_b | 7.7752 | 中性（同上） | 同上 |
  - **晋级门（预注册，随 candidate commit `fc047598` 先于回执落档）**：
    数值失败或华为 <11.77（-5%）→ 删除 `_ascend` vendor、恢复 E1 字节
    （generic 本就未动）；华为 ≥15 保留 / ≥18 判轴确认；平台均值
    >11.63934286（我队当前 TB）换 TB；其余 6 芯字节未变，任一 -5% 只可能是
    噪声/平台漂移，触发异常排查而非结构回退。本节之前无该轴任何平台读数；
    门与签名在平台提交前落账，读数后不得改判据。
