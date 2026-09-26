# Task 101 `post_reorder_deepgemm` 实验记录

```current
task: 101
operator: post_reorder_deepgemm
batch: 7
validity: valid
platform: e3(21627)valid 7/7 avg9.74534286(-16.27% vs e1 TB);天数9.306(-23.20%)/沐曦5.3242(-16.05%)/海光12.4122(-39.77%)/card_a9.607(-10.98%)四rider芯破e1基线-5%线(card_b7.46,-4.05%未单独破)/昆仑11.3878(-0.57%,冻结字节)/华为12.7202(+0.003% vs e2,_ascend实跑,≥12.08不回滚);TB仍e1(21509)avg11.63934286;is_team_best=False/ranking0.71112221(16:31复核)
candidate_stage: e3(fired,读数落账见E3平台终态节)
team_best_stage: e1
team_best_speedup: 见platform行
sealed: no
next: 预注册generic-rider回滚门已触发(四芯破-5%):按门文本generic回滚e1字节(blob b405f3cf@0f9cf744);天数exec227983ms vs e2同芯152283ms慢窗未排除(海光读数腰斩但exec仅+21%),窗口归因与回滚执行归orchestrator
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

## E2 平台终态（2026-09-26 发射，sub 21551）：valid 7/7 avg 11.56768571——华为 +2.68% 落未申报中间带，TB 守 e1

- **发射记录（单 preflight + 单 submit）**：一次 preflight（nonce
  `e26efa416acab32a6e04906fff223a99`，tuple 与上膛五元组全匹配：commit
  `87d0e627` / zip `c53082a0…` / 3 成员 / test-sha `3f5e02a0…` / receipt
  `579feb0c…`）+ 一次 submit --confirm → state=submitted，submission_id
  **21551**，daily_seq 11；watch 绑定 file_url_sha256 `8cdc7d30…`、
  after-epoch 1790366407。无 uncertain/sending 状态。
- **终态（落账员独立复核：`platform_cli.py status --race 782kzq4m
  --batch 7 --task 101`，observed 2026-09-26T11:07）**：status=completed，
  validity=valid，**7/7 GPU passed**（raw_result failed_cases 全空），
  average_speedup **11.56768571**（vs e1 11.63934286 = **-0.62%**）。
- **逐芯（e2 21551 vs e1 21509；selected_file 佐证字节实跑）**：
  | 芯 | e2 | e1 | Δ | selected_file |
  |---|---|---|---|---|
  | 华为 huawei | **12.7198** | 12.3874 | **+2.68%** | `post_reorder_deepgemm_ascend.py`（vendor 实跑；exec 39239ms vs e1 39378ms） |
  | 天数 tianshu | 11.9694 | 12.1168 | -1.22% | generic（E1 字节） |
  | 沐曦 muxi | 6.4378 | 6.342 | +1.51% | generic（E1 字节） |
  | 海光 haiguang | 19.8214 | 20.6092 | -3.82% | generic（E1 字节） |
  | 昆仑 kunlunxin | 11.5302 | 11.453 | +0.67% | `_kunlunxin`（e1/s0 冻结字节） |
  | card_a | 10.8056 | 10.7918 | +0.13% | generic（E1 字节） |
  | card_b | 7.6896 | 7.7752 | -1.10% | generic（E1 字节） |
- **预注册门判决（判据=上节原文，读数后未改）**：
  1. 数值失败未发生（7/7 passed）→ 不触发删 vendor；
  2. 华为 12.7198 ≥11.77 回滚下限（不删）但 <15 保留线、<18 判轴线 →
     落在**预注册未申报的中间带 [11.77,15)**，轴**未确认**；
  3. 均值 11.56768571 ≤11.63934286 → **TB 不换**，e1(21509) 仍为我队
     按均值最优；
  4. 六个字节未变芯最大跌幅海光 -3.82%，未破 -5% 异常排查线。
- **平台动态字段双记（两次读数均如实保留，不互改）**：发射侧最终轮询记
  21551 `is_team_best=true / ranking_score 1.98333333`；本轮 11:07 status
  复核记 `is_team_best=False / ranking_score 1.41666667`（榜单相对动态
  字段，随榜单变动）。
- **额度**：发射后 19/30 剩余；本轮 11:07 复核时 17/30（已含同窗
  T104/T97 各一发）。
- **遗留（orchestrator 待决）**：预注册门对 [11.77,15) 未申报动作——沿轴
  继续（e3）或删 vendor 收敛 generic 均不违规，本账本不擅自判决。

## E3 候选记录（2026-09-26 评审 r1/r2 修订，源就绪未上膛，armed 前预注册）

- **假设**：e2 落 [11.77,15) 中间带后沿轴继续——slot-skip + num_stages
  移植上游 SGLang main 主形态到 E1/E2 几何（grid.y hidden / BLOCK2048 /
  w8 / 两段式臂保留），主靶华为（-17.7304 vs 金狐狸 30.1178，占毛缺口
  72.0%）。无效槽（topk_ids<0）整槽跳过：`if eid >= 0` 标量分支包住
  dst/weight 标量读与整块 gather，padding 槽不再白付 gather+2 标量读+1
  乘法；门保持题面 reference 语义（topk_ids>=0 且 clamp 保留——上游
  `if dst_idx>=0` 门会丢掉 eid>=0∧dst=-1 的 clamped 贡献，由
  `test_slot_skip_reference_gate` 钉死）。行（token）grid-stride 循环改
  `tl.range(..., num_stages=3)` 软件流水（上游 ep_moe_kernels.py@
  5f6dd44 实读核验：`tl.range(start, num_tokens, step,
  num_stages=NUM_STAGES)`，NUM_STAGES=3，weight+gather 包在标量分支内）。
  generic 改动同时作用于现跑 generic 的 rider 芯（e2 逐芯
  selected_file=generic：天数/沐曦/海光/card_a/card_b）。
- **实现状态**：generic 与 _ascend vendor（两段式双臂）同改，
  _kunlunxin 冻结 s0 字节不动。r1 commit `4e817d95`（实现+回归
  RELEASE_REQUIRED_TESTS 9→11）；r2 commit（本 commit）=评审两项 P2
  处置：P2-1 非有限分歧落账为预注册有意接受、P2-2 generic 数值失败门
  补进两处 docstring+本账本。**未建 ZIP、无 release 回执、未消耗额度**；
  上膛时另立五元组（candidate_stage 届时 e2→e3 递增）。
- **评审 P2-1 判决（预注册，armed 前锁定）**：literal reference 对无效槽
  计算 `down[clamp(-1)]*(w*0)`，down[0] 或 w 非有限时 0×NaN/Inf 传播
  NaN；slot-skip 分支输出 0。该分歧为**有意接受**：题面与 skill/参考
  文档均无 harness 数据生成器书面契约（"randn/有限"系未验证的开发者
  假设，评审已检索核实），randn 族数据下两语义逐位一致；同一提交内
  _kunlunxin 冻结 vendor 仍 NaN 传播——同接口跨后端在非有限无效槽数据
  上不一致同样有意接受（昆仑字节不因 e3 解冻）。**护栏**：任何芯平台
  数值失败即触发下述回滚门，不靠 docstring 单独披露；回归钉在
  `test_slot_skip_nonfinite_semantics`（`SLOT_SKIP_SEMANTICS` 模块标记
  分派期望：generic/_ascend 断言 skip 语义，冻结 kunlunxin 按 literal
  reference + equal_nan 校验）。
- **评审 P2-2 判决（预注册，armed 前锁定）**：generic 臂数值失败门原缺、
  现补死——**任一 generic-rider 芯（天数/沐曦/海光/card_a/card_b）数值
  失败或编译失败 → generic 回滚 e1 字节**（e1 generic blob sha256
  `b405f3cf8c8d25f67208372ea1a5099631f785290bf91c5a50738af2ad889827`
  @0f9cf744）。风险披露：循环级 `tl.range(num_stages=)` 在仓库内仅
  _enflame vendor 有平台兑现先例（group_norm_silu E12 sub 16770 valid，
  且在燧原读数为性能 no-op：0.45866667 vs E11 0.459；本仓 grep 核验
  tl.range+num_stages 仅命中 _enflame 三文件与本候选两文件），
  muxi/haiguang/card_a/card_b 四个 Triton fork 从未执行过该 kwarg——
  任一 fork 编译/数值失败将按 T104 e2 先例使整发 invalid_correctness
  （烧一发额度）；NVIDIA 代理只做灾难门。评审给出的拆分选项（先发
  slot-skip A、正信号再叠 num_stages B）已评估**不采用**：一次发射同时
  取抢榜与 [11.77,15) 处置的信息收益，两风险均已预注册字节级回滚路径。
- **晋级门（预注册，读数前锁定，读数后不得改判据）**：
  1. 华为数值失败或 <12.08（e2 12.7198 的 -5%）→ _ascend 回滚 e2 字节
     （vendor blob sha256
     `607a80b46ec5511a2f93596f958c099e0c299ef2251a01c0f897df699adcd0bf`
     @87d0e627）；
  2. 任一 generic-rider 芯数值失败或较 e1 逐芯基线（天数 12.1168/沐曦
     6.342/海光 20.6092/card_a 10.7918/card_b 7.7752）-5% → generic
     回滚 e1 字节；
  3. 平台均值 >11.63934286 → 换 TB（TB 自 e1 21509）；
  4. 昆仑字节未变（冻结），读数异常只排查不回滚。
- **回归矩阵增量**（r1 已入 RELEASE_REQUIRED_TESTS，9→11）：
  `test_slot_skip_reference_gate`（全无效行精确 0、eid≥0∧dst=-1 clamped
  贡献钉 reference 门防 dst 门误植、65537 行 pipelined 第二遍
  grid-stride 含活跃 skip 分支）、`test_slot_skip_nonfinite_semantics`
  （非有限分歧按 SLOT_SKIP_SEMANTICS 分派期望）。

## E3 上膛（2026-09-26，armed-unfired）

- **候选**：slot-skip + num_stages（假设、实现、评审 P2-1/P2-2 判决与
  预注册门全部锁定于上文 E3 候选记录节，本节不复述判据——读数后按该节
  原文判读，不得改判据）。generic 与 `_ascend` vendor 为 r1 `4e817d95`
  + r2 `b0b927f0` 字节；`_kunlunxin` 冻结 s0 字节不动。
- **代理证据（NVIDIA RTX 5070 Ti / torch 2.13.0+cu130 / triton 3.7.1 /
  python 3.12.13）**：release v2 回执（source=verification commit
  `ffe2b99e75b6d70da61d58d7159c81ba6588c389`，mode=release，
  `--proxy-vendor ascend --proxy-vendor kunlunxin`）：**11/11
  RELEASE_REQUIRED 全过**（expected=passed=11，含
  `test_slot_skip_reference_gate` / `test_slot_skip_nonfinite_semantics`
  两个新回归），0 fail/0 error/0 skip，121 case；入口调用与实际 kernel
  launch generic/_ascend/_kunlunxin 各 30/27（30-27=3 为 grid 捕获
  mock 拦截，与 E1/E2 同口径）；非空张量 shape 249 条；远端
  `Ran 11 tests in 2.626s OK`，exit 0（log sha256
  `7b4886ebe26e600f998ac01baaef4c6f93befa52d00e9e664b20259fce5ab4cf`，
  与本地回执 log 逐字节重算一致）。**壁钟披露**：2.626s 显著快于 E1
  225.865s / E2 159.539s；kernel launch 与 shape 计数由 runner 从实际
  执行 hook 记录（非零、三源对称），执行真实性以回执为准；两数量级差
  推测为远端 triton 编译缓存已热——评审期同字节是否曾远端执行无法从
  本会话证据排除，如实留痕。回执
  `artifacts/competition/day5prep-20260921/post_reorder_deepgemm e3 slot-skip+num_stages-wf/verification.json`
  / `verification.log`。_ascend 与 _kunlunxin 均在
  target_unverified_sources——NVIDIA 代理不构成华为/昆仑背书：华为由
  e2 平台 12.7198 逐芯通过 + 本候选同构几何延续佐证；昆仑由 s0 平台
  通过 + 字节冻结（成员 sha256 与 e1/e2 ZIP 相同）保证。
- **五元组（上膛身份）**：
  - source_commit：`ffe2b99e75b6d70da61d58d7159c81ba6588c389`
  - verification_commit：`ffe2b99e75b6d70da61d58d7159c81ba6588c389`（=本回执）
  - ledger_commit：本 commit（E3 上膛记账）
  - ZIP：`artifacts/competition/post_reorder_deepgemm/e3-ffe2b99/post_reorder_deepgemm.zip`
    （20937 字节，zip_sha256 =
    `46cfc990496cf8e386be57e1f1dce07f71bd6ac909fb700ff2d43a7fc6377489` =
    canonical_zip_sha256；≠ e2 `c53082a0…` 且成员字节数全变（generic
    3954→6325、ascend 8206→10948、kunlunxin 3236 冻结），平台去重键
    zip_sha256 成立）
  - stage：`e3`（CURRENT candidate_stage e2→e3 递增）
  - ZIP 成员名单（与 `zipfile.namelist()` 实际核对一致，无夹带；`unzip -t`
    无错；均为 UTF-8 `.py`，无测试/缓存/目录前缀/macOS 垃圾；逐成员
    sha256 = 打包器申报 = git blob @ffe2b99e = 回执 files 哈希 = 远端实际
    执行字节）：
    | 成员 | 字节 | sha256 |
    |---|---|---|
    | `post_reorder_deepgemm.py` | 6325 | `933499b326616f3e993fb591edbefa5b29d7531f0567b554db2050af913dcbdc`（E3 slot-skip+num_stages，=回执执行字节） |
    | `post_reorder_deepgemm_ascend.py` | 10948 | `f74aac822e54e5cbe9378391289480cfc764d6f4ff2d4d3649a4d07b5a7c14f5`（E3 同改双臂，=回执执行字节） |
    | `post_reorder_deepgemm_kunlunxin.py` | 3236 | `c14b806eb825cc11db35f71d1a59128ce89790e60614061025260e2bc579d879`（=e1/e2/s0 冻结字节） |
- **预注册门（引用 E3 候选记录节原文，读数前已锁定）**：华为数值失败或
  <12.08（e2 -5%）→ _ascend 回滚 e2 字节（blob
  `607a80b4…`@87d0e627）；任一 generic-rider 芯（天数/沐曦/海光/
  card_a/card_b）数值失败或较 e1 逐芯基线 -5% → generic 回滚 e1 字节
  （blob `b405f3cf…`@0f9cf744）；平台均值 >11.63934286 → 换 TB；昆仑
  冻结字节读数异常只排查不回滚。

## E3 平台终态（2026-09-26 发射，sub 21627）：valid 7/7 avg 9.74534286——四 rider 芯破 -5% 触发 generic 回滚门，TB 守 e1

- **发射记录（单 preflight + 单 submit，无 uncertain/sending）**：发射链
  status 预查（16:15:07）task=competing / can_submit=true / 额度 17/30 /
  最小间隔久过 → preflight（nonce `d19045b5`，tuple 与 E3 上膛五元组
  全匹配）→ submit --confirm → submission_id **21627**（daily_seq 14，
  created 16:16:26）→ watch 绑定 file_url_sha256
  `cd52c3a2f6388f618c0ec85c71bc6d9685dace1f1c087e7a64010e31a99666bc` /
  after-epoch 1790391527 轮询至终态（发射侧 observed 16:21:16）。
- **终态（落账员独立复核：`platform_cli.py status --race 782kzq4m
  --batch 7 --task 101`，observed 2026-09-26T16:31:22）**：status=
  completed，validity=valid，**7/7 GPU passed**（raw_result errors/
  failed_cases 全空），average_speedup **9.74534286**，is_team_best=
  **False**，ranking_score 0.71112221；额度 15/30 剩（本次复核时点，
  已含同窗 T96 sub 21628 一发；发射侧发射后读数为 16/30，如实双记）。
- **逐芯（e3 21627 vs e1 逐芯基线〔预注册门基准〕vs e2 21551；
  selected_file/exec_ms 取自逐芯 raw_result）**：
  | 芯 | e3 | e1 基线 | Δ vs e1 | e2 | Δ vs e2 | selected_file（exec_ms） |
  |---|---|---|---|---|---|---|
  | 天数 tianshu | 9.306 | 12.1168 | **-23.20%** | 11.9694 | -22.25% | generic（227983ms，vs e2 同芯 152283ms） |
  | 沐曦 muxi | 5.3242 | 6.342 | **-16.05%** | 6.4378 | -17.30% | generic（31493ms） |
  | 海光 haiguang | 12.4122 | 20.6092 | **-39.77%** | 19.8214 | -37.38% | generic（10923ms，vs e2 9013ms） |
  | 昆仑 kunlunxin | 11.3878 | 11.453 | -0.57% | 11.5302 | -1.23% | `_kunlunxin` 冻结字节（15866ms） |
  | 华为 huawei | 12.7202 | 12.3874 | +2.68% | 12.7198（门基准） | +0.003% | `_ascend` vendor 实跑（39635ms） |
  | card_a | 9.607 | 10.7918 | **-10.98%** | 10.8056 | -11.09% | generic（8714ms） |
  | card_b | 7.46 | 7.7752 | -4.05% | 7.6896 | -2.99% | generic（21955ms） |
  （华为门基准为 e2 12.7198：`_ascend` 自 E2 起实跑，E3 同改双臂。）
- **均值验算**：(9.306+5.3242+12.4122+11.3878+12.7202+9.607+7.46)/7
  = 68.2174/7 = 9.745342857，与平台 9.74534286 一致（截位）；vs e1 TB
  11.63934286 = -16.27%。
- **预注册门判决（判据=E3 候选记录节原文，读数后未改；本节只记录门
  触发事实，回滚执行归 orchestrator）**：
  1. 华为 12.7202 ≥12.08（e2 -5% 回滚线）且 7/7 无数值失败 →
     `_ascend` **不回滚**；
  2. **generic-rider 门触发**：天数 -23.20% / 沐曦 -16.05% / 海光
     -39.77% / card_a -10.98% 四芯破 e1 基线 -5% 线（card_b -4.05%
     未单独破）→ 按预注册文本 **generic 回滚 e1 字节**（blob
     `b405f3cf8c8d25f67208372ea1a5099631f785290bf91c5a50738af2ad889827`
     @0f9cf744）；
  3. 均值 9.74534286 ≤11.63934286 → **TB 不换**，e1(21509) 仍为我队
     按均值最优；
  4. 昆仑 11.3878（-0.57%）冻结字节噪声带内 → 只排查不回滚。
- **exec_ms 观察（窗口归因线索，非判决）**：天数 exec 227983ms 显著
  拉长（e2 同芯 152283ms，约 +50%）；海光读数较 e1 腰斩（-39.77%）但
  exec 10923ms 仅 +21%（vs e2 9013ms）——评测慢窗（reference 同窗
  退化）对读数的贡献未排除；读数如实上报，窗口归因与回滚执行归
  orchestrator/后续账本会话。
