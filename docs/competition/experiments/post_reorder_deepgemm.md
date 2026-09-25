# Task 101 `post_reorder_deepgemm` 实验记录

```current
task: 101
operator: post_reorder_deepgemm
batch: 7
validity: valid
platform: s0(21443)valid 7/7 avg10.02:昆仑11.4(#2)/华为12.3(#2)/海光15.0/天数11.0;均值距榜首11.10一步之遥
candidate_stage: e1
team_best_stage: s0
team_best_speedup: 见platform行
sealed: no
next: E1上膛待发射(hidden块上grid.y+BLOCK 2048,昆仑冻结s0 vendor);按预注册门判读后定E2轴
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
