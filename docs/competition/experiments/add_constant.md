# Task 93 `add_constant` 实验记录

```current
task: 93
operator: add_constant
batch: 7
validity: valid(7/7,s0)
platform: e3(21510)valid:华为0.332(+34%,两段式无mask兑现,门0.5-0.9保留带)/昆仑0.652;avg0.838 vs榜首0.919
candidate_stage: e3
team_best_stage: e3
team_best_speedup: 0.841(avg)
sealed: no
next: 华为0.33→0.92仍2.8x;方向已二次验证(T97同族+28%)
updated: 2026-09-26
```

## 2026-09-26 E3 预注册：华为两段式无mask热路径（_ascend vendor重写）

- 缺口（分诊员实跑核验 climb-loop.json s2t1op093）：我方 huawei 0.2472 vs
  wangteam 0.9878 / OpeGoodn 0.9860；第三名 GuanghuLab 0.388 断崖 = 结构分化。
  我方 avg 0.83271429（#6）vs 榜首 RSI 0.91932143——单芯全额兑现
  +0.739/7 = +0.106 即 0.9384 反超。
- 病理双证：旧 `_ascend` 字节（`5cfdb654`）每 tile `m = offs < numel`
  int32 向量比较（chip-rulesets.md:36 Vector CMP 不支持 int32 降标量）+
  `tl.load other=0` 预填（chip-rulesets.md:39 串行化 MTE2）。已排除轴：
  persistent cap 48 vs 512 无差（e1/e2，上current块）；chip-rulesets.md:43-45
  四假说全负（int32 化/launch 数/并行度族）不含热路径去 mask。
- 结构（source commit `72364b4c`，评审 r2 docstring 补披露后为本轮 commit）：
  grid = numel//16384 整块 program + 恰 1 个尾块 program；kernel 内标量分支
  `pid < n_full`（两路均 Triton 计算，非 PyTorch fallback，无模块级可变容器）：
  热路径 load/store 双无 mask 无 other，尾块 masked load 不带 other + 同 mask
  store（T92 e22 与 FlagGems pointwise_dynamic 形态，上游 masked load 无
  other 已 `gh api` 核对 L608）。BLOCK=16384/num_warps=16 预注册（T92 e22
  华为现行已证字节档；chip-rulesets.md:40 阶梯 ≥4096→16）。int32 寻址。
- 测试：新增 `AddConstantTest.test_two_segment_block_boundaries`
  （16383/16384/16385/32767/32768/32769/65536：B±1、2B±1 与整倍数纯热路径）
  并列入 RELEASE_REQUIRED_TESTS；既有矩阵 1023/1025/4097/2^20+1/3*2^20+7
  覆盖双臂与尾块。本地 `python3 -m py_compile` 两文件通过（工作树与提交
  字节均验）；本机无 torch/CUDA，代理回归与华为首验留待验证轮（华为
  target-runtime-unverified）。
- 负证据披露（评审 r2 P2-1）：同概念同芯先例 **T92 E23**（本目录
  unpad_draft_extend_output.md「E23 平台终态（20457）」）：整块 load+store
  双去掩码 + 尾块保留 mask，平台 8/8 正确，但华为 316.2016 vs e22 444.1506
  （−28.8%，八芯净 −92.0072，华为独自 −127.949），命中预注册均分门，自
  不可变 e22 ZIP 回滚（source 回滚 commit `47dc1382`）。与本候选差异：E23
  int64 地址 + (bs,tiles) 2D grid-stride + 循环内逐 base 标量守卫；本候选
  int32 + 1D 每 program 一 tile + kernel 级 pid 分支。家族证据无法分解 E23
  回归由哪个因子驱动，故本候选华为方向真开放，以下性能门约束下行风险。
- 预注册门（评审 r2 P2-2，screening/preflight 前锁定）：
  1) 数值失败 → 回滚 `_ascend` 至 `5cfdb654` 字节（member SHA-256
     `7b3ee54ba624ad0d7bac434ea3b2a3d3fa68986a4ac17d903c01aadaa0997040`）；
  2) 华为 <0.5 → 回滚同上；华为 ≥0.9 → 换 TB；0.5–0.9 保留；
  3) 均分 ≤0.841（e1 TB）→ 回滚 `_ascend` 字节；同字节他芯读数按
     ±10-35% 水位波动同窗判读（chip-rulesets.md:58），不得单点判涨跌。

### E3 上膛（2026-09-26 01:46，armed-unfired）

- **远端 release 矩阵（绑定 HEAD）**：`verify_release.py prepare add_constant
  --source-commit HEAD --verification-commit HEAD --proxy-vendor ascend
  --proxy-vendor kunlunxin`（该题现有全部 vendor；mode=release，schema v2），
  上传 `/tmp/wf-e3-huawei-two-segment-nomask-release` 后远端
  `/home/kevin/notebook/.venv/bin/python` 执行 `run`，**RC=0**：
  **5 测试 / 127 case 全过**（5/5 RELEASE_REQUIRED，含
  `test_two_segment_block_boundaries`，0 fail/error/skip/xfail），
  **每源 31 次 kernel launch**（generic/`_ascend`/`_kunlunxin` 各 31，
  source_calls 同为 31×3），69 条非空张量 shape 记录，环境 NVIDIA RTX
  5070 Ti / torch 2.13.0+cu130 / triton 3.7.1 / cuda 13.0。回执
  `artifacts/competition/day5prep-20260921/e3-huawei-two-segment-nomask-wf/verification.json`
  （sha256 `a7e760ce27b90b0713cb6cbbaea76e50b579dc58a15cbc5c1a25e18a22846379`）
  / `verification.log`（sha256
  `aa7eed6b13ca71e94963149fa5c48ad54492f2ab8235dc77e7a1cfa37f5dbc3e`，
  `Ran 5 tests in 1.957s OK`）。源码字节谱系 = 72364b4c（候选）→
  d8b13f67（评审 r2 docstring 披露）→ HEAD 8b7394af
  （`git diff d8b13f67 HEAD -- src/…/add_constant*.py tests/…` 为空，
  其后两个 commit 均为纯 docs）。
- **五元组（上膛身份）**：
  - source_commit：`8b7394af73c9489061de0bda06f269356e63c583`
  - verification_commit：`8b7394af73c9489061de0bda06f269356e63c583`（=本回执，
    发射 preflight 要求 commit 字段等于它）
  - ledger_commit：本 commit（E3 上膛记账）
  - ZIP：`artifacts/competition/add_constant/e3-8b7394a/add_constant.zip`
    （8628 字节，zip_sha256 =
    `4543911ea8c4d1d4cad0e0d5e891e8b01b070fdc4dd960e3c58f53054e7a6bb6` =
    canonical_zip_sha256；≠ e2 `dcd04a21…` ≠ e1 `74fd8b45…` ≠ s0
    `cad85701…`；vs e2 仅 `_ascend` 成员由 `7b3ee54b`（回滚档字节）变为
    `b8ce66f2`，generic `a2c0b1ba` 与 kunlunxin `3e5d489c` 冻结——单变量
    纪律成立，平台 zip_sha256 去重键成立）
  - stage：`e3`（CURRENT candidate_stage 已预写 e3 且无 e3 ZIP 存在，
    上膛即落 e3，序列 s0→e1→e2→e3 连续不跳号）
  - ZIP 成员名单（与 `zipfile.namelist()` 实际核对一致，无夹带；
    `unzip -t` 无错；UTF-8 `.py`，无测试/缓存/目录前缀/macOS 垃圾；
    成员字节 = git blob @8b7394af = 回执 files 哈希 = 远端实际执行字节）：
    | 成员 | 字节 | sha256 |
    |---|---|---|
    | `add_constant.py` | 1182 | `a2c0b1baf3633464b4fda27955f8376c1c076f9274e2a6cbb88c7883835d31fb` |
    | `add_constant_ascend.py` | 5297 | `b8ce66f22b47eded68ea4c7aa7f15436e2189fdbc71e3ae3d62032b63087e106` |
    | `add_constant_kunlunxin.py` | 1775 | `3e5d489c951c38abe030bf0c95b3ef21e9dd22b888166c0861ec8caacb90fc5b` |
- **状态**：armed-unfired——发射前预注册门不变（上节三条：数值失败或华为
  <0.5 或均分 ≤0.841（e1 TB）→ 回滚 `_ascend` 至 `5cfdb654` 字节
  `7b3ee54b…`；华为 ≥0.9 → 换 TB；0.5–0.9 保留，同字节他芯 ±10-35%
  水位同窗判读），发射 preflight 通过后执行一次性 submit；华为
  target-runtime-unverified 状态不变（NVIDIA 代理不背书目标芯）。

## 契约

- `def reference(src, constant)`：1D 连续 int32，`out = src + constant`，精确。
- baseline 是 C++ 模板（常量编译期折叠 + kMaxVecBytes 向量化）；>2^20 元素走向量路径。

## 榜单靶标（逐芯 #1 读数，score=Σ1/rank 口径，见 plan 文档）

wangteam 4.15 分领跑但全场均值 <1.13：tianshu 1.129 / muxi 1.012 / haiguang 0.971 /
kunlun 0.910 / huawei 0.988 / A 1.005 / B 1.008。全芯 ~1.0+ 即可拿 5.5+ 分。
关键分化：华为其他队 0.14-0.34（wangteam 0.988 独家配方），昆仑次优 0.48-0.72。

## S0 结构（commit b6641097）

- `CONSTANT: tl.constexpr` 编译期折叠（对齐 baseline 模板 trick）；
- flat map，BLOCK=1024 / num_warps=4（4M 元素代理 11.69µs ≈ torch 11.70µs）；
- i64 仅寻址；mask 全覆盖；`if numel` 防 0 元素。

## 证据

- NVIDIA 代理 screening（BLOCK=4096 旧配置）4 测试 0 失败；1024/4 新配置复筛中。
- 代理 launch 开销实测：Triton ~4.8µs vs torch.add ~2.6µs（小 shape 全场 <1.0 的成因假设）。
- KernelGen 华为 autotune job `6e09dda3`：3 轮后 failed，服务端已清理记录（poll 404），失败详情不可恢复；华为专配轴待平台逐芯反馈后重开（generate_kernel 新 job 或依 s0 华为读数定位）。
- KernelGen 昆仑后端仍 502（T98 请求已归档 `t98-kunlun-tune-req.json`）。

## 情报

- 快照 `docs/competition/data/batch7-intel-20260924.json`
  SHA-256 `061475d31505bb81f16a5a212fa7cfd6e4b1142188cb02da2c202962efedadb2`。

## 不可变身份（s0 v2，2026-09-24，契约 spec 高阶复审后）

- source commit = verification commit = `daf77923f6913447b65a8c226f34297ce13b4440`（review v1 4项（1P1+3P2）已修；本轮 spec 复审 T93 无新发现）。
- ZIP：`artifacts/competition/add_constant/s0-daf7792/add_constant.zip`，SHA-256 `cad85701d335dbe17a747e10280eec4920859563391c783cfd5c523b1ede6cf9`（单成员 `add_constant.py`，generic-only）。
- release 回执：`artifacts/competition/b7-s0-release2-20260924/add_constant/verification.json`（mode=release，exit 0，绑定该 commit 字节），SHA-256 `89d7b8ba239b5d04f21d8540e3450825367bd0cb0cf6db96a47b0ce4ec10efb3`；
  日志 SHA-256 `e654eed4ed5f5167f19d6dc96677a068d5adb3f9b59653bfe2ce4d7ec132370d`。NVIDIA RTX 5070 Ti / torch 2.13.0+cu130 / triton 3.7.1。
- 提交脚本：`artifacts/competition/b7-s0-release2-20260924/submit-batch7-s0.sh`（v2，全部参数预烘焙）。
- review 历史：v1（commit 级，gpt-6-astra medium）4 项全修；v2（--base 对照契约 spec，high）5 项 P2 全修；
  三轮修复均经 screening + release 双门禁复跑全绿。
