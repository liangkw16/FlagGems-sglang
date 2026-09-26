# Task 96 `fused_pack_qkv` 实验记录

```current
task: 96
operator: fused_pack_qkv
batch: 7
validity: valid(7/7,e3)
platform: e3(21628)valid 7/7 avg1.95460714 is_team_best=True(TB换e3;my_best 1.72675旧口径差已由本次实读对齐):沐曦2.1715(+13.10%)/海光3.45875(+9.85%)/card_a1.44175(+16.65%)/card_b1.40175(+7.85%)四rider升,华为0.50075(-39.23%,破回滚线0.7828→generic回滚门触发待执行)/天数4.138(-0.78%,_iluvatar冻结)/昆仑0.56975(+2.47%,冻结);vs e2平台实读1.88703571=+3.58%
candidate_stage: e3(fired,读数落账见E3平台终态节)
team_best_stage: e3
team_best_speedup: 1.95460714(21628)
sealed: yes
next: 华为回滚门已执行(generic+tests回fde2d02c字节,blob db53db9a验签通过);平台TB e3(21628,1.9546)已锁定不受影响;下轮决策=四rider增益(+7.9%~+16.7%)vs华为-39.2%的取舍(可华为单芯vendor切分保四rider)
updated: 2026-09-26
```

## 契约

- `(q,k,v,indices)`：`[B,S,H,D]` 三张量按 flat `B*S` 索引 gather 成三个
  `[total_valid,H,D]`；indices int32/int64；exact 纯 gather。

## 榜单靶标

EvokeAgent 6.00 分（仅 2 队可见）：tianshu 5.512 / muxi 2.446 / haiguang 3.266 /
**kunlun 0.606** / **huawei 0.617** / A 1.481 / B 1.392。两队长芯 kunlun/huawei 均 <1：
单 kernel 行 gather 结构大概率直接拿下这两芯 #1。

## S0 结构（commit b6641097）

- 单 kernel：BLOCK_R=4 行 × BLOCK_C=1024 列 2D 瓦片；`indices` 每行 load 一次按列广播；
  一个 program 内完成 q/k/v 三份拷贝（1 launch vs baseline 3 gather + long cast）；
  `idx.to(tl.int64)*row_elems` 防溢出；num_warps=4（sweep @HD2048 26.46µs vs torch 28.38µs）。

## 证据

- NVIDIA 代理 screening（r8 旧配置）5 测试 0 失败 11 launch；r4 新配置复筛中。
- proxy L2 常驻下 HD512 1.65x / HD2048 1.07x；平台冷数据口径下结构优势更大。

## 情报

- 快照 SHA-256 `061475d31505bb81f16a5a212fa7cfd6e4b1142188cb02da2c202962efedadb2`。

## 不可变身份（s0 v2，2026-09-24，契约 spec 高阶复审后）

- source commit = verification commit = `daf77923f6913447b65a8c226f34297ce13b4440`（review v2：补 indices.stride(0) + 非连续 q/k/v 归一化（替代断言））。
- ZIP：`artifacts/competition/fused_pack_qkv/s0-daf7792/fused_pack_qkv.zip`，SHA-256 `4edde6d3aacc05c8dd63dee86dfd0d782bc707ae3fa217a463701dcb80705da0`（单成员 `fused_pack_qkv.py`，generic-only）。
- release 回执：`artifacts/competition/b7-s0-release2-20260924/fused_pack_qkv/verification.json`（mode=release，exit 0，绑定该 commit 字节），SHA-256 `98b7f6f9c603979336da0f3a66b9f9dab8696dd65b2fa289491b20ee761e3f94`；
  日志 SHA-256 `7e5058c05673e57ab61e2f9c9dc6563feee7d090443a8b9048d37f8a56b3f578`。NVIDIA RTX 5070 Ti / torch 2.13.0+cu130 / triton 3.7.1。
- 提交脚本：`artifacts/competition/b7-s0-release2-20260924/submit-batch7-s0.sh`（v2，全部参数预烘焙）。
- review 历史：v1（commit 级，gpt-6-astra medium）4 项全修；v2（--base 对照契约 spec，high）5 项 P2 全修；
  三轮修复均经 screening + release 双门禁复跑全绿。

## E2 候选：天数 `_iluvatar` vendor（per-row 1D + int32 寻址域 + 单 alloc，评审 r1，2026-09-26）

**假设**：用上游原型 per-row（grid=n_valid，一行一程序）1D 单段拷贝替换自研
BLOCK_R=4×1024 2D tile，叠加全链 int32 寻址（域内）与 3 alloc→1 单 buffer 三 view，
可把天数从 2.95 拉进对手群众带 4.6-6.1。

**缺口证据**（climb-loop s2t1op096，2026-09-26T01:50 快照，分诊员实跑复核）：我方天数
2.9505 全场 14 队垫底，其余 13 队 3.447-6.148（榜首 c2flow 5.5245、Sweetdeath 6.148）；
天数→5.52 = 均值 +0.366，占均值缺口 58%。platform_cli status 实查（只读）
s0(578dfe02)→e1(b25a11c2) 天数同 generic 字节读数 2.95575/2.9505（差 0.2%），排除水位、
判定纯结构差。

**机制证据**（4 条）：
1. 上游 sglang 原型 `varlen_pack_pad_triton.py::_fused_pack_qkv_kernel` 即 per-row
   （grid=(n_valid,) 每程序一行）+ BLOCK_HD=next_pow2(hd) 1D 单段 + 标量 idx load；
   对手群众带 4.6-6.15 与我方 2D tile 2.95 构成结构对照。
2. T51 E9 平台证据：行块（16 行/程序）在天数 -20%（fla_layernorm_gated.md:386-388）。
3. 本题内部同族：昆仑 2D tile 0.010→1D 行程序 0.583（57x，`_kunlunxin` vendor e1 平台兑现；
   分诊员实跑复核 s2t1op096 kunlunxin=0.5830 ✓）；天数从未测过任何非 2D 形态。
4. int64 全链（generic `fused_pack_qkv.py:30-38` 五处 int64：rows/idx/src/dst/cols）是
   三题族共同特征且天数系统性落后（T95 per-row int64 82.3 vs AICity 139.8；T98 2D int64
   2.37 vs c2flow 2.91；跨芯负资产先例：燧原 enable_i64=False、华为 Vector ADD 无 int64，
   chip-rulesets.md）。

**实现**（`src/flaggems_sglang/runtime/backend/_iluvatar/ops/fused_pack_qkv.py`，新增文件）：
- `_pack_row_i32_kernel` / `_pack_row_i64_kernel` 双 Triton 孪生：per-row grid=(n,)、
  标量 idx load、`BLOCK_C=min(65536, max(128, next_pow2(row_elems)))` 单段掩码拷贝；
  列循环仅在 H*D>65536 时兜底（宽行回归 70000 覆盖）。i32 版全链不 cast（int32 寻址），
  i64 版与 generic 五处 cast 逐位对齐（溢出域安全网）。
- host 分派 `_use_int32(q_numel, n_rows, idx_s0)`：数值域判定（非设备判断，T80 e30 同构）。
  界：q.numel() < 2^31 且 (n_rows-1)*idx_s0 < 2^31（覆盖 strided indices load 深度；
  src=idx*row_elems<numel、dst=n*row_elems≤numel 由语义 n≤B*S 保证）。
- 单 alloc：`torch.empty(3*n*row_elems)` 一块平 buffer，q/k/v 三输出为三个连续 view
  （T77 e6/e17 先例：天数 +8.6%、昆仑 +91%）。
- num_warps=4（generic 在 HD2048 的 proxy sweep 值）；纯 Triton、无 try/except、
  无模块级可变容器。
- caveat：grid=(n,) 在 i64 域需 n≥2^31 且 row_elems=1 的退化形态才触 CUDA grid-x 上限，
  公开域（B*S*H*D≤6.3M）不可达，记录不设防。

**测试**：`tests/test_fused_pack_qkv.py` 新增 `test_iluvatar_int32_domain_guard`
（2^31 边界四断言：最大合法域 True / numel==2^31 False / strided indices 越界 False /
空 gather True），列入 RELEASE_REQUIRED_TESTS；既有矩阵（shapes/tails、int64+重复索引、
fp32/bf16、sliced indices、非连续 q/k/v、70000 宽行、全量+空）自动经
`_op_variants.load_operator_modules` glob 发现并全量跑 i32 路径。

**覆盖度声明**：本候选 target-runtime-unverified（天数 lowering 本地不可验证）；
i64 孪生 kernel 在 CI 域（<2^31 元素）不可达、未被执行，仅靠与 generic 已证 i64
数学逐位同构 + 守卫单测锚定边界；NVIDIA 代理只验数学/JIT，不可外推天数性能，
平台单发裁决。本轮（评审 r1）本地验证为 py_compile + 守卫函数实跑断言，
代理 GPU 矩阵留待 release 轮。

**预注册门**（armed-unfired）：天数 ≥4.0 保留 / ≥4.6 进场带（13 队群众带下沿）/
均值 >1.72675 换 TB；数值失败或天数 <2.95（回退）即回滚删 vendor 文件、树回 e1 字节；
同字节重掷 ≤2 次；其余六芯 vendor 隔离（仅 tianshu 选中 `_iluvatar`，
映射 E9 五发实证 fused_recurrent_gdn.md:548）读数应不变。

**归因注记**：三变量打包（形态+i32+单 alloc）归因模糊，E3-generic-i32 单变量留作
后续归因补充而非本轮同题并发。单芯兑现不翻榜（均值→约 2.09 < 榜首 2.3555），
属大幅抢分 + 打开天数轴。

### E2 评审 r2 修复（2026-09-26）

**P1（正确性，必修）**：`_use_int32` 漏检输出侧寻址域。原 guard 只查
`q_numel<2^31` 与 `(n-1)*idx_s0<2^31`，依赖「n≤B*S ⇒ n*row_elems≤q.numel()」——
该不变量题面并未强加：纯 gather 允许重复/膨胀 indices（本题矩阵自测 duplicates），
评审实跑复现 q=(1,1,1,32768)、indices=65537 个零时 guard 放行 i32，kernel 内
`dst=65536*32768=2^31` int32 回绕为 -2^31 → 负偏移 OOB 写；generic int64 同输入
正确，属 vendor 相对 generic 的正确性回归（且账本预告 E3 将把 i32 引入 generic，
缺陷会扩散）。修复：`_use_int32` 增参 `row_elems` 并直接检查
`n_rows*row_elems<2^31`（输出侧独立设界，不再从 q.numel() 推导），docstring 的
错误论证同步改正；wrapper 调用点更新。测试：守卫测试补评审复现值
`(32768,65537,32768,1)→False` 与精确边界 `2*2^30→False / 2*(2^30-1)→True`
（纯 host 断言，OOB 复现不可安全执行）；`test_indices_int64_and_duplicates`
补 GPU 膨胀回归（n=2048>B*S=512 全零索引，域内走 i32 真跑）。
**P2（发布配置陷阱，必修）**：守卫测试原 `skipTest("iluvatar not in
FLAGOS_TEST_SOURCES")` 违反仓库零 skip 规范——verify_release.py require_success
拒绝任何 skipped（e1 回执 8 测试零 skip），评审实跑复现默认 nvidia 域
execution_sources=[generic] → skipped=1 → ValueError『verification contains
skipped』，即任何不带 iluvatar 源的该题 release/screening 必然以误导性错误失败。
修复：删 skip 分支改 `assertIsNotNone` 硬断言——vendor 文件进 ZIP 则其源必须进
验证配置，缺席即响亮失败（信息指明 FLAGOS_TEST_SOURCES 缺 iluvatar），不放宽
全局 skip 门禁；后续本题 release/screening 命令必须以 --proxy-vendor 含
iluvatar（连同既有 kunlunxin）。

### E2 上膛（2026-09-26 02:37，armed-unfired）

- **远端 release 矩阵（绑定 HEAD）**：`verify_release.py prepare fused_pack_qkv
  --source-commit HEAD --verification-commit HEAD --proxy-vendor iluvatar
  --proxy-vendor kunlunxin`（该题现有全部 vendor；mode=release，schema v2），
  上传 `/tmp/wf-e2-iluvatar-perrow-1d-release` 后远端
  `/home/kevin/notebook/.venv/bin/python` 执行 `run`，**RC=0**：
  **9 测试 / 64 case 全过**（9/9 RELEASE_REQUIRED，含 r2 加固的
  `test_iluvatar_int32_domain_guard` 与 `test_indices_int64_and_duplicates`
  膨胀重复索引 GPU 回归，0 fail/error/skip/xfail/unexpected-success），
  **每源 16 次入口调用 / 15 次 kernel launch**（generic/`_iluvatar`/
  `_kunlunxin` 三源各自实际执行），环境 NVIDIA RTX 5070 Ti /
  torch 2.13.0+cu130 / triton 3.7.1 / cuda 13.0 / python 3.12.13。
  回执 `artifacts/competition/day5prep-20260921/e2-iluvatar-perrow-1d-wf/verification.json`
  （sha256 `bb0dfc1a4bfabd630a361e6a242f3eabe2f0b1c1975bad7a4766528b583d1d1b`）
  / `verification.log`（sha256
  `9cc40ce0fe3b21edb46fd5d0cd8005a6c93eb2703ba6318d81a8b88b8612f97e`，
  `Ran 9 tests in 1.482s OK`，与回执内 log_sha256 一致）。
  `target_unverified_sources` = iluvatar + kunlunxin（目标芯未验证，维持
  E2 节披露口径）；`unexecuted_sources` = 空。源码字节谱系 =
  ef74f308（r1 候选）→ fde2d02c（r2 修复）= HEAD，回执直接绑定 r2 字节。
- **五元组（上膛身份）**：
  - source_commit：`fde2d02c51c9879370b6024ba2a8cae7941bba34`
  - verification_commit：`fde2d02c51c9879370b6024ba2a8cae7941bba34`（=本回执，
    发射 preflight 要求 commit 字段等于它）
  - ledger_commit：本 commit（E2 上膛记账）
  - ZIP：`artifacts/competition/fused_pack_qkv/e2-fde2d02/fused_pack_qkv.zip`
    （13180 字节，zip_sha256 =
    `15e4006625bc0069d384a88048ba1fc609468a83a7878b1b83694fc09ea1c585` =
    canonical_zip_sha256；≠ e1 `00cac1db…`（5abd626）/ `711be763…`
    （d3e86e9）≠ s0 `4edde6d3…`（daf7792）/ `81767f23…`（90d732f）；
    vs e1-5abd626（昆仑 vendor 现行字节）generic 与 kunlunxin 成员逐字节
    冻结（`db53db9a`/`bf011714` 不变），仅新增 `_iluvatar` 成员
    `1bc958df`——单芯 vendor 追加纪律成立，平台 zip_sha256 去重键成立）
  - stage：`e2`（CURRENT candidate_stage 已预写 e2 且无 e2 ZIP 存在，
    上膛即落 e2，序列 s0→e1→e2 连续不跳号；账本已预留 E3-generic-i32
    后续归因候选名号不受挤占）
  - ZIP 成员名单（与 `zipfile.namelist()` 实际核对一致，无夹带；
    `unzip -t` 无错；UTF-8 `.py`，无测试/缓存/目录前缀/macOS 垃圾；
    成员字节 = git blob @fde2d02c = 回执 files 哈希 = 远端实际执行字节）：
    | 成员 | 字节 | sha256 |
    |---|---|---|
    | `fused_pack_qkv.py` | 2703 | `db53db9a196e3fb08206725bf945c34aac22dec186f119fa0d0bab913c286e52` |
    | `fused_pack_qkv_iluvatar.py` | 6746 | `1bc958dfa1676da41becd7cee725965ec2eed8a90bbcf910da9231c9f16a8cc4` |
    | `fused_pack_qkv_kunlunxin.py` | 3341 | `bf01171404b813b9792938716b24a214f3a130f14c618951cd00ca4cae51812c` |
- **状态**：armed-unfired——发射前预注册门不变（E2 节原文：数值失败或
  天数 <2.95（回退）→ 回滚删 `_iluvatar` vendor 文件、树回 e1字节；
  天数 ≥4.0 保留 / ≥4.6 进场带（13 队群众带下沿）/ 均值 >1.72675 换 TB；
  同字节重掷 ≤2 次；其余六芯 vendor 隔离（仅 tianshu 选中 `_iluvatar`）
  读数应不变）。发射命令必须带 `--proxy-vendor iluvatar
  --proxy-vendor kunlunxin`（P2 修复后该源缺席会响亮失败）。

## E3 候选与上膛：generic-perrow（2026-09-26 16:07，armed-unfired）

**候选**（r1 `e03934bd` → 评审 r2 修复 `9bad92cb`，实现细节见两 commit）：
把 s0 2D 行瓦片（BLOCK_R=4×1024、int64 全链、3 alloc）换成与 E2
`_iluvatar` vendor 同构、平台已在天数兑现 +41% 的 per-row 1D 三件套，一次抬
五个仍乘 generic 字节的 rider 芯（muxi/huawei/card_a/card_b/haiguang）：
per-row 1D 程序 + 65535 程序数帽（`rows_per_prog=cdiv(n,65535)`，Ascend
coreDim）；kernel 内全 1D 张量（拆 muxi「1D grid + in-kernel 2D tensor」
TTGIR 险与 Ascend Vector CMP int64 罚，chip-rulesets.md:29/31/36）；
BLOCK_C 帽 2048（muxi max_tile_size，旧 4096 超 2 倍），宽行走分块列循环
（H*D=70000 回归）；int32 寻址域 `_use_int32`（r2 补 padded row space
`grid*rows_per_prog<2^31` 域界与输出侧 `n*row_elems<2^31` 独立设界）+
i64 孪生逐位镜像 s0 数学；单 alloc 三 view；`_iluvatar`/`_kunlunxin`
vendor 字节冻结。测试 +2：`test_generic_int32_domain_guard`、
`test_generic_rows_per_prog_multirow`（RELEASE_REQUIRED 9→11）。
本轮为 E2 节预告的归因补充臂（三变量打包 → generic 臂单变量迁移），非同题
并发新轴。

### 上膛证据（远端 release，绑定 HEAD `7e0c58b0`）

`verify_release.py prepare fused_pack_qkv --source-commit HEAD
--verification-commit HEAD --proxy-vendor iluvatar --proxy-vendor
kunlunxin`（该题现有全部 vendor；mode=release，schema v2），目录
`/tmp/wf-fused_pack_qkv E3 generic-perrow-release` 上传 gpu 后以
`/home/kevin/notebook/.venv/bin/python` 执行 `run`（timeout 900），**RC=0**：

- **11 测试 / 83 case 全过**（11/11 RELEASE_REQUIRED，含两个 E3 新回归与
  r2 加固的 padded-row-space 守卫断言；0 fail/error/skip/xfail/
  unexpected-success）。
- **每源 20 次入口调用 / 19 次 kernel launch**（generic/`_iluvatar`/
  `_kunlunxin` 三源各自实际执行）；96 非空 shape。
- 环境 NVIDIA RTX 5070 Ti / torch 2.13.0+cu130 / triton 3.7.1 /
  cuda 13.0 / python 3.12.13；`Ran 11 tests in 1.515s OK`（与 E2 的
  1.482s 同量级，无墙钟异常需披露）。
- 回执 `artifacts/competition/day5prep-20260921/fused_pack_qkv E3
  generic-perrow-wf/verification.json`（sha256
  `a81e89182aee0d3b8de826515d2e7fd4e42bc3ae29c3d78549bc0e7c32faf646`）
  / `verification.log`（sha256
  `c5b76917c0b5f50c1341e71e7a43eb33481215be41692dd8cd21d80f05f45b9c`，
  与回执内 `log_sha256` 一致，本地落盘后逐字节复核）。
- `target_unverified_sources` = `_iluvatar` + `_kunlunxin`（目标芯未验证，
  维持 E2 节披露口径）；`unexecuted_sources` = 空。
- 源码字节谱系 = `e03934bd`（r1 候选）→ `9bad92cb`（r2 修复）→ HEAD
  `7e0c58b0`：r2 之后 4 个并行会话提交均未触及本题五路径
  （`git log 9bad92cb..HEAD -- <generic/tests/两vendor/账本>` 为空），
  回执直接绑定 r2 字节。

### 五元组（上膛身份）

- source_commit：`7e0c58b061b639568f7b817c678906affe0b1566`
- verification_commit：`7e0c58b061b639568f7b817c678906affe0b1566`（=本回执；
  发射 preflight 要求 commit 字段等于它）
- ledger_commit：本 commit（E3 上膛记账）
- ZIP：`artifacts/competition/fused_pack_qkv/e3-7e0c58b/fused_pack_qkv.zip`
  （18182 字节，zip_sha256 =
  `bbfe10ece54c527831ae032d2d04dd35d4363d24f9f147e89fbcf2a5ab06421c` =
  canonical_zip_sha256；≠ e2 `15e40066…`（fde2d02）≠ e1 `00cac1db…`
  （5abd626）/ `711be763…`（d3e86e9）≠ s0 `4edde6d3…`（daf7792）/
  `81767f23…`（90d732f）——平台 zip_sha256 去重键成立，新候选产出了
  新 ZIP 字节、非同字节重掷；vs e2-fde2d02 仅 generic 成员重写
  （`db53db9a`→`e6fe9f40`），`_iluvatar` 成员 `1bc958df` 与
  `_kunlunxin` 成员 `bf011714` 逐字节冻结——generic 臂单变量纪律成立）
- stage：`e3`（CURRENT candidate_stage 由 e2 递增一号；上膛前无 e3 ZIP
  存在，s0→e1→e2→e3 连续不跳号）

ZIP 成员名单（与 `zipfile.namelist()` 实际核对一致，无夹带；`unzip -t`
无错；UTF-8 `.py`，无测试/缓存/目录前缀/macOS 垃圾；成员字节 = git blob
@7e0c58b0 = 回执 files 哈希 = 远端实际执行字节，三方核对全中）：

| 成员 | 字节 | sha256 |
|---|---|---|
| `fused_pack_qkv.py` | 7705 | `e6fe9f407712979eee16ac939be1ae83aa192207d9db35459d00cd997464ffc7` |
| `fused_pack_qkv_iluvatar.py` | 6746 | `1bc958dfa1676da41becd7cee725965ec2eed8a90bbcf910da9231c9f16a8cc4` |
| `fused_pack_qkv_kunlunxin.py` | 3341 | `bf01171404b813b9792938716b24a214f3a130f14c618951cd00ca4cae51812c` |

### 预注册门（armed-unfired，发射后判决）

- **影响面**：generic 重写只落在五个 rider 芯（muxi/huawei/card_a/
  card_b/haiguang）；tianshu 选 `_iluvatar`、kunlunxin 选 `_kunlunxin`
  （两 vendor 字节冻结）——隔离芯读数异常只排查不回滚（T101 e3 同处置）。
- **回滚门**：任一 rider 芯数值或编译失败，或较 e2 读数 -5% → 回滚
  generic 至 e2 字节（`db53db9a` @fde2d02c）。基线与回滚线：huawei
  e2=0.824（本账本 CURRENT 记读）→ <0.7828；muxi 1.920 / haiguang
  3.1485 / card_a 1.236 / card_b 1.29975（climb-loop s2t1op096 我队
  my-best 行 2026-09-25T00:52:49；四芯 e1→e2 乘同一 generic 字节、e2
  提交未单列四芯读数，以同字节 my-best 行为基线并披露来源）→
  分别 <1.824 / <2.9911 / <1.1742 / <1.23476。
- **判轴（记账性，非回滚）**：huawei ≥1.1 记 per-row 轴在 generic 臂
  显著兑现（2D-tile 地板 0.824）；huawei ≥1.40525（c2flow 读数）或
  muxi ≥2.181（现榜首 EvokeAgent 读数）记进对手带。
- **换 TB**：均值 >1.89（e2 账本记读）。口径披露：平台 `my_best` 双快照
  （batch7-intel-20260926-day 13:26 与 climb-loop 2026-09-26T15:15）仍读
  1.72675（best_submitted_at 09-25T00:52:49），与账本 e2 avg 1.89 存差；
  取较高者为换线避免假记 TB，差异留观，发射后以 platform status 实读对账。
- 同字节重掷 ≤2 次。
- 发射命令必须带 `--proxy-vendor iluvatar --proxy-vendor kunlunxin`
  （E2 P2 修复后 vendor 源缺席会响亮失败）。

## E3 平台终态（2026-09-26 发射，sub 21628）：valid 7/7 avg 1.95460714，is_team_best=True——均值/TB 双升，华为单独触发 generic 回滚门

- **发射记录（单 preflight + 单 submit，无 uncertain/sending）**：与
  T101 e3（sub 21627，16:16:26）间隔约 9 分钟（≥125s 最小间隔）；
  发射链 status 预查（16:23:42）task=competing / can_submit=true /
  额度 16/30 → preflight（nonce `c2a08f24`，tuple 与 E3 上膛五元组
  全匹配：commit `7e0c58b` / zip `bbfe10ec…` / 3 成员
  generic+iluvatar+kunlunxin / test-sha `4af416d8…` / receipt
  `a81e8918…`，回执 source=verification=`7e0c58b`，11/11，proxy
  iluvatar+kunlunxin）→ submit --confirm → submission_id **21628**
  （daily_seq 15，created 16:25:43）→ watch（发射侧原样 watch_command）
  绑定 file_url_sha256
  `274d388d05053633dc287cf959d815e9d8c89d5ba106cb070c04f830a4c7495b` /
  after-epoch 1790410586 轮询至终态（发射侧 observed 16:29:09）。
- **终态（落账员独立复核：`platform_cli.py status --race 782kzq4m
  --batch 7 --task 96`，observed 2026-09-26T16:32:08）**：status=
  completed，validity=valid，**7/7 GPU passed**（raw_result errors/
  failed_cases 全空），average_speedup **1.95460714**，is_team_best=
  **True**，ranking_score 1.1777369；额度 15/30 剩。
- **逐芯（e3 21628 vs 预注册基线；基线与回滚线见上方 E3 预注册门节；
  selected_file/exec_ms 取自逐芯 raw_result）**：
  | 芯 | e3 | 基线 | Δ | selected_file（exec_ms） |
  |---|---|---|---|---|
  | 天数 tianshu | 4.138 | e2 4.17075 | -0.78% | `_iluvatar` 冻结字节（163333ms，vs e2 同芯 220720ms） |
  | 沐曦 muxi | 2.1715 | 1.920 | **+13.10%**（>回滚线 1.824；距对手带线 2.181 差 0.0095 未进带） | generic（29402ms） |
  | 海光 haiguang | 3.45875 | 3.1485 | **+9.85%**（>2.9911） | generic（9173ms） |
  | 昆仑 kunlunxin | 0.56975 | e2 0.556 | +2.47% | `_kunlunxin` 冻结字节（9886ms） |
  | 华为 huawei | 0.50075 | 0.824 | **-39.23%（<0.7828，破回滚线）** | generic（28872ms） |
  | card_a | 1.44175 | 1.236 | **+16.65%**（>1.1742） | generic（6705ms） |
  | card_b | 1.40175 | 1.29975 | **+7.85%**（>1.23476） | generic（19063ms） |
- **均值验算**：(4.138+2.1715+3.45875+0.56975+0.50075+1.44175+
  1.40175)/7 = 13.68225/7 = 1.954607143，与平台 1.95460714 一致
  （截位）；vs e2 平台实读 avg 1.88703571 = **+3.58%**，vs 账本 e2
  记读 1.89 = +3.42%。
- **预注册门判决（判据=E3 预注册门节原文，发射后未改；本节只记录门
  触发事实，判决执行归 orchestrator/上膛会话）**：
  1. **回滚门触发**：华为 0.50075 < 0.7828（-39.23%；数值/编译均无
     失败，纯读数破 -5% 线）→ 按门文本 **generic 回滚 e2 字节**
     （`db53db9a196e3fb08206725bf945c34aac22dec186f119fa0d0bab913c286e52`
     @fde2d02c）；
  2. 判轴门：华为 ≥1.1 与 ≥1.40525 均未达（0.50075）——per-row 轴在
     华为 generic 臂未兑现反大幅回退；沐曦 ≥2.181 未达（2.1715，差
     0.0095）；
  3. **换 TB 门触发**：均值 1.95460714 > 1.89，且本次 status 实读
     is_team_best=True——E3 预注册门节披露的平台 my_best 双快照仍读
     1.72675 的口径差异已由本次实读对齐（留观项闭环）；
  4. 隔离冻结芯：天数 -0.78% / 昆仑 +2.47% 噪声带内 → 只排查不回滚。
- **混合结局读数**：五 rider 芯四升（沐曦 +13.10% / 海光 +9.85% /
  card_a +16.65% / card_b +7.85%）一大幅回退（华为 -39.23%）；均值与
  TB 双升。平台 TB 已由 21628（e3 字节）取得，回滚树字节不撤销该提交，
  但下一发回到 e2 generic 字节将同时回吐四芯增益与均值——回滚执行与
  「整臂回滚 vs 保均值」取舍归 orchestrator（预注册门文本为回滚，未
  申报保留选项，如实留痕）。
