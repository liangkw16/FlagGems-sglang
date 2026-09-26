# Task 98 `get_mla_kv_buffer` 实验记录

```current
task: 98
operator: get_mla_kv_buffer
batch: 7
validity: valid(7/7,e1)
platform: e4(21514)valid但昆仑0.131<0.350触发预注册回滚门(sglang行形态窄向量倒挂第三次复现);华为0.726;TB保e1(avg1.669,昆仑0.350#2)
candidate_stage: e5(dev就绪待验证:天数iluvatar单行骨架+T96三变量包,轮1开发员;源码+回归已commit,NVIDIA代理release/ZIP/发射待上膛轮)
team_best_stage: e1
team_best_speedup: -
sealed: yes
next: e5评审卡点(两轮同因py_compile账本.md)→r3=评审范围更正交接非新候选(见E5咨询节),过门后天数轴单发裁决(预期avg1.34-1.38>1.328可翻榜);昆仑墙三证不动,芯走已回滚地板字节;发射须--proxy-vendor iluvatar ascend kunlunxin
updated: 2026-09-26
```

## 契约

- `(kv_buffer, loc, cache_k_nope, cache_k_rope)`：按 `loc[i]` gather 行并拆 NoPE/RoPE 两半；
  `cache_k_nope/rope` 仅作 dtype/shape 模板（返回两个新张量，store 隐式转换 RTNE）；exact。

## 榜单靶标

EvokeAgent 4.59 分：muxi 1.418 / haiguang 1.886 / **kunlun 0.131** / **huawei 0.508** 全芯第一
（昆仑四队全部 ~0.13 = 结构性踩坑）；tianshu 第一 2.911（EA 自己仅 2.412 排 #7，幽灵池有界）。
任意正常行 gather 即可昆仑 #1（>0.131）+ 华为 #1（>0.508）。

## S0 结构（commit b6641097）

- 单 kernel：BLOCK_R=8 行瓦片，loc 一次 load；行内 BLOCK_C=512 列向量两段
  （NoPE 段 + RoPE 段偏移 nope_dim）；store 时 `.to(dtype)` 转换；i64 寻址；num_warps=8。
- 代理 sweep @n=16384：torch 基线 56.69µs vs 我方 13.65µs（**4.1x**）；r8c512w8 最优。

## 证据

- NVIDIA 代理 screening（r16 旧配置）6 测试 0 失败 19 launch；r8 新配置复筛中。
- 昆仑 0.131 之谜假设：torch 参考在昆仑走 vendor gather+cast 4-5 kernel 全速，
  Triton gather 在昆仑（XPU 端口）慢；单 kernel 减 launch 数是正确方向，昆仑专配
  待 KernelGen kunlun 后端恢复（请求已归档 `artifacts/competition/batch7-kernelgen-20260924/t98-kunlun-tune-req.json`）。

## 情报

- 快照 SHA-256 `061475d31505bb81f16a5a212fa7cfd6e4b1142188cb02da2c202962efedadb2`。

## 不可变身份（s0 v2，2026-09-24，契约 spec 高阶复审后）

- source commit = verification commit = `daf77923f6913447b65a8c226f34297ce13b4440`（review v1 零宽半边已修；review v2：补 kv_buffer.stride(1) 与 loc.stride(0)）。
- ZIP：`artifacts/competition/get_mla_kv_buffer/s0-daf7792/get_mla_kv_buffer.zip`，SHA-256 `1d6e5fe55092e2a1081283b1448c1b918480af1c22b229c9bfa07a728f017002`（单成员 `get_mla_kv_buffer.py`，generic-only）。
- release 回执：`artifacts/competition/b7-s0-release2-20260924/get_mla_kv_buffer/verification.json`（mode=release，exit 0，绑定该 commit 字节），SHA-256 `64ff8b794bf824abcab0e096a9d5eb6848008f6391972532756373d89fd5bcce`；
  日志 SHA-256 `bc5213f527c79040273dd9e5c225fde3ccc97d24a10c4493f105e8f3df74fa16`。NVIDIA RTX 5070 Ti / torch 2.13.0+cu130 / triton 3.7.1。
- 提交脚本：`artifacts/competition/b7-s0-release2-20260924/submit-batch7-s0.sh`（v2，全部参数预烘焙）。
- review 历史：v1（commit 级，gpt-6-astra medium）4 项全修；v2（--base 对照契约 spec，high）5 项 P2 全修；
  三轮修复均经 screening + release 双门禁复跑全绿。

## E4 上膛（上膛员，2026-09-26；候选=轮2研究员-昆仑臂 commit `2f071f41`）

**候选**：昆仑臂换骨架为上游 sglang 官方单行 kernel 形状（2026-09-26 经 gh api
复核 `sgl-project/sglang` `mla_buffer.py`：`grid=(n,)` 每程序一行、标量 loc load
铸 i64 行基、constexpr 维度/步长、两段 exact-width `tl.arange` 零 mask 零循环），
host 形状分支限定 pow2 宽度 ≤65536 lane / kv 列步长=1 / loc 步长=1 / n≤65535，
其余形状保留 e1 masked-rows kernel（`aef2c1f1` 回滚档 0.350 已证字节，kernel 体
与 launch 参数不动，仅插入分支）。字节谱系核对：`_kunlunxin` blob
`2f071f41~1` = `aef2c1f1` = `1a7e55ed` = `75eb1301…`（地板），`2f071f41` =
`1c564ceb` = `00fd72ea…`（本候选唯一改动）；`2f071f41→1c564ceb` 本题
generic/_ascend/_kunlunxin 三源 + 两测试文件 diff 为空（其后 5 个 commit 均为
T96/T97 纯 docs），上膛字节=研究提交字节。

**远端 release 回执**（source=verification=`1c564ceb`，prepare 前后 HEAD 复核一致）：
`artifacts/competition/day5prep-20260921/轮2-研究员-T98（昆仑臂）-wf/verification.json`
（SHA-256 `1260d4c62bf29960686ffef5afcac33df3620fcd03b525fd4ae21c1e200ae8cd`）与
`verification.log`（SHA-256 `36eb05fe6dc9d7951ce6301c438baa3a9129c2fc8fac972c3cecfe951f4ca436`，
与回执内 log_sha256 逐字一致）。mode=release、exit_code=0、NVIDIA RTX 5070 Ti /
torch 2.13.0+cu130 / triton 3.7.1、proxy_vendors=[ascend, kunlunxin]、
`timeout 900` 内完成。**14 测试 / 144 case 全过**（RELEASE_REQUIRED_TESTS 14 项
——含 5 个新 `test_row_form_*` 回归——与 expected_tests 集合全等且全 passed），
0 failures / 0 errors / 0 skipped / 0 expected_failures / 0 unexpected_successes；
generic/_ascend/_kunlunxin 三源各 **36 entry calls / 35 kernel launches**、各
108 条非空张量 (path,dtype,shape) 覆盖（实际 kernel 执行，非空 bfloat16）。

**不可变 ZIP**：`artifacts/competition/get_mla_kv_buffer/e4-1c564ce/get_mla_kv_buffer.zip`，
13999 字节，zip_sha256 = canonical_zip_sha256 =
`a1820da000d9c852be445576a985fbff05eac7a28d1768f176c00aad2984a058`，≠ e3
`30498e69ee3c7545be43389e4cc97f385b5a2b8e019f74d69b39f3bfb2d0d078` ≠ e2
`67983fe5e6f4e8d5a5907ea77e1db274f059c6161aaa5675f5e0e85682a1f624` ≠ e1
`37cec89bc9e2883867189a72a23c62a13f57d678d226b4b4855409222aa8b221` /
`72435f9beed964242cc20313bade9bf958d8caa3bef3a9bebe5924d976fe39e5` ≠ s0
`1d6e5fe55092e2a1081283b1448c1b918480af1c22b229c9bfa07a728f017002` /
`a3643b550f444e117ca19075d1084fdf71551bcc4766bc129e84f2df0ac74906`
（平台元组去重键 zip_sha256，新候选字节成立）。

- **五元组（上膛身份）**：
  - source_commit：`1c564ceb16e6ebb1053689630d48f250a0f685ec`
  - verification_commit：`1c564ceb16e6ebb1053689630d48f250a0f685ec`（=本回执，
    发射 preflight 要求 commit 字段等于它）
  - ledger_commit：本 commit（E4 上膛记账）
  - ZIP：`artifacts/competition/get_mla_kv_buffer/e4-1c564ce/get_mla_kv_buffer.zip`
    （13999 字节，哈希见上；vs e3 仅 `_kunlunxin` 成员重写，generic 与
    `_ascend` 成员逐字节冻结——单变量纪律成立）
  - stage：`e4`（CURRENT 旧值 `candidate_stage: s0` 为 e1/e2/e3 出包后未回写的
    陈旧值；按产物连续性 s0→e1→e2→e3 已全部存在取下一号，与研究提交
    `2f071f41` 预注册命名「T98 e4」一致；无既有 e4 ZIP，不跳号不撞号）
  - ZIP 成员名单（与 `zipfile.namelist()` 实际核对一致，无夹带；`unzip -t`
    无错；仅 UTF-8 `.py`，无测试/缓存/目录前缀/macOS 垃圾；成员字节 = git
    blob @1c564ceb = 回执 files 哈希 = 远端实际执行字节）：

    | 成员 | 字节 | sha256 | 变更说明 |
    |---|---|---|---|
    | `get_mla_kv_buffer.py` | 2580 | `eefba286fbbc29f55c3a90e27b0eff29762e9082132629bfb2832cf0fe703a2a` | 自 s0 review `daf77923` 起冻结（= e3 同哈希） |
    | `get_mla_kv_buffer_ascend.py` | 3758 | `3b698faefc68a9970c802c6915749a084b566cf55a4f38d166b112df33e7a793` | = e3 同哈希冻结（5cfdb65 引入的 persistent gather-split） |
    | `get_mla_kv_buffer_kunlunxin.py` | 7257 | `00fd72ea973e5de3a7fff00a5c378921a7957519cb2bf4c2e261a8f73e4aba3d` | 本候选唯一重写：轮2单行骨架 + host 形状分支，保留 aef2c1f1 地板行为 e1 masked-rows 路（e3 ZIP 内为失误带出的 exact-width `bf3f356e…`，树内已由 `aef2c1f1` 修正为地板 `75eb1301…`） |

**发射命令约束**：verify/发射须带 `--proxy-vendor ascend --proxy-vendor kunlunxin`。

**预注册门**（沿用 vendor docstring 预注册，一字未改；按平台昆仑 speedup 裁决）：
<0.350 → 回滚 `_kunlunxin` 至 `aef2c1f1` 字节（blob `75eb1301…`）；0.350–0.49 →
保留观察；≥0.49（过 eatabigwatermelon 0.4905）→ 骨架轴判兑现；~0.976 为
OpeGoodn 天花板。若窄向量倒挂复现，备用 E4b = 同骨架 + 单次 1024 宽整行 load +
两次移位 store（load lane 减半）。armed-unfired：昆仑/华为 target-runtime-
unverified 状态不变（NVIDIA 代理不背书目标芯，vendor docstring 亦自记「Not
compiled on kunlunxin hardware locally」），发射 preflight 通过后执行一次性
submit，发射后以平台 status/watch JSON 回写逐芯结果。

## E5 开发（轮1开发员-T98-e5-iluvatar-perrow-1d，2026-09-26）

**候选**：天数（iluvatar）装 per-row 1D vendor——上游 sglang 官方单行骨架
（2026-09-26 经 gh api 复核 `sgl-project/sglang`
`python/sglang/kernels/ops/kvcache/mla_buffer.py::get_mla_kv_buffer_kernel`：
`grid=(n_loc,)` 每程序一行、标量 loc load、constexpr 维度/步长、两段
exact-width `tl.arange` 零 mask 零循环）+ T96 已平台兑现的三变量包（int32
寻址域守卫 + nope/rope 单 alloc 双 view；T96 fused_pack_qkv e2 平台天数
2.95→4.17 +41%，本题为该包第二个载体），替换 generic 的 (8,512) 2D masked
全链 i64 行瓦片（`src/flaggems_sglang/ops/get_mla_kv_buffer.py:29-53` 五处
`.to(tl.int64)`）。

**缺口结构**（快照 climb-loop.json s2t1op098 @2026-09-26T15:15:23，本会话实读）：
EA（EvokeAgent）#1 avg 1.32814286，天数 2.7945（芯内 #4）；我方 #6 avg 1.212，
天数 2.305（芯内 #25）/muxi 1.37133333/kunlunxin 0.13016667/huawei 0.7695。
EA−我 天数 +0.4895（+21.2%）为单芯最大；天数榜首 c2flow 2.911、Sweetdeath
2.88783333 佐证 2.9+ 可达。÷7 算术：天数 +30~41% → avg +0.099~0.135，昆仑
地板恢复 0.130→0.350 → +0.031，合计 +0.13~0.166（申报 0.13，带内中低段）；
单发可翻榜（1.212+0.13>1.328）。成功率支撑：同芯同构三变量包 T96 E2 平台
兑现 +41%（唯一在天数平台兑现过的同族先例，其机制证据 4 条中 2 条直接点名
本题——T98 2D int64 2.37 vs c2flow 2.91、int64 全链天数系统性落后，
`fused_pack_qkv.md` CURRENT 与机制证据节本会话复核）；T51 E9 行块在天数
-20% 佐证 per-row 偏好。

**实现**（`src/flaggems_sglang/runtime/backend/_iluvatar/ops/get_mla_kv_buffer.py`，
新增文件，本 commit）：
- `_mla_split_row_i32_kernel` / `_mla_split_row_i64_kernel` 双 Triton 孪生：
  上游单行骨架逐行镜像（标量 loc、constexpr KV_S0/NOPE_DIM/ROPE_DIM/双出
  步长、exact-width arange 零 mask）；i32 版 `row/idx` 全链不升 i64，
  i64 孪生承接溢出域（公开测试矩阵不可达，镜像 e4 已证 i64 行数学）。
  num_warps=4（上游默认档，与 T96 兑现 launch 同值）。
- `_use_int32(kv_rows, kv_s0, total_dim, n, nope_dim, rope_dim, loc_s0)`
  数值域守卫：源深 `(kv_rows-1)*kv_s0+total_dim < 2^31`（r2 教训：按
  stride 深度而非 numel——行步长视图 kv_s0>total_dim 时 numel 低估深度）；
  输出侧 `n*nope_dim` / `n*rope_dim` 各自独立 < 2^31（纯 gather 允许重复
  索引，输出不由输入规模导出）；loc 深 `(n-1)*loc_s0 < 2^31`。
- `_upstream_row_form(n, nope_dim, rope_dim, loc_s0, kv_s1)`：host 形状
  分支，与 e4 昆仑臂同一谓词同一签名（pow2 宽度 ≤65536 lane、kv 列步长=1、
  loc 步长=1、1≤n≤65535）。
- `_mla_rows_kernel`：generic 逐字节镜像回退（(8,512) 2D masked i64 全链，
  launch 参数同 generic）——不满足行形态的一切形状落回平台已在天数读过
  2.21-2.37 的字节，单变量纪律（vs e4：generic/_ascend/_kunlunxin 三源零改动）。
- 单 alloc 双 view：nope/rope 同 dtype 时一份平坦 buffer 两个连续 view
  （2×torch.empty→1，T77/T96 先例）；dtype 域分支——nope/rope dtype 交叉
  （`test_cross_dtype_store` 三组合）回退双 empty。
- 反作弊合规：int32 走数值域守卫非设备判断；无 try/except fallback；
  无模块级可变容器。

**测试**（`tests/test_get_mla_kv_buffer.py`，本 commit）：新
`test_iluvatar_int32_domain_guard`（守卫真值表 12 例：源深/行步长深/输出
两侧/loc 深各自边界，T96 同名测试模式）加入 RELEASE_REQUIRED_TESTS（现
15 项）；e4 五项 `test_row_form_*` 回归原样复用（谓词签名一致，现由本
vendor 的 `_upstream_row_form` 承接——本会话核实回滚后的 `_kunlunxin`
树内字节 = aef2c1f1 地板（blob `05459d36`），不再暴露该谓词）；全矩阵经
`tests/_op_variants.py` 自动装载新模块（generic/_ascend/_kunlunxin/
_iluvatar 四源同跑）。本地无 triton/torch，CUDA 数值矩阵未在本机执行
（远端 release 阶段补齐）；本会话已跑：`python3 -m py_compile` 两源码文件
通过，stub 导入验证谓词 10/10、守卫 12/12 真值表与平台形状族（n≤4096、
512/64、单位步长）全部走 i32 单行 kernel。

**预注册门**（按平台天数 speedup 裁决）：≥2.9（超 EA 2.7945）判轴兑现；
≥4.0 进场带（T96 兑现带）；<2.19（-5% vs 2.305）或任何数值失败 → 删本
vendor 文件，天数回 generic 字节（e1 地板组合）；其余六芯 vendor 隔离读数
应不变（昆仑走已回滚地板字节，发射背景本身含 0.130→0.350 恢复）。发射
命令必须 `--proxy-vendor iluvatar --proxy-vendor ascend --proxy-vendor
kunlunxin`。备选臂（预注册，rope 段 64-lane 窄段风险——天数无 exact-width
前科亦无窄段正证据）：同 vendor 改整行 1024-lane masked 单 load + 两次
移位 store（账本 E4b 同思路，load lane 减半）。归因补充预留：generic-i32
单变量（T96 同样打包，E3 位）。天数 lowering target-runtime-unverified
（NVIDIA 代理只验数学/JIT），平台单发裁决。

## E5 评审卡点与新方向（轮1咨询员 codex-ask，2026-09-26）

**卡点**：e5 评审两轮同因否决——评审员对
`docs/competition/experiments/get_mla_kv_buffer.md` line 14
（`updated: 2026-09-26`）执行 `python3 -m py_compile`，报 leading-zeros
SyntaxError。本会话复现同错；未触碰的 T96 `fused_pack_qkv.md` 同样不过
（line 13 U+2192）→ **类别错误**：prose markdown 恒不可能通过 py_compile。
项目自证口径：`references/remote-validation.md:89`「目标源码和测试的
py_compile」——py_compile 范围本就限定为源码与测试。

**r2 论断修正（本会话 `git diff-tree` 逐 commit 实测）**：1a6c6dfc 声称
「先例轮全为 .py-only」不成立——`ef74f308`/`fde2d02c`（T96 E2 r1/r2）、
`87d0e627`/`3a1fe387`/`7d87c4cc`（T97 e3/e4）五个先例评审轮的 commit
都改了账本 .md 且评审通过；只有 `2f071f41`（T98 e4）纯 .py。→ e5 评审员
对 .md 编译并据以否决是相对五个先例的**异常行为**，不是系列不变量。
r2「只 touch .py」无效（第二轮意见逐字相同）→ 评审员文件集不是「最新
commit 的 diff」；存活假设：H-cumulative（候选系列累计 diff
`7ee1f7a6..HEAD`，含 r1 账本）或 H-canonical（任务规范文件集，账本恒在）。
评审员不读 commit message（r2 整段论证零效果）。

**codex-ask 裁决**（ask-codex.sh，gpt-6-astra / reasoning_effort=high /
read-only，问题文件含候选假设、两轮 diff、两轮评审意见、逐芯证据；
`remote-validation.md:89` 引文与「基线账本失败在 line 9、现版在 line 14」
诊断特征均已本地核实）：

- **不做账本回滚**：不把 CURRENT/E5 节恢复到 `7ee1f7a6` 基线字节（若回滚
  后评审仍报 line 14 = 旧诊断/旧对象复用，不能再归因于回滚未做好）。追加
  独立记账 commit 无隔离作用——仍在累计评审窗口内，账本照旧被选中。
- **r3 = 评审范围与证据更正，不是新性能候选**：把类别错误反驳 + 检查契约
  写进**轮间交接字段**（实际传给评审员的材料），首段模板：「本轮候选实现
  未变。上一轮失败对象 `docs/competition/experiments/get_mla_kv_buffer.md`
  属账本文档；`remote-validation.md:89` 将 py_compile 范围规定为目标源码和
  测试。请保留账本内容审查，将编译输入限定为 Python 源码、测试及相关
  Python 依赖。请在本轮结果中记录实际检查的 commit、文件集生成规则/base、
  完整编译 argv、退出码与失败文件哈希，以确认本轮确实检查了当前候选。」
  随附：r3 commit 号、Python 清单（generic + 三 vendor + 规范测试 + 评审
  入口）、vendor 字节仍等于 `23d4561d` 的 `git diff --exit-code` 证据、
  累计 diff 全部 13 个 .py 逐一 py_compile 通过记录（本会话已实测全绿）、
  五个先例 commit 文件列表反例。
- 若工作流强制 r3 产生新 commit：只更新
  `tests/test_T98-e5-iluvatar-perrow-1d.py` 的 docstring（记录本轮静态验证
  范围与复现命令），不改导入与测试行为；不新增 .md 文件；关键说明不只放
  commit message。
- **成功判据（把 r3 设计成有信息增益的一轮）**：评审绑定当前 r3 commit、
  实际编译输入无 .md、后续意见指向真实源码/测试/门禁。若第三轮仍报同一句：
  不再做 r4 空修复（不加评审入口、不改日期），只向工作流主控上报四项诊断
  ——实际 evaluated commit / 文件集生成规则 / py_compile argv / 失败文件
  SHA-256；机械固定且输入范围不可改时，现有约束下不存在保证通过的干净
  修复，应精确定位工作流阻断而非继续消耗候选轮次。
- 长期方向：显式分离「审阅清单」与「语言检查清单」（账本照常审阅，Python
  只编译 Python）；若证实检查按净差异工作且工作流允许指定评审 base/head，
  可用独立评审分支/worktree 让账本先存在于评审基线、再只应用冻结的 .py
  差异（该前提目前无证据，待查）。

**代码面无需改动**：累计 diff（`7ee1f7a6..HEAD`）全部 13 个 .py 本会话
逐一 `python3 -m py_compile` 全绿；vendor/规范测试冻结于 `23d4561d`
（`git diff --exit-code 23d4561d HEAD -- <vendor> <tests>` 为空）。评审
通过后走既有上膛流程：远端 NVIDIA release 回执 → 不可变 ZIP →
`--proxy-vendor iluvatar ascend kunlunxin` 平台单发裁决，预注册门不变。
