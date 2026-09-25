# Task 98 `get_mla_kv_buffer` 实验记录

```current
task: 98
operator: get_mla_kv_buffer
batch: 7
validity: valid(7/7,e1)
platform: e3(21269)valid:ascend persistent vendor+7%(华为0.722->0.770)但昆仑vendor字节错误带精确宽(0.130)已回滚1024地板;TB保e1(avg1.669);教训:打包前未核对vendor字节
candidate_stage: e4(armed未发射:昆仑单行骨架上行形状,轮2臂2f071f41;release@1c564ceb全绿14测试/144case零skip;ZIP e4-1c564ce sha256 a1820da0)
team_best_stage: e1(在途)
team_best_speedup: -
sealed: yes
next: E4上膛完成待发射(armed-unfired,回执绑定1c564ceb,ZIP e4-1c564ce sha256 a1820da0,五元组见E4上膛节;14/14全绿零skip,--proxy-vendor ascend+kunlunxin);门:昆仑<0.350回滚_kunlunxin至aef2c1f1字节,0.350-0.49观察,>=0.49判骨架轴兑现,0.976天花板,E4b备用
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
