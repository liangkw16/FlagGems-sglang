# Task 77 `compute_position` 实验记录

```current
task: 77
operator: compute_position
batch: 6
validity: candidate(未提交;代理双源全绿)
platform: 未提交
candidate_stage: s0
team_best_stage: -
team_best_speedup: -
sealed: no
next: s0 待发(09-22 首发);燧原 int64 布局以首发为探针(B=打包/int64=标准);8/8 即成功,仅燧原数值失败则换双词形态补一发
updated: 2026-09-21
```

## 过程摘要（2026-09-18 开发，09-21 补燧原 vendor 定稿）

- 题面：[Task 77](../tasks/batch-6/77-compute_position.md)。extend batch 的
  fused position + start：`start = exclusive_cumsum(extend_seq_lens)`（int32
  `[bs]`），`positions[start[i]:start[i]+s_i] = p_i + arange(s_i)`（int64
  `[sum]`）。精确整数；prefix 可空。reference 是 **Python for 循环逐段
  arange**（bs 次 launch + host 循环）→ 榜首 EvokeAgent 1731.16x 的来源；
  16 队达标，纯增量空白题。
- generic（09-18，`721476e7` 家族 + codex-ask 两轮修复）：双路径——小批
  （bs<64）每请求一程序；大批 striped（≤64 stripes × ROWS_PER_STRIPE=16，
  超 64 自动扩行），串行链 O(min(bs,64)²)。边界矩阵含 bs=1/63/64/65/1023/
  1024/1025/2049、零长度段、空 prefix、513×65、int32 边界 prefix（2^31-4
  跨界，`a47602d2` 补 RELEASE_REQUIRED_TESTS）。
- 燧原墙与解法（09-21，`ca037504`）：GCU 签名级 i64 禁令（T60 八轮）+
  torch-gcu int64 物理窄化（4N 字节装 N 逻辑元素，t60 审计钉
  `gcu_empty_tensor.cpp:50-64`）+ T60 E8 双词写 100% 失配 ⇒ 窄化打包假设 B。
  vendor 内一次性 host 探针（arange(4) 读 int32 view：`[0,1,2,3]`=B 打包 /
  `[0,0,1,0]`=A 标准，模块级缓存）分发：B→int32 打包内核（元素 i 写 view
  下标 i，4N 字节物理界内）；A→int64 标准内核。代理走 A 分支 ⇒ 数值矩阵
  双源可完整验证；GCU 走 B。两路均为 Triton（T60 clamp_position 的 host
  守卫先例）。launch 几何按官方 gcu300 规则：stripes 封 12、num_warps=2、
  BLOCK_TOKENS=2048。
- 值域说明：B 布局下 >2^31 的 position 值物理不可表示，但平台 reference
  自身（torch.arange int64 同窄化存储）同样不可表示 ⇒ benchmark 不含该
  区间；代理 A 分支保留完整 int64 数学（int32 边界测试在代理通过）。

## 验证与产物（09-21 晚）

- screening（`3dd3b342` 工作树字节）：T77 双源（generic+enflame）4 测试
  0 失败 0 skip，generic 20 次 / enflame 真实 launch；回执
  `/tmp/flagos-t77-screen2/verification.json`（远端）。
- release（HEAD `eb1e2c6d`，source=`ca037504` 家族字节）：4 测试 0 失败，
  generic/enflame 各 20 次非 warmup launch；回执
  `artifacts/competition/day5prep-20260921/compute_position/verification.json`
  SHA `da5f7ed3f98615dc4e19687ee3e26554fb93e4131075c25e72af5ca9923ad81d`；
  ZIP `artifacts/competition/compute_position/s0-eb1e2c6/compute_position.zip`
  （成员 generic+enflame）SHA
  `32073a813138087473937abd447a1c79aecfba80b5af6b9ce1edecba8ad0e402`。
- codex-review：09-21 `--uncommitted` 轮对本 vendor 无发现；T78 P1 教训
  （输出侧 rope 间隔）在本题为 N/A（positions 单输出连续）。

## 靶子与下一步

- 榜首 EvokeAgent 1731.16（天数 4338/沐曦 1181/燧原 427/海光 2562/昆仑
  64/华为 682/A 2426/B 2170，09-21 00:13 快照）。我方 striped 结构带宽
  型任务应落在同量级；昆仑 64 与燧原 427 为后续轴。
- 发射预案（09-22 00:01 首发）：s0 = generic + enflame vendor 双成员 ZIP。
  预注册门：8/8 任意有效=成功；仅燧原数值失败 = 假设 A 成立，e2 换双词
  （A 布局）形态补一发。
