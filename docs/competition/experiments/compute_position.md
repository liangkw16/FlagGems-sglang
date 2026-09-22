# Task 77 `compute_position` 实验记录

```current
task: 77
operator: compute_position
batch: 6
validity: valid(8/8,e3,1051.15x TB)
platform: e5(20169)invalid_correctness:华为数值100%失配(int64布局假设在ascend端不成立?);其余七芯强天数2694/muxi630/haig1628/kunl55;TB 1051.15(e3)守
candidate_stage: e5
team_best_stage: e3
team_best_speedup: 1051.15x
sealed: no
next: 华为152vs682仍为最大缺口(需ascend专属形态,flat二分在ascend编译失败);天数2696vs4338;树已回滚e3字节,e4思路封存
updated: 2026-09-23
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


## 2026-09-22 S0/E2 平台终态：344.34 有效上榜（未提交题补位成功）

- S0（19369）：**7/8 燧原被判反作弊扫描拒绝**——"Module-level mutable
  container detected: '_packed_cache'. Global dict/set variables can cache
  results across benchmark iterations"。布局假设根本没被执行；教训：探针
  缓存这类全局可变容器直接踩扫描规则。
- E2（19383，`8a514f1`，去全局缓存改每次调用探针）：**8/8 valid 344.34**。
  逐芯：天数 584.9 / 沐曦 160.0 / **燧原 51.7（B 布局打包写平台验证
  成立）** / 海光 453.8 / 昆仑 6.75 / 华为 217.5 / A 649.1 / B 630.9。
  回执 `day5prep-20260921/compute_position-e2/`；ZIP
  `compute_position/e2-8a514f1/`。
- 榜首 EvokeAgent 1731（沐曦 1181/天数 4338 为大头）。后续轴：沐曦、
  昆仑 6.75→64、燧原 51.7→427。per-call 探针成本（~100µs）在沐曦/天数
  高分下不构成瓶颈的读数成立。


## 2026-09-22 E3 平台终态：1051.15 新 TB（+205%，全并行结构引爆）

- E3（19587，`405928ad`，codex-review P2 修复后发射：i64 扫描偏移 +
  int32 契约输出截断同 reference）：8/8 valid **1051.15**（344.34 起
  +205%）。逐芯：天数 2696 / 沐曦 702 / 燧原 116.3 / 海光 1582 /
  昆仑 55.1 / 华为 152.3 / A 1602 / B 1503。K1 单程序向量化 cumsum +
  K2 每请求一程序预 start 平铺——O(bs²) 串行链移除后大带宽芯全面
  4-8x（天数 4.6x / 沐曦 4.4x / 昆仑 8x / 海光 3.5x）。
- 华为 217→152 回退：ascend 对 bs 个小程序的 launch 或两段开销，
  榜首 682 说明形态仍差——下一轴。


## 2026-09-22 E4 平台终态：flat+二分实验判负，e3 字节回滚

- E4（19621）：flat 位置网格 + 每元素无分支二分。**多数大带宽芯回退
  30-50%**（天数 2696→1778 / 沐曦 702→461 / 海光 1582→1209 / 昆仑
  55→9.4）——二分的 log2(bs) 次掩码加载开销远超负载均衡收益；
  华为 static_range 展开编译失败（7/8 后判 invalid）。负结果入档：
  此题负载倾斜不是瓶颈，e3 的每请求并行形态即优。


## 2026-09-22 深夜 E5 上膛（09-23 午夜首发第 3 发）

- E5（`423699e`，新增 ascend vendor）：e3 两 kernel 核心 + fill 网格
  封 64 程序请求跨步（block↔核强绑定，e2 striped ≤64 程序时华为曾读
  217）+ 掩码 fp32 比较（Vector CMP 无整数）。codex-review 零发现
  （14 边界 batch 调度模型恰一次覆盖）；release 绿；ZIP
  `compute_position/e5-423699e/`；回执 `.../compute_position-e5/`。
  预注册门：华为 ≥300 视为带突破；均值 >1051.15 换 TB。
