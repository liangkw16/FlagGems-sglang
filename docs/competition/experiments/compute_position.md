# Task 77 `compute_position` 实验记录

```current
task: 77
operator: compute_position
batch: 6
validity: valid(8/8,e9,1233.7119x TB)
platform: e9(20739)valid 8/8 avg 1233.7119新TB(旧1100.744075=e6);昆仑105.223回水位带;ts3089.8/muxi704.3/hg1626.0/A2088.8/B1999.1;燧原115.4(vendor)/华为141.1
candidate_stage: e9
team_best_stage: e9
team_best_speedup: 1233.7119
sealed: no
next: 距c2flow 1747.5差513.8;下一轴e10=无循环2D grid平铺(攻华为141vs612与天数3090vs4773的fill吞吐/动态循环/并行度缺口);enflame vendor单launch化(115vs483)备选
updated: 2026-09-24
```

## 2026-09-24 E9 平台终态（20739）：8/8 valid 1233.71 新 TB（+12.1%），昆仑 vendor 隔离兑现

- E9（submission **20739**，daily_seq 3，created 2026-09-24T07:2x+08，
  commit `375fa767`）：completed/valid，8/8，均值 **1233.7119** 新
  TB（+133.0 vs e6 1100.744075），is_team_best=true，rank 10→8。
- 逐芯：天数 **3089.77** / 沐曦 704.28 / 燧原 115.39（vendor 未动）/
  海光 **1625.96** / **昆仑 105.223**（vendor 隔离成功，e6 水位带
  105.25 精确复现）/ 华为 141.14 / **A 2088.78** / **B 1999.15**。
- 预注册门核对：8/8 ✓；昆仑 ~105 带 ✓；均值 >1100.744 ✓（+12.1%）。
- 剩余缺口归因（vs c2flow 1747.5001，需 +4110 sum）：天数 +1683 /
  海光 +899 / 华为 +471（141→612，4.3x）/ 燧原 +368（vendor，115→483）/
  A +300 / 沐曦 +257 / B +107 / 昆仑 +25。华为与天数为主要结构缺口，
  假说：fill 的 `range(0, seq_len, BLOCK)` 动态 trip-count 循环 +
  grid=(batch,) 在小 batch 下 CTA 并行度不足 → e10 无循环 2D grid。

## 2026-09-24 E8 平台终态（20738）：7/8——昆仑 fused 自扫数值错，单 launch 方向在七芯兑现

- E8（submission **20738**，daily_seq 2，created 2026-09-24T07:09:31+08，
  commit `c03cb3cc`）：completed/**invalid_correctness**（昆仑失败），TB 守
  e6 1100.744075。
- 逐芯（vs e6）：天数 **3206.93**（2927.56，+9.5%）/ 沐曦 687.41（674.68）/
  燧原 115.17（116.42，vendor 未动）/ 海光 1616.69（1615.02，持平）/
  **昆仑 None（失败）**/ 华为 146.58（147.71，持平）/ **A 2036.91**
  （1687.14，+20.7%）/ **B 2036.14**（1532.17，+32.9%）。
- 昆仑失败指纹（raw_result）：`test_compute_position[0/4/5]` 断言
  mismatch 63-81%，第二输出（int32 extend_start_loc）出现垃圾大值——
  fused 路径的 `tl.sum(tl.where(lanes < program_id, seg, 0), 0)` 标量
  program_id 广播在 XPU 上错算（与向量整除、标量/向量混编同族的 XPU
  lowering 缺陷）。单 launch 方向本身在七个芯兑现（B +33% 为最大）。
- E9（commit `375fa767`）：generic e8 字节不动，新增
  `compute_position_kunlunxin.py` vendor = e6 generic 函数体逐字节
  （双 launch + low/high 拆分），隔离昆仑。codex-review 一条 P2
  （tools/select_tests.py 不映射 vendor 路径→CI 不触发，属仓内 CI 覆盖
  缺口，非提交门禁；release runner 以 `--proxy-vendor kunlunxin` 执行
  vendor 字节补执行证据）。
- 预注册门（e9）：8/8 正确；昆仑回到 ~105 水位带；均值 >1100.744 换 TB
  （七芯 e8 读数 + 昆仑 105 外推 ≈1243，+13%）。

## 2026-09-24 E7 平台终态（20655）：8/8 有效但均分 1021.7663，低于 E6 最佳

- 00:03:15+08 单次正式提交，八芯 00:05:16+08 全部 completed/valid；
  均值 **1021.7663 < e6 1100.744075**，按预注册门判负（-78.977775，
  -7.17%）。逐芯：天数 2403.631 / 沐曦 724.182 / 燧原 116.9506 /
  海光 1552.9658 / 昆仑 51.3672 / 华为 218.8826 / A 1627.859 /
  B 1478.2922。相对 e6：华为 +48.2%、沐曦 +7.3%，但天数 -17.9%、
  昆仑 -51.2% 主导净退。NVIDIA 小批 1.14–1.96 倍的代理优势未跨芯兑现。
- 最终 codex-review：`--base c31f5278..7395eea5` 首轮指出性能样本映射和
  T80 账本两项 P2 证据问题；修正提交 `f2788b91` 经 `--commit` 复审为
  “未发现可靠、可复现缺陷”。题面检查未发现可确认违规。两轮 review 均成功退出。
- 00:00:53+08 preflight 绑定账号 15600308080 / 团队 SoulCoder / tid
  `s2t1op077` / source+verification `0e2f173e` / ZIP SHA
  `01bcb2d0…dc8d7` / release SHA `9b970ce4…3628ef4`，返回一次性 nonce。
  首次执行在发送前实时 GET 网络超时，intent 仍 `prepared`，没有上传；
  00:02:59+08 状态 GET 确认额度30/30、tuple未变，复用同 nonce 后
  **仅一次上传与正式 POST** 成功，平台返回 20655。CLI 提交后远端验签
  因未设置 `FLAGOS_REMOTE_ZIP_HOST` 显示 unavailable；随后用已核实
  `flagos.ks3-cn-beijing.ksyuncs.com` 无认证只读 GET，远端 10,217 B、
  SHA-256 `01bcb2d049e06a921e558f8f4f5e872a262b5f640d3e36df42930ff3ac3dc8d7`
  与本地不可变 ZIP 完全一致。
- 原始 watch `artifacts/competition/compute_position/e7-0e2f173/platform-watch-20655.jsonl`
  SHA `a039c3cc8f26631391bd5a46d5672abe92fbc51d7782010c8fb1e61f1e2baa42`；
  终态账户额度 29/30。源码/测试在 `675e6ec3` 按 e6 不可变 ZIP
  两成员与 09-23 测试字节回滚，generic SHA
  `186ca75b5bfee5c0c10309af0a9f42ee7781e2776c0e4701f1db19612f2a1ff2`、
  enflame SHA `ea06d694d1627bff676c43e4afef0907679949a4f3a05eab362e21681920b714`，
  与 e6 ZIP `c800163c7da6131f0ee620e832dc194cb475d7d085215d7f5143c94f87dd5044` 成员逐字节相同。
- 决策：小批单核 generic 轴关闭。Huawei-only 分派即使完全保留其他 e6
  读数，按本次增益仅约 +8.9 均分，离榜首 1747.5001 仍远；当前冲榜
  额度优先给差距更窄的 T80。

## 2026-09-23 E7 小批量单发射候选（历史开发证据）

- 榜单：23:20 平台全量 GET 快照 `docs/competition/data/batch6-intel-20260923-2323.json`
  SHA `d3c7d96d881f9662f60f7136941d5cc22cc0e8e784e87a0241b4ac73fcde9ca3`；
  榜首 c2flow 1747.5001、我方 e6 1100.744075（第 10），差 646.756025；
  23:20 当日额度 0/30，目标截止 09-24 19:59:59+08。
- 瓶颈与结构：e6 在小批量仍启动 scan/fill 两核并分配共享宽 start 缓冲；
  e7 对 `batch<=512` 复用历史 e2 单核每请求前缀求和，直接写契约 start；
  `batch>512` 保留 e6 双核和宽地址路径。小批量前缀总功为 O(batch²)，
  因此 513 以上不进入单核路径。燧原 vendor 仅 Black 排版变化，
  新旧源码 AST 完全相同；新测试覆盖 512/513、无 prefix 的 513、
  两条路径的 int32 越界 prefix。
- source commit = verification commit
  `0e2f173e56d35173805f35c45bc237c7c472346e`；源码 SHA：generic
  `35d57497d7c02689d914fcf4ee8a5597f14721704e0b5ecf7051988eac0e134f`，
  enflame `d379122b303f0c2e30885b08b6b88718288fa3c958ceedb9f456a05a54ce1008`；
  测试 SHA `1d1416ba221fdfe4385897e3152ed796b11a64a23db28cbb96f10b547cd1abc9`。
- release：`python .agents/skills/flagos-operator-race/scripts/verify_release.py prepare
  compute_position --source-commit 0e2f173e56d35173805f35c45bc237c7c472346e
  --verification-commit 0e2f173e56d35173805f35c45bc237c7c472346e
  --proxy-vendor enflame --directory /tmp/flagos-t77-e7-release-v4`；远端
  `/tmp/flagos-t77-e7-release-v4.QHxUeH`，RTX 5070 Ti / PyTorch 2.13.0+cu130 /
  Triton 3.7.1；`run` 4/4 测试、零失败/错误/skip/xfail，generic/enflame
  各 26 次源码调用、32/50 次非预热 launch；目标芯仍未验证。
  `artifacts/competition/compute_position/e7-0e2f173/verification.json` SHA
  `9b970ce4db0b86d6c7fdac2cc38b6b1c74159ad8464006aa3699a7f893628ef4`，
  相邻 `verification.log` SHA
  `05a5f3e41eeda891f23e3d5fbd5d0603502543737513899977f419db2d627c3b`。
  py_compile、Black、isort、flake8 均过；全部三份代码测试文件远端 SHA 与 Git 一致。
- 五轮交替 AB/BA、wrapper 计时，候选与 e6 等值：远端
  `/tmp/flagos-t77-e7-bench-v2.1dRulD`，脚本 SHA
  `3144a2b941e94b3b0ae433e21ba6b66116d7a5c60bfa126e9428c610e9766b55`，
  `old.py` = e7 候选 = 上列 generic SHA（留存
  `artifacts/competition/compute_position/e7-0e2f173/old.py`，与 Git 源码逐字节相同）；
  `new.py` = e6 对照 SHA
  `186ca75b5bfee5c0c10309af0a9f42ee7781e2776c0e4701f1db19612f2a1ff2`
  （留存 `artifacts/competition/compute_position/e7-0e2f173/new.py`）；
  原始样本 `artifacts/competition/compute_position/e7-0e2f173/bench.out` SHA
  `6de62bf0592ec0d13d4621a9fab2ae938c749558b88e7eaaee3984617c7782d3`。
  `bench.py` 的 `new_us/old_us` 即 e6/e7；中位比：batch 1/16/63/256
  为 1.953/1.909/1.941/1.962，
  512 短/长为 1.291/1.140，513/1024/2049 为 1.001/1.002/1.001；
  无 prefix batch16 为 1.960。以上仅 NVIDIA 代理性能证据。
- 不可变 ZIP：`artifacts/competition/compute_position/e7-0e2f173/compute_position.zip`
  (10,217 bytes)，SHA
  `01bcb2d049e06a921e558f8f4f5e872a262b5f640d3e36df42930ff3ac3dc8d7`；
  成员仅 `compute_position.py` / `compute_position_enflame.py`，分别对应上列
  generic/enflame SHA。dry-run、`--verify-existing` 和 `unzip -t` 一致。
- 预注册：正式平台须 8/8 正确且每芯加速比 ≥0.1；均值 > e6 TB
  1100.744075 才保留，否则恢复 e6 两成员字节。首轮 codex-review 的 P3
  指出大批无 prefix/宽 prefix 测试缺口，已补回归；最终复审在提交前完成。

## 2026-09-23 E6 平台终态（20390）：valid 8/8 均值 1100.744 新 TB；昆仑 +91%

- 结构（`68a5e3db` + P2 修复 + e5 残留清除）：3 alloc→2 alloc——单一
  int32[2·batch] 共享缓冲，前半=契约输出 extend_start_loc（与
  reference 返回的 int32 截断一致），后半=宽 start 高 32 位；K2 以
  (hi<<32)|(lo&0xFFFFFFFF) 恢复 int64 地址（codex-review P2：累计
  start 越 2^31 时 r1 形式会写越界）。e5 失败的 _ascend vendor
  仓库残留已删（第二次打包夹带坑，本次打包前抓到）。
- 逐芯：天数 2927.6 / 沐曦 674.7 / 燧原 116.4 / 海光 1615.0 /
  **昆仑 105.25（+91%）** / 华为 147.7 / A 1687.1 / B 1532.2。
- 门判定：均值 >1051.15 ✓ 换 TB；最差回退 -4%（沐曦）未破 -10% ✓。
- 五元组：source/verification `HEAD（e6 终形）`；ZIP SHA
  `c800163c…`（2 成员 generic+_enflame，e5 残留 ascend 已剔）；
  回执 `day6-climb-20260923/t77e6-wf/`（2 源 4 测试 0 败）。
- 可迁移结构知识：**每 call torch 分配在昆仑芯计价极高**（一个
  batch 级 int64 scratch 的移除=+91%）；对各题昆仑轴先查 per-call
  alloc。代理 GPU 对 alloc 数完全不敏感（全形状 ~14us 恒定）——
  代理不可见该轴，只有平台逐芯能暴露。

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
