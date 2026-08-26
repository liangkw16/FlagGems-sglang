# 第二批候选与提交队列

本页汇总第二批 Task 08–24 的本地产物；逐项契约、测试、性能和风险以算子账本
为准。公开榜单快照见 [赛题索引](../task-index.md)，同步时间为
`2026-08-24T02:17:26+08:00`。

当前覆盖：17/17 个算子都有已提交源码、远端 NVIDIA 代理验证和不可变 ZIP。
Task 08、09、12、17、19、20、21 和 24 已经平台 8/8；其余 9 个尚无八芯平台结果，
代理加速比不能外推。

产物路径统一为：

```text
artifacts/competition/<operator>/<stage>-<commit>/<operator>.zip
```

## 候选清单

下方建议提交顺序只覆盖尚无 8 芯结果的任务；已完成任务保留当前结论。

| Task | 算子账本 | 候选 | ZIP SHA-256 | 当前证据 | 建议 |
| ---: | --- | --- | --- | --- | --- |
| 08 | [`apply_token_bitmask`](apply_token_bitmask.md) | `e2-86fca87` | `88d2e8387ac2e7de785cf1574ad9c762df54c0baa79e4ada67fad7252987c1dc` | E2 平台 8/8、4.686925x、第 12/13；燧原 0.4292x → 2.8510x，团队当前最佳 | 保留 E2，转其他算子 |
| 09 | [`bmm_chunk`](bmm_chunk.md) | `e4-93035e4` | `a967742cbeee3053b8f2b9af261a30cf3bae7b9a745c8adaf15abc61cd434f85` | E4 平台 8/8、1.27375x；官方 GCU300 cap 6 使燧原 0.149→0.153x，但平均未超过 E3e 的 1.283x | 保留 E3e team best；Task 09 永久停止 |
| 10 | [`chunk_cumsum`](chunk_cumsum.md) | `s2b-63e7943` | `c822f75d719f8919269c7566b1210b6e31dc6ce3292229a838723f7945b15923` | S2b 平台 8/8 正确、invalid_threshold；华为 UB tile 上限修复生效；燧原 0.0375x/昆仑 0.012x 为 cumsum lowering 固有瓶颈 | 已按两次规则停止；重试需改写 cumsum 算法形式 |
| 11 | [`chunk_local_cumsum_vector`](chunk_local_cumsum_vector.md) | `e1b-dddef74` | `cf4dcaf05640599fe5b50ee9633ba19d2a4f13f2b47f856e88259242e975bab9` | 两次提交均 7/8；昆仑 0.016x/华为 0.0255x 与折叠形态无关，cumsum 固有；燧原三形态编译失败 | 已按两次规则停止；同 Task 10 结论 |
| 12 | [`chunk_state`](chunk_state.md) | `e7-294990c` | `583b55a2518091cd707ff6dbf1080a10bd1fe2fd690da2885d0bfd48daae04a8` | E7 平台 8/8、**4.0371875x team best**；官方 Ascend Cube 路线使华为 0.329→2.1185x（6.44 倍），其余七芯稳定 | Task 12 永久停止；保留 E7 |
| 13 | [`chunk_state_varlen`](chunk_state_varlen.md) | `e3-d795ed3` | `8ea09900fe8995f59f63e649ca0b3a3fd97215e1ed09511d0a4b96a97424151c` | E3 平台 7/8；燧原从全 case 编译失败恢复至 1.4735x，昆仑全 padding 无 mask 后五个 case 仍数值失败 | E3 stop gate 已触发，永久停止；保留 GCU i32 模板 |
| 14 | [`context_attention`](context_attention.md) | `e1a-6246fa8` | `8bfc8843bb6951de12160d83dbd56428c3697262ad331a05183217b9aa2d7861` | E1a 终态 5/8：天数 vendor 1.99x、nvidia vendor 6.32x；燧原超时疑似死循环、昆仑评测器崩溃、华为 flash 边界 bug | 保留第 2 次额度；三芯失败互独立无单变量解 |
| 15 | [`decode_attention`](decode_attention.md) | `e2a-5add38c` | `b2fdcbc98b098165c3defe61cb9b0a5f5e021dfe04dbb5798dfd684b0fac8751` | 两次提交 6–7/8；华为 case 8 整行重复指纹（Ascend flash 边界 bug）两种 grid 均现；昆仑评测超时崩溃 | 已按两次规则停止 |
| 16 | [`decode_grouped_attention`](decode_grouped_attention.md) | `e1a-9801c56` | `c8dd889f7820f52e73bfc2ea1c88c007b2a969c3811e0970b260f911e25a5b2b` | E1a 平台 5/8；天数 vendor 生效；华为 case 7 与 Task 15 同型指纹，燧原段错误、昆仑评测崩溃 | 三芯失败互独立无单变量解，保留第 2 次额度，记 5/8 停止 |
| 17 | [`embedding_lora_a`](embedding_lora_a.md) | `e2a-i32-fb1235d` | `eb4b40d4703f5c6ea8d9bc3e5c3b896310f5bfe7a9c0d40637dc0c746d126081` | E2a-i32 平台 8/8、13.8620625x、team best；消除 GCU 64-bit IR 使燧原从全 case 编译失败恢复至 0.3885x | 闭环完成，保留 E2a-i32 |
| 18 | [`fused_recurrent_gdn`](fused_recurrent_gdn.md) | `e6-b528e9c` | `6093114cf384aa3fe81a5291b6a48bd64d0b72f2c4969fc6906062967cf97764` | E6 平台仅国际 A vendor 通过；官方 `[BK,BV]` 轴转置 E8 offline 的两组错误数与 E6 逐字相同，昆仑仍有系统性 1830s 超时 | 官方唯一新归约形态已离线否决，Task 18 永久停止 |
| 19 | [`fused_rmsnorm`](fused_rmsnorm.md) | `e2-a5b2986` | `04e24fd06f26144bb6b5824b720678edd48a731f34ced48eab8600c30c65c124` | E2 平台 8/8、4.5467x、第 7；Kunlun vendor 被选中但仅 0.9308x，未过 1.05x 门禁 | 停止同一 multi-row 假设；转其他算子 |
| 20 | [`mamba_layernorm_gated`](mamba_layernorm_gated.md) | `e3-374e06c` | `afe450702c551fc83395432733dd22e98840125a31247c5e43117983ee30bb3d` | E3 平台 8/8、4.2526x、第 6/6；华为由启动失败恢复至 1.8838x，团队当前最佳 | 保留 E3，转其他算子 |
| 21 | [`moe_sum_reduce`](moe_sum_reduce.md) | `s3-1ca7dd2` | `159911639601002f9be5e083309d9a5cac1d1d32617e1fe31207486cc267b2f8` | S3 平台 8/8、2.7096x、valid、team best；E4 燧原 BLOCK 4096 screening 仅 0.555x，已拒绝 | 保留 S3；停止同一大 tile 假设 |
| 22 | [`qkv_lora_b`](qkv_lora_b.md) | `s2c-7857dca` | `357e8a690cca68123aabebdbb5500a86ebd66fe328105a8b91f7c1afe489cb38` | S2c 终态 6/8；六芯高分（海光 82.6x、天数 47.8x）；燧原 case 2 编译失败、昆仑评测异常 | 已按两次规则停止 |
| 23 | [`sgemm_lora_b`](sgemm_lora_b.md) | `e11-dd4632a` | `c23266792c400636b2a7a4aa418defa2eb15f19623e8419316133dceb4463ff7` | E11 官方 XPU legacy masked-memory 使昆仑五个 case 全部跑完，但均数值失败；燧原回调待定，其余六芯已通过 | 官方两种 masked-memory 路径均验证失败，Task 23/22 永久停止 |
| 24 | [`softcap_out`](softcap_out.md) | `s2e-1a5ea26` | `8469beb23dbaa27fbfbd7f6f74b650ee899586f43bacfb7e3fdd40e8dd566ed0` | S2e 平台 8/8、valid、2.0179x team best；S3c grid=12 interleave 仍正确但昆仑仅 0.2418x、平均 1.9516x | 保留 S2e；永久停止 interleave 假设 |

## 建议提交顺序

按实现复杂度、公开达标队伍数、代理覆盖和跨芯风险排序：

```text
12 → 09 → 17 → 23 → 22 → 11 → 10 → 15 → 16 → 13 → 14
```

Task 18 暂缓，仅作状态资源实验。Task 21 已闭环（S0c 6/8 → S1 7/8 → S2 7/8
→ S3 8/8、2.7096x、team best；昆仑 XPU 2D grid 展平总数上限 65535 的证据
链完整，华为 capped grid-stride 第三次平台验证）。队列恢复新算子首投。
Task 13 已按公开 reference 可返回域修复；
其余 shape 仍需赛方澄清。Task 14 已有 NVIDIA 优化，但七芯 generic 风险仍需平台
证明。Task 08 的 E2 已通过 8/8，燧原 BLOCK 优化完成。Task 19 的 E2 已
通过 8/8，但昆仑专项优化未生效；燧原为 1.5049x。Task 20 E3 已通过 8/8，
Ascend capped grid-stride 将华为从启动失败恢复至 1.8838x。
Task 24 S2c 已通过，Enflame-only BLOCK 4096 将燧原从 0.3458x 提升至
1.1892x，平均 1.9855x。Task 19 在
2026-08-24 17:05:45 CST 二投后平台显示当日剩余 `11/15`；Task 08 S0c 首投后，
2026-08-24 18:00:04 CST 只读状态为 `10/15`；S1 提交后在 19:24:01 CST 为
`9/15`，E2 提交后在 19:55:10 CST 为 `8/15`。Task 20 E2 在 20:38:59 CST
提交，20:53:09 CST 只读状态为新口径剩余 `22/30`；E3 于 22:34:19 CST 提交，
22:35:10 CST 终态只读状态为 `21/30`。Task 21 S0c 于 22:46:50 CST 提交，
22:50:18 CST 终态时剩余 `20/30`；S1 于 23:02:23 CST 提交，23:03:08 CST 终态时
剩余 `19/30`；S2 于 23:13:00 CST 提交，23:13:47 CST 剩余 `18/30`；S3 于
23:20:43 CST 提交，23:21:15 CST 终态时剩余 `17/30`。这些只是历史观察，每次
上传前必须重新读取实时额度。

## 自动提交门禁

每次只为一个不可变 ZIP 建立 preflight intent，tuple 必须同时包含：

- race ID/赛季、登录账号、登录团队、batch、Task 编号和 operator；
- source commit、stage、成员集合、ZIP 绝对路径和完整 SHA-256；
- 平台实时剩余额度，以及本次消耗 1 次。

当前任务包含平台提交、完整闭环或继续既有竞赛闭环，且完整 tuple 与本地证据一致时，
按项目 Skill 立即执行一次性 submit 命令，无需再询问。每个新 ZIP 都重新 preflight；
`sending`、`uncertain`、`stale_after_upload` 或已提交状态只读核对，绝不自动重试。
逐芯结果写回对应账本。

## 2026-08-25 收官记录

第二轮全部 17 个任务已按"每任务 ≤2 次提交、按序尝试"规则处理完毕（早期
T08/T09/T17 按当时的 3 次规则执行）。终态：

- **8/8 有效（7 题）**：Task 08（4.687x）、19（4.547x）、20（4.253x）、
  21（2.710x）、24（**2.0179x**）、12（**2.0966x**）、09（**1.283x**）。
  加粗为 2026-08-25 下午冲刺轮（5 次提交全部 valid）更新：T09 E3e 过线
  新增第 7 道有效题；T12 E3/E4 燧原 0.116→0.743→0.939x；T24 S2d/S2e
  昆仑 0.444→0.764→0.864x。
- **尝试后停止（9 题）**：T23/T11 7/8；T22/T15/T13 6/8（T15 一轮
  6–7/8）；T10 8 芯正确但燧原 0.0375x/昆仑 0.012x 低于 0.1x 门槛；
  T16/T14 5/8（各保留 1 次额度，三芯失败互独立无单变量解）；T18 平台
  `pending_challenge` 拒绝提交（候选 `e2-2ba2813` 已留证，恢复
  competing 后可直接复用 preflight 流程）。
- **额度**：2026-08-25 Task 24 S2e 提交后剩 1/30（当日 6 次机会用 5 次，
  1 次留作截止前回归储备）；08-26/08-27 每日仍有 30 次；截止
  2026-08-27 19:59:59。

### 2026-08-25 冲刺轮沉淀的跨芯新知识

1. 燧原 grid-stride 折叠（cap 64）两次平台验证：T09 +66%（0.090→
   0.149x）、T12 +26%（0.743→0.939x）；与 "64/64/128+stages2" 组合
   是燧原 dot vendor 当前最优模板。
2. 燧原 dot 配置迁移性：fp16 操作数 + 64/64/128 + stages2 在 T12 上
   0.116→0.743x（6.4 倍），病理配置（ieee-fp32 dot + 32 tile +
   stages1）跨题复现。
3. 昆仑 elementwise BLOCK 曲线：256→1024→4096 对应 0.444→0.764→
   0.864x（T24），仍未饱和；与 T21 reduction BLOCK 1024 证据互补。
4. 平台按团队最佳计分（`is_team_best` 字段），追投无下行风险。
5. 远端 venv black 升级至 26.5.1（hug_parens）与仓库既有字节冲突，
   属工具漂移；release 门禁以本地 black 25.12.0 等价执行并记录。
6. 昆仑 fp16 操作数 `tl.dot` 正确性失败（T12 E5 平台证据，代理 NVIDIA
   全对、七芯全对仅昆仑错）；fp32-ieee 操作数在 E2d 通过。与天数
   "fp32 操作数 dot 静默不可执行"互为镜像：两芯 dot 操作数 dtype 兼容集
   相反，generic 低精度 dot 必须搭配 `_kunlunxin` 回退 vendor。
7. generic 低精度 dot 的分芯收益谱（T12 E5）：card_a +282%（张量核心
   解锁）、海光 +1.4%、沐曦 -30%、国际 B -56%——低精度 dot 不是普适
   提速，逐芯 vendor 选择是必要配套。

### 已平台验证的跨芯知识（均有逐芯证据，详见各账本）

1. 天数：fp32 操作数 `tl.dot` 静默不可执行；fp16 操作数（宽松容差）或
   split-fp16 三点积（fp32 1e-4 容差）可执行。
2. 华为：2D/3D grid 展平总数 ≤65535（超限 launch 失败，capped
   grid-stride fold 五次平台验证）；UB tile 上限（`block_h ≤
   512//block_size`）；flash 型 kernel 存在整行重复的边界 bug（T14/T15/T16
   同指纹，两种 grid 均现）。
3. 昆仑：2D grid 展平总数 ≤65535（编译期失败）；`num_warps/num_stages`
   为 invalid 参数；dot kernel 走 SDNN 路径可过（规整结构），ragged/
   varlen/cumsum 结构编译爆炸或固有 0.012–0.016x 慢；BLOCK 是唯一有效
   调参轴（Task 21 BLOCK 1024 唯一成功案例）。
4. 燧原：dot kernel 需 stages≥2 + ≥64 tile（Task 09 0.001→0.090x、
   Task 23 4.05x）；grid.x ≤65535；含运行时分支或 cumsum 的结构
   `Pipeline run failed` 编译失败；32×32+fp16-dot 组合编译失败。
5. cumsum 家族（T10/T11 四轮）：昆仑/燧原 lowering 固有瓶颈，非
   grid/配置/stages 可解，重试需改写算法形式（两阶段分块扫描）。

### 后续选项

- T14/T16 各有 1 次保留额度，待燧原/昆仑评测器恢复或新证据（当前三芯
  失败互独立，无单变量解）。
- 已 8/8 任务的性能冲刺：昆仑（多题 0.012–0.34x）与燧原（0.09–2.85x）
  提升空间最大，vendor 模板已全部入库。
- T18 平台恢复后直接复用 `e2-2ba2813` 候选。
