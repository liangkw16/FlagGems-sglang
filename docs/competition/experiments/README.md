# 第二批候选与提交队列

本页汇总第二批 Task 08–24 的本地产物；逐项契约、测试、性能和风险以算子账本
为准。公开榜单快照见 [赛题索引](../task-index.md)，同步时间为
`2026-08-24T02:17:26+08:00`。

当前覆盖：17/17 个算子都有已提交源码、远端 NVIDIA 代理验证和不可变 ZIP。
Task 08、12、19、20、21 和 24 已经平台 8/8；其余 11 个尚无八芯平台结果，
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
| 09 | [`bmm_chunk`](bmm_chunk.md) | `e3d-57b7130` | `ad546c3942c40689649c48b4399bf72da318fba04179df1d93d9a85d823fd5ee` | 4 次提交均 8/8 正确、燧原 0.001→0.090x 未达 0.1x 门槛；天数/华为 vendor 已验证；已按三次规则停止 | 待燧根 dot 配置知识成熟后重试（stages≥2 起步） |
| 10 | [`chunk_cumsum`](chunk_cumsum.md) | `s2b-63e7943` | `c822f75d719f8919269c7566b1210b6e31dc6ce3292229a838723f7945b15923` | S2b 平台 8/8 正确、invalid_threshold；华为 UB tile 上限修复生效；燧原 0.0375x/昆仑 0.012x 为 cumsum lowering 固有瓶颈 | 已按两次规则停止；重试需改写 cumsum 算法形式 |
| 11 | [`chunk_local_cumsum_vector`](chunk_local_cumsum_vector.md) | `e1b-dddef74` | `cf4dcaf05640599fe5b50ee9633ba19d2a4f13f2b47f856e88259242e975bab9` | 两次提交均 7/8；昆仑 0.016x/华为 0.0255x 与折叠形态无关，cumsum 固有；燧原三形态编译失败 | 已按两次规则停止；同 Task 10 结论 |
| 12 | [`chunk_state`](chunk_state.md) | `e2d-3d31481` | `3c06525a76dd00e338d40107feca666e43dd7f99b097e00da5e87f7ca548623b` | E2d 平台 8/8、1.948x、valid、team best；天数 fp16-dot、华为 capped grid vendor 均选中 | 闭环完成；燧原 0.116x 贴门槛为后续优化点 |
| 13 | [`chunk_state_varlen`](chunk_state_varlen.md) | `s1b-1975cf7` | `0319c0e26b7cd6fb12f33b43771572a058306e89ac5982234531552daa0203d1` | 两次提交均 6/8；天数 vendor 132x、华为 27.7x 生效；燧原/昆仑对该 varlen 结构编译失败（结构性） | 已按两次规则停止 |
| 14 | [`context_attention`](context_attention.md) | `e1a-6246fa8` | `8bfc8843bb6951de12160d83dbd56428c3697262ad331a05183217b9aa2d7861` | E1a 评测中：5/8 已过（天数 vendor 1.99x、nvidia vendor 6.32x）；燧原/昆仑/华为待评 | 终态见账本 |
| 15 | [`decode_attention`](decode_attention.md) | `e2a-5add38c` | `b2fdcbc98b098165c3defe61cb9b0a5f5e021dfe04dbb5798dfd684b0fac8751` | 两次提交 6–7/8；华为 case 8 整行重复指纹（Ascend flash 边界 bug）两种 grid 均现；昆仑评测超时崩溃 | 已按两次规则停止 |
| 16 | [`decode_grouped_attention`](decode_grouped_attention.md) | `e1a-9801c56` | `c8dd889f7820f52e73bfc2ea1c88c007b2a969c3811e0970b260f911e25a5b2b` | E1a 平台 5/8；天数 vendor 生效；华为 case 7 与 Task 15 同型指纹，燧原段错误、昆仑评测崩溃 | 三芯失败互独立无单变量解，保留第 2 次额度，记 5/8 停止 |
| 17 | [`embedding_lora_a`](embedding_lora_a.md) | `s1c-e12c7a9` | `a247a9dd500ae4b110248f8b9954c9c7e1ae429763115c0061344aee360f3a4f` | 3 次提交均 7/8；华为/昆仑 token 折叠 vendor 平台验证成功；燧原三种 kernel 形态均编译失败 | 已按三次规则停止；燧原标量载入嫌疑待 GCU 环境定位 |
| 18 | [`fused_recurrent_gdn`](fused_recurrent_gdn.md) | `e2-2ba2813` | `4be0a8135cc5dcc23a33b31852b6754fa44a2959e8d035e49a113d07edaf14eb` | 3/3 release；低精度 K65–128 为 1.479–1.525x；大 K 八芯状态资源仍高风险 | 最后受控提交 |
| 19 | [`fused_rmsnorm`](fused_rmsnorm.md) | `e2-a5b2986` | `04e24fd06f26144bb6b5824b720678edd48a731f34ced48eab8600c30c65c124` | E2 平台 8/8、4.5467x、第 7；Kunlun vendor 被选中但仅 0.9308x，未过 1.05x 门禁 | 停止同一 multi-row 假设；转其他算子 |
| 20 | [`mamba_layernorm_gated`](mamba_layernorm_gated.md) | `e3-374e06c` | `afe450702c551fc83395432733dd22e98840125a31247c5e43117983ee30bb3d` | E3 平台 8/8、4.2526x、第 6/6；华为由启动失败恢复至 1.8838x，团队当前最佳 | 保留 E3，转其他算子 |
| 21 | [`moe_sum_reduce`](moe_sum_reduce.md) | `s3-1ca7dd2` | `159911639601002f9be5e083309d9a5cac1d1d32617e1fe31207486cc267b2f8` | S3 平台 8/8、2.7096x、valid、team best；昆仑 BLOCK 1024 与华为 capped grid-stride 均选中通过 | 闭环完成；燧原 0.206x 连续低读数为后续优化点 |
| 22 | [`qkv_lora_b`](qkv_lora_b.md) | `s2c-7857dca` | `357e8a690cca68123aabebdbb5500a86ebd66fe328105a8b91f7c1afe489cb38` | S2c 终态 6/8；六芯高分（海光 82.6x、天数 47.8x）；燧原 case 2 编译失败、昆仑评测异常 | 已按两次规则停止 |
| 23 | [`sgemm_lora_b`](sgemm_lora_b.md) | `s2b-4c184b6` | `3b022a2b66b170c99d3aa0f94c9f5f878489df1fad729bc43d94ba09af993db0` | 两次提交均 7/8；燧原 64/128+stages3 达 4.05x、天数 34x、华为 18x；昆仑 SDNN 对 ragged 结构编译爆炸 | 已按两次规则停止；昆仑需规整 batched-GEMM 改写 |
| 24 | [`softcap_out`](softcap_out.md) | `s2c-5cd6019` | `999f2dea69774c2f9756748a2a113c7ad54d3e2fdce18bfd24b014a96fed1f46` | S1 平台 8/8、1.90x；S2 Enflame 大 shape 代理提升 3.53–5.66x；canonical 包已验签 | 优化候选；实时预检后确认提交 |

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
Task 24 已通过，S2 只优化 Enflame。Task 19 在
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
