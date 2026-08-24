# 第二批候选与提交队列

本页汇总第二批 Task 08–24 的本地产物；逐项契约、测试、性能和风险以算子账本
为准。公开榜单快照见 [赛题索引](../task-index.md)，同步时间为
`2026-08-24T02:17:26+08:00`。

当前覆盖：17/17 个算子都有已提交源码、远端 NVIDIA 代理验证和不可变 ZIP。
Task 24 已经平台 8/8；其余 16 个尚无八芯平台结果，代理加速比不能外推。

产物路径统一为：

```text
artifacts/competition/<operator>/<stage>-<commit>/<operator>.zip
```

## 候选清单

| Task | 算子账本 | 候选 | ZIP SHA-256 | 当前证据 | 建议 |
| ---: | --- | --- | --- | --- | --- |
| 08 | [`apply_token_bitmask`](apply_token_bitmask.md) | `s0-3fac516` | `394d287484e04c62eba5deea0c3f698787b1bd053ee7803598a7e9c98567a4b7` | 5/5 回归；E1 仅 0.9983x，保留 S0；未平台 | 第 2 个提交 |
| 09 | [`bmm_chunk`](bmm_chunk.md) | `e3-a5afc19` | `d8577b2ee314cad758f756d47794685240448d2654baf6b685a7e53fac415b95` | 6/6 release；低精度 affected 1.4306x，FP32 controls 1.0000x，资源与 S0 相同；未平台 | 第 6 个提交 |
| 10 | [`chunk_cumsum`](chunk_cumsum.md) | `s1-a4e84aa` | `f9fd0d595aeb5a4a4da76514321790815fbad9ccc39faa447c8bfa120f0e7db9` | 4/4 回归；E1–E4 均拒绝，E4 大 shape 有效但 medium/low 仅 1.0091/0.9999x；保留 S1，未平台 | 第 11 个提交 |
| 11 | [`chunk_local_cumsum_vector`](chunk_local_cumsum_vector.md) | `e1-528a2bb` | `7f0484b9b2ae078bf284e4fda1c5a1a0ffb0c8545b907e801d9fa21200fde7d8` | 2/2 release；tiny chunks 2-warps affected 1.032–1.162x；controls 0.997–1.002x；未平台 | 第 10 个提交 |
| 12 | [`chunk_state`](chunk_state.md) | `e2-67350fa` | `35f11803055ccc0a7e6bff71c974ad3671032c1cec35d2a556367789206de9e3` | 3/3 E2 release；E3 聚合 1.1207x，但最差 0.9591x 且 FP32 新增 spill，拒绝；未平台 | 第 5 个提交 |
| 13 | [`chunk_state_varlen`](chunk_state_varlen.md) | `s1-7911930` | `fcc17df06adf338578402f315e4dab75bf2361e641885bb99f94e23be46efd49` | 5/5 release；低精度 dot E1 因确定性 `0.0625>0.03` 反例拒绝；未平台 | 第 14 个提交 |
| 14 | [`context_attention`](context_attention.md) | `e1-a085dc4` | `1bd5f7483bac887f92c6be3e2aea81ac2c69f519aeafd28f267585f37a7da777` | 6/6 release；NVIDIA vendor 1.685–2.056x over S0，原最差点达 1.066x reference；未平台 | 第 15 个提交 |
| 15 | [`decode_attention`](decode_attention.md) | `e2-59cb094` | `0170fd15d5da5e0bd268fa1c5d12c7e9ee36e5cb5af50625a33da26e6ef4da62` | 7/7 release；NVIDIA 长序列 1.197–1.501x，短序列门控最差 1.000x，0 spill；未平台 | 第 12 个提交 |
| 16 | [`decode_grouped_attention`](decode_grouped_attention.md) | `e1-bc729bd` | `088a9ebfcae10a608528e5614a684997753cd8693ac13f49496383ced4ca80c0` | 6/6 release；grouped KV reuse 1.326–1.845x，controls 1.0009x；未平台 | 第 13 个提交 |
| 17 | [`embedding_lora_a`](embedding_lora_a.md) | `s1-d101ebe` | `49d7a33648c31d2b13e46c7e3dba8e7a4b88ecadce7da444c2ed5bac6b0ac09f` | 5/5 release；修复空 segment metadata 越界；2-warps E1 仅 0.999994x，保留 S1；未平台 | 第 7 个提交 |
| 18 | [`fused_recurrent_gdn`](fused_recurrent_gdn.md) | `e2-2ba2813` | `4be0a8135cc5dcc23a33b31852b6754fa44a2959e8d035e49a113d07edaf14eb` | 3/3 release；低精度 K65–128 为 1.479–1.525x；大 K 八芯状态资源仍高风险 | 最后受控提交 |
| 19 | [`fused_rmsnorm`](fused_rmsnorm.md) | `s0-3fac516` | `93780caf704341737ddfe5925cfacdcd7115ccefc2f38edf3c7ff006716d1820` | 3/3 回归；multi-row E1 仅 0.195–0.981x，保留 S0；跨芯风险已审计；未平台 | 第 1 个提交 |
| 20 | [`mamba_layernorm_gated`](mamba_layernorm_gated.md) | `e2-345413d` | `78c56c2955981833242d9fc2ed13dca1373014fc49f12072d469f34987875f03` | 3/3 回归；E2 受影响点 1.0803x；未平台 | 第 4 个提交 |
| 21 | [`moe_sum_reduce`](moe_sum_reduce.md) | `s0-3fac516` | `ef3c30e416d24d8268a1c252261676f3e540910a8836a93d2520917580f514bf` | 4/4 回归；512/1024/2048 tile 均未过 1.05x，保留 S0；未平台 | 第 3 个提交 |
| 22 | [`qkv_lora_b`](qkv_lora_b.md) | `s1-11ae343` | `bec21ac8d198d0eefd3d7c0ef68bf3a2c654017c00656c230ab12bc04f0f4d9c` | 3/3 release；修复空段 metadata 越界并跳过窄 slice 无效 GEMM；affected 1.006–1.371x；未平台 | 第 9 个提交 |
| 23 | [`sgemm_lora_b`](sgemm_lora_b.md) | `s1-222dd77` | `4223927a48608887b322b87611001f65102cd0e6fa2bf432b4efb50a7773a03f` | 4/4 release；N256/N128 均在低精度 ragged 回退且资源失败，保留 S1；未平台 | 第 8 个提交 |
| 24 | [`softcap_out`](softcap_out.md) | `s2c-5cd6019` | `999f2dea69774c2f9756748a2a113c7ad54d3e2fdce18bfd24b014a96fed1f46` | S1 平台 8/8、1.90x；S2 Enflame 大 shape 代理提升 3.53–5.66x；canonical 包已验签 | 优化候选；实时预检后确认提交 |

## 建议提交顺序

按实现复杂度、公开达标队伍数、代理覆盖和跨芯风险排序：

```text
19 → 08 → 21 → 20 → 12 → 09 → 17 → 23 → 22 → 11 → 10 → 15 → 16 → 13 → 14
```

Task 18 暂缓，仅作状态资源实验。Task 13 已按公开 reference 可返回域修复；
其余 shape 仍需赛方澄清。Task 14 已有 NVIDIA 优化，但七芯 generic 风险仍需平台
证明。Task 24 已通过；S2 只优化 Enflame，不重复消耗正确性额度。Task 24 在
2026-08-24 01:03:51 CST 提交后页面显示当日剩余 `13/15`，这只是历史观察；
每次上传前必须重新读取实时额度。

## 上传确认门禁

每次只确认一个不可变 ZIP，确认信息必须同时包含：

- race ID/赛季、登录账号、登录团队、batch、Task 编号和 operator；
- ZIP 绝对路径和完整 SHA-256；
- 平台实时剩余额度，以及本次消耗 1 次。

旧的“继续”或“上传”不授权后来生成的 ZIP。用户当次确认只授权上述 tuple 的一次
提交点击；确认后才按项目 Skill 执行网页选择文件和提交，并把逐芯结果写回对应账本。
