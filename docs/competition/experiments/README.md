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
| 09 | [`bmm_chunk`](bmm_chunk.md) | `s0-b05bfeb` | `058b016c309c0affa5ecbbcb125de415a6565be93e2b76a9535473021169c4e3` | 5/5 回归；N64 E1 仅 0.9011x，K64 E2 的 FP16 仅 1.0055x，保留 S0；未平台 | 第 6 个提交 |
| 10 | [`chunk_cumsum`](chunk_cumsum.md) | `s1-a4e84aa` | `f9fd0d595aeb5a4a4da76514321790815fbad9ccc39faa447c8bfa120f0e7db9` | 4/4 回归；修复尾块 dA carry；E1 仅 0.9995x，保留 S1；未平台 | 第 11 个提交 |
| 11 | [`chunk_local_cumsum_vector`](chunk_local_cumsum_vector.md) | `e1-528a2bb` | `7f0484b9b2ae078bf284e4fda1c5a1a0ffb0c8545b907e801d9fa21200fde7d8` | 2/2 release；tiny chunks 2-warps affected 1.032–1.162x；controls 0.997–1.002x；未平台 | 第 10 个提交 |
| 12 | [`chunk_state`](chunk_state.md) | `e2-67350fa` | `35f11803055ccc0a7e6bff71c974ad3671032c1cec35d2a556367789206de9e3` | 3/3 回归；K>=256 受影响点 1.1198x；未平台 | 第 5 个提交 |
| 13 | [`chunk_state_varlen`](chunk_state_varlen.md) | `s0-b05bfeb` | `bd23ddad1c833c8f9ba2c8e0e551fa5e4c3d7ad446351d74a346af14c850603b` | NVIDIA 613.340x 单 case；题面 reference 跨 chunk 语义冲突 | 暂缓，先确认语义 |
| 14 | [`context_attention`](context_attention.md) | `s0-fbbf74f` | `38ce76db6fee2121a765a1cd741138b9c2ded2478fdd85b1bfb4bba3d0f97456` | NVIDIA 0.5797–6.4198x；大 D 资源风险 | 暂缓，受控实验 |
| 15 | [`decode_attention`](decode_attention.md) | `s0-f431ba4` | `850cf12333241a450b342edbd2e108dca5841ddfb4f576129df45d863e5123b9` | 5/5 回归；tile64 E1 为 1.198–1.449x 但新增 12 spills，保留 S0；未平台 | 第 12 个提交 |
| 16 | [`decode_grouped_attention`](decode_grouped_attention.md) | `s0-f431ba4` | `4ed5e04d8453e100a38feff3d8986801fab9a13c4d77481e070a3260855136ef` | NVIDIA 5.285x；未平台 | 第 13 个提交 |
| 17 | [`embedding_lora_a`](embedding_lora_a.md) | `s1-d101ebe` | `49d7a33648c31d2b13e46c7e3dba8e7a4b88ecadce7da444c2ed5bac6b0ac09f` | 5/5 release；修复空 segment metadata 越界；2-warps E1 仅 0.999994x，保留 S1；未平台 | 第 7 个提交 |
| 18 | [`fused_recurrent_gdn`](fused_recurrent_gdn.md) | `s0-de1530b` | `cf27e0e48f41fc1948075cd3bc22864e45d2387d8e61b5b6371fe1147fe9ce7f` | NVIDIA 21.92–110.36x；八芯状态资源高风险 | 暂缓，受控实验 |
| 19 | [`fused_rmsnorm`](fused_rmsnorm.md) | `s0-3fac516` | `93780caf704341737ddfe5925cfacdcd7115ccefc2f38edf3c7ff006716d1820` | 3/3 回归；multi-row E1 仅 0.195–0.981x，保留 S0；跨芯风险已审计；未平台 | 第 1 个提交 |
| 20 | [`mamba_layernorm_gated`](mamba_layernorm_gated.md) | `e2-345413d` | `78c56c2955981833242d9fc2ed13dca1373014fc49f12072d469f34987875f03` | 3/3 回归；E2 受影响点 1.0803x；未平台 | 第 4 个提交 |
| 21 | [`moe_sum_reduce`](moe_sum_reduce.md) | `s0-3fac516` | `ef3c30e416d24d8268a1c252261676f3e540910a8836a93d2520917580f514bf` | 4/4 回归；E1 仅 1.0111x，保留 S0；未平台 | 第 3 个提交 |
| 22 | [`qkv_lora_b`](qkv_lora_b.md) | `s1-11ae343` | `bec21ac8d198d0eefd3d7c0ef68bf3a2c654017c00656c230ab12bc04f0f4d9c` | 3/3 release；修复空段 metadata 越界并跳过窄 slice 无效 GEMM；affected 1.006–1.371x；未平台 | 第 9 个提交 |
| 23 | [`sgemm_lora_b`](sgemm_lora_b.md) | `s1-222dd77` | `4223927a48608887b322b87611001f65102cd0e6fa2bf432b4efb50a7773a03f` | 4/4 release；修复空段 metadata 越界并跳过无效 GEMM；ragged 1.881–4.541x；未平台 | 第 8 个提交 |
| 24 | [`softcap_out`](softcap_out.md) | `s2-5cd6019` | `3746930f19d1a255571906fd4defd59b4a7ee272a65343f519969cd265e3db20` | S1 平台 8/8、1.90x；S2 Enflame 大 shape 代理提升 3.53–5.66x | 优化候选；覆盖其他任务后再投 |

## 建议提交顺序

按实现复杂度、公开达标队伍数、代理覆盖和跨芯风险排序：

```text
19 → 08 → 21 → 20 → 12 → 09 → 17 → 23 → 22 → 11 → 10 → 15 → 16
```

Task 13、18、14 暂缓，分别等待语义确认、仅作状态资源实验、仅作 attention
资源实验。Task 24 已通过；S2 只优化 Enflame，不重复消耗正确性额度。Task 24 在
2026-08-24 01:03:51 CST 提交后页面显示当日剩余 `13/15`，这只是历史观察；
每次上传前必须重新读取实时额度。

## 上传确认门禁

每次只确认一个不可变 ZIP，确认信息必须同时包含：

- race ID/赛季、登录账号、登录团队、batch、Task 编号和 operator；
- ZIP 绝对路径和完整 SHA-256；
- 平台实时剩余额度，以及本次消耗 1 次。

旧的“继续”或“上传”不授权后来生成的 ZIP。用户当次确认只授权上述 tuple 的一次
提交点击；确认后才按项目 Skill 执行网页选择文件和提交，并把逐芯结果写回对应账本。
