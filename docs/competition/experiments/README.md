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
| 08 | [`apply_token_bitmask`](apply_token_bitmask.md) | `s0-3fac516` | `394d287484e04c62eba5deea0c3f698787b1bd053ee7803598a7e9c98567a4b7` | NVIDIA 3.333–5.154x；未平台 | 第 2 个提交 |
| 09 | [`bmm_chunk`](bmm_chunk.md) | `s0-b05bfeb` | `058b016c309c0affa5ecbbcb125de415a6565be93e2b76a9535473021169c4e3` | NVIDIA 1.670–2.199x；未平台 | 第 6 个提交 |
| 10 | [`chunk_cumsum`](chunk_cumsum.md) | `s0-3fac516` | `81a1cff508d5ca8a7eb921d8644e4061b40382ea2ab9e4ce12a231118e48c607` | NVIDIA 3.184–29.044x；未平台 | 第 11 个提交 |
| 11 | [`chunk_local_cumsum_vector`](chunk_local_cumsum_vector.md) | `s0-3fac516` | `b4ab7b21ecd5a4f23b0d53aab00e8ef504c2e2f329c27b1bbf77306db5daab3a` | NVIDIA 1.605–2.919x；未平台 | 第 10 个提交 |
| 12 | [`chunk_state`](chunk_state.md) | `s0-b05bfeb` | `c689def894513d211ae96a1085d9e937a6b2da6dbc40e3db4aa5e9c9cb0a9686` | NVIDIA 2.611–4.406x；未平台 | 第 5 个提交 |
| 13 | [`chunk_state_varlen`](chunk_state_varlen.md) | `s0-b05bfeb` | `bd23ddad1c833c8f9ba2c8e0e551fa5e4c3d7ad446351d74a346af14c850603b` | NVIDIA 613.340x 单 case；题面 reference 跨 chunk 语义冲突 | 暂缓，先确认语义 |
| 14 | [`context_attention`](context_attention.md) | `s0-fbbf74f` | `38ce76db6fee2121a765a1cd741138b9c2ded2478fdd85b1bfb4bba3d0f97456` | NVIDIA 0.5797–6.4198x；大 D 资源风险 | 暂缓，受控实验 |
| 15 | [`decode_attention`](decode_attention.md) | `s0-f431ba4` | `850cf12333241a450b342edbd2e108dca5841ddfb4f576129df45d863e5123b9` | NVIDIA 16.213x；未平台 | 第 12 个提交 |
| 16 | [`decode_grouped_attention`](decode_grouped_attention.md) | `s0-f431ba4` | `4ed5e04d8453e100a38feff3d8986801fab9a13c4d77481e070a3260855136ef` | NVIDIA 5.285x；未平台 | 第 13 个提交 |
| 17 | [`embedding_lora_a`](embedding_lora_a.md) | `s0-f431ba4` | `e0fd0124cece568d536efaa89d05779c1f7457d9f0abf13efba8d190c482567e` | NVIDIA 25.228–134.092x；未平台 | 第 7 个提交 |
| 18 | [`fused_recurrent_gdn`](fused_recurrent_gdn.md) | `s0-de1530b` | `cf27e0e48f41fc1948075cd3bc22864e45d2387d8e61b5b6371fe1147fe9ce7f` | NVIDIA 21.92–110.36x；八芯状态资源高风险 | 暂缓，受控实验 |
| 19 | [`fused_rmsnorm`](fused_rmsnorm.md) | `s0-3fac516` | `93780caf704341737ddfe5925cfacdcd7115ccefc2f38edf3c7ff006716d1820` | NVIDIA 1.830–4.720x；未平台 | 第 1 个提交 |
| 20 | [`mamba_layernorm_gated`](mamba_layernorm_gated.md) | `s0-f431ba4` | `0bf5d8f26c6e3b3b827e2541bc58c058dc6b6fec05efe7bcff127492dfaedf76` | NVIDIA 3.016–6.333x；未平台 | 第 4 个提交 |
| 21 | [`moe_sum_reduce`](moe_sum_reduce.md) | `s0-3fac516` | `ef3c30e416d24d8268a1c252261676f3e540910a8836a93d2520917580f514bf` | NVIDIA 1.063–2.917x；未平台 | 第 3 个提交 |
| 22 | [`qkv_lora_b`](qkv_lora_b.md) | `s0-b05bfeb` | `ec395510ac56ccd289f53f95dab584c9502950e7a8b5d30d0681a3e2a1ab8a30` | NVIDIA 49.133–150.855x；未平台 | 第 9 个提交 |
| 23 | [`sgemm_lora_b`](sgemm_lora_b.md) | `s0-b05bfeb` | `d3a05c053120e9bf575125f28798eac0c5b5fdf9a9bf25f57fb83a8d1df2e348` | NVIDIA 60.948–103.665x；未平台 | 第 8 个提交 |
| 24 | [`softcap_out`](softcap_out.md) | `s1-fe2348e` | `698a7d9652d973868941e6e9e773d7d62ec1dceb87e0e392430b4b1c9cc69ded` | 平台 8/8，平均 1.90x，第 8 名 | 已完成；暂不再投正确性包 |

## 建议提交顺序

按实现复杂度、公开达标队伍数、代理覆盖和跨芯风险排序：

```text
19 → 08 → 21 → 20 → 12 → 09 → 17 → 23 → 22 → 11 → 10 → 15 → 16
```

Task 13、18、14 暂缓，分别等待语义确认、仅作状态资源实验、仅作 attention
资源实验。Task 24 已通过，不重复消耗正确性额度。Task 24 在
2026-08-24 01:03:51 CST 提交后页面显示当日剩余 `13/15`，这只是历史观察；
每次上传前必须重新读取实时额度。

## 上传确认门禁

每次只确认一个不可变 ZIP，确认信息必须同时包含：

- Task 编号和 operator；
- ZIP 绝对路径和完整 SHA-256；
- 平台实时剩余额度，以及本次消耗 1 次。

旧的“继续”或“上传”不授权后来生成的 ZIP。确认后才按项目 Skill 执行网页选择
文件和提交，并把逐芯结果写回对应账本。
