# 2026-09-16 PR 复扫与第五批结构机会

本报告只读取 GitHub API、固定 SHA 源码、本地契约和历史实验。未连接 GPU，未提交平台。榜单依据 `data/top1-intel-20260916-refresh2.json`，观测时间 11:18:32 +08:00；额度 26/30。源码审计基线 HEAD `bc5dbb42a55709520daaa385cfd2cafcec6568ed`。

## 最新公开事实

通过 `gh api repos/flagos-ai/FlagGems-sglang/pulls?state=all&sort=updated&direction=desc&per_page=100` 读取最新 PR。当前最高编号 #79；没有第四、第五批算子 PR。最新更新 #56 是 2026-09-16 02:11:17Z；#79/#78/#77/#75/#74/#70 等第三批 PR 仍开放。第二批 #49/#50/#59 等已在 9 月 14 日合并。PR 作者关于平台验证的描述仅视作提交者声明，不代替本任务执行证据。

| 固定来源 | 最新状态 | 实际读取的可迁移结构 |
| --- | --- | --- |
| [#56 apply_token_bitmask](https://github.com/flagos-ai/FlagGems-sglang/pull/56)，`9e36df5e4bfe146982bf8fa4b1f98bb496fe952c` | open | Ascend 由同一个 NWB 计算任务数和访存偏移；小 grid direct，超大 grid persistent fallback |
| [#78 softcap_inplace_logits](https://github.com/flagos-ai/FlagGems-sglang/pull/78)，`077fdc3a0d8d7af02b8b84f131e20399c962c1d3` | open | Ascend `_direct` 无 runtime loop；较大输入另走 persistent |
| [#50 mamba_layernorm_gated](https://github.com/flagos-ai/FlagGems-sglang/pull/50)，`987e6070a6b9435bf8cf19298468291e08277552` | merged；merge `4fa4ba7ff5fd444e9429d4ebc972a8cddfa8db5b` | Enflame 固定 group 的 weight/bias 在 persistent row loop 外加载并复用 |
| [#70 gelu_and_mul](https://github.com/flagos-ai/FlagGems-sglang/pull/70)，`cef74eaa1ae076d386d3a0addb3aaa7a628b2f51` | open | Ascend 行打包/持久循环；注释明确该 tile-area probe 未在 Ascend 编译测速，不能当目标性能证据 |
| [#75 per_token_quant_int8](https://github.com/flagos-ai/FlagGems-sglang/pull/75)，`3d35241b537a81aca991ac558ca69fa5c65ca7ba` | open | Enflame 多行向量归约，未发现能直接填补第五批缺口的新算法 |
| [#77 per_token_group_quant_int8](https://github.com/flagos-ai/FlagGems-sglang/pull/77)，`482f74016a35874e8e45b29e9f53b994558a3cbf` | open | Ascend 小 group 打包，严格 div_rn；与已试 packing 家族重合，不能据此重开 |

## 按根因证据排序的三个机会

### 1. T64：修正 Enflame 任务数与实际块宽不一致

**可直接准备候选，优先于猜测性调参。** 当前及 TB E5 的 `_enflame/ops/deepep_permute.py:61` 用 `tiles=cdiv(hidden,512)`，但第 78 行 launch `BLOCK=2048`，kernel 第 41 行按 `tile*BLOCK` 访存。因此 hidden=4096 时每 token 8 tasks 仅 2 个有有效 lane；另外 6 个仍循环读取 topk 路由和执行分支。修复仅使 task count 使用同一个 `_BLOCK=2048` 常量，保留 clone、cap24、数学、scatter、dtype、strides 和其他 vendor。

这不是重试 BLOCK 阶梯，而是删除旧 512 调度遗留的空任务。[PR #56 固定源码](https://github.com/flagos-ai/FlagGems-sglang/blob/9e36df5e4bfe146982bf8fa4b1f98bb496fe952c/src/flaggems_sglang/runtime/backend/_ascend/ops/apply_token_bitmask.py#L122-L152) 提供同一宽度导出任务数、grid 与内核 offset 的成熟形式；本仓 generic 也始终保持 512/512 一致。`rg` 查出算子只有 generic、Ascend、Enflame 三个 wrapper；不一致仅 Enflame。

TB source `f8e1aec19ae52a82070e48ffdc3fd72a81074011`，包 `artifacts/competition/deepep_permute/e5-f8e1aec/deepep_permute.zip` SHA-256 `a3b7a76e3ed5700d58fbaecfca5b573cc2467d761df6d3f4ddf5d55c21955741`。三个成员均与 Git 对象逐字节一致：generic `7857c235db4f253b025d9890b444215455bde6c8ea99bbe9cecd3f6a8d69fc1a`；Ascend `c032bc09a05cbeaa217e4fa03207eb08854d2ca6d2548298dd99c2aebcd8d5f1`；Enflame `5ec8fe68cf7f82df5709c5e157ad3130bcdef0ee2b7c0483906c2965917eab55`。错误由 `ea69f3a51b05e409ee85a7aeb5e7017a124d916b` 延续。

成本收益：hidden 为 2048 整倍数时 task/route-loop 工作减少 75%，但真实数据搬运量、clone 和 kernel 数均不变，不能把 4 倍 task 缩减说成 4 倍端到端提速。需要完整回归（尤其 512/2048 边界、stride、无效路由）、wrapper-inclusive 五轮 AB/BA，额外统计 kernel-only 以定位 clone 遮蔽。小 hidden 或小 token 数可能改变有效 program 分布，必须纳入控制组。

实时排名 11/14，我方 7.17105，第一 26.486775；燧原 6.666 vs 第一 12.0746。仅本芯追平第一约增加平均 0.6761，不能独立冲第一；单芯独立登顶需约 161.1918。执行理由是根因清楚、变更小，不是保证 Top1。

### 2. T75：Ascend 有界 direct no-loop，超大输入保持循环

**有成熟源码，可在 T64 后准备；当前未开发。** 当前 Ascend 使用 BLOCK4096、num_warps8，即使只有一 tile 仍携带 runtime grid-stride loop。[PR #78 Ascend `_direct`](https://github.com/flagos-ai/FlagGems-sglang/blob/077fdc3a0d8d7af02b8b84f131e20399c962c1d3/src/flaggems_sglang/runtime/backend/_ascend/ops/softcap_inplace_logits.py#L22-L39) 和 [#56 有界 direct 分派](https://github.com/flagos-ai/FlagGems-sglang/blob/9e36df5e4bfe146982bf8fa4b1f98bb496fe952c/src/flaggems_sglang/runtime/backend/_ascend/ops/apply_token_bitmask.py#L133-L152) 是固定一手来源；本题昆仑 E3 同数学 no-loop 平台 0.263→0.8994 提供异芯实证。

候选应只改 Ascend 控制流，保留 TB E9 的 BLOCK4096/warps8 和 fp32 sigmoid 公式，`cdiv(n,4096)<=65535` 用 direct，超过则保留原循环。不得复制 #78 的数学/块宽/阈值调参，不得删尾 mask。新增边界必须覆盖分派两侧，可用独立小 cap 的测试入口验证 fallback，正式源码不改变真实阈值。

NO-GO：若编译 IR 已消除原单轮循环，则源码改写没有根因证据；NVIDIA 微秒噪声不足以证明 Ascend 收益。昆仑 no-loop 成功不能推出 Ascend 成功，T71 同套路也未过门。E9 warps8 和全部 Enflame BLOCK 阶梯已关闭，不能重试。

实时排名 10/14：2.89671667 vs 5.13165833，华为 1.27826667 vs 15.93793333。华为差额占平均分净差约 82%，单华为独立登顶需约 19.1578；direct 结构本身尚无支撑 15 倍提速的证据。

### 3. T73：broadcast gate 在 persistent 行循环外驻留

**只作为低成本条件研究，不直接发射。** 当前 Enflame `_rga_capped2d` 的 cols、mask、gate 指针均与 row 无关，但 `tl.load(g_ptr+cols)` 在 row loop 内。已合并 [PR #50 `_performance_fixed_multi_mixed_kernel`](https://github.com/flagos-ai/FlagGems-sglang/blob/987e6070a6b9435bf8cf19298468291e08277552/src/flaggems_sglang/runtime/backend/_enflame/ops/mamba_layernorm_gated.py#L226-L258) 在循环外加载共享 weight/bias，提供真实同芯结构来源。只复用加载位置，不复制其 fp16 affine、精度折中或 shape 特判。

先检查编译 IR 是否已经 LICM；若没有，外提一份 gate 向量，保留逐元素乘积先舍入到输入 dtype、加法 fp32、fusion=False。仅影响 broadcast 且 rows>24 的重复行循环，same-shape 不变。理想流量从三读一写变为两读一写加摊薄 gate，最大约 4/3，缓存命中时更低；寄存器延长生命周期也可能负向。

NO-GO：当前两路径是二选一单 kernel，gate.reshape 是 view，没有额外 launch 或 memcpy 可删。不能以两路径名字声称“两次 kernel”，不能改 FMA/舍入语义。实时 5/9：4.0993125 vs 6.41778125，燧原 2.10325 vs 18.29125；仅本芯登顶需 20.651，外提流量上限明显不够，应低于 T64/T75。

## 排除的重复与失败轴

- T63：E5 已去掉 wrapper clone；当前非空为单 kernel，E10 平铺映射、E11 scalar-base、E12 Metax split2048 已判负，E13–E15 BLOCK 扩大属已走完阶梯。没有发现成熟的新根因消除，不再用“clone/launch 税”解释。
- T64：destination-gather 去 clone 历史代理 0.35x，宽行变体 0.25–0.60x；不可混入本次最小修复。E3/E4 宽度和 E5 topk constexpr 已试，空 task 由旧宽度计算遗留，才是本次新增发现。
- T67：单写融合已做；列分块、固定块、n_cols constexpr、load 行谓词均已阴性。#56/#70 的一般分块形式不能构成重复发射理由。
- T72：E9 15841 已八芯正确但均值 2.48429167、华为 1.1215 未过门，小 group 驻留轴关闭；#50/#77 不构成重投理由，旧 uncertain 不再触碰。
- T66：split-K 正由主线独立推进；#49/#72 是可参考 GEMM 家族，此报告不重复创建候选。

下一步已授权范围：仅在 artifacts 隔离目录准备 T64 TB 单变量修复、持久回归与 AB/BA 计划；主树算子、测试、账本不改，GPU 等调度授权，commit 与平台提交由主线程负责。
