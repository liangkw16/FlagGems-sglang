# 第五批一轮优化与首次提交（2026-09-11）

用户要求“联网搜索 + 本地经验，先优化一轮，再尝试提交”。本轮检查 T59–T64，
T59/T63/T64 晋级 E1，T60/T61/T62 保留已验证 S0。提交结果随后在各题 CURRENT 与本页更新。

## 依据与取舍

- [Triton 官方分块示例](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)说明按 program 划分独立输出块并使用尾部 mask；本轮沿用这一方式。
- [SGLang KV 索引源码](https://github.com/sgl-project/sglang/blob/8014d9d062c3cc5d393596ecdf2f7009191965df/python/sglang/kernels/ops/kvcache/kv_indices.py)的 FlashMLA 相邻实现用 grid 轴 1 拆页块；T63 将此结构用于长段，并保留长度驱动的 grid-stride 循环。
- [SGLang MoE 源码](https://github.com/sgl-project/sglang/blob/8014d9d062c3cc5d393596ecdf2f7009191965df/python/sglang/kernels/ops/moe/ep_moe_kernels.py)对每个 token 循环 hidden 块；T64 将 hidden tile 改为独立任务，仍在 tile 内向各有效目的行复用输入。
- 本地 T48 教训要求长段不能只处理第一个 tile；T21 的昆仑 grid 上限经验要求展平任务数受限，超出的任务继续 grid-stride；T58 的向量除法编译失败记录促使 T59 将行/块索引改为标量。跨芯收益均为待平台验证的假设。
- T60 已是单次融合，T61 已是直接逆排列 scatter，T62 已有 head 共享与 int64 地址修复；缺少新的可证实瓶颈，本轮保持。

实时逐芯榜单已读取，原始快照见证据清单。其分数用于识别目标芯风险；不将他队异常读数解释为确定的计时原因，不把 NVIDIA 比值外推为八芯均值。

## 配对筛选与发布验证

预注册晋级门：两个代表负载至少一个 E1/S0 中位收益 >=1.05，另一个 >=0.95；完整正确性通过，spill/shared memory 为 0，寄存器 <=64。
GPU 无其他计算进程；同一进程、同一输入五轮交替 AB/BA，含 wrapper、warmup 20ms / rep 50ms。

| Task | 负载 1 / E1 相对 S0 | 负载 2 / E1 相对 S0 | E1 寄存器 |
| --- | --- | --- | --- |
| T59 | 8×128 / 1.0663x | 64×2048 / 1.0083x | 38 / 40（S0 为 29） |
| T63 | 1×8193 / 1.9101x | 32×1024 / 1.1045x | 34 / 30（S0 为 40） |
| T64 | 8×4×4096 / 1.5346x | 512×4×4096 / 1.0027x | 19 / 19（S0 为 21） |

所有测得编译资源均无 spill 和 shared memory。T59 寄存器上升，目标芯表现仍需平台确认。
筛选只执行代表负载的参考对比，不能替代完整回执。源码/测试提交后从 Git 对象重新准备 release，
三个 E1 的完整 13 个测试方法全通过；加上未变更 S0 的既有回执，六题合计 **24 个测试方法、284 条 test/subTest 记录、146 次实际 kernel launch**，无失败/错误/skip/xfail。
新增覆盖 T59 65537 行、T63 32769 长段/129 请求、T64 超过 65535 个逻辑 tile；继续覆盖尾块、stride、空输入及原输出未写区域。

设备：NVIDIA RTX 5070 Ti，PyTorch 2.13.0+cu130，Triton 3.7.1，CUDA 13.0。目标芯 runtime 仍为 `target-runtime-unverified`，由首次平台提交补齐。

## 不可变证据

- E1 source / verification commit：`29754c1dfcbc87c9b193e750145b668c8e35d852`。
- T60/T61 S0：`4836fe0378d6290cbcaa91958693e13ed1acdf3e`；T62 S0：`7b53fdeb58878f8eeaf08eaabbde8bb38015e523`。
- [完整证据清单](data/batch5-optimization-20260911.json)：全部 ZIP/源码/测试/回执/日志/测速哈希、五轮样本、编译资源、启动记录和逐芯榜单。
- 清单 SHA-256：`87a560dd86ab7e9dfb9607f8f05a007060d48b3aab876b165bfc0fec591f77db`。
- 原始产物：`artifacts/competition/batch5-optimize-20260911/`；提交记录：`artifacts/competition/batch5-submit-20260911/`。
