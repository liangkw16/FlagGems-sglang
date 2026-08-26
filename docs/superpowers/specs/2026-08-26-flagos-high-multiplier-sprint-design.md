# FlagOS 第二批高倍数冲刺设计

日期：2026-08-26（Asia/Shanghai）

## 目标与边界

在 2026-08-27 19:59:59 截止前，按以下固定顺序尝试高倍数候选：

```text
T23 sgemm_lora_b
→ T13 chunk_state_varlen
→ T17 embedding_lora_a
→ T22 qkv_lora_b
→ T15 decode_attention
```

每题只改当前失败芯片的 vendor，已经通过的芯片文件逐字节冻结。一次平台提交只验证
一个可解释变量；候选通过测试、release、不可变 ZIP 和实时 preflight 后，按现有授权
自动单次提交。`sending`、`uncertain`、`stale_after_upload` 或已提交候选不重试。

本轮不穿插 T10/T11，也不追 T8 的 709x 异常跃升；仅在上述五题全部停损后重新排序。

## 1. T23 `sgemm_lora_b`

现状：七芯合计 192.419x；昆仑只要达到 0.1x，平均即约 24.0649x；昆仑达到
37.013x 可超过当前 28.679x 榜首，提交目标取 39x。

首选 E7 只重写 `_kunlunxin`：

1. 简单 Triton kernel 将 ragged/permutation 输入打包为连续 padded X，并选择权重；
2. 规整 batched GEMM 只处理连续 `[bs,max_len,K] × [bs,K,N]`；
3. scatter-add 写回 permutation 行并与 FP32 base 相加，最后 cast 到输出 dtype。

这样把 ragged metadata 与昆仑 SDNN dot 完全隔离。E7 若为 15–39x，只再投一个把
scatter 融入 BMM 的 E8；E7 若仍编译失败，最多投一个无 dot 的 FP32
multiply+sum 保底。最多 3 次提交。

停损：E7/E8 任一达到昆仑 39x 即停止；得到有效但低于 15x 也停止追分；三个结构
候选均失败则进入 T13，不再调 BLOCK、warps 或 stages。

## 2. T13 `chunk_state_varlen`

现状：六芯合计 1301.519x；燧原和昆仑各按 0.1x 计时，平均约 162.7149x。

E2a 分别新增燧原/昆仑 vendor：把 runtime `sequence_length` 循环改为以
`CHUNK_SIZE` 为编译期上界的静态循环，保留 mask，避免 varlen loop 进入 dot
lowering。若仍有芯片编译失败，E2b 用第一阶段预计算每段边界和 scale，第二阶段只做
规整 per-sequence/head GEMM。

最多 5 次提交。首个结构候选若两芯都未通过则直接使用两阶段方案；两阶段仍不能让
任一失败芯片通过时停止。本题目标是获得约 162x 的有效成绩，不以追 707x 榜首为门禁。

## 3. T17 `embedding_lora_a`

现状：七芯合计约 110.118x；燧原达到 0.1x 后平均约 13.777x；追平榜首要求燧原
约 80.9x。

候选顺序：

1. E2a：删除 runtime rank loop，使用 `next_power_of_2(weights.shape[1])` 的单个
   编译期 rank block；
2. E2b：若仍失败，使用安全地址钳制并移除 scalar masked metadata load；
3. E2c：有效后再以 8 个 segment 为一组向量化 metadata，减少 CTA 数。

先用 3 次解决编译；只有候选已经有效才追加最多 3 次调 `BATCH_BLOCK=4/8/16` 或
grid cap。总上限 6 次。

## 4. T22 `qkv_lora_b`

现状：六芯合计 328.264x；燧原和昆仑各按 0.1x 计时，平均约 41.058x。

只有 T23 的 pack/规整 BMM/scatter 在平台证明可行后才开始。复用该数据流，将
`output_offset` 定义的 Q/K/V slice 展平到规整 BMM；仅为燧原和昆仑提供 vendor，
其余六芯保持原字节。若 T23 的两个规整 BMM 候选都未通过，则跳过 T22、直接进入
T15。T22 最多 5 次提交。

停损：首个移植候选若两个失败芯片都未通过，最多再做一个无 dot 保底；仍失败即转
T15，不继续复制 T23 的 tile 参数。

## 5. T15 `decode_attention`

现状：五芯合计 314.7038x；三个失败芯片各按 0.1x 计时，平均约 39.3755x。

燧原/昆仑先把 runtime sequence loop 改为静态长度上界；资源或编译仍失败时使用
split-KV + merge。华为单独使用两阶段实现，将 logits/softmax 与 value accumulation
拆开，并按 UB 安全宽度分块，绕开现有整行重复的 load/broadcast lowering 指纹。

最多 6 次提交。首个结构候选若未恢复至少两个失败芯片则停止本题；不做普通 tile
微调。

## 验证、发布与记录

每个候选均执行同一最短闭环：

1. 先补一个会覆盖平台失败结构的最小回归；
2. 本地 py_compile、Black/isort/flake8；
3. 远端 NVIDIA screening 后台运行，主线程继续静态审查；
4. 晋级源码和测试 commit/push；
5. 从 commit Git 对象执行 release，生成规范 ZIP 并记录完整 SHA-256；
6. 实时 preflight 核对账号、团队、Task、成员、commit、ZIP、额度和截止时间；
7. 门禁全部匹配后只执行一次 submit，等待八芯终态；
8. 平台逐芯结果写入对应实验账本，再 commit/push。

候选只因正确性、编译恢复或明确性能增益晋级。远端 NVIDIA 不能证明 vendor runtime；
目标 vendor 无代理环境时只做静态检查，并把平台提交作为唯一运行证据。五题理论上限
共 25 次高信息提交，保留剩余额度用于实际正信号后的单变量收口。
