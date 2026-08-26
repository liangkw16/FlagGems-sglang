# FlagOS 第二批高倍数杠铃冲刺设计

日期：2026-08-26（Asia/Shanghai）

## 目标与事实基线

本轮采用“高倍数有效题 + 现实夺首题”的杠铃策略。目标不是耗尽额度，也不是继续
用 BLOCK、warps、stages 碰运气，而是：

1. 用少量结构候选补齐已有高底盘题的失败芯片；
2. 只把大部分后续额度投给已经出现正信号的任务；
3. 并行保留一条离榜首差距较小的有效题夺首线。

2026-08-26 12:14 CST 的平台只读状态为账号
15600308080、团队 SoulCoder、今日额度 29/30，截止时间
2026-08-27 19:59:59。按用户给定的两日约 60 次额度和今天已使用 1 次计算，
理论剩余上限为 59 次；每次发布前仍以实时平台返回为准。

公开实时榜单与团队最近一次逐芯结果形成以下决策基线。valid floor 把每个当前失败
芯片按最低门槛 0.1x 计入；“夺首所需”只表示算术下限，不代表可实现性。

| Task | 当前榜首 | 团队通过底盘 | valid floor | 夺首所需失败芯表现 |
| ---: | ---: | ---: | ---: | ---: |
| 23 sgemm_lora_b | 40.561375x | 7 芯合计 190.407x | 23.813375x | 昆仑 134.084x |
| 13 chunk_state_varlen | 707.0045x | 6 芯合计 1301.519x | 162.714875x | 两芯平均 2177.2585x |
| 17 embedding_lora_a | 25.583x | 7 芯合计 109.952x | 13.7565x | 燧原 94.712x |
| 22 qkv_lora_b | 181.7155x | 6 芯合计 328.264x | 41.058x | 两芯平均 562.73x |
| 15 decode_attention | 78.2958x | 5 芯合计 314.7038x | 39.375475x | 三芯平均 103.8875x |

因此 T23 E7b 不再承担夺首目标；它只用一枪争取约 23.81x 的有效成绩。T13 是绝对
倍数主战场，但同样不以追平 707x 为现实门禁。现实夺首副线从 T19/T21 中选择：
T19 团队 4.5467x、榜首 5.55235x，差距约 22%；T21 团队 2.7096x、榜首
3.69085x，差距约 36%。

## 组合、顺序与总止损

选择杠铃式预算，不选择激进广撒网或保守只调已有 valid 题：

    高倍数主线：
    T23 E7b（1 次）
    → T13 两阶段规整 BMM（最多 2 次）
    → T17 路由展开 + 二维 gather（最多 2 次）
    → T15 pack + split attention + merge（最多 2 次）
    → T22（仅 T23 昆仑全 case 通过后，最多 1 次）

    现实夺首副线：
    T19（先过零提交准入门槛）
    → T21 flat fused reduction（先过代理收益门槛）

同一任务两个结构不同的候选均失败即停止。结构性编译失败后禁止普通
BLOCK/warps/stages 微调；只有像 E7→E7b 一样已经定位到精确资源点时，才允许一个
最小资源收口候选。高倍数补齐候选冻结已通过芯片的 generic/vendor 字节；夺首副线
只允许修改预先声明的 affected vendor，其余芯片字节冻结。

“结构不同”要求改变 kernel 边界、数据布局或核心数据流；只改 launch 配置不算第二个
结构。“显著缩小榜首差距”要求候选投影后的绝对差距至少减少 10%。

“正信号”至少满足一项：

- 恢复一个此前失败芯片；
- 任务首次变为 8/8 valid；
- 平台或可迁移代理证据显示候选显著缩小实时榜首差距；
- 目标芯片跨过预先声明的性能晋级阈值。

没有正信号时不为消耗额度继续提交。

## 高倍数主线架构

### T23 E7b：诊断桥与一次性有效化候选

E7 已把昆仑失败从 1830 秒编译超时缩短为 8.4 秒内的 pack-W
uni_sram 错误。E7b 保持以下数据流：

    ragged/permutation X ──pack──> 连续 padded X
    adapter weights ───────pack──> 连续 [B,K,N] weights
                                  ↓
                        regular 32×32×32 BMM
                                  ↓
                        scatter + base + scaling

pack-W 改为每个 program 复制一条 rank 行上的连续 N256 元素，不再让 vector lane
执行运行时除法/余数；BMM/scatter 收敛到 32×32 资源形态。source commit 为
a5c5d0bd74d399716e1e614c9b0e897e30cda034，最终 screening 6/6 通过，
NVIDIA 主形状代理中位为 1.6145 ms，慢于 E7 的 1.4817 ms，因此它是可编译性
候选，不是提速突破。

决策：

- 昆仑失败：停止 T23，不做 no-dot 保底；
- 8/8 且昆仑 <15x：保留 valid 成绩，停止 T23；
- 昆仑 ≥15x 且候选平均达到实时榜首的 70%：记入放大池，但不阻塞 T13；
- 当前实时夺首需要昆仑 134.084x，不把该值作为 E7b 的合理预期。

### T13：metadata pack 与 regular BMM 解耦

不再先试 runtime loop 静态化小修，直接使用两阶段 vendor：

1. metadata kernel 读取 cu_seqlens、末 chunk 边界、dt/dA scale，把 X 与
   B*scale 整理成连续 padded 数据；
2. regular BMM 只读取连续张量，输出
   [batch,nheads,headdim,dstate]。

每条序列只 pack 题面要求的最后一个 chunk，padded 长度上限为 chunk_size，不复制
整条历史序列。

昆仑采用 32×32×32 FP32 IEEE 规整 BMM；燧原采用已在同族任务证明可执行的大 tile
低精度 dot 结构，并服从题面 3e-2 容差。公开 reference 的 slice/broadcast 语义由
pack 阶段完整保留，不把 chunk_states 数值带入计算。

首个候选若恢复至少一芯，第二次只改仍失败芯片；两芯均失败则停止。

### T17：路由展开与规则二维 gather

只改燧原 vendor：

1. 简单 kernel 将 segment metadata 展开成逐 token 的 adapter/rank 路由表；
2. gather kernel 只处理规则的 token × rank 向量块。

该结构删除运行时 rank 循环、early return 和 masked scalar metadata load；extra
embedding 与普通 embedding 使用向量 masked load。首版失败时只允许一次安全地址
钳制、无 scalar mask 的变体。

### T15：pack、split attention 与 merge

放弃单 kernel 在线 softmax。失败芯片采用三阶段：

1. 展开 CSR/page 索引，形成连续 KV 访问；
2. 每个 KV split 计算局部 (max, sum, weighted_value)；
3. merge kernel 以稳定公式合并 partial state。

只 pack 当前 KV split，不物化完整 logits。若局部状态为 (m_i, l_i, a_i)，merge
使用 m=max(m_i)、l=sum(l_i*exp(m_i-m))、
a=sum(a_i*exp(m_i-m))，最终输出 a/l。临时状态规模与 split 数和输出维度成正比，
不与完整注意力矩阵大小成正比。

华为使用 1D capped grid，避免 coreDim > 65535；昆仑把 ragged metadata 与 dot
隔离；燧原不让 cumsum/runtime 分支进入 Pipeline pass。首个结构候选若未恢复至少
两个失败芯片，则停止 T15。

### T22：条件复用

只有 E7b 在昆仑全 case 通过时才允许开始。复用 pack/BMM/scatter，但
output_offset 和 slice metadata 只能出现在 pack/scatter，不得进入 dot kernel。
该题只有一次结构提交预算；T23 结构失败则直接跳过。

## 现实夺首副线

### T19：零提交准入

旧的全局 multi-row 候选在 NVIDIA affected 矩阵几何均值仅 0.5715x，昆仑专用
multi-row 也未改善平台分数。因此没有新证据时不提交 T19。

只有结构不同的候选在固定 shape 矩阵上达到以下条件才进入平台：

- wrapper-inclusive affected 几何平均 ≥1.25x；
- 任一 guard 点 ≥0.95x；
- 无新增 spill、scratch 或 local load/store；
- 能解释约 22% 的榜首差距，而非只优化一个低权重单点。

### T21：flat fused reduction

若 T19 未过准入，转 T21。候选按连续输出块处理，每个 lane 对 TOP_K 做编译期
循环，减少现有二维 grid 和 hidden-block 调度开销；优先只新增燧原 vendor。昆仑
保留已经平台验证的 BLOCK 1024，其他通过芯片字节不变。

affected 代理几何平均需 ≥1.20x、最差点 ≥0.95x，才允许一次平台提交。普通
BLOCK 4096 已被 E4 代理否决，不作为候选。

若 T19/T21 均无正信号，不用 T20/T24 的普通 BLOCK 微调填额度，放大池全部转给
高倍数主线中已经出现正信号的任务。

## 验证、发布与错误处理

每个候选使用相同最短闭环：

1. 一个永久最小回归覆盖平台失败 shape，并覆盖题面 dtype、非连续输入、空段和
   相关 metadata 边界；
2. py_compile、Black、isort、flake8；
3. 远端 NVIDIA screening 后台运行，包含 correctness、wrapper-inclusive A/B 与
   编译资源检查；
4. screening 字节与 source/test Git blob 逐项比较；
5. 只从明确 source commit 的 Git object 建立 release，重跑同一正确性门禁；
6. 生成确定性 ZIP，验证成员、UTF-8、大小、unzip -t 和 SHA-256；
7. 实时 preflight 核对账号、团队、Task、stage、commit、成员、额度和截止时间；
8. tuple 全匹配时执行一次 submit，等待八芯终态并写回账本。

性能候选默认要求 affected 几何平均 ≥1.10x、最差点 ≥0.95x 且无新增
spill/scratch；T19/T21 使用上文更严格阈值。目标 vendor 无代理环境时，NVIDIA
只证明语法、数值和资源，平台第一次提交是运行证据；只有平台正信号才允许一个收口
候选。

平台状态为 sending、uncertain、stale_after_upload 或已提交时绝不重试。
候选失败按根因分类：数学错误修 generic；单芯错误只改该 vendor；精确资源错误允许
一次最小收口；无法定位或第二个结构失败则退出该任务。

## 59 次动态预算

额度是上限，不是目标：

| 资金池 | 上限 | 使用条件 |
| --- | ---: | --- |
| 结构探索 | 8 | E7b 1、T13 2、T17 2、T15 2、T22 条件 1 |
| 正信号确认 | 8 | 每个正信号只允许一次单变量收口 |
| 性能放大 | 39 | 70% 给当前边际收益最高任务，30% 给第二名 |
| 最终储备 | 4 | 截止前最佳候选回归与不可变发布 |

每轮平台终态后重新读取公开榜首与账号额度，计算当前候选的边际平均分和夺首差距，再
决定下一笔预算。没有达到晋级门槛的候选不进入 ZIP/preflight；没有正信号时允许额度
留空。

## 非目标

- 不在本轮穿插 T10/T11/T14/T16/T18；
- 不追 T8 的 709x 异常高分；
- 不为一个失败芯复制或改动已经通过的七芯实现；
- 不使用设备判断、异常 fallback 或纯 PyTorch 核心路径；
- 不把 NVIDIA 代理结果表述为昆仑、燧原或华为平台性能证明。
