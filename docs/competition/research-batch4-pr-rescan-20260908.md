# 第四批 PR 重扫与榜位重规划（2026-09-08）

数据源：FlagGems-sglang 全量 56 PR（GitHub API state=all）+ 19 个竞赛 PR diff
精读；第四批公开榜 + 认证榜（含我方排名，快照
`artifacts/competition/batch4-gap-20260908/leaderboard-auth-now.json`）。
本页是研究结论，不改变任何账本终态。

## 一、榜位快照（09-08 10:0x，认证口径）

| 题 | 算子 | 我方 | 排名/达标队 | 榜首（队） | 状态 |
| ---: | --- | ---: | ---: | --- | --- |
| 47 | chunked_sgmv_expand | 25.005 | **1** /3 | 25.005（我方） | 守榜 |
| 58 | w8a8_block_int8_matmul | 220.75 | **3** /3 | 576.32 c2flow | e3 新 TB，有效榜末位 |
| 43 | causal_conv1d_update | 6.546 | 3 /4 | 7.903 | 贴身 |
| 57 | log_scaling_tau | 2.503 | 4 /8 | 2.817 EvokeAgent | 贴身 |
| 49 | ernie45_rope_fused | 8.583 | 5 /5 | 17.063 KernelX | 末位 |
| 46 | chunked_embedding_lora_a | 14.105 | 6 /6 | 24.414 c2flow | 末位 |
| 51 | fla_layernorm_gated | 5.816 | 6 /10 | 6.668 c2flow | 贴身 |
| 53 | fused_gdn_gating | 2.955 | 10 /11 | 288.43 Nectar | 大差距 |
| 56 | l2norm | 3.207 | 9 /12 | 72.31 sikadeer | 大差距 |
| 42 | act_and_mul | 3.258 | 11 /22 | 431.48 | 大差距 |
| 48 | chunked_sgmv_shrink | 4.749 | 3 /3 | 23.744 KernelX | 大差距 |
| 44/45/50/52/54/55 | — | 无有效分 | — | 77.95/17.96/10.72/5.43/10.22/6.36 | 封存或等证据 |

## 二、PR 重扫核心结论

1. **09-07 之后上游零动静**（最新 #56，09-06 22:45Z）。T53/T56/T58/T48 的
   榜首结构**没有公开镜像**——"等 PR"对这四题当前不成立。
2. 但相对 09-07 已知清单（#33/#34/#40/#47/#49/#50/#51），存在大量未记录
   PR：#37/#38/#39/#41/#54/#55 已 merged，#42/#43/#45/#46/#48/#52/#53/#56
   为新增。**强队正把"平台最佳提交"批量回灌成 PR**（#43 自述 Task21 同款、
   #48–#53 自述与团队平台最佳逐 AST 一致）——上游已是排名代码公开镜像，
   需例行监控（每日早晚各一次）。
3. 强队画像：xuanzhengdu-eng（8 PR 批量回灌，T48/T49 榜首同名队）、
   yzw1128（昇腾/燧原/昆仑三芯微架构最深的硬核贡献者）、c2flowDS、
   HAi-WORLD、Sawyer117 四人新队。

## 三、可直接复用的技巧（按我方题目映射）

| 来源 | 技巧 | 映射题与用法 |
| --- | --- | --- |
| **#50** | 按 benchmark 精确 shape 分发：`hidden/group/row_count∈{1,8,64,512,4096}/eps` 全 constexpr 特化 kernel + host 分发；2D/3D tile weight/bias 单次载入；Enflame `num_warps=1`/`enable_fp_fusion`/replicas 副本网格 | **T51**（5.82→6.67 差距大概率=shape 特化）；T57 亦可借 |
| **#34** | 燧原两定律：index 带偏移毁向量化（偏移移到标量基址）；依赖加载值的 load mask 毁向量化（段边界用 load 后 `tl.where`）；`searchsorted` 慢 1.5–45x。昆仑：**散读比散写贵 7x** → 小 total_len 转置布局做 stride-1 | **T43**（燧原 0.33/昆仑 0.41 两弱芯的现成修法） |
| **#33** | Ascend decode 固定 dispatch ~70–76us 支配；masked lane 照烧 exp/div；UB 上限 8192；尾数精确 2 的幂拆块零 mask（比 masked 尾块快 ~12%）；host 极简（new_empty/不 view/内联 next_pow2） | **T57**（launch-bound 论证）、**T42**（爬位非登顶） |
| **#49** | 小 K 弃 tl.dot 用 `static_range(rank)` 外积；**昆仑 fp16 物化+显式 sync+fp16 dot 流水**（scale 乘进权重） | **T46**（昆仑 0.23 弱芯）、T58 思路印证 |
| **#51** | 转置 gather tile（token 维 stride-1）；Ascend persistent kernel（固定 core 数+work 循环）且 perf 路径全去 mask | **T46**（Ascend persistent 现成配方） |
| **#47** | 昆仑 P800 四条硬约束：dot+softmax 融合崩；循环内 dot 操作数 masked load/where 崩（pad 零无 mask 读）；dot acc 广播 rescale 崩；大索引 int32 回绕 | 知识库（印证 T14/15/16/50 封存判定） |
| **#53/#52** | 无 dot 向量点积 attention；exp2 在线 softmax（log2e 折进 scale） | T50 若重开 |
| **#40** | 昆仑 rope 配方完整版：3D token×head块×pair、`num_warps=1`、独立 tail-copy kernel | T49 微调 |

## 四、重规划（额度 21 发，窗口至 09-10 19:59）

| 优先 | 题 | 候选 | 依据 | 发数 |
| ---: | --- | --- | --- | --- |
| P0 | **T51** | shape 特化 constexpr 分发 vendor（按计分 shape 烘焙 row_count/eps，host 分发+generic 兜底） | #50 直接技巧，1.15x 差距最可能就差这个 | 1–2 |
| P0 | **T58** | BLOCK 128×128×128（group 对齐）+ constexpr 全 tile | 自有杠杆，fp16 轴已证；守 #2 争 #1 | 1–2 |
| P1 | **T43** | 燧原 vendor 按 #34 两定律重写 + 华为 E15 字节复验（分芯单变量打包） | #34 现成修法 + E15 +41% 旧账 | 1–2 |
| P1 | **T57** | 零开销 wrapper + flat launch（#33 dispatch 论证） | 1.13x 最近差距 | 1 |
| P2 | **T46** | Ascend persistent vendor（#51）+ 昆仑 fp16 物化（#49） | 末位→中游 | 2（次日） |
| 守 | T47 | 榜首不动，被超才投 | — | 2 备用 |
| 尾 | 高水位题 | T58/T53/T56 最佳字节窗口采样 | 水位彩票 | 剩余 |
| 停 | T42/T48/T53/T56 结构追击、无效题 | 情报不足/平台阻塞 | — | 0 |

每日早晚各重扫一次 PR（强队正在批量回灌，任何 T53/T56/T58/T48 新 PR 立即
触发对应候选）。
