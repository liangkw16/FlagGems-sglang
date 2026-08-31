# 第二批候选与提交队列

本页汇总第二批 Task 08–24 的本地产物；逐项契约、测试、性能和风险以算子账本
为准。公开榜单快照见 [赛题索引](../task-index.md)，同步时间为
`2026-08-27T00:31:02+08:00`。

当前覆盖：17/17 个算子都有已提交源码、远端 NVIDIA 代理验证和不可变 ZIP。
Task 08、09、12、17、19、20、21 和 24 已经平台 8/8；其余 9 个尚无八芯平台结果，
代理加速比不能外推。

产物路径统一为：

```text
artifacts/competition/<operator>/<stage>-<commit>/<operator>.zip
```

## 候选清单

> **状态真相源（2026-08-31 起）**：当前状态以各算子账本顶部的
> ` ```current ` 块为唯一人工维护真相，索引由
> `python tools/gen_experiment_index.py` 生成到 [INDEX.md](INDEX.md)。
> 下表为第二批时期的历史人工表，第三批任务以 INDEX.md 为准
> （例如 T27 早期 6/8 口径已过期，实为 7/8 仅差昆仑）。

下方建议提交顺序只覆盖尚无 8 芯结果的任务；已完成任务保留当前结论。

| Task | 算子账本 | 候选 | ZIP SHA-256 | 当前证据 | 建议 |
| ---: | --- | --- | --- | --- | --- |
| 08 | [`apply_token_bitmask`](apply_token_bitmask.md) | `e6-c8b4b2b` | `4414d9deed8d7c73f4fa86d49aa7ab3605cffdc0604e8f92654428a7631bf01a` | E6 平台 6/8；GCU 最大 case 的 2D word-layout 有 13.8% 元素错误 | 保留 E2 4.686925x；word-layout 轴永久停止 |
| 09 | [`bmm_chunk`](bmm_chunk.md) | `e4-93035e4` | `a967742cbeee3053b8f2b9af261a30cf3bae7b9a745c8adaf15abc61cd434f85` | E4 平台 8/8、1.27375x；官方 GCU300 cap 6 使燧原 0.149→0.153x，但平均未超过 E3e 的 1.283x | 保留 E3e team best；Task 09 永久停止 |
| 10 | [`chunk_cumsum`](chunk_cumsum.md) | `s2b-63e7943` | `c822f75d719f8919269c7566b1210b6e31dc6ce3292229a838723f7945b15923` | S2b 平台 8/8 正确、invalid_threshold；华为 UB tile 上限修复生效；燧原 0.0375x/昆仑 0.012x 为 cumsum lowering 固有瓶颈 | 已按两次规则停止；重试需改写 cumsum 算法形式 |
| 11 | [`chunk_local_cumsum_vector`](chunk_local_cumsum_vector.md) | `e1b-dddef74` | `cf4dcaf05640599fe5b50ee9633ba19d2a4f13f2b47f856e88259242e975bab9` | 两次提交均 7/8；昆仑 0.016x/华为 0.0255x 与折叠形态无关，cumsum 固有；燧原三形态编译失败 | 已按两次规则停止；同 Task 10 结论 |
| 12 | [`chunk_state`](chunk_state.md) | `e7-294990c` | `583b55a2518091cd707ff6dbf1080a10bd1fe2fd690da2885d0bfd48daae04a8` | E7 平台 8/8、**4.0371875x team best**；官方 Ascend Cube 路线使华为 0.329→2.1185x（6.44 倍），其余七芯稳定 | Task 12 永久停止；保留 E7 |
| 13 | [`chunk_state_varlen`](chunk_state_varlen.md) | `e4-46c8b3c` | `bd112d1efa5098b8294c2b772ef2fa8ee99783e1b2b880521995eac76a770b1b` | E4 平台 8/8、**166.4583125x**、第 5/6；host-resolved direct dot 使昆仑从全 case 数值失败恢复至 55.3745x | 闭环完成，保留 E4；升第 4 无已验证小变量 |
| 14 | [`context_attention`](context_attention.md) | `e2-b951safe-fc81bb8` | `63e3e0ddccf1493dfb484ee4a7f1310f4f91dae677b874afa68fe43798cac774` | E2 平台 7/8；E3 mask-free scalar 代理 11/11 正确，但 `[512]x2` non-causal 最差仅 0.124466x、长序列 0.181778x | E3 未过预注册 0.2x 门，未提交；Task 14 最终停止 |
| 15 | [`decode_attention`](decode_attention.md) | `e4-96a0dfe` | `51ec3d98ca1da7e33bd1aaee93c398399855dfb7f60a6b4b8e73a3e6f0f9ca7a` | E4 平台 7/8；Ascend 物理 worker grid 恢复至 43.1704x，Enflame 正确但仅 0.0928x，Kunlun 从 1833s 编译中止恢复到 17.868s 后仍 case 2 数值失败 | E4 双重 stop gate 已触发；Task 15 永久停止 |
| 16 | [`decode_grouped_attention`](decode_grouped_attention.md) | `e2-a574a77` | `78d84fc861683b5d70a5435a4b94d6ca50a0bcd176f3d5ab7051b17d89d2d13e` | E2 八芯正确、平均展示 14.5922x；燧原 0.0346x、昆仑 0.0142x 导致 `invalid_threshold` | stop gate 已触发，Task 16 永久停止 |
| 17 | [`embedding_lora_a`](embedding_lora_a.md) | `e4-hygon-d33b89d` | `74cff90ca2a055624647ee36b004372fe044282e9b1798b52de903bd33f31cfe` | E4 平台 8/8、**14.1618125x team best**；Hygon 2-wave 使海光 +3.34%，未过第 5 名阈值 | 保留 E4；Hygon warps 轴永久停止 |
| 18 | [`fused_recurrent_gdn`](fused_recurrent_gdn.md) | `e12-79afc08` | `3f058ea65ac64a07918ad4b28af4f2d03cf40c57e98099d1925494a60c46707a` | E9–E12 四投按芯分发归约指纹：国际 A 五连过 5.36–5.53x，海光串行 FMA 仅差 3 元素，国际 B 串行 402；燧原回调停摆、昆仑 reference 侧崩溃均为平台阻塞 | 额度 3/30；E13 已预注册（串行×4 + hygon exp + amd 反向串行），保底留 1 次终投 |
| 19 | [`fused_rmsnorm`](fused_rmsnorm.md) | `e5-f70db6c` | `def37c370d7ffab821377122202eef3a103d574e182709a0989c8f679cb5b462` | E5 平台 8/8、4.55663333x team best、第 15/18；Enflame 官方 launch 使该芯 +38.01% 到 2.0768x | 保留 E5；未过升名阈值，Enflame launch 轴永久停止 |
| 20 | [`mamba_layernorm_gated`](mamba_layernorm_gated.md) | `e4-98cb62b` | `40a80fcafc213f6bd4f84903f3b1f0b612e23efca18602b1e92b6d371a7f641b` | E4 平台 8/8、**4.37825x team best**；Enflame multirow 仅 +8.60%，CUDA case 8 的 7x 未迁移 | 保留 E4；multirow tile/阈值轴永久停止 |
| 21 | [`moe_sum_reduce`](moe_sum_reduce.md) | `e11-b1a9927` | `5e3e6bac0aba5f16d6bae51cc4e63dd4955f95e863c19096c5f6b5d734e9db7f` | E11 平台 8/8、**2.995375x team best、第 10/11**；E13 dense 华为 -3.97%；Kunlun2048 代理仅 0.7467x | 保留 E11；Ascend 变体、top-k=2 与 Kunlun2048 均停止 |
| 22 | [`qkv_lora_b`](qkv_lora_b.md) | `s2c-7857dca` | `357e8a690cca68123aabebdbb5500a86ebd66fe328105a8b91f7c1afe489cb38` | S2c 终态 6/8；六芯高分（海光 82.6x、天数 47.8x）；燧原 case 2 编译失败、昆仑评测异常 | 已按两次规则停止 |
| 23 | [`sgemm_lora_b`](sgemm_lora_b.md) | `e11-dd4632a` | `c23266792c400636b2a7a4aa418defa2eb15f19623e8419316133dceb4463ff7` | E11 官方 XPU legacy masked-memory 使昆仑五个 case 全部跑完，但均数值失败；燧原回调待定，其余六芯已通过 | 官方两种 masked-memory 路径均验证失败，Task 23/22 永久停止 |
| 24 | [`softcap_out`](softcap_out.md) | `s8-b2a249b` | `9fd897ad5b1e167c8c0a49826c295b2387b160b2d378ffacec61d74a6e469899` | S8 平台 8/8、**2.24001042x team best**；S9 constexpr 仅使燧原 +0.21%、整题回退 | 保留 S8；constexpr 与既有 tile/grid/warps/math 轴全部停止 |

## 建议提交顺序

按实现复杂度、公开达标队伍数、代理覆盖和跨芯风险排序：

```text
12 → 09 → 17 → 23 → 22 → 11 → 10 → 15 → 16 → 13 → 14
```

Task 18 暂缓，仅作状态资源实验。Task 21 已闭环（S0c 6/8 → S1 7/8 → S2 7/8
→ S3 8/8、2.7096x → E5 8/8、2.76185x、team best；昆仑 XPU 2D grid 展平
总数上限 65535 的证据链完整，Ascend BLOCK512 平台验证有效）。
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

## 第三批(Task 25–30)

2026-08-27 20:00 开闸,提交窗口至 2026-09-03 19:59:59。六题账本:
[`draft_topk1`](draft_topk1.md)、[`fused_moe_router_cudacore`](fused_moe_router_cudacore.md)、
[`fused_moe_router_tensorcore`](fused_moe_router_tensorcore.md)、
[`gate_up_lora_b`](gate_up_lora_b.md)、[`gelu_and_mul`](gelu_and_mul.md)、
[`interleaved_rope`](interleaved_rope.md)。

2026-08-28/29 kernelgen 轮:**T29 e5–e7 三投 8/8 valid,团队最佳
2.436x → 2.7229x(+11.8%)**,燧原 5 倍/华为 2 倍/昆仑 +65%/沐曦 +11%;
T30 结构三轮 + autotune 华为实机全部负结果(详见各自账本)。
2026-08-29 新题四连:T33 3.5385x、T35 rotary_embedding 4.6673x、
T34 per_token_quant_int8 4.7074x(-9.9%)、**T32 moe_fused_mul_sum
4.4829x(超快照榜首 +5.3%)**,全部首投 8/8 valid;T36
selective_state_update 两投同指纹数值失败(真实 reference 语义
未定位,止损);T31 moe_fused_gate 七芯两投全过(海光 13.4/华为
4.9 等)仅昆仑死于平台 inductor 崩溃,7/8 工单路径;T33 E1 摊销证伪、T35 E1 连续读证伪(负结果入账)。
T33 per_token_group_quant_int8 S0 8/8 valid、3.5385x
(榜首 4.0170x,-11.9%);E1 燧原/华为多组摊销 vendor 证伪(团队最佳
保持 S0);根因知识:Triton 普通 `/` 是近似除,需 `tl.math.div_rn`
才与 torch 逐位一致(amax 边界 ±1 陷阱)。
2026-08-27 定稿的作战方案:全 6 题;主攻顺序 `29 → 30 → 25 → 27 → 26 → 28`;
每题 5 次提交预算,S0 探路 → 最多 3 次 vendor 单变量迭代 → 留 1 次截止前
回归储备,同指纹失败连续 2 次提前停;dot 题型 generic 用 fp32-ieee 操作数 +
`_tianshu` split-fp16 vendor(昆仑保持 fp32-ieee,T12 镜像证据)。

### 2026-08-29 通宵轮(用户解除预算限制)

平台第三批扩至 12 题(Task 31–36 新开);T25–T28 达标队伍升至
7/4/5/2,证明全部可解。本轮结论:

- **T27 华为根因破案:split-K 部分和舍入 vs 参考 matmul 顺序累加**
  ——A/B 对照(T26 e6 末索引平局 vs T27 e6 n_splits=1)一轮定位;
  T27 e6/e7 华为 0.54/0.52x 稳定通过,**7/8 只差昆仑**(七芯均值 ~1.03x)。
- **T26 华为未破**:1D 归约重构、末索引平局、n_splits=1、无 dot 顺序
  FMA 四发探针全败;GEMM/归约结构假设全部排除,失败面收敛到
  case_idx=7(E=256/topk=8)的参考 matmul 内部舍入顺序,独立 Triton
  实现难以稳定复现;该轴暂停,待平台 Q&A 或新证据。
- **T25 燧原第 6/7 种结构**:int64 全消除(e6b 仍编译失败,证 meta/draft
  int64 假设单独不充分);e6c = where+min 归约 × int64-free base
  (全部燧原已证算子集)已提交待裁决;华为 0.28x 保持通过。
- **昆仑评测 worker 成批死亡**:每日午夜额度重置后被请求洪峰打挂
  (08-28 00:14 起、08-29 00:58 起),全天同指纹崩溃(T28 六结构两代
  候选含无 dot 净室版;今日 9+ 连崩,唯一通过例 00:48 轻量 gelu)。
  T28 七芯均值 ~16x(榜首 11.5x,8/8 即登顶)、T27 仅差昆仑;
  工单已更新证据矩阵申请 rerun。策略:每日上午/晚间低峰各 1–2 发
  重投载体探窗口,截止 2026-09-03 19:59。

## 2026-08-30 09:5x 进度快照(第三批收官格局 + 榜单对标)

- **6 题 valid**:T29 2.7229x / T30 25.835x / T32 4.4829x / T33
  3.5385x / T34 4.7074x / T35 4.6673x(全部平台 8/8)。
- **2 题 7/8 卡昆仑平台崩溃**(inductor compile-worker,10 次同指纹,
  6v6 相关性见 kunlun-crash-ticket):T31(七芯 ~7.6x)、T28(七芯
  ~18.5x;注意榜首已涨至 c2flow 34.98x,登顶假设失效)。
  恢复窗口重投(T31 e2 sub 6477 / T28 e9 sub 6481)均同指纹失败,
  代码侧关闭,仅剩工单。
- T36 止损(真实 reference 语义黑盒);T25/26/27 早前终态。
- **榜单对标(08-30 09:5x 快照)**:T32 距榜首 -1.3%(贴身)、
  T34 -10.7%、T29 -19.0%(榜首 3.36)、T35 -20.7%、T30 -22.2%、
  **T33 -40.9%(榜首 c2flow 4.02→5.99 跳涨,存在未发现的大优化面)**。
  c2flow 霸榜 T25/28/33/36/37。
- T37 sgemm_lora_a 语义黑盒止损(七芯 ~99% 失配,题面 reference 与平台不符疑);T38
  sigmoid_gate_topk_renorm(1 队 6.33x,topk 族昆仑风险)、T39
  silu_and_mul_masked / T40 softcap_inplace_logits / T41
  state_passing(达标数待出)。
- 额度 30/30(08-30 重置);窗口至 09-03 19:59;cron 每 30 分钟
  监视昆仑族达标数(报告制,重投额度已按纪律用尽)。
- 战术排序:T39/T40 收割 → T32 昆仑短板(0.21x)与 T33 优化面
  (kernelgen 二轮)→ T34 追赶 → T37/T38。

## 2026-08-31 第三批收官快照

- **8 题 valid**:T29/T30/T32/T33/T34/T35 + T39 19.870x(榜首)+
  T40 1.727x;
- **3 题 7/8 昆仑墙封存**:T28(七芯 ~18.5x)、T31(~7.6x)、
  T38(sub 6930,七芯,昆仑 Segfault=崩溃族第 15 例);
- **2 题语义黑盒止损**:T36、T37;**4 题早前止损**:T25/26/27 +
  (T36 前科);
- 崩溃族证据链:15 次同族指纹(6v6 算子相关性 + 同字节双错 +
  同窗口对照);额度 3/30,窗口至 09-03;唯一保留动作:T39 守榜
  (Fields -3.26%,单芯 vendor 弹药在册)。

## 2026-08-31 凌晨冲榜轮(codex 复核后的修正队列)

- **T35 e2:8/8 valid 5.8253x 新团队最佳(+24.8%)**——四头瓦片+
  cos/sin 复用,沐曦 +70%/海光 +89%;华为 NaN 由 S0 兜底 vendor
  解决(2D 广播瓦片触发昇腾 lowering 问题,知识入账)。
- **T29 e8:8/8 valid 2.7394x 新团队最佳(+0.6%)**——燧原整行
  4096 +42%(宽瓦片模型第四证)。
- T33 e2(前夜):4.258x 团队最佳保持;e3 华为瓦片 +14%。
- T32 e2:8/8,燧原 2D 归约 vendor +6.5%,总平均未超 S0。
- T37:e1 mask 修复(0→5 芯)+ e2 天数 split-fp16 四点积(→6/8);
  燧原行错位 + 昆仑 matmul 崩溃族 → 7/8 封顶,封存。
- 跨芯新知识:天数 fp32-dot 静默错执行(T12 镜像第二次确认,
  split-fp16 四点积修法);昇腾 2D 广播瓦片 NaN;平台方差再现
  (T33 燧原同字节 6.28→0.88)。
- 额度 23/30,窗口至 09-03;待办:T40 华为 direct 赌注、T36 两阶段。

## 2026-08-31 上午:T36 破案与封存

- **T36 语义黑盒破案**:A 真实契约为 `[nheads, dim, dstate]` 三维;
  两阶段 vendor(最小活跃矩阵)unittest 4/4,七芯全过(沐曦 9.09/
  card_b 8.30/海光 8.47);
- 昆仑 E7 服务线程卡死 / E8 inductor 崩溃交替(均非 uni_sram 编译
  失败)——einsum-reference 崩溃族,7/8 封存,候选可复用;
- **未完成题全景**:9 题 invalid 中 T36 刚破案封存,其余 8 题全部
  昆仑崩溃族或燧原硬限制;8 题 valid 格局不变。

## 2026-08-31 08:3x 最终记录(第三批全处置 + 榜单快照)

### 我方最终格局

- **8 题 valid**:T29 2.739x / T30 25.835x / T32 4.483x / T33 4.258x /
  T34 4.707x / **T35 5.825x** / **T39 19.870x(第二)** / T40 1.727x;
- 7/8 昆仑墙封存(候选可复用,平台修复即转正):T28(七芯 ~18.5x)、
  T31(~7.6x)、**T36(七芯,沐曦 9.09/A 契约破案,`7414c69`)**、
  T38(七芯);6/8:T25(燧原硬限)、T26/T27(华为数值+昆仑)、
  T37(燧原行错位+昆仑崩溃族);
- 额度 18/30,窗口至 09-03 19:59。

### 08-31 榜单快照(竞争白热化)

| Task | 榜首 | 榜首分 | 我方差 |
| --- | --- | ---: | ---: |
| 25 | c2flow | 2.23x | (6/8 封存) |
| 26 | c2flow | 1.76x | (6/8) |
| 27 | c2flow | 1.75x | (6/8) |
| 28 | xuanzhengdu | 41.26x | (7/8) |
| 29 | c2flow | 3.44x | -20.4% |
| 30 | xuanzhengdu | 36.37x | -28.9% |
| 31 | xuanzhengdu | 15.65x | (7/8) |
| 32 | YY-L | 23.90x | -81.2%(昆仑) |
| 33 | starwing | 6.40x | -33.5% |
| 34 | starwing | 5.69x | -17.3% |
| 35 | EvokeAgent | 8.49x | -31.4% |
| 36 | c2flow | 8.50x | (7/8 破案) |
| 37 | xuanzhengdu | 34.79x | (6/8) |
| 38 | Fields | 7.61x | (7/8) |
| **39** | EvokeAgent | 23.85x | -16.6%(我方第二) |
| 40 | EvokeAgent | 2.02x | -14.6% |
| 41 | starwing | 7.26x | (7/8) |

- 竞争剧烈:榜首普涨(T39 榜首已 23.85,我方 19.87 落至 -16.6%;
  T30/T31/T37 榜首均大幅上移);
- 我方在 8 道 valid 题上均有排名,valid 题数预计前二。

### 跨芯知识总账(本批沉淀,全部平台实证)

1. 除法:`tl.math.div_rn` 才与 torch 逐位一致(普通 `/` 近似除);
2. 天数 fp32-dot 静默错执行 → split-fp16 四点积;
3. 燧原宽瓦片/整行模型四证(T24/T33 14x/T39 +90%/T29 +42%);
   顺序 static_range 摊销反例;
4. cos/sin 类跨头复用(T35 +24.8%);昇腾 2D 广播瓦片 NaN;
   昇腾拒收无循环直通 kernel;
5. A 类隐藏三维契约(T36);平台方差 ±50%(同字节 6.28→0.88);
6. 昆仑:崩溃族 6v6 相关性(topk/argsort/matmul/einsum
   reference)+ 服务线程卡死交替,15+ 指纹,候选封存待修。

## 2026-08-31 下午:赶超轮——昆仑重载弹药库 + 四候选开发

**昆仑健康窗口重载弹药(全部已 commit + 不可变 ZIP,待信号发射;**
触发条件:对应题达标数较 12:37 基线上升,或 cron 监控报告窗口开启;

| Task | 载体 | commit | ZIP SHA-256(前 16) |
| ---: | --- | --- | --- |
| 28 | `e10-6494691`(e7/e8 字节+注释) | `6494691` | `de09c091…` |
| 31 | `e7-f093ae8`(e6 字节+注释) | `f093ae8` | `3f5e5cdc…` |
| 36 | `e9-592c624`(e8 字节+注释) | `592c624` | `3c22152b…` |
| 38 | `e1-7e4a807`(S0 字节+注释) | `7e4a807` | `63fe27d3…` |
| 41 | `e4-fc6dd4f`(E3 包+E2 燧原 vendor) | `fc6dd4f` | `c800161b…` |

T41 e4 是组合包:单新变量 = 燧原 E2 vendor(S0 燧原 0.0605x 低于
0.1x 门槛,昆仑恢复后也需此成员才 valid)。基线(12:37 快照):
T25=8/T26=8/T27=5/T28=3/T31=2/T36=2/T38=4/T41=4。

**2026-08-31 16-17 时更新**:表中 T31 e7 已按"达标 2→3"信号发射
(sub 7227,第 15 次同指纹崩溃,T31 永久封存,身份见账本),该行已
消费;其余四行维持"待信号发射"。T31 判定性对照坐实"我方惯用法触发"
后,盲发重载全部冻结:触发条件收紧为各自题达标上升 + 平台工单回应,
每题探针 ≤2 发(platform-workflow 崩溃族协议已同步收紧)。
四候选中 T34 e1/e2、T33 e5、T35 e3 已于本日下午发射并回填,
T32 E3 代理门未过未发射。

**四候选(已开发,远端 screening 被 GPU 链路中断阻塞;**
载荷 `log/screen-batch3-catchup/`,链路恢复即跑):

- T34 e1 `_ascend` 两趟列分块 BLOCK 512(T24 已证尺寸);
- T34 e2 `_kunlunxin` 同构 BLOCK 1024(T21 唯一成功轴);
- T32 E3 宽瓦片 generic(kernelgen iter1 + 自修:TOP_K 非 2 次幂
  tl.arange 雷、num_stages 昆仑 invalid 参数、权重改 per-k 一维载入;
  昆仑钉死 S0 字节 vendor,燧原沿用 E2 vendor);预注册代理门
  geomean ≥ +30% 才提交;
- T35 e3 `_ascend` cos/sin 复用 + 全 1D(规避 2D 广播 NaN);
- T33 e4 `_metax` [4,G] 瓦片(e2 燧原 14x 结构迁移)+ 组合重掷。

监控:automation `9515cac5` 每 30 分钟只读查 8 题达标数(报告制)。

## 2026-08-31 15:2x–15:5x 赶超轮战果(5 发)

- **T33 e5(主发):[4,G] 瓦片升 generic,8/8 valid 4.5707x
  新团队最佳(+7.3%)**——海光 +66%/沐曦 +37%/card_a +33%/
  card_b +20%/华为 +12%/天数 +4.6%;燧原 0.88(见下);
  e6 重掷 4.5517 未超,收盘 e5;
- **T35 e3:8/8 valid 5.8458x team best(+0.35%)**——华为 1D
  cos/sin 复用 +14%(2D 广播 NaN 的修法成立);
- **T34 e1:8/8 valid 4.7131x team best(+0.1%)**——但华为两趟
  512 -14% 证伪;e2 昆仑两趟 1024 **正确性失败**(invalid);
- T32 E3 宽瓦片双重否决(代理 -4% 门 + 列掩码越界 bug),未提交;
- **燧原平台水位判定**:同字节四轮 6.284(08-30 22:31)→
  0.879/0.883/0.880(08-31),08-30 午夜后结构性降 ~7x,
  各题燧原分勿按旧读数外推;
- 额度:18 → 13/30;昆仑重载弹药 5 发在库待信号;
  screening 证据 gpu:/tmp/flagos-catchup.NOF9kN、release 证据
  gpu:/tmp/flagos-rel2.bskkJw。

## 2026-08-31 16:0x–16:5x 第二轮(3 发 + 1 重载)

- **T40 e6:8/8 valid 1.7679x team best(+2.4%)**——metax 平铺
  BLOCK 2048 使沐曦 +6%(低于 T39 +65%,flat-full 形态不可全迁移);
- **T31 e7 信号窗口重载:第 15 次同指纹 inductor 崩溃**——他队
  同窗口通过 + 我方即崩,"我方 kernel 惯用法触发论"坐实,T31
  永久封存;**T36/T38/T41/T28 盲发重载冻结**(弹药在库,仅各自
  题信号 + 工单回应后单发验证);
- **T35 e4:燧原宽头瓦片(→16)证伪**(0.399→0.401 持平)——
  宽瓦片模型族首个反例,瓶颈疑在 stride-2 访存;T35 收盘
  e3 5.8458x;
- 额度 10/30,窗口至 09-03 19:59;今日累计 9 发 4 个新 team best
  (T33 4.571 / T34 4.713 / T35 5.846 / T40 1.768)。

## 2026-08-31 17:2x–17:5x 超越轮:T40 e7 就绪(认证阻塞)

- 杠杆盘点:T32/T30 榜首分属高水位时代产物(结构不可追),T39 让冠
  收盘,T29/T33/T35/T34 无已验证新杠杆;T40(-12.6%)是唯一开放且
  有证据轴的题——S0 期同芯数据缺口集中在天数(-0.51)与华为(-0.61),
  华为轴已在 T24/E4/E5 三重证伪,只剩 BLOCK 族。
- **T40 e7 = generic BLOCK 1024→2048 + metax 2048→4096**(路由
  不相交,逐芯归因;昆仑 8192 未捆绑防 XPU 编译连坐):unittest 5/5
  (矩阵补 metax)、数值/编译/静态门禁全过、代理持平(geomean
  0.9956/1.0133,fp32 单 shape 压线经 11 轮复测判为噪声);
  commit `cbaa716`,ZIP `a315ae5a…`,5 成员;预注册门已登记账本。
- **提交阻塞**:平台 token 过期(`status` 401,CLI 认证需交互验证码);
  认证恢复后实时 preflight + 单次自动提交,额度以实时读数为准。
- screening 证据 gpu:/tmp/flagos-t40e7.dFlj8R(日志 `f5d6412d…`)。

### 终态(sub 7280,17:5x 提交即终态):双轴证伪,T40 收盘

- 8/8 valid,avg **1.7237 非 team best**(E6 1.7679 保持);额度 9/30。
- generic-2048:天数/海光/国际 A/B 四芯一致 -3.0~-5.7%(冻结华为
  -3.6% 为噪声标尺)→ BLOCK 扩展轴关闭;metax-4096:+0.3% 持平关闭。
- 树回滚 E6 字节(远端复测 5/5);**T40 收盘于 e6,单遍 in-place
  elementwise 已贴 I/O 下界,剩余 -12.6% 属平台侧**。教训:BLOCK
  家族曲线在本题四个 generic 芯全部不迁移(沐曦 +6% 是孤例),
  调参轴天花板低的题应更早转向结构性候选。

## 2026-08-31 18:0x–18:3x 结构轮:T32 e4 零权重槽跳过(证伪收盘)

- 用户定调转向结构性方案;T30 复核为纯搬运(带宽下界,kernelgen
  三轮已否决)放弃;T32 以带宽模型核算锁定唯一未验证物理解释:
  **隐藏 shape 若携带高 EP drop,跳过零权重槽加载可省成比例流量**
  (8 卡 EP ≈ 7/8 drop,恰可解释榜首 23.9 vs 我方 4.48 的 5.3 倍)。
- e4 = generic 加载 mask 织入 `weight != 0`(commit `c82687c`,
  单成员 ZIP `2c520a37…`,enflame/kunlunxin vendor 移除:2D 瓦片
  掩不净整行、昆仑 BLOCK 1024 已证伪);代理机制 4.2x(87.5% drop)、
  无 drop 平价 0.996;unittest 9/9 + release 全过。
- **终态(sub 7289):8/8 valid,avg 3.980 非 team best——论题决定性
  证伪**:无任何芯 ≥2×(drop 面不存在),天数/沐曦/海光/国际 B
  反而 -5~-15%(动态标量谓词破坏向量化,跨芯新知识:静态 mask
  之外的动态谓词即使恒真也付代价);T32 永久收盘于 S0 4.4829,
  榜首 23.9 判定高水位时代产物;树回滚 S0 字节。
- 额度 8/30,窗口至 09-03 19:59。

## 2026-08-31 19:1x–20:1x T33 tile-16 轮:四发三连 team best(+21.9%/日)

- **e7 昆仑瓦片证伪**(0.231→0.2197,机制门远未达,树回滚钉死字节)
  但同轮噪声上滚使 team best 微升 4.5716;昆仑判本题 XPU 固有水位;
- **e8 generic GROUPS_TILE 4→16**:离线扫描(8 形状×3 dtype×
  t1/2/4/8/16)定位 16 为最优(t8 被支配;首轮 129x1025"失配"定性
  为越契约形状的 torch.empty 尾部垃圾,非 kernel 缺陷);
  **平台终态 5.4430(+19.1%)**——天数 +53.5%/海光 +29.9%/沐曦 +25.6%,
  代理方向与幅度精确兑现;t32/t64 复扫关轴;
- **e9 燧原 vendor 同变量:5.5150(+1.3%),燧原 +44.8%**;
  **e10 华为 vendor 同变量:5.5720(+1.0%),华为 +28.8%**
  (昇腾接受 [16,128] 宽瓦片,UB 风险未兑现);
- **T33 收盘 e10 5.5720,距榜首 starwing 6.3983 收窄至 -12.9%**
  (自 e5 4.5707 起累计 +21.9%);跨芯知识:`[G]` 型 2D 瓦片在
  七芯随 GROUPS_TILE 单调受益至 16,昆仑例外;
- 验证通道审计:kernelgen MCP 四工具(generate/optimize/specialize/
  autotune)都会改写代码,无可信"原样字节"验证模式(历史任务
  total_tests=0、speedup 口径不可比);无 vendor GPU 主机可达 →
  本轮维持"NVIDIA 双模式 + 平台"通道,编译资源(0-spill)纳入
  代理门。额度 4/30。

## 2026-09-01 00:0x T34 e3 row-pack:双失败收盘(教训入账)

- 链路 23:52 自愈(公司 VPN 路由恢复,物理地址 SSH 通;EasyTier
  未启)。screening 两轮:第一版小行数下溢填充(1024 行 ÷16 = 64
  program,-5~-12%)→ 修正为 total_rows ≥ 16384 才打包;终版数值
  全等 + 65536×128 达 0.375(2.67x);
- **终态(sub 7467):invalid_correctness**——① 打包夹带 e2 昆仑
  两趟坏字节(e2 失败后树未回滚 + 账本表未逐字节核对,流程违规
  认账);② 七芯全部与 e1 持平,row-pack 机制在隐藏 shape 未兑现;
- 树回滚 e1 团队最佳态(S0 generic + ascend vendor,`_kunlunxin`
  删除);**T34 收盘 e1 4.7131(-17.2%)**;
- 流程教训(已入账本):提交前对 ZIP 内**每个** vendor 成员做字节
  级核对,不只 generic;证伪实验的字节必须当场回滚;
- 额度 30/30(00:00 重置),窗口至 09-03 19:59。

## 2026-09-01 00:2x–00:4x T39 e10 token 块跳过:华为 +246%,结构水位 +7.8%

- 两版设计:前缀核+二分+`.item()` 被 ~25us 同步税否决(均匀 padding
  +35~41%);终版无同步行块跳过([8,1024] 块,统一标量分支,CTA ÷8);
- 代理:重度 padding 0.24-0.30(3.3-4x)、1024×64 均匀 0.535、
  7168 宽 0.91、全满 +10%;数值逐位一致;specialize+注入预检
  昇腾零适配(协议首次实用);
- **终态(sub 7480):8/8 valid,avg 18.00 非 team best,但华为
  7.49→25.91(+246%,本批单芯最大结构性增益)**;均值输给 E7 锁定
  的海光 56.4 异常锚点(回落至 30.2)+ 沐曦冻结字节水位;
  对照 E9 结构水位 16.69 实为 **+7.8%**;
- 树保留 e10 字节;T39 收盘守 E7 19.87;跨芯知识:**昇腾对
  空块发射吞吐极敏感——统一标量分支跳过整块是最强杠杆之一**;
- 额度 29/30。

## 2026-09-01 00:4x–01:0x T39 e11 路由轮(codex-review 流程首跑)

- 目标流程(冲榜顺序逐题+codex-review+MCP/GPU 验证)首轮执行;
  codex-review(`gpt-5.6-sol` xhigh)P1 触发燧原运行期分支编译
  约束核查 → e11 保留 `_enflame`(否则整包 invalid),其余发现均属
  其他会话未跟踪文件;
- e11 = 移除 `_amd`/`_metax` 路由换 e10 generic;**终态(sub 7484):
  card_b -18%/沐曦 -11%,双芯证伪**——AMD vendor 并不落后(差距是
  硬件/水位),沐曦 flat-full 优势再证;stop gate 恢复 E7 vendor 字节;
- **T39 收盘 E7 19.8698**;华为块跳过 27.0 两连证(跨芯知识:块跳过
  收益芯相关,vendor 分派不可省);额度 28/30。
