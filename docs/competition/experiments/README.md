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
4.4829x(超快照榜首 +5.3%)**,全部首投 8/8 valid;T33 E1 摊销证伪、T35 E1 连续读证伪(负结果入账)。
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
