# 第六批算子调研与作战方案（T76–T81）

> 依据：题面同步 `2026-09-17T20:50:10+08:00`；逐芯榜单快照
> `data/leaderboard-batch6-perchip-20260917.json`
> （observed_at `2026-09-17T20:51:15+08:00`，SHA-256
> `edb1a797aa63c0d84062c642e930c143463df196c5c68c4901144fc421491c34`）。
> 提交窗口 2026-09-17 20:00 → 09-24 19:59:59；评审 09-25 → 09-30。
> 今日额度 30/30 已在 batch5 收官日用尽（status 实测 `remaining=0`），
> 首批提交最早 09-18。实时数字以平台为准。
>
> **修订记录**：2026-09-17 晚经 codex-ask（gpt-6-astra，high effort）独立
> 评审后逐条核实修订——修正 T76/T78 配方与事实引用、T77 窄化机制表述与
> 探针设计（对齐 `artifacts/competition/t60-narrow-storage-audit-20260916/`）、
> T78/T79 exact 语义、T80 门槛芯与 grid 方案、T81 分裂触发与榜差数字；
> Codex 的"PR #31284 Ascend 阈值为 256"一条与本会话抓取的 PR diff 文本
> （"Ascend NPU crosses ~bs 50"）矛盾且其无网络访问，按证据弃用。

## 总判断

1. **6 题全部出自 sglang 新 `kernels/ops` 模块**（sgl-project/sglang），上游
   Triton/CUDA 参考实现与引入 PR 全部可得（下文逐题钉了 commit）。这是六批
   以来上游情报最透明的一批。
2. **3 题是 batch5 已验证资产的直系姐妹**：T78↔T62 concat_mla_k（我方 TB
   1.270875x）、T79↔T63 create_flashinfer_kv_indices（我方 TB 199.70x）、
   T81↔T53+T75（多行门控融合 + sigmoid 门控族）。s0/e1 可以直接复用已验证
   字节起步，是本批收益确定性最高的三题。
3. **两题是"reference 带 host sync"的巨分题**：T77（Python for 循环逐段
   arange，榜首 1296x）与 T80（`nonzero().tolist()` + Python 循环，榜首
   140x）。正确性是主要墙（T77 15 发仅 1 队有效），过墙即进榜。
4. T77 的**有效性大门是燧原 int64**：positions 输出 int64 撞 GCU 签名级
   i64 禁令（T60 八轮实证）。c2flow 8/8 通过证明可解；我方 T60 的
   torch-gcu 物理窄化假设是现成突破口，但需目标机验证后才能定稿。
5. 榜单逐芯核查：**六题榜首均无 >20x 同芯彩票指纹**（T77 巨分来自
   reference 本身慢，不是慢窗），靶子就是榜首逐芯读数本身。

## 优先级

| 序 | 题号 | 算子 | 榜首/均值 | 达标队 | 我方资产 | 首发预估 | 理由 |
| ---: | ---: | --- | --- | ---: | --- | ---: | --- |
| 1 | 79 | create_flashmla_kv_indices | HAiWORLD 187.47x | 3 | T63 全套（generic+4 vendor，逐芯配方齐） | ~190x | 我方 T63 banked 逐芯等效 ≈199x，直移即争第一 |
| 2 | 78 | concat_and_cast_mha_k | c2flow 1.2760x | 1 | T62 e12 字节 + cast | ~1.27x | 榜首≈我方 T62 TB；燧原 gap（0.15→0.96）与华为 persistent 是已证增项 |
| 3 | 81 | fused_gate_sigmoid_mul_add | c2flow 4.4852x | 7 | T53 多行结构 + T75 sigmoid 族配方 | ~4.0x | 7 队拥挤但榜首仅 +1.95% 于次席；华为 2.0-2.5 有结构性上行假设（方向性） |
| 4 | 80 | fixup_zero_kv | HAiWORLD 139.95x | 3 | 上游 CUDA 直接对照移植 | ~100x+ | kernel 极简、正确性风险低；比拼最小 launch 开销 |
| 5 | 77 | compute_position | c2flow 1296.06x | 1 | 上游 striped kernel（PR #31284）+ T60/T61 燧原 i64 情报 | 有效性待验 | 过正确性墙即 #2；燧原 i64 未定验前不投 |
| 6 | 76 | add3 | GuanghuLab 1.1611x | 9 | T73/T75 燧原 streaming 配方（T61 为 scatter 反例） | ~1.0x | 9 队全过、分差 0.03，纯挤水分；低成本顺手打 |

排序依据：期望名次增量 × 我方资产确定性 ÷ 风险。T76 虽排序末位但开发成本
最低，可与 T80 同日并行出 s0。

---

## T79 create_flashmla_kv_indices（第一优先）

**契约**：`reference(req_to_token, req_pool_indices, page_kernel_lens, kv_start_idx, page_size)`；
每 page 一个 entry：`slot = req_to_token[pool_i, kv_start_i + p*page_size]`，
`kv_indices[i,p] = slot // page_size`；kv_indices 2D `[bs, max_pages]` int32。
题面文字写"原地写入"但 reference 实际 `clone()` 后**返回新张量**——以
reference 为准：clone 基底、只覆写每行前 `ceil(len/page_size)` 个 page，
**行尾保留原 kv_indices 值**（这是 2D 行尾保留问题，不能直接套 T63 的
indptr 间隙恢复代码）。exact。

**上游**：`create_flashmla_kv_indices_triton`
（`kernels/ops/kvcache/kv_indices.py` @ `92d831d`）：grid `(bs, page_blocks)`，
`NUM_PAGE_PER_BLOCK = 4096 // PAGED_SIZE`，步长 page_size 的 strided gather +
`data // PAGED_SIZE`（constexpr 除法，pow2 时编译为移位）。

**我方资产（T63 终局，2026-09-17）**：
- generic：flat/分块 gather + gap 恢复写（空段间隙由 kernel 补写，见
  `src/flaggems_sglang/ops/create_flashinfer_kv_indices.py`）；
- 燧原：GCU 配方 persistent24+stages3 + BLOCK 4096 峰档（36.0→38.6）；
- 昆仑：i64 寻址算术 + 算术化 select（T58 教训族）；海光/沐曦冻结字节。
- banked 逐芯（T63 e8/e21 族）：天数 513-558 / 沐曦 70-93 / 燧原 32-39 /
  海光 267-322 / 昆仑 7.9-8.0 / 华为 69-77 / A 253-258 / B 293-308
  → 等效均值 ≈ 190-199。

**差距与靶子**（榜首 HAiWORLD 187.47）：华为 92.5 / 昆仑 17.8 / 燧原 51.3 /
天数 349.7 / 沐曦 164.2 / 海光 303.9 / A 321.7 / B 198.6。逐项对照我方
banked：燧原 38.6 vs 51.3（差 1.3x）、华为 ~73 vs 92.5（差 1.3x）、
昆仑 8 vs 17.8（差 2.2x）是我方三个真实缺口；天数/海光/A/B 我方占优。
**纯直移估算 ≈ 190x 已过榜首**；昆仑轴是最大增量项（+1.2 avg）。

**s0/e1 计划**：T63 generic 改 per-page 步长 gather（`offset*p * page_size`
寻址 + `// page_size` store，page_size 走 constexpr 特化），**输出基底改为
clone + 每行有效 page 覆写、行尾保留**；先 NVIDIA 代理 exact 回归（含非整除
page、kv_start=None 与非零、零长度行、非连续 stride、行尾非零哨兵、
max_pages 不足容量行为——reference 对容量不足会直接赋值形状不匹配报错，
不能未经证据把"静默截断"当契约），vendor 直接带 T63 四件套。预注册门：
8/8 且均值 ≥170 才算直移兑现，≥187.5 争第一。

**风险**：`slot // page_size` 若 slot 允许负值，Triton 整数 `//` 向零截断
与 Python 向下取整不同（`-1//3` 应为 -1）——token slot 域按非负假设，
回归矩阵加负值哨兵确认题面不会出负 slot，若出现需 floor 修正；
page_size 非 constexpr 且非 2 幂时昆仑向量除法有 PassManager 前科
（T58）——page_size 必须 constexpr 特化或标量化。

## T78 concat_and_cast_mha_k（第二优先）

**契约**：`reference(k, k_nope, k_rope)`；`k[t,h,:nd]=k_nope[t,h,:]`、
`k[t,h,nd:]=k_rope[t,0,:]`（单头 broadcast 到 num_heads）；store 隐式
`.to(k.dtype)`（**k 与输入 dtype 可能不同**——"低精度 cache"）；返回新张量，
exact。

**上游**：`concat_mla.py` @ `fb207b7`（CUDA JIT，PDL）；flashinfer
`concat_ops.py` 同源。与 T62 唯一差异是 store cast。

**我方资产（T62 终局）**：TB e12 1.270875x。e12 = `_ascend` persistent
（华为 0.284，+35.9%）+ `_enflame` 为 E9 语义（去 warps 钉）**加
num_stages=3**（注：纯净 launch 是 e13 形态；e12/e13 燧原 0.151/0.209 均
低于 e9 的 0.28，T62 燧原轴从未回到 e9 水位）+ hygon/kunlun 冻结；逐芯
华为 0.284 / 昆仑 0.182 / 燧原 0.151 / 沐曦 1.019 / 海光 2.911 / 天数
2.317 / A 1.664 / B 1.638。**e12 是团队最佳总分，不等于各芯最佳结构
组合**——T78 直移起步后仍按芯单变量重扫。

**差距与靶子**（榜首 c2flow 1.2760）：华为 0.245 / 昆仑 0.284 / 燧原 0.960 /
天数 2.419 / 沐曦 0.951 / 海光 1.956 / A 1.625 / B 1.767。与 T62 e12 逐芯
几乎同型，**唯燧原 0.15 vs 0.96 差 6 倍**——这是本题主战场：T63 已证燧原
gather/copy 族 BLOCK 阶梯与 persistent24 配方（T78 输出 dtype cast 改变
store 宽度，BLOCK 阶梯需重扫 512→4096）。华为 persistent 上行已证
（0.28→0.33 仍在涨）；昆仑 0.28 vs 我方 0.18 亦有小差距。

**s0/e1 计划**：T62 e12 字节 + store 前显式 cast（按**输出指针元素类型**
cast，不能把 `torch.dtype` 对象当 Triton dtype 用）；**先解除
`src/flaggems_sglang/ops/concat_mla_k.py:67` 的"三输入必须同为 bf16"
断言**（generic 与全部 vendor 都要解，仅改 store 不够），s0 即 ~1.25；
e1 起燧原轴单变量扫 BLOCK 阶梯（预注册门：燧原 ≥0.5、均值 >1.28 换发）。
dtype 组合未公开——本地代理矩阵至少覆盖：fp32→bf16/fp16 降精度、
bf16↔fp16 互转、NoPE/RoPE 混合 dtype、舍入中点值、溢出、NaN/Inf 传递，
平台首个回执后再收窄。

**风险与门槛**：exact 比较的对象是 reference 执行
`torch.cat(...).to(k.dtype)` 的结果——cast 按同一舍入规则匹配即可，
**不要求输入值在目标 dtype 下可精确表示**（此前"无解"表述作废）；
不要自作聪明做 round-to-nearest 之外的舍入，也不要用混合 dtype 的
`tl.where` 合并分支（Triton fp16/bf16 提升规则可能引入额外降精度）。
**门槛优先于均值**：T62 历史上华为 0.0932 跌破 0.1 一次、燧原 0.109/
昆仑 0.18 长期贴线——先保证这三芯余量（≥0.2），再追燧原 0.96 的结构差。

## T81 fused_gate_sigmoid_mul_add（第三优先）

**契约**：`reference(hidden_states, gate_weight, shared_output, final_hidden_states)`；
`gate = Σ(hidden*gate_weight, -1)` fp32；`out = final + sigmoid(gate)[:,None]
* shared`（fp32 中乘加）→ cast 回 final dtype。`[num_tokens, hidden]`，
`gate_weight [hidden]`。标准 bf16/fp16 容差。

**上游**：`_fused_append_shared_experts_with_weights_kernel`
（`kernels/ops/moe/fused_moe_triton_kernels.py` @ `d4ad368`）FUSE_GATE 路径：
`logit = tl.sum(h*w)` 后 `tl.sigmoid(logit)*scale`；调用方
`qwen2_moe.py shared_expert_gate`（Linear(hidden,1) 无 bias = 行点积）。

**我方资产**：T53 fused_gdn_gating **多行复用结构**（一 program 多行摊销
gate_weight 载入；天数 +96%、海光 +47% 的已证增益形态）；T75 sigmoid_gate_mul
的燧原 2.8x 配方、i32/i64 双 kernel 结构、华为 persistent vendor（+15.6%）。

**差距与靶子**（榜首 c2flow 4.4852）：华为 2.04 / 昆仑 0.99 / 燧原 3.89 /
天数 6.53 / 沐曦 4.57 / 海光 7.44 / A 5.60 / B 4.83。次席 HAiWORLD 4.399
（榜首仅 **+1.95%**；华为 2.454 最高、燧原 1.42 最低）。7 队挤在 2.86-4.49。
**华为上行假设（仅方向，非可得增量）**：T75 sigmoid 族他队华为 16-20x、
我方 1.48x——单遍全融合 + persistent 在 Ascend 的结构空间本榜尚未见
exploitation（榜首仅 2.0-2.5）；**T75 与本题 reference 形态不同，该数字
不可外推为保证**，只能作为华为轴值得优先探的结构方向。燧原轴
（3.89 榜首 vs 次优 2.20）亦有 1.8x 缺口。

**s0/e1 计划**：单 kernel 两阶段（阶段 1 行点积 fp32 累加——hidden 只读
一次；阶段 2 load final+shared FMA store），**首版一 program 一行**、全
fp32 中间量（乘、累加、sigmoid、FMA 的舍入位置照 reference，首版不引入
未验证的乘加收缩），s0 目标 8/8 + ≥3.5。e 轴顺序：华为 persistent
vendor → 燧原（去钉 launch / BLOCK）→ 多行宽度（T53 多行是逐元素布局的
增益证据，**不能直接证明本题归约形态多行正确/高效**；且复盘记录过二维
`tl.sum/cumsum` 的后端错误，多行版要带独立对照）。**双 kernel 分裂的触发
条件修正**：单 kernel 方案 hidden 只读一次、阶段 2 不再读 hidden，不存在
"重读瓶颈"；真正该触发分裂的证据是寄存器/UB 压力、spill、长行串行化、
或 token 数太少导致输出阶段并行度不足——比较时计入额外 launch 与中间
gate 张量成本。

**风险**：容差内仍要控制抵消误差——测试矩阵覆盖严重抵消（正负相消的
行点积）、gate 近零、sigmoid 饱和（|gate|>20）、输出抵消；sigmoid 用
`tl.sigmoid`（T51 已证 Ascend 可用等价形态，若数值差异大再换）。

## T80 fixup_zero_kv（第四优先）

**契约**：`reference(out, lse, kv_lens, cum_seq_lens, max_seq_len)`；
对 `kv_lens[i]==0` 的请求：`out[cum[i]:cum[i+1]]=0`、`lse[...]=-inf`；
返回克隆。out `[total_tokens, H, Vd]` bf16/fp16、lse fp32。精确，lse 用
`equal_nan=True`。

**上游**：`fixup_zero_kv.cuh` @ `74338e9`（PR #37525）：2D grid
`(max_seq_len, batch)`，y 早退 `kv_lens>0`，float4 向量清零。**reference
带 `nonzero().tolist()` host sync** → 巨分（105-140x）主要量的是 launch/
sync 开销。

**靶子**（榜首 HAiWORLD 139.95）：华为 51.6 / 昆仑 3.24 / 燧原 6.02 /
天数 218 / 沐曦 195 / 海光 248 / A 265 / B 133。三队分差 105→140 由弱芯
拉开（华为 1.9→51.6 = 27 倍、昆仑 1.1→3.2）。**华为又是最大杠杆**：零填充
是 streaming store 族，T62/T64 已证 Ascend persistent 对 copy 族 +35-44%。
**门槛风险芯是燧原不是昆仑**：cgzhou 燧原 0.1924 是全榜距 0.1 最近的读数
（昆仑最低 0.2456）——tiny-kernel launch 开销主导，wrapper 必须极薄
（clone×2 + 单 kernel = 3 次 launch，clone 本身也是成本）。

**s0 计划**：wrapper `out.clone()` + `lse.clone()` + 单 kernel：每程序早退
（`kv_lens>0`）+ BLOCK_V 向量清零 + lse -inf（**lse 的索引与 store mask
按 [total_tokens, H] 独立处理，不随 V lane 重复写**；未触碰段的 NaN/原值
必须保留）。**grid 方案**：燧原 grid.y≤255 且 grid.x 也有上限——2D
`(bs, token_blocks)` 若 `ceil(max_seq_len/BLOCK_T)>255` 仍越界，首版直接
**一维展开**（`grid=(bs * token_blocks_max,)` kernel 内拆下标）或 y 封顶 +
kernel 内 token 块循环。**`max_seq_len` 只是分派提示**：reference 不使用它，
段界以 `cum_seq_lens` 为准（kernel 内按 cum 循环上界），不信任 host 侧
max_seq_len 覆盖所有 token。预注册门：8/8 且 ≥100；华为轴 persistent
vendor 为 e1。**后续融合机会**：`empty_like` 替代 clone、kernel 内对
零 KV 段写常量、非零段从原张量拷贝——单 launch 完成 clone+修补（去掉
显式 clone ≠ 改原地写，仍返回新张量）。

**风险**：clone 是全量拷贝（与 reference 同价），不要试图跳过 clone 改
原地写——judge 对比的是返回张量，原地写会污染输入侧基准；`-inf` 写入用
`float('-inf')` 常量；`tl.where` 谨慎仅限有复现证据的后端形态，不作跨芯
通用禁令。测试矩阵：全零/全非零/混合 KV、空段、batch 与 token 块数各跨
255/256 边界。

## T77 compute_position（第五优先，先验证后投）

**契约**：`reference(extend_prefix_lens, extend_seq_lens, extend_seq_lens_sum)`；
`start = exclusive_cumsum(extend_seq_lens)`（int32 `[bs]`）；
`positions[start[i]:start[i]+s_i] = p_i + arange(s_i)`（int64 `[sum]`）。
精确整数。prefix 可为空（`has_prefix=False` 时 p=0）。

**上游**：`position.py` @ `6ed9843` 原版（grid=(bs,) 串行 O(bs²) 前缀和，
"slow for large bs"，main 无 striped）；**PR #31284（OPEN，head
`454f6bb`）striped kernel**：≤64 stripes × ROWS_PER_STRIPE（超 64 stripes
后 `rows_per_stripe = cdiv(bs, 64)` 自动扩行覆盖，不会丢行），串行链封顶
O(min(bs,64)²)。PR 注释称 H200 交叉点 bs=1024、"Ascend NPU crosses ~bs 50"
（作者环境注释，非本赛题平台实测，分派阈值需我方自调）。

**差距与靶子**（榜首 c2flow 1296.06x，唯一有效队）：昆仑 33.96 / 燧原
71.0 / 华为 354 / 天数 3268 / 沐曦 739 / 海光 1772 / A 2121 / B 2009。
16 发 8 队仅 1 队有效（20:50 索引口径；次数≠失败数，部分提交可能仍在
评测）→ **正确性墙**，而非速度墙。过墙即 #2，且 1296x
巨分使任何名次都有账面价值。

**瓶颈判定——燧原 int64**：positions 是 int64 输出。GCU 签名级 i64 禁令
（`gcu64-type-verifier` 拒绝 `!tt.ptr<i64>`，T60 八轮/T59-T61 七轮实证）
意味着 kernel 签名必须 i64-free：wrapper 以 int32 view 传 positions。
**T60 固定源码审计**（`artifacts/competition/t60-narrow-storage-audit-20260916/`，
torch-gcu `f17a922`）已证：`gcu_empty_tensor.cpp:50-64` int64 常规路径
**物理按半 itemsize 分配 4N 字节、StorageImpl 元数据仍宣称 8N**；
dtype-view 无特化注册、平台 dispatcher 未查明。因此两种物理布局假设：
- A（标准双词）：物理 8N 字节，元素 i 的词在 view 下标 2i/2i+1；
- B（narrow packed）：物理 4N 字节，元素 i 物理槽位=字节偏移 4i，即
  **view 下标 i**（view 元数据按 itemsize 比例仍报 2N 个元素——元数据
  长度与物理分配无关，写 2N 下标会越物理界）。T60 E8 的 2-词写 100%
  失配（最大差 1.13e9）与 B 相容。
未裁决前不投。**探针设计（对齐审计报告，输出向）**：
1. 先记录平台 torch/torch-gcu 版本、设备 major（S60/L600 决定 narrow
   行为）、`TORCH_GCU_ENABLE_INT64_AND_UINT64` 状态与 `aten::view.dtype`
   实际 dispatcher；
2. 物理布局判别：设备 int64 `[1,2,3,4]` → `view(int32)` 后**只读前 4 个
   int32**（两种假设下都在已分配 16 字节内）：A 应为 `[1,0,2,0]`，B 应为
   `[1,2,3,4]`；不做任何写；
3. **输出路线探针**（T77 的 int64 在输出侧，T60 的输入保真探针不覆盖）：
   新分配 int64 输出 → Triton 按 A/B 各写一版 → CPU 读回比对；
4. **输出值域探针**：用 T77 合法输入域（int32 prefix/seq）测
   `prefix≈INT32_MAX, seq_len=2` 等可跨 2^31 的输出——A 布局高词非零，
   B 布局下该值域物理不可表示，需按判别结果确认题面实际 shape 不会
   落进该区间。
其余七芯用标准 i64 签名 kernel（tl store int64 无碍）+ vendor 仅燧原。

**s0 计划**（七芯先行）：striped kernel 直移（ROWS_PER_STRIPE=16、
BLOCK_TOKENS=512、≤64 stripes；小 bs 走原版单程序路径），本地 exact 回归
覆盖 bs=1/2/1023/1024/1025/空 prefix/巨 bs（≥交叉点两侧）/零长度段/
非整除 BLOCK_TOKENS，positions 与 start 分别断言。
昆仑向量 `//` 无此题（无除法）；cumsum 串行链标量累加即可。

## T76 add3（第六优先，低成本顺手）

**契约**：`reference(a, b, c)` → `(a+b)+c`，**双重舍入是契约**
（`bf16(bf16(a+b)+c)`，与未融合对逐位一致；**禁止** fp32 单次舍入）。
bf16 连续同 shape，numel ≡ 0 (mod 16)。标准 bf16 容差。

**上游**：`add3.py` @ `197832b`（CUDA JIT + PDL + 16B 向量化 + prefetch；
PDL 是 NVIDIA 专属，**generic 不许带**）。

**靶子**（榜首 GuanghuLab 1.1611，9 队全 8/8）：华为 0.417 / 昆仑 0.730 /
燧原 1.197 / 天数 1.655 / 沐曦 1.276 / 海光 1.360 / A 1.378 / B 1.277。
全榜分差 0.91→1.16，**无结构分化，纯挤水分**。参考模型：reference 2 kernel
约 12B/elem、融合后 8B/elem——同带宽假设下约 1.5x，**是简化模型推算不是
硬上限**（天数 1.65 已高于它，说明带宽假设各芯并不成立）；华为全员 ≤0.462
另有结构性原因待查。

**瓶颈**：华为全员 <0.5（killgame 0.4623 最高，0.1444 最低——9 队有效
不等于无门槛风险）——Ascend 上融合 Triton 反而慢于两条 torch add，疑
bf16 逐元素向量化宽度/launch 差，需 `_ascend` vendor 专项（vectorcore
数量适配 + persistent copy 族配方）；昆仑 0.73-0.82、燧原 0.44-1.20
分化大。**燧原 streaming elementwise 配方的正例是 T73（+92%）/T75
（+113%）**——T61 E8 已证该配方对 scatter 无效（-7%），引用时勿错位；
add3 属 streaming elementwise，配方适用。

**s0 计划**：flat 单 kernel，`t=(a.f32+b.f32).to(bf16); out=(t.f32+c.f32)
.to(bf16)` 显式双舍入（downcast 显式 rtne，与 torch eager 每算子 fp32 计
算后按 RTNE 出 bf16 的语义对齐）；BLOCK 1024、grid-stride；**numel%16 只
保证 16 元素向量化安全，BLOCK=1024 的尾块 mask 仍必需**。本地用
`(a.bf16+b.bf16)+c.bf16` 的 torch 双算子做 exact 对照 + 容差对照双跑，
并加入**双舍入 vs 单舍入的最小反例**（`a=1, b=1/256, c=-1`：双舍入=0、
fp32 单舍入=1/256，防止实现意外融合成单舍入仍骗过宽松容差的错觉——
题面明令禁止单舍入）。e1 起华为/昆仑 vendor。预注册门：均值 >1.17
才值得继续投轴。

---

## 跨芯瓶颈与配方速查（本批投影）

| 芯 | 本批暴露题 | 已证配方（来源） | 反例/禁区 |
| --- | --- | --- | --- |
| 华为 | T76 全员<0.5、T78 0.25、T81 2.0-2.5、T80 1.9-51.6 | persistent 拷贝/散布族（T62 +35.9%、T64 +44%；NVC 封顶） | T63 宽 gather 反例（无增益）；多分支/字面量 i64 store 挂 |
| 昆仑 | T78 0.28、T80 0.25-3.2、T81 0.7-1.0、T79 8-18 | i64 仅寻址算术；整除崩溃即标量化（T58）；index 类 `.long()` 入口（T58 昆仑 vendor） | 向量 `//` runtime 标量（PassManager）；fp16 dot 数值失败 |
| 燧原 | T77 i64 签名墙、T78 0.15→0.96 主战场、T79 51.3 | streaming/gather：persistent24+stages3、BLOCK 阶梯至 4096、launch **不钉 warps**（T19/T51/E9 三证） | `!tt.ptr<i64>` 签名直接编译失败；任何 launch 钉 -50% |
| 天数/海光/A/B | 无短板 | 冻结 generic 字节即可 | 慢窗 ±25% 读数波动，判读以同窗他芯为准 |
| 沐曦 | T79 164 领先我方 93 | T63 metax 2048 曾回退，banked 92-93 即水位 | 低精度 dot 回退（勿外推） |

## 额度与日程（09-18 起 7 天 × 30 发，全局共享）

每日额度为**上限而非必须消耗的指标**：没有新假设就不必用满；已有 banked
TB 不需要例行重投"守榜"；保留额度当日有效、不跨日累计。

- **D1（09-18，预算 ≤12 发）**：T79 s0+T63 vendor 直移首发（2-3 发）；
  T78 s0（T62 e12+cast，1 发）；T81 s0（1-2 发）；T80 s0（1 发）；T76 就绪
  即发。目标当日 4-5 题 8/8 进榜，取得真实逐芯基线。
- **D2（预算 ≤18 发）**：燧原目标机 T77 探针（不耗额度，见 T77 节）→
  裁决 A/B 后 T77 s0 条件首发；修 D1 暴露的正确性/0.1 缺口；T78 燧原
  BLOCK 阶梯轴；T79 昆仑轴。
- **D3-D5（每日 ≤24 发，留 6 发当日应急）**：按本队同题
  `Δ平均分/开发成本` 排序单变量推进（华为 persistent vendors ×4 题、
  T81 多行对照、T76 vendor）。
- **D6-D7（每日 ≤18 发，留 12 发缓冲）**：有证据的最后优化与必要修复；
  D7 按排队/回调时长**提前停风险候选**（评测往返可能 >1h），最终有效
  候选尽量在 09-24 18:00 前提交完毕。
- 全程遵守：每发预注册晋级门 + 止损；崩窗族/慢窗判读按 retrospective
  协议，不连续重掷。**慢窗注意**：T77/T78 各只有 1 队在榜，缺同芯对照，
  首轮回执的逐芯读数与执行时长要与他题同窗横向对照后再下结构性结论，
  不用固定"±25%"概括所有窗口变化。

## 未决问题（开题前必答）

1. T78 k dtype 组合（cast 是否真跨精度）——平台首个回执裁决；本地先三种全测。
2. T77 燧原 int64 物理布局（假设 A/B）——目标机保真探针裁决，未验不投。
3. T79 page_size 值域与 2 幂性——题面未公开；constexpr 特化按平台回执补录。
4. T81 hidden/num_tokens 量级（多行宽度选型）——首个回执的 shape 线索。
5. 各题平台 shape 范围未公开，代理矩阵按保守超集覆盖（非整除 tile 每轴
   B-1/B/B+1）。

## 引用

- 上游文件均已钉 commit：position `6ed9843`、kv_indices `92d831d`、
  fixup `74338e9`、add3 `197832b`、moe gate `d4ad368`、concat_mla
  `fb207b7`（均为 sgl-project/sglang main）。
- 姐妹题账本：`experiments/concat_mla_k.md`、`create_flashinfer_kv_indices.md`、
  `sigmoid_gate_mul.md`、`clamp_position.md`、`compute_src2dst.md`、
  `fused_gdn_gating.md`；跨批教训 `season2-retrospective.md`。
- PR #31284（striped compute_position，OPEN，head `454f6bb`，striped 函数体
  在本会话 PR diff 中完整取得）、#37525（fixup_zero_kv JIT 预建）。
- T60 窄化审计：`artifacts/competition/t60-narrow-storage-audit-20260916/
  report.md`（torch-gcu `f17a922` 固定源码，T77 探针设计的证据基础）。

## 2026-09-18 D1 晚重规划（codex-ask 第三轮，两条事实主张已核实）

> 当前 TB：T79 173.36 / T80 200.92 / T78 1.0908 / T81 3.0704 / T76 1.0429，
> 五题 8/8 有效；T77 封存。今日 24 发已用，剩 6；窗口余 6 个提交日。

### 额度部署（窗口上限 94 发，多数应不花）

| 日期 | 新结构/正信号 | T77 | 窗口复测 | 储备 | 上限 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 09-18 余 | 2 | 0 | 1 | 3 | 6 |
| 09-19 | 8 | 4 | 0 | 4 | 16 |
| 09-20 | 8 | 4 | 2 | 4 | 18 |
| 09-21 | 10 | 0 | 2 | 6 | 18 |
| 09-22 | 6 | 0 | 2 | 4 | 12 |
| 09-23 | 2 | 0 | 4 | 6 | 12 |
| 09-24 | 0 | 0 | 4 | 8 | 12 |

值得花的四种情形：新机制首验 / 已定位根因的修复 / 目标芯正信号确认 /
满足预注册窗口门的复测。一律不花：已关轴换包装、纯代理无机制差异、
额度过夜心理、钟点/排队/旧榜首触发重掷、sending/uncertain 状态、T77 布局
未明时的任何投递。每假设首发 1 发；根因/窗口混淆才第 2 发；第 3 发必须
新根因证据。

### 剩余可检验结构轴（均需代理证据前置）

1. **T81-A 静态展开（第一优先）**：`do_not_specialize=["rows","hdim"]` 使
   hidden 循环保持 runtime（已核实源码）；5120/7168 均被 1024 整除，hdim
   特化后可全展开+消尾掩码。不改 program 数/launch/tile 宽，未被三结构
   证伪覆盖。门：受影响形状 wrapper ≥10%、控制形状回退 ≤5%、无 spill。
2. **T81-B 双累加链**：两独立 fp32 累加向量交替处理相邻块降依赖链；仅
   测 2 链不扫参数。门：沐曦或海光 ≥20% 才隔离 vendor。
3. **T78-C 一维输出 tile**：按输出连续地址解码 token/head/col、双路掩码
   加载、单 store（替代 BH=16 二维 broadcast store）；首发只动 hygon/metax
   vendor。门：海光 ≥1.82 或沐曦 ≥1.21。
4. **T79-D affine 访存表达（条件立项）**：需 Ascend IR 或目标执行证据
   证明旧路径标量化——**无目标编译证据预算为零**。

### T77 解锁（半天工作量上限，09-20 晚无渠道即暂停）

路径排序：赛事同栈一次性探针 > KernelGen 类真实通道（绑定源码+测试数）>
短租匹配硬件（先核驱动/容器）> 版本不匹配机（仅方向性）。探针顺序：
环境记录 → [1,2,3,4] 前 4 词只读判别 → **新分配输出**写入读回 → 跨 2³¹
保真 → 全矩阵+发布门。预算 4+4 包；窄化布局无法表达合法输出时必须继续
解决存储路线，不以"平台可能不测大值"放行。int32 边界测试已补入
RELEASE_REQUIRED_TESTS（本轮核实修复）。

### 收盘纪律（09-23/24）

- 窗口触发门（预注册）：同题同芯 ≥2 个可比观测 reference 耗时 +20% 且
  候选稳定 ±10%；或历史证明该状态改善总分；且估算足以越过具体名次线。
- 复测优先 T80；每窗口事件每题最多 2 发；跨午夜不清零止损。
- 新结构停点 `min(09-24 12:00, 截止−2L−1h)`；已验证候选最后提交
  `min(09-24 18:00, 截止−L−30min)`；TB 已保留不例行重交。
- 停止条件：T76 停止主动开发；T80 停常规结构搜索；T79 华为无目标证据
  即停；T81/T78 新轴各最多两次有理由验证，09-22 无正信号收口。

## 2026-09-18 D1 午后重规划（codex-ask 第四轮，三条源码主张已核实为真）

> T81 e5 静态展开 TB 3.9045 后的再平衡；T80 batch=0 边界已当场修复
> （commit 见账本）；T77 口径订正：13:10 threshold_team_count=5 为最新，
> "唯一有效 c2flow"是 09-17 旧读数。

### 静态机制迁移排序（本轮主轴）

1. **T81 燧原 vendor**（门：燧原 ≥1.80 或耗时 -15%；单发判决）
2. **T81 昆仑 vendor**（门：昆仑 ≥0.88；低成本完成不追打）
3. **T78 维度特化**（heads/groups/nd/rd 出 do_not_specialize；先 IR 确认
   除模强度削减；门：海光或沐曦可比 ≥10%；首轮只动该两 vendor）
4. T81 外层行循环单波特化（rows≤2048 消 grid-stride；与 hidden 静态化
   分开测；今日零预算）
5. T76 add3 单波直通探针（今日零预算）
- T79：**工作树 generic 是已判负静态版，后续候选必须回 e3 `975df79e`
  动态基座重建**；无目标 IR 证据零预算。
- T80：HV 已静态、token/gap 循环数据依赖；不再批量迁移。

### 判决基线与纪律增补

- T81-B 双链仍排直接迁移之后（对 e5 单链比较；门：多块形状 ≥10%×2、
  控制回退 ≤5%；平台沐曦或海光 ≥20% 才隔离 vendor）。
- T78-C 一维 tile 保留，先做维度特化对照（与新基线比，不与旧混算）。
- T77：**不以 7/8 无效换"首攻克"（无规则收益）**；探针口径改为"足以
  裁决输出存储路线的同栈证据"（赛事执行通道可替代自有主机）；09-19
  半天争取、09-20 晚无新证据暂停；可预留 1 发条件首验（需完整值域+
  无已知失败的候选）。
- T80 B 轴 164 vs 532：待归因（分数差无法区分 reference 变慢/候选变快）；
  复测必须用含 gap 修复的 e5 后继版本，不得回退 e4 字节。
- 额度对账：今日 8 + 6×30 = **188** 发窗口余量（前述 168 有误）。
  今日默认花 2-4 发；候选不成熟花 0 发也正确。

### 收官曲线（上限）

| 日期 | 实验/复测 | 修复余量 | 重点 |
| --- | ---: | ---: | --- |
| 09-19/20 | 各 16 | 各 6 | 静态迁移、B/C 筛选、T77 证据 |
| 09-21 | 12 | 6 | 正信号兑现与组合 |
| 09-22 | 8 | 6 | 无正信号轴关闭 |
| 09-23 | 4 | 8 | 修复+条件窗口复测 |
| 09-24 | 2 | 8 | 已验证候选收尾 |
