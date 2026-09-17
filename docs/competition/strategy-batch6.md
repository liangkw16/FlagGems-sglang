# 第六批算子调研与作战方案（T76–T81）

> 依据：题面同步 `2026-09-17T20:50:10+08:00`；逐芯榜单快照
> `data/leaderboard-batch6-perchip-20260917.json`
> （observed_at `2026-09-17T20:51:15+08:00`，SHA-256
> `edb1a797aa63c0d84062c642e930c143463df196c5c68c4901144fc421491c34`）。
> 提交窗口 2026-09-17 20:00 → 09-24 19:59:59；评审 09-25 → 09-30。
> 今日额度 30/30 已在 batch5 收官日用尽（status 实测 `remaining=0`），
> 首批提交最早 09-18。实时数字以平台为准。

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
| 3 | 81 | fused_gate_sigmoid_mul_add | c2flow 4.4852x | 7 | T53 多行结构 + T75 sigmoid 族配方 | ~4.0x | 7 队拥挤但榜首仅 +3.7% 于次席；华为 2.0-2.5 有结构性上行假设 |
| 4 | 80 | fixup_zero_kv | HAiWORLD 139.95x | 3 | 上游 CUDA 直接对照移植 | ~100x+ | kernel 极简、正确性风险低；比拼最小 launch 开销 |
| 5 | 77 | compute_position | c2flow 1296.06x | 1 | 上游 striped kernel（PR #31284）+ T60/T61 燧原 i64 情报 | 有效性待验 | 过正确性墙即 #2；燧原 i64 未定验前不投 |
| 6 | 76 | add3 | GuanghuLab 1.1611x | 9 | T61 燧原 streaming 配方 | ~1.0x | 9 队全过、分差 0.03，纯挤水分；低成本顺手打 |

排序依据：期望名次增量 × 我方资产确定性 ÷ 风险。T76 虽排序末位但开发成本
最低，可与 T80 同日并行出 s0。

---

## T79 create_flashmla_kv_indices（第一优先）

**契约**：`reference(req_to_token, req_pool_indices, page_kernel_lens, kv_start_idx, kv_indices, page_size)`；
每 page 一个 entry：`slot = req_to_token[pool_i, kv_start_i + p*page_size]`，
`kv_indices[i,p] = slot // page_size`；kv_indices 2D `[bs, max_pages]` int32
**原地写入**（clone 自 reference 语义）。exact。

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
寻址 + `// page_size` store，page_size 走 constexpr 特化），保留 gap 恢复写；
先 NVIDIA 代理 exact 回归（含非整除 page、kv_start 非零、max_pages 截断），
vendor 直接带 T63 四件套。预注册门：8/8 且均值 ≥170 才算直移兑现，≥187.5
争第一。

**风险**：`slot // page_size` 若 page_size 非 constexpr 且非 2 幂，昆仑向量
除法有 PassManager 前科（T58）——page_size 必须 constexpr 特化或标量化；
bs×max_pages 的 2D 原地写要求 gap 恢复逻辑与 T63 同型。

## T78 concat_and_cast_mha_k（第二优先）

**契约**：`reference(k, k_nope, k_rope)`；`k[t,h,:nd]=k_nope[t,h,:]`、
`k[t,h,nd:]=k_rope[t,0,:]`（单头 broadcast 到 num_heads）；store 隐式
`.to(k.dtype)`（**k 与输入 dtype 可能不同**——"低精度 cache"）；返回新张量，
exact。

**上游**：`concat_mla.py` @ `fb207b7`（CUDA JIT，PDL）；flashinfer
`concat_ops.py` 同源。与 T62 唯一差异是 store cast。

**我方资产（T62 终局）**：TB e12 1.270875x。e12 = `_ascend` persistent
（华为 0.284，+35.9%）+ 纯净 launch `_enflame` + hygon/kunlun 冻结；逐芯
华为 0.284 / 昆仑 0.182 / 燧原 0.151 / 沐曦 1.019 / 海光 2.911 / 天数
2.317 / A 1.664 / B 1.638。

**差距与靶子**（榜首 c2flow 1.2760）：华为 0.245 / 昆仑 0.284 / 燧原 0.960 /
天数 2.419 / 沐曦 0.951 / 海光 1.956 / A 1.625 / B 1.767。与 T62 e12 逐芯
几乎同型，**唯燧原 0.15 vs 0.96 差 6 倍**——这是本题主战场：T63 已证燧原
gather/copy 族 BLOCK 阶梯与 persistent24 配方（T78 输出 dtype cast 改变
store 宽度，BLOCK 阶梯需重扫 512→4096）。华为 persistent 上行已证
（0.28→0.33 仍在涨）；昆仑 0.28 vs 我方 0.18 亦有小差距。

**s0/e1 计划**：T62 e12 字节 + store 前显式 `.to(k.dtype)`（Triton 按指针
元素类型 cast），s0 即 ~1.25；e1 起燧原轴单变量扫 BLOCK 阶梯（预注册门：
燧原 ≥0.5、均值 >1.28 换发）。dtype 组合未公开——本地代理按
（bf16→bf16 / fp16→fp16 / fp32→bf16）三种做 exact 回归，平台首个回执后
再确认实际组合。

**风险**：exact + cast 意味着若 k.dtype 精度低于输入，题目必然保证值可精确
表示（否则无解）；不要自作聪明做 round-to-nearest 之外的舍入。

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
（华为 2.454 最高、燧原 1.42 最低）。7 队挤在 2.86-4.49。
**华为上行假设**：T75 同族题他队华为 16-20x（我方 1.48x）——单遍全融合
kernel + persistent 在 Ascend 的结构空间未被本榜 exploitation（榜首仅 2.0-
2.5）；若 T75 的他队华为结构可移植，+8-16x 单芯 = +1-2 avg。燧原轴
（3.89 榜首 vs 次优 2.20）亦有 1.8x 缺口。

**s0/e1 计划**：单 kernel 双趟（趟1 行点积累加 fp32，趟2 load final+shared
FMA store），B_ROWS 多行/program（T53 形态）；s0 目标 8/8 + ≥3.5。e 轴
顺序：华为 persistent vendor → 燧原（去钉 launch / BLOCK）→ 多行宽度。
**两趟 vs 双 kernel 分裂**：hidden 行 10KB 级 L2 可容，首版两趟单 kernel；
若代理显示第二趟 hidden 重读是瓶颈，再试 dot-kernel + fma-kernel 分裂。

**风险**：fp32 累加顺序影响容差内逐位（sum 顺序与 reference 不同即可，
有容差）；sigmoid 用 `tl.sigmoid`（T51 已证 Ascend 可用 `1/(1+exp(-x))`
等价形态，若数值差异大再换）。

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
昆仑 0.1 门槛风险真实存在（hbmu 0.2456 贴地）——tiny-kernel launch 开销
主导，wrapper 必须极薄（clone + 单 launch，无任何额外 torch op）。

**s0 计划**：wrapper `out.clone()` + `lse.clone()` + 单 kernel：grid
`(bs, cdiv(max_seq_len, BLOCK_T))` 每程序早退 + BLOCK_V 向量清零 + lse
-inf。**燧原 grid.y≤255：bs 放 grid.x、token 块放 grid.y**（或折叠）。
预注册门：8/8 且 ≥100；华为轴 persistent vendor 为 e1。

**风险**：clone 是全量拷贝（与 reference 同价），不要试图跳过 clone 改
原地写——judge 对比的是返回张量，原地写会污染输入侧基准；`-inf` 写入用
`float('-inf')` 常量，勿用 `tl.where` 分支形态。

## T77 compute_position（第五优先，先验证后投）

**契约**：`reference(extend_prefix_lens, extend_seq_lens, extend_seq_lens_sum)`；
`start = exclusive_cumsum(extend_seq_lens)`（int32 `[bs]`）；
`positions[start[i]:start[i]+s_i] = p_i + arange(s_i)`（int64 `[sum]`）。
精确整数。prefix 可为空（`has_prefix=False` 时 p=0）。

**上游**：`position.py` @ `6ed9843` 原版（grid=(bs,) 串行 O(bs²) 前缀和，
"slow for large bs"）；**PR #31284 striped kernel**：≤64 stripes ×
ROWS_PER_STRIPE，串行链封顶 O(min(bs,64)²)，H200 交叉点 bs=1024、
**Ascend 交叉点 ~bs=50**（few-core，对本批八芯正合适）。

**差距与靶子**（榜首 c2flow 1296.06x，唯一有效队）：昆仑 33.96 / 燧原
71.0 / 华为 354 / 天数 3268 / 沐曦 739 / 海光 1772 / A 2121 / B 2009。
15 发 8 队仅 1 队有效 → **正确性墙**，而非速度墙。过墙即 #2，且 1296x
巨分使任何名次都有账面价值。

**瓶颈判定——燧原 int64**：positions 是 int64 输出。GCU 签名级 i64 禁令
（`gcu64-type-verifier` 拒绝 `!tt.ptr<i64>`，T60 八轮/T59-T61 七轮实证）
意味着 kernel 签名必须 i64-free：wrapper 以 int32 view 传 positions。
**两种物理论假设**：
- A（标准）：逻辑 int64 = 2×int32 词，store (val, 0) 对；
- B（torch-gcu 物理窄化，T60 新证据：`gcu_hardware.h` Long→Int 映射 +
  半 itemsize 偏移 + 按窄 dtype 分配）：物理上每元素仅 4 字节，int32 view
  的 numel 是 N（逻辑）而非 2N，按 **同 numel 单词写** 才落位——T60 E8 的
  2-词写 100% 失配（最大差 1.13e9）恰与假设 B 相容。
未定验前不投。**首验路径**：燧原目标机（或 KernelGen）跑 int64 保真探针
（CPU→GCU→CPU 全域含 stride/offset），一次性裁决 A/B；裁决后 wrapper 定稿。
其余七芯用标准 i64 签名 kernel（tl store int64 无碍）+ vendor 仅燧原。

**s0 计划**（七芯先行）：striped kernel 直移（ROWS_PER_STRIPE=16、
BLOCK_TOKENS=512、≤64 stripes；小 bs 走原版单程序路径），本地 exact 回归
覆盖 bs=1/2/空 prefix/巨 bs（≥1024 交叉点两侧）/非整除 BLOCK_TOKENS。
昆仑向量 `//` 无此题（无除法）；cumsum 串行链标量累加即可。

## T76 add3（第六优先，低成本顺手）

**契约**：`reference(a, b, c)` → `(a+b)+c`，**双重舍入是契约**
（`bf16(bf16(a+b)+c)`，与未融合对逐位一致；**禁止** fp32 单次舍入）。
bf16 连续同 shape，numel ≡ 0 (mod 16)。标准 bf16 容差。

**上游**：`add3.py` @ `197832b`（CUDA JIT + PDL + 16B 向量化 + prefetch；
PDL 是 NVIDIA 专属，**generic 不许带**）。

**靶子**（榜首 GuanghuLab 1.1611，9 队全 8/8）：华为 0.417 / 昆仑 0.730 /
燧原 1.197 / 天数 1.655 / 沐曦 1.276 / 海光 1.360 / A 1.378 / B 1.277。
全榜分差 0.91→1.16，**无结构分化，纯挤水分**。理论：reference 2 kernel
12B/elem，融合后 8B/elem，上限 ~1.5x；天数 1.65 已贴顶。

**瓶颈**：华为全员 <0.5（killgame 0.462 最高）——Ascend 上融合 Triton 反而
慢于两条 torch add，疑 bf16 逐元素向量化宽度/launch 差，需 `_ascend`
vendor 专项（vectorcore 数量适配 + persistent copy 族配方）；
昆仑 0.73-0.82、燧原 0.44-1.20 分化大。**我方 T61 燧原 streaming 配方
（grid 封顶 24 + BLOCK 4096）直移**。

**s0 计划**：flat 单 kernel，`t=(a.f32+b.f32).bf16; out=(t.f32+c.f32).bf16`
显式双舍入；BLOCK 1024、grid-stride；numel%16 保证向量化安全。本地用
`(a.bf16+b.bf16)+c.bf16` 的 torch 双算子做 exact 对照 + 容差对照双跑。
e1 起华为/昆仑 vendor。预注册门：均值 >1.17 才值得继续投轴。

---

## 跨芯瓶颈与配方速查（本批投影）

| 芯 | 本批暴露题 | 已证配方（来源） | 反例/禁区 |
| --- | --- | --- | --- |
| 华为 | T76 全员<0.5、T78 0.25、T81 2.0-2.5、T80 1.9-51.6 | persistent 拷贝/散布族（T62 +35.9%、T64 +44%；NVC 封顶） | T63 宽 gather 反例（无增益）；多分支/字面量 i64 store 挂 |
| 昆仑 | T78 0.28、T80 0.25-3.2、T81 0.7-1.0、T79 8-18 | i64 仅寻址算术；整除崩溃即标量化（T58）；index 类 `.long()` 入口（T58 昆仑 vendor） | 向量 `//` runtime 标量（PassManager）；fp16 dot 数值失败 |
| 燧原 | T77 i64 签名墙、T78 0.15→0.96 主战场、T79 51.3 | streaming/gather：persistent24+stages3、BLOCK 阶梯至 4096、launch **不钉 warps**（T19/T51/E9 三证） | `!tt.ptr<i64>` 签名直接编译失败；任何 launch 钉 -50% |
| 天数/海光/A/B | 无短板 | 冻结 generic 字节即可 | 慢窗 ±25% 读数波动，判读以同窗他芯为准 |
| 沐曦 | T79 164 领先我方 93 | T63 metax 2048 曾回退，banked 92-93 即水位 | 低精度 dot 回退（勿外推） |

## 额度与日程（09-18 起 7 天 × 30 发）

- **D1（09-18）**：T79 s0+T63 vendor 直移首发（预计 2-3 发）；T78 s0
  （T62 e12+cast，1 发）；T81 s0（1-2 发）；T80 s0（1 发）。目标当日 4 题
  8/8 进榜。
- **D2**：燧原目标机 T77 int64 保真探针（不耗额度）→ 裁决 A/B 后 T77 s0；
  T78 燧原 BLOCK 阶梯轴；T79 昆仑轴。
- **D3-D5**：单变量轴逐题推进（华为 persistent vendors ×4 题、T81 多行/
  双 kernel 对比、T76 s0 顺手）。
- **D6-D7（09-23/24）**：冲榜 + 收盘回归，各题保留 ≥2 发防御额度；
  09-24 19:59 硬截止前完成最后一轮 banked-TB 守榜。
- 全程遵守：每发预注册晋级门 + 止损；崩窗族/慢窗判读按 retrospective 协议，
  不连续重掷。

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
- PR #31284（striped compute_position）、#37525（fixup_zero_kv JIT 预建）。
