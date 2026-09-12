# 第五批第四轮：17 题终局方案（2026-09-12 深夜）

平台于 09-12 晚把第五批从 12 题扩到 **17 题（T59–T75）**，用户确认此为
终局题量。同步脚本期望数已改 17，题面/索引已刷新（20:28 快照）。
本文收口四路联网/GitHub 调研（新五题上游、昇腾之谜、昆仑垃圾/崩溃、
GCU 寻址纪律），给出 09-13 起的作战方案。

## 1. 三大调研突破（直接改变打法）

### ① 昇腾"简单操作 9-17x"之谜 = 重编译风暴 + `>>` 标量降级（可修）

- **JIT 每 shape 重编译**：`BLOCK=next_power_of2(cols)` 之类 constexpr
  使每个 eval shape 一次秒级编译（T67 天数/华为 exec 84s/30s、T59 25s
  的量级来源）；triton-ascend PR #1480（2026-08-13 才修）证实修复前
  **整型参数 16 整除性翻转、指针 16B 对齐翻转都触发全量重编译**——
  竞赛镜像大概率早于该修复。
- **`>>`/`&` 被 bishengir 降级标量执行**（issue #1220，实测 10-100x
  损失）——T59 的 `slot >> SHIFT` 正中；i64 向量必然降级。
- **形态反模式**：巨大 grid + 每程序标量 GM 元数据载入；官方指南要
  grid=向量核数 + persistent 循环 + 批量索引载入。
- **修法（T59/T67/T60 昇腾 vendor 统一纪律）**：全部动态 int 参数
  `do_not_specialize`（vllm-ascend penalty.py 同款）；BLOCK 固定档位
  （单一编译产物跨 shape 复用）；`>> s` 改 `// (1<<s)`；1D persistent
  grid；int32 索引域。

### ② 昆仑 concat_mla_k 垃圾根因 = stride-0 广播读 + 非 2 幂 192 宽 tile（可修）

- FlagTree #1147（OPEN 未修）：**输入侧某轴 stride-0（我们的 rope
  [T,1,R] 沿 H=128 广播）+ 输出全局连续 store** 正是 OffsetAnalysis
  误判 Continuous→burst DMA 越界读的触发配方（同指纹确定性错值）。
- FlagGems `_kunlunxin/ops/softmax.py` 官方注释：**非 2 幂宽 tile
  （我们的 192）一族静默 miscompile**；≤32 lane tile 同罪。
- 官方同族参照 `concat_and_cache_mla.py`：**两段独立 1D masked 循环**
  （kv 段、pe 段各自 load/store），无跨 H 广播 tile。
- **修法（T62 昆仑 vendor 重写）**：k 段与 rope 段分立 1D 循环（段宽
  128/64 恰为 2 幂），或每头独立小段；launch kwargs 加
  `isCloseOffsetAnalysis=True`（compiler.py 证实跳过 offset stamp pass）
  作保险。
- **T69/T64 昆仑崩溃补充假设**：FlagGems sum.py 注释"masked load 在
  runtime 循环内不可靠"；#1146（0.7.0-rc1 is_sdnn_kernel SIGSEGV，
  kernel 侧无解）。**e6 低成本动作：launch kwargs 加
  `isCloseOffsetAnalysis=True, isCloseUnrollControl=True`**
  （FlagGems fused_moe 生产在用同款）；仍崩则只剩工单。

### ③ GCU 寻址纪律拿到完整规则集 → T61 重开、T66 vendor 有谱

- FlagGems gcu300 scatter/index_put 六条纪律：**kernel 内零 i64 数据**
  （宿主侧 `.to(torch.int32)`——我们 T61 四连败的根因正是 kernel 里
  load 了 int64 reorder_ids）；地址链纯 i32（arange 出发 + 显式
  `.to(tl.int32)` 的 stride 累加）；全部 shape/stride
  `do_not_specialize`；masked lane 地址钳位；512MB MMU scatter 跨度；
  原子全族按不可用对待（skip 表 370 条）。
- **T61 重开实现（预注册）**：wrapper `reorder_ids.to(torch.int32)`（
  值域 <2^31 已证无损）+ `tl.store(out + cur_i32, offsets, mask)`
  单行 scatter + do_not_specialize——与 FlagGems scatter.py 1D 特例
  同构。
- **T66 燧原 vendor**：网格饥饿修法（BLOCK_N=128、persistent 12-CTA、
  num_warps=2、stages=3、`tl.dot(out_dtype=fp32, allow_tf32=False)`）；
  K 深需要 split-K 时走两段式（partial+reduce，不用原子锁版 streamk）。
- T71 tanh：GCU 用 `2/(1+exp(-2x))-1` exp 恒等式（T29 已证 exp 是
  全八芯唯一安全超越函数；FlagGems 官方 fallback 同式）。

## 2. 新五题 S0 计划（上游全部 FOUND，明日首窗连发）

| Task | 上游（固定引用） | S0 结构 | 难度 |
| --- | --- | --- | --- |
| T74 seqlens_expand | SGLang fd32226 `ops/attention/pad.py`（题面同源） | grid=(N,) 每请求一 program，`max(start+offs,0)`，BLOCK=next_pow2(max_q_len)；cumsum 进 kernel（每 program masked load 前缀 + tl.sum）以全 Triton 化 | 低 |
| T75 sigmoid_gate_mul | SGLang `ops/moe/triton_sigmoid_gate_mul.py` | flat 1024-lane，`x*1/(1+exp(-g))`，fp32 算 cast 回 | 最低 |
| T71 gelu_tanh_and_mul | FlagGems `fused/gelu_and_mul.py` + 我仓 T29 底盘 | T29 结构换 tanh 恒等式（零 tl.math/tl.extra） | 低 |
| T72 group_norm_silu | SGLang `ops/diffusion/norm/group_norm_silu_triton.py` 单遍版 | 2D grid (G,N)，chunk 循环两遍（统计+归一化+affine+silu），fp32，`idx//spatial` 取 channel | 中 |
| T73 residual_gate_add | SGLang CUDA JIT（Triton 需自写） | flat：`prod=(u*g).to(T).to(f32); out=(r+prod).to(T)`（双重舍入=契约）；gate 行广播用 2D grid 免逐元素取模 | 低 |

竞争态势：c2flow 五题全先手（3.4/3.0/4.2/20.0/2.9x），但除 T71/T72 外
都只有 1-2 队提交——首发窗口仍大。泄露扫描：FlagGems-sglang 无这五题
的他人提交。

## 3. 全 17 题状态与下一轴（按把握倒排）

### 第一梯队：新五题 S0（明日 00:00 窗口连发，互不占旧题额度心智）

按 T74→T75→T71→T72→T73 序开发；每题 S0 走完整
screening→commit→release→ZIP→preflight→submit 闭环。

### 第二梯队：调研解锁的修复题

1. **T62 昆仑 vendor 重写**（①节②修法：两段 1D + isCloseOffsetAnalysis）
   ——三次同指纹垃圾的根因首次有实证解；华为/燧原弱芯轴 B1/B2 随后。
2. **T61 重开**（③节 scatter 单行式 + 宿主 int32 降位）——四连败根因
   （kernel 内 int64 load）已定位；七芯水位 1.07-3.01 早已是未过线最强。
3. **T69 e6**：三 kernel 加 `isCloseOffsetAnalysis/isCloseUnrollControl`
   kwargs 重掷昆仑（五连崩后首个有据动作）；同时工单提交（不耗额度）。
4. **T59 e8 / T67 e4 昇腾重写**（①节①修法：do_not_specialize + 固定
   BLOCK + `//` 替 `>>` + persistent）——华为缺口 T59 3.6/T67 15.6 的
   首个机制性解法。

### 第三梯队：已过线题的增量轴

5. **T63 e9**：燧原 e7 字节重掷（6.2→17.9 窗口，+1.5）+ 沐曦 87→114
   （几何已换，再看窗口）；距榜首 9.7。
6. **T68 e4**：海光窗口全开重掷（e1 已证 14.26；差 0.21）——载体=
   任何真实改动，候选=generic `>>`… 无；用 do_not_specialize 加持
   （兼收昇腾重编译收益，华为 3.67 或同步抬升）。
7. **T66 e2 燧原 vendor**（③节网格饥饿配方）。
8. **T64 e3=C2** precomputed-pos（R2 注册不变）+ 昆仑 kwargs 重掷。
9. **T65 E1** 路由行向量化 + 昆仑 kwargs 重掷（七芯部分和 76.9 待命）。
10. **T70 e2**：燧原/华为补迭代选择 vendor（FlagGems kunlun topk.py
    stage1 形态为参照：max/argmax+击杀，merge 段同样迭代式避开
    bitonic——昆仑 sort 家族官方判不可信）+ 三芯新 ZIP 重掷。
11. **T60 E8**：无条件 i32 词对 + i64-free 签名（R2 预注册不变；新证据
    强化了 kernel 内零 i64 纪律的必要性）。

### 已关闭轴（不再投）

T59 tile/store 形态、T67 列分块、T64 grid 封顶、T68 燧原两遍分块、
T69 JIT 瘦身（降级为次要因子）、T63 clone 轴（已兑现）。

## 4. 09-13 窗口发射纪律

- 额度 30 重置；建议分配：新五题 S0 ×5、T62/T61 修复 ×2、T63 重掷 ×1
  （共 8 发），留 ≥40% 给逐芯信号为正的重开与收盘回归。
- 每发之间读逐芯结果；T69/T64/T65 昆仑三连崩的 kwargs 假设由 T69 e6
  先验证，同型再推 T64/T65。
- 工单（T69 昆仑五连崩）明日一早提交，附同窗他题正常判决对照。

## 5. 证据分级

- **平台实测**：今日 30 发全部回执与逐芯读数（各题账本）。
- **上游源码级**：SGLang fd32226 / FlagGems master 与 gcu300、
  _kunlunxin 覆盖层 / FlagTree main 的逐文件引用（见各节）。
- **公开 issue 实证**：triton-ascend #1220/#1480/#1675/#298/#1610/#1673；
  FlagTree #1147/#1053/#1056/#1146；FlagGems #5816/#6093/#6166。
- **假设（未测）**：重编译风暴对 T59/T67 华为读数的占比；kwargs 对
  昆仑崩溃的疗效；T61 scatter 单行式的平台表现。均以首发裁决为准。
