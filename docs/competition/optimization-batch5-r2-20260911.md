# 第五批第二轮优化方向（2026-09-11 收盘后）

本文是只读调研产出：给出下一额度窗口的追加方向、预注册晋级门和证据订正。
它**不产生平台结论，也不改变任何已就绪候选的状态**——候选的 commit、
ZIP SHA-256、回执 SHA-256 以各题账本 CURRENT 块为准，本文只登记**尚未开发**
的候选提案。

基线：`1d37f0f`（2026-09-11 13:13）。今日额度已打到 `daily_seq 29`，
下一窗口为 09-12。截止 2026-09-17 19:59:59。

## 1. 已就绪队列（复述，勿重复开发）

按 T59 账本 CURRENT 写明的发射序，四发已在库待发：

| 序 | Task | 候选 | source commit | 目标 |
| ---: | ---: | --- | --- | --- |
| 1 | 62 | `e4-95e5292`（去 `do_not_specialize` 对照） | `95e5292` | 裁决昆仑垃圾是否来自非特化多参数绑定 |
| 2 | 61 | `e5r-95e5292`（E5 同字节注释载体重掷） | `95e5292` | 区分 GCU 挂死 vs 评测机窗口 |
| 3 | 63 | `e3-95e5292`（`_enflame`/`_kunlunxin`/`_metax` 冻结 E1 字节） | `95e5292` | 恢复 E2 回退的窄带宽芯，预期均值 ≈134.4 |
| 4 | 59 | `e5-169dd6e`（`_ascend` vendor） | `169dd6e` | 华为 8.68→≈20 即重夺第一 |

本文以下内容是**这批之后**的追加方向，不与该队列竞争同一次发射。

## 2. 追加方向 A（最高期望）：T63 去掉 wrapper 的 `kv_indices.clone()`

### 2.1 假设

E1 平台读数对榜首 Nectar 的逐芯比值是：天数 0.559、沐曦 0.516、燧原 0.491、
海光 0.628、昆仑 0.257、华为 0.959、国际 A 0.630、国际 B 0.616——
**八芯里六芯挤在 0.49~0.63**。跨芯一致的比值是加性固定开销的特征，
不是算法差距（算法差距会随带宽/核数发散）。

wrapper 里的固定开销正是 `out = kv_indices.clone()`：clone 的流量是
"读 N + 写 N"，与紧随其后的 gather kernel 流量同量级，两者相加约为纯
kernel 的两倍，**预测比值 0.5，落在实测区间内**。旁证：`1c0381c` 为 T64
测得 "clone ceiling confirmed at 30-53%"，说明该量级在本仓是可测事实。

### 2.2 关键细节：未写区不是尾巴，是行间空隙

参考实现与测试把 `kv_indptr[i+1]` 构造成 `kv_indptr[i] + len_i + 3`，
`old` 长度为 `pointers[-1] + 5`。所以未写区 = **每两行之间 3 个元素的空隙
+ 末尾 5 个元素**，不是一段连续尾巴。因此"拷贝尾部"的写法是错的。

T63 的结构允许逐行补齐：每行的 program 已经知道自己写的是
`[indptr[r], indptr[r]+len_r)`，也知道 `indptr[r+1]`，于是同 program 再从
`kv_indices` 拷贝 `[indptr[r]+len_r, indptr[r+1])` 即可；最后一行补到
`out.numel()`，另需单独补 `[0, indptr[0])`。生产形状下空隙为 0，
补齐代价为零，而 clone 的 2N 流量被整体省掉。产出用 `torch.empty_like`
（测试断言 `actual.data_ptr() != kv_indices.data_ptr()`）。

该改法**不改变访问模式**，这是它与 T64 被证伪的 destination-gather
重写的本质区别——后者改了访存方向，在 L2/DRAM 区间输给现形态（代理 0.35x）。

### 2.3 预注册门

- 代表负载：沿用 E1 筛选的 `1×8193` 与 `32×1024`。
- 晋级门：代理 wrapper-inclusive 实测 clone 占比 **≥25%**（在任一代表负载）；
  完整正确性全过；spill/shared memory 为 0；寄存器 ≤64。
- 预期：占比 30~50% 时均值 132.099→**185~260**；占比 <10% 时放弃该轴。
- 止损：若实测占比 <25%，直接关闭 clone 轴（不投平台）。
- 证据等级：**假设**。先做代理测量，再决定是否开发。

### 2.4 同一候选要一并处理的隐患：splits 可能越燧原 `grid.y` 硬限

E2 起的 generic 用 `splits = min(cdiv(width, 256), max(1, 512 // batch), 512)`，
splits 落在 grid 轴 1；而燧原 `grid.y` 硬限 255（已记录 `OutOfResources:
grid.y, Required: 256, Hardware limit: 255`）。当 `batch ≤ 2` 且
`width ≥ 65536` 时 splits 可达 256~512，**generic 路径在这类形状上会越界**。
E2 平台 8/8 说明当次 bench 形状未触发，但该隐患对仍走 generic 的五芯持续存在；
E3 的三芯 vendor 冻结 E1 字节（splits ≤32），恰好绕开了它。

建议直接采用上游 `sgl-project/sglang#37659`（merged 2026-09-10，
"[AMD] Parallelize aiter spec-decode KV index building over token blocks"）
的公式，而不是自己放大 cap：

```text
cap  = ceil(table_width / 8192)      # 每块至少 8192 token
want = 512 // max(1, batch)
splits = max(1, min(cap, want))      # 256K 上下文 cap 仅 32，天然不越 255
```

并额外加一条显式 `min(..., 255)` 守卫。另外 #37659 的同一 request 内多
program 是**跨步瓜分** copy 循环、开头 `if blk >= num_loop: return` 提前退出；
我们的 grid-stride 归零循环等价，可保留。参考 `#35245`（stride 改运行时参数
防 per-shape 重编译）——我方的 `ps0/ps1/rs/ls/ips/ss/os` 已是运行时参数，
这一点不要退回去。

E2 已实测的代价也印证"过拆有害"：它把大芯片抬上去（天数 285.8→337.7、
海光 179.9→199.4、A 168.9→183.9、B 189.3→202.2）的同时压低了窄带宽芯
（燧原 21.71→**9.01，−58%**、沐曦 60.28→55.05、昆仑 2.88→2.45）。

## 3. 追加方向 B：T62 的两个弱芯轴（门槛危机，非均值问题）

T62 的当前处境是**两层同时塌**：昆仑连续垃圾/崩溃族，且燧原 0.2314、
华为 0.1594 都只在 0.1 门槛的 1.6~2.3 倍以内——**只要一次窗口退化就会
跌破门槛**（历史已出现华为 0.0932）。所以 T62 需要的不是"更高均值"，
而是"把两个弱芯抬到安全水位"。两条候选：

### B1：把 grid 封顶（一行改动）

现 launch 为 `_concat_mla_k[(min(tasks, 65535),)]`，`tasks = tokens *
cdiv(128, 16) = tokens * 8`；`tokens=8193` 时即 65544，被封在 65535。
两个弱芯都是"大 grid 吃亏"的已知受害者：skill 硬事实表记着燧原 GCU
**启动开销无法被掩盖**、硬件仅 24 SIP、推荐 GridDim 6~24，而我方多题
燧原 grid 在 1e3~1e4（超发 100~2700 倍）；华为侧 capped grid-stride 有
七次平台验证先例、T51 拿到过 +38%。循环体本来就是 grid-stride，
改 cap 数值即可，不是重构。**这条同源适用于 T64。**

### B2：生产形状去 mask（窄守卫下的 constexpr 专化）

契约写明生产 shape 为 H=128、N=128、R=64，全是指数幂。当前
`BN=next_power_of_2(nd)`、`BR=next_power_of_2(rd)` 走运行时标量，
`n < nd`、`r < rd`、`h < heads` 三个谓词在多数形状上恒真却永久留存，
谓词化的 2D store 很可能让编译器放弃向量化。做法是在**窄守卫**
（例如 `nd == 128 and rd == 64 and heads % BH == 0`）下走 constexpr 专用
变体，generic 路径保持运行时标量——重编译变体数是 2，不是随 shape 爆炸。
注意 `1c0381c` 引入 `do_not_specialize` 防的正是重编译风暴，但它顺手也让
掩码永久保留；两件事需要分开处理。

另一个需要**在同一轮里一起量**的反向风险：`1c0381c` 把 BH 从 4 提到 16，
per-program 的 program 数降到四分之一，代价是每 token 的 rope 行按组重读。
对燧原/华为这两个"弱芯加访存"的组合，方向可能是相反的，必须与 B1 分开
做单变量，不要打包成一次提交。

### 预注册门（B1、B2 各自独立）

- 代表负载：沿用 E1 筛选的 `8×4×4096` 与 `512×4×4096`（batch×heads×dims 口径见账本）。
- 晋级门：华为与燧原中位收益 **≥1.15x**，且其余六芯无任何一芯回退 >5%；
  完整正确性通过；spill/shared memory 为 0。
- 止损：若华为与燧原均 <1.05x，关闭该轴并转昆仑裁决（E4 先行）。
- 证据等级：B1 有**平台实证先例**（他题同芯），B2 为**假设**。

## 4. 追加方向 C：T64 的归因纠正（目标轴是燧原，不是昆仑）

T64 当前 CURRENT 写的是"优先改善昆仑 0.2538x 的余量"。按**均值增量**排序，
这个口径是错的。以次席 Nectar（7.61745）为真实靶子（榜首 EvokeAgent
21.649025 的 81% 来自华为单芯 100.305，而第二名同芯仅 3.0316，差 33 倍，
远超 20x 判据，是不可复现的窗口彩票）：

| 芯片 | 我方 | Nectar | 追平后均值增量 | 占 0.86965 差距 |
| --- | ---: | ---: | ---: | ---: |
| enflame | 2.5326 | 5.5018 | **+0.3711** | 42.7% |
| haiguang | 10.3318 | 12.7074 | +0.2970 | 34.2% |
| muxi | 5.5116 | 6.7576 | +0.1558 | 17.9% |
| tianshu | 16.1606 | 17.1312 | +0.1213 | 13.9% |
| kunlunxin | 0.2538 | 0.1396 | −0.0143（已反超） | — |
| huawei | 3.7614 | 3.0316 | −0.0912（已反超） | — |
| card_a | 8.4868 | 8.7262 | +0.0299 | 3.4% |
| card_b | 6.9438 | 6.9442 | +0.0001 | 0.0% |

即：把昆仑从 0.2538 提到同题第三名的 0.9048 只值 **+0.081**，而把燧原从
2.53 提到同题最优的 6.68 值 **+0.519**。**燧原是第一轴，海光第二，昆仑可忽略。**

对应到动作：`1c0381c` 新增的 `_enflame/ops/deepep_permute.py` 只是把
generic 的 clone+scatter 字节冻结成载体，**不含优化**，救不了 2.17x 的缺口。
燧原侧可用的两条已证事实是 grid 封顶（同 B1）与"避免 load 取值直接作
store 索引"（T61 E1/E3 的平台实证毒点）——现 kernel 的
`dst = tl.load(routes)` 后直接 `tl.store(out + dst * os0 + ...)` 正落在
该形态上，值得按 precomputed-pos 路线改写（wrapper 预计算目的地地址张量，
kernel 纯乘法掩码）。

clone 轴**不要再投**：上限虽为 1.43~2.13x，但 `1c0381c` 的
destination-gather 重写实测只有 0.35x，且宽行变体 0.25~0.60x，两条都已
被测量否掉；T63 的 clone 之所以是另一回事，是因为它不需要改访问模式。

- 预注册门：燧原中位收益 **≥1.3x**（封顶候选）或 **≥1.5x**（precomputed-pos 候选），
  其余七芯无回退 >5%；完整正确性通过。
- 证据等级：grid 封顶有平台先例；precomputed-pos 为**结构假设**。

## 5. 追加方向 D：T60 把 E7 的开放问题变成可证伪候选

T60 账本 CURRENT 仍停在"等待 E2 逐芯回调"，但正文已经有 E1–E7 七轮和
止损结论——**这是 CURRENT 与正文分叉，已在本次修订中同步**。

E7 的 guard 是"值全部落于 `[0, 2^31)` 时走 i32 快路径，否则回退 64 位词内核"。
账本把 E7 的失败归因为二：①平台 case 3 的值域不在 `[0, 2^31)`，guard 必然
落入 64 位路径；②wrapper 的 `view(torch.int32)` 小端重解释有错。
**这两条都意味着：i32 路线迄今从未在失败用例上真正被执行过**，
所以"i32 化失败"并未被建立。

由此得到一条零额度即可推进、且可证伪的候选：

- **E8 提案**：把 i32 词对算法做成**无条件**路径（borrow 逻辑覆盖
  `INT64_MIN` 等全域，不依赖值域 guard），并同时把 kernel 签名做成
  i64-free（见 §6）。门：先做离线签名审计与逐元素对拍（不投平台），
  两项都过才用 1 发验证。
- 若 E8 仍复现"三种内核相同垃圾"，则根因确定在内核之外（wrapper 组装或
  输出覆盖），T60 应停在内核改造、转查 wrapper 的写覆盖是否有空洞。

另需核实一处归属：账本把 `view(torch.int32)` 嫌疑记为 "XMLIR 的 view
语义"，但 T60 的失败芯是燧原，XMLIR 是昆仑运行时。这条嫌疑的归属**未经
证据支持**，重启前应先确认到底是谁的重解释行为可疑，避免按错误坐标系找根因。

## 6. 证据订正（三条，含出处）

### 6.1 燧原 i64 规则收窄为"签名级"

skill 硬事实表现行口径是"燧原 GCU 拒绝 i64 数据流（编译层）"。读 FlagTree
源码后，真实检查点要窄得多：`gcu64-type-verifier`
（`third_party/enflame/triton_gcu/triton_gcu300/lib/Transforms/GCUSupportVerifier.cpp`）
**只遍历 `triton::FuncOp` 的签名输入**，只判两件事：

1. pointee 是 64 位整数/浮点的指针类型（`!tt.ptr<i64>`）；
2. 裸的 64 位标量参数（`i64`/`f64`）。

**函数体内用 i64 做寻址算术不受该 pass 影响。**
gcu300 在 `make_ttir` 里默认就挂它（`if not options.enable_i64`），
gcu400 需 `ENABLE_I64_CHECK` 环境变量 opt-in。

对 T60/T61 的直接含义：让 kernel 签名里既无 `!tt.ptr<i64>`、也无 i64
标量参数，同一算法就能过编译；i64 张量在 wrapper 以 `view(torch.int32)` 传入、
标量一律 i32 域。对照证据：FlagGems `#5344`（closed 未合并）写明
"GCU300's `make_gcuir` PassManager rejects 64-bit dtypes at triton codegen"；
`libtriton_jit` issue `#41` 中厂商本人回复 "enflame/gcu300 not support int64
due to hw limitation. suggest to use int32 cases instead"；FlagGems `#3471`
只在 **gcu400** 上开了 11 个算子的 int64。若平台跑 gcu300，int64 是硬件限制，
int32 词视图是唯一通路——这与 T60 E1 已在纯 Python 仿真下验证过的
(lo,hi) 小端词对算法吻合，差的只是把它变成签名干净的 kernel。

### 6.2 报错文本订正

`RuntimeError: Pipeline run failed: PassManager execution failed` **不是燧原
上游的报错措辞**：按该字面串在 enflame 侧检索不到任何 issue/PR，它是 Triton
后端 `make_gcuir` 的外层包装；上游真实文本是
`64-bit data type not supported on GCU300!`（`GCUSupportVerifier.cpp` 的
`funcOp.emitError` 后 `signalPassFailure()`）。记这条订正是为了避免后续
会话按错误的关键词检索。

### 6.3 atomic 路线需要重新论证

T61 账本记的"重启候选方向 `tl.atomic_xchg`（非 DMA lowering 路径）"缺乏
支撑：FlagTree `#1019`（closed 2026-08-20）已确认昆仑 XPU **不支持
`tl.atomic_cas`**（`failed to legalize operation 'tt.atomic_cas'`），燧原的
atomic 能力无公开证据。该路线在投入前需要独立证据，不能只凭"非 DMA 路径"
的推断。

## 7. 额度纪律与发射顺序

今日额度已耗尽（`daily_seq` 到 29）。下一窗口建议按下面的顺序，且每一发
发射前必须满足两件事：预注册门写在对应账本、源码/测试/ZIP 身份一致。

1. 已就绪队列四发（62-e4 → 61-e5r → 63-e3 → 59-e5），按 T59 账本 CURRENT 的序。
2. T63 追加方向 A 的**代理测量**（零额度）：先量 clone 占比，再决定是否开发。
3. T62 B1（grid 封顶）与 B2（生产形状去 mask）各自单变量、各自一发。
4. T64 燧原 grid 封顶一发；precomputed-pos 改写作为其后的独立候选。
5. T60 E8 仅在离线签名审计 + 逐元素对拍双过之后占用一发。

留 ≥20% 额度给"均值门失败但芯片信号为正"的重开，不做无预注册收益假设的
同字节重掷（第四批教训：最后 10 发中 8 发重复测量仅换来 2 项小新高、零名次
提升）。

## 8. 证据分级

- **平台实测**：本文引用的全部逐芯 speedup、榜单读数、燧原/昆仑/华为的
  失败形态与门槛，均来自各题账本与 `data/batch5-*.json`。
- **上游源码级**：§6 的 i64 规则、报错文本、atomic 能力缺口，来自 FlagTree /
  FlagGems / libtriton_jit / sglang 的固定 commit 与 issue。
- **假设（未测）**：T63 clone 占比与 2x 预测、T62 B2 去 mask 收益、
  T64 precomputed-pos 收益、T60 E8 两条。这些在本文中一律标为假设，
  不得作为账本结论引用。

## 9. 相关产物

- 首轮与深度优化轮：[optimization-batch5-20260911.md](optimization-batch5-20260911.md)
- 开发契约：[strategy-batch5.md](strategy-batch5.md)
- 跨批次经验：[session-mining-retrospective.md](session-mining-retrospective.md)、
  [season2-retrospective.md](season2-retrospective.md)
- 提交与逐芯证据：[data/batch5-submissions-20260911.json](data/batch5-submissions-20260911.json)、
  [data/batch5-optimization-20260911.json](data/batch5-optimization-20260911.json)
