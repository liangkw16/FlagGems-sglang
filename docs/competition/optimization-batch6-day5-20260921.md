# 第六批 Day5 冲榜方案（2026-09-21 晚预研，09-22 起三日执行）

> 依据：09-21 20:56 榜单快照（commit `3dd3b342`）+ 09-21 00:13 全 17 题逐芯快照
> （`data/leaderboard-snapshots/batch6-20260921-0016.json`）+ GCU 官方源码调研
> （FlagGems `_enflame` gcu300 codegen / FlagTree enflame backend，经 gh api 逐文件验证）
> + SGLang/vLLM/FlagGems 上游结构扫描（研究 agent 逐文件核对）。
> 窗口：09-24 19:59:59 截止；额度 09-21 已 30/30，09-22 起每日 30 发，共 ~90 发。

## 一、逐芯情报结论（榜差性质）

对每题榜首做逐芯分解后，差距高度集中，不是均匀落后：

| 模式 | 题目（榜首单芯极端值 vs 我方） | 性质 |
| --- | --- | --- |
| 燧原单芯 | T78（15268x vs 0.37）、T76（7.96 vs 0.44）、T82（18.7 vs 0.85）、T86（9.0 vs 1.03）、T89（14.7 vs 3.86）、T91（10.5 vs 0.67） | 我方 GCU kernel 结构病理 |
| 华为单芯 | T79（1212 vs 97.5）、T85（179 vs 8.2）、T87（27.3 vs ~2.6）、T92（843.7） | 华为专属结构未破译 |
| 全芯均匀 | T84（917.8 vs 163.1，5-8x/芯） | 算法/launch 结构差距 |
| 未提交 | T77（榜首 1731x，16 队达标） | 纯增量空白 |
| 多芯 3-5x | T80（569 vs 201）、T88（30.5 vs 16.9） | 待破译 |

## 二、三条结构性主线（按 EV 排序）

### 主线 1：T77 补位（唯一未提交题，榜首 1731x）

- 09-18 已完成 generic 双路径（小批 per-request + striped ≤64 stripe）+ 全边界
  测试 + 两轮 codex-ask 修复；本日补齐燧原 vendor。
- **燧原 int64 墙的解法**：GCU 签名级 i64 禁令（T60 八轮）+ torch-gcu int64
  物理窄化（`gcu_empty_tensor.cpp:50-64`，4N 字节装 N 个逻辑元素）+ T60 E8
  双词写 100% 失配 ⇒ 窄化打包（元素 i 的物理槽 = int32 view 下标 i）。vendor
  内做**一次性 host 探针**（arange(4) 读 int32 view：`[0,1,2,3]`=打包 /
  `[0,0,1,0]`=标准），按结果分发 int32 打包内核或 int64 标准内核——代理上
  走标准分支故数值矩阵可完整验证，GCU 上走打包分支。两条计算路径都是 Triton。
- 发射即探针：8/8 = 假设 B 定案；仅燧原数值失败 = 假设 A，下一发换双词形态
  （内存安全，两种情况下都不会越界写）。
- **预注册门**：任何 8/8 有效即成功（预估数百 x 均值，进榜即前 5 量级）。

### 主线 2：燧原 GCU 结构规则集（一揽子收割 ~6 题）

官方源码验证的 GCU300 规则（此前只经验性知道一部分）：

1. **grid 封顶 12 CTA**（`max_grid_size=(12,1,1)`，超出走 grid-stride）；
2. **num_warps=2**（`enflame_heuristics_for_num_warps` 钉死；4-SIPS 模式 4）；
3. **tile 超大**：32K 元素基准，4 字节元素 ×2、2 字节 ×4；
4. **stride 必须编译期互整除才走 DMA**，否则 tile 缩 4 倍走非 DMA——
   **运行时 stride 传参 = DMA 判定失败**（我们 vendor 的通病！）；
5. 全链 **int32 寻址**（`enable_i64=False`，int64 偏移算术软仿真）；
6. gather/scatter 单 block 跨度 < 512MB（MMU 窗口，超出 BLOCK 塌缩为 1）；
7. topsaten 原生算子：i64 全部 NOT_SUPPORT 走 CPU；**非连续 stride 不受支持**
   （解释 T78 参考 `expand+cat+cast` 在 GCU 上病理 → 15268x 的来源）。

应用：T78 e3（已写：constexpr 化全部 shape/stride + int32 寻址 + 12 CTA +
warps 2 + nope 1D 平铺）；随后 T82/T86/T89/T91/T76 的 `_enflame` 按同一规则
集改造（各自调宽度/分组）。**预注册门**：每题燧原 ≥ 当前 2 倍且其余七芯
不动（vendor-only 改动）；T78 特例：燧原 ≥2 即保留，≥50 视为基线病理捕获。

### 主线 3：T84 launch/分配结构（163 → 目标 400+）

差距全芯均匀 5-8x、shape 小、参考带 Python 专家循环+nonzero host sync ⇒
launch-bound。SGLang 现行结构 = 2 kernel + 0 memset + 0 clone（直方图共享内存
+ 3 级 warp scan + 二分填 expert_ids；scatter 用 caller 的 cumsum_buffer 当
cursor）。我方 e11 = memset + 4 kernel + clone + 3 alloc。e12（已写）：

- 输出**原位写**（返回传入 buffer；eids 未定义尾=原值，与 reference 的 clone
  语义逐位一致，含测试的全量断言）；
- `_compute` 融合 launch：P 个程序原子直方图 + **ticket 最后程序做向量化
  `tl.cumsum` scan**（256 次串行迭代 → 1 次向量）+ 逐 lane 变长填 expert_ids；
  其余程序并行毯填 sentinel；
- cursor 落在 caller 的 `cumsum_buffer`（题面 scratch），counts+ticket 一次
  1KB memset；
- `_scatter` 字节不动；
- 发射：5+clone → **3**（memset + compute + scatter）。
- 昆仑/华为/燧原 vendor 字节保持 e11 验证形态（单变量：仅 generic）。
- **预注册门**：均值 >163.14 换 TB；任一走 generic 的芯（天数/沐曦/海光/A/B）
  相对 e11 回退 >10% 即判负回滚。

### 华为线（待破译，第二优先级）

T79/T85/T87/T92 的榜首华为值 10-22x 于我方。已知资产：persistent（copy 族
+30~97%）、无原子模板（T84）、gather best-practice。09-22 用当日 T77/T84
发射的华为回执校准后，再决定是否给 T79/T92 各发一发 persistent+constexpr
化 vendor 探路。**不盲投**。

## 三、09-22 发射序（额度 30）

| 序 | 候选 | 预注册门 | 备注 |
| ---: | --- | --- | --- |
| 1 | T77 s0 | 8/8 任意有效 | 00:01 首发；燧原失败=A/B 数据 |
| 2 | T84 e12 | >163.14 | 三发射结构 |
| 3 | T78 e3 | 燧原 ≥2 | GCU 规则集首验 |
| 4-8 | T82/T86/T89/T91/T76 enflame | 燧原 ×2 | 规则集验证后批量 |
| 9+ | 按回执迭代 | - | 华为线/守榜/窗口重掷 |

每发前置：codex-review 通过 → release 回执 → ZIP 验签 → preflight。

## 四、今晚开发清单

1. ✅ T77 燧原 vendor（探针分发）——screening 双源 4 用例 0 失败已过；
2. ✅ T78 e3 燧原 vendor（constexpr+int32+12CTA+warps2）——screening 进行中；
3. ✅ T84 e12 generic（原位+ticket+向量化 scan）——screening 进行中；
4. T82/T86/T89/T91/T76 `_enflame` 规则集改造（screening 过后批量）；
5. 全部过 codex-review 后 commit → release → ZIP → 待 00:01。
