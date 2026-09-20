# Task 92 `unpad_draft_extend_output` 实验记录

```current
task: 92
operator: unpad_draft_extend_output
batch: 6
validity: valid
platform: completed(18360,e5,8/8,110.055x<TB;保e4 115.661;metax回退验证-8%一致)
candidate_stage: e5
team_best_stage: e4
team_best_speedup: 115.661
sealed: no
next: 首发即108x;轴=燧原3.5/昆仑8.9 vendor(gather/streaming配方);榜首295.62窗口待判
updated: 2026-09-20
```

## 2026-09-20 S0 首发记录 + 根因复盘

- 结构（`b7ae92fc`）：torch searchsorted 预计算行映射（T49 先例）+
  纯 gather kernel（2D grid 行×块）。8/8 首个有效 108.207x。
- **两小时"miscompile 追查"实为自身索引 bug**：src_row 已是展平行号
  （b*tpb+t），源行 stride 应为 H*D 而非 stride(0)=tpb*H*D（双重展平）。
  期间构造了 constexpr/runtime/if/mask/i32/i64 六维排查矩阵、多份"语义
  等价"探针——均因探针无意改用了正确 stride 而通过，形成"后端随机
  miscompile"假象。教训入账：**怀疑编译器前先字符化每个索引的物理
  含义**（与 T48 维度角色教训同构）。

## 2026-09-20 E1 平台终态：燧原 streaming vendor +143%，TB 109.45

- submission **18310** completed/valid，8/8，均值 **109.45 新 TB**（+1.1%）。
  **燧原 3.5→8.5（+143%，24-SIP+stages3 配方在 staged-gather 形态兑现）**；
  海光 170→184；华为 68.9→58.1（窗口回落）。

## 2026-09-20 E2 平台终态：华为 persistent +83%，TB 115.112

- submission **18321** completed/valid，8/8，均值 **115.112 新 TB**（vs
  109.45，+5.2%）。**华为 58.1→106.1（+83%，persistent 配方在 staged-gather
  形态兑现——与 T79 反例互补：gather 族的 persistent 效果按题分化）**；
  天数 191.7/A 198.7 窗口新高；燧原 8.9 持平 e1。

## 2026-09-20 E3 平台终态：BLOCK 4096 中性偏负，TB 保 e2 115.112

- submission **18329** completed/valid，8/8，均值 113.795 < TB 115.112（保 e2）。
  沐曦 99.9→113.0（+13%）但天数 191.7→184.3 / 海光 172.8→166.4 / B 133.7→
  129.1 同步回落——BLOCK 阶梯按芯分化再添一例。榜首 295.6 的逐芯形态
  （沐曦 245/海光 510）未破译，差距为结构级。

## 2026-09-20 E4 平台终态：BLOCK 回 1024 + metax w8，TB 115.661

- submission **18340** completed/valid，8/8，均值 **115.661 新 TB**（+0.5%）。
  海光 172.8→201.2（+17%）、B 133.7→142.5（+7%）；沐曦 99.9→82.6（warps8
  在此题为负——又一次 op×chip 分化）。榜首 295.6 结构差距仍在。

## 2026-09-20 E5 平台终态：metax 回退一致，TB 保 e4 115.661

- submission **18360** completed/valid，8/8，均值 110.055 < TB 115.661（保
  e4）。沐曦 82.6→100.2（回退 warps8 后恢复 e2 水位，-18% 判决二次确认）；
  燧原 8.4→6.2 / A 197→187 窗口回落。

## 2026-09-21 夜 e6 候选就绪（午夜第 5 弹，咨询第一优先结构轴）

- 结构（`4c7b9fed`）：**batch 连续段拷贝**——题面 cu/seq_lens 直接表达
  "每 batch 一段连续 H×D 拷贝"（padded 布局每 batch 接受前缀连续），
  彻底删除 searchsorted 行映射的全部框架开销；内核只用标量段起点 +
  连续向量偏移，无张量索引 gather。两个 decode bug 被数字探针在提交前
  拦下；codex-review 抓出 P1（lens/cum 直接相邻读破坏切片语义）→
  stride(0) 修复 + 交错切片回归测试。
- 预注册门：映射阶段占比假设下 wrapper 中位耗时 -15%+；均值 >115.661
  换 TB；八芯 exact（含 strided/零长度/非整除 tile 回归）。
- 五元组：commit `4c7b9feda87e40cd9c51661e791ce75e28e95913`；ZIP
  `e6-4c7b9fe` SHA `5914e4c4e7b3559d729c84d92151389e4170b4f84896d6ab8830f0621f1f7825`；
  test SHA `6c9f5e2624cde0d03723a01adee2f771af90bdadf2ff7cf5728dccb36ea467c6`；
  回执 SHA `51123accdda39b06e18f68fac5cb5549b168925f885989c0954e14caf2465c42`
  （`ready3-wave-20260921/`，四路径）。

## 2026-09-21 00:16 E6 平台终态：215.91（+100.2，结构性兑现！）TB 大刷新

- submission completed/valid，8/8。逐芯：天数 **405.9** / 沐曦 258.9 /
  燧原 **8.22** / 海光 **169.0** / 昆仑 17.1 / 华为 **97.7** / A **494.4**
  （反超榜首同芯 490.3）/ B 276.1。
- **根因定位**：跑 generic 的五芯全部起飞；三洞全是旧结构 vendor
  （_enflame/_hygon/_ascend 仍为 searchsorted 行映射），慢 2-5×。
  榜首 350.30 逐芯健康（148-609）。
- **E7 轴（已锁定）**：把 e6 段拷贝内核移植进三个 vendor 文件
  （纯移植、逐芯同 generic 语义）→ 预计海光 ~600 / 华为 ~500 /
  燧原 ~150 / 昆仑 vendor 后续 → 均值 ~340+ 冲 Top1。

## 2026-09-21 E7R 候选就绪并发射（段拷贝移植三 vendor + int64 修复）

- 结构（`ea745b6d`）：e6 段拷贝内核逐字节移植进 _enflame/_hygon/_ascend
  （e6 首判：generic 五芯 259-494 vs 旧 vendor 8/169/98）；codex-review
  P2 修复——`elems = n*span` 与循环基座 int64 化（≥2^31 段元素数静默不写，
  四文件同修，含 generic）。
- 预注册门：海光 ≥400 / 华为 ≥300 / 燧原 ≥100（generic 带读数下限）；
  均值 ≥300 进入 e8 精调（昆仑/沐曦 vendor，对标 Fields 379 / EvokeAgent 480）。
- 五元组：commit `ea745b6db332ef3b2bd62822d6eca9e1d7862be0`；ZIP `e7r-ea745b6` SHA
  `9dca7c26e2af189388601aa61b75046ae116ff14de4264c50781340d37206331`；
  test `6c9f5e2624cde0d03723a01adee2f771af90bdadf2ff7cf5728dccb36ea467c6`；
  回执 `top1day-20260921/unpad_draft_extend_output/` SHA
  `07cfc0f5cfac54234d42f61cb94478ece0c0105a84ad06d2511b816cd071b163`。

## 2026-09-21 00:33 E7R 平台终态：289.92（+74），海光 619.8 反超榜首同芯

- submission completed/valid，8/8。逐芯：天数 395.8 / 沐曦 242.3 / 燧原
  **18.7**（门 ≥100 未过，generic 字节仅 2.3×）/ 海光 **619.8**（>榜首
  609.1）/ 昆仑 17.5 / 华为 **257.9**（门 ≥300 差 14%）/ A 492.8 / B 274.6。
- 三 vendor 结构移植整体兑现（+74 均值）；天数/沐曦 -10~-17 属 int64 化
  或窗口噪声（A/B 稳定）。
- **E8 轴（三 vendor 单变量，一弹三芯归因）**：燧原 BLOCK 8192（T76
  streaming +21% 同源）；沐曦新增 _metax vendor + num_warps 8（relu2
  metax +20% 同源）；华为 BLOCK 2048（更小 tile 更多程序）。潜在
  +40~60 均值 → 330-350 进榜首带。

## 2026-09-21 E8 候选就绪并发射（三 vendor 单变量 pin，一弹三芯归因）

- 结构（`4e5cdc02`）：_enflame BLOCK 8192（T76 阶梯顶 +21% 同源）；
  **新增 _metax** vendor（generic 字节 + num_warps 8，relu2 metax +20%
  同源，对标 Fields 沐曦 379）；_ascend BLOCK 2048（更多驻留程序，
  对标 506-701 带读数）。generic/海光/昆仑/天数/A/B 字节不动。
- 预注册门：沐曦 ≥300 / 华为 ≥350 / 燧原 ≥40；均值 >320 进终段调优。
- 五元组：commit `4e5cdc02b22f3c9654405160d648e02dc6182ab8`；ZIP `e8-4e5cdc0` SHA
  `d0402fd9e0f16a2334c01791376c9d501be8b51537d4bd22cd9b1663ad7e6166`；
  test `6c9f5e2624cde0d03723a01adee2f771af90bdadf2ff7cf5728dccb36ea467c6`；
  回执 `top1day-20260921b/unpad_draft_extend_output/` SHA
  `719f69b7d638c80d3cc5561017fdd6074c4d9ad5ab7229d8b5505c5dc6eb4106`。

## 2026-09-21 00:44 E8 平台终态：269.65 判负，宽度阶梯机制浮出

- 三 pin 归因：燧原 BLOCK 4096→8192 = 18.7→**33.6**（+80%，宽度正效）；
  华为 4096→2048 = 257.9→**129.6**（腰斩，宽度强正效的反向确认）；
  沐曦 warps8 = 242.3→**213.3**（warps 负效）。均值回落至 269.65，TB
  仍保 e7r 289.92。
- **机制结论**：该 streaming copy 在华为/燧原/沐曦全部呈宽度阶梯
  （~1.8-2×/翻倍）；GCU 模型三败确证（行形式 0.5-0.6、flat 全网格 0.2-0.5、
  少程序+超宽 2.61@relu2 / 33.6@unpad）——**E9 全走宽度阶梯**：
  燧原 BLOCK 16384 + ≤24 程序（relu2-GCU 同源）；华为 8192；沐曦 8192
  （去 warps pin 回默认）。
