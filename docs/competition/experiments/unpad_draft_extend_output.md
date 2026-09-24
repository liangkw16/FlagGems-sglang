# Task 92 `unpad_draft_extend_output` 实验记录

```current
task: 92
operator: unpad_draft_extend_output
batch: 6
validity: valid(8/8,e25,344.59575x TB)
platform: e25(20780)valid 8/8 avg 344.59575微幅新TB:amd int32中性(280.9)/昆仑80.2(水位站稳,兼除i64大向量雷)/沐曦248.1回温/华为435.5(窗内回落)
candidate_stage: e25
team_best_stage: e25
team_best_speedup: 344.59575
sealed: no
next: 距c2flow 416.58差72(-17.3%);b 281vs437与华为435-465vs665两结构缺口未破;收盘前视水位对e25字节防御重掷≤2次(昆仑≥80且沐曦≥248为进场带)
updated: 2026-09-24
```

## 2026-09-23 E23 候选：华为整块无掩码，尾块保留 e22 路径

- 榜单 17:50：我方 e22 **341.3097** vs c2flow **400.920925**，总分需
  增加 **476.8898**；华为 444.1506 vs 575.3874、B 281.2718 vs
  441.9676 是最大两洞。仅把华为追到 c2flow 水位，均分预计只加
  16.4，仍不能独立登顶；本弹检验能否突破 e22 的 444 水位，后续
  仍需 B 等多芯收益。Triton [tl.load 语义](https://triton-lang.org/main/python-api/generated/triton.language.load.html)
  明确未指定 `other` 的 masked-out lane 为未定义值；e22 已以同 mask store
  安全处理尾块。
- 结构只改 `_ascend`：保留已 8/8 有效的 e22 `(bs,tiles)` grid、
  `BLOCK=16384`、int64 地址和尾块 mask；每个 base 若
  `base+BLOCK<=elems`，load/store 整块不带 mask，否则沿用 e22 的
  masked load/store。e21r 同时改 persistent rotation、fp32 尾和 warps16，
  华为 case 3 有 1520/164352 失配；e23 不继承这些变量。平台未公开
  该 case 的完整输入，现有 `test_unmasked_main_masked_tail_split` 精确覆盖
  全块、单尾、跨段和零长，不能代替目标芯实测。
- screening 源 SHA-256
  `d8cc43d332e0d251d770b69374ab1daf2385228d8f56eb6d930de59c5f3cd7c8`；
  代理回执 `e23-screen-20260923/verification.json` SHA-256
  `42faadf9a5f9177e39298b49729da593d3a79f45e4617dfe2cf11a27746a4b02`，
  6/6 方法、0 失败/错误/skip；三组五轮 AB/BA wrapper-inclusive
  速度比 e22/e23 **0.991 / 1.001 / 1.002**，原始样本
  `e23-screen-20260923/bench.out` SHA-256
  `87859df4f4851fbb06c3285b8c44b303bff60786e82147097b4368a28724ba9d`。
  NVIDIA 仅证明代理侧无明显退化，华为收益仍未知。
- source/verification commit
  `34dad5bad4421274216126101195d44c81e234be`；测试 SHA-256
  `cd676efb1d1f525f5d0dce5e14f4df6398cb4564278e8ba9c79f982b870c3357`。
  八成员 ZIP `e23-34dad5b/unpad_draft_extend_output.zip`，19090 B，
  SHA-256 `e0e7e7dd12c996b9f33e20e4a9dde5d1795460ce64e997bfaa1907686aa51728`；
  打包器 `--dry-run` / `--verify-existing` 与 `unzip -t` 通过。
  release 回执 `e23-34dad5b/verification.json` SHA-256
  `b3f29cd166ce6d30c03aa04cb48807998f938ab5a4351058f603f39450ebb6c6`，
  日志 SHA-256 `8a5684a2fe0bc4b777013fad5fe70cf2763cfda0909178ca886716fc0cd2cad5`；
  RTX 5070 Ti 代理 6/6、0 失败/错误/skip/xfail，八成员各 13 次真实
  launch。华为目标 `target-runtime-unverified`。
- **预注册门**：codex-review 无可靠缺陷才 preflight；平台 8/8 正确、
  每芯 speedup≥0.1 才是有效候选；均分 > e22 的 341.3097 才保留新 TB。
  华为 ≥575.3874 表示本结构达到榜首同芯水位。若华为编译/数值失败
  或均分未超 e22，恢复 e22 `_ascend` 不可变 ZIP 字节，不重投 e23。
  其他七芯源成员逐字节冻结；其读数变化先按同字节环境波动核对。
- `codex-review --commit 34dad5ba --spec .../92-unpad_draft_extend_output.md`
  成功完成，`gpt-6-sol/max`；Spec 和 Standards 均未发现可确认的新增缺陷。
  评审枚举 11,308 组长度、tile 与网格组合，完整 tile 和尾 tile 均恰好
  覆盖有效元素一次；评审不证明华为目标芯编译或性能。审查门通过。

## 2026-09-23 E23 平台终态（20457）：8/8 有效，但结构判负并回滚

- 一次性提交 **20457**（daily_seq 26，18:09:28 +08），远端 ZIP SHA/大小
  验签通过。平台 18:13:05 终态 `completed/valid`，8/8 正确，每芯 ≥0.1；
  均分 **329.8088 < e22 TB 341.3097**，排名仍按 e22。codex-review 的
  静态正确性结论成立，但平台性能否定了本轴。
- 逐芯 e23 / e22（同字节成员也记录波动）：天数 394.1164 / 406.8622，
  沐曦 265.3566 / 249.4908，燧原 125.8396 / 126.704，海光
  662.6268 / 662.2952，昆仑 **79.6568 / 33.1488**（同字节 +140%，
  表明强窗口波动），华为 **316.2016 / 444.1506**（唯一变更成员，
  −28.8%），A 527.9116 / 526.5542，B 266.761 / 281.2718。
  净八芯和 −92.0072，华为独自贡献 −127.949；即使昆仑同字节读数升高，
  仍未抵消华为下降。
- 命中预注册“均分未超 e22”回滚门：从不可变 e22 ZIP
  `e22-5d57614/unpad_draft_extend_output.zip` 取回
  `unpad_draft_extend_output_ascend.py`，SHA-256
  `e11c642072ac46164a38054f6b7747d814638f434c7e4b43f9d810a41cd9c03e`；
  工作树同 SHA，源码回滚 commit `47dc1382`。e23 ZIP 和平台记录保留，不重试。
  剩余额度 **4/30**。完整 GET 保存在
  `artifacts/competition/unpad_draft_extend_output/e23-34dad5b/platform-status-20457.json`，
  SHA-256 `9b03eda13ffee6984c6ccb41f1fadf11bf3c69e8d1750794a6e37833c3c2f829`。

## 2026-09-23 E22 平台终态（20358）：valid 8/8 均值 341.31 新 TB

- 结构（`5d576147`）：e19r 证明形态（(bs,tiles) grid + BLOCK=16384 +
  int64 offs/elems integer mask）+ 单变量 drop `other=0` masked-load
  预填（chip-rulesets.md:25 MTE2 串行化）。预注册门达成：均值
  >331.10 换 TB ✓；华为 444.15 落在 [378, 480) 区间——未触判负
  （<378 或数值失败）也未达 480 大胜带。
- 逐芯：天数 406.9 / 沐曦 249.5 / 燧原 126.7 / 海光 662.3 / 昆仑
  33.1 / **华为 444.2** / A 526.6 / B 281.3。八芯全部 pass。
- 五元组：commit `5d576147da38822458fd7d911e2ec4053ac4caf1`；ZIP SHA
  `02c6855acd1c2b13507b09fa1359cfef156f4c767dfd721760ed68787a213d99`
  （8 成员）；test `cd676efb1d1f525f5d0dce5e14f4df6398cb4564278e8ba9c79f982b870c3357`；
  回执 `day6-climb-20260923/t92e22-wf/`（8 源 × 13 launch，
  verification_commit=5d576147）。
- 判读：华为轴 drop-other-prefill 兑现部分（e19r ~378-442 → 444，
  带内上沿）但 680+ 场带（金狐狸/CosmosMind）未破——纯 copy 的
  MTE2 串行化只是华为缺口的一部分，剩余疑在 launch/网格形态。
  e21r 数值败未复现（e22 = e19r 字节 + 单 token 差异，八芯过）。

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

## 2026-09-21 E9 候选就绪并发射（三 vendor 宽度阶梯）

- 结构（`e380fc5b`）：_enflame BLOCK 16384 + tiles=min(需求数, max(1,24//bs))
  （relu2-GCU 同源程序模型）；_ascend BLOCK 8192；_metax BLOCK 8192 回默认
  warps。预注册门：华为 ≥400 / 沐曦 ≥300 / 燧原 ≥50；均值 >320。
- 五元组：commit `e380fc5b5295ae43051354e45f9a2233d00deda1`；ZIP `e9-e380fc5` SHA
  `87653e3005a1d4f5d96ef752f5627f1191ce74880bfcd5e8489fa90961a1033d`；
  test `6c9f5e2624cde0d03723a01adee2f771af90bdadf2ff7cf5728dccb36ea467c6`；
  回执 `top1day-20260921c/unpad_draft_extend_output/` SHA
  `b484b622f08a1b239713707c7662467d2bd3992c22115c850ed2b1dabd3736bc`。

## 2026-09-21 00:52 E9 平台终态：296.98 新 TB（+7），燧原/华为阶梯续爬

- 逐芯：天数 389.9 / 沐曦 249.6（8192 仅 +3%，宽度饱和）/ 燧原 **55.1**
  （16384+24 程序兑现，18.7→33.6→55.1）/ 海光 594.1 / 昆仑 16.6 /
  华为 **315.6**（2048→129.6 / 4096→257.9 / 8192→315.6，边际递减）/
  A 488.8 / B 266.2。**E10 轴**：燧原 32768、华为 16384 续爬；
  沐曦换轴（宽度无效）、天数/昆仑待结构情报。

## 2026-09-21 E10 候选就绪并发射（燧原 32768 / 华为 16384 续爬）

- 结构（`36d980af`）：_enflame BLOCK 32768（55.1@16384 续爬）；_ascend
  BLOCK 16384（315.6@8192 探拐点）。预注册门：燧原 ≥80 / 华为 ≥380。
- 五元组：commit `36d980af7a95eae8143ca3e9e0e687a378d93155`；ZIP `e10-36d980a` SHA
  `154b8252cad4eaf58bf34a55be382e97bc5c24dedc99a41e9bc24fc79e64dc1d`；
  test `6c9f5e2624cde0d03723a01adee2f771af90bdadf2ff7cf5728dccb36ea467c6`；
  回执 `top1day-20260921d/unpad_draft_extend_output/` SHA
  `369a390218c44bb9f69adb9fbd1c9fe8749cc05eafa01887ac2451ac5f63d5ac`。

## 2026-09-21 01:07 E10 平台终态：314.97 新 TB（+18），华为 442 再加速

- 逐芯：天数 395.6 / 沐曦 247.7 / 燧原 51.9（32768 饱和）/ 海光 603.9 /
  昆仑 17.8 / 华为 **442.0**（129.6→257.9→315.6→442，每翻倍 +40% 再加速）/
  A 478.1 / B 282.7。**E11 轴**：华为 32768（冲 600+）+ _nvidia/_amd
  卡轴 8192（band 266-494 → 298/515）。

## 2026-09-21 E11 候选就绪并发射（华为 32768 + 新增卡轴 vendor）

- 结构（`8b47cce3`）：_ascend BLOCK 32768（442@16384 冲 600+）；新增
  _nvidia/_amd（generic 字节 BLOCK 8192，若 card_a/b 映射即得卡轴宽度
  阶梯，不映射则无害）。预注册门：华为 ≥550；卡任一 +20。
- 五元组：commit `8b47cce370af2555d7060be37e17095b02e6e1bf`；ZIP `e11-8b47cce` SHA `416e50a33db6ee90a4e679f0994144518eefc05652eb6997c3d838f2625068f1`；
  test `6c9f5e2624cde0d03723a01adee2f771af90bdadf2ff7cf5728dccb36ea467c6`；
  回执 `top1day-20260921e/unpad_draft_extend_output/` SHA `cb2dd04e10408fa0f5a08bbeab69c7193554694efe8ee1f855614ace57344a45`。

## 2026-09-21 01:25 E11 平台终态：310.56，华为 32768 过拐点，卡轴映射成功

- 逐芯：天数 393.2 / 沐曦 247.5 / 燧原 53.5 / 海光 **637.0**（band 顶）/
  昆仑 16.6 / 华为 **378.1**（峰值在 16384=442，32768 过头）/ A **493.9**
  （_nvidia 8192 兑现 +16）/ B 264.7（_amd 8192 微负，需窄向）。
  最佳保 e10 314.97。**E12 轴（六文件）**：ascend 16384+stages2、nvidia
  16384、amd 2048、enflame 16384+stages4、metax stages4、新增 _kunlunxin
  1024（昆仑窄向假设）。

## 2026-09-21 E12 候选就绪并发射（六文件逐芯探针）

- 结构（`1558b04e`）：ascend 16384+stages2（442 回峰点+流水探针）、
  nvidia 16384（493.9@8192 续爬）、amd 2048（窄向）、enflame 16384+stages4
  （55.1 回峰点+深流水）、metax st4、**新增 _kunlunxin 1024**（窄向假设）。
- 五元组：commit `1558b04e8ae614f506c4cb86b826c0486625e72a`；ZIP SHA `a8a1fd51d436ccb02f0b40f8e5be5b78c71470333d0e493b6852a2e6601f3aed`；test
  `6c9f5e2624cde0d03723a01adee2f771af90bdadf2ff7cf5728dccb36ea467c6`；
  回执 `top1day-20260921f/` SHA `7604fc12f5d24e582577fea7065f7063ef045ce999ae191e85fd3f28496b9566`。

## 2026-09-21 E13 候选就绪并发射（天数轴 + 六探针组装）

- 结构（`55531b9c`）：generic（**天数专用**——七 vendor 槽齐备后 generic
  只跑天数）BLOCK 4096→8192；同时吸收 e12 全部探针（ascend 16384+st2、
  nvidia 16384、amd 2048、enflame 16384+st4、metax st4、kunlunxin 1024）。
  e12 因上传中断未发射，探针并入本弹。预注册门：天数 ≥420；均值 >322。
- 五元组：commit `55531b9c2ab30a10f900ea292abfad2af8519950`；ZIP SHA `74217bdb82107d48b95d8fe1ed884763b563b38119567499f2d5db1b690c5180`；test
  `6c9f5e2624cde0d03723a01adee2f771af90bdadf2ff7cf5728dccb36ea467c6`；
  回执 `top1day-20260921g/` SHA `4b49a510f52379dd3ece17d74cdfb2e8b20952ce3f54a8346f868950fe87ee03`。

## 2026-09-21 07:06 E13 平台终态：310.63，三轴归因 + 峰值表更新

- 天数 8192 负效（393→372.8，**回 4096**）；昆仑 1024 灾难（16.6→6.1，
  **昆仑要宽**——与海光相反）；卡B 2048 兑现（264.7→**279.7**）；华为
  st2 微负（414.4 vs plain 442，回 plain）；燧原 st4 微正（56.5）；沐曦
  st4 无效（247.2）；卡A 16384 饱和（495.8）。
- **E14 峰值组装**：generic 4096、ascend 16384 plain、enflame 16384+st4、
  metax 8192 plain、nvidia 16384、amd 1024 探窄、kunlunxin 16384 探宽。

## 2026-09-21 E14 候选就绪并发射（峰值组装弹）

- 结构（`c7973b17`）：各芯已知峰回填——generic 4096（天数）、ascend
  16384 plain（华为 442 峰）、enflame 16384+st4（56.5）、metax 8192 plain、
  nvidia 16384（卡A 495.8）、amd 1024（卡B 窄向探底）、kunlunxin 16384
  （昆仑宽向探针）。预注册门：均值 >314.97 换 TB；昆仑 ≥25。
- 五元组：commit `c7973b17ed109ab0ce6f354141a3531dacd630c8`；ZIP SHA `1b2fe4bc82f87b2d88be15f52f943d84aa20c4c137dff5cbf7f64ecda0decf6e`；test
  `6c9f5e2624cde0d03723a01adee2f771af90bdadf2ff7cf5728dccb36ea467c6`；
  回执 `top1day-20260921h/` SHA `afb7f57128d978bb40fdcf98e44861bdeb048a57482373f9bd30b357e79a0dd7`。

## 2026-09-21 07:18 E14 平台终态：316.07 新 TB，昆仑洞关闭，燧原慢窗

- 逐芯：天数 395.8 / 沐曦 250.9 / 燧原 **2.3**（同字节 e13 读 56.5——
  慢窗，吞 6.6 均值）/ 海光 588.1 / 昆仑 **34.6**（16384 宽向兑现，
  16.6→34.6 进场带 26-39）/ 华为 **492.0**（同字节 442-492 窗口带，
  逼近 506-701）/ A 487.2 / B 277.6（1024≈2048 平）。
- 若燧原读 ~55：均值 322.7。**E15**：注释载体重掷燧原慢窗（崩溃族/
  慢窗协议，≤2 次）+ 昆仑 32768 续爬。剩余结构洞：沐曦 -128、天数 -84、
  燧原 -93（真带宽 vs 窗口差）。

## 2026-09-21 E15 候选就绪并发射（燧原慢窗重掷 + 昆仑 32768）

- 结构（`3c1107c1`）：燧原注释载体（字节同 e14，2.3 慢窗重掷，协议
  ≤2 次之第 1 次）；昆仑 16384→32768 续爬。预注册门：燧原 ≥40（窗口
  恢复）；昆仑 ≥30。
- 五元组：commit `3c1107c10f02b2af85f83408e14375359a15626b`；ZIP SHA `2f201f0e9d2b56ec6565d5cf2bcbfedb639facb542162cbbee666a1a20372dba`；test
  `6c9f5e2624cde0d03723a01adee2f771af90bdadf2ff7cf5728dccb36ea467c6`；
  回执 `top1day-20260921i/` SHA `bb57f5db12567e1b958c284d83b07844671bf616ebb2b63a001ef1fb15f413a0`。

## 2026-09-21 07:27 E15 平台终态：319.90 新 TB，燧原窗口恢复

- 逐芯：天数 388.8 / 沐曦 248.5 / 燧原 **54.7**（2.3 确系慢窗，重掷
  兑现）/ 海光 **643.6**（新顶）/ 昆仑 27.3（32768<16384 峰，回退）/
  华为 437.4（437-492 窗口带）/ A 490.5 / B 268.4。
- **E16（咨询第六轮后首个结构弹）**：燧原固定 P 输出分区内核（均衡
  slab + 段行走，codex 首选）；沐曦/天数 uint32 位宽重解释路径（指令
  数减半假设）；昆仑回 16384。

## 2026-09-21 E16 候选就绪并发射（咨询结构弹：分区内核 + uint32 路径）

- 结构（`0e79dd3f`）：_enflame 固定 P=24 输出分区内核（均衡 slab + 标量
  段行走；两次内部修复：u32 view 守卫 dim 偶+偏移对齐、段边界 pos 推进
  ——均被 release 门/审查拦截于发射前）；generic(天数)/_metax(沐曦)
  uint32 位宽路径（指令数减半假设，奇维/奇偏移回退元素路径，新增
  回归契约测试）；_kunlunxin 回 16384 峰。预注册门：沐曦 ≥300 /
  天数 ≥420 / 燧原 ≥70；均值 >325 视为结构兑现。
- 五元组：commit `0e79dd3f96a88700bad67efb87591beca722a04b`；ZIP SHA `c1f91600be744d82c51082623a374712af3bbd7f9b5799d00ab3c2246f74a1cd`；test
  `6c9f5e26` 起版本以 verification commit 为准
  （`0e79dd3f:tests/test_unpad_draft_extend_output.py`）；回执
  `top1day-20260921j/` SHA `a426893954dda4d00e1dfbf21086887c664d2d30029439e50a97cc03ef3299f0`。

## 2026-09-21 07:53 E16 平台终态：305.52 三结构探针全负

- 燧原输出分区内核 54.7→**19.7**（GCU 不奖励均衡 slab，(bs,tiles)
  形态更优——分区假设证伪）；天数 u32 388.8→**368.1**、沐曦 u32
  248.5→**245.2**（位宽重解释无增益）。昆仑回 16384 兑现（33.7）。
  TB 保 e15 319.90。三结构洞（沐曦/天数/燧原）今日已知轴全部穷尽，
  待第三方结构情报。

## 2026-09-21 E17 候选就绪并发射（傍晚窗口重掷，e15 峰值字节）

- 结构（`1f20b086`）：注释载体，内核字节同 e15（TB 319.90）。目标：
  抽华为 437-492 / 海光 588-643 窗口带的高位。预注册门：均值 >319.90
  换 TB；燧原 ≥40（窗口健康）。
- 五元组：commit `1f20b086bdf611a1bc168d552151e0a18e161949`；ZIP SHA `b605e9d504c9a47ee91e648afda28542fccb7c0fc7e210c4a8316d3e0d0d7781`；test
  `a244d03ac9281f585da62c5d39de2b93b941b248d8d0662768a0871606c38d4a`；回执 `top1day-20260921m/` SHA `12d5292bc08a54c3b98e8ea9f49b95f760eeed46824a520d7e1511f1d905baea`。

## 2026-09-21 08:37 E17 平台终态：329.64 新 TB（+9.7），窗口重掷命中

- 逐芯：天数 394.3 / 沐曦 252.7 / 燧原 54.3 / 海光 615.5 / 昆仑 34.7 /
  华为 **507.4**（历史最高，进榜首带 506-701）/ A **503.9**（新峰）/
  B 274.3。当前榜首带 350.3/349.9/349.3/336.0——我方 329.6 居 #5。

## 2026-09-21 E18 候选就绪并发射（当日最后一发窗口重掷）

- 结构（`d546e391`）：峰值字节注释载体（同 e17 的 329.64 字节族）。
  预注册门：均值 >329.64 换 TB。
- 五元组：commit `d546e3915d20bc2f1505e8a68985b8c21db73109`；ZIP SHA `dc4efb42fb47320b1b7148e204644ab855d864993bed581d3b3acb82080e15ca`；test
  `a244d03ac9281f585da62c5d39de2b93b941b248d8d0662768a0871606c38d4a`；回执 `top1day-20260921o/` SHA `f7115c0d02e1f63a128bcb663ccee1873ea2f1fdc70a76da0a2cc29a07d20c6c`。

## 2026-09-21 09:07 E18 平台终态：324.97（窗口抽 454），TB 保 e17 329.64

- 当日 T92 战线收束：115.66 → **329.64**（+185%，7 个新 TB），距榜首
  350.30 差 6.2%；#5 居位（前四 350.3/349.9/349.3/336.0）。剩余结构洞：
  沐曦 -120、天数 -85、燧原 -94（今日已知轴穷尽，待新情报）。


## 2026-09-22 E19/E19R：燧原规则集兑现 + 天芯评测挂死重掷

- E19（19456，enflame 眧原生真 gcu300 几何：constexpr stride（含非连续
  lens 契约）+ int32 + ≤12 CTA + warps2）：**燧原 54.3→126.1（+132%）**，
  海光 666/华为 427/A 532 同发健康；**天芯评测执行超时 3630s/3600s**
  （子进程 S 态阻塞：锁/IO/驱动）判 invalid_correctness——评测挂死族，
  非字节问题（generic 未动，e17 同字节天数曾读 394）。
- E19R：comment-only 载体重掷（挂死族 1/1）。回执
  `day5prep-20260921/unpad_draft_extend_output-e19r/`；发射后回执待判。


## 2026-09-22 E19R 平台终态：331.10 新 TB（挂死重掷命中）

- E19R（19475）：8/8 valid **331.10 > 329.64 换 TB**。逐芯：天数 394.6
  （挂死恢复）/ 沐曦 247.4 / 燧原 126.2 / 海光 663.9 / 昆仑 33.1 /
  华为 377.9 / A 526.7 / B 278.9。
- 五元组：commit `075a0a5346dca01596829f0bed0febead55842b5`；ZIP `artifacts/competition/unpad_draft_extend_output/e19r-f9abadc/unpad_draft_extend_output.zip`
  SHA `a9b2ba0a96e96036076bf8444d6466978cbc499654aa4449cf4d76cbc025734f`（8 成员）；回执
  `day5prep-20260921/unpad_draft_extend_output-e19r/verification.json`
  SHA `0ae3c379a5b55a4e5f91a1597541450a5cf0c8b3b2e1bec8edea380450249825`。
- 重掷额度（1/1）用尽，该轴关闭。距榜首 12.3%（华为 843 单芯为主）。


## 2026-09-22 深夜 E20 上膛（09-23 午夜首发第 2 发）

- E20（`423699e`，ascend 规则集版）：int32 寻址（Vector ADD 无 i64）+
  fp32 尾掩码（Vector CMP 无整数路径；标量分支门在 2^24 域内，codex-
  review 抓的 fp32 边界漏写已修）+ 去 masked-load other 预填 + warps16
  （≥4096 tile 档）。release 绿；ZIP `unpad_draft_extend_output/
  e20-423699e/`；回执 `.../unpad_draft_extend_output-e20/`。
  预注册门：华为 ≥500 换 TB 方向确认；均值 >331.10 换 TB。

## 2026-09-23 E21R 上膛（ascend UB-fit：unmasked-main + 小 fp32 尾，待发射）

- 结构（ascend 字节最早出现于 `91928ef1`，ZIP 打包自 `bcf63bbc`——其树中
  `_ascend/ops/unpad_draft_extend_output.py` 与
  `tests/test_unpad_draft_extend_output.py` 与 `91928ef1` 逐字节一致）：修
  e20（20162）华为 BiShengHIR `ub overflow, requires 3145984 bits while
  1572864 bits available`（384KB > 192KB UB；超额 ~196,640B = BLOCK=16384
  宽度下物化的 fp32 offs 转换 + fp32 mask，16384×4B×3 量级——e14 同
  BLOCK 整数 mask 形态曾编译并读 442-507，宽 fp32 向量是仅有的新大
  buffer）。主循环整 BLOCK=16384 tile **无 mask** 流式拷贝（宽度阶梯峰
  2048→129.6 / 8192→315.6 / 16384→442 / 32768→378，保 16384；接受前缀
  内所有触及元素落在 `[src, src+full_end)`，内存安全；T40 E16
  unmasked-main 结构先例华为 +130%），余量经单条 BLOCK_TAIL=2048 fp32
  masked 尾循环排空——其比较操作数为循环局部且被 BLOCK 界定
  （<16384<<2^24），e20 的 2^24 域 bug 无法触发，标量守卫分支删除；每个
  循环的活跃向量集是 e14 已编译形态的子集，UB 需求单调不增。**e21 的
  capped persistent rotation grid=(min(bs,64)) + warps16 一并在此字节内**
  （`395c6d7f`，e21 本体未单独上膛）。新增回归
  `test_unmasked_main_masked_tail_split` 进 RELEASE_REQUIRED_TESTS。
- release v2 绿（回执 `day5prep-20260921/e21r-ascend-ubfit-unmasked-main-wf/`，
  verification_commit=`bcf63bbc`）：5 测试 101 case 0 败 0 skip，8 源 ×
  11 non-warmup kernel launch，RTX 5070 Ti / torch 2.13.0+cu130 /
  triton 3.7.1（nvidia-proxy；ascend 仍 target-unverified，交平台）。
- 五元组：commit（ZIP source=verification）`bcf63bbcb25babe654c2a29f23248c5067e2dd5f`；
  ZIP `artifacts/competition/unpad_draft_extend_output/e21r-bcf63bb/unpad_draft_extend_output.zip`
  SHA `e916ae46bd96ed9bfb782ab8573a69f3cf7c4551695fb908b3fa6c988c329fb1`
  （22002B，canonical=实际哈希一致，≠e20 新 zip_sha256）；test
  `c44c9a460d24060cd937a8c196d9a0f5dca76498e4c6d2c98740f69102147a84`；
  回执 SHA `91aacb54caccde41a5e905ed26c91e0e49e81a6168e2f204dbbdbb236dd3415c`。
- 逐成员（zipfile 实读 SHA 全部与 `bcf63bbc` git blob 一致，防打包器
  夹带；前 7 成员与 e20 ZIP 逐字节相同——单变量；ascend 为新字节）：
  - `unpad_draft_extend_output.py`（generic）`f1175d80…d0c0` — 同 e20
  - `unpad_draft_extend_output_amd.py` `1e9dbfb5…4032` — 同 e20
  - `unpad_draft_extend_output_ascend.py` `6d7b7aba…ec12` — **新**
    （3118→5220B，e21r 本体）
  - `unpad_draft_extend_output_enflame.py` `204c8208…9af7` — 同 e20
  - `unpad_draft_extend_output_hygon.py` `031847ed…bda7` — 同 e20
  - `unpad_draft_extend_output_kunlunxin.py` `c29bd06c…dbcf` — 同 e20
  - `unpad_draft_extend_output_metax.py` `2ac05704…44ce` — 同 e20
  - `unpad_draft_extend_output_nvidia.py` `f39143ac…ecdc` — 同 e20
- 预注册门：华为 8/8 正确且无 'ub overflow' 编译错且读数 ≥378（e19r
  水位带 378-507，峰对标 e14 BLOCK=16384 的 442）；任何 ub-overflow
  判 invalid 即证伪，ascend 回滚至 e19r ZIP 的 ascend 字节（e14 整数
  mask 形态）；均值 >331.10 换 TB。

## 2026-09-23 E21R 平台终态：invalid_correctness 7/8——华为 ub-overflow 已消除，转为数值失败

- E21R（submission **20313**，daily_seq 9，created 2026-09-23T12:31:00+08）：
  completed/invalid_correctness，7/8（华为失败），均值 None（无效提交不
  计均值），is_team_best=false——TB 守 e19r 331.10。
- 逐芯（七芯全过，括注平台 selected_file）：天数 389.2154（generic）/
  沐曦 244.1776（_metax）/ 燧原 125.8252（_enflame）/ 海光 659.4556
  （_hygon；唯一 raw errors 条目为 pytest-asyncio PytestDeprecationWarning
  stderr 噪音，passed=true，0 failed_cases，良性）/ 昆仑 33.1278
  （_kunlunxin）/ A 519.3046（_nvidia）/ B 280.1564（_amd）；华为失败
  （`unpad_draft_extend_output_ascend.py` 被平台选中）。
- 华为关键变化（raw_result failed_cases 实读）：e20（20162）的 BiShengHIR
  `ub overflow, requires 3145984 bits while 1572864 bits available` 编译错
  **已消除**——e21r UB-fit 修复在编译层兑现（编译并运行，
  execution_time_ms=32914），但转为数值失败：
  `test_unpad_draft_extend_output[3]`（case_idx 3，npu:0，bf16，容差
  atol/rtol 0.015），Mismatched 1520/164352（0.9%），max abs diff
  4.515625 @ (143,4,41)，max rel diff 631.8447265625 @ (143,3,7)。
- 预注册门核对（b17f47a0 上膛时记录）：'华为 8/8 正确且无 ub-overflow
  且 ≥378（e19r 水位带）'——无 ub-overflow ✓，但华为不正确 ✗（晋级门
  失败），≥378 未达成；'any ub-overflow rolls ascend back to e19r ZIP
  bytes' 触发条件**未发生**（是数值错非 ub-overflow），后续处置（修复
  ascend 数值或回滚）按预注册条款归编排方裁决；重掷轴已关闭（1/1 用尽）。
- 额度：本发后 21/30 remaining（observed_at 12:34:00+08，submission
  20313/daily_seq 9）；三发全部落地后复核 20/30（used 10，observed_at
  12:42:11+08，本次 status JSON 实读）。
- remote_verification=unavailable（FLAGOS_REMOTE_ZIP_HOST 未设，仅远端
  字节未复核，不影响已成功提交）；未出现 sending/uncertain，未重试。
