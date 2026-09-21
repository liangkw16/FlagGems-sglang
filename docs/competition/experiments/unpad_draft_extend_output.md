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
