# Task 90 `sigmoid_gate_mul_broadcast` 实验记录

```current
task: 90
operator: sigmoid_gate_mul_broadcast
batch: 6
validity: valid
platform: e6(20163)invalid_threshold:燧原0.033(flat流式在GCU崩,远低于2.0保留门);kunl0.118贴门;TB 2.5247(e4)守
candidate_stage: e7
team_best_stage: e4
team_best_speedup: 2.5247
sealed: no
next: e7评审2轮通过(唯一发现=py_compile口径,裁定门禁仅作用.py、markdown账本不在编译范围;代码零评审发现,字节=f43cf87e)待release代理验证+e7-ZIP上膛;门=燧原≥2.0保留/≥2.5进场带,均值>2.5247换TB,任一其余芯-5%判负,回退档=纯1D BLOCK=W标量row;e6字节绝不再入包
updated: 2026-09-23
```

## 2026-09-20 S0/E1 首发记录

- 开发+验证+codex-review 门后 00:12-00:27 发射（详见
  `artifacts/competition/nx-s0-20260920/`）。

## 2026-09-20 E1 平台终态：燧原 +12% 但均值持平，TB 保 2.472

- submission **18315** completed/valid，8/8，均值 2.433 < TB 2.472（保 s0）。
  燧原 0.8→0.9（+12%）；**海光 4.2→3.8（-10%，warps16 在此题为负——与
  T86 正例互补，warps 分化按 op 而非只按芯）**；华为 persistent 无增益。

## 2026-09-20 E2 平台终态：hygon 回退兑现，TB 2.495

- submission **18336** completed/valid，8/8，均值 **2.495 微幅新 TB**（vs
  2.472）。海光 3.8→4.3（回退 warps16 后恢复）；燧原 0.9 持平。距榜首
  3.21 差 29%。

## 2026-09-20 E3 平台终态：微幅新 TB 2.517

- submission **18352** completed/valid，8/8，均值 **2.517 微幅新 TB**（+0.9%）。
  天数 4.3→4.7；A 3.0 持平（warps8 在 sgmb 的 generic 路径收益小于 relu2
  ——同配方跨 op 部分迁移）。

## 2026-09-21 夜 e4 候选就绪（午夜第 4 弹）

- 结构（`f3b8cad1`）：`_ascend` persistent vendor（NVC 封顶）——榜首华为
  2.1 vs 我方 1.2。预注册门：华为 ≥1.5；均值 >2.517 换 TB。
- 五元组：commit `f3b8cad1`；ZIP `e4-f3b8cad` SHA
  `66e2793d107769a3c02e7884d3b7f58562a5a78e20b13d8f686c24f36866e0ff`；
  回执 `ready2-wave-20260921/sigmoid_gate_mul_broadcast/`（四路径）。

## 2026-09-21 00:13 E4 平台终态：2.5247 新高（+0.007）

- submission completed/valid。逐芯：天数 4.69（≈榜首 4.71）/ 沐曦 2.51 /
  燧原 **0.77** / 海光 4.26 / 昆仑 0.75 / 华为 1.25 / A 3.05 / B 2.91。
- 榜首 3.206（金狐狸）Δ -0.68 集中四洞：燧原 -1.74、沐曦 -1.03、
  B -0.97、华为 -0.89。天数/昆仑已追平——**vendor 轴（燧原/沐曦/
  B=card_b 无 vendor 位）是下一弹**。

## 2026-09-21 E5 候选就绪并发射（燧原 vendor 去 24-SIP 行封顶）

- 结构（`76982f26`）：_enflame 回退 generic 字节——24-SIP 网格封顶使
  one-program-per-row 只有 24 个程序（读数 0.77 vs 场带 2.5-3.4），
  GCU 需要程序级并行而非 SIP 数行循环。预注册门：燧原 ≥2.0；均值
  >2.5247 换 TB。
- 五元组：commit `76982f269271ac4c0def557403940ddfe7610bd6`；ZIP `e5-76982f2` SHA
  `cd0bd7a66201478b16c750df6564d161d397b9a134b9de4ff2fe4e6dc29b7e78`；
  test `449328d06e44d64f3add4b2ee8c36db6c770140e7dedea3ef78d078ba5874d2d`；
  回执 `top1day-20260921/sigmoid_gate_mul_broadcast/` SHA
  `fc831bff01a55d3fbca07d36ea28267b86921c8af8eba2a65b7fa8a68df7956c`。

## 2026-09-21 00:37 E5 平台终态：燧原 0.63 判负，行形式在 GCU 双向失败

- submission completed/valid，均值 2.5205 ≈ e4（2.5247，保 e4 TB）。
  燧原 0.77→**0.63**：24-SIP 封顶（0.77）与全网格 generic 字节（0.63）
  均远低于场带 2.5-3.4——行形式+HDIM 静态展开在 GCU 病理，程序数多寡
  两个方向都更差。**燧原轴关闭**（需第三方结构情报）。

## 2026-09-22 e6 候选实现（第 1 轮：燧原重开，flat 流式形态）

- 结构（本轮 commit）：`_enflame` 弃行形式，移植 T89 relu2 终态已验证
  配方——numel 一维 flat grid-stride + `min(cdiv(numel,65536),12)` 12-CTA
  封顶 + BLOCK 65536（relu2 阶梯宽度顶：2.61@16384→3.3@32768→
  **3.9@65536**→3.86@131072，12-CTA 终态 3.94）+ num_stages 3 + warps
  不钉。gate 经 `offs // HDIM`（constexpr 整除）gather 且带同一 m 掩码
  （OOB 防线）；int32 寻址域断言 `numel < 2^31-65536`（含一个尾块余量，
  防 masked lane 回绕）；vendor 内 assert x/gate row-major 连续（relu2
  先例 relu2.py:30）。e5 关轴时缺的第三方结构情报 = 同芯 relu2 已验证
  宽块阶梯 + 模板字节 8/8 先例。
- 回归：`test_flat_block_boundary_gate_gather` 新增并列入
  RELEASE_REQUIRED_TESTS——65536 整块无尾 / 块边界恰在行边界 / 块边界
  跨行（67584）/ 16 块 >12 CTA grid-stride 二趟 / hdim=1 逐元 gate；行间
  distinct gate（sigmoid 展布 0.047..0.953）使 gather 错位必超 2e-2 容差。
- 预注册门（e5 原门保留）：燧原 ≥2.0 保留 / ≥2.5 进场带（场带 2.5-3.39，
  EvokeAgent 3.2440）；均值 >2.5247 才换 TB；宽度阶梯 32768 为回退档。
  折扣因子：relu2 无 sigmoid+gather 的额外 ALU 未实测；expectedAvgGain
  0.1 = 进场带档 +0.22×约四成成功率（非 EvokeAgent 满配 +0.33）。
- 待办：release 代理验证 + e6-<commit> ZIP 后上膛；同题落选的 _metax
  flat 候选（17/18 队 ≥2.88 场带先验，天花板 +0.14 均值）与本候选文件
  互不重叠，e6 回执后可作沐曦轴下一发。

## 2026-09-22 e6 第 2 轮：评审修复（P1 int32 域 / P2 契约收窄 / P3 标注）

- **P1（必修，已修）**：第 1 轮断言 `numel < 2^31-65536` 只保证尾 lane
  （offs=base+65535）不回绕，未覆盖 grid-stride 归纳步进 base+12*65536。
  本会话独立 int32 回绕扫描复现评审结论：安全上界精确为
  `numel ≤ 2^31-786432 = 2146697216`（`safe(2146697216)=True`、
  `safe(2146697217)=False` 边界对验证）；评审示例 262128x8192=
  2147352576 落在不安全窗（回绕负 base 使 `offs<numel` 全真 → 负地址
  OOB）。解析证明：visited base < numel，步进至多 786432，故
  `base+step ≤ numel-1+786432 ≤ 2^31-1`。断言收紧为
  `numel < 2^31 - 12*65536`（vendor 内注释同步写明归纳覆盖）。
- **P2（已修）**：第 1 轮 `x.is_contiguous()` 硬断言把 generic 契约
  （`x.stride(1)==1` 行间隙 x 走 xs0，ops/:39）收窄成 AssertionError。
  改为：gate 连续断言保留（与 generic 同），行间隙 x 走
  `x = x.contiguous()` 布局拷贝（gating 乘法仍在 Triton kernel 内，
  反作弊合规）；新增回归 `test_row_gated_strided_x`
  （base[::2] 行间隙，generic 与 _enflame 拷贝路径同测）。
- **P3a（同批修）**：(33,2048) 注释失实——65536%2048==0 是行对齐边界；
  换 (33,2047)（65536=32*2047+32，真 mid-row）并修正注释。
- **P3b（同批修）**：头注释/launch 注释的 "relu2 final recipe/geometry"
  改为 "宽度顶档 65536"——relu2 落地终态是 131072 档（relu2.py:38-39，
  3.94 出自该形态），e6 选 65536 依据是宽度峰 3.9@65536。
- **附带存量风险（不在本候选文件集，未改字节）**：relu2.py 同型循环
  （:15-24）无任何 numel 断言，同病；已在此登记，T89 下一轮 vendor
  改动时一并补 `numel < 2^31 - 12*65536`（其 BLOCK=131072、grid≤12，
  步进同 12*BLOCK 量级，断言需按其几何取 12*131072）。
- e6 候选身份更新为本轮 commit（第 1 轮 `9d080b0d` 字节作废，未上过
  release/ZIP）；预注册门不变。

## 2026-09-22 e6 上膛（round-2 commit 回执绿 + ZIP 验签）

- 回执（`artifacts/competition/day5prep-20260921/sigmoid_gate_mul_broadcast-wf/`）：
  release 模式 NVIDIA 代理（RTX 5070 Ti，torch 2.13.0+cu130 / triton 3.7.1），
  `run` exit 0；RELEASE_REQUIRED_TESTS 3 用例全过（`test_shapes_and_saturation` /
  `test_flat_block_boundary_gate_gather` / `test_row_gated_strided_x`），
  0 失败 / 0 错误 / 0 skip / 0 xfail；62 case ×4 模块
  （generic/ascend/enflame/hygon）；4 源各 12 次 kernel launch；
  三 vendor 路径 proxy-executed，目标芯 target-runtime-unverified 保守标注。
  回执 SHA `3b721d286977c0004d6350eecd799ef03bdbc20619b52e5e6ea194dd64c3d908`，
  log SHA `898678c6030d27de839254e911020b2924eee2cd89d840e08d3200a1e327f1a9`。
- 五元组：source commit `6aaa394f33a522047bff5f54023c66de42e0175c`（=HEAD，
  round-2 修复后单 commit，kernel 字节=9d080b0d 第 1 轮相同、注释/断言/
  测试更新）；verification commit 同上；test
  `tests/test_sigmoid_gate_mul_broadcast.py` SHA
  `b0f26e58be5af427ae402f03992d21de0384d6005c4942ecee1ad787156e29f0`；
  ZIP `e6-6aaa394/sigmoid_gate_mul_broadcast.zip` SHA
  `320f38345988e79334dba051f97351aeae9698e10c497e46266bb2f5bf5a5955`
  （9434 B，新字节 vs e5 `cd0bd7a6…`，平台元组 zip_sha256 去重无冲突）；
  回执目录 `day5prep-20260921/sigmoid_gate_mul_broadcast-wf/`。发射
  preflight 的 verification_commit 必须等于本回执的 `6aaa394f…`。
- ZIP 成员名单（zipfile 实际 namelist 与预期 4 成员精确相等、无夹带；
  逐成员字节与 `6aaa394f` git blob 比对一致；`unzip -t` 无错）：
  `sigmoid_gate_mul_broadcast.py`（`7b653c1f`，← `src/flaggems_sglang/ops/`）/
  `sigmoid_gate_mul_broadcast_ascend.py`（`32e9a26b`，← `_ascend/ops/`）/
  `sigmoid_gate_mul_broadcast_enflame.py`（`090d7d46`，← `_enflame/ops/`）/
  `sigmoid_gate_mul_broadcast_hygon.py`（`c9e3756e`，← `_hygon/ops/`）。
- 预注册门（e6 两轮一致，发射前锁定）：燧原 ≥2.0 保留 / ≥2.5 进场带
  （场带 2.5-3.39，EvokeAgent 3.2440）；均值 >2.5247 才换 TB；宽度阶梯
  32768 为回退档；折扣因子见第 1 轮段（expectedAvgGain 0.1）。

## 2026-09-23 e7 候选实现（第 1 轮：燧原行块流式形态，_enflame 整成员替换）

- 逐芯差距复核（climb-loop.json `s2t1op090` 快照，本轮会话实读）：我方
  enflame 0.77473333（rank 19）vs EvokeAgent 3.39373333（Δ-2.62，全题最大
  单洞）；双峰确认：仅 4/20 队 ≥2.5（金狐狸 2.50533333/varphi 2.76406667/
  c2flow 2.97546667/EvokeAgent 3.394），其余 16 队 0.771-1.504=结构门
  （快照实测带宽，较启动假设的 0.77-1.24 略宽，结构门结论不变）。
- 根因链（已证伪形态收口）：①行形式+运行时 xs0+逐行标量 gate=e4 0.77
  （24-SIP 封顶）/e5 0.63（全网格）——chip-rulesets.md:9「stride 必须编译期
  互整除才走 DMA，运行时 stride 传参=DMA 判定失败、tile 缩 4 倍」直接解释；
  ②flat 65536+逐元素 offs//HDIM 除法 gather=e6 0.033（sub 20163
  invalid_threshold）。同芯 T89 relu2 已着陆配方（relu2.py：numel flat
  grid-stride、12-CTA、num_stages 3、warps 不钉）平台 3.94——与 e6 的唯一
  结构差=逐元素除法+per-element gather（relu2 亦带掩码与 tl.where ALU，
  证明掩码/ALU 非杀手），故 e6 崩因锁定除法 gather。
- 结构（本轮 commit）：`_enflame` 整成员替换为 [RB,W] 行块瓦片——W=HDIM
  最大 2 幂因子（`hdim & -hdim` 封顶 65536）、RB*W=65536 对齐 relu2 宽度带；
  寻址 `(base_row+arange(RB))[:,None]*HDIM+(h0+arange(W))[None,:]`，HDIM
  constexpr→编译期整除→DMA 通路，列向 W|HDIM 恒成立→零列掩码、列循环
  `range(0,HDIM,W)` 编译期计数（奇数维 1023/2047 次迭代不静态展开）；gate
  按行块一次 [RB] 向量载入+sigmoid+[RB,1] 广播（无逐元素 gather）；行尾
  掩码仅末行块激活；launch 沿 T89 配方：行块 grid-stride + gcu300 12-CTA
  封顶 + num_stages 3 + warps 不钉。
- int32 域（e6 P1 纪律沿用）：断言 `(rows+RB)*hdim < 2^31`——覆盖行块归纳
  与 masked 尾 lane 的全部计算地址（最大计算地址 =nblocks*RB*HDIM-1≤
  (rows+RB)*hdim-1，全加法非负→不可能回绕成负索引过 `row_offs<rows` 检查）；
  本轮纯 Python 循环结构仿真证明 17 个测试 shape 逐元素恰写一次、grid-stride
  分发与尾行掩码正确。
- 契约：gate 连续断言保留；行间隙 x（stride(1)==1, stride(0)>hdim）沿用
  e6 P2 contiguous 拷贝先例（gating 乘法仍在 Triton kernel 内）。
- 回归：新增 `test_rowblock_tile_boundary`（rows=RB-1/RB/RB+1、769 行块
  >12 CTA 二趟、5120=5×1024/7168=7×1024 多列块、96=3×32 亚 512 宽带、
  65536 W 封顶 RB=1；逐行 distinct gate sigmoid 展布 0.047..0.953）并列入
  RELEASE_REQUIRED_TESTS；`test_flat_block_boundary_gate_gather` 矩阵保留、
  注释改写为 e7 行块语义（e6 flat 语义已不适用）；`test_row_gated_strided_x`
  保留。
- 硬约束（单变量）：generic/_ascend/_hygon 三成员冻结 e4 字节；e6 字节
  （0.033<0.1）绝不再入包——本候选即 _enflame 整成员替换。
- 预注册门（e6 门沿用+连坐条款）：燧原 ≥2.0 保留 / ≥2.5 进场带（金狐狸
  2.51 为场带下沿）；均值 >2.5247 换 TB；任一其余芯 -5% 判负；回退档=纯 1D
  BLOCK=W 每迭代标量 row=base//HDIM（标量除法/迭代）+标量 gate。
- 主要不确定性：GCU 对 2D 瓦片 lowering 未实测（无燧原主机，NVIDIA 代理仅
  数值门，target-runtime-unverified），回退档纯 1D 标量 gate 形态兜底；
  W<512 的奇数维退化为 [RB,1] 正确但慢（基准维 2048/4096/5120/7168 均
  W≥1024 不受影响）；昆仑同字节读数 0.7476→0.118 贴门为平台方差
  （e4/e6 三共同成员字节逐一相同已核实），任何提交均有 invalid_threshold
  连坐风险，非本候选可控。
- 待办：评审通过后 release 代理验证 + `e7-<commit>` ZIP 验签上膛。

## 2026-09-23 e7 第 2 轮：评审门禁口径裁定（代码零发现，候选字节不变）

- 第 1 轮唯一评审发现：py_compile 失败——评审把命令作用到全部触碰文件，
  账本 line 13 `≥`(U+2265)、README line 7 `（`(U+FF08) SyntaxError。本轮
  会话原样复现（同文件同行同字符，`python3 -m py_compile` 实跑）；两个
  .py 触碰文件 exit 0 通过。
- 升级裁定（编排方）：py_compile 门禁只作用于触碰的 Python 源码文件；
  markdown 账本/README 是记录文档，不在编译范围；为过 Python 编译重写冻结
  历史账本会破坏 `gen_experiment_index.py` 的 ```current 解析契约，禁止。
  该评审条按此口径驳回；.py 层面 round-1 实现零评审发现。
- 候选身份不变：代码/测试字节 = round-1 commit `f43cf87e`（op sha256
  `a4f8df21fdf86287eb29cfe50fe6bab63c96ec470f735cea75b36f0eea5df63a`、
  test sha256 `e46a497908c440ecc1201baa0d91ea753c84b37a30e3dea6e0481d36a10c6313`，
  本轮会话逐一核对工作树与该 commit git blob 逐字节相同）；本轮 commit
  仅账本/README/INDEX 记录，四个 ZIP 源成员路径字节在两 commit 间不变。
- 后续不变：release 代理验证 + `e7-<commit>` ZIP 上膛（source commit 取
  载有上述 .py 字节的 commit，`f43cf87e` 与本轮 commit 等价）；预注册门
  与回退档同第 1 轮（燧原 ≥2.0 保留 / ≥2.5 进场带；均值 >2.5247 换 TB；
  任一其余芯 -5% 判负；回退档=纯 1D BLOCK=W 标量 row）。
