# Task 73 `residual_gate_add` 实验记录


## 2026-09-13 E2：enflame grid 封顶 24（已发射 13905）

- 依据：skill 已证燧原超发系统缺陷（grid 1e3-1e4 vs 24 SIP，T19-E5/
  T51-E5 +38% 中位）。vendor 行维封顶 24+grid-stride，列维自然块数，
  免除法寻址（运行时向量整除=GCU/昆仑毒点）；BLOCK 随行宽自适应；
  语义与 generic 完全一致（双重舍入保留）。
- source commit：`037a3f4350493bec8f78d2cb197883c63899e09b`；ZIP `e2-037a3f4`，
  SHA-256 `3896d122358cf09af96e8b77937679b9bc5ecbee99deabcc590dc23589ccb17b`；
  release 回执 SHA-256
  `3d41eed4f62dbd89305638606ea563a4e88114d28e3fe5c35308cfa356171701`；
  3 方法 0 失败，generic 25 + enflame 25 launch。
- submission 13905；裁决点=enflame 1.02 是否被抬升（+38% 门 ⇒ ≥1.4），
  若 ≥2.3 即追平榜首同芯、均值 →4.11（#2），叠加窗口可争 #1。


## 2026-09-13 E3 平台终态：8/8 VALID 4.0308x 新 team best（燧原 +92%）

- **enflame 1.017 → 1.9549（+92%！）**，其余七芯不变（vendor 只影响
  燧原，读数逐项对齐 s0）→ 均值 4.0308 新 TB，距榜首 Nectar 4.302
  缩到 0.271（原 0.35）。
- **配方定型（二连证）**：燧原 elementwise = **grid 封顶 24 + BLOCK
  4096**。T73 e2（仅封顶 1024）得 +18%；e3 叠加 4096 得 +92%
  （4096 是主增益，封顶是辅助；与 skill "GCU BLOCK 倾向远大于 GPU"
  及 T71/T75 的 4096 证据一致）。
- 榜首同芯 2.29 vs 我 1.955：差距收到 1.17x，仍差。


## 2026-09-14 E4：flat-1D 燧原配方（候选就绪待发射）

- e1 证明 flat-vs-2D 八芯中性 ⇒ 去掉 2D grid 窄行浪费是免费结构简化；
  叠加配方（BLOCK 4096 + 封顶 24 grid-stride + 免除法 same-shape gate，
  广播 gate 保留 modulo）。gap 0.27 的最后一推。
- source commit：`b4010e80ff717cec968d1e0bd893e3ea58813151`；ZIP `e4-b4010e8`，
  SHA-256 `d73405ed6970cdad60a95d859f9608c06cde92ae2e23cef5b49917f1723c3e94`；
  release 回执 SHA-256
  `9f69f3392246db211a59ad40ba753feffa0aeb37f9f916a85f0b7892c97ae87e`。

## 2026-09-14 E4 修订版（Codex 审查后，候选就绪待发射）

- 审查发现三问题已修：①broadcast 分支保留 e3 capped-2D（向量 %d 无
  燧原先例，flat 中性证据只覆盖 same-shape）；②乘积改元素 dtype 直乘
  （Triton 会把寄存器内 f32 链折叠跳过中间舍入——探针实证只有内存
  往返能强制，偏差 ≤1 ulp 在平台容差内）；③新增跨 tile/第二轮步进/
  广播覆盖（97×1025、D=4095/4096/4097）+ 双舍入 ulp 界测试。
- source commit：`220aa32d18a1c3a4aca829b09b79e2e906471e43`；ZIP `e4-220aa32`，
  SHA-256 `062fdbe8b19c6cee11e9f7320232fea40781fe1c2a8a64b0aa62511fe1d19bcb`；
  release 回执前缀 `aed4ff41`；5 方法 0 失败。

## 2026-09-14 E4 终版（第三轮审查：双舍入契约强制执行）

- 审查反例全部实证（12 分支超容差：fp16 0.0039>0.001 / bf16
  0.0625>0.01 / fp32 3.8e-6>1e-6）——**此前"≤1 ulp 在容差内"的论证
  不成立**（残差抵消放大）。修复：**全部 launch 加
  `enable_fp_fusion=False`**（探针证明完全恢复契约，三 dtype 均回
  eager 的 0）。同时：广播 capped-2d BLOCK 恢复 e3 的 4096 上限
  （修订版曾回退到 1024——审查抓到的性能变量回归）；空维除零守卫；
  消没矩阵进 required tests。
- source commit：`ec213b44fb19f1fb45b5bc005fb79813d68049b9`；ZIP `e4-ec213b4`，
  SHA-256 `695c3242ed6925624fdb4365918f937ab6c31d1f2320613c29e5fce29d7483ca`；
  release 回执 SHA-256
  `c5e8b3d2ae80b55d68b39254888949843a263ac6d14a3374321943392d456bfb`；
  6 方法 0 失败。
- 注意：generic 也变了（fusion-off 影响全部八芯）——不再"只影响燧原"，
  平台回归风险由全芯共担；但契约语义正确性优先。


## 2026-09-14 E4 平台终态：8/8 valid 4.027（低于 TB e3 4.031，fusion-off 无回归）

- 燧原 2.0959（e3 1.955→+7%，flat 路径小幅增益）；天数 +5%；其余
  持平——**fusion-off 对七芯无回归**（重要：契约修复零性能代价）。
- 均值 4.0274 < e3 TB 4.0308（差 0.009，窗口噪声）。TB 保留 e3。
- 榜首已通胀至金狐狸 4.388（差 0.357）；燧原榜首 2.95 vs 我 2.10。

```current
task: 73
operator: residual_gate_add
batch: 5
validity: valid
platform: completed(e11/sub15484,8/8,4.08910938x；TB仍E6 4.0993125x)
candidate_stage: enflame-row-group-development
team_best_stage: e6
team_best_speedup: 4.0993125
sealed: no
next: E7已投且昆仑0.6295未过0.65门，关闭宽度轴；开发燧原行分组新结构，其他芯冻结TB
updated: 2026-09-16
```

## 契约与实现（S0）

- 完整题面：[Task 73](../tasks/batch-5/73-residual_gate_add.md)（2026-09-12 晚新增五题之一）。
- 2D grid avoids gate modulo; product rounds to dtype before add (double rounding contract)。
- 核心计算 Triton，无 fallback；八芯 0.1x；截止 2026-09-17 19:59:59。

## 不可变身份

- source / verification commit：`e6b450fd4beb001255079ade929d3b2de51796d6`（五题同批提交）。
- source SHA-256：`9ec431b99fbfcb016299f1ede9fa370badfc0083b2be50b1a64fb046ed8ef6b4`。
- test SHA-256：`65b50e81bd82507f68a2e7d50463e1e98c61d0b9bc51c32393c6a5a320c027ac`。
- ZIP：`artifacts/competition/residual_gate_add/s0-e6b450f/residual_gate_add.zip`，SHA-256 `942643b4f828362fb5f97e4f9a14894e8986fc5950fe8e656f85f5dd1dce54e2`（单成员 `residual_gate_add.py`）。
- release 回执：`artifacts/competition/batch5-new5-validate-20260913/residual_gate_add/verification.json`，
  SHA-256 `bc29bd54560f16941293e46ea8f7fe2fc4f12452442ecb2f637626008d47b74a`；日志 SHA-256 `b0258bcd84a6a28f7274383004361bf559e819c51c17cd4120a2ead082638b59`。

## 验证状态

- screening 多轮门禁拦下并修复的缺陷已记录于提交说明；
  最终 release：0 失败/错误/skip，非空 kernel launch，NVIDIA 代理范围。
- 所有八芯目标 `target-runtime-unverified`；裁决权在平台。

## 2026-09-13 E1：同形 gate 扁平化（已发射 13872）

- Codex 咨询建议：非广播 gate 改 flat 1D（免 2D grid 窄行浪费，免
  %D），广播保持 2D——纯结构调整，双重舍入语义不变。
- source commit：`dd043007f917f3406566167cc54240df4f719887`；ZIP `e1-dd04300`，
  SHA-256 `2f2d64146c540cca957d13fc06a67a8c2ab6a5523063aad020210f7baaec020f`；
  release 回执 `batch5-r5-20260913/residual_gate_add/verification.json`
  SHA-256 `f301e7c62bae010808c13d6555bd8cd31ff35b46f60a77e6f95a5e33a5c2ae04`；
  3 方法 0 失败。

## 2026-09-13 E1 平台终态：8/8 valid 3.903（订正：扁平化中性，此前归因有误）

**数据订正**（API 真值回查）：
- s0(13764)：tianshu 6.944 / muxi 5.372 / enflame 1.017 / haiguang 6.992 /
  kunlunxin 0.233 / huawei 1.125 / card_a 5.765 / card_b 4.165（均 3.9518）
- e1(13872)：tianshu 6.890 / muxi 4.789 / enflame 1.008 / haiguang 7.243 /
  kunlunxin 0.232 / huawei 1.091 / card_a 5.793 / card_b 4.178（均 3.9031）

此前账本把 s0 记为 enflame 3.99 / kunlun 1.42 / huawei 3.81 等，与 3.95
均值不自洽——系转录错误。**真实结论：扁平 1D 与 2D 在八芯上无显著差异
（±11% 内=窗口噪声），此前"扁平在 GCU/昇腾/昆仑回退"的跨芯规则作废**。

**本条教训（数据完整性纪律）**：逐芯对比必须从 API 回读原值，不得用
凭记忆转写的基线；"某形态在某芯回退 N%"的结论落账前须与 submission 的
gpu_results 逐项核对。

**T73 真实差距结构**（vs Nectar 4.302，总差 0.35）：enflame +1.27
（我 1.02 vs 2.29，最大项）、haiguang +0.55、huawei +0.47、kunlun +0.43、
card_b +0.31、tianshu +0.09；muxi/card_a 我方领先。燧原窗口摆动实证
（T63 跨提交 6.2↔21.7 最强；T73 自身三读 1.017/1.008/1.197 较稳 ⇒
差距更像结构性）。e2 用真实改动（enflame vendor grid 封顶 24+grid-stride）
重掷。

## 2026-09-13 E2 平台终态：8/8 valid 3.930（enflame 封顶 +18%，低于 TB）

- 真值：enflame **1.017→1.197（+17.6%，vendor 被选中）**、kunlun 0.234、
  huawei 1.014、tianshu 6.88 → avg 3.9301 < TB 3.9518。
- 结论：grid 封顶对燧原是真实但有限的正向（+18%；skill 的 +38% 中位
  未完全兑现）；仍差榜首同芯 1.9x（2.29 vs 1.197）。
- **E3 假设（双杠杆叠加）**：封顶 + BLOCK 1024→4096——T71（+44%）/
  T75（+71%）双证 4096 对燧原正向，两杠杆叠加预期 enflame 1.5-1.8，
  均值 →4.0-4.05。

## 2026-09-14 E5 候选就绪：昆仑 no-loop 双路径 vendor（Codex 激进菜单，待发射）

- 依据：T75 同 diffusion 族昆仑 no-loop 3.4x（0.26→0.90）；T73 昆仑
  现 0.234 vs c2flow 1.264。载体 = 双路径都去 runtime 循环：
  broadcast 保持 2D (row, col-block) 网格、同形路径恰好
  cdiv(n,1024) 个 program 直线射出；BLOCK=1024、双重舍入契约
  enable_fp_fusion=False 字节保留。Codex v3 限制：广播路径本无循环
  不包装成实验，同形路径为独立重评（T71 反例后重新登记）。
- source / verification commit：`ffa1d7fc…`；ZIP `e5-ffa1d7f`，
  SHA-256 `704c351d92c87560ac03b970ee5b8170b63609b43e101a0e3c69b5fe27de4ee5`。
- release 回执 `batch5-submit-20260914-finale/residual_gate_add-t73e5/verification.json`
  SHA-256 见 artifacts（--proxy-vendor kunlunxin：vendor 41 launch
  全矩阵 0F0E0S）。
- 预注册晋级门：**昆仑 ≥ 0.65 且均值 ≥ 4.07**；失败则 T73 昆仑后续
  链整体取消。

## 2026-09-14 E5 平台终态：8/8 VALID 4.0893x 新TB（submission 14766）

- **昆仑 vendor 被选中、passed 0.4904**（generic 循环形态 0.234-0.237
  → **2.1x**）——no-loop 机制在 T73 复制成功（同 diffusion 族），
  但未到 0.65 门；均值 4.0893 ≥ 4.07 且创新高，TB = e5。
- 其余芯窗口内（天数 6.88/6.92、沐曦 4.84/4.79、燧原 2.10/2.10、
  海光 7.24/7.11、华为 1.12/1.13、A 5.70/5.75、B 4.34/4.18）。
- 榜首 c2flow 4.9749，差 0.886。剩余昆仑缺口（0.49 vs 1.26）与本
  族 T75 的 0.90 之间还有空间，但今晚不再追加（健康门/额度纪律）。

## 2026-09-14 E6 候选就绪：昆仑 flat BLOCK 1024→4096（待发射）

- e5 no-loop 已拿 2.1x（0.234→0.490）；昆仑唯一有效调参轴=BLOCK，
  4096 为本 skeleton 首个加宽档。代理 kunlunxin vendor 82 launch
  0F0E0S。门：**昆仑 ≥ 0.65**；未过宽度轴停。
- source / verification commit：`343f9d57…`；ZIP `e6-343f9d5`，
  SHA-256 `eff880664d2f7fac6223bbb6e475b7031d94402cf6b1a5c4e177a09c6227a034`。

## 2026-09-14 E6 平台终态：8/8 VALID 4.0993x 新TB（submission 14838）

- **昆仑 vendor passed 0.6033**（e5 的 0.4904 → +23%）——4096 档宽度
  轴兑现，距 0.65 门仅 0.047。no-loop + 宽度曲线（1024→4096）双正，
  下一档 8192 待明日信号。均值 4.0993 > e5 的 4.0893，新 TB = e6。
- 其余芯窗口内。榜首 c2flow 4.9749，差 0.876。

## 2026-09-14 E7 候选就绪：昆仑 flat BLOCK 4096→8192（宽度曲线续推，待发射）

- e5 no-loop 2.1x → e6 4096 +23% 双正曲线的下一档。代理 kunlunxin
  vendor 全矩阵 0F0E0S。门：**昆仑 ≥ 0.65**；未过则宽度轴停在 4096。
- source / verification commit：`934120a…`；ZIP `e7-934120a`，
  SHA-256 `ba4ac2a777af4b04d264ec061b857493f0abff4fc478b7207fda07089a709b13`。
- release 回执 `batch5-submit-20260914-finale2/residual_gate_add-t73e7/verification.json`
  SHA-256 `c26a184e0468dfee65075d08bcbd0be19c5610476f4f2f6b99e80054d6fb15d2`。

## 2026-09-15 23:20 夜间 warps 批扫终态：E11 平:燧原2.08≈TB2.10;warps轴无效

- 单变量=燧原 vendor 钉 num_warps=4（夜间批 T71/T73/T68/T62 四题）。
  批量结论：燧原 warps 为逐题特性，非统一旋钮；本批 2 负 2 平，
  轴关闭。TB 各自保持。

## 2026-09-16 广播 gate 外提：机制成立，筛选未过门，不晋级

- 从TB E6 `343f9d57` 冻结全部成员，只将Enflame广播路径的cols/mask/gate load移到row循环前；保留cap24、BLOCK4096及原默认warps、fp16/bf16乘法中间舍入、fp32加法、fusion=False、same-shape路径和所有wrapper语义。不是历史E11 warps4轴；本轮没有改主树。
- 成熟同芯来源为前批PR #50循环外weight/bias驻留；exact TB的6形状TTGIR/PTX确认gate载入未外提，候选6/6又确认load在rowloop之前，机制确实改变。GCU编译器也有LICM，NVIDIA现象不能代替目标GCU代码生成证据。
- **9/9正确性、0F0E0S**，generic/Enflame/昆仑每源89实际launch，覆盖broadcast、same-shape、尾块、精度抵消、empty、alias与特殊浮点。32桶×5轮wrapper AB/BA：18affected中位 **1.003498x**、几何均值 **1.029655x**，均未达1.05；14controls保留、全桶最差0.984615，zero spill。**NO-GO**，不挑大shape缩域，不晋级、不打ZIP、不提交平台，TB仍E6 4.0993125。
- 候选Enflame SHA-256 `9127fb02cdbf5d2c86835be962f9c8d614afb103f7d3aff08a15ab6119c2259a`，test `aeaf4c6322f88de34825ef2d2c1fdffbc3001145d9ac71112c4ac136343f245b`。输入、完整32桶与24份IR在 `artifacts/competition/t73-gate-hoist-screening-20260916/`；screening.json SHA-256 `885a799ded6ce8c772e560c82baea291c1e1df73467e7226fbff955103914db4`；日志 `c698970074b23886595442dfedff975e11c2dc860b80570ad7dab0920c9ff56c`；analysis.json含逐IR指针来源/行号审查。
- 前两次取证脚本中止，不是数值失败：rows=1时gate与residual同shape，wrapper实际选flat，但runner误抓capped2d。早期“热缓存返回None”诊断已更正；attempt1/attempt2各9桶及9/9原件保留。最终按真实shape穷举32桶（22capped2d/10flat），只修资源抓取，source/test/门/样本不变；完整最终轮独立裁决，不拼接前两轮样本。
- 最终远端 `/tmp/flagos-t73-hoist-final.XLAryQ`、PID388539、timeout600、EXIT0，输入/全部输出双端验签，GPU已释放。最终script SHA-256 `2a209ce19ce392d165a0fe08fcc6c01a36dcc60e7a486e66e69a0ec90d5ac99f`，tar `20102fb55ad84ade5856a3f40b753f1ed6fa7a1d2fc587978d16e77dade0631d`。

## 2026-09-16 23:12 历史提交对账与并行候选

- 实时查分与本地submitted intent交叉核实：E7早已于2026-09-14T23:42:41提交，submission **14856**，8/8、均分 **4.08110938**、昆仑 **0.6295**，未超E6均值且未达0.65门。intent `8972fdf47a1120ee91329a5e1d52ae03` 的commit934120a795ad2f6f15b12159515cdd15a48ee777、ZIP SHA ba4ac2a777af4b04d264ec061b857493f0abff4fc478b7207fda07089a709b13与该record file_url SHA完全匹配；旧“待发射”段仅保留历史，不再视为队列。
- E11/sub15484于2026-09-15T23:39:58，8/8、4.08910938，燧原2.0805、昆仑0.631625；未超E6 **4.0993125**。8192宽度和燧原warps轴不重复。新候选仅研究Enflame广播行分组，具体门槛在执行前冻结，未通过发布门不提交。
- 本次状态观察 `2026-09-16T23:11:43.998592+08:00`，账号额度 **15/30**。状态快照 `artifacts/competition/pair-grouped-platform-20260916/t73-before-parallel-status.json` SHA-256 `06b19907fde9bd38a9c228e3f1ae4f41299e6c53333e6ecb61b46b22156da637`。
