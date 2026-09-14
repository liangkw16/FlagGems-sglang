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
platform: completed(14379,e4,8/8,4.027x;TB e3 4.0308x)
candidate_stage: e4
team_best_stage: e3
team_best_speedup: 4.03084375
sealed: no
next: e4 fusion-off 无回归（契约修复零代价）；TB e3 守；榜首金狐狸 4.388 差 0.357 需多芯叠加
updated: 2026-09-13
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
