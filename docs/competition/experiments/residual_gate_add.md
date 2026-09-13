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

```current
task: 73
operator: residual_gate_add
batch: 5
validity: valid
platform: completed(13905,e2,8/8,3.930x;TB s0 3.9518x)
candidate_stage: e2
team_best_stage: s0
team_best_speedup: 3.9518
sealed: no
next: e2=grid封顶 enflame+18%（真实但有限）；e3=封顶+BLOCK4096 双杠杆（T71/T75 双证 4096 对燧原+44~71%）
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
