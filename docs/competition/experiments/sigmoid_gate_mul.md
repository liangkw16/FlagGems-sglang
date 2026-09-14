# Task 75 `sigmoid_gate_mul` 实验记录


## 2026-09-13 E2 平台终态：8/8 VALID 2.6065x 新 team best（燧原 +113%）

- **enflame 0.827 → 1.7606（+113%！）**；tianshu 经 1024 vendor 恢复
  （4.191→4.721，s0 4.769）；其余稳定。均值 2.6065 新 TB，距榜首
  Albedo 3.054 缩到 0.45（原 0.52）。
- **配方二连证**：燧原 elementwise = grid 封顶 24 + BLOCK 4096
  （T73 e3 +92% / T75 e2 +113%）。含天数分芯 vendor（1024）的
  "双赢组合"打法成立。

```current
task: 75
operator: sigmoid_gate_mul
batch: 5
validity: valid
platform: completed(14598,e4,8/8,2.739325x 新TB)
candidate_stage: e4
team_best_stage: e4
team_best_speedup: 2.739325
sealed: no
next: e4 8192燧原2.16(+23%)新TB 2.739;宽度轴封顶;剩余燧原缺口需形态情报
updated: 2026-09-14
```

## 契约与实现（S0）

- 完整题面：[Task 75](../tasks/batch-5/75-sigmoid_gate_mul.md)（2026-09-12 晚新增五题之一）。
- flat 1024-lane x*sigmoid(gate); SGLang triton_sigmoid_gate_mul form。
- 核心计算 Triton，无 fallback；八芯 0.1x；截止 2026-09-17 19:59:59。

## 不可变身份

- source / verification commit：`e6b450fd4beb001255079ade929d3b2de51796d6`（五题同批提交）。
- source SHA-256：`df083ec35d96b1f3d08a7db4fa4f051d5fee6e747ab30bfe068276634c7f06af`。
- test SHA-256：`1f8d9b035c8706c9c98f6c186924a3d158f4ae40d5c5dbdb203853635cdb481a`。
- ZIP：`artifacts/competition/sigmoid_gate_mul/s0-e6b450f/sigmoid_gate_mul.zip`，SHA-256 `f608dbaf766c96df37e60f5d0c6a6a7ef89ffcd789f617b23165c479eee265ff`（单成员 `sigmoid_gate_mul.py`）。
- release 回执：`artifacts/competition/batch5-new5-validate-20260913/sigmoid_gate_mul/verification.json`，
  SHA-256 `93ea7bbb4a56b52982b89d21447b4afe9529bad5756c9d38af48d6c23aeae1d5`；日志 SHA-256 `a8f10c821397f48f969b52b395b846fe627e6466eb2a83ae7d80eda01914a24b`。

## 验证状态

- screening 多轮门禁拦下并修复的缺陷已记录于提交说明；
  最终 release：0 失败/错误/skip，非空 kernel launch，NVIDIA 代理范围。
- 所有八芯目标 `target-runtime-unverified`；裁决权在平台。

## 2026-09-13 E1：BLOCK 1024→4096（13878）8/8 valid 2.511（API 真值订正）

**真值**：s0(13767) enflame 0.827 / kunlun 0.295 / huawei 1.028 / tianshu 4.769
（avg 2.5382）；e1(13878) enflame **1.417（+71%）** / kunlun 0.262 /
huawei 1.369 / tianshu 4.191（avg 2.5107）。

**订正结论**：4096 对燧原 **+71%**（与此前"回退"记载相反——同样是
转录错误）；但天数 -12%（4.769→4.191）吃掉净收益，均略低于 TB。
e2 假设：**燧原再推 8192 + 昆仑 no-loop 形态**（T61 no-loop 读 1.29
vs 本题 0.26）。

## 2026-09-14 E3 候选就绪：昆仑 no-loop 探针（Codex top1 队列 3 号槽，待发射）

- 逐芯榜单修正认知：T75 昆仑其余各队 0.78-0.86，我方 0.263；T61
  "no-loop 已证"前提被 Codex 证伪（T61 generic 含 grid-stride）——
  本发是降级后的未验证假设探针，单变量=去 runtime 循环（一 program
  一 tile，BLOCK=4096、数学与 fp32 契约字节不变）。
- source commit：`185170f1…`；ZIP `e3-185170f`，4 members，SHA-256
  `09293f3a1dde27b21efac65715803d6d74fa00da337bad5596d52394fa880321`。
- release 回执 `batch5-submit-20260914/sigmoid_gate_mul/verification.json`
  SHA-256 `765dc87d3557c7a36559b0d7c856e18274072853042c27447bf842a1dca18b7a`
  （--proxy-vendor kunlunxin：vendor 17 次真实 launch 全矩阵 0F0E0S）。
- 预注册晋级门：**昆仑 ≥ 0.50**；<0.35 则暂停 no-loop 复制链
  （T71/T73 不再跟发）。

## 2026-09-14 E3 平台终态：8/8 VALID 2.696x 新 TB（submission 14570）——no-loop 假设验证

- **昆仑 vendor 被选中、passed 0.8994**（generic 循环形态 0.263 →
  3.4x）——no-loop 探针兑现，"runtime 循环形态是昆仑 streaming 低
  读数主因"升级为单题实证；队列 6/7 号槽（T71/T73 昆仑）解锁。
- 逐芯：天数 4.7862 / 沐曦 2.6003 / 燧原 1.7565 / 海光 4.0185 /
  **昆仑 0.8994** / 华为 1.3183 / A 3.2262 / B 2.9627 → 均值 2.696
  （前 TB 2.607）。榜首 GuanghuLab 3.688，差 0.99。
- 下一轴：燧原 8192（队列 9 号，leader 7.84 证明可达）与昆仑带宽
  剩余差距待逐芯榜单刷新后评估。

## 2026-09-14 E4 候选就绪：燧原 BLOCK 4096→8192（Codex 队列 9 号，待发射）

- 单变量：配方宽度轴推到 8192（该轴在本芯单调为正：1024→4096 曾
  +71%）；leader GuanghuLab 燧原 7.84 vs 我方 1.76 证明余量存在。
  cap 保持 24。
- source / verification commit：`e49a7a5…`；ZIP `e4-e49a7a5`，
  SHA-256 `0afa3136ee787eb5a92336e6e94337fe5b4ced1a8ac3187b2837376b6f65a467`。
- release 回执 `batch5-submit-20260914/sigmoid_gate_mul_e4/verification.json`
  SHA-256 `6f90a0d96c6a4ac92d05a51d3656ca6133e145e1dc44e668129b46a9a87ca663`
  （--proxy-vendor enflame：vendor 17 launch 0F0E0S）。
- 预注册晋级门：**燧原 ≥ 2.2**；无增益则宽度轴封顶（不平推 T71/T73）。

## 2026-09-14 E4 平台终态：8/8 VALID 2.739x 新TB（submission 14598）——8192 边际为正但未过门

- **燧原 2.1562**（4096 的 1.7565 → +23%）：宽度轴单调性延续但边际
  递减，差预注册门（≥2.2）一线——按纪律**不平推 T71/T73 的 8192**，
  宽度轴就此封顶。均值 2.739 > e3 的 2.696，TB 更新为 e4。
- 逐芯：天数 4.7794 / 沐曦 2.6066 / **燧原 2.1562** / 海光 3.9015 /
  昆仑 0.9003 / 华为 1.3999 / A 3.258 / B 2.9127。榜首 GuanghuLab
  3.688，差 0.95。
- 剩余燧原缺口（2.16 vs 7.84）已非宽度轴可及，需他队形态情报或
  燧原侧 profiling；今日收口守 TB。
