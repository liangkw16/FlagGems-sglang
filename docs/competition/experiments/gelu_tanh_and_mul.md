# Task 71 `gelu_tanh_and_mul` 实验记录


## 2026-09-13 E3 平台终态：8/8 VALID 2.6748x 新 team best（配方第三证）

- **enflame 1.484 → 1.7434（+17%）**；其余七芯持平/微升 → 均值
  2.6748 新 TB，距榜首 c2flow 3.44 缩到 0.77（原 0.81）。
- **配方分解定理（三题数据交叉）**：
  - BLOCK 1024→4096：燧原 **+44~71%**（T71 e1 / T75 e1）
  - 再叠加 grid 封顶 24：**+17~18%**（T71 e3 / T73 e2 均为仅封顶）
  - 从 1024 直上 4096+封顶：**+92~113%**（T73 e3 / T75 e2）
  - **不适用 scatter**（T61 e8:-7%）
- 三新 TB 收官：T73 4.031 / T75 2.607 / T71 2.675。

```current
task: 71
operator: gelu_tanh_and_mul
batch: 5
validity: valid
platform: completed(13917,e3,8/8,2.674825x team best)
candidate_stage: e3
team_best_stage: e3
team_best_speedup: 2.674825
sealed: no
next: e3 新 TB 2.675（配方第三证:封顶+4096 分解定理成立）；距榜首 0.77；明日复刻 T74/T63/T64/T68/T66/T67/T62
updated: 2026-09-13
```

## 契约与实现（S0）

- 完整题面：[Task 71](../tasks/batch-5/71-gelu_tanh_and_mul.md)（2026-09-12 晚新增五题之一）。
- T29 chassis; tanh via exp identity (no tl.math); fp32 compute。
- 核心计算 Triton，无 fallback；八芯 0.1x；截止 2026-09-17 19:59:59。

## 不可变身份

- source / verification commit：`e6b450fd4beb001255079ade929d3b2de51796d6`（五题同批提交）。
- source SHA-256：`19bc8de0beb978bda557dc97fc9600fa802f21a473516ab36e53e5b6abf73c8f`。
- test SHA-256：`1b6d62fb4b910d5c400640565f7194346fef5cc9b3e1f8a267aeae55bc2f725f`。
- ZIP：`artifacts/competition/gelu_tanh_and_mul/s0-e6b450f/gelu_tanh_and_mul.zip`，SHA-256 `b123f0114f244e0d7f0aa494ccd289eebf39f9ae5beb2ec4181c02b28f87c6c5`（单成员 `gelu_tanh_and_mul.py`）。
- release 回执：`artifacts/competition/batch5-new5-validate-20260913/gelu_tanh_and_mul/verification.json`，
  SHA-256 `0d4850777b93b1fd6349205ce2abd45bc48a1fc446f011269067ab520486084d`；日志 SHA-256 `771a8af8086ff4a38bde695cbd34400ea7beccb08b3460ed0cba1eeeb44cc803`。

## 验证状态

- screening 多轮门禁拦下并修复的缺陷已记录于提交说明；
  最终 release：0 失败/错误/skip，非空 kernel launch，NVIDIA 代理范围。
- 所有八芯目标 `target-runtime-unverified`；裁决权在平台。

## 2026-09-13 E1/E2（API 真值订正）

**真值**（submission API 逐芯）：
- s0(13763)：avg 2.5656 | enflame 1.028 / kunlun 0.265 / huawei 1.954 / tianshu 4.738
- e1(13875,BLOCK=4096)：avg 2.6265 | enflame **1.484（+44%）** / kunlun 0.247 / huawei 1.88 / tianshu **5.091（+7%）**
- e2(13898,enflame/kunlun 冻结 1024)：avg 2.6208 | enflame **1.028（-31%,自伤）** / kunlun 0.263 / huawei 2.079 / tianshu 5.202

**订正结论**：此前记录的"4096 回退燧原 -50%/昆仑 -89%"系转录错误（对照
基线非 API 原值）。真相：**BLOCK=4096 对燧原是 +44% 的增益**，与 skill
"GCU BLOCK 倾向远大于 GPU"一致（T75 同构 +71%）；e2 的"冻结 1024"
vendor 反把燧原打回 1.028——**该 vendor 是自伤，应改回 4096**。kunlun
对 1024/4096 不敏感（0.25-0.27）。

**真实差距**（vs c2flow 3.440）：燧原 1.484 vs 3.83（2.6x）、昆仑 0.247
vs 1.14（4.6x）——两芯均为结构性差距。TB 保持 e1 2.6265。
**下一假设**：①燧原再推 BLOCK 8192（4096 已证正向）；②昆仑 no-loop
形态（T61 的 no-loop 简单 1D 读 1.29 vs 本题 grid-stride 读 0.25）。

## 2026-09-13 E3：燧原配方复刻（BLOCK 4096 + 封顶 24，已发射 13917）

- 取代 e2 的冻结-1024 自伤 vendor；套用 T73 e3(+92%)/T75 e2(+113%)
  双证配方。预期 enflame 1.484 → ~2.9（均值 →2.8+）。
- source commit：`d7f0d6f83151a432bfbabd350814bbe2a817fb27`；ZIP `e3-d7f0d6f`，
  SHA-256 `58531549a6fbac30b280d50d61aef6f090cd4bd1336011e9020570ee3cf576c8`；
  release 回执 SHA-256 `b793d827900cdc8c53407e9a4ba51662ee5d85386f2fe6e9e946f2bbcb9c9b2d`。

## 2026-09-14 E4 候选就绪：昆仑 no-loop（队列 6 号，T75 正信号解锁，待发射）

- 载体 = 保持本 op 已证的 BLOCK_COL=1024（4096 曾把昆仑打到 0.25），
  去 runtime grid-stride 循环：flat grid=(rows×col_blocks)，一 program
  一 tile；数学/fp32 契约与 generic 字节一致。单变量=T75 已验证的
  去循环改动平移到本题。
- source / verification commit：`39838fbf…`；ZIP `e4-39838fb`，
  SHA-256 `a7895afe770b6ccc83a336f6989cbb902ce49878184fa2dbd5cd2e051965794e`。
- release 回执 `batch5-submit-20260914/gelu_tanh_and_mul/verification.json`
  SHA-256 `aeb717873dba19a0579837f7edc95bdd8bc5179fcb9727b6b80d7403d6758ea7`
  （--proxy-vendor kunlunxin：vendor 15 次真实 launch，0F0E0S）。
- 预注册晋级门：**昆仑 ≥ 0.50**；失败不推广至 T73（其 gate 分支另判）。
