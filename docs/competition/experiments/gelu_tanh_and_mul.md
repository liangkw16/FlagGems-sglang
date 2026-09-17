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
platform: completed(e10/sub16462,8/8,2.78165833x新TB；e9/sub16473八芯2.78134167x未超过)
candidate_stage: e10
team_best_stage: e10
team_best_speedup: 2.78165833
sealed: no
next: 用户要求不再新提交；保留E10最佳，E9未提分；第15，追平Top1需92.97%
updated: 2026-09-17
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

## 2026-09-14 E4 平台终态：8/8 VALID 2.7109x 新TB（submission 14573）——no-loop 复制链失败

- **昆仑 vendor passed 0.3043**（循环形态 0.2627，+16%）——预注册门
  （≥0.50）未过。no-loop 假设在 T75（0.26→0.90, 3.4x）成立但**不
  推广到 T71**（BLOCK 1024 / exp 超越函数 / 行结构差异需另证）；
  按预注册不向 T73 跟发，昆仑 no-loop 复制链就此收口。
- 均值 2.7109 微破 TB e3 2.6748（其余七芯窗口方差内：天数 5.19/5.12、
  燧原 1.74/1.74、海光 3.94/3.88、华为 2.05/1.93、A 3.12/3.14、
  B 3.15/3.16、沐曦 2.19/2.17）——TB 更新为 e4。
- 榜首金狐狸 3.441，差 0.73；剩余轴：昆仑真因未破（0.30 vs 1.0）、
  燧原 8192 未试（leader 3.67）。

## 2026-09-14 E5 探针夭折（代理失败神谕，零额度）：tanh 路线不可移植

- 假设：官方 gcu400 gelu 用 shim tanh（FlagGems 生产在用），若 GCU
  有原生 tanh 则可解释我方 exp 恒等式与燧原 3.7-4.1 聚集的差距。
- 代理两次拦截：`tl.math.tanh` 在 Triton 3.7.1 编译错误（15 errors、
  vendor 0 launch）；`tld.tanh` 的 JIT 语义解析路由到不存在的
  `tl.math.tanh` 同样失败。FlagGems 自身的 shim 是 backend 注册的
  `triton.language.extra.<extra>.libdevice` + 缺符号 exp 补丁——GCU
  的模块名与原生 tanh 可用性无法在代理侧确认，且存在"Python 层
  存在但 lower 为 None"的编译陷阱。
- 处置：按 Codex 门"源码差异只剩已试过项则不提交"——燧原 tanh 轴
  暂停，字节恢复 e4 载体（exp 恒等式）。重开条件=FlagTree-GCU 的
  extra.libdevice 模块名与 tanh 符号的真实确认（KernelGen 或镜像内
  introspection）。
- 额度：未消耗（两次均为代理拦截）。

## 2026-09-14 E6 候选就绪：燧原 BLOCK 4096→8192（Codex v3 队列槽 2，解禁一发，待发射）

- 解禁依据：本芯宽度轴单调（1024→4096 +71%）；同族 T75 4096→8192
  +23%（虽未过其 2.2 门，方向性证据）。单变量=BLOCK_COL 4096→8192，
  其余=TB e4 字节。
- source / verification commit：`3db8540…`；ZIP `e6-3db8540`，
  SHA-256 `9d4018cd0e8253b85797c172649225d339757905aa0d3a3180dd6fc9b0ca485f`。
- release 回执 `batch5-submit-20260914/gelu_tanh_and_mul_e6/verification.json`
  SHA-256 见账本 artifacts（enflame vendor 代理 0F0E0S）。
- 预注册晋级门：**燧原 ≥ 2.10 且完整均值 ≥ 2.75**；未过停宽度轴，
  不再 16384 梯度。

## 2026-09-14 E6 平台终态：8/8 VALID 2.689x < TB（submission 14690）——8192 边际递减

- **燧原 vendor passed 1.8725**（4096 的 1.7391 → +7.7%）——门（≥2.10）
  未过。同族 T75 的 +23% 未复制到本任务（task 形状差异），宽度轴
  按预注册关闭，不试 16384。TB 守 e4 2.7109。
- 其余芯窗口内（天数 5.11/5.19、沐曦 2.18/2.19、海光 3.90/3.94、
  华为 1.91/2.05、A 3.11/3.12、B 3.11/3.15、昆仑 0.30/0.30）。
- 榜首金狐狸 3.4409，差 0.73；剩余燧原缺口（1.74 vs 3.67-4.08 聚集）
  的形态待 FlagTree-GCU introspection 或他队情报。

## 2026-09-15 23:20 夜间 warps 批扫终态：E8 平:燧原1.88(+8%)均值2.712≈TB;warps=4微正不换TB

- 单变量=燧原 vendor 钉 num_warps=4（夜间批 T71/T73/T68/T62 四题）。
  批量结论：燧原 warps 为逐题特性，非统一旋钮；本批 2 负 2 平，
  轴关闭。TB 各自保持。

## 2026-09-16 23:28 真实TB对账与并行结构候选

- live status确认E8/sub **15474**，2026-09-15T23:32:26，8/8、**2.712425**、is_team_best=true；对应source `06ba0bdf1f1db411ebfa0d2456956e4a684d0d12`，URL SHA `7f666e4314c898b7d7ec772ecdab38be894f45524aff95104908eebf508587c7` 与submitted intent一致。订正旧段“未换TB”：涨幅很小但平台确实更新了最佳。
- 新候选只研究Enflame短行四行合并，保留E8其他成员、宽shape实现/参数/exp算式；不是重做8192或warps。主树未验证E9 runtime resolver不进入ZIP。发布前冻结shape矩阵与门并完整release，平台门为八芯有效均值>2.712425且燧原>1.88246667；尚未通过开发门。
- 观察 `2026-09-16T23:28:06.035023+08:00` 额度 **13/30**；状态 `artifacts/competition/pair-grouped-platform-20260916/t71-before-parallel-status.json` SHA-256 `76326ba3f251a89cb1d896ff3a2a4592ddfd093b544a5599b21f2b4660cdc4a4`。

## 2026-09-16 23:44 E10 短行分组发布门通过

- 基线真实TB E8 `06ba0bdf1f1db411ebfa0d2456956e4a684d0d12`，只Enflame rows≥4且0<half_width≤1024合并4行，减少固定8192-lane短行掩码浪费与逐行循环；宽路径/其他芯逐字冻结E8。主树未验证E9动态resolver移除，不携带探针/fallback；数学表达式与运算顺序仍E8 exp恒等式。
- source/verification commit `0d0cedb8a7b5e7d7d2d6604d1ae681a53e6bbd6c`。独立冻结18affected+8controls、5轮AB/BA，门主中位≥1.05、每桶≥0.95、zero spill；实测主中位 **2.6857767624**、GM **3.32658055**、全桶最差 **0.9987995107**，五轮主中位2.678–2.698。主任务独立复算130pairs与Git/SHA通过。T73平台仅微涨是反例，不用它给本题目标芯背书。
- 六份IR probe真实4×BLOCK、FP32 exp原算序/i64地址，零spill/shared；宽kernel AST/哈希与E8一致。原4方法保留，增加group/8192边界、非连续与零宽，共7方法。exact release **7/7**，0fail/error/skip/xfail，generic/Enflame/Kunlun各55入口/52真实JIT；NVIDIA代理，Enflame目标仍未验证。
- ZIP `artifacts/competition/gelu_tanh_and_mul/e10-enflame-group4-0d0cedb/gelu_tanh_and_mul.zip`，9040bytes，SHA-256 `f522c23147d8ba4851961cf3c58b047a5b0a781e884d4c7589378f5fcee5b489`，成员gelu_tanh_and_mul.py及_enflame/_kunlunxin.py；主任务验签receipt/CRC/Git/成员再过。测试SHA `a7ec5422922ebbfc97cd682f1e2cd9adfe03e7780e3cefbfd34dbb9e13db8d7b`。
- `src/flaggems_sglang/ops/gelu_tanh_and_mul.py` SHA-256 `2e620ac3e4fa9c7964b02f2f034f311fdde3fb0ccee36994c1b43c588ee4d7cb`。
- `src/flaggems_sglang/runtime/backend/_enflame/ops/gelu_tanh_and_mul.py` SHA-256 `16b3e71a036b739cd6274c2048addd11161bc7d3da390630978542aa42687170`。
- `src/flaggems_sglang/runtime/backend/_kunlunxin/ops/gelu_tanh_and_mul.py` SHA-256 `b2dc4c7706bd0d5fd3c93d7d56fc6dbd2551280350ce6c50321d3b445f3c4112`。
- `artifacts/competition/t71-group4-screening-20260916/screening.json` SHA-256 `0dc05a9aece19849f20c22f1a40e6679db5f05a4ce5af3e7aa183e2e9d84e042`。
- `artifacts/competition/t71-group4-screening-20260916/release/verification.json` SHA-256 `c17e78c6ad64ecd204b76e3c19415494f349bfa8a7dbb923dd40cd882d54972c`。
- `artifacts/competition/t71-group4-screening-20260916/release/verification.log` SHA-256 `76141549809231875d8e0d79276a1f504fda614250f6f70fad791f059cc6ed81`。
- 平台目标沿用预注册：全部八芯正确且每芯≥0.1，均分>**2.712425**且Enflame>**1.88246667**。只提交一次；失败定位，不重复旧width/warps/native-tanh轴，也不把代理倍数当平台均值倍数。

## 2026-09-17 00:02 E10/E9 平台终态对账（sub 16462 / 16473）

- **E10 group4**：source/verification `0d0cedb8a7b5e7d7d2d6604d1ae681a53e6bbd6c`，于 **2026-09-16 23:52:42** 提交16462、daily_seq19；nonce `fe883a4456be47aae2ccd75de7992726` 为submitted，上传和正式POST各一次，远端9040bytes与ZIP SHA完全匹配。首轮预检IAM GET超时在创建intent之前，第二轮成功后仅提交一次。
- 8/8 valid，均分 **2.78165833，新TB +2.55245%**。逐芯 tianshu5.11513333 / muxi2.10086667 / enflame2.041 / haiguang3.90206667 / kunlunxin0.86533333 / huawei1.94226667 / card_a3.12593333 / card_b3.16066667。燧原相对E8 1.88246667提高约8.42%；昆仑字节冻结E8，0.30273333→0.86533333不能归因于本次group4或新no-loop改动。
- **E9 native-tanh** 由并行会话于 **2026-09-16 23:59:59** 提交16473、daily_seq21；00:02:31只读GET确认8/8 valid、**2.78134167**，未超过E10。燧原1.87953333，未改善E8水位；本次平台数据不含实际命中哪个tanh分支的证据，不将解析未命中或目标符号缺失写成已证事实。
- 两次提交相隔 **7分17秒**。E10燧原2.041高于E9的1.87953333是正向观察；缺少目标芯重复配对计时，不能只凭相邻提交排除窗口影响或证明稳定收益。保持E10团队最佳，不继续重投。
- 00:05榜单第15，Top1 c2flow **5.36765**，追平需 **92.97%**。用户明确停止新提交。
- `artifacts/competition/pair-grouped-platform-20260916/t71-e10-preflight-v2.json` SHA-256 `255b20c6ae8fb1a6c2dd70b7b0e2feb30ec9333cc5f91cad9845b88a67cf4ad5`。
- `artifacts/competition/pair-grouped-platform-20260916/t71-e10-submit.json` SHA-256 `f39bebd73d0b69380d3637ae750ba969c35689b67a19f422d0bab26627961909`。
- `artifacts/competition/pair-grouped-platform-20260916/t71-e10-final-status.json` SHA-256 `b53da825a4ef428dc5c64e5c6a5151b89d87ae99553f7a9e908b6ec85571b936`。
- 完整status观察00:00:22.405134 +08，按submission_id16462提取E10终态；文件同时包含当时尚在评测的16473，不能把文件名当作所有记录均终态。该status确认新日额度30/30、已用0。

## 2026-09-17 E11：Ascend persistent vendor，已提交

- 结构（`f8357985`，第二轮收割）：新增 `_ascend` persistent vendor（grid 封顶 NVC，body 已有 stride 循环，launch-only）。其余成员字节冻结；本题为已证配方家族单变量投放。
- screening（RTX 5070 Ti）：unittest 7 项全绿；black/flake8 过。
- release v2（commit `f83579853cc0163321fd11e6301a1ad61d46eb58`）：7 项全过 0F/E/S/X，changed-vendor 真实 launch 52 次，exit 0。回执 `artifacts/competition/round2-20260917/gelu_tanh_and_mul/verification.json` SHA-256 `55f10fde8a75a592edf365afb4edff68d37a743850b811f06e16c48a27e65728`。目标芯 target-runtime-unverified。
- ZIP：`artifacts/competition/gelu_tanh_and_mul/e11-f835798/gelu_tanh_and_mul.zip（12159B）`，SHA-256 `8f112e005479bac32c3824e4df83fbc112adbc8401c53734d40c12eab18a0a95`，4 成员（enflame/kunlunxin 冻结）。
- 预注册门：8/8 有效且均值 > 2.78165833；华为（1.94 起） ≥ 3.9 为正信号。一次候选一次判决。

## 2026-09-17 E11 单次平台提交

submission **16582**，evaluating；八芯终态另节记录。
