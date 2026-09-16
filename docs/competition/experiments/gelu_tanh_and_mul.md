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
platform: completed(16462,e10,8/8,2.781658x新TB;燧原2.04/昆仑0.87)
candidate_stage: e10
team_best_stage: e10
team_best_speedup: 2.781658
sealed: no
next: e9 native-tanh平台关闭(16473,燧原1.88未动);宽度/warps/tanh轴全封;燧原2.04 vs 全场3.7-4.1结构差与榜首5.37需新证据
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

## 2026-09-17 00:0x E10/E9 平台终态补记（sub 16462 / 16473）：group4 新 TB 2.7817；native-tanh 轴平台关闭

- **E10 group4（sub 16462，`0d0cedb`，11:5x 发射）**：8/8 valid，均值 **2.781658 新 TB**（>E8 2.712425）。燧原 **2.041**（+8.5%，短行分组形态部分兑现）、**昆仑 0.8653**（0.30→0.87，+186%，昆仑 no-loop vendor 形态在平台窗口首次大规模兑现）、天数 5.1151 / 沐曦 2.1009 / 海光 3.9021 / 华为 1.9423 / A 3.1259 / B 3.1607。
- **E9 native-tanh（sub 16473，`8c2d6ba`，11:5x 发射）**：8/8 valid，均值 2.781342 ≈ E10（-0.0003）。**燧原 1.8795 ≈ E8 1.8825（±0.2%）**——FlagGems 式 native tanh 解析链（活跃 backend 探测 + builtin 过滤 + 编译探针 + exp 恒等式兜底）在 GCU 平台未产生增益：解析未命中（GCU Triton 无带 builtin 标记的 tanh extern）或 native tanh 非 3.7-4.1 聚集的缺口形态。轴按平台证据关闭；解析链基础设施（防跨后端符号污染 + 防 stub 陷阱）保留在 `8c2d6ba` 的 vendor 字节中供后续题复用。
- 同窗口读数对比注意：E9/E10 相邻发射（间隔 ~3 分钟），燧原 1.88 vs 2.04 的差为同窗可比，分组形态增益真实；昆仑 0.87 在两发同读，窗口一致。
