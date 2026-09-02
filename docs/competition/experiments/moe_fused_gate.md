# Task 31 `moe_fused_gate` 实验记录

```current
task: 31
operator: moe_fused_gate
batch: 3
validity: invalid_correctness
platform: E9 sub8270 7/8;Kunlun 1833723ms 同指纹(第16例)
team_best_stage: e7(=e6字节载体)
team_best_commit: f093ae8
team_best_speedup: 七芯~7.73
blockers: 四微核全拆仍同指纹;崩溃面不在源码复杂度层,超出本地可达
sealed: yes
next: 永久封存;仅平台工单回应+他队结构公开或昆仑修复后以 e7 载体单发重验
updated: 2026-09-02
```

状态:S0 候选就绪

## 契约锁定

- 签名:`reference(scores, bias, topk, scoring_func="sigmoid",
  num_fused_shared_experts=0, renormalize=True, routed_scaling_factor=1.0,
  apply_routed_scaling_factor_on_output=False, moe_softcapping=0.0,
  num_expert_group=1, topk_group=1)`
- `scores [M,N]` fp32/fp16/bf16;`bias [N]` fp32;输出
  `(weights [M,topk] fp32, indices [M,topk] int32)`
- 三种 scoring(sigmoid/sqrtsoftplus/softmax+tanh 软封顶)、DeepSeek
  分组 top-2/topk_group 掩码、融合共享专家槽、renorm、输出缩放全分支
- 容差:fp32 1e-4;indices 需精确;八芯

## S0(2026-08-29,commit `1b32d9a`)

- kernelgen 生成语义底稿 + 手工重写:单 program 一行,N 整块
  (BLOCK=next_pow2(N));复用平台验证组件(router 的 tanh 多项式
  软封顶、迭代 argmax+mask topk + min-index 平局、max 减 softmax);
  分组掩码用**精确边界 static_range 循环**(reshape 在非 2 次幂
  epg 下会错位组界);共享专家/renorm/缩放全 constexpr 分支。
- screening(gpu:/tmp/t31.107MPt,字节与 blob 逐项一致):unittest
  5/5(3 dtype × 3 scoring × 5 flag 组合 × 分组 × 平局 × 非 2 次幂
  × 4096 行);bench 6/7 shape **2.15–6.03x**;唯一索引差异为
  65536 行中 1 行的 1-ulp sigmoid 近平局翻转(fp16 量化固有,
  非次序语义)。
- ZIP `s0-1b32d9a`,SHA `a5f91a23fd6b5c398f1bd97f3b153dbdcb10ffcbc74dd5298a045f561afb574e`,单成员。

### 跨芯风险

- 与 T26/T27 同族:华为 reduce 数值问题、昆仑平台崩溃风险;
  sqrtsoftplus 的 sqrt/log 未有昆仑实证。

### S0 提交与七芯结果(sub 6368,2026-08-29 17:1x CST)

- preflight 全过(tid `s2t1op031`,额度 10/30 消耗 1 → 9/30);
  **七芯全过**:海光 13.364x、card_a 12.778x、card_b 11.467x、
  天数 7.006x、华为 4.046x(本族题华为首过!T26/T27 的华为 reduce
  失败未复现)、沐曦 3.581x、燧原 0.886x。
- **昆仑 waiting_callback**(评测器不稳,等待中);若过即 8/8,
  七芯均值 ~7.9x 已超榜首 11.62x 之外的竞争带。
- 后台观察器轮询昆仑终态。

### S0 昆仑终态 + E1 重载(sub 6368 终态 7/8)

- 昆仑 completed passed=False:**平台侧 inductor 崩溃指纹**
  (执行超时 1830s/1800s + torch inductor compile_worker
  Fatal Aborted,failed_cases=0)——与 T25–T28 同族服务端问题,
  非本 kernel 缺陷;当日 T33/T34 昆仑正常完成,评测器间歇病发。
- **七芯证据完整**:海光 13.36/card_a 12.78/card_b 11.47/天数
  7.01/华为 4.05/沐曦 3.58/燧原 0.89。
- E1 = 重载载体(commit `9e3d10c`,除注释外与 s0 逐字节一致),单发
  重投昆仑窗口;额度 9→8/30。

### E1 提交记录(2026-08-29 18:0x CST)

preflight 全过(额度 9/30 消耗 1 → 8/30);单次 confirm 提交,评测
入队,昆仑终态待回填。

### E1 终态(sub 6374,2026-08-29 18:2x CST)

- 七芯第二次全过(华为 4.92/card_a 13.03/海光 12.71/card_b 11.48/
  天数 6.86/沐曦 3.56/燧原 0.88);
- 昆仑 completed passed=False:**同一 inductor 崩溃指纹**
  (1830s + compile_worker Aborted,failed_cases=0)——与 S0 完全
  同指纹,重载无效,符合"reference 含 topk → 昆仑验证侧必崩"的
  6v6 相关性结论(见 kunlun-crash-ticket 文档)。
- **T31 终态 7/8,工单路径**;两投同指纹,止损。额度 8/30。
  保留候选:评测器修复后 e1 可直接复用(七芯证据完整)。

### E2 恢复窗口重投(sub 6477,2026-08-29 21:2x CST)

健康信号:T25 达标 7→8。e2 = e1 + 注释载体(commit `dee4ed3`);
额度 8/30 消耗 1 → 7/30;昆仑终态待回填。

### E2 终态(sub 6477,2026-08-29 22:3x CST)

昆仑 completed failed:**同一 inductor 崩溃指纹**(1830s + Aborted,
failed_cases=0),排队 ~55 分钟后返回。T31 三投同指纹,恢复窗口
(T25 达标 +1)对本题无效,候选重投额度用尽,关闭;仅剩赛方修复/
工单路径(证据链:6v6 相关性 + 三投同指纹 + 七芯全过)。

### E3 双确认恢复窗口重投(sub 6581,2026-08-30 10:5x CST)

同窗口(T26 5→6 + T28 2→3)。e3 = e2 + 注释载体(commit
`da06494`);七芯结果与昆仑终态待回填。

### E3 终态(sub 6581,2026-08-30 12:0x CST)

昆仑同指纹崩溃(第 11 次)。**解读反转**:T26 已 6 队、T28 已 3 队
八芯通过——纯平台 reference 侧理论不成立,触发面疑在我们 kernel
的实现惯用法(经 torch.compile/inductor 编译 Triton 的昆仑路径):
- 崩溃组共性:T31/T36 均含 `tl.log`(softplus),T31 另有 `tl.sqrt`,
  T25-28 共享迭代 argmax+mask topk;
- 通过组(T29-E6 exp/T33 div_rn/T32/T35 纯算术)从未用过 log/sqrt;
- T29 先例:erf 崩 → A&S 多项式替换即过,libdevice 替换是已验证
  修复模式。
- 下一步(若再投):kunlunxin vendor 用多项式 log1p 替代 tl.log
  (精度 1e-7,容差 1e-4 内),一发验证假设。优先级让位于 T39/T40。

### E4 定向假设投(sub 6xxx,2026-08-30 17:4x CST)

- 窗口:T26 达标 6→7(第三个 crash 族新增信号);
- 假设:`tl.log` 是崩溃族共享嫌疑算子(T31/T36 均含,所有昆仑通过
  的 kernel 均无);kunlunxin vendor 用 atanh z 级数多项式 log1p
  替代(误差 <5e-9,容差 1e-4 内),其余字节同 e3(T29 erf→A&S
  同款修复模式);
- 代理:vendor 过完整 5/5 unittest(设备参照、索引精确匹配);
- commit `70c14a5`,ZIP SHA `5415c2ed…`,2 成员;额度 15/30。
  若过 → log 假设成立 + 第 7 题 valid;若再崩 → 嫌疑转移至
  sqrt 或 topk 机器,题内止损。

### E4 终态 + E5 重投(2026-08-30 18:4x CST)

- **E4(sub 6661)昆仑返回新错误:"服务线程卡死自动恢复,请重新
  提交"——非 inductor 崩溃指纹**(T28 E3 先例,平台明示可重投的
  exec 服务端问题)。重要推论:log 多项式版未触发编译崩溃,
  log 假设未被证伪。
- E5 = e4 + 注释载体(commit `290c84e`,ZIP SHA `6d5bf63a…`),
  preflight 全过(额度 12/30),单次提交;昆仑终态待回填。

### E5 终态:T31 止损(sub 6680,2026-08-30 19:4x CST)

- **同一 vendor 字节两次投出两种错误**:e4 = 服务线程卡死(可重投
  类),e5 = inductor 崩溃指纹(第 13 次)——**同字节不同错**证明
  两者均为平台侧间歇故障,log 假设无法证实或证伪。
- T31 五投(s0/e1/e2/e3/e4/e5 含载体)终态 **7/8**,按纪律题内
  止损;额度 11/30。若后续昆仑健康期(达标数显著上升)可复用
  e5 候选(`290c84e`,七芯 ~7.6x)。

### E6:T31 专属信号重投(2026-08-30 21:2x CST)

- 信号:**T31 达标 1→2**——他队刚在本题八芯通过(昆仑成功处理
  T31 的 topk reference),首个本题专属恢复信号;
- e6 = e5(log 多项式 vendor,七芯 ~7.6x)+ 注释载体(commit
  `4b09547`,ZIP SHA `b8d2732f…`);preflight 全过(额度 10/30),
  单次提交;昆仑终态待回填。

### E6 终态:T31 永久封存(sub 6712,2026-08-30 22:0x CST)

- 昆仑第 14 次 inductor 同指纹崩溃。**决定性对照**:他队在本窗口
  通过 T31 八芯(达标 1→2),我方同窗口 e5/e6 两投皆崩——平台
  间歇论削弱,我方 kernel 惯用法触发论回升(嫌疑收敛至 topk
  机器或 sqrt;log 已多项式化仍崩)。
- T31 六投终态 7/8,永久封存;e6 候选(`4b09547`)留档。

### E7:信号窗口重载(sub 7227,2026-08-31 16:0x CST)

- 触发:15:46 监控发现 **T31 达标 2→3**——他队刚在 T31 八芯通过
  (含昆仑),按 E6 同型预注册协议立即重载 e6 字节载体;
- 七芯全过(天数 6.98/海光 13.00/card_a 13.59/card_b 11.52/华为
  4.46/沐曦 3.66/燧原 0.88),均值 ~7.73;
- **昆仑终态:第 15 次同指纹 inductor 崩溃**(1830s 超时 + Fatal
  Python error: Aborted + subproc_pool.py,空 failed_cases,
  validation 9a82040f)。
- 身份补录(2026-08-31 17:4x 回填自本地 intent `7192cca7`):source
  commit `f093ae8`,ZIP SHA-256
  `3f5e5cdce7d1e10677bc8c08540fd47e3bb0770a9f46ddc7512ad3c97d3d5ba1`
  (16890 字节,成员 `moe_fused_gate.py` + `moe_fused_gate_kunlunxin.py`,
  与 e6 载体相同,已经 `unzip -Z1` 复核);sub 7227 = 当日 daily_seq 18,
  提交 `2026-08-31T16:03:55+08:00`,提交后剩余 12/30(按 30−18 推得)。

### E7 判定:T31 永久封存

- 他队同窗口通过 + 我方重载即崩 = **我方 kernel 惯用法触发论坐实**
  (嫌疑:topk 机器的 tl.where/索引形态;log 已多项式化仍崩);
- 平台"健康窗口"假说对 T31 失效,连带冻结 T36/T38/T41/T28 的
  盲发重载——后续只在各自题出现达标上升 + 平台修复工单回应后
  单发验证;弹药(全部已 commit+ZIP)继续在库。

## E8:昆仑三阶段结构改写(2026-09-02)

状态:E8 已单次提交；平台 7/8，Kunlun 再次于约 1830 秒
compile-worker 崩溃。三阶段同候选封存，禁止重试。

### 决策与预注册门

- 实时只读状态(`2026-09-02T09:19:23+08:00`):E7 sub `7227` 仍为
  7/8，昆仑 `1833721ms` 后 compile-worker/subproc_pool Fatal Aborted、
  `failed_cases=[]`；额度 `27/30`，Task 开放至 `2026-09-03 19:59:59`。
- T31 自身共八投，其中 S0/E1/E2/E3/E5/E6/E7 七次为 1830s 指纹，E4
  为服务线程卡死；历史"第 15 次"是跨题崩溃族累计，不是 T31 投了 15 次。
- E7 七芯和约 `54.09`。E8 只用于恢复 8/8/进入有效榜；即使昆仑仅过
  `0.1x`，预计均值约 `6.77375x`，不把它表述为可信夺冠候选(实时榜首
  `59.192025x`)。
- 晋级门:generic 字节冻结；generic+vendor 完整正确性通过；全部代理形状
  `>=0.1x`；0 spill/0 scratch；commit-bound release 与 ZIP 验签一致。
  平台 success gate 为 8/8 且昆仑 `>=0.1x`。任一数值失败、昆仑低于
  `0.1x` 或再次出现同一 1830s 指纹即关闭本三阶段轴，不重试同候选。

### 单变量与 KernelGen

- 单变量只替换 `_kunlunxin`；generic SHA-256 冻结为
  `7bf2a6f48691ab4eb601206b1176d34e54b8426c571be5f67bc0686fd129f3fc`。
- 根因假设从"平台间歇"收敛为原 vendor 巨型 AST:评分、group top-2、
  group top-k、expert top-k、共享槽、renorm/scale 和 device persistent
  `tl.range` 全在一核。E8 参考 T27 E8 的昆仑成功证据，改为三核:
  1. 物化 fp32 selector；2. 只做 grouped/expert 选择并写 indices；
  3. 按 indices 重算/gather activated 并一次写最终权重。
- KernelGen MCP transport 正常(4 tools)。`optimize_kernel` 首轮因
  `tl.arange(0,K_ROUTED)` 非 2 次幂、group top-2 并列删除错误和额外 ids
  workspace 在落盘前被拒；反馈回灌后的第二轮产出当前三阶段模块。只做一处
  明确单行修复:stage2 store 指针改为 `+ slots`，使 vector value/mask 与
  pointer shape 一致。生成请求约束保存于
  `log/kernelgen-round/req_t31_e8_opt_desc.txt`(ignored evidence)。
- 随后只发起一次 `generate_kernel(device="kunlun")` clean-room
  对照；MCP 入口已进入 chunked SSE 响应，但约 60 分钟内无任何
  JSON-RPC 终态/错误/验证结果，客户端终止于 `resp.read()`。按预注册
  规则记为 KernelGen Kunlun verifier 不可用，不重试、不冒充芯片通过；
  目标芯证据交由平台本次唯一提交。
- 新 vendor 还修复了旧实现的契约边界:组内两个并列最大值只排除一个确定
  index，再取第二大；group/expert 均用 max + min-id 精确选择。softmax 的
  grouped/top-k 严格使用 reference 的 pre-softmax biased logits，只有输出
  权重使用 softmax probability。

### Screening

| 项目 | 值 |
| --- | --- |
| base/source/verification commit | `9b6911de7aa2f794b6d0499ea7a346ea06e7c2b8` |
| Kunlun vendor SHA-256 | `907e9aaf201423515b622af941677e7e4d1cbe9c0b5d78cbd33967094e5a9926` |
| test SHA-256 | `9600bcaaf60f6437809434ee34342de84f1c8e729232597ad6ed604c94401f56` |
| `_op_variants.py` SHA-256 | `cdc5fe3e4cb5a85976f0a3414cd194bb53c79f6f2830be01f685f996b97ca0d7` |
| benchmark helper SHA-256 | `a5f254739747de9b59ae72e22648873094be81bceb7d7665e23cf2f3ebe83513` |
| screening 目录 | `gpu-et:/tmp/flagos-moe-fused-gate.ZpSxQb`(0700) |
| final test PID/PGID/SID | `265840` |
| test/benchmark log SHA-256 | `99082f81fc050428c0d55661d32681debe893c7bcb59f437a1f8dbd40618061b` / `43eef2f6170511e11ba741fa79f6b3fca0fe9a0c4807c6414ae0c9fa3ed66507` |

- 环境:RTX 5070 Ti、PyTorch `2.13.0+cu130`、Triton `3.7.1`、CUDA
  `13.0`。py_compile/Black/isort/flake8 通过，generic+vendor unittest
  **6/6**；覆盖三 dtype/三 scoring、group/shared/renorm/scale、group
  top-2 精确并列、softmax selector 语义、N=65/G=5、topk 3/6/8、空行及
  M=70000 grid fold。
- 五轮 AB/BA、warmup25/repeat100 的 vendor/reference speedup:
  `2.7016/4.2421/4.2618/4.3559/2.9445/2.2466/4.5959x`。65536×128
  fp16 的历史 1-ulp 近平局由冻结 generic 与 vendor 同字节复现，显式标记
  `KNOWN-TIE`；M=70000 精确 indices 回归通过。
- vendor 最大 `63` registers、0 spill、0 scratch、最多 16 bytes shared，
  三核均 `num_warps=4,num_stages=1`；generic 最大 103 registers/stages3。

### Commit-bound release

- release 目录:`gpu-et:/tmp/flagos-moe-fused-gate-release.Xkvyno`(0700)，
  source/test/helper 全由 commit `9b6911d` Git 对象导出；test PID/PGID
  `266017`，benchmark PID/PGID `266128`。静态检查、6/6 unittest、同七
  shape benchmark、资源检查和前后完整 SHA 均通过。
- release test log SHA-256
  `24f9d7ca2d5b0e87361ae3736b309d871a0d89fbeac5094bc7205753c171fa35`；
  release benchmark log SHA-256
  `450a53350b91bf2fb3a4ef4ab03793d08b77c428213adae4c7058929687c5eef`。
- canonical ZIP 已由 source commit `9b6911d` 创建，并通过
  `--verify-existing`/`unzip -t`。路径
  `artifacts/competition/moe_fused_gate/e8-9b6911d/moe_fused_gate.zip`，
  18755 bytes，SHA-256
  `f7ec5b4fab04ceb1fa78ad7df9931f7fc74beff82d48563445674e1d15c19820`；
  根目录仅含 `moe_fused_gate.py` 与
  `moe_fused_gate_kunlunxin.py`。

### E8 平台结果(sub 8148)

- `2026-09-02 10:36 CST` 实时 preflight 重验 race/season/账号/
  团队、batch 3/Task 31/tid `s2t1op031`、source/verification commit、
  test/release/ZIP SHA 及两个成员全部匹配；Task 为
  `competing/submitting/can_submit=true`，提交前额度 `27/30`。
  消费该一次性 intent 后正式提交成功，daily_seq 4，额度
  `26/30`，禁止重发。
- submit 内置远端验签因未预设受信对象存储 host 而为
  `unavailable`；随后对平台返回的精确 HTTPS URL 做无认证、
  禁止重定向的独立回读，得到 18755 bytes、SHA-256
  `f7ec5b4fab04ceb1fa78ad7df9931f7fc74beff82d48563445674e1d15c19820`，
  与本地 canonical ZIP 完全一致。
- 七个非 Kunlun 芯片全部通过：天数 `6.827x`、沐曦
  `3.6706x`、燧原 `0.886x`、海光 `12.924x`、华为
  `4.9838x`、国际 A `13.1628x`、国际 B `11.464x` 均通过；
  七芯和 `53.9182`。
- `11:07:03 CST` 终态:`completed/invalid_correctness`、7/8。Kunlun
  执行 `1833425ms`后 `passed=false`，无数值 case 失败，
  `failed_cases=[]`；错误仍为验证阶段超时、子进程先退出，
  `Fatal Python error: Aborted` 于
  `torch/_inductor/compile_worker/subproc_pool.py::_recv_msg`。该读数与
  E1/E2/E3/E5/E6/E7 的 1830s 指纹同型；不重试。
- 结构归因:拆除 `tl.log`、device persistent loop、split/partials 与
  大部分复合职责后仍同型崩溃，将主要未隔离因子收窄到
  stage2 内 `NUM_GROUPS` 两轮静态处理叠加 `TOPK_GROUP`/
  `K_ROUTED` 多轮 max/min/sum 的巨型 grouped-route AST；`tl.sqrt`
  仅作次级未隔离风险。若继续 T31，只接受把 group score/
  group select 与 expert select 再分核的新结构，不再扫
  BLOCK/warps/stages 或重投三阶段字节。

## E9:四微核全拆结构(2026-09-02)

状态:E9 已单次提交;平台终态待回填。

### 决策与预注册门

- E8 终态后唯一被接受的轴:把 group score / group select 与 expert
  select 再分核。E9 落地为**四微核全拆**——stage1 selector 与
  stage3 finalize 保留 E8 字节(已核与 `9b6911d` 逐字节一致),
  stage2 整体替换为:
  1. `_group_score`(每行,runtime `range(n_groups)` 循环):top-2
     和用 max + count(max)>=2 + 严格小于 max 三件套,零整数归约、
     零 static_range;
  2. `_group_select`(每行):rank 法(BLOCK_G 广播比较 + sum,
     (值降序、id 升序)全序)选 TOPK_GROUP,零循环;
  3. `_apply_group_mask`(每行):gather `eg[]`(host torch 预计算
     组 id,clamp 防 N%G 余数越界)物化 selw,零循环;
  4. `_pick_slot`(每行,host 每 slot 一发,共 K_ROUTED 发):单次
     masked argmax + min-id 平局,写 indices 并置 selected=1。
- 与 E8 的对照:E8 stage2 单核携带 NUM_GROUPS 两轮 static 处理 ×
  TOPK_GROUP 循环 × K_ROUTED 循环的巨型 AST;E9 每核只剩**单次
  max(+至多一次 min-id)或一个 runtime 小循环**,static_range 全部
  消除,循环只剩 GEMM 已两证昆仑可过的 runtime `range` 形态
  (T28/T37 规则 GEMM 范式);host-stepped 多发射是 T41 四指针同款。
- KernelGen MCP:`.mcp.json` 已配置 kernelgen-server,但本会话工具面
  未暴露 kernelgen 工具;按 E8/T28 先例(基础设施故障不阻塞,E8 模块
  本身即 MCP optimize 产物)在 E8 字节上直接分核实现,请求约束存档
  `log/kernelgen-round/req_t31_e9_split_desc.txt`。
- 晋级门:generic 字节冻结(`7bf2a6f4` 复核一致);generic+vendor
  unittest 全过(6 法含 grouped/ties/N=65 G=5/M=70000);flake8/isort
  通过 + black 25.12.0(79 列)格式化;release 与 ZIP 验签一致。
  平台 success gate:**8/8 valid 且昆仑 >=0.1x**;七芯维持 E8 映射
  (~53.9 和),昆仑过 0.1x 时期望均值约 6.77x,定位为恢复上榜,非
  夺冠候选(实时榜首 59.19x)。
- stop gate:昆仑再现同型 1830s compile-worker 崩溃 → 本分核轴关闭,
  **T31 永久封存**(E8 预注册的最后一类结构已用,无后续轴);
  任一芯数值失败 → 需新 commit 修复,不得重投同字节。

### Release(commit `9b0c1bc`,2026-09-02 16:1x CST)

- 首轮 release(`8abc266`)被 unittest 门禁拦截:host 侧 `Tensor.div_`
  对 int32 走浮点除(`result type Float can't be cast to Int`,5 法
  全体 error)——release 先行的价值实证;修复为
  `torch.div(rounding_mode="floor")`,新 commit `9b0c1bc`,旧 ZIP 删除
  重建。
- release 目录 `gpu:/tmp/flagos-rel-t31e9b.8IsJhm`(0700,源/测试全由
  commit `9b0c1bc` Git 对象导出):py_compile/isort/flake8 通过;远端
  black 26.5.1 漂移照录(本地 25.12.0 79 列格式化 + 裸检通过);
  远端 SHA 复验 5/5 逐字节一致;**unittest 6/6 OK**(matrix 15 组合 +
  grouped 3 + ties + kunlun_grouped_edges + N=96/4096/70000/N=65G5 +
  empty);
- 资源探针:6 微核全部编译,最大 `_group_score`/`_pick_slot` 20
  registers,**全核 0 spill/0 scratch**;release log SHA-256
  `1c9398bea6322525120f1b60e00ce169b14e116da7b65e8c77bac53b5ada09af`;
- canonical ZIP `e9-9b0c1bc/moe_fused_gate.zip`(2 成员:generic
  `7bf2a6f4` 冻结、kunlunxin `240f1f75`),SHA-256
  `8eeca0ef34bb06fa2b478f44b540d2343c979dbb57af6be6a91f2e81e4e7a8d4`,
  `--verify-existing`/`unzip -t` 通过。

### E9 平台提交(sub 8270,2026-09-02 16:05 CST)

- preflight 全过(task competing/can_submit,额度 17/30),nonce
  `39fbbf5eee659881f8feba27aada3773` 一次性消费,submission **8270**
  (daily_seq 14),额度 17→16/30;file_url_sha256
  `35220a654f351b073b8ac2221e1c65f7d79713c94b41c6377a5d66154835da45`;
- 平台终态待回填(若再现 1830s 指纹,评测约 31 分钟出终态)。

### E9 终态(sub 8270,2026-09-02 16:3x CST):第 16 次同指纹,T31 永久封存

- 七芯全部通过且读数健康(天数 6.8166/沐曦 3.645/燧原 0.8832/海光
  13.8264/华为 4.8704/card_a 12.529/card_b 11.5736,和 54.24);
  **昆仑 1833723ms 后 `passed=false`、`failed_cases=[]`,与 E1-E8 的
  1830s compile-worker Fatal Aborted 完全同指纹**(T31 内第 8 次
  1830s 型,加 E4 服务线程卡死为九投九败);额度 16/30。
- **决定性排除**:E9 已把每个核的 AST 压到机制性下限——单核最大
  20 registers/0 spill、无 static_range、host-stepped 每 slot 一发
  单 argmax——仍触发同一指纹。E8 归因的主嫌疑"巨型 grouped-route
  AST"不成立;tl.sqrt 也已在 stage1/3 之外无处可删。崩溃面不在
  我方 Triton 源的复杂度或惯用法层面,结合 E7"他队同窗通过",唯一
  剩余差异空间在平台 validation 侧与特定实现的编译交互,超出源码
  单变量可达范围。
- **T31 永久封存**(预注册 stop gate 兑现:分核轴已用,无后续轴);
  树回滚 E8 字节 `9b6911d`(`907e9aaf`,revert commit `f76e46f`)。
  重启条件:平台工单回应且他队通过样例字节结构公开,或昆仑平台
  修复后以 e7 载体(`f093ae8`)单发重验;不再做本地结构迭代。
- 七芯资产价值:E9 七芯读数与 E8 一致,证明四微核结构本身跨芯
  健壮;若未来昆仑恢复,候选可复用,无需重新开发。
