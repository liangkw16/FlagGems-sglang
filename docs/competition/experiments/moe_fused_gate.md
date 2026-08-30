# Task 31 `moe_fused_gate` 实验记录

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
