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
