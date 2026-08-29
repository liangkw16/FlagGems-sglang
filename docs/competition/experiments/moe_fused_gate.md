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
