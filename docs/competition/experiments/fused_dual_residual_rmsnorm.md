# Task 52 `fused_dual_residual_rmsnorm` 实验记录

```current
task: 52
operator: fused_dual_residual_rmsnorm
batch: 4
validity: invalid
platform: 6/8(e4,燧原两核拆分也不行;精度天花板)
team_best_stage: -
team_best_speedup: -
sealed: no
next: E5预注册:燧原/昆仑tl.rsqrt配方(T19同平台正证据,1发);败则raw_result分诊后封6/8
updated: 2026-09-05
```

## S0（reciprocal: v*(1/rms)*w）: 5/8
- 天数✗(1/16.7M) 燧原✗(1/33.5M) 昆仑✗(4/33.5M) card_b✓

## E1（div_rn: div_rn(v,rms)*w）: 5/8（tianshu✓ card_b✗）
- 天数✓ 燧原✗ 昆仑✗ card_b✗(NEW!)

## E2（plain /: v/rms*w）: 5/8（tianshu✓ card_b仍✗）
- 天数✓ 燧原✗ 昆仑✗ card_b✗
- 结论：三种除法形式各自修一芯破另一芯，1/33M 元素在 bf16 舍入边界

## E3 AMD reciprocal vendor → card_b 翻绿（submission 10143）
- **card_b PASS 8.09x**（reciprocal 形式恢复 S0 通过路径）
- tianshu PASS 8.62（direct / generic 继续生效）
- **6/8**：天数/沐曦/海光/华为/A/card_b 全过
- 仅燧原+昆仑败（同款 1-4/33M 精度边界，两种除法形式均不过）

## E4 燧原两核拆分（submission 10151）
- 燧原仍败——memory 物化中间 bf16 也没修复 1/33M 边界翻转
- **判定：T52 6/8 是当前方法的天花板**（燧原+昆仑均不可修）

## 2026-09-05 Codex 会诊方案与 E5 预注册

Codex 配方矩阵部分兑现：E3 `_amd` reciprocal 修好 card_b（8.09x）与
预测一致。剩余两芯（燧原/昆仑）的**未试首选是 `tl.rsqrt` 配方**——
T19 fused_rmsnorm 同平台用 tl.rsqrt 在燧原/昆仑均通过（最强直接正
证据）；E1 div_rn ≠ rsqrt，该轴尚未消费。

**E5（预注册，1 发）**：`_enflame`/`_kunlunxin` 两 vendor 的两处 rms
均改 `v * tl.rsqrt(var+eps) * w`，其余芯字节冻结 E4 载体。
次选：`tl.sqrt_rn + tl.math.div_rn`（最贴 eager 源码顺序；sqrt_rn 在
厂商 fork 的可编译性先离线验证）。
已证伪勿重复：plain /（E2）、两核拆分 mid 物化（E4）、fp64 兜底
（燧原无原生 fp64）。E5 仍败 → 只读拉 raw_result 分诊 mid vs out
首个分叉点；无新证据则封 6/8，额度转 T42/T53。
