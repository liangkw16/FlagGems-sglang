# Task 52 `fused_dual_residual_rmsnorm` 实验记录

```current
task: 52
operator: fused_dual_residual_rmsnorm
batch: 4
validity: invalid
platform: 6/8(e5,五种数学形态同指纹边界失配;已分诊)
team_best_stage: -
team_best_speedup: -
sealed: yes
next: 封存6/8;仅sqrt_rn+div_rn或torch归约树复刻等全新结构证据可重开
updated: 2026-09-06
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

## E5 rsqrt 配方（2026-09-06 00:12，submission 10322，daily_seq 1）

- 载体：generic(E2 plain /) + `_amd`(reciprocal) + `_enflame`/`_kunlunxin`
  （单核 `v * tl.rsqrt(var+eps) * w`）；source `a0298a2`，
  ZIP `e5-a0298a2` SHA-256 `5b1d42b1…c8a`（4 成员）
- screening：远端 `/tmp/flagos-t52e5.6EODJa`（NVIDIA 代理 unittest 5/5 OK
  含 vendor 矩阵；日志 SHA `fc188f5d…`；black diff 仅既有字节工具漂移）
- **终态 6/8 invalid_correctness**：天数 8.5836 / 沐曦 5.2425 / 海光
  8.7104 / 华为 3.7427 / A 8.1713 / B(_amd) 7.9668 过；燧原/昆仑各 1 case 败
- **失败情报（raw_result）**：两芯同败 `test[18]`（33.5M 元素大 case）；
  燧原 1 元素 abs 0.0171（容差 0.015）、昆仑 4 元素 abs 0.0195 + 1 处
  expected=0 的 inf 相对差——第五种数学形态（plain//、div_rn、reciprocal、
  两核拆分、rsqrt）同指纹边界失配
- **判定**：分叉点不在除法形式（T19 单 norm rsqrt 正证据未迁移到
  双 norm + residual 链；疑燧原/昆仑 torch 的 bf16 add/mid 舍入路径
  差异，无目标芯探测通道）。**T52 按 E5 预注册止损封存 6/8**；仅
  `sqrt_rn+div_rn`、torch 归约树复刻等全新结构证据可重开。额度 29/30
