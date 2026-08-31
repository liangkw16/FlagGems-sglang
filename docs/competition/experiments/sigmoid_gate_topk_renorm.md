# Task 38 `sigmoid_gate_topk_renorm` 实验记录

```current
task: 38
operator: sigmoid_gate_topk_renorm
batch: 3
validity: invalid
platform: 7/8
team_best_stage: S0
team_best_commit: 311570f
blockers: 昆仑 topk 族 Segfault(崩溃族第15例)
sealed: yes
next: 平台工单;候选封存可复用
updated: 2026-08-31
```

状态:S0 候选就绪

## 契约锁定

- 签名:`reference(logits, k, n_shared_experts, route_scale, global_scale, bias)`
- `logits [T, N+S]`(末 S 列共享);`bias [N]`;`global_scale [1]` fp32
- sel=sigmoid(routed)+bias → topk(argsort 降序);routed_vals 取
  **原始 logit**(非 sigmoid);probs=sigmoid(cat(routed_vals,shared));
  归一化 ×route_scale×global_scale;输出
  `(routed_weights [T,k] 输入dtype, indices [T,k] int32, shared_weights [T,S])`
- indices 精确匹配;容差 fp32 1e-4 / bf16 1.5e-2 / fp16 1e-2;八芯

## S0(2026-08-31,commit `311570f`)

- T31 机器(迭代 argmax+min-index 平局)**双确定性选轮**:第一轮
  indices + sigmoid 概率总和,第二轮复现选点写归一化权重——规避
  同 program store→load 可见性风险;共享专家恒活跃参与归一化;
  1D capped grid。
- screening(gpu:/tmp/t38.z8kOOd,字节与 blob 一致):unittest 3/3
  (3 dtype × 7 形状含 S=0/S>T 边界);bench 4/5 shape 1.42–2.61x,
  65536×64 小 N 档 0.536x(两轮选点在大 T 小 N 下偏慢,过门槛,
  平台 mix 待证)。
- 风险:topk 族昆仑前科(T25/26/27/31 皆崩);fp32 无 dot、无
  libdevice 超越函数(sigmoid=1/(1+exp))是本次差异点。

### S0 提交记录(2026-08-31 12:2x CST)

preflight 全过(tid `s2t1op038`,额度 4/30 消耗 1 → 3/30);单次
confirm 提交,评测入队,逐芯结果待回填。

### S0 平台终态(sub 6930,2026-08-31 14:0x CST)

**7/8**——七芯全过(卡_B 1.854/沐曦 1.580/华为 1.255/海光 1.215/
card_a 1.146/天数 0.819/燧原 0.421),**昆仑 1830s 验证段
Segmentation fault**(崩溃族第 15 例,与 Aborted 同族不同信号量)。

- topk 族昆仑结论第三次复现(T25/26/27/31/38);本 kernel 无 dot、
  无 libdevice 超越函数仍崩——"最干净形态"也未能幸免,题内止损
  (1 发即停,候选 `311570f` 封存可复用)。
- 第三批 17 题全部处置完毕:8 valid + 3 题 7/8 昆仑墙封存 +
  2 题语义黑盒止损 + 4 题早前止损。额度 3/30。
