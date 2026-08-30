# Task 37 `sgemm_lora_a` 实验记录

状态:S0 候选就绪

## 契约锁定

- 签名:`reference(x, weights, batch_info, stack_num=1)`
- `x [S,K]`;`weights [num_lora, R, K]`(R=stack_num*r);batch_info
  含 seg_indptr/weight_indices/permutation(可选)
- 每 segment:`out[rows] = x[rows].float() @ weights[w].float().T`,
  转回 x.dtype;输出 `[S,R]`,torch.zeros 起步
- 容差:fp32 1e-4 / bf16 1.5e-2 / fp16 1e-2;八芯

## S0(2026-08-31,commit `4b77dae`)

- T23/T28 家族模板:1D capped grid 折叠(constexpr 块数除法)、
  fp32-ieee dot、BLOCK 64/64/64 + stages=2、permutation 可选、
  zeros 基底;max_len 缺失时 host 端由 seg_indptr 差分。
- screening(gpu:/tmp/t37.1k5OeX,字节与 blob 一致):unittest 3/3
  (3 dtype × 7 形状含 permutation/空段/S=1);bench 5/5,代理
  **1.71–5.82x**(大 R×K 档最弱,调参余量在 BLOCK_N/K)。
- 风险:昆仑(家族 0/3 前科:matmul-reference 崩溃族),fp32-ieee
  dot 是 T12 平台验证过的昆仑兼容路径。

### S0 提交记录(2026-08-31 11:0x CST)

preflight 全过(tid `s2t1op037`,额度 5/30 消耗 1 → 4/30);单次
confirm 提交,评测入队,逐芯结果待回填。

### S0 平台终态(sub 6927,2026-08-31 11:2x CST)

- 七芯全部数值失败(~99% 元素失配,最大绝对差 19–51)——系统性
  语义差异,非精度/调参问题;疑平台真实 reference 与题面摘录不符
  (T36 同款黑盒:疑点 permutation 方向、stack_num>1 语义、
  batch_info 附加字段),代理无法复现(本地参照恒绿)。
- **按 T36 教训立即止损**(1 发即停),额度 4/30;恢复条件:
  赛方公开真实 reference 或他队结构证据。
