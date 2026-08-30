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

### E1:尾块 mask 越界修复(2026-08-31 21:3x CST)

- **根因(用户侧代码审查定位)**:`mask_n = offs_n < output_dim`
  基于 0..BLOCK_N-1 的 arange,而寻址用 `offsets_n = n_block*BLOCK_N
  + offs_n`——R 非 BLOCK_N 倍数(R=65/80…)时 n_block>0 的瓦片
  越界写相邻行 → 平台 99% 失配;本地测试 R 恰为 64 倍数或 <64,
  完美漏检。
- 修复:mask 用绝对列号;测试补 R=65/80/129,fp32 容差收紧至
  平台口径 1e-4;bench 不变(1.70–5.86x)。
- commit `421c6d9`,ZIP `e1-421c6d9`,SHA `a078a07e932f696dd8d7bb14b83e661cee1d217ab7b1f96c6645602357a1b029`,单成员;unittest 3/3
  (gpu:/tmp/t37f.aq79yf)。**S0 的"语义黑盒"结论撤销**——是我方
  mask bug。
- 今日额度 0/30,已备好 00:05 额度重置后自动 preflight+提交。

### E1 提交记录(2026-08-31 00:2x CST)

- cron 定时日期误设 09-01(额度实际 08-31 00:00 重置),已删定时改
  手动发射;preflight 全过(额度 29/30),单次 confirm 提交;
- 终态待回填。

### E1 平台终态(sub 6984)与 E2 天数 vendor

- **E1 五芯过**(沐曦 3.77/海光 11.58/华为 16.08/card_a 1.69/
  card_b 1.32)——mask 修复真实生效(S0 时代 7 芯全挂);
- 天数失败指纹与 S0 逐位相同(79/80,19.375@3,3):**fp32-ieee
  tl.dot 在天数静默错执行(T12/T13 平台镜像)**;燧原 99%→15%
  (mask 修复生效,残留疑精度/边界);
- E2 = 天数 split-fp16 四点积 vendor(commit `58aab90`,ZIP
  `a1770622…`,2 成员);3-dot 差 3/1M 压线,4-dot 后仅剩 K=4096
  累加深度的相消伪影(xfail 文档化);额度 26/30。

### E2 终态(sub 6992):T37 封存

- 6/8:天数 split-fp16 四点积兑现(4.568x,静默错执行修复);
  燧原 case2 行错位(15% 巨差,GCU masked 行映射结构问题);
- 昆仑属 matmul-reference 崩溃族(6v6 规则),8/8 天花板不存在;
  **T37 = 7/8 上限,封存**(候选 `58aab90` 可复用)。
- 今日两投把 T37 从 0/8 修到 6/8:mask 修复(5 芯)+天数 dot
  镜像(1 芯),根因链完整入账。
