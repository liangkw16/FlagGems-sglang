# Task 34 `per_token_quant_int8` 实验记录

状态:S0 候选就绪

## 契约锁定

- 签名:`reference(x)`(单参);`x [M, N]` 任意浮点 dtype、连续
- 计算:题面 reference 直接委托 T33 group reference 且
  `group_size = N`(整行一组)——语义即 T33:**trunc 截断**
  (题面文字"round"与代码不符,以代码为准)
- 输出:`(x_q [M,N] int8, x_s [M,1] float32)`
- 容差:fp32 1e-4 / bf16 1.5e-2 / fp16 1e-2;八芯

## S0(2026-08-29,commit `0159b26`)

- **直接复用 T33 平台 8/8 验证过的 kernel**(amax→scale→
  `tl.math.div_rn`→clamp→截断 int8),wrapper 单参、整行一组、
  x_s 形状 [..., 1]。
- screening(gpu:/tmp/t34.YAaQC0):unittest 5/5 OK;bench 7/7;
  代理 3.53–6.56x(整行大组比 T33 小组更快:每 program 工作量大)。
  screening 字节与 commit blob 逐项一致。
- ZIP `s0-0159b26/per_token_quant_int8.zip`,SHA
  `e8ea05af9b69de9b08cc69e917102566b1576c962fa55a6ea07a21b5d8f7c3cf`,
  单成员。

### 跨芯风险

- 与 T33 S0 同 kernel(平台 8/8 实证,含昆仑 div_rn 通过);
  唯一新变量是 BLOCK = next_pow2(N) 可达 4096+(大组),T24 昆仑
  大 BLOCK 曲线曾到 4096 不饱和,风险低。
