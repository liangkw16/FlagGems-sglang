# Task 35 `rotary_embedding` 实验记录

状态:S0 候选就绪(首 screening 全绿)

## 契约锁定

- 签名:`reference(x, cos, sin, interleaved)`;`x [T,H,D]`
  bf16(兼容 fp16/fp32);`cos/sin [T, D//2]`
- 计算(题面 reference 忽略 interleaved,恒偶奇对):x1=x[...,0::2]、
  x2=x[...,1::2];o1=x1*c-x2*s;o2=x1*s+x2*c;交错重组,fp32 计算,
  转回 x.dtype
- 容差:fp32 1e-4 / bf16 1.5e-2 / fp16 1e-2;八芯;纯 Triton

## S0(2026-08-29,commit `5406218`)

- kernelgen `generate_kernel` 生成 + 三处契约修正:while→for range
  (已知编译雷)、恒偶奇对拆分(生成版 False 分支半宽拆分与题面
  reference 不符)、store 显式 cast。
- 结构:每 program 一 (t,h) 行,constexpr HEADS 除数,偶奇 strided
  读写,1D capped grid-stride,int32。
- screening(gpu:/tmp/t35.Pw0Tob):unittest 5/5 OK;bench 7/7;
  代理加速比:4096×32×128 **11.79x**、128×4×192 4.56x、
  256×8×128 4.20x、65536×1×64 4.13x、16×32×128 4.03x、
  1024×16×128 3.72x。
- ZIP `s0-5406218/rotary_embedding.zip`,SHA
  `7f768cc3ef4896a6ad4c3326a2c9cd37406d034d209ecdc11fd604cca8482b81`,
  单成员。

### 跨芯风险

- 无超越函数、无归约、无 dot;strided 读写是常规模式;风险面小。
- 小 shape launch 主导(代理 ~4x 与 T30 平铺同水位)。

### S0 提交记录(2026-08-29 17:1x CST)

screening 字节与 commit blob 逐项一致(晋级验签 ✓);preflight 全过
(tid `s2t1op035`,额度 15/30 消耗 1 → 14/30);单次 confirm 提交,
评测入队。
