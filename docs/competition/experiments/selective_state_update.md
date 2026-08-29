# Task 36 `selective_state_update` 实验记录

状态:S0 候选就绪

## 契约锁定

- 签名:`reference(state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False)`
- `state [B,H,dim,N]`;`x/dt [B,H,dim]`;`A [H,N]`(负);`B/C [B,G,N]`
  广播 g=h//(H//G);可选 D/z(silu 门)/dt_bias/softplus;fp32 计算
- 输出 `(y.to(x.dtype), state_new.to(state.dtype))`;输入 state 不变
- 容差:fp32 1e-4 / bf16 1.5e-2 / fp16 1e-2;八芯

## S0(2026-08-29,commit `a14d36b`)

- kernelgen 生成 + 契约重写(生成版 state 双写、while 循环、C 借用
  B stride 三处缺陷):flat 1D capped grid + 块级 div/mod,[64, N]
  tile,fp32 全程,溢出安全 softplus(max+log1p(exp(-|t|))),
  constexpr 旗标,int32,显式输出 cast。
- screening(gpu:/tmp/t36.n5vM5A,字节与 commit blob 逐项一致):
  unittest 4/4(含 D/z/bias/softplus 全组合 × 3 dtype × 6 形状、
  非连续、softplus 极值、2048 大 batch);bench 7/7,代理
  **7.51–8.34x**(fp32 全旗标 3.65x)。
- ZIP `s0-a14d36b`,SHA `744ab4ea66a23f42cc818f7201d48018aecc112788b18b82c943dc53296352f5`,单成员。

### 跨芯风险

- `tl.exp` 平台实证安全(T24/T29);`tl.log`(softplus 路径)与
  `tl.sigmoid` 未有昆仑实证,若昆仑崩溃则 A&S 类多项式替换;
- 归约仅 axis-1 sum(T21 先例,燧原可过)。
