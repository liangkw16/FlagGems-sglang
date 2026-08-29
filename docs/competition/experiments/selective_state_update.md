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

### S0 平台终态(sub 6356,2026-08-29 16:4x)

**8/8 全部数值失败**(逐芯同指纹):5 个 case 各 ~80% 元素超差。
- 失败签名:y 首元素精确、其余中等无规偏差(最大绝对差 1.19、
  最大相对差 42x)——与"平台传入 vllm 式 1-D [nheads] D/dt_bias、
  kernel 按 [nheads,dim] 索引越界读垃圾(H=1 时 p=0 恰好读对)"
  完全吻合;题面快照摘录的 reference 本身不可运行(dA 广播错),
  已证其与真实 harness 有出入。
- 代理无法复现(本地测试均按题面 2-D 形状构造)。

### E1:1-D 形状归一(commit `a5b6f04`)

- wrapper 将 1-D A/D/dt_bias 广播展开为 2-D(纯形状处理);
  测试参照同步归一;unittest 5/5(新增 1-D 变体用例)。
- ZIP `e1-a5b6f04`,SHA `f35e70c2e843820943ec6042f86cafc405c28ccec1cb575a36faaa39f2d9eee1`。额度 12→11/30。
