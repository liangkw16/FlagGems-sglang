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

### S0 平台终态(sub 6341,2026-08-29 17:2x CST)

**8/8 全过,valid,平均 4.66725x**(榜首 MakeYUNAGreatAgain 5.8830x,
-20.6%,首投即达标)。

| 芯片 | speedup |
| --- | ---: |
| tianshu | 9.986 |
| card_b | 8.987 |
| card_a | 8.354 |
| haiguang | 4.936 |
| muxi | 3.793 |
| huawei | 0.715 |
| kunlunxin | 0.298 |
| enflame | 0.269 |

短板三芯与 T29/T33 同画像;E1 假设:stride-2 偶奇双读在弱芯代价高,
改整行连续读 + 寄存器 `tl.reshape/tl.split/tl.join` 解交错。

### E1 负结果(未提交,2026-08-29)

连续整行读 + `tl.reshape/tl.split/tl.join` 寄存器解交错候选:代理
全面回退(4096×32×128 11.79→9.31x,小 shape 4.20→3.34x,余 -3%)。
NVIDIA 上 stride-2 访存硬件合并良好;且 T30 的逐元素 gather 在
燧原 8.7x/昆仑 5.8x,反证"非连续访问是弱芯瓶颈"。假设不支持,
不提交。T35 弱芯(燧原 0.27/昆仑 0.30/华为 0.72)无已证杠杆,
本题收敛:S0 = 团队最佳 4.6673x(-20.6%)。

### E1:四头瓦片 + cos/sin 复用(sub 6xxx,2026-08-31 00:1x CST)

- codex 收窄后的可迁移模型:真实全局数据复用(cos/sin 只依赖
  (token,col),每瓦片载一次广播到 4 头)+ 保持 stride-2 访问;
  昆仑 vendor 钉死 S0 字节兜底,新 generic 只达其余七芯;
- commit `581fc8c`,ZIP `8ea6dc5a…`,2 成员;
- 代理:+3%~+62%(4096×32×128 11.78→15.59x、1024×16×128
  3.76→6.09x)全 shape 无回退,unittest 5/5;preflight 全过。

### E1 平台终态(sub 6982)与 E2(2026-08-31 01:1x CST)

- **E1 七芯全过且大涨**:沐曦 3.79→**6.43(+70%)**、海光 4.94→
  **8.97(+81%)**、天数 9.99→10.33、card_a 8.35→9.59、card_b
  9.10;华为 9 case NaN(99.6%,[4,64] 2D 广播瓦片触发昇腾
  lowering 问题)——cos/sin 复用模型七芯验证成立;
- E2 = 华为钉 S0 旧字节 vendor(commit `7ed663c`,ZIP
  `90e429be…`,3 成员);预计 8/8 且平均 ~6.7(现 4.667)。
- 额度 27/30。

### E2 平台终态(sub 6990,2026-08-31 01:4x CST)

**8/8 valid,平均 5.82525x —— 新团队最佳(S0 4.6673 → +24.8%)。**

| 芯片 | S0 | E2 | 变化 |
| --- | ---: | ---: | ---: |
| tianshu | 9.986 | 10.385 | +4% |
| **muxi** | 3.793 | **6.448** | **+70%** |
| **haiguang** | 4.936 | **9.332** | **+89%** |
| card_a | 8.354 | 9.917 | +19% |
| card_b | 8.987 | 9.148 | +2% |
| huawei(S0 vendor) | 0.715 | 0.679 | 持平 |
| enflame | 0.269 | 0.397 | +48%(新 generic) |
| kunlunxin(S0 vendor) | 0.298 | 0.295 | 持平 |

- cos/sin 复用 + 四头瓦片模型六芯兑现;剩余短板燧原 0.40/昆仑 0.30
  /华为 0.68(后两者为钉死的 S0 兜底字节)。
- 距榜首 EvokeAgent 8.488 收窄至 **-31.4%**;额度 26/30。
