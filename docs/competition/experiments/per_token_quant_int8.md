# Task 34 `per_token_quant_int8` 实验记录

```current
task: 34
operator: per_token_quant_int8
batch: 3
validity: valid
platform: 8/8
team_best_stage: e1
team_best_speedup: 4.7131
sealed: no
next: e3 大行数 row-pack 已提交待评测(代理 2.67x@65536x128,ZIP 4c770895)
updated: 2026-08-31
```

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

### S0 平台终态(sub 6344,2026-08-29 17:5x CST)

**8/8 全过,valid,平均 4.707425x**(榜首 c2flow 5.2281x,-9.9%)。

| 芯片 | speedup |
| --- | ---: |
| haiguang | 7.689 |
| tianshu | 7.477 |
| card_b | 6.238 |
| card_a | 5.599 |
| muxi | 4.099 |
| **enflame** | **3.804**(T33 小组时仅 0.45——整行大组正对燧原) |
| huawei | 2.147 |
| kunlunxin | 0.607 |

- 跨题知识:燧原对"每 program 工作量"敏感——小组(≤256)调度压垮,
  整行大组(≥2560)反而 3.8x;昆仑仍是短板但 0.61x 已过门槛。
- 额度 13/30;后续单变量轴:昆仑 vendor(大 BLOCK 曲线在整行组下
  未探)、huawei 2.1x 结构轴。

## E1:华为两趟列分块 BLOCK 512 vendor(sub 7217,2026-08-31)

- 假设:S0 的 next_pow2(N) 全行单趟瓦片(可达 8192 宽)远超昇腾
  偏好;T24 平台已证 BLOCK 512。vendor = 每行两趟列分块扫描
  (constexpr 列循环静态展开、分块 running max 数学不变、div_rn 保留),
  commit `088a916`;
- screening(gpu:/tmp/flagos-catchup.NOF9kN):unittest 5/5;代理上
  华为专属 vendor 回退 -4~-17%(两趟读 2x 流量的必然代价,
  vendor-only 不阻断);release(gpu:/tmp/flagos-rel2.bskkJw)5/5;
- ZIP `e1-088a916`,SHA `c0833690…`,2 成员;preflight 17/30。

### E1 终态(sub 7217)

**8/8 valid,平均 4.71314167x —— team best(S0 4.7074 → +0.1%)。**
华为 2.147→**1.847(-14%,预注册门失败)**;平均微升来自其余芯噪声。
**结论:两趟列分块在昇腾证伪——T24 的 BLOCK 512 pointwise 证据
不可迁移到两趟归约结构;ascend 两趟轴关闭。**

## E2:昆仑两趟列分块 BLOCK 1024 vendor(sub 7218)

- 同构 BLOCK 1024(T21 昆仑归约唯一成功轴的尺寸),commit `a9a860f`,
  ZIP `b57c28dd…`,3 成员(e1 包 + kunlunxin)。

### E2 终态(sub 7218):昆仑正确性失败

- **invalid_correctness**——昆仑芯 passed=False(无 speedup),
  其余七芯正常(华为 1.994 为 e1 vendor 噪声波动);
- 跨芯知识:**昆仑对"两趟列分块 + static_range + 分块 running max"
  结构正确性失败**(单趟全行 S0 在昆仑正确)——非尺寸问题,
  kunlun 两趟轴关闭;team best 保持 e1 4.7131。

## E3:大行数 row-pack 静态分派(2026-08-31 23:5x CST)

状态:候选就绪待单次提交。41 算子核验建议的"小行多 row"轴,经两轮
screening 修正为"仅大行数打包"。

### 假设与修正过程

- 单遍行级 amax 要求整行进同一 program → 唯一单遍泛化是 [ROWS, N]
  行打包;分派为 host 静态选择(规避 T32 e4 动态谓词学费);
- 第一版按"每 program ~2048 元素"打包,小行数形状下溢填充
  (1024 行 ÷16 = 64 个 program 喂不满 GPU,回退 5-12%)——
  **打包只在行数巨大时有利**(少而肥的 program 赢调度开销);
- 终版规则:total_rows ≥ 16384 才启用,ROWS = min(2048//pow2(N), 16),
  其余形状逐字节走 S0 kernel。

### screening 证据(RTX 5070 Ti,gpu:/tmp/flagos-t34e3b.92fS64)

- 日志 SHA-256 `0219e116396648bb44eb767a569797b7f35cf3dca3d9dd8ab375c8c1062d368d`;
  第一版(含下溢填充教训)目录 gpu:/tmp/flagos-t34e3.2Zd3Un,日志
  SHA-256 `e75be02285346a8ab779562f59419440fff02c1688f8f40bca05c783c484cd29`;
- py_compile/isort/flake8/unittest **5/5**;数值 **NUMERIC_FAILS=0**
  (16 形状 × 3 dtype 全部与 HEAD 对照逐位一致);
- 性能(五轮 AB/BA median,新/旧):**65536×128 `0.375`(2.67x)**、
  65536×256 `0.488/0.479`(2.05x)、32768×512 `0.671/0.704`(1.5x)、
  16384×64 `0.619/0.602`(1.65x);65536×1024/×4096 与全部小形状
  ~1.0(S0 路径,纯噪声);分派边界 16383×128 = 1.01 正确回落。

### 构建身份与 release

| 项目 | 值 |
| --- | --- |
| source / verification commit | `969a34cf1bbd44f78207bd2dcf17131a19a0d50b` |
| generic SHA-256 | `9ed3eb8ddad614c5fce18bb82a51622516ddc3688c19855d6b0414c651b6f05a` |
| test SHA-256 | `cd52f72f8316f268ffbeff86e4d60a65109394e5b509db34f5737c8167ea6f23` |
| canonical ZIP | `e3-969a34c/per_token_quant_int8.zip`,11058 bytes,SHA-256 `4c77089517be3b9727adcbb107a63a6fcaf876de9a6e3931398c87c25a270d52` |
| ZIP 成员 | generic + `_ascend`(e1) + `_kunlunxin`(S0 钉死);`unzip -t` 无错,成员哈希与 blob 逐项一致 |
| release | gpu:/tmp/flagos-t34e3-rel.cv5rpG,unittest OK、py_compile OK,release.log SHA-256 `3c8d0f575dfbb8424396b55ee9c6ebae569bf83428743081b252ba110d9ba2c2` |

### 平台预注册门

- 基础门:8/8 valid。
- 晋级门(team best):平均严格高于 e1 `4.71314167x`。
- 机制门:generic 路由五芯合计较 e1 读数均值 +5%(大行数 shape 存在
  则应显著兑现,参照 T33 e8 的代理→平台传导)。
- stop gate:平均回落 >3% 且无单芯 ≥ e1 → 打包分派证伪,回滚 S0;
  华为/昆仑 vendor 字节未变,读数按噪声/水位处理不归因。
