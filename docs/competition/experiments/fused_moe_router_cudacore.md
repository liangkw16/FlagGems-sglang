# Task 26 `fused_moe_router_cudacore` 实验记录

状态:S0 候选就绪,待额度重置后提交(排在 29→30→25(e1)→28→27 之后)

## 契约锁定

- 签名:`reference(x, router_weight, topk, moe_softcapping, correction_bias=None)`
- 输入:`x [B, H]`;`router_weight [E, H]`;`topk` int;`moe_softcapping`
  float(0 不启用);`correction_bias [E]` float32 或 None
- 计算:fp32 GEMM logit → 可选 tanh 软封顶 → 可选 bias → 全 E softmax →
  top-k(`argsort(logits, descending=True)[:, :topk]`)→ gather 权重
- 输出:`(topk_weights [B, topk] fp32, topk_ids [B, topk] int32)`
- 容差:fp32 1e-4 / bf16 1.5e-2 / fp16 1e-2;`topk_ids` 精确相等
- 关键语义:权重来自全 E softmax,不对 topk 子集重归一化;topk 可大于 2

## 方案

- 不用 tl.dot(cudacore 变体):H 分块 FMA 归约,累加顺序贴近 torch fp32 GEMM
- E 整块加载,行内全 E softmax(数值稳定 max 减);topk 迭代抽取
- 风险:topk>2 argsort 精确匹配、近平局翻转;排 T27 之后做

## 提交预算与止损(2026-08-27 定稿)

- 每题 5 次提交预算;S0 首投探路 → 最多 3 次 vendor 单变量迭代 → 剩 1 次留作
  截止前回归储备。
- 同指纹失败连续 2 次提前停,不烧满 5 次;额度只花在有单变量假设的候选上。
- generic dot 策略(如涉及 `tl.dot`):fp32-ieee 操作数 generic + `_tianshu`
  split-fp16 vendor;昆仑保持 fp32-ieee(T12 镜像证据,昆仑 fp16-dot 数值失败
  有平台实证)。

## S0:generic baseline(split-K + 向量化 topk)

状态:候选就绪,未提交(额度阻塞)
时间:2026-08-27 22:30–23:55 CST
source/verification commit(同一提交):`3642a65ac1dfaafdf4f39a053924aad9dd6af57f`

### 构建身份

| 项目 | 值 |
| --- | --- |
| 源文件 | `src/flaggems_sglang/ops/fused_moe_router_cudacore.py` |
| 源文件 SHA-256 | `743d5c52a4604bd4b589d2621564f6dac012fc807b213123cfd66f58d0f0e982` |
| 测试 SHA-256 | `34262a3c9912c73b2895f78def4e62732a4d02349152cf403c336627015f2981` |
| ZIP | `artifacts/competition/fused_moe_router_cudacore/s0-3642a65/fused_moe_router_cudacore.zip` |
| ZIP SHA-256 | `8ff7a761ef0b0614ee759264c75504e6f70f90171befaac2a2298e0896ecdb55`(与 canonical 一致) |
| screening 目录 | `gpu:/tmp/flagos-batch3-rest.oTBskH/fused_moe_router_cudacore`(round2) |
| release 目录 | `gpu:/tmp/flagos-rel3.Fp3vo7/fused_moe_router_cudacore`,文件取自 Git 对象 |

### 候选配置

- 与 T27 共用 split-K GEMM + 向量化归约骨架;topk 泛化为
  `tl.static_range(TOPK)` 轮选:selected 布尔掩码逐轮累积,每轮
  axis-1 max + 首索引,全向量化无串行扫描;topk 1–8。
- 生成器缺陷修复记录:排除链曾用 Python `or`(zsh 生成 + `|` 优先级
  低于 `==` 的双重陷阱),round1 49 错;改为掩码累积方案后 8/8。

### 正确性

screening(round2)与 release 两次均 8/8:dtype × (B,E,H)(含非 64 倍 H)
× topk {1,2,3,4,8} × softcap/bias;近平局次序;非连续;不变性;B=0;
70000 行折叠。

### 远端 NVIDIA 代理性能(五组 AB/BA p50 中位数)

| dtype | B×E×H | op p50 (ms) | torch p50 (ms) | speedup |
| --- | ---: | ---: | ---: | ---: |
| float16 | 64×256×4096 | 0.083968 | 0.049152 | 0.5854x |
| float16 | 1024×256×4096 | 0.288768 | 0.193536 | 0.6702x |
| float16 | 4096×256×4096 | 0.960576 | 0.612384 | 0.6375x |
| float16 | 4096×64×2048 | 0.137216 | 0.184320 | 1.3433x |
| float16 | 16384×32×1024 | 0.260096 | 0.331856 | 1.2759x |
| float32 | 64×256×4096 | 0.088128 | 0.038912 | 0.4415x |
| float32 | 1024×256×4096 | 0.280576 | 0.156672 | 0.5584x |
| float32 | 4096×256×4096 | 0.927840 | 0.474176 | 0.5111x |
| float32 | 4096×64×2048 | 0.139360 | 0.130176 | 0.9341x |
| float32 | 16384×32×1024 | 0.251904 | 0.227328 | 0.9024x |

最差 0.4415x(topk=8 轮选略慢于 top2,E=256 fp32)。

### 已知边界与 E1 假设

- 同 T27:E ≤ 256;天数 fp32-ieee dot vendor 预案;平局分歧风险。
- topk=8 小 B(64×256)轮选 8 轮 axis 归约有固定开销,E1 候选:轮间
  早停(slot ≥ topk 无需)与 BLOCK_R 适配。

### 提交计划

- preflight tuple:season 2、race `782kzq4m`、account `15600308080`、
  team `SoulCoder`、batch 3、task 26、operator
  `fused_moe_router_cudacore`、stage `s0`、commit
  `3642a65ac1dfaafdf4f39a053924aad9dd6af57f`、ZIP
  `artifacts/competition/fused_moe_router_cudacore/s0-3642a65/fused_moe_router_cudacore.zip`、
  SHA-256 `8ff7a761ef0b0614ee759264c75504e6f70f90171befaac2a2298e0896ecdb55`、
  member `fused_moe_router_cudacore.py`。

## 平台提交记录

- 2026-08-28 00:07 CST 额度重置(30/30)后,按 29→30→25→28→27→26 顺序自动
  preflight + 一次性提交;全部 tuple 与账本一致后执行 confirm。
- 提交时间约 2026-08-28 00:33 CST;submission_id `5743`;ZIP SHA-256
  `8ff7a761ef0b0614ee759264c75504e6f70f90171befaac2a2298e0896ecdb55`;state `submitted`、validity `pending`、评测入队。
- 提交后团队当日额度剩余 24/30(6 投全记录)。


### 八芯结果(S0 首投,sub 5743,截至 00:55,昆仑芯评测中)

与 T27 同指纹:5 过 2 败(tianshu topk ids mismatch、huawei
MLIRCompilationError),kunlunxin 评测中。E2 vendor 计划同 T27。
