# Task 27 `fused_moe_router_tensorcore` 实验记录

状态:S0 候选就绪,待额度重置后提交(排在 29→30→25(e1)→28 之后)

## 契约锁定

- 签名与计算语义与 cudacore 变体完全相同,区别:底层用 `tl.dot`,
  且 `topk <= 2`、`H` 须为 64 的倍数
- 输出:`(topk_weights [B, topk] fp32, topk_ids [B, topk] int32)`
- 容差:fp32 1e-4 / bf16 1.5e-2 / fp16 1e-2;`topk_ids` 精确相等

## 方案

- generic fp32-ieee `tl.dot`(昆仑可过、天数静默失败 → `_tianshu`
  split-fp16 三点积 vendor,fp32 1e-4 容差,T12 已验证镜像规则)
- topk≤2 → 行内两遍扫 max/second-max;近平局次序需与 torch.topk 一致,
  代理验证先测 tie 行为
- H%64 对齐 BLOCK_SIZE_K

## 提交预算与止损(2026-08-27 定稿)

- 每题 5 次提交预算;S0 首投探路 → 最多 3 次 vendor 单变量迭代 → 剩 1 次留作
  截止前回归储备。
- 同指纹失败连续 2 次提前停,不烧满 5 次;额度只花在有单变量假设的候选上。
- generic dot 策略(如涉及 `tl.dot`):fp32-ieee 操作数 generic + `_tianshu`
  split-fp16 vendor;昆仑保持 fp32-ieee(T12 镜像证据,昆仑 fp16-dot 数值失败
  有平台实证)。

## S0:generic baseline(split-K + 向量化)

状态:候选就绪,未提交(额度阻塞)
时间:2026-08-27 22:30–23:55 CST
source/verification commit(同一提交):`d44c85bad974cf2c920fc04cb145a9bc78b2f999`

### 构建身份

| 项目 | 值 |
| --- | --- |
| 源文件 | `src/flaggems_sglang/ops/fused_moe_router_tensorcore.py` |
| 源文件 SHA-256 | `102d07d2d15dab579c04aff1f5f06c00cfb810f9bd3055bb08ac3474d8bbb56f` |
| 测试 SHA-256 | `fbd727711e09b35a7a72a3e913edbc9867776d0810020b6876eca2e4573aa79a` |
| ZIP | `artifacts/competition/fused_moe_router_tensorcore/s0-d44c85b/fused_moe_router_tensorcore.zip` |
| ZIP SHA-256 | `a3d7e7a5fbc9f8085769fcd5a0bc1d491a9ab91302790b19c4a5506815d31521`(与 canonical 一致) |
| screening 目录 | `gpu:/tmp/flagos-batch3-rest.oTBskH/fused_moe_router_tensorcore`(round2) |
| release 目录 | `gpu:/tmp/flagos-rel3.Fp3vo7/fused_moe_router_tensorcore`,文件取自 Git 对象 |

### 候选配置与演化

- kernel A:split-K GEMM,fp32-ieee `tl.dot`,64/64/64 tile + `num_stages=2`
  (燧原规则),K 按 512 分片、至多 8 片,partials `[SK,B,E]` fp32 工作区;
  1D grid 65535 折叠。
- kernel B:整行向量化归约(BLOCK_R=32 × E_pow2≤256,axis-1 max/exp-sum/
  top2),softcap 用 T24 双形式 tanh(近零五阶奇多项式 + 稳定 exp;
  Triton 3.7 无 `tl.tanh`,round1 实证),bias 后置;平局首索引。
- 设计演化:初版两 kernel 串行 chunk 扫描 + 无 split-K,代理最差仅
  0.124x(小 B 单 tile);split-K + 向量化后 0.518x(4.2 倍)。

### 正确性

screening(round2)与 release 两次均 8/8:fp32/fp16/bf16 × (B,E,H) 矩阵
(H 含非 64 倍数 100/192/200/333 由 K mask 支持)× topk 1/2 × softcap/bias
四组合;近平局次序断言;精确平局 torch.topk 顺序为设备实现细节,记为
已知分歧风险(随机输入下概率≈0),不逐字断言;非连续;输入不变性;
B=0;70000 行折叠。

### 远端 NVIDIA 代理性能(五组 AB/BA p50 中位数)

| dtype | B×E×H | op p50 (ms) | torch p50 (ms) | speedup |
| --- | ---: | ---: | ---: | ---: |
| float16 | 64×256×4096 | 0.051200 | 0.047136 | 0.9206x |
| float16 | 1024×256×4096 | 0.258144 | 0.191520 | 0.7419x |
| float16 | 4096×256×4096 | 0.935968 | 0.607232 | 0.6488x |
| float16 | 4096×64×2048 | 0.133184 | 0.183296 | 1.3763x |
| float16 | 16384×32×1024 | 0.255968 | 0.330816 | 1.2924x |
| float32 | 64×256×4096 | 0.057408 | 0.037888 | 0.6600x |
| float32 | 1024×256×4096 | 0.251968 | 0.155648 | 0.6177x |
| float32 | 4096×256×4096 | 0.903232 | 0.468000 | 0.5181x |
| float32 | 4096×64×2048 | 0.136256 | 0.130048 | 0.9544x |
| float32 | 16384×32×1024 | 0.245824 | 0.224256 | 0.9123x |

最差 0.5181x(E=256 fp32;fp32-ieee dot 对 cuBLAS SGEMM 的固有差距,
门槛 0.1x 有 5 倍余量)。

### 已知边界与 E1 假设

- E ≤ 256(整行归约单 program 约束),超出即 assert 失败可见。
- 若平台某芯 fp32-ieee dot 不下沉(天数静默失败风险),E1 = `_tianshu`
  split-fp16 三点积 vendor(T12 已验证配方)。
- 精确平局 topk_ids 与 torch 分歧风险同 T26,见上。

### 提交计划

- preflight tuple:season 2、race `782kzq4m`、account `15600308080`、
  team `SoulCoder`、batch 3、task 27、operator
  `fused_moe_router_tensorcore`、stage `s0`、commit
  `d44c85bad974cf2c920fc04cb145a9bc78b2f999`、ZIP
  `artifacts/competition/fused_moe_router_tensorcore/s0-d44c85b/fused_moe_router_tensorcore.zip`、
  SHA-256 `a3d7e7a5fbc9f8085769fcd5a0bc1d491a9ab91302790b19c4a5506815d31521`、
  member `fused_moe_router_tensorcore.py`。

## 平台提交记录

- 2026-08-28 00:07 CST 额度重置(30/30)后,按 29→30→25→28→27→26 顺序自动
  preflight + 一次性提交;全部 tuple 与账本一致后执行 confirm。
- 提交时间约 2026-08-28 00:29 CST;submission_id `5742`;ZIP SHA-256
  `a3d7e7a5fbc9f8085769fcd5a0bc1d491a9ab91302790b19c4a5506815d31521`;state `submitted`、validity `pending`、评测入队。
- 提交后团队当日额度剩余 24/30(6 投全记录)。


### 八芯结果(S0 首投,sub 5742,截至 00:55,昆仑芯评测中)

已出 7 芯:5 过 2 败(与 T26 同指纹):

- 通过:muxi、haiguang、huawei? 否——见下表(以平台为准):
  tianshu 失败 `topk expert ids mismatch`(fp32-ieee dot 静默算错,
  T12 镜像证据);huawei 失败 `MLIRCompilationError`(`input_precision=
  "ieee"` 或 split-K 结构在昇腾不支持编译);其余 muxi/haiguang/card_a/
  card_b 通过,kunlunxin 评测中。

### E2 计划(预算剩 4 次)

- `_tianshu`:split-fp16 三点积 dot;
- `_huawei`:去掉 `input_precision="ieee"`(用默认精度需过 fp32 1e-4
  容差评估,或 split-fp16)。

## E1 vendor 轮提交(sub 5765,2026-08-28 01:4x CST)

- 首轮逐芯失败指纹对应的 vendor 修复;vendor commit
  `0e3d58715ec0c5b1d3b841e2cf6b277b48fd8f9c`;ZIP SHA-256 `45fc5be6ab332db9cd825a3e2b0b1bd17a24769ab66bbb54cb74be50af2fc535`。
- 成员:generic + `fused_moe_router_tensorcore_iluvatar.py` /
  `_ascend.py`(均 split-fp16 三点积,兼顾天数数值与华为编译)。
- 远端 NVIDIA 代理:router/lora/enflame/ascend vendor 数值全对;
  `_kunlunxin` last-index 在 NVIDIA 失配为设计内现象(NVIDIA torch 平局
  取首索引)。
- 提交后当日额度 20/30 剩余;评测中。

## E2 第二轮 vendor(sub 5802,2026-08-28 03:0x CST)

- E2/e2 结果回填:T28 天数+燧原已修(split-fp16 与 1D 无分支模板均有效);
  T27/T26 天数已修;华为失败根因确认为 **UB overflow(2.89M/1.57M bits)**,
  即 batch-2 已知昇腾 UB tile 上限,非 input_precision;T25 昆仑 actual 为
  未初始化垃圾(标量访存静默失效),燧原编译失败落在 finalize i64 标量 store。
- 本轮修复:router `_ascend` reduce BLOCK_R 32→8;draft `_ascend` 改纯拷贝
  + finalize 融合列写入;draft `_enflame`/`_kunlunxin` 改行内串行 argmax,
  lane 向量 store、无工作区、无标量访存(kunlunxin 回归首索引)。
- vendor commit `e62a27eb7f41819461fb981c44adf397f25b8729`;远端代理 9/9 绿。
- 评测中;额度 17/30。

