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

## E1 vendor 轮提交(sub 5766,2026-08-28 01:4x CST)

- 首轮逐芯失败指纹对应的 vendor 修复;vendor commit
  `0e3d58715ec0c5b1d3b841e2cf6b277b48fd8f9c`;ZIP SHA-256 `7a24b21933e659880ccaa5c987aedab0792b601effa011bdbfb5acaf84ee7199`。
- 成员:generic + `fused_moe_router_cudacore_iluvatar.py` /
  `_ascend.py`(split-fp16 三点积)。
- 远端 NVIDIA 代理:router/lora/enflame/ascend vendor 数值全对;
  `_kunlunxin` last-index 在 NVIDIA 失配为设计内现象(NVIDIA torch 平局
  取首索引)。
- 提交后当日额度 20/30 剩余;评测中。

## E2 第二轮 vendor(sub 5803,2026-08-28 03:0x CST)

- E2/e2 结果回填:T28 天数+燧原已修(split-fp16 与 1D 无分支模板均有效);
  T27/T26 天数已修;华为失败根因确认为 **UB overflow(2.89M/1.57M bits)**,
  即 batch-2 已知昇腾 UB tile 上限,非 input_precision;T25 昆仑 actual 为
  未初始化垃圾(标量访存静默失效),燧原编译失败落在 finalize i64 标量 store。
- 本轮修复:router `_ascend` reduce BLOCK_R 32→8;draft `_ascend` 改纯拷贝
  + finalize 融合列写入;draft `_enflame`/`_kunlunxin` 改行内串行 argmax,
  lane 向量 store、无工作区、无标量访存(kunlunxin 回归首索引)。
- vendor commit `e62a27eb7f41819461fb981c44adf397f25b8729`;远端代理 9/9 绿。
- 评测中;额度 17/30。

## E3 第三轮 vendor(sub 5806,2026-08-28 04:0x CST)

- E3 结果:华为 UB 溢出已修(可编译)但 split-fp16 在昇腾数值错误(昆仑
  fp16-dot 的镜像);T25 华为失败为 E3 wrapper 顺序 bug(draft=None 时
  finalize 未执行→topk_index 垃圾);T25 燧原三轮不同结构均挂在
  tl.max/tl.min 轴归约(对照 T21 tl.sum、T28 无归约均过)。
- 本轮:router `_ascend` 回归 fp32-ieee(保留 BLOCK_R=8);draft `_ascend`
  finalize 提升到函数级;draft `_enflame` 改全展开 reshape/split 两两树
  (纯 where 元素运算,零 max/min 归约)。
- vendor commit `91838b855b3691e220f5bb03eeac48b1cdf4b6f2`;远端代理 9/9 绿;
  额度 14/30;T27/T26/T25 均剩最后 1 次预算。

## E4 终投(sub 5809,2026-08-28 04:5x CST)

- E4 结果:华为 fp32-ieee 仍 topk ids 失配 → 剩余唯一假设为 stages=2
  流水化数值 bug;终投镜像 batch-2 平台已证的 `chunk_state_ascend`
  配置(ieee dot + `num_stages=1`),vendor commit
  `7001c8156d098a130cfc9a238c943008661f5760`。本题预算 5/5 用尽。
- E4 伴随记录:T25 华为 0.322x 修复(wrapper 顺序 bug);T25 燧原第 4 种
  结构(纯 where 树)仍编译失败 → 按同指纹 4 连败止损,T25 燧原轴永久
  停止,T25 封顶 7/8、无法 valid,保留最后 1 次额度不投。

## 终态(2026-08-28 05:1x)

- sub 5808/5809:华为在 `num_stages=1` + ieee(镜像已证 chunk_state_ascend
  配置)下仍 topk ids 失配 → 失败面收敛到 E2 以来始终未变的
  softmax/topk reduce kernel 在昇腾的执行;预算 5/5 用尽,本题停止。
- 最好成绩 6/8:天数(split-fp16 1.13x/0.90x)、沐曦、海光、燧原、card_a/b
  通过;华为数值失败、昆仑平台评测器崩溃(待恢复,理论上限 7/8)。
- 跨芯沉淀:昇腾 fp16-dot 数值失败(split-fp16 镜像昆仑);UB 溢出量化
  (2.89M/1.57M bits,reduce tile 32×256 fp32 即超限,BLOCK_R=8 可编);
  路由 reduce kernel 在昇腾存在未定位的数值错误,后续若有他人通过证据
  可重启。


## E5:ascend 归约 kernel 逐行 1D 重构(2026-08-29)

- 复盘:E2–E4 四轮全部只动 GEMM 侧(精度/tiles/stages),softmax/topk
  归约 kernel 自 S0 起逐字节未变。归约 kernel 中唯一无平台先例的操作是
  2D (BLOCK_R, BLOCK_E) tile 上的 `tl.max/tl.min(axis=1)`;对照 T25
  `_ascend` argmax(1D axis=0 max + where+min 首索引)在华为三次平台
  通过,T21 `tl.sum` 亦 1D/2D 已证。假设:华为失败面 = 2D axis=1
  归约在昇腾的执行,与 dot/stages 无关。
- 单变量改动:`_ascend` 归约 kernel 改为逐行循环(`for row in
  tl.range(pid, n_rows, n_programs)`),所有归约 1D axis=0
  (T25 华为已证结构),软封顶 tanh 形式(T24 已证)与索引 where+min
  (路由 generic 燧原已证)保持不变;GEMM kernel 与 launch
  (ieee + stages=1)不动。
- screening(2026-08-29 01:0x,gpu RTX 5070 Ti):
  `gpu:/tmp/flagos-router-asc.6i1BpG`;unittest 8/8;
  `_ascend` vendor 全矩阵(fp32/fp16/bf16 × 6 shape × topk ×
  softcap/bias × 平局/非连续/70k 折叠)通过,仅 1 例 fp16 k8 精确平局
  ids 翻转——与旧 kernel 同种子同 row 复现(fp32 logits slot8==slot9
  完全相等),属 torch.topk 平局顺序设备细节,非本改动回归。
- source commit `77c34c8`(blob SHA `6f5cf2cb…` 与 screening 逐字节一致);
  ZIP `artifacts/competition/fused_moe_router_cudacore/e5-77c34c8/`,
  canonical SHA-256 `a0d5f474a7ea08f4ab650548e77545dc034f182514b30be68c2fa1c7d601b3c7`;
  成员 generic + `_ascend` + `_iluvatar`(3,与 E4 集合一致)。
- release 验证:见文末 release 记录。

## E5 平台结果(sub 6187,2026-08-29 02:2x 终态前)

- 华为仍 topk ids 失配 → 1D/2D 归约结构假设证伪;其余六芯通过
  (天数 0.9234、沐曦 1.0924、燧原 0.1924、海光 0.9392、
  card_a 0.5008、card_b 1.0216);昆仑 waiting_callback。
- 失败面再收窄:归约 axes 不是根因;weights 全过仅 ids 失配,
  指向边界 logit 对的**排序差异**(精确平局取序或 split-K 舍入翻转)。

## E6:平局取后索引对照实验(sub 待填,2026-08-29 02:4x)

- 单变量:`_ascend` 归约 tie-break 首索引 → 末索引
  (`tl.max(where(cand==cand_value, experts, -1))`),与 T27 e6 的
  n_splits=1 变体构成同指纹 A/B 对照。
- 代理:unittest 8/8;仅 fp16 精确平局例翻转(设计内,5.0-0.001
  在 fp16 即 5.0);非平局全对。
- source commit `5c510f5`;ZIP `e6-5c510f5`,SHA-256
  `7ede5e938356952b286831cb8b95024b39a95879d0af23b2dc5118573c612473`。

### E6 平台结果(sub 6201,华为芯已完成)

- **华为仍 ids 失配** → 平局取序假设排除。六芯通过,昆仑 waiting。

### E7 平台结果(sub 6212)

- 华为仍 ids 失配(case_idx=7,E=256/topk=8)→ n_splits=1 在 cudacore
  上不充分;T27 同修复却通过 → T26 存在 T27 没有的额外失败面。
  GEMM kernel 两题逐字节相同 → 剩余差异:topk 8 轮选 vs ≤2、
  非 64 倍 H 的 k-mask 路径(仅 T27 合同排除)、或 case7 恰好踩
  dot 累加与参考 matmul 的舍入差。
- 昆仑 waiting_callback。

### E8:dot-free FMA GEMM 探针(sub 6228,2026-08-29 03:4x)

- 单变量:`_ascend` GEMM 内 `tl.dot(ieee)` → 逐 k 顺序 FMA
  (`tl.range(k_begin,k_end)`,(B,1)×(1,E) 广播乘加),一次覆盖
  "dot lowering 数值 bug" 与 "累加顺序 vs 参考 matmul 舍入" 两个
  剩余假设。
- 代理:全矩阵对(唯一失败为已知 fp16 精确平局信息性);
  代理 perf 0.055–0.58x,华为 0.1x 阈值有风险,correctness first。
- source commit `04b97cb`;ZIP `e8-04b97cb`,SHA-256
  `6bbbd6376b5ce5e56b25b7e8e4557b9fb99762036fed3cc69a83f5a3aef55877`。
