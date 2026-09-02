# Task 27 `fused_moe_router_tensorcore` 实验记录

```current
task: 27
operator: fused_moe_router_tensorcore
batch: 3
validity: valid
platform: 8/8(e9,1.039975x)
team_best_stage: e9 1.016425x
blockers: 榜首1.754125x;差0.7377
sealed: yes
next: 昆仑需约6.5772x才登顶,无可信路径;转T36
updated: 2026-09-01
```

状态:E8 昆仑 direct vendor 平台 8/8 valid,彻底消除旧结构 1830s 超时;
当前第 3,按登顶可能性转 T36。

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

## E3 第三轮 vendor(sub 5805,2026-08-28 04:0x CST)

- E3 结果:华为 UB 溢出已修(可编译)但 split-fp16 在昇腾数值错误(昆仑
  fp16-dot 的镜像);T25 华为失败为 E3 wrapper 顺序 bug(draft=None 时
  finalize 未执行→topk_index 垃圾);T25 燧原三轮不同结构均挂在
  tl.max/tl.min 轴归约(对照 T21 tl.sum、T28 无归约均过)。
- 本轮:router `_ascend` 回归 fp32-ieee(保留 BLOCK_R=8);draft `_ascend`
  finalize 提升到函数级;draft `_enflame` 改全展开 reshape/split 两两树
  (纯 where 元素运算,零 max/min 归约)。
- vendor commit `91838b855b3691e220f5bb03eeac48b1cdf4b6f2`;远端代理 9/9 绿;
  额度 14/30;T27/T26/T25 均剩最后 1 次预算。

## E4 终投(sub 5808,2026-08-28 04:5x CST)

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

- 同 T26 e5 假设与改动:`_router_softmax_top2_kernel` 改逐行循环,
  best/second 两轮均 1D axis=0 归约(T25 华为已证结构);GEMM
  (ieee + stages=1)不动。
- screening 同轮同目录:unittest 8/8;`_ascend` vendor 全矩阵通过
  (近平局/非连续/70k 折叠含 topk≤2 分支);唯一 ids 翻转例同 T26,
  为旧新共有的精确平局背景噪声。
- source commit `2650ee1`(blob SHA `964ff259…`);ZIP
  `artifacts/competition/fused_moe_router_tensorcore/e5-2650ee1/`,
  canonical SHA-256 `3b9302b0a889616791bbd4c764674dcf13647779f9512a7641127d9c5093ff23`;
  成员 generic + `_ascend` + `_iluvatar`(3)。

## E5 平台结果(sub 6190,2026-08-29 02:2x 终态前)

- 同 T26:华为仍 ids 失配,1D 归约假设证伪;六芯通过(天数 1.1926、
  沐曦 1.4372、燧原 0.1972、海光 1.7986、card_a 0.755、card_b 1.287);
  昆仑 waiting_callback。

## E6:顺序 K 累加对照实验(sub 待填,2026-08-29 02:4x)

- 单变量:`_ascend` wrapper `n_splits=1`(去掉 split-K 部分和,
  累加顺序贴近参考 matmul 的顺序舍入)。与 T26 e6 后索引平局变体
  构成 A/B 对照:谁修复华为,根因即谁。
- 代理:unittest 8/8 全对(平局规则未变);64x256x4096 代理 0.129x
  (并行度下降,昇腾 perf 风险已知,correctness first)。
- source commit `3e81a5b`;ZIP `e6-3e81a5b`,SHA-256
  `27f4c615f61c96a59fba688810465e16352f62b6dc9c47ca37d712c6efcf7cff`。

### E6 平台结果(sub 6204,华为芯已完成,昆仑 waiting)

- **华为 0.5404x 通过!** n_splits=1 顺序累加修复 ids 失配——
  **根因锁定:split-K 部分和舍入与参考 matmul 顺序累加的差异翻转
  边界 logit 对**(E1–E6 全程 weights 容差内、仅 ids 精确失配的
  唯一自洽解释;平局取序对照(T26 e6)同轮排除)。

### E7 重投载体(sub 6220,2026-08-29 03:3x)

- e6 内容(n_splits=1)换注释载体重投,探昆仑评测窗口。
- source commit `6f2ce2a`;ZIP SHA-256
  `427937028c84fb6b30ed4b36ba0c122ada3805da3975d39ed3014848988a3fd5`。

## kernelgen MCP 轮负结果(2026-08-29 06:0x)

- 用户指示 kernelgen mcp 优化第三批;目标选 T27 generic(华为/天数走
  vendor,改动零正确性风险;muxi/海光/燧原/card_a/b 吃 generic)。
  工具链:kernelgen-server JSON-RPC optimize_kernel,RTX 5070 Ti 代理,
  12 shape 门控+AB/BA 基准(目录 log/kernelgen-round-t27/)。
- HEAD 基线 geomean **0.8399x**(vs torch 参考,12 shape);
  预注册提交信号 ≥1.08x。
- 三轮 MCP 候选全部否决:iter1 融合 kernel smem 超限(278KB>101KB);
  iter2 修复后 0.3857x(大 B 全面回退至 0.10–0.4x);iter3 按指令
  "HEAD split-K 仅留大 B + 融合只留 n_rows≤32"仍 0.6090x(split 路径
  仍回退)。唯一真实赢面:融合单 kernel 在 launch-bound 小 B
  (7x64x192 2.01x vs 1.45x),不足以支撑整题提交。
- 结论:fp32-ieee dot vs cuBLAS 的 GEMM 差距为结构性的;当前 HEAD
  (e7 平台结构)即代理可见轴的局部最优。与 T30 轮先例一致,本轮
  无候选、不消耗平台提交。报告 log/kernelgen-round-t27/final_report.md。

## E8:昆仑 direct 两阶段 vendor(2026-09-01 10:4x–11:0x CST)

状态:screening、source/test commit、commit-bound release、canonical ZIP
和平台 8/8 均通过。

### 根因假设与单变量

- E7 在昆仑选择 generic:64×64×64 split-K GEMM、至多 8 片、
  `partials[n_splits,B,E]` 工作区、device-side grid-stride 循环,再由
  32×E 二维归约 kernel 汇总。平台记录为 validation execution 约
  1830s、超过 1800s,compile worker 随后 abort。
- E8 不再把该现象归为不可处理的平台故障,而以编译结构复杂度为首要根因:
  只新增 `_kunlunxin` vendor;generic、Ascend、Iluvatar 字节冻结。
- GEMM 改为 32×32×64、`num_stages=1` 的 direct program,K 维按
  compile-time `HIDDEN_DIM` 顺序累加;无 split-K、partials 工作区或
  device-side persistent 循环。host 将物理 grid 分段至每次不超过 65535。
- GEMM 直接写 `[B,E]` fp32 logits;第二个 kernel 每行一个 program 完成
  softcap、bias、全局 softmax 与 top-2。softcap/bias 后重新 mask padding
  experts,修复 `-inf` 经 softcap 变成有限值后可能进入 top-k 的根因。
- top-2 使用 max + `min(where(...))`,保证 expert 不重复;精确平局只锁定
  两个并列 expert 的集合,不依赖 torch 未保证的先后顺序。

### 构建身份

| 项目 | 值 |
| --- | --- |
| source / verification commit | `140a632f80f03e2a093c0e82788f7b67920807ef` |
| generic SHA-256 | `102d07d2d15dab579c04aff1f5f06c00cfb810f9bd3055bb08ac3474d8bbb56f` |
| Ascend SHA-256 | `c49b2448ad1791773e97737e0d23885f6156f9a139179a62b03a0b935997c643` |
| Iluvatar SHA-256 | `0c10544de3a1747b1964220ebf12e5ecff68f52b0c3b4b374bebf2bf0c5842e6` |
| Kunlun SHA-256 | `1809af5cb6bd3832496ab2b876062e84ee825493e659b053bb09bd7f23fd2088` |
| test SHA-256 | `6200ac64c2775bb1cacd25a85076a9fb2fea2fe7ff5593a830d57babfaee9e74` |
| `_op_variants.py` SHA-256 | `cdc5fe3e4cb5a85976f0a3414cd194bb53c79f6f2830be01f685f996b97ca0d7` |
| screening payload SHA-256 | `15f3a8197183edaa610f96bae1992b8aa5fdf216f3264d7064b243ad061f9cc9` |

### Screening 与代理性能

- 目录:`gpu-et:/tmp/flagos-fused-moe-router-t27-screen2.Q78rXm`;环境 RTX
  5070 Ti、Python 3.12.13、PyTorch `2.13.0+cu130`、Triton `3.7.1`、
  CUDA 13.0。
- 完整 unittest **12/12**,耗时 `24.934s`;日志 SHA-256
  `d31dceaa6f6644c7ee4e91968dfc12d990ffe770eaafb0d3e08177aa2a2ab6d7`。
- 覆盖 generic、Ascend、Iluvatar、Kunlun 四份实现;包含三 dtype、H=576/
  E=65 尾块、平台 case 7、topk 1/2、softcap/bias 四组合、单 expert、
  平局集合、全局 softmax、padding remask、非连续输入、输入不变性、空行及
  70000 行 grid fold。
- benchmark 为五轮 AB/BA、warmup 25、repeat 100;harness SHA-256
  `0804f5a1b1ff2b2cb49551cbf36442cfd40aae2884788c85e887e2c19b14e4d7`,
  日志 SHA-256
  `0a2d8a2d25f64d9dc182ae83e24a3594c220423769388ae2f53f5be4ed288bdf`。

| dtype / B×E×H / 配置 | 候选 ms | torch ms | speedup | peak bytes |
| --- | ---: | ---: | ---: | ---: |
| fp16 / 64×256×4096 / cap=0+bias | 0.1380675 | 0.0321901 | 0.233104x | 41487360 |
| fp16 / 1000×128×4096 / cap=30+bias | 0.2565482 | 0.0953802 | 0.371846x | 62214656 |
| fp32 / 64×256×4096 / cap=0+bias | 0.1474794 | 0.0248682 | 0.168604x | 38934016 |

三点均超过题面 `0.1x` 代理门;此结果只证明 NVIDIA 上的编译、正确性和最低
性能余量,不外推昆仑正确性或实际分数。

### Commit-bound release 与不可变 ZIP

- release 目录:`gpu-et:/tmp/flagos-fused_moe_router_tensorcore-release.d1pQoQ`
  (0700,保留);PID/PGID/SID `234420`;从 commit `140a632` Git 对象导出。
- pycompile、Black 79、isort 80、flake8、逐文件前后哈希和完整 unittest
  均通过;12/12 PASS;release 日志 SHA-256
  `46a971fbda4bc69d4104129557a236bf625ebaf69ca21a8f82302c1eb887abd1`。
- canonical ZIP:
  `artifacts/competition/fused_moe_router_tensorcore/e8-140a632/fused_moe_router_tensorcore.zip`,
  28733 bytes,SHA-256
  `279065a663b2d321be66292f0674d0039d3a6d2afb9af4bf048edcd0bd63c3e5`;
  dry-run、实际构建与 `--verify-existing` 三者一致。
- ZIP 成员:generic、`_ascend`、`_iluvatar`、`_kunlunxin` 四文件;各 member
  SHA 与上述 commit blob 完全一致。

### 平台门禁

- 2026-09-01 11:03 CST preflight 精确绑定 race `782kzq4m`、season 2、
  account `15600308080`、team `SoulCoder`、batch 3、Task 27、stage e8、
  source/verification commit `140a632`、四成员、test/release/ZIP 完整 SHA;
  实时额度 26/30、时间窗和最小间隔均通过。
- 一次性 intent 正式提交成功:submission_id `7571`,daily_seq 5;
  提交后额度 25/30。平台文件 URL SHA-256
  `bd3957db45e8fcec23441b6b72f80780c9e593ac28e6f36f900615844916844d`;
  因未配置 `FLAGOS_REMOTE_ZIP_HOST`,上传后远端 ZIP 回读状态为
  `unavailable`,已按一次性规则停止,未重试。

### E8 平台结果(sub 7571,2026-09-01 11:04 CST)

| 芯片 | selected file | speedup | execution ms | 结果 |
| --- | --- | ---: | ---: | --- |
| tianshu | `_iluvatar` | 1.1516x | 25445 | pass |
| muxi | generic | 1.4440x | 26311 | pass |
| enflame | generic | 0.1964x | 8014 | pass |
| haiguang | generic | 2.0464x | 15074 | pass |
| kunlunxin | `_kunlunxin` | **0.6756x** | **12604** | **pass** |
| huawei | `_ascend` | 0.5796x | 37688 | pass |
| card_a | generic | 0.7502x | 15450 | pass |
| card_b | generic | 1.2876x | 13665 | pass |

- 终态:`completed`、8/8、`valid`,平均 **1.016425x**,团队 best,实时第 3;
  榜首 c2flow `1.754125x`,绝对差 `0.7377`。
- 根因结论:旧昆仑 generic 的 split-K + partials + device persistent 组合
  触发编译/执行灾难;direct 两阶段 vendor 将昆仑从 1830s timeout 降至
  12.604s 并正确通过,平台本身并非不可用。
- 若其余七芯保持本轮分数,登顶要求昆仑约 `6.5772x`,即本轮的 9.74 倍;
  当前无足够代理或公开实现证据支持该跃迁,按登顶概率将优化资源转向 T36。

## 2026-09-02 水位采样战役(T27)

- 载体采样(注释载体,核字节=原团队最佳):r1 sub 见 README 战役表;
  新团队最佳 e9 1.039975x(水位采样 r1;距榜首 1.7541 仍 -40.7%,维持封存)。
