# Task 104 `rmsnorm_hf` 实验记录

```current
task: 104
operator: rmsnorm_hf
batch: 7
validity: valid
platform: s0(21439)valid 7/7 avg4.25:沐曦4.26(#1!)/天数6.4/海光6.2;昆仑1.36/华为1.57是拖累(榜首Nectar 4.58)
candidate_stage: e1
team_best_stage: s0
team_best_speedup: 见platform行
sealed: no
next: E1 generic精确子块两遍(79ade34d)代理门全绿(7/7回执+配对中性),待平台arm;预注册各芯签名/回滚预留见E1节
updated: 2026-09-26
```

## 不可变身份（s0，2026-09-25 第二批释放夜）

- 四题同批提交（21439/21440/21442/21443），review 5 项 P2 全修后门禁全绿。
- release 回执：`artifacts/competition/b7b-s0-release-20260925/rmsnorm_hf/verification.json`。
- ZIP：`artifacts/competition/rmsnorm_hf/s0-*/rmsnorm_hf.zip`（T102 绑定 2cad7b0d，其余 f5b7add4）。

## E1 generic 精确子块两遍（2026-09-26，评审 r1 实现 + r2 补证与预注册）

- **假设**：整行单遍 masked kernel 是六芯共同拖累——`BLOCK=next_pow2(hidden)` 使
  fp32 整行驻寄存器（hidden≥4096 时 32KB/program，超沐曦 `max_tile_size=2048`），
  且全路径 int mask + masked-load `other` 预填（华为 Vector CMP int32 降标量 +
  预填串行 MTE2，chip-rulesets.md:36-39；relu2 e9 去 other 预填华为 +26%）。
  改 per-row program + `D_TILE=min(hidden&-hidden,1024)` 精确子块两遍
  （pass1 累加 sumsq / pass2 重读 scale+store），热路径零 mask 零预填；x 二读
  由行驻 L2 吸收。同族实证：T51 fla_layernorm_gated E10 同结构（D>1024 且
  256|D）华为 +10%、8/8 新 TB；昆仑 `fused_rmsnorm.py:68` "XPU OffsetAnalysis
  needs exact N for block DMA; do not pad"（现 next_pow2 对 3072/5120 浪费
  33-60% lane）；海光要窄 1024（relu2.md:119，宽度方向互补）。
- **s0 next 轴的替换依据**：原 next（华为 persistent vendor，T93/T97 同族
  +31~115%）已被排除——T93 add_constant e2 grid48 vs 512 华为 0.247/0.282
  噪声内且 CURRENT 明示 persistent 参数空间证伪；T97 fused_sigmoid_mul e2
  中性（1.311≈1.340）；账本引用的散布归约族收益不可平移。
- **实现**（source commit `79ade34d4e0802c2e79deeb75e1a49f08633c3c7`，仅
  generic `src/flaggems_sglang/ops/rmsnorm_hf.py` + `tests/test_rmsnorm_hf.py`）：
  - 分派门 `hidden > 1024 and hidden % 256 == 0` → 子块两遍 kernel
    （`D_TILE` 256/512/1024，`N_SUB=hidden//D_TILE` 精确整除，num_warps=4/
    num_stages=1 沿 T51 E10 结构）；否则整行 kernel 及其 launch **字节保持
    S0 原样**（非整除如平台 333 探针、hidden≤1024 decode 形状不受影响）。
  - 语义顺序逐位保持：fp32 sumsq → `tl.rsqrt(ms+eps)` → y round 到 out
    dtype → fp32 权重乘 → 单次 store round。归约为逐子块累加（数值重结合，
    容差内，T51 E10 同款）。
  - 回归：`test_exact_subblock_large_hidden`（1280..8192，D_TILE/N_SUB 全档）、
    `test_subblock_gate_boundary`（1024/1025/1200/2176/333）、
    `test_strided_input_large_hidden_subblock`（hidden 2048 跨步 x_s1=w_s0=2），
    全部列入 RELEASE_REQUIRED_TESTS（7/7）。
- **对 T51 E10 先例的三点已申报偏离**（评审 r2 P2-2）：(a) generic 全芯部署
  而非 vendor 限定——本题拖累是六芯共性且沐曦/昆仑有各自机制证据，T51 账本
  "行块配方芯相关且算子相关，不可跨题外推"以预注册签名+回滚预留对冲（见下）；
  (b) `tl.range` 替代 vendor 的 `range`——依据 chip-rulesets.md:37（static 展开
  按迭代累计 UB 占用），仓库 `_kunlunxin/deepep_post_reorder.py:44` 等有先例，
  Ascend 真机未验证，计入华为观测签名；(c) 非 persistent per-row grid
  （persistent 参数空间已被 T93/T97 证伪，不引入）。
- **代理证据（评审 r2 P2-1 补齐，NVIDIA RTX 5070 Ti / torch 2.13.0+cu130 /
  triton 3.7.1）**：
  - release v2 回执（source=verification commit `79ade34d`，模式 release）：
    7/7 方法全过（含 3 个新回归，日志 `Ran 7 tests ... OK`），41 次 kernel
    launch 与逐 subTest 推算（10+1+1+2+16+10+1）精确吻合，子块形状
    [1280..8192] 均有真实非空张量与 JIT launch。
    `artifacts/competition/rmsnorm_hf/e1-79ade34/validation/verification.json`
    (sha256 `52e1a396ae20d2d590f6fc97e441dc5bd7ffb09775f85074946ecd18d957e39d`)
    / `verification.log` (sha256 `afcda51c6b7d621b03658d20cfecf2de32ad1d78a5aa0a9ce99fe95a594d6df0`)。
  - 配对计时（wrapper-inclusive，6 轮 AB/BA 交替 ×50 iters，S0=f5b7add4 字节
    vs e1=79ade34d 字节，14 形状 × bf16/fp16）：
    对照组（hidden≤1024 整行路径，字节未变）paired median **0.9996-1.0021**
    完全中性；子块路径（hidden>1024）paired median **1.029-1.043**——代理上
    无回退、方向中性偏正（该 GPU 上 decode 小形状 enqueue-bound ~8µs，读数
    是"结构不伤代理"的签名而非重排收益；UB/CMP 收益按 T51 E10 先例只在
    华为类芯片兑现）。`pair-e1.json`
    (sha256 `e9bc6f5e94b47204f8b580890f866a9206fe30161c61cb5747641181b364f88a`)。
    远端目录 `/tmp/flagos-t104e1-rel`、`/tmp/flagos-t104e1-pair`，总日志
    `/tmp/flagos-t104e1-run.log`（两阶段 RC=0）。
- **预注册各芯预期签名**（s0 基线 sub 21439 → e1 预期；climb-loop s2t1op104）：
  | 芯 | s0 | 机制预期 | 兑现/回退判读 |
  |---|---|---|---|
  | 华为 | 1.5687 | ↑ 主靶（CMP 降标量+去预填+UB 驻留三机制；T51 锚 +10%） | ≥1.70 判轴兑现 |
  | 昆仑 | 1.3585 | 中性~↑（exact-N DMA 去 padding；E23 idle-core 前科计入正确性观察） | 正确性优先观察 |
  | 沐曦 | 4.2601(#1) | 中性~↑（4096/8192 超 max_tile_size=2048 → 回到带内） | 重点保护，<-5% 触发回滚 |
  | 海光 | 6.1592 | 中性~↑（要窄 1024 方向一致） | 噪声带内即过 |
  | 天数 | 6.4061 | 中性（无芯专属证据） | 噪声带内即过 |
  | A/B | 4.9772/5.0375 | 中性（代理配对 1.00-1.04 同源） | 噪声带内即过 |
  - **晋级门（预注册）**：7 芯无 -5% 回退；华为 ≥1.70 才判轴兑现；平台均值
    >4.25246667 换 TB。任一芯 -5% 回退触发下述回滚而非全量回滚。
  - **回滚旋钮（预留）**：`runtime/backend/_metax`/`_tianshu` 等 vendor 切分——
    把 S0 整行字节冻结进该芯 vendor（平台 vendor 优先于 generic 生效），即可
    单芯回滚不动其余六芯；vendor 文件经 `tests/_op_variants.py` 自动进矩阵。
    华为若 tl.range 出问题，同机制回 `_ascend` vendor（range 版 T51 字节）。
  - 本节之前无该轴任何平台读数；以上门与签名在平台提交前落账，读数后不得
    改判据。
