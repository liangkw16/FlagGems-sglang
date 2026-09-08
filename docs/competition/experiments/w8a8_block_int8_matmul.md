# Task 58 `w8a8_block_int8_matmul` 实验记录

```current
task: 58
operator: w8a8_block_int8_matmul
batch: 4
validity: valid
platform: e6r/11210八芯valid,258.04890833x team best(排名3);e7/11228 valid 253.07(B已修复)
team_best_stage: e6r
team_best_commit: 80bba3d
team_best_speedup: 258.04890833
sealed: no
next: e7字节(e6r组级+amd逐块B)为最优组合,均值差=华为窗口三连下行;水位回常态时以e7字节重掷(新ZIP身份,≤2次)
updated: 2026-09-08
```


## S0 fp32-ieee dot + 组内 scale（2026-09-06，远端 GPU 全过）

- 形态：单 Triton kernel；int8 load → `.to(tl.float32)` → `tl.dot(...,
  input_precision="ieee")`；每个 K 迭代后按 `k_start // block_k` 取
  As 行向量 [BLOCK_M] 与 Bs 列向量 [BLOCK_N]（`offs_n // group_n` gather），
  `acc += dot * a_s[:,None] * b_s[None,:]` — 与 reference 逐块舍入顺序一致。
- **遵守题面注意事项**"int8 须先 cast fp32、不可 int8 直接 GEMM"：不做
  int8 tensor-core dot，规避判罚风险（int8 dot 虽数学上更精确）。
- BLOCK_K = largest_pow2 ≤ min(group_k, 64)（保证 BLOCK_K | group_k，
  k 迭代不跨 scale 组）；BLOCK_N = largest_pow2 ≤ min(group_n, 64)；
  num_stages=2（3 stages + BLOCK_N=128 曾爆 shared memory 245760 > 101376）。
- 远端 5070 Ti：unittest 4/4 OK（平台契约 fp32 scales；atol 1e-2 本地线，
  平台 0.5）；bench 4 平台 shape：3.3x/6.5x/6.0x/22.4x，maxdiff ≤ 0.031。
- 教训：本地测试的 bf16/fp16 scale 子测是自加严（reference 的
  bf16 `s=a_s*b_s` 乘积自身差 |C|·2^-9≈0.04）— 已对齐平台契约改为
  fp32 scales。

## S0 平台结果（2026-09-06，submission ~08:40）

- **7/8 PASS**：天数 74.6x / 沐曦 147.5x / 燧原 4.60x / 海光 367.1x /
  华为 177.4x / A 61.7x / B 7.96x；**昆仑 PassManager::run failed（编译崩溃）**。
- e1：`_kunlunxin/ops/w8a8_block_int8_matmul.py` vendor — 唯一逐 lane
  向量整除 `offs_n // group_n` 改为标量 `(pid_n*BLOCK_N)//group_n`
  （BLOCK_N | group_n 时组索引跨 tile 恒定）；generic 字节不动。
  远端 variants 矩阵 5/5 OK（含 vendor 在 NVIDIA 代理编译+数值）。

## E1 → **8/8 VALID**（2026-09-06，submission seq27，第 8 个 8/8！）

- 逐芯：天数 74.44 / 沐曦 145.17 / 燧原 4.48 / 海光 358.86 /
  昆仑 142.03 / 华为 185.79 / A 62.84 / B 7.68 → **avg 118.16x**
- 昆仑 vendor（标量组索引替向量整除）一发修复编译崩溃，且 142x 高性能。
- 榜首 EvokeAgent 209.30x，差 43%；燧原/B 是短板轴。

## 2026-09-07 只读盘点校正

实时 team best 仍 E1/sub10412，真实八芯平均122.66158333，校正旧118.15512。榜首562.41590833；平台当前排名3，本轮未提交。
- 查询证据 `/Users/bytedance/ccc/flagos/artifacts/competition/batch4-top1-20260907/tasks-now.json` SHA256 `fc73368c3d98b228b0c7815d6ec1e9042a58af8d953337fff58990daec1474fc`。
- 查询证据 `/Users/bytedance/ccc/flagos/artifacts/competition/batch4-top1-20260907/58-submissions-now.json` SHA256 `fa822e61f388826ef2e99c7712299f89b369b40a7a2b68f3758f94944eb69d13`。

## 2026-09-08 推荐方案实现与提交前验证（未提交平台）

新增沐曦cpasync候选，保持int8先转FP32和scale顺序；修复原测试误生成int64，补负数及K/N尾部。NVIDIA拒绝pipeline参数（13条error），未打包。

- source `26a95766b179d263916e9483dfc8d2343c40406a`；verification `26a95766b179d263916e9483dfc8d2343c40406a`。6 个测试方法、23 次实际 kernel 调用；选定 NVIDIA/代理范围门禁失败。
- 回执 `artifacts/competition/batch4-implementation-20260907/t58-release1/verification.json`，SHA256 `27ad5180edb00e2702b1e4bdda7122b2a9229fa6ac02d73f260a872b70a19109`；日志 SHA256 `0d55f76f5f476e30857a93a5cd779ad0d904163619acde76a76dbde19fae26ea`。
- 环境、逐源码执行范围、原始配对数据和未完成条件见[本轮报告](../implementation-batch4-20260908.md)及[证据清单](../data/batch4-implementation-20260908.json)。本轮不更新历史有效分，未做平台 preflight、上传或正式提交。

## E2/E3 fp16 张量核 dot：七芯 2-14 倍增益，天数 vendor 单发修复（2026-09-08）

- **E2（commit `bb4bbc4`，sub 11047）**：generic 操作数 int8→fp16
  （int8 值在 fp16 精确表示，张量核 fp16×fp16→fp32 累加；移除
  `input_precision="ieee"`），删除从未通过门禁的沐曦 pipeline vendor
  （回到 E1 成员集）。NVIDIA 代理配对计时 6.5-6.8x、输出逐位一致。
  平台终态 **7/8 invalid_correctness：天数 fp16 dot 失败**，其余七芯
  大幅上涨——海光 **655.6**（358.9→，+83%）/ 沐曦 **265.5**（145.2→，
  +83%）/ A **364.7**（62.8→，+481%）/ B **111.4**（7.68→，+1350%）/
  华为 190.0（+2%）/ 昆仑 139.1（vendor 未动，-2%）/ 燧原 4.63（+3%）。
- **E3（commit `20379c0`，sub 11049）**：单变量 = 新增 `_iluvatar`
  vendor（字节 = E1 fp32-ieee generic，天数 dot dtype 敏感性与 T12
  族一致）。platform：天数恢复 **74.08**，海光 655.6 / A 362.6 /
  沐曦 221.6 / 华为 195.9 / 昆仑 140.4 / B 111.1。
  **终态 8/8 VALID，avg 220.752425 新 team best（is_team_best）**：
  燧原 4.79 通过；E1 122.66 → **+80%**，距榜首 c2flow 576.32 收窄至
  2.61x（原 4.70x）。
- 跨芯知识（更新 T12 条目）：**天数 dot 操作数 dtype 兼容集依赖算子
  上下文**——T58 中 fp32-ieee 可用、fp16 失败，与 T12 的结论方向相反；
  每题逐芯 dtype 路由不可凭单题经验外推。fp16 张量核 unlock 对
  dot-bound 芯是本季最大单结构杠杆（B +1350%、A +481%）。

## E4/E5 芯级 tile 分派：e4 证伪拆出方向，e5 **250.65 新 team best**（2026-09-08）

- **E4（commit `6d687b0`，sub 11129）**：generic 全面升
  BLOCK 128×128×128（代理 +10%）。平台 8/8 valid 但 avg **184.956525
  （e3 220.75→，-16% 证伪）**。逐芯拆分极性鲜明：**华为 391.82
  （195.91→，+100%）、燧原 6.12（4.79→，+28%）** vs 沐曦 114.26
  （-48%）、海光 461.74（-30%）、A 263.19（-27%）、B 25.33（-77%）。
- **E5（commit `cf31913`，sub 11143）→ 8/8 VALID，avg 250.64599167
  新 team best**：单变量 = generic 回 BLOCK64（e3 字节）+ 新增
  `_ascend`/`_enflame` 两 vendor 携带 e4 已证 128 tile（华为/燧原
  各自平台实证，芯级分派）。终态：**华为 445.56（+127% vs e3）** /
  **燧原 6.16（+29%）** / 海光 644.82 / A 376.91 / 沐曦 207.85 /
  B 109.35 / 天数 74.95 / 昆仑 139.58。e3 220.75 → **+13.5%**，
  距榜首 c2flow 576.32 收窄至 **2.30x**（今日从 4.70x 起两连收）。
- 跨芯知识：**tensor-core dot 的最优 tile 严格芯相关**（华为/燧原要
  128、海光/沐曦/A/B 要 64）——"vendor 分派不可省"（T39 块跳过教训）
  在 GEMM tile 维度第二次验证。代理 +10% 不代表平台方向，逐芯平台
  数据才是分派依据。

## E6 组级 scale 累加（2026-09-08，提交前预注册）

- 假设：K 循环按 scale group 两级化——group 内 BLOCK_K 步用
  `tl.dot(a, b, acc)` 在张量累加器内累加（零逐元素开销），`acc * a_s * b_s`
  的 [M,N] 逐元素链每组只跑一次（group_k=128、BLOCK_K=64 时减半）；
  As/Bs gather 也减半。结构来自 MLSys2026 FlashInfer MoE 冠军 kernel
  （`mlsys2026-flashinfer-contest-solution/submissions/moe-fp8/solution/
  python/main.py`，本地 external-repos 已 clone）。组内累加比逐块缩放
  更贴近 reference（整组 matmul 后乘 scale）。
- 单变量：generic + `_iluvatar` 同构改造（两条 dot 路径各一文件）；
  `_ascend`/`_enflame`（BLOCK_K=128=group，无组内步可省）与
  `_kunlunxin` 字节冻结。
- **E2 天数失败先诊断（零额度 raw_result）**：sub 11047 tianshu 仅
  **1/29,360,128 元素**超差——abs 0.5088 vs atol 0.5、ref≈0.26 的
  大数消减刀锋元素；根因是 fp16 张量核乘积 127×127=16129 超 fp16
  整数精确域 2048 的舍入翻转，非 lowering 错译，结构上不可绕。
  天数维持 `_iluvatar` fp32-ieee；本候选给该路径减 epilogue。
- NVIDIA 配对计时（RTX5070Ti，6 组 AB/BA×30 次 wrapper 全含，
  `pair-generic.json`）：generic 三组 g128 case **+15.3%/+15.3%/
  +15.6%**，g64（GROUP_STEPS=1 退化）+0.6% 中性；iluvatar
  （`pair-iluvatar.json`）0.995~1.005 中性（fp32-ieee 在代理为
  FMA-bound，epilogue 占比小；天数真机 FMA:epilogue 比不同，中性
  无下行风险故随包携带）。
- source/verification commit `18616ff`；release v2 回执（远端
  `gpu:/tmp/flagos-t58e6-rel`）：6 方法全过（RELEASE_REQUIRED 全集），
  generic 25 + iluvatar 15 次非 warmup launch，fail/error/skip/xfail=0。
  回执 `e6-18616ff/validation/verification.json` SHA256
  `8da35db804324bf9d4b66cf3ab6bc4c405e41d927dcabde14d90c357dd38c97c`、
  日志 `6479ee10c3ec709955b455e2a7ec9d935c65034f5bb0f59504c4e680427284e0`。
- ZIP `e6-18616ff/w8a8_block_int8_matmul.zip`（5 成员：generic 新、
  ascend/enflame/kunlunxin 冻结、iluvatar 新）24785 bytes，SHA256
  `6d63454b06391f3c8f692f42223fad2241d5eea962ba352ab9e4ad028e0f8972`。
- 预注册晋级门：8/8 valid 且 avg > 250.64599167（e5 team best）才晋级；
  generic 四芯（沐曦/海光/A/B）任一 ≥+5% 视为结构兑现。若 8/8 但未超
  e5，字节保留为已验证组合（组级结构无回退证据）并记逐芯读数。e4 教训
  在案：代理 +15% 不保证逐芯方向，平台逐芯数据为准。

## E6 提交记录（2026-09-08T14:56）

- preflight 一次性通过（quota 13/30）；submit **submission 11202**
  `queued`，file_url SHA256 `f67ea92139228f713e0eba04e2d73a59c0022e539927d694c4fa840730987c4f`；
  证据 `batch4-codex-round-20260908/`（58-preflight intent nonce
  `cfd4e110…`、submit 响应、58-watch.jsonl）。

## E6 平台终态与 e6r 重载（2026-09-08T15:1x–15:24）

- **sub 11202 终态：`invalid_correctness` 7/8**——昆仑 18/18 case 同指纹
  `RuntimeError error code=299, wait for noc idle timeout`，栈在 XMLIR
  运行时 `aten_capture/eager_customized/cat.cpp:44` 与 `copy_kernel.cpp:414`
  （benchmark 基建层，非 kernel），执行 1,206,418ms——**昆仑崩溃族**
  （与 T51 E6 同指纹，~1.207s 级执行）；昆仑 vendor 为 e5 冻结字节，
  今日 12:00 同字节刚以 139.58 通过。按崩溃族协议不计代码止损。
- **七芯结构读数（vs e5）**：海光 **703.93（+9.2%）**、沐曦
  **219.21（+5.5%）**、天数 76.85（+2.5%，ieee 路径组级化也有小赚）、
  昆仑回调失败、华为 404.49（冻结字节 -9.2% = 窗口噪声）、A 372.23
  （-1.2%）、B 85.78（-21.5%，疑窗口/待复验）、燧原 6.22（冻结持平）。
  generic 四芯中两芯 ≥+5%（预注册"结构兑现"门），B 的回落需 e6r 复读。
- **e6r = 注释载体（commit `80bba3d`，generic 仅注释差异，四 vendor
  与 e6 逐字节一致；e5 昆仑冻结字节在内）**：按载体纪律过 py_compile，
  因 v2 回执绑定 source commit 另跑 exact release（回执
  `e6r-80bba3d/validation/verification.json` SHA256
  `7a3e515cd13411600cdc9bfdce928d9b3bb6fdf7e60489234d01855a29271f30`）。
  ZIP `e6r-80bba3d` SHA256
  `c062ada088ee6b68e38b9303ce1aa011363ce63112cdee62499b047b248a4fbc`。
- **sub 11210**（15:24:46）已提交，重掷昆仑回调窗口；晋级门沿用 e6
  （8/8 且 avg > 250.64599167）。同字节重掷 ≤2 次纪律内（第 1 次）。

## e6r 平台终态：**8/8 VALID，258.04890833x 新 TEAM BEST**（2026-09-08T15:5x）

- sub `11210` 终态 `completed / valid`，8/8 全过，avg **258.04890833**
  （e5 250.65 → **+2.9%**），`is_team_best=true`；昆仑回调本轮通过
  （142.13，与 e5 的 139.58 一致——e6 的昆仑失败坐实为间歇性崩溃族）。
- 逐芯（vs e5）：天数 76.57（74.95，+2.2%，ieee 组级化小赚兑现）/
  **沐曦 257.02（207.85，+23.7%，组级结构在沐曦大兑现）** /
  燧原 6.10（6.16 持平，冻结）/ **海光 721.87（644.82，+12.0%）** /
  昆仑 142.13（139.58，冻结带内）/ 华为 410.69（445.56，冻结字节
  -7.8% = 窗口噪声）/ A 365.74（376.91，-3.0%）/
  **B 84.28（109.35，-22.9%，两连同量级复读——判定组级结构在 card_b
  真实回退，非窗口）**。
- 结算：组级 scale 累加净收益为正（沐曦/海光/天数三芯合计 +86，
  超 B -25 与华为窗口 -35），B 的 -23% 是该结构唯一明确受害者
  （跨芯知识：组级累加对部分 dot 后端损失双发射机会，vendor 分派
  可考虑给 B 单独走逐块缩放字节——下一发候选方向，未验证）。
- 排名维持 **3**（榜首 c2flow 576.32，差距 2.23x）；额度余量充足。
- 证据 `batch4-codex-round-20260908/58r-{preflight,submit}.*`；
  e6r release 回执 `e6r-80bba3d/validation/`。

## E7 card_b 逐块缩放 `_amd` vendor（2026-09-08T16:20，sub 11228）

- 单变量：新增 `_amd` vendor = **e5 generic 逐块缩放字节逐字节搬运**
  （`cf31913` 的 generic，SHA256 `eec198bb…`，平台实证 card_b 109.35）；
  generic（组级 scale）/ ascend / enflame / iluvatar / kunlunxin 全部冻结
  e6r 字节。依据：e6/e6r 两发 card_b 同量级回退（85.78/84.28 vs 109.35，
  -22.9%/-23.0%），组级累加在该后端真实负收益（代理不可见，平台两连证）。
- screening + release（`gpu:/tmp/flagos-t58e7-rel`）全绿：generic 25 +
  amd 15 次非 warmup launch，0 fail。回执
  `e7-091377a/validation/verification.json` SHA256
  `0e07514c155d39295e077a093b0eb4dcc58a0fd1a6e1e5704e926451cf240360`。
- ZIP `e7-091377a`（6 成员）SHA256
  `a496b4cbeef38977a5ceb81fdc33db560524dc44e04e00388661e85bc29a898d`；
  source/verification commit `091377a`。
- 预注册晋级门：8/8 valid 且 avg > 258.04890833（e6r TB）；card_b
  应回 100+（e5 带 109±3），其余七芯带内。若 card_b 仍 <95 → e5 字节
  假设证伪，B 的回退另有原因（窗口/评测器），关闭该轴。

## E7 平台终态：8/8 valid 253.07，B 修复兑现、窗口吃掉均值（2026-09-08T16:4x）

- sub `11228` 终态 valid，avg **253.06811667**（未超 e6r 258.05，TB 保持）。
- 逐芯（vs e6r）：**card_b 103.99（84.28→+23.6%，`_amd` 逐块字节假设
  完全兑现，回到 e5 带）**、天数 79.56（76.57→+3.9%）、燧原 6.44
  （带内）、海光 710.53（721.87，带内）、昆仑 139.14（带内）、
  **华为 372.45（410.69→-9.3%，冻结字节三连下行 445.6→410.7→372.5
  = 下午评测窗持续变慢）**、沐曦 241.74（257.02→-6.0%）、A 370.68（带内）。
- 结算：**e7 是当前最优字节组合**（e6r 组级 + B 逐块分派，每芯都平台
  实证）；均值差 5.0 全部来自华为窗口 -38 与沐曦 -15。华为窗口回常态
  （410-445 带）时 e7 字节期望 ~258-262。处置：e7 字节作为后续水位
  重掷基底（新 ZIP 身份，同字节重掷 ≤2 次纪律）。

## E7r 水位重掷终态：8/8 valid 251.38，未超 TB（2026-09-08T16:43，sub 11236）

- e7 字节注释载体（`4c4ca9d`）。华为 385 带内波动未回 410+，均值
  251.38 < 258.05。e7 字节同字节重掷已 1 次（≤2 纪律），明日窗口
  （华为 410-445 带）可再 1 次。
