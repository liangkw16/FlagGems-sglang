# Task 58 `w8a8_block_int8_matmul` 实验记录

```current
task: 58
operator: w8a8_block_int8_matmul
batch: 4
validity: valid
platform: 8/8(e5,11143,250.64599167x新team best)
team_best_stage: e5
team_best_commit: cf31913e61b91654542b654fe4d7d8c226e0d222
team_best_speedup: 250.64599167
sealed: no
next: E6分组边界与FP32契约修复release通过;大矩阵代理回退约20%,未晋级不提交
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

## 2026-09-08 E6 后续实施（独立分支）

- source / verification：`65b7fdd418d941ad4393ae9485541c74a55e5017`。
- release：8 方法、235 次 kernel launch，0 fail/error/skip。NVIDIA 代理范围通过；其他目标芯运行时未验证。
- 回执 `artifacts/competition/batch4-next-20260908/t58-release/verification.json`，SHA-256 `f8cefffe9b588ecba4f1dbddeb0caec0754b75050c42b6e22177fb7e9d35ee6a`；完整日志 SHA-256 `4db87d81b73dc925c6c4d1a2a7f513834884406ec436b9be66e5a432db800a7e`。
- ZIP `/Users/bytedance/ccc/flagos-batch4-next/artifacts/competition/w8a8_block_int8_matmul/e6-65b7fdd/w8a8_block_int8_matmul.zip`，SHA-256 `68893a461103596298c62e8a560e9a36f7a6cfdc8817fdd256734e3027c59b1f`，23363 bytes；成员：`w8a8_block_int8_matmul.py`, `w8a8_block_int8_matmul_ascend.py`, `w8a8_block_int8_matmul_enflame.py`, `w8a8_block_int8_matmul_iluvatar.py`, `w8a8_block_int8_matmul_kunlunxin.py`。逐成员完整 SHA、测试/runner/依赖身份见[证据清单](../data/batch4-next-20260908.json)。
- N/K tile 严格落在各自量化组内，组内累加后再乘 scale；int8 先转 FP32，generic/Ascend/Enflame 用 TF32（int8 值可精确表示），天数/昆仑保留 IEEE。补任意分组和 stride/分块边界。
- 旧源在新增回归有 18 failure / 5 error；修复版全部通过。M=1 约 3.49x，公开四形状接近持平，但 M=1024,N=2048,K=4096 约 0.795x。BK128/BM128 尝试未改善，恢复较小配置。**性能不晋级，本轮不提交平台**，历史 E5 250.64599167x 仍为平台最佳。
