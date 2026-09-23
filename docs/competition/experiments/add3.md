# Task 76 `add3` 实验记录

```current
task: 76
operator: add3
batch: 6
validity: candidate(e4代理release 3/3,待平台);有效TB e2 1.0564x
platform: e2(17393)valid 8/8 avg1.0564 TB; e3(19376)invalid_threshold昆仑0.056(同generic字节e2=0.64),燧原1.06
candidate_stage: e4
team_best_stage: e2
team_best_speedup: 1.0564x
sealed: no
next: e4燧原大输入tile 8192→16384；代理1.80-2.62x，release和ZIP验签通过，待codex-review；平台门8/8各芯≥0.1、均分>1.0564、燧原≥1.5；昆仑同字节0.056风险；额度4/30
updated: 2026-09-23
```

## 2026-09-23 E4 候选：燧原满波后扩大流式 tile

- 最新 17:50 逐芯榜：我方有效 e2 **1.0564** vs varphi **1.98311667**，
  八芯和差 **7.41373336**。燧原 0.91393333 vs 7.96046667 占缺口
  7.04653334；若单芯完全追平仍需其他芯共加 0.3672。e3（19376）
  同一 generic 字节昆仑 0.056 <0.1，故其燧原 1.06 不能作为有效总分。
- [FlagGems GCU300 配置](https://github.com/flagos-ai/FlagGems/blob/d03a66b71cbd586e9408b81ff8f2c4f8329b77a1/src/flag_gems/runtime/backend/_enflame/gcu300/utils/codegen_config_utils.py)
  对 bf16 pointwise 给到 32K×4 元素上限；
  [tile 选择](https://github.com/flagos-ai/FlagGems/blob/f148752746cee390bdbe53b3eaac44bbebb4220b/src/flag_gems/runtime/backend/_enflame/gcu300/utils/shape_utils.py)
  在小于 512K 的任务还会按半上限取值。它是上游生成策略，非当前手写 kernel
  的性能保证。e4 仅改 `_enflame` 大输入分块：`numel>=12*16384`
  时 BLOCK=16384，否则保持 e3 的 8192；grid cap12、2 warps、3 stages、
  双舍入计算和其余两个 ZIP 成员不变。门槛保证扩大分块前有完整一波 CTA。
- 第一档 65536 元素试验已淘汰：NVIDIA 代理 source SHA
  `94394907fbd37f9e08f236460352e2db89265866eb19f67eea8cc0e32c00225d`
  逐位正确、3/3 测试过，但编译 `n_spills=2420`（基线 0），
  四档配对 `base/new=0.128/0.089/0.109/0.181`，慢约 5–11 倍；
  原始计时 `e4-screen-20260923/bench.out` SHA-256
  `69e5d543609b215e10acd9bf8625a0541e0d45c98ba0cadb666bfa9dac0517b0`。
- 收敛到 16384：代理 `n_spills=10`，真实候选 screening 回执
  `e4b-screen-20260923/verification.json` SHA-256
  `9a8b677f390315ff2c1aae90c9f37285d12865bad4725e1c974e8df53c842c42`，
  generic/Enflame 各 12 次 launch、3/3 测试全过。四档
  786432/1048576/4194304/16777216 元素五轮 AB/BA wrapper-inclusive
  中位 `base/new=1.837/1.796/2.104/2.619`，原始计时
  `e4b-screen-20260923/bench.out` SHA-256
  `8a2b5b929746294b6affc46705c6c8e3840ed71ceb2b7633cb61a01659ba284e`；
  仅能说明 NVIDIA 代理收益，燧原目标 `target-runtime-unverified`。
- source/verification commit `cc81499fe5790845218ba581115f7b9064c0605b`；
  Enflame source SHA-256
  `eb606d7c114d90b725d14b2edad00a5210b5608b9bf3bbe54bc1fbe25515bd32`；
  test SHA-256
  `4aacf7f0fe9a1e7120e49aa55af6e4e595ab32b3809f0c197a4010b669f8eaa1`。
  release 回执 `e4-cc81499/verification.json` SHA-256
  `3c42430f9900ba0f9bc1a99e76124050b993c91741def262d6414f8f4a7a6c7a`，
  日志 SHA-256
  `651a1de65d3a0cf0f8ec6374dd3905ea7b1a4fc3618573eb4034e638eb568e15`，
  3/3 测试、2 实际路径各 12 launch、0 失败/错误/skip/xfail；
  Ascend 原成员未在 NVIDIA 代理执行。
- 不可变 ZIP `artifacts/competition/add3/e4-cc81499/add3.zip`（4810 B）
  SHA-256 `77f029e7ece00ced445d0abff5387c0a0b229ab0d0b900b2cd6a46c9b8577f4f`；
  三成员 `add3.py` `5f89fb24…d296`、`add3_ascend.py`
  `4609850b…b14f`、`add3_enflame.py` `eb606d7c…bd32`，与 source
  commit manifest 逐项一致；dry-run/verify-existing/unzip -t 通过。
- **预注册门**：codex-review 无可靠缺陷再实时 preflight；平台 8/8 正确、
  各芯 ≥0.1、均分 > e2 1.0564 才晋级 TB；燧原 ≥1.5 才说明目标轴
  兑现。若整发 invalid（尤需看昆仑同字节低水位），不自动重投 e4；若燧原
  <1.5 或有效均分不超 e2，回滚燧原成员到 e3 提交前字节。

## 契约与实现（S0）

- 题面：[Task 76](../tasks/batch-6/76-add3.md)。`out = bf16(bf16(a+b)+c)`，
  **双重舍入是契约**（与未融合对逐位一致，禁 fp32 单舍入）；bf16 连续、
  numel%16==0；标准 bf16 容差。
- 实现：flat grid-stride kernel，`(a.f32+b.f32).bf16` 显式中间舍入再
  `+c` 出 bf16（RTNE，与 torch eager 每算子语义对齐）；BLOCK 1024、
  grid≤4096；对照 SGLang 197832b add3（PDL 为 NVIDIA 专属未引入）。
- 测试：双舍入最小反例（a=1,b=2^-8,c=-1 → 双舍入 0 / 单舍入 2^-8）、
  尾块边界、±inf 相消与 NaN 位级对照。

## 不可变身份

- source commit：`370923bfb83a8b7e681051acb2a3b1585dde12fc`（五题同批）。
- verification commit：`686092eb443f6a7b9514f8890fa9fdaa2d339a02`（含测试侧修复轮）。
- source SHA-256：`5f89fb2445a68fad815a89f298a29ce207d4e3e4c11369baeeaf4f23b820d296`。
- test SHA-256：`c4f0a92ceaf002c88bd47564eb825b09d61cd483219e3dec03c0d69d196149a6`。
- ZIP：`artifacts/competition/add3/s0-370923b/add3.zip`，SHA-256
  `6a50c36f2787094b347c45f0ed466f14309b14162971b4405cf56e924312f9c7`（单成员 `add3.py`，generic-only）。
- release 回执：`artifacts/competition/b6-s0-release-20260917/add3/verification.json`
  （3 测试 0 失败，9 次非 warmup kernel launch，
  NVIDIA RTX 5070 Ti / torch 2.13.0+cu130 / triton 3.7.1），
  SHA-256 `e7a372866f53bc1f4a7ae4612c7467f2ead8df6a6aa4790789c3136366f4eb93`；日志 SHA-256 `ed0eb9ef80b57877051e0ccf409d8c29c67867bc04a50e52b7fac2e7585720da`。

## 靶子与下一步

- 榜首 GuanghuLab 1.1611x（9/9 队有效，无彩票指纹）；主轴华为（全员<0.5）与昆仑 vendor。
- 验证三轮教训已固化在测试侧：numel%16 契约、token 边界与 kv_lens 解耦、
  归约噪声容差、行内连续 stride 构造。
- 八芯目标 `target-runtime-unverified`（NVIDIA 代理证据），裁决权在平台。

## 2026-09-18 S0 平台首回执（submission 17220，observed_at 01:0x +08）

- 状态：8/8 valid；均值 0.99170833。
- 逐芯：天数 1.4622 / 沐曦 1.1103 / 燧原 0.4382 / 海光 1.3677 / 昆仑 0.6405 / 华为 0.2956 / A 1.3511 / B 1.2681。
- 首回执与榜首 GuanghuLab 1.1611 对照：天数 1.46 vs 1.66、A 1.35 vs 1.38 接近；差距集中在华为（0.30 vs 0.42）、燧原（0.44 vs 1.20）、昆仑（0.64 vs 0.73）。

## 2026-09-18 E1 平台终态：双 vendor 兑现，TB 1.042875

- E1（17294，`8b65a43e`）：燧原 streaming vendor 0.4382→**0.7553（+72%）**、
  华为 persistent vendor 0.2956→**0.3631（+23%）**、A 1.351→1.392；均值
  **1.042875 新 TB**（vs s0 0.9917，+5.1%）。release 三路径 9+9+9 launch；
  ZIP `e1-8b65a43` 3 成员。
- 榜首 Sweetdeath 1.194（17 队过线）；我方差距集中在燧原/华为/昆仑。

## 2026-09-18 E2 平台终态：燧原 8192 档过门，TB 1.0564

- 结构（`12de5169`，单变量）：enflame vendor BLOCK 4096→8192（阶梯越
  过 T63 峰值档的一档——add3 是纯 streaming，与 T63 的 gather 不同族）。
- submission **17393** completed/valid，8/8，均值 **1.0564 新 TB**
  （vs 1.0429，+1.3%）。**燧原 0.7553→0.9139（+21%，预注册门 ≥0.9
  刚过）**；其余芯窗口持平。
- 族边界再证：BLOCK 阶梯的峰值档按访存形态分化——gather（T63）峰在
  4096、纯 streaming（add3）8192 仍上行。


## 2026-09-21 E3 候选就绪（燧原官方 gcu300 规则集，待 09-22 发射）

- 官方源码调研（FlagGems _enflame gcu300 codegen / FlagTree enflame
  backend）：max_grid_size=(12,1,1)（grid-stride 吸收超额）、
  enflame_heuristics_for_num_warps 钉 2、stride 需编译期互整除才走 DMA。
  本题 vendor 应用：grid 24→12+ num_warps=2。
- 五元组：commit `106517ef`；ZIP
  `artifacts/competition/add3/e3-14437ad/add3.zip`
  SHA `c93ebcc4a87b65647f7abde46e87a5303deaafa7a41dcba8f682ee40d75199d0`；成员 generic/ascend/enflame（generic 与非燧原 vendor 字节
  不变，账本明确列全）；回执 `day5prep-20260921/add3/verification.json`
  SHA `d9ed04149fb27554…`。
- 预注册门：燧原 0.44→≥0.88(×2)；其余七芯不动（vendor-only 单变量）。codex-review
  零发现（grid-stride 边界模拟无漏算）。
