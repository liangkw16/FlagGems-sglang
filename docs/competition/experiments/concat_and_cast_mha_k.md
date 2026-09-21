# Task 78 `concat_and_cast_mha_k` 实验记录

```current
task: 78
operator: concat_and_cast_mha_k
batch: 6
validity: valid
platform: e3(19373)invalid_threshold:昆仑0.024环境崩(e2=0.281,未变字节)+燧原0.547(+47%但<2门);待昆仑恢复重掷
candidate_stage: e3
team_best_stage: e2
sealed: no
next: e3/e3r两发昆仑同读0.024(确定性,非窗口)=昆仑评测环境对未变generic字节持续劣化,重掷轴封存;燧原0.55(+47%规则集部分兑现);c2flow 15268x未破译;TB e2 1.0908守
updated: 2026-09-22
```

## 契约与实现（S0）

- 题面：[Task 78](../tasks/batch-6/78-concat_and_cast_mha_k.md)。T62 姐妹题：
  NoPE 拷贝 + RoPE 单头 broadcast，store 侧 `.to(k.dtype)`（k 与输入 dtype
  可不同）；exact。
- 实现：T62 e12 家族几何（BH=16 分组、grid-stride）+ store 前按输出指针
  元素类型显式 cast；解除 T62 的三输入同 bf16 断言（k_nope/k_rope 同
  dtype、k 任意 fp16/bf16/fp32）。
- 测试：dtype 全矩阵（含跨精度方向、RTNE 舍入中点、fp32 溢出/次正规）、
  位级 exact 对照、特殊值同精度透传。

## 不可变身份

- source commit：`370923bfb83a8b7e681051acb2a3b1585dde12fc`（五题同批）。
- verification commit：`370923bfb83a8b7e681051acb2a3b1585dde12fc`（含测试侧修复轮）。
- source SHA-256：`3af5914e673ea1298df554f624ab001e80cd2b0f4d477c91264b0ffa6914a170`。
- test SHA-256：`149d63f1639220b9489385e3d4b2dcfc7ed28a2183fb12490ba9a840402691a4`。
- ZIP：`artifacts/competition/concat_and_cast_mha_k/s0-370923b/concat_and_cast_mha_k.zip`，SHA-256
  `c414ff0c4c20c485750e6f6c5d2e917198161de9ec0324926a755e4e27dd1c10`（单成员 `concat_and_cast_mha_k.py`，generic-only）。
- release 回执：`artifacts/competition/b6-s0-release-20260917/concat_and_cast_mha_k/verification.json`
  （4 测试 0 失败，37 次非 warmup kernel launch，
  NVIDIA RTX 5070 Ti / torch 2.13.0+cu130 / triton 3.7.1），
  SHA-256 `146451ab6d115038fdc2e4b24b5647fb5ecdda4989dc9b00c302324502a0b39b`；日志 SHA-256 `da7c6d16aa896706af39c0858af8249e93abdc7e67bf37c205a5b18cff0c6ce1`。

## 靶子与下一步

- 榜首 c2flow 1.2760x ≈ 我方 T62 TB 1.270875；主战场燧原（0.15→0.96）BLOCK 阶梯 + 华为 persistent；门槛余量优先（T62 华为曾 0.0932 跌破）。
- 验证三轮教训已固化在测试侧：numel%16 契约、token 边界与 kv_lens 解耦、
  归约噪声容差、行内连续 stride 构造。
- 八芯目标 `target-runtime-unverified`（NVIDIA 代理证据），裁决权在平台。

## 2026-09-18 S0 平台首回执（submission 17210，observed_at 01:0x +08）

- 状态：8/8 valid；均值 1.07345。
- 逐芯：天数 2.049 / 沐曦 0.9992 / 燧原 0.3716 / 海光 1.5462 / 昆仑 0.281 / 华为 0.2464 / A 1.535 / B 1.5592。
- 与 T62 e12 逐芯对照：燧原 0.151→0.372(+146%)、昆仑 0.182→0.281 改善；海光 2.911→1.546(-47%)、天数 2.317→2.049 回落——cast 改变 store 宽度或窗口；与榜首 c2flow 差距=燧原 0.59+海光 0.41+天数 0.37。

## 2026-09-18 E1/E2 平台终态：燧原 +8% 后华为 persistent +97%，TB 1.090775

- E1（17292，`8b65a43e`）：燧原 streaming vendor 0.3716→0.4016（+8%，远未及
  c2flow 0.96）；华为 0.2464→0.1664（-32%，快窗回落）；均值 1.0663 < TB。
- E2（17300，`0051b001`）：`_ascend` persistent（T62 e13 配方）华为
  0.1664→**0.3272（+97%）**，超 c2flow 0.2454；均值 **1.090775 新 TB**
  （vs s0 1.07345，+1.6%）。release 双路径（generic+ascend）37+37 launch；
  ZIP `e2-0051b00` 含 3 成员（generic/ascend/enflame）。
- 轴状态：华为 persistent **在 copy 族连续第三次兑现**（T62 e12/e13、
  T78 e2）；燧原 streaming 配方在 T78 仅 +8%（与 T62 的 0.15-0.33 水位
  相容，c2flow 0.96 结构未破译）。

## 2026-09-18 维度特化：代理筛失败，不发射

- 候选：`heads/groups/nd/rd` 移出 do_not_specialize（重规划轴 3）。
  代理对照（513×128×128×64 等三形状）：ratio 0.947-1.052 **平手**——
  编译器按 %16 特化桶已做等价强度削减，显式 constexpr 无增量。
  门（wrapper ≥10%）未过，不发射。T78 剩余可试轴仅一维输出 tile
  （需先过特化对照的新基线，现无正信号支撑）。


## 2026-09-21 E3 候选就绪（燧原官方规则集重构，待 09-22 发射）

- 榜单情报更新：c2flow 均值 1.276→**1909.92**，全部来自燧原单芯 15268x
  （其余芯 0.28-2.96 与我方同量级）。机制（源码验证）：topsaten 非连续
  stride 不受支持 + cat/copy 在 _enflame CUSTOMIZED_UNUSED_OPS 禁用 +
  i64 NOT_SUPPORT 走 CPU——参考 `expand+cat+cast` 在 GCU 上病理。我方
  0.37x = 自身 kernel 病理（运行时 stride 阻断 DMA 判定 + int64 寻址）。
- e3（`1784de75`）：constexpr 化全部 shape/stride（连续性 wrapper 断言）、
  int32 寻址、12 CTA + num_warps=2（官方 gcu300 几何）。2D per-head 行存储
  保持 rope 间隔（codex-review P1：flat 跨头块在输出侧错误——已修，数值
  矩阵绿）。复审仅剩理论 P2（2^31 元素 assert 边界，GCU 不可达，接受）。
- 五元组：commit `1784de75`（HEAD `eb1e2c6d`）；ZIP
  `artifacts/competition/concat_and_cast_mha_k/e3-eb1e2c6/concat_and_cast_mha_k.zip`
  SHA `79709c72710d5a1bfe5394f045eefc7ee80a445c162976128b4016b7e4d9cb1f`
  （3 成员 generic/ascend/enflame）；回执
  `day5prep-20260921/concat_and_cast_mha_k/verification.json` SHA
  `a5a12f0bd8b7e9644c5f85dc96ef9bdd9146e69a299198a2d05938e7dd5b6bbb`
  （4 测试 0 失败，三源各 37 launch）。
- 预注册门：燧原 ≥2 保留（当前 0.40）；≥50 视为基线病理捕获成功；其余
  七芯不动（vendor-only 改动，generic/ascend 字节不变）。


## 2026-09-22 E3/E3R 平台终态：昆仑 0.024 双发同值，重掷轴封存

- E3（19373）：invalid_threshold。燧原 0.547（e2 0.372，+47%，未达 ≥2
  门）；**昆仑 0.024**（e2 同 generic 字节读 0.281）。
- E3R（19410，comment-only 载体 `21f60bb9` 前身）：昆仑**再次 0.024**
  ——同值复现 = 确定性环境劣化（评测侧 reference 变快或环境变化），非
  慢窗。按纪律停止重掷。
- TB e2 1.090875 守住。剩余：昆仑需 flat-1D vendor（0.024→≥0.1 才可能
  有效）；燧原 15268x 结构仍未破译（我们最好 0.55）。
