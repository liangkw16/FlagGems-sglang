# Task 78 `concat_and_cast_mha_k` 实验记录

```current
task: 78
operator: concat_and_cast_mha_k
batch: 6
validity: valid
platform: completed(17300,e2,8/8,1.090775x新TB;华为persistent+97%兑现)
candidate_stage: e2
team_best_stage: e2
sealed: no
next: 华为轴关闭(0.327超c2flow 0.245);剩余缺口=海光 1.52→1.96+/天数 2.11→2.42/燧原 0.39→0.96;e3 候选=海光窗口观察+燧原结构(未破译)
updated: 2026-09-18
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
