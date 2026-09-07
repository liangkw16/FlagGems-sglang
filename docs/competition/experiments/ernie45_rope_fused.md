# Task 49 `ernie45_rope_fused` 实验记录

```current
task: 49
operator: ernie45_rope_fused
batch: 4
validity: valid
platform: 8/8(e3,8.58253125x)
team_best_stage: e3
team_best_commit: 5a6b8cb3eac758d266765f96cdf52a4581deb2a9
team_best_speedup: 8.58253125
sealed: no
next: head16候选代理正确/ZIP验签;需昆仑live buffer和完整wrapper收益后晋级
updated: 2026-09-08
```

## S0: 6/8（燧原PassManager + 昆仑uni_sram）
## E1（precomputed-pos vendor）: **7/8**（燧原翻绿0.552x！昆仑仍uni_sram）
- 逐芯：天数 16.24 / 沐曦 8.85 / **燧原 0.552** / 海光 9.33 /
  昆仑 FAIL / 华为 2.48 / A 22.14 / B 9.88
- Precomputed-pos vendor：wrapper PyTorch 预计算 [T, half_rd] 位置
  张量（纯元数据），kernel 变纯 gather+RoPE 零分支——PassManager
  毒点彻底消除

## E2 昆仑 HEADS_TILE=1 + coreTiling（submission 10035）
- 昆仑仍 uni_sram（3 投：S0 generic / e1 precomputed-pos / e2 最小
  tile+coreTiling）→ **昆仑 conclusive 封轴**
- 逐芯：天数 16.26 / 沐曦 8.98 / 燧原 0.577 / 海光 9.32 /
  昆仑 FAIL / 华为 2.47 / A 21.54 / B 9.87

## E3 昆仑 pair-per-program（2026-09-07 已提交）

- 借用官方 PR40 `mrope_fused` 昆仑工作划分：3D grid
  `(tokens, head_groups, pairs)` + 标量位置/cos/sin + `BLOCK_HEADS=8` +
  `num_warps=1` + 尾部独立 copy kernel；仅换轴规则为 ERNIE 的 H/W
  奇偶交替 + 尾部 T（mrope 的连续 T|H|W 段不适用）。
- 与 e2（HEADS_TILE=1，整段旋转向量仍驻留）不同，每 program 活跃状态
  缩至 8 lane 标量对，直接针对 uni_sram 墙；grid 总数继承 PR40 已过
  昆仑的证据（未在本地复证 65535 上限，风险随平台裁决记录）。
- variants 矩阵补非 128 head size、t 段切分与 bf16；NVIDIA 代理
  screening 5 tests / 3 sources 通过，release 回执（7 launches ×
  50）0 失败。source `5a6b8cb`，ZIP `e3-5a6b8cb` SHA-256
  `36db259a70a94e5a950f6a96b0c016c9cee139a7b1f3637d664e80aaba1e1c96`。
- 2026-09-07 seq2 提交评测中；晋级门：昆仑 ≥0.12x 且其余七芯不回退。

## E3 终态 → **8/8 VALID**（2026-09-07，seq 2，第 9 个 8/8）

- **昆仑 0.66425x——pair-per-program 击穿 uni_sram 墙**（S0/e1/e2 三投
  均败后，第 4 发换结构通过）；燧原 0.577→0.6265，华为 2.47→2.5365。
- **8/8 valid，avg 8.58253125x**：天数 16.10 / 沐曦 8.78 / 燧原 0.627 /
  海光 8.95 / **昆仑 0.664** / 华为 2.54 / A 21.11 / B 9.90。
- PR40 工作划分在昆仑的 grid 总数上限风险未兑现；后续可选轴：昆仑
  `BLOCK_HEADS`（8→16/4）与 A 芯 21.11x 的参数微调，非必需。

## 2026-09-08 推荐方案实现与提交前验证（未提交平台）

昆仑保留单 pair 结构，BLOCK_HEADS 8→16；补3/4/5/7/8/9/15/16/17 head边界。代理大head案例约1.15x，小案例持平。

- source `26a95766b179d263916e9483dfc8d2343c40406a`；verification `26a95766b179d263916e9483dfc8d2343c40406a`。6 个测试方法、204 次实际 kernel 调用；选定 NVIDIA/代理范围门禁通过。
- 回执 `artifacts/competition/batch4-implementation-20260907/t49-release1/verification.json`，SHA256 `99e82fc4a2640ca9d7f691870553773b84fec3a1ca1d95d68f0639328f9f22c9`；日志 SHA256 `e70ce8849ceea5342b7b8c247e24ce3c23a6070862adf4235e75012c928fe5ba`。
- 不可变 ZIP `artifacts/competition/ernie45_rope_fused/research-20260908-26a9576/ernie45_rope_fused.zip`，SHA256 `f7c5a9a4a34bc4ee97619b7225cd9d4bfe0b0cc5774d83983751a071130b6ed0`；与 dry-run manifest、构建和 existing 验签一致。ZIP 是候选产物，不等于目标芯或平台已通过。
- 环境、逐源码执行范围、原始配对数据和未完成条件见[本轮报告](../implementation-batch4-20260908.md)及[证据清单](../data/batch4-implementation-20260908.json)。本轮不更新历史有效分，未做平台 preflight、上传或正式提交。
