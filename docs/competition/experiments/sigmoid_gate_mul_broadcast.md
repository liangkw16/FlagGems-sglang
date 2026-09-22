# Task 90 `sigmoid_gate_mul_broadcast` 实验记录

```current
task: 90
operator: sigmoid_gate_mul_broadcast
batch: 6
validity: valid
platform: completed(e4,8/8,2.5247新TB;e5 2.5205判负保e4)
candidate_stage: e6
team_best_stage: e4
team_best_speedup: 2.5247
sealed: no
next: e6 燧原flat流式(relu2终态配方)待代理验证+ZIP上膛;轴=沐曦flat候选(e6回执后下一发)/华为1.25;昆仑0.75贴门槛
updated: 2026-09-22
```

## 2026-09-20 S0/E1 首发记录

- 开发+验证+codex-review 门后 00:12-00:27 发射（详见
  `artifacts/competition/nx-s0-20260920/`）。

## 2026-09-20 E1 平台终态：燧原 +12% 但均值持平，TB 保 2.472

- submission **18315** completed/valid，8/8，均值 2.433 < TB 2.472（保 s0）。
  燧原 0.8→0.9（+12%）；**海光 4.2→3.8（-10%，warps16 在此题为负——与
  T86 正例互补，warps 分化按 op 而非只按芯）**；华为 persistent 无增益。

## 2026-09-20 E2 平台终态：hygon 回退兑现，TB 2.495

- submission **18336** completed/valid，8/8，均值 **2.495 微幅新 TB**（vs
  2.472）。海光 3.8→4.3（回退 warps16 后恢复）；燧原 0.9 持平。距榜首
  3.21 差 29%。

## 2026-09-20 E3 平台终态：微幅新 TB 2.517

- submission **18352** completed/valid，8/8，均值 **2.517 微幅新 TB**（+0.9%）。
  天数 4.3→4.7；A 3.0 持平（warps8 在 sgmb 的 generic 路径收益小于 relu2
  ——同配方跨 op 部分迁移）。

## 2026-09-21 夜 e4 候选就绪（午夜第 4 弹）

- 结构（`f3b8cad1`）：`_ascend` persistent vendor（NVC 封顶）——榜首华为
  2.1 vs 我方 1.2。预注册门：华为 ≥1.5；均值 >2.517 换 TB。
- 五元组：commit `f3b8cad1`；ZIP `e4-f3b8cad` SHA
  `66e2793d107769a3c02e7884d3b7f58562a5a78e20b13d8f686c24f36866e0ff`；
  回执 `ready2-wave-20260921/sigmoid_gate_mul_broadcast/`（四路径）。

## 2026-09-21 00:13 E4 平台终态：2.5247 新高（+0.007）

- submission completed/valid。逐芯：天数 4.69（≈榜首 4.71）/ 沐曦 2.51 /
  燧原 **0.77** / 海光 4.26 / 昆仑 0.75 / 华为 1.25 / A 3.05 / B 2.91。
- 榜首 3.206（金狐狸）Δ -0.68 集中四洞：燧原 -1.74、沐曦 -1.03、
  B -0.97、华为 -0.89。天数/昆仑已追平——**vendor 轴（燧原/沐曦/
  B=card_b 无 vendor 位）是下一弹**。

## 2026-09-21 E5 候选就绪并发射（燧原 vendor 去 24-SIP 行封顶）

- 结构（`76982f26`）：_enflame 回退 generic 字节——24-SIP 网格封顶使
  one-program-per-row 只有 24 个程序（读数 0.77 vs 场带 2.5-3.4），
  GCU 需要程序级并行而非 SIP 数行循环。预注册门：燧原 ≥2.0；均值
  >2.5247 换 TB。
- 五元组：commit `76982f269271ac4c0def557403940ddfe7610bd6`；ZIP `e5-76982f2` SHA
  `cd0bd7a66201478b16c750df6564d161d397b9a134b9de4ff2fe4e6dc29b7e78`；
  test `449328d06e44d64f3add4b2ee8c36db6c770140e7dedea3ef78d078ba5874d2d`；
  回执 `top1day-20260921/sigmoid_gate_mul_broadcast/` SHA
  `fc831bff01a55d3fbca07d36ea28267b86921c8af8eba2a65b7fa8a68df7956c`。

## 2026-09-21 00:37 E5 平台终态：燧原 0.63 判负，行形式在 GCU 双向失败

- submission completed/valid，均值 2.5205 ≈ e4（2.5247，保 e4 TB）。
  燧原 0.77→**0.63**：24-SIP 封顶（0.77）与全网格 generic 字节（0.63）
  均远低于场带 2.5-3.4——行形式+HDIM 静态展开在 GCU 病理，程序数多寡
  两个方向都更差。**燧原轴关闭**（需第三方结构情报）。

## 2026-09-22 e6 候选实现（第 1 轮：燧原重开，flat 流式形态）

- 结构（本轮 commit）：`_enflame` 弃行形式，移植 T89 relu2 终态已验证
  配方——numel 一维 flat grid-stride + `min(cdiv(numel,65536),12)` 12-CTA
  封顶 + BLOCK 65536（relu2 阶梯宽度顶：2.61@16384→3.3@32768→
  **3.9@65536**→3.86@131072，12-CTA 终态 3.94）+ num_stages 3 + warps
  不钉。gate 经 `offs // HDIM`（constexpr 整除）gather 且带同一 m 掩码
  （OOB 防线）；int32 寻址域断言 `numel < 2^31-65536`（含一个尾块余量，
  防 masked lane 回绕）；vendor 内 assert x/gate row-major 连续（relu2
  先例 relu2.py:30）。e5 关轴时缺的第三方结构情报 = 同芯 relu2 已验证
  宽块阶梯 + 模板字节 8/8 先例。
- 回归：`test_flat_block_boundary_gate_gather` 新增并列入
  RELEASE_REQUIRED_TESTS——65536 整块无尾 / 块边界恰在行边界 / 块边界
  跨行（67584）/ 16 块 >12 CTA grid-stride 二趟 / hdim=1 逐元 gate；行间
  distinct gate（sigmoid 展布 0.047..0.953）使 gather 错位必超 2e-2 容差。
- 预注册门（e5 原门保留）：燧原 ≥2.0 保留 / ≥2.5 进场带（场带 2.5-3.39，
  EvokeAgent 3.2440）；均值 >2.5247 才换 TB；宽度阶梯 32768 为回退档。
  折扣因子：relu2 无 sigmoid+gather 的额外 ALU 未实测；expectedAvgGain
  0.1 = 进场带档 +0.22×约四成成功率（非 EvokeAgent 满配 +0.33）。
- 待办：release 代理验证 + e6-<commit> ZIP 后上膛；同题落选的 _metax
  flat 候选（17/18 队 ≥2.88 场带先验，天花板 +0.14 均值）与本候选文件
  互不重叠，e6 回执后可作沐曦轴下一发。
