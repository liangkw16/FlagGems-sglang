# Task 53 `fused_gdn_gating` 实验记录

```current
task: 53
operator: fused_gdn_gating
batch: 4
validity: valid
platform: 8/8(e4,2.1267x)
team_best_stage: e4
team_best_commit: 743ebb7f68109cdfd3b8c651aa88d8025d4d9d0c
team_best_speedup: 2.1267
sealed: no
next: e4沐曦2.7402x(+104%,过门,is_team_best);榜单:达标10队,榜首EvokeAgent 4.3306x(差51%);后续轴:generic二代[ROWS_TILE,H]
updated: 2026-09-07
```

## S0（2026-09-05 凌晨，submission 9866）

- 全 8 芯 correctness 通过；昆仑 0.0625x < 0.1x → invalid_threshold
- 逐芯：天数 2.88 / 沐曦 1.31 / 燧原 0.165 / 海光 2.79 / 昆仑 0.0625 /
  华为 1.19 / A 2.53 / B 2.73

## E1 昆仑向量化 vendor → **8/8 VALID**（submission 9877）

- 昆仑 vendor：标量循环→1024-lane 向量化（flat [B*H] 索引 + wrapper
  contiguous+view(-1)）
- **昆仑 0.0625→0.7168x（11.5x 跃升）** → **8/8 valid，avg 1.7691x**
- 逐芯：天数 2.33 / 沐曦 1.34 / 燧原 0.17 / 海光 2.98 / 昆仑 0.72 /
  华为 1.26 / A 2.65 / B 2.70

## 2026-09-05 Codex 会诊作战方案（预注册）

根因（已验证）：generic 每 program 一个标量 + grid-stride + 逐元素
`idx//H` 取模——昆仑 flat 向量化 0.0625→0.7168（11.5x）构成因果证据；
燧原 0.17/华为 1.26/沐曦 1.34 仍在跑标量形态 = -50% 榜差主因。

⚠️ **anchor 纪律（前置，0 发）**：E1 ZIP（`450e6fe`）与当前源码
（`6ed1fa9` 改过 softplus）字节已分叉，team_best_commit=TO_FILL。
下一发前二选一：从 `450e6fe` 精确字节建候选，或当前字节先做一次
8 芯 anchor，再叠加 vendor。

候选（按序单变量）：
1. **燧原 `_enflame`**：`grid=(B,)` 一 program 一行 + H 全宽向量、
   无循环无运行期分支（T51 同法把燧原从超时修到 2.34x）；门 ≥0.5x
   （0.3 为结构兑现下限）
2. **华为 `_ascend`**：行结构化 head-vector，lane 内无整除/取模，
   `H==BLOCK_H` 时整行无 mask；门 >1.60x（≥1.70 与 T42 +35% 先验一致）
3. **沐曦 `_metax`**：昆仑 flat 骨架 BLOCK=2048（勿先试 4096）；
   门 >1.50x
4. **generic 二代 `[ROWS_TILE, H]`**：A_log/dt_bias 广播 + `exp(A_log)`
   按行摊薄复用（lane 预算 ≤1024）——逼近榜首 3.54x 的广谱步骤

测试矩阵：B={0,1,3,32,257}、H={1,7,8,31,32,127,128,129}、threshold
两侧、beta={0.5,1,2}、a/b/A_log/dt_bias 极值与非连续布局；发射前
IR 检查（燧原无 scf.if/grid-stride、华为热路径无整数除法/取模）。

## E2 燧原行向量 vendor（2026-09-06，submission 10384）

- anchor 纪律兑现：generic+`_kunlunxin` 回滚 E1 平台已验证字节
  （`450e6fe`），唯一新变量 = `_enflame` 行向量 vendor（grid=(B,) +
  H 全宽 + 零循环零分支，softplus 用 E1 数学的 where 形态）；
  source `42d98e6`，ZIP `e2-42d98e6` SHA-256 `be3c6b9e…ec97`（3 成员）
- 附带修复：并行会话 6ed1fa9 的 "stable softplus" 改写从未过平台且
  在 NVIDIA 代理上可概率性放大误差（log(1+exp) 舍入 6e-8 ×
  exp(A_log) 超 1e-4）——已随回滚消除；测试种子化 + A_log 限幅
  ×2（平台实证范围），screening 5/5 OK
- **七芯已过**：天数 2.434/沐曦 1.3396/海光 3.1538/昆仑 0.7182/
  华为 1.29/A 2.446/B 2.7264——**燧原（目标芯）卡病态盒子
  waiting_callback**（T46 同日 1830s 前科）
- 待燧原裁决：≥0.5x → 轴兑现且 ~1.83x 新 TB；1830s 超时 → 平台侧
  invalid，候选封存等恢复窗口重载（需逐发授权）

## E2 燧原终态（2026-09-07 核实）

- submission 10384 燧原 `执行超时(1830s/1800s)`，子进程 R 状态
  （评测机忙，同字节其余七芯全过）——按病态盒子协议判平台侧超时，
  非代码回归；燧原轴不定罪，等健康窗口随 e3 顺带重试。
- 七芯成绩：天数 2.434 / 沐曦 1.3396 / 海光 3.1538 / 昆仑 0.7182 /
  华为 1.29 / A 2.446 / B 2.7264；team best 仍 e1 1.7691x。

## E3 华为行向量 vendor（2026-09-07 已提交）

- `_ascend`：一 program 一行 + H 全宽向量、lane 内无整除/取模、
  `H==H_PAD` 时整行无 mask（constexpr 分支）；a/b 显式行/列 stride，
  并按 SGLang #22312 教训在 variants 增加非连续 a/b 回归
  （`wide[:, ::2]` / `wide[:, 1::2]`）。
- softplus 保持 E1 平台已验证 where 形态；NVIDIA 代理 screening
  6 tests / 5 sources、release 25 launches 0 失败。source `d2d6ea9`，
  ZIP `e3-d2d6ea9`（4 成员）SHA-256
  `956f0bab7b132005a6d43c6cf74ed5809cd378295df753ea2fd0f64350740ea3`。
- 2026-09-07 提交评测中；门：华为 >1.60x；燧原同字节顺带重试 e2 裁决。

## E3 终态 → **8/8 VALID，avg 2.014025x 新 team best**（2026-09-07，seq 4）

- **华为 1.654x（1.29→+28%，过 >1.60 门）**；**燧原 1.605x**——e2 的
  1830s 超时确系评测机忙，行向量 vendor 实际兑现 0.17→1.605（9.4 倍）。
- 逐芯：天数 2.452 / 沐曦 1.3402（generic）/ 燧原 1.605 / 海光 2.763 /
  昆仑 0.7286 / 华为 1.654 / A 2.9108 / B 2.6586。
- 双门全过 → 按预注册顺序解除 e4 阻塞。

## E4 沐曦 flat vendor（2026-09-07 已提交）

- `_metax`：昆仑 E1 骨架 BLOCK=2048（勿先试 4096），flat [B*H] 索引
  改为显式 a/b 行/列 stride 寻址（免 contiguous 拷贝，#22312 类）。
- source `743ebb7`，ZIP `e4-743ebb7`（5 成员）SHA-256
  `536ea4c7f77954526ba19da566910b9bca120b8d2a21dd7d7d3a84c23fec0f0d`；
  release 回执 0 失败（`artifacts/competition/fused_gdn_gating/e4-743ebb7/`）。
- 2026-09-07 提交评测中；门：沐曦 >1.50x（当前 generic 1.34）。

## E4 终态 → **8/8 VALID，avg 2.1267x 新 team best**（2026-09-07，seq 6）

- **沐曦 2.7402x（1.3402→+104%，远过 >1.50 门）**——flat 向量化在沐曦
  与昆仑同源兑现。逐芯：天数 2.4364 / 沐曦 2.7402 / 燧原 1.5856 /
  海光 2.9286 / 昆仑 0.7162 / 华为 1.3676（ascend 同字节 -17%，评测机
  波动，e3 记录 1.654）/ A 2.6012 / B 2.6378。
- E2→E3→E4 三发单变量全部兑现；预注册剩余轴：generic 二代
  `[ROWS_TILE, H]`（A_log/dt_bias 广播复用），目标逼近榜首 3.54x。
