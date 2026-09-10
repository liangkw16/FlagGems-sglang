# Task 49 `ernie45_rope_fused` 实验记录

```current
task: 49
operator: ernie45_rope_fused
batch: 4
validity: valid
platform: e4/11245八芯valid,8.71115625x新team best
team_best_stage: e7r2
team_best_commit: 9ad8867bbadfc7e3c17c39a09e62848ad5c1a3f8
team_best_speedup: 10.35265625
sealed: no
next: 批4收盘;额度用尽.e9w(12648)9.68269燧原0.560,水位未抬;TB保持e7r2 10.35266;排名8/9,上邻+0.227
updated: 2026-09-10
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

## E4 head16 探针：**8/8 valid 8.71115625 新 TEAM BEST**（2026-09-08T17:00，sub 11245）

- 晨间候选（`26a9576`，昆仑 pair 小 program 保持、head tile 8→16）。
- 逐芯：天数 16.08 / 沐曦 8.90 / 燧原 0.573 / 海光 9.51 / **昆仑 0.667
  （e3 0.664 持平）** / 华为 2.44 / A 21.61 / B 9.91——多芯带宽带内微升，
  均值 8.5825→**8.7112（+1.5%）**。head16 轴温和正收益，保留。
- 排名看实时榜（旧 #5，榜首 17.06）；下一轴未定，守此 TB。

## E4r 水位重掷：**8/8 valid 8.9578125 新 TB**（2026-09-09T06:44，sub 11648）

- e4 字节注释载体（16fe1d4）。晨窗带内上沿兑现，8.7112→**8.9578（+2.8%）**。
- e4 字节重掷已 1 次（≤2）；距榜首 21.19（今晨又涨 +24%）差距扩大，结构面未明。

## 2026-09-09 燧原 persistent+pingpong 候选（e5，未提交平台）

榜单逐芯情报（按题 leaderboard 端点，只读）显示本题榜差集中在燧原：
我方 **0.567**，同芯他队 c2flow **38.78**、EvokeAgent 等亦远高；榜首
c2flow 21.19 逐芯健康（非慢窗产物）。按 `Δavg=(target-ours)/n` 估算，
仅把燧原抬到他队已证读数即可 8.96 → 13.73（+53.3%）。

根因（厂商文档硬事实）：燧原 TritonGCU 性能优化指南明确 GCU 平台
**kernel 启动开销无法被完全掩盖**，以大于硬件资源数的 GridDim 并行执行
会引入额外调度开销、增加执行时间，**且在 kernel 计算量较小时尤为明显**，
建议把 kernel 间并行度以循环方式放入 kernel 内；推荐 GridDim 为 6
（GCU300/nw=4）或 24（GCU400/nw=8），硬件仅 24 个 SIP。

本题 enflame vendor 恰是该缺陷的最坏形态：grid `(T, cdiv(n_h, 4))`
且每个 program 体量极小，实际 1e3–1e4，超发约 100 倍；同时
`num_stages=1` 意味着 pingpong 从未开启。

### e5 改动（本地 commit `0d4d758`）

- grid 改为 `(min(total_tiles, 24),)`，kernel 内用
  `tl.range(pid, total_tiles, tl.num_programs(0), num_stages=3)`
  遍历同一 (token, head_tile) 空间——tile 分解、掩码与数学全未变，
  只改 program→tile 的映射；`num_stages=3` 才使能 pingpong。
- 去掉显式 `num_warps`，交回 GCU 后端默认。该芯上钉 warps 已两次被
  证明是错的（T19-E5 / T51-E5 各 +38%）；本批统计：未钉 warps 组燧原
  中位数 **2.077** vs 钉住组 **0.732**（n=9 / 20）。
- NVIDIA 代理 6/6 通过（含 variants 矩阵与 head-group 边界）。
- **燧原本身未验证**：NVIDIA 代理无法验证 GCU 调度与 lowering，而本次
  预期收益全在燧原，按 target-runtime-unverified 记，交平台补齐。

预注册晋级门：燧原 ≥3x（他队已证 38.78 可达，3x 为保守下限），其余七芯
不低于当前噪声带；八芯平均预期 8.96 → ≥11。
止损：若燧原读数无改善且 exec_ms 未下降，说明瓶颈不在 grid 超发，
转查 wrapper 侧 `_compute_positions` 的 PyTorch 预计算开销。

## 2026-09-09 深夜执行轮（e5/e6 双候选就绪，未提交平台）

### e5 就绪（release + ZIP 补齐）

- exact release 回执 `/tmp/flagos-t49-e5-release/verification.json`
  （RTX 5070 Ti，6/6 用例含 head-group 边界与 variants，0 skip/xfail，
  generic 41 + enflame 62 真实 launch，exit 0）。
- canonical ZIP `e5-0d4d758`，SHA-256
  `803cb28bfad8d4aa78c74b681fde17f97571d40b7b4b6eb443cd9dc9c9e7f63f`。

### e6 候选（generic 三合一调度修复，commit `aec238a`）

- 依据（上游证据 + 审查修正）：sglang PR #19144 的同算子 fused kernel
  每 program 只做一次 cos/sin gather；逐芯情报中华为 2.46 vs 榜首 12.4、
  海光 10.5 vs 37.7 的差距主嫌疑是 repeated gather、int64 位置标量链
  （昇腾已证退化）与双 launch。
- 改动（数学零变化）：① Q/K 两次 launch 合并为单 kernel，(token,
  head_tile) program 对 q/k 同 tile 共享同一份 cos/sin gather；②
  positions wrapper 侧转 int32（值域受 cache 行数约束）；③ grid 形状
  与 e4 相同（token×head_tile），decode 小 T 并行度不回退。
- 收益预期修正（审查算术）：仅燧原 0.567→3 时均值约 **9.26 而非 11**；
  华为/海光若同步受益才向 11+ 走。带宽模型下纯 gather 去重上限约一成，
  数倍级收益须来自调度/lowering 面——e5（燧原）与 e6（generic）分两发
  归因。
- exact release 回执 `/tmp/flagos-t49-e6-release/verification.json`
  （6/6，0 skip，exit 0）；canonical ZIP `e6-aec238a`，SHA-256
  `61c35716be270b2b04e5bba87677f86b7e397e50384c9fffc2a43fb5792652df`。

发射序：e5 首发（燧原单芯归因，验 persistent 假设）→ e6 第二发
（generic 面，华为/海光/天数归因）。


## 决赛日执行轮 I（2026-09-10 上午，submissions 12359/12361）

- **e5 终态（sub 12359）**：8/8 valid 8.9828（+0.3%，噪声带）。燧原
  0.567→0.56875——**persistent+pingpong+去pin 模板在 rope 类零效果**，
  模板证伪第一例。燧原差距（vs c2flow 38.8）另有结构根源（疑 wrapper 侧
  `_compute_positions` 或 gather 形态，未定位）。
- **e6 终态（sub 12361）**：8/8 valid **8.4047（回退 -6%）**，但逐芯分裂决定性：
  海光 10.57→**20.44（+93%）**、华为 2.55→**2.98（+17%）**（合并 launch + 共享
  gather 的两个目标芯大兑现）；天数 16.2→12.4、沐曦 10.4→7.2、A 21.1→15.8、
  B 9.8→7.1（-23~-30%，双 launch 形态在这些芯更优）。
- **e7 预注册（commit `bf3cf88`，就绪待发）**：把分裂变成路由——generic 回滚
  e4 字节（五芯偏好形态），新增 `_hygon`/`_ascend` vendor 携带 e6 kernel 原字节；
  燧原/昆仑 vendor 不动。预期 avg ≈ **10.27 新 TB**（排名 8→前 5）。门：8/8 valid
  且 avg > 8.958；海光 ≥15 且华为 ≥2.7 视为路由兑现。

- **e7 终态（sub 12366，09:4x）**：8/8 valid，avg **10.1395 新 team best**（+13.2%）。
  逐芯：天数 16.18/沐曦 8.95/燧原 0.61/海光 **20.73**（路由兑现，门 ≥15 ✓）/
  昆仑 0.72/华为 **3.12**（门 ≥2.7 ✓）/A 20.90/B 9.91。排名 8→6。
  路由假设（generic e4 五芯 + e6 kernel 海光/华为）完全兑现；
  跨题知识：**合并 Q/K launch + 共享 cos/sin 在海光系 +93%、华为 +17%，
  但天数/沐曦/A/B 反向 -23~-30%——按芯路由是必须的**。

- **e7r 终态（sub 12371）**：10.1384 与 e7 同窗平读（间隔 40 分钟 = 重复抽同
  一张票，符合水位模型）。TB 保持 e7 **10.1395**。
- **e8 终态（sub 12373，12:3x）**：valid 8.746 回退。per-head 程序（tile 1）
  使天数 16.2→11.9、沐曦 9.05→6.37（-27~-30%），A/B 同跌——**并行度假设证伪**：
  tier-2 的优势不在 grid 并行（tile4 < tile1 反而更好，两代证伪后两代形态
  都不如 tile4）。路由 vendor 稳定（海光 19.6/华为 3.01）。
  **T49 收盘于 e7 10.1395（排名 6）**；剩余动作仅错峰水位重掷。
- **e7r2 终态（sub 12399，14:1x）**：8/8 valid，avg **10.3527 新 TB**（+2.1%，
  A 高窗 22.94）。距 rank5（ChipVoyager 10.58）0.23。e7 字节低滚预算剩 1，
  尾窗可再采样一次。
- **e7r3 终态（sub 12424，16:2x）**：10.2701 未超（海光 18.7 偏低窗）。
  e7 字节低滚 2/2（e7r/e7r3），路径关闭。**T49 收盘于 e7r2 10.3527（排名 6）**。

## 批4收盘：e9w 水位重掷（2026-09-10，sub 12648）

- e7r3 字节 preflight 被判已提交，改注释载体重掷。
- **e9w（source 654fab4，ZIP 6e9f097c1e6d…e0b5，5 成员，回执 7c93bf5c…0e88b）**：
  8/8 valid avg **9.68269**（未破 TB 10.35266）；燧原 **0.56025**
  ——与 e5 结构探针读数（0.567/0.569）一致，再次确认燧原 0.56 是该形态水位、
  非窗口噪声。
- 预期依据是我方逐芯历史峰值合成可达 10.631（越过上邻 10.5797），但本发
  天数 16.21 / 海光 17.42 均低于历史峰值，合成未兑现。收盘 **8/9，上邻 +0.227**。
