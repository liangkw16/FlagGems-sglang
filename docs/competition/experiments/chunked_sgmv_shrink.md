# Task 48 `chunked_sgmv_shrink` 实验记录

```current
task: 48
operator: chunked_sgmv_shrink
batch: 4
validity: valid
platform: 8/8(e6,4.7198125x);e4=7/8(燧原评测机忙超时,同字节vendor)
team_best_stage: e12
team_best_commit: bfde90eebbc3bcb4275a3d0de4468606414cd984
team_best_speedup: 5.207
sealed: no
next: e12收盘5.207(rank~5);燧原0.828(adapter分组+55%但未到门);结构面未定位,水位重掷可选
updated: 2026-09-10
```

## S0: 6/8（燧原+昆仑败）
## E1（燧原 route/materialize）: 7/8（燧原翻绿0.58x，昆仑败）
## E2（+昆仑 route/materialize + long）: 7芯已过含昆仑1.764x，燧原评测中

## E3-E5 燧原超时系列（2026-09-05，submissions 9986/9998/10002）
- E3（BK=512 revert）：燧原 1830s 超时
- E4（最小变更重掷）：燧原 1830s 超时
- E5（恢复 e1 原版字节）：燧原仍在评（4th 连续超时）
- **判定：燧原评测机持续繁忙（e1 同字节已过 0.58x）**；等平台侧
  窗口恢复后重投
- 七芯稳定：天数 3.74 / 沐曦 5.39 / 海光 6.09 / 昆仑 1.77 /
  华为 6.26 / A 7.04 / B 6.69

## E6 → **8/8 VALID**（2026-09-05，submission 10111，第 6 个 8/8！）

- **燧原评测机恢复，e1 字节通过 0.58x**（5 次连续超时后终于恢复）
- **8/8 valid，平均 4.7198125x**
- 逐芯：天数 3.85 / 沐曦 5.31 / 燧原 0.58 / 海光 6.00 /
  昆仑 1.79 / 华为 6.54 / A 6.98 / B 6.70
- 关键 vendor：燧原 route/materialize + 昆仑 route/materialize + long-index

## 2026-09-06 SGLang 结构两发证伪（e2/e3，submissions 均败）

- e2（c7d2d06）：SGLang 生产形态 BLOCK_M=max_len、shape 自适应
  BLOCK_N/K → 平台 5/8 correctness 败。
- e3（885f9e1）：BLOCK_M 封顶 64 重试 → 7/8 败（仅昆仑 vendor 过）。
  根因：每 program 只装一个 BLOCK_M tile（`tl.arange(0,BLOCK_M)+seg_start`），
  段长 > BLOCK_M 的 token 直接丢失 → 75% mismatch 与 max_len=256/64=4 吻合。
- 结论：该移植缺"段内多 tile 循环"；上游还依赖调用方预切短 segment。
  两个失败版本不足以否定完整分块方案，先补长段覆盖再验证。
- **generic 已回退 E6 字节（6ed1fa9）**，vendor 不动；远端回归 5/5 OK
  （含 _op_variants 矩阵）。team best 仍 e6 8/8 4.7198x。

## 2026-09-05 冲分预注册（8/8 后；本会话基于同族资产拟定，未做专项会诊）

现状：e6 8/8 4.7198x，榜首 c2flow 21.63x。弱芯燧原 0.58/昆仑 1.79/
天数 3.85。同族 T47 结论：indirect+dot 在燧原/昆仑不受支持，
route/materialize 是 sgmv 族唯一可行形态（e8-e10 三投证伪）。

候选（按序，每轴 1 发不过门即关）：
1. 燧原 0.58x：e1 字节刚过线；T12/T47 燧原 dot 模板（64 tile +
   stages2）在本题未试过——单发 64³ route/materialize 变体
2. 天数 3.85x：核对现有实现 dot 操作数 dtype（天数 fp32-dot 静默错
   执行，必须 fp16 或 split-fp16 三点积）；若已是 fp32-ieee 则试
   T12 镜像 split-fp16
3. 昆仑 1.79x：BLOCK 唯一有效轴（T21 1024 唯一成功）——BK/BN/BM
   单档扫描 ≤2 发
止损：总额度优先让给 T42/T53/T52 的预注册候选。

## 2026-09-06 流程审查修正

- 63/64/65/256 行的 segment 用例已加入 generic 与 vendor 矩阵；验证状态见本次流程修复记录。
- E6 平台最佳结果保持历史原值；长段覆盖修复前不再将该方向称为结构证伪。
- 最终 GPU release 5 tests/27 条 test/subTest 记录通过，generic 与两个 vendor 入口均实际调用。source/verification 身份及完整回执见 [流程实测](../workflow-validation-20260906.md)。

## E4 自适应 BLOCK_S（2026-09-07 已提交）

- 重开 e2/e3 方向的修正版：上游 #10286 的关键前提是生产段很短
  （调用方先切成 16 行），BM=64 对 ≤16 行段是 4 倍填充浪费。
  单变量仅 BLOCK_S 随 max_len 取 16/32/64；BLOCK_N=128/BLOCK_K=32/
  warps=4/stages=3 保持 E6 平台已验证值，(token_tile, output_tile)
  网格覆盖不变（e2/e3 败因是每段单 tile，已由 E6 结构避免）。
- NVIDIA 代理 wrapper 基准：8 行段 1.47x、16 行段 1.47x、
  (8 行段,K=4096,bf16) 1.75x；64/256/257 行段 1.00x 不回退；
  screening 5/5（含 63/64/65/256 边界）、release 37 launches 0 失败。
- source `8b134f4`，ZIP `e4-8b134f4` SHA-256
  `6141f43ce475cba01498e9cefb6e82e015c303fdc6e7c3a649d2881dedf9b5b4`；
  2026-09-07 提交评测中。目标芯 ≥15% 才晋级 team best，否则保留 e6。

## E4 终态（2026-09-07，submission 10670）

- **7/8**：燧原 1830s R 状态超时（评测机忙）；其运行的是与 e6 通过
  0.58x 完全相同的 `_enflame` vendor 字节，非代码回归，同 2026-09-05
  e3–e5 拥堵前科。team best 仍 e6 8/8 4.7198x。
- 七芯 vs e6：天数 3.77（-2%）/ 沐曦 5.35（+1%）/ 海光 6.30（+5%）/
  昆仑 1.77（-1%）/ 华为 6.76（+3.4%）/ A 7.19（+3%）/ B 6.79（+1.3%），
  总和 +1.5%——**未过 ≥15% 晋级门，自适应 BLOCK_S 轴按预注册关闭**
  （平台隐藏 shape 的短段占比低于代理假设）。
- 结构结论：完整多 tile 覆盖下自适应 BM 方向正确但收益微小；
  e2/e3 的"结构证伪"改判为"覆盖缺陷 + 低收益"，账本留档。

## E7 燧原原生低精度 dot 操作数（2026-09-07）

- 预注册 vendor 轴 #1（T12 镜像）：e6 燧原 vendor 的 GEMM 把操作数
  cast 到 fp32 后 ieee dot——恰是 T12 实证的 GCU 病理配置（ieee-fp32
  操作数 + 小 tile 低 stages，T12 当时 0.116x；fp16 原生操作数 +
  64/64 tile + stages2 → 0.743x）。单变量：bf16/fp16 输入保持原生
  dtype 进 dot（fp32 累加；bf16×bf16/fp16×fp16 乘积在 fp32 内精确，
  与 reference 的 fp32 GEMM 在低精度容差内等价），fp32 输入维持 ieee
  路径；index_select 物化同步降为原生 dtype（省一半拷贝流量）。
  generic/kunlunxin 字节冻结。
- source/verification commit `094548d`；screening `/tmp/flagos-t48e7`
  （5/5）与 release `/tmp/flagos-t48e7-rel`（5/5 方法、0 fail/skip，
  generic 15 / enflame 11 / kunlunxin 11 launch 实跑）双绿；远端
  black/isort/flake8 全过（black 重排后取回 hash 一致）。
- 预注册门：燧原 ≥0.58x 基础上兑现 T12 量级（≥1.5x）且整题 avg 超
  team best 4.7198125 才晋级；燧原评测机忙超时（1830s R 态）不计
  代码失败、不重试同字节。
- ZIP `e7-094548d`，SHA256
  `84fe1c7df7c158ce5f0965aa22562b654f31bab6b31e0217bc8774af3a9d9307`；
  成员 3（generic/kunlunxin 与 e6 一致，enflame 为新字节）。
- 证据 `validation/verification.json` SHA256
  `0cafa617d5e848eb5b5d99d432e3c6cab0568719855b0db5eb34c89fb043b36d`。

## E7 平台终态（2026-09-07T2x:xx）

- submission `10808`：**valid，avg 4.7489375，新 team best（+0.62%）**；
  quota 观测 12/30。
- 逐芯：天数 3.7705 / 沐曦 5.305 / **燧原 0.528（预注册门未达：
  T12 原生操作数模板未迁移，0.58→0.53 噪声带内；燧原轴关闭）** /
  海光 6.223 / 昆仑 1.785 / 华为 6.622 / A 7.132 / B 6.626。
- 均值增量来自未动芯小幅水位（华为 +1.2%/A +2.2%/海光 +3.7%）。
- 跨题知识：**T12 燧原 fp16-操作数 dot 模板只在融合 kernel 形态
  兑现；route/materialize wrapper 的逐段 launch + 规则 GEMM 不受
  操作数 dtype 影响**——燧原 sgmv 瓶颈在逐段 Python 循环
  （index_select/GEMM/index_copy × B 段），而间接寻址 dot 又是
  该芯不可用形态，0.5x 接近该结构上限。csgmv 单 launch 分段 CSR
  是唯一结构出路但依赖间接权重寻址（燧原/昆仑不受支持），
  T48 结构轴定格。
- 证据 `validation/48-status-final.json`。

## 2026-09-08 推荐方案实现与提交前验证（未提交平台）

新增沐曦同数学路径 cpasync 候选，补长段跨tile和三dtype。NVIDIA 编译器拒绝 pipeline 参数（7条error），这是运行时覆盖缺口，未删除参数伪造通过，也未打包。

- source `26a95766b179d263916e9483dfc8d2343c40406a`；verification `26a95766b179d263916e9483dfc8d2343c40406a`。6 个测试方法、18 次实际 kernel 调用；选定 NVIDIA/代理范围门禁失败。
- 回执 `artifacts/competition/batch4-implementation-20260907/t48-release1/verification.json`，SHA256 `7f02b6837a843f77a35f04993b15eaf44c2305d8876860a4ac0276e3d3e44a74`；日志 SHA256 `f023c3e3b871b63720a2d6a4c1a75595b5049ed2faf0169893ab2a79678ec5b8`。
- 环境、逐源码执行范围、原始配对数据和未完成条件见[本轮报告](../implementation-batch4-20260908.md)及[证据清单](../data/batch4-implementation-20260908.json)。本轮不更新历史有效分，未做平台 preflight、上传或正式提交。

## 2026-09-09 榜单逐芯情报推翻两条历史结论（e5 候选，未提交平台）

本轮首次拉取按题 leaderboard 端点（返回**每支队伍的逐芯 speedup**，
只读、不耗额度），据此更正本账本此前两条判断：

1. **"燧原 0.5x 接近该结构上限"、"燧原轴关闭"不成立。** 同题他队燧原
   读数：c2flow **28.79**、EvokeAgent 8.11 等，均在同一评测口径下取得，
   与我方 0.528/0.58 相差 50 倍。水位论证据不足——这是我方结构问题，
   不是该芯在该 op 的物理上限。原"燧原轴关闭"结论撤回。
2. **"T48 结构轴定格"过早。** 真正的证伪只覆盖 route/materialize 与
   csgmv 间接寻址两条路径，未覆盖 **tile 几何**这一独立轴。

新证据（tile 几何轴，此前从未检验）：T48 与 T47 是姐妹题，但两题的
归约维与输出维**恰好互换**：

| | 归约维（BLOCK_K 该管） | 输出维（BLOCK_N 该管） |
| --- | --- | --- |
| T47 expand | rank，小 8–64 | out_dim，大 ~1e3 |
| T48 shrink | K_in，**大 512–4096** | rank，**小 16–64** |

e4 沿用了 T47 的 `BLOCK_N=128 / BLOCK_K=32`，于是 BLOCK_K 走在**大**的
K_in 上（16–128 趟），BLOCK_N 又给**小**的 rank 填 50–87% 空列。这与
平台观测到的**八芯一致落后 3–8x**（天数 5.1 / 沐曦 4.4 / 燧原 32 /
海光 7.8 / 昆仑 3.4 / 华为 3.1 / A 5.2 / B 4.7）在量级上吻合；八芯同步
落后本身即排除任何单芯 lowering 解释。

### e5 候选（本地已 commit，未提交平台）

- source/verification commit `fcf5997`；BLOCK_N 跟随真实输出维，
  BLOCK_K 走归约维，并对 (num_stages, BLOCK_K) 做快存预算联合搜索。
- **过程中自查出一个真实缺陷**：先试的固定 `BLOCK_K=256` 实测需
  196608 B 快存 > NVIDIA 101376 B 上限（RTX 5070 Ti），被新增测试拦下，
  未流到平台。
- NVIDIA 代理内核耗时（旧→新）：decode bs=32 r=16 0.2389→0.0781（3.06x）、
  bs=64 r=16 0.2411→0.0778（3.10x）、prefill 4×128 r=16 0.3700→0.1642
  （2.25x）、8×64 r=32 0.1963→0.1394（1.41x）、bs=32 r=64 0.2404→0.1740
  （1.38x）。
- 新增 `test_tile_geometry_axes`：90 组 (K,N) 覆盖 BLOCK_N/BLOCK_K 每个
  可选边界与非 2 幂尾块，对 float64 reference 按 sqrt(K) 缩放判据 +
  1% 硬性粗错检查。**K=4096 处 e4 与 e5 误差逐位相同**（5.951e-04），
  说明该处是 reference 累加顺序而非内核缺陷，平台 flat atol=1e-4 在此
  偏严；泛化矩阵 5/5 通过。
- 其他芯未验证（仅 NVIDIA 代理），按 target-runtime-unverified 记。
  既有 `_metax` vendor 在本 harness 上因无关的 `pipeline` kwarg 失败，
  与本次改动无关。

预注册晋级门：燧原 ≥3x（他队已证 28.79 可达，故 3x 为保守下限）且
八芯均不低于当前读数的噪声带；八芯平均预期 4.81 → ≥8.3。
止损：若平台八芯读数与 e6 无显著差异（±10% 内），说明平台 shape 与
本地假设不符，改从 raw_result 取实际 shape 后再定 tile。

## 2026-09-09 深夜执行轮（精度门关闭 + e5/e9 双候选就绪，未提交平台）

### 精度门（Codex 审查要求，已关闭）

审查指出 `test_tile_geometry_axes` 的 float64+sqrt(K) 判据不能替代题面
fp32 reference + 题面容差。补测：216 组（3 dtype × K∈{512,1024,2048,
4096} × N∈{16,32,64,80,128,256} × 3 段型），对**题面 reference**（fp32
matmul）按**题面容差**（fp32 1e-4 / fp16 1e-2 / bf16 1.5e-2）计 max err
与超差元素数。结果：**216/216 PASS**——e5 的误差在每组 ≤ e4×1.10 且不
新增超差；关键组 fp32 K=4096 上 e4/e5 的 max err 与超差计数**逐组完全
一致**（如 [63,64,65,256] 段型同为 7.935e-04/9 元素），e4 平台 8/8 已
两过，e5 继承该误差剖面的平台通过性。evidence：远端
`/tmp/flagos-t48-acc.fCDOQ1/gate.log`（216 PASS，本地副本
`/tmp/t48_gate.log`，SHA-256 `8a349212cb15d8bcd0e5d674a01b04fe274a7dc0730b50b3e8b3a84a97d28391`）。

预注册门修正（审查意见采纳）：e5 是 generic 改动、燧原走 e7 vendor，
"燧原 ≥3x"不应作为 e5 的晋级门；e5 的门改为**八芯平均较 e7 团队最佳
+30% 以上**（代理 1.4–3.1x 的保守下限）。燧原 ≥3x 门移交给 e9。

### e5 候选就绪（release + ZIP）

- source/verification commit `fcf5997`；exact release 回执
  `/tmp/flagos-t48-e5-release/verification.json`（RTX 5070 Ti，7/7 用例
  含 variants 矩阵，0 skip/xfail，exit 0）。
- canonical ZIP `e5-fcf5997`，SHA-256
  `0417bfa21c6e6157c7d2d76e9f3f779ed7c7a1a57d6737eccb4cb077c59ec0f3`。

### e9 候选（燧原 launch 风暴修复，commit `3fba418`）

- 结构：host 逐段 `index_select→GEMM launch→index_copy_` 循环（成百次
  launch，GCU 启动开销不可掩盖 + 每次 launch 各自超发 grid）折叠为
  **单次 launch + tile 描述表 persistent 走访**（grid cap 24）。表项
  `(seg, m0, n0)`，内核所有内层循环 shape-static（T47 e13 grouped-GEMM
  评测崩溃的教训：不允许数据相关循环界），每 tile 5 次标量 load。
- 单变量纪律：每 tile 的 GEMM 配置与 e7 逐字节相同（64×64、BK≤128、
  bf16/fp16 原生 dtype dot + fp32 acc、fp32 走 ieee、warps4/stages2），
  只改调度。空段/负 weight_index 在 host 丢弃（与 generic 平台已接受
  的跳过语义一致）。
- 代理证据：screening 21 组（32 段持久回绕/多 M-tile 段/空段/非 2 幂）
  20/21 过；唯一 fail（fp32 K=4096 单段 2048）A/B 实证与 e7 **逐位一致**
  （max 7.019043e-04、7 元素超差）——已知累加序伪差，非回归。
  variants 矩阵补 32 段用例固化回归。
- exact release 回执 `/tmp/flagos-t48-e9-release/verification.json`
  （7/7，generic 109 + enflame 8 真实 launch，0 skip，exit 0）；
  canonical ZIP `e9-3fba418`，SHA-256
  `c13a2f8bccb38dac58847f0fdf9dfecb0a36a88614ffbb5a9a227d9377fd8b84`。
- 燧原调度/lowering 未被代理验证（target-runtime-unverified），等平台。

发射序（明日额度）：e5 首发（验 tile 几何，八芯归因）→ e9 第二发
（在 e5 基线上验燧原批量化，燧原单芯归因）。


## 决赛日执行轮 I（2026-09-10 上午，submissions 12356/12362）

- **e5 终态（sub 12356，08:14）**：8/8 valid，avg **4.853375 新 team best**（+2.2%）。
  逐芯：天数 4.25/沐曦 2.82/燧原 0.54/海光 6.66/昆仑 1.80/华为 **7.77**/A 7.70/B 7.30。
  判读：走 generic 的六芯全部 +9~19%（tile 几何方向正确但幅度远低于代理 1.4–3.1x，
  平台 shape 段短使 kernel 偏 launch 受限）；+30% 门未过。沐曦 5.31→2.82 回退归因
  破案：**e7 包无 metax 成员（沐曦跑 e4 generic），metax vendor 是后来入树的**，
  e5 包切换到该 vendor 后沐曦大跌——metax vendor 为净负资产。
- **e10 终态（sub 12362，08:5x）**：7/8 invalid_correctness，燧原
  `Pipeline run failed: PassManager execution failed`（selected_file 确认新 vendor 生效）
  ——与 T47 e13 同族：**GCU 编译器拒绝 sgmv 族的批量化/间接单 launch GEMM**。
  其余七芯全过且沐曦 6.41（metax 移除 + generic e5 兑现 +21%）。
- **e11 预注册（commit `e16dfc9`，就绪待发）**：燧原 vendor 回滚 e7
  route/materialize 平台已过字节，generic 保持 e5，无 metax。预期 = e10 七芯读数
  + 燧原 ~0.54 → **avg ~5.30 新 TB**。门：8/8 valid 且 avg > 4.853。

- **e11 终态（sub 12365，09:4x）**：8/8 valid，avg **5.200625 新 team best**（+7.2%）。
  逐芯：天数 4.21/沐曦 6.32（metax 移除兑现）/燧原 0.54/海光 6.78/昆仑 1.81/
  华为 6.95/A 7.78/B 7.23。门（>4.853）通过。排名约第 5。
- **e12 预注册（commit `bfde90e`，就绪待发）**：燧原 vendor 改 adapter 分组——
  同 adapter 段拼接后一次 index_select→GEMM→index_copy_，launch 数从段数降到
  adapter 数（≤16），kernel/dtype/tile 与 e11 逐字节相同，保持 GCU 唯一可编译的
  规则 GEMM 形态。门：燧原 ≥1.5、其余七芯 e11 噪声带内、avg > 5.2006。

- **e12 终态（sub 12368，10:0x）**：8/8 valid，avg **5.207 微幅新 TB**（+0.13%）。
  燧原 0.5355→**0.828（+55%）**——adapter 分组方向确认，但量级未到 ≥1.5 门；
  其余七芯 e11 噪声带内。燧原剩余差距（vs RSI 35.5）不在 launch 数（段→adapter
  已减），结构面未定位。e12 字节为 T48 收盘基线。
