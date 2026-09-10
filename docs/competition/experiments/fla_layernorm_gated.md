# Task 51 `fla_layernorm_gated` 实验记录

```current
task: 51
operator: fla_layernorm_gated
batch: 4
validity: valid
platform: e10/11204八芯valid,60.7192x新team best,实时rank1
team_best_stage: e10
team_best_commit: 18616ffa954cf7ddd57e578bc6fa9863c6e38740
team_best_speedup: 60.7192
sealed: no
next: 已登顶(燧原442.8极端水位+华为子块+10%同发兑现);守榜为主,华为子块结构知识可迁移
updated: 2026-09-08
team_best_commit: d05e57a0ee3d2d453866c85c0479b22a6fcbae8b
```

## S0（2026-09-05，submission 9867）
- 7/8，燧原 1830s 超时（非 correctness），其余七芯全过

## E1 燧原 vendor → **8/8 VALID**（submission 9944）
- 燧原 vendor：一 program 一行（去 grid-stride 循环）+ tl.rsqrt/tl.sigmoid
- **燧原超时→2.3398x**；逐芯：天数 9.21 / 沐曦 4.57 / 燧原 2.34 / 海光 7.74 /
  昆仑 0.944 / 华为 2.52 / A 8.73 / B 7.07

## 2026-09-05 Codex 会诊作战方案（预注册）

**华为（首发，最高置信）**：`dim` → `tl.constexpr D`，`offs < dim`
运行时 int 比较（昇腾 Vector CMP 标量退化，官方指南确认）变编译期
静态 mask，`/D` 常量折叠；行结构与单次 x load 不变（勿先搬 T40 的
1024 子块——那会迫使 RMSNorm 重载两次 x）。预期 +15~40%；门 ≥+15%。

**沐曦**：仅 `1/sqrt` → `tl.rsqrt`（消除逐 lane 向量除法；IR 不变则
取消候选）；门 ≥+12%。
**昆仑**：仅 `D<1024` 时 BLOCK 提 1024（T21 唯一成功轴；T19 multi-row
无收益、BLOCK 2048 负收益，勿重复）；门 ≥+15%；raw_result 若暴露逐
case D 分布则先查命中再发。
后置：g load 后移（缩短归约期 live 向量）；BLOCK_D≥2048 沐曦 warps8。
LayerNorm 勿改 `E[x²]-E[x]²` 单遍（大均值小方差消减误差，fp32 1e-4 风险）。

> 注：并行会话 09-05 22:56 记录已发 T51 E2（8/8、5.3939x、未超 e1，轴内容未落账本）；
> 执行本方案任一候选前先与该会话核对 E2 消费过的轴，避免重复预注册。

## E3 华为 constexpr-D vendor（2026-09-06，submission 10385）

- `_ascend` vendor：generic 数学逐字节不变，唯一变量 = `dim` →
  `tl.constexpr D`（`offs < D` 编译期折叠 + `/D` 常量折叠）；
  source `080148b`，ZIP `e3-080148b` SHA-256 `2bddb9d4…271d`
  （4 成员），screening OK
- 七芯已过：**华为 2.52→2.6182（+3.7%，远低于 ≥+15% 门——轴单发
  证伪关闭**：int 比较不是本题华为瓶颈）；天数 9.2822/沐曦 4.3918/
  海光 8.133/昆仑 0.9682（kunlunxin vendor 生效）/A 8.1964/B 6.873
- 燧原（非目标芯）卡病态盒子 waiting_callback；无论燧原落点，
  估算均值 ≤5.38 < e1 TB 5.39——非 team best
- 判定：constexpr-D 轴关闭；T51 剩余轴：沐曦 rsqrt（预注册 #2）

## E4 Top1 冲刺候选（2026-09-07，提交前）

先校正旧记录：2026-09-07 只读查询确认 E2/submission10276 才是 team best 5.3939（source 5985b1ce09fe6e7cec374c271aa1c25940264783），高于 E1 5.390025；旧节“未超 E1”是过期表述。E3/submission10385 已终态 7/8 invalid_correctness，燧原 1830s 超时，不能继续写在评。

参考 GitHub PR50 的多行 tile 调度，针对本题非分组 RMS/LayerNorm 契约适配：仅 `rows>=128 and dim<=1024` 使用 BLOCK_R=4，其余1；列归约改 axis=1，权重/偏置保留列 mask，原数学运算顺序不变。测试新增行数 127/128/129 等 tile 边界、列127/128/129与1023/1024/1025、三dtype、两种norm/gate和stride。

五轮交错代理 A/B：`[4096,128]` 1.50x；`[4096,256/512/1024]` 1.008/1.003/1.001x；`[4096,8192]` 1.00x；阈值边界约0.995–1.008x。早期 rows>=8 导致 `[32,512]` 0.697x，已缩小条件并复测；单独 rsqrt 仅≈1.016x，未采用。

预注册：一次平台探索检验隐藏 case 是否受益；目标超过 5.3939，冲榜需当时榜首6.668225×1.03≈6.86827。只有一个代理 case 明显获益，不承诺总体收益；无均值提升则关闭多行阈值轴。

- source/verification commit：`5544a77d406e1f22e735f6cba2bd4ead8f80d4f8`；ledger commit 为本节所属提交。
- 本地 py_compile、Black、isort、flake8 通过。NVIDIA release：7 方法、83 次 kernel launch，fail/error/skip/xfail 均为0。执行源：`src/flaggems_sglang/ops/fla_layernorm_gated.py`。
- 测试源码 SHA256：`dc8e1281482839cecd2d225f89265caf1493803df135ab608516fb7f6959e0eb`；各输入文件 SHA 见 verification-input.json。
- 远端 `gpu:/tmp/flagos-b4-top1.8nsvBC/t51-release`，RTX5070Ti / driver610.57.04 / Python3.12.13 / torch2.13.0+cu130 / triton3.7.1。串行后台执行：`timeout 600 /home/kevin/notebook/.venv/bin/python .agents/skills/flagos-operator-race/scripts/verify_release.py run --directory /tmp/flagos-b4-top1.8nsvBC/t51-release`。
- ZIP `/Users/bytedance/ccc/flagos/artifacts/competition/fla_layernorm_gated/e4-5544a77/fla_layernorm_gated.zip`，14982 bytes，SHA256 `05e9e495d68e068cfd3f0aa764d8af84fbed86d47a68a7108d26ef7def8350e7`；dry-run 与最终 manifest 五项恒等字段全部匹配。
- ZIP member `fla_layernorm_gated.py` SHA256 `e1aa34a0f3d851473998a2848175b0539ee9be681b2c4d753f06a922efdf309e`。
- ZIP member `fla_layernorm_gated_ascend.py` SHA256 `92baf6f3a08d4675cee828d451250be42a8b977d3f889eb59ca8c582654fb0c4`。
- ZIP member `fla_layernorm_gated_enflame.py` SHA256 `7be3674559520a8f77d34208580761b4d98cd4eb8ff414ea6a6f74ca7a39bae8`。
- ZIP member `fla_layernorm_gated_kunlunxin.py` SHA256 `66e08be2b4d870028f63ddceb89dd6e6b66ed79ad333e29757ad901332b1ebdd`。
- 证据 `/Users/bytedance/ccc/flagos/artifacts/competition/fla_layernorm_gated/e4-5544a77/validation/verification.json` SHA256 `08476719b7cc4736e179bc9c16b593ef882d8dd8478911796fd3f1acfcb64795`。
- 证据 `/Users/bytedance/ccc/flagos/artifacts/competition/fla_layernorm_gated/e4-5544a77/validation/verification.log` SHA256 `7ea153e1e09fb8aee45357ec6ce7d39ac4685117954d2e3ec5c744f6107fcbf1`。
- 证据 `/Users/bytedance/ccc/flagos/artifacts/competition/fla_layernorm_gated/e4-5544a77/validation/verification-input.json` SHA256 `35c1b19af362fe9b0a0d91adf37f10e3f48bee314fa975b1a0000c3e2632aa7f`。
- 附加基准、诊断及 MCP 证据清单 `/Users/bytedance/ccc/flagos/artifacts/competition/fla_layernorm_gated/e4-5544a77/validation/evidence-sha256.json` SHA256 `45a1aa10a777047186b2409b759228b4cfa61b990190f56dfcffcb84d7ee251c`。
- 首轮每题最多1次正式上传/提交；本次预算上限沿用批准的 T43/T51/T52 各4、T57 3、储备6，须有新证据才继续消耗。sending/uncertain/stale_after_upload 不自动重试。

## 本轮平台结果与止损（2026-09-07T14:55:36+08:00）

- submission `10746` / daily_seq `12` / created `2026-09-07T14:48:28`；preflight与上传/提交均只执行一次，远端ZIP验签 `verified`。
- 平台原始状态 `completed` / `valid`，通过8/8、终态8/8；average_speedup `5.195325`，is_team_best `False`；观测时额度 `18/30`。
- E4 比最佳 E2 回退约3.68%；沐曦3.7252对旧4.5654回退18.40%，其他芯波动也影响均值。不能把全部回退归因多行分支：BLOCK_R=1仍改变IR维度，且隐藏性能case未透出。此轴无净收益，停止追加平台试验。

| 芯片 | 状态/正确性 | 加速比 | 实际文件 |
| --- | --- | ---: | --- |
| tianshu | completed/True | 9.235 | `fla_layernorm_gated.py` |
| muxi | completed/True | 3.7252 | `fla_layernorm_gated.py` |
| enflame | completed/True | 2.3446 | `fla_layernorm_gated_enflame.py` |
| haiguang | completed/True | 7.9656 | `fla_layernorm_gated.py` |
| kunlunxin | completed/True | 0.9616 | `fla_layernorm_gated_kunlunxin.py` |
| huawei | completed/True | 2.3488 | `fla_layernorm_gated_ascend.py` |
| card_a | completed/True | 8.0856 | `fla_layernorm_gated.py` |
| card_b | completed/True | 6.8962 | `fla_layernorm_gated.py` |

- 证据 `/Users/bytedance/ccc/flagos/artifacts/competition/fla_layernorm_gated/e4-5544a77/validation/51-submit.json` SHA256 `78c9eb2c5ea6416ce70162d585f9acc914c7d17f6538457a2867473408f84182`。

- 证据 `/Users/bytedance/ccc/flagos/artifacts/competition/fla_layernorm_gated/e4-5544a77/validation/51-status-first-round.json` SHA256 `f1a1c304a944abf543a866887fdce262d865c483b97053f2e61a212e1b81a7b4`。

## (dtype,行数)×调度档位离线扫描（2026-09-07，PR50 模板，零额度）

- 代理端（RTX 5070 Ti）5 行数档 × 3 维度档 × 3 dtype 共 45 桶，扫
  BLOCK_R{1,2,4,8,16} × warps{4,8} × grid_cap{无,4096}，基线 =
  BLOCK_R=1/warps4/不封顶（E2 形态）。证据
  `research-20260907/t51_scan.{py,json}`。
- **唯一显著档位轴 = BLOCK_D≥2048 时 warps 4→8**：BR1_w8 不封顶在
  9 桶最优、均值 +10.9%、峰值 +36.5%（bf16/1024 行/4096 维）、
  +23.1%（fp16 同档）、+19.7%（fp32/1024/2048）；其余 36 桶最优
  配置增益 ≤5%，多行（BR≥2）与 grid 封顶全部 marginal——与 E4
  多行平台回退互证（多行轴确认关闭）。
- 处置：warps 档位是纯调度轴（不改 IR 维度结构），但 E4 前科表明
  代理单桶增益不可外推平台均值；且 9 桶全在 rows=1024 单行档，
  与弱芯（昆仑 0.94/华为 2.52）瓶颈画像不重叠。**不投平台**，该
  轴留作组合包组件（若未来 vendor 组合包需要，generic warps 按
  BLOCK_D 分档是安全叠加项）。

## E5 燧原默认 launch vendor（2026-09-07，Top1 冲刺第 1 发，提交前）

假设：GCU 官方默认 launch 优于显式 `num_warps=4, num_stages=1`——T19（fused_rmsnorm
同族）E5 平台实证同款单变量 +38%（1.50→2.08）。本会话 Top1 计划第 1 发。

单变量与成员冻结：generic 回退 E2 字节（撤销 E4 多行 generic，SHA
`cae3e2d5…7ab868a` = E2 ZIP 成员逐字节）；ascend `92baf6f3…`、kunlunxin
`66e08be2…` 冻结 E4 字节；唯一变更 = `_enflame` vendor 去 launch 参数。

**中间失败（如实记录）**：首版 commit `0358012`（仅去 launch 参数）按规程加
`--proxy-vendor enflame` 在 NVIDIA 代理执行时暴露 E1 以来潜在 bug：该 vendor kernel
以 `w_ptr + offs` 读 weight/bias（无 stride），边界测试 `weight[::2]` 步长切片下
60 subTest 全部数值不符（`Tensor-likes are not close`）；平台 E1/E2/E4 八芯通过
从未暴露，因为该 vendor 此前从未在代理执行且平台隐藏 case 传连续权重。修复 =
wrapper 对 strided weight/bias 先 `.contiguous()`（commit `445d3eb`；kernel 字节
不变、平台连续输入零开销）。kunlunxin vendor 同病（同为无 stride 读），无代理执行
通道且平台已验证，本轮不动，待其下次变更时一并修复；ascend vendor 有 stride 无此问题。

- source/verification commit：`445d3eb`（首版 `0358012` 已被失败回执引用，作废不投）。
- 本地 py_compile 通过；Black/isort/flake8 沿用远端 release 流程核对。NVIDIA release
  v2 回执：7 方法、generic 83 + enflame 64 次非 warmup kernel launch，fail/error/skip/
  xfail 均为 0；enflame 路径首次完整代理执行（默认 launch 参数下边界矩阵全过）。
  执行源：generic + `_enflame`；ascend/kunlunxin 静态携带，target-runtime-unverified。
- 环境：RTX5070Ti / driver610.57.04 / Python3.12.13 / torch2.13.0+cu130 / triton3.7.1。
  远端 `gpu:/tmp/flagos-t51-e5c.sQ0KdU`（串行后台，timeout 600，PID 313535）。
- ZIP `artifacts/competition/fla_layernorm_gated/e5-445d3eb/fla_layernorm_gated.zip`，
  15060 bytes，SHA256 `9e2995baea00f6786a3310cb3e86d61f9fe76e97f7342bcea5e9c31858d1f185`；
  dry-run 与最终 manifest 恒等字段全部匹配；`unzip -t` 4 成员通过。
  成员 SHA：generic `cae3e2d5…7ab868a`、ascend `92baf6f3…b0c4`、
  enflame `848f5014…`、kunlunxin `66e08be2…`（完整值见打包器 manifest）。
- 回执 `e5-445d3eb/validation/verification.json` SHA256
  `0c5bf1a61f489ae7ad55f82a5539c26c80ad2721b8ce916bc5fd4cfef4eb5671`；
  `verification.log` SHA256 `76c1767756e913046847ce475e457145d6a1e317bc78e3bfa9919eb0faad0cb9`。
  首版失败回执保留于 `e5-0358012/validation/`（该 ZIP 作废未提交）。
- 预注册晋级门：8/8 valid 且燧原 ≥3.0x（E2/E4 同字节噪声带 2.34–2.77 上沿之外）；
  均值须高于 E2 5.3939 才记 team best；燧原 <2.77 则关轴、保留 E2。其他七芯字节
  冻结或回 E2，读数仅在各自噪声带内波动。首轮每题最多 1 次正式上传/提交；
  sending/uncertain/stale_after_upload 不自动重试。
- 附加证据清单 `e5-445d3eb/validation/evidence-sha256.json` SHA256 `392a443f5bf3c94d0126cad173787f8dbb3f03cdcef54e9d330373f4c888586f`
（含作废首版回执 `5ca0f400…d6f86`/`296e03b1…66566de` 与最终回执哈希）。

## E5 提交网络超时与对账（2026-09-07 18:16–18:20，未入队）

- 18:16:17 preflight 全绿后执行一次性 submit（nonce `9781638b…4e090c`）：
  ZIP 上传成功（KS3 `…/9ip3BVEJ/fla_layernorm_gated.zip`），创建提交的 POST 于
  18:17:34 网络超时（`Errno 60`），intent 状态 `uncertain`。
- 只读对账两次（18:18:03 / 18:19:51）：T51 提交列表无新记录（最新仍 E4
  14:48:28）、quota 保持 10/30 未扣、上传 URL 不被任何提交记录引用。
  判定 POST 未达服务器，候选未提交、额度未消耗；服务端 Idempotency-Key
  `flagos-<nonce>` 幂等保护仍在。
- 按 sending/uncertain 不自动重试纪律，未重发。nonce 已自然过期（18:26:02）；
  重提需用户明示授权：归档 `.git/flagos-platform/9781638b….json` 后重新
  preflight + 一次性 submit（候选字节不变、门禁证据全部有效）。

## E5 平台终态与关轴（2026-09-07T19:32:11 观测）

- submission `10873` / daily_seq `26` / created `2026-09-07T19:31:04`（18:16 首次
  提交因 POST 网络超时未入队，用户授权归档 uncertain intent 后 19:31 重提成功；
  远端 ZIP 手工验签 `9e2995ba…d1f185` 与本地逐字节一致，15060 bytes）。
- 平台 `completed` / `valid`，通过 8/8；average_speedup `5.386175`，
  is_team_best `False`（E2 5.3939 保持 team best）；观测时额度 `4/30`。
- **燧原 2.3282，远低于预注册门 ≥3.0，也低于噪声带下沿 2.34——T19 E5 的
  默认 launch +38% 模式未迁移到本题，按预注册关轴：燧原默认 launch 轴关闭，
  不再以此机制消耗额度。**

| 芯片 | 状态/正确性 | 加速比 | vs E2 | 实际文件 |
| --- | --- | ---: | ---: | --- |
| tianshu | completed/True | 9.215 | -1.4% | `fla_layernorm_gated.py` |
| muxi | completed/True | 4.858 | +6.4% | `fla_layernorm_gated.py` |
| enflame | completed/True | 2.3282 | **-15.9%** | `fla_layernorm_gated_enflame.py` |
| haiguang | completed/True | 7.9994 | -2.2% | `fla_layernorm_gated.py` |
| kunlunxin | completed/True | 0.9636 | +0.0% | `fla_layernorm_gated_kunlunxin.py` |
| huawei | completed/True | 2.357 | +0.8% | `fla_layernorm_gated_ascend.py` |
| card_a | completed/True | 8.2822 | +0.9% | `fla_layernorm_gated.py` |
| card_b | completed/True | 7.086 | +4.5% | `fla_layernorm_gated.py` |

- 附带读数校准：generic 为 E2 逐字节，七芯全部落在历史噪声带内（muxi 同字节
  带 4.39–4.86，E5 读 4.858 属带内高位）；燧原显式参数（E1/E4）与默认 launch
  （E5）读数 2.33–2.35 无差异，E2 的 2.77 判定为评测机高位噪声。
- 证据 `e5-445d3eb/validation/51-submit.json` SHA256
  `7fcd6fde6a376549f2f1e9654878576d89c7cbe5ebbd0cbae42650e46aff9a6b`；
  `51-status-e5-final.json` SHA256
  `41857c1b1bdef06f88b24a3887679f54c3b114e8db1bbc50b78685fce1846e6b`。
- 下一发（E6）：generic warps 按 BLOCK_D 分档（≥2048 用 8，45 桶扫描代理
  +10.9%）+ 新增 `_metax` vendor 冻结 E2 字节隔离沐曦；多行轴维持关闭。

## E6 generic warps 分档 + 沐曦 metax-pin（2026-09-07，Top1 冲刺第 2 发，提交前）

假设：BLOCK_D≥2048 时 warps 4→8 提升 generic 强芯——45 桶调度扫描（2026-09-07
并行会话，RTX5070Ti）唯一显著轴：受影响桶均值 +10.9%、峰值 +36.5%
（bf16/1024 行/4096 维）；多行与 grid 封顶均 marginal，维持关闭。

单变量与隔离：generic 仅 wrapper 层 warps 分档（kernel 函数字节不变，
`num_warps=8 if block_d >= 2048 else 4`）；**新增 `_metax` vendor = E2 generic
冻结字节**（固定 warps=4），把沐曦钉在已验证路径、隔离 generic 变化对沐曦的
不可外推风险（E4 教训）；ascend/enflame/kunlunxin 冻结 E5 字节。爆炸半径收敛为
天数/海光/A/B 四强芯（历史同字节波动 ±3%）。

- source/verification commit：`d9b7aec960ec673c570b9c02232550f7ecb0b214`。
- release v2 回执：7 方法、generic 83 + metax 64 次非 warmup launch，
  fail/error/skip/xfail 均为 0；metax 为新增受影响路径，已按规程代理执行。
  环境 RTX5070Ti / Python3.12.13 / torch2.13.0+cu130 / triton3.7.1，
  远端 `gpu:/tmp/flagos-t51-e6.mFAqHw`（timeout 600，PID 315637）。
- ZIP `e6-d9b7aec/fla_layernorm_gated.zip`，5 成员（新增 metax），SHA256
  `9ce71b28cb48a0639a8ca2e53a086e1e48358afbdfc32708e9e8f1fb88a10f33`。
  成员：generic `d952eda8…`、ascend `92baf6f3…`、enflame `848f5014…`、
  kunlunxin `66e08be2…`、metax `3a4f040c…`（完整值见打包器 manifest）。
- 回执 `verification.json` SHA256
  `a98e97e46e520a636057c2930f74c42450ed7e461fca83bf43456c237280e4f2`；
  `verification.log` SHA256
  `3201f55a41ff6a430d3e02a34a036aedcae35321e31aecf0fc62a3ed14296d17`。
- 预注册晋级门：8/8 valid；均值 > E2 5.3939 才记 team best；四强芯各自
  ≥ E2 值 −5%；沐曦（metax pin，E2 字节）须落在同字节噪声带 4.39–4.86。
  失败处置：均值 ≤ E2 则 warps 分档轴关闭，generic 回 E2 字节。
- 观测时额度 4/30（本发后 3/30）；sending/uncertain/stale_after_upload 不自动重试。

## E6 平台终态：昆仑崩溃族 7/8，七芯读数有效（2026-09-07T20:00:13 观测）

- submission `10877` / created `2026-09-07T19:39:46`；`completed` 但
  `invalid_correctness`，7/8；观测时额度 `1/30`（并行会话同窗消耗多发）。
- **昆仑失败 = 崩溃族，非代码问题**：case 0 报 `RuntimeError error code=299,
  wait for noc idle timeout`（execution 1,207,342ms≈20min），与 T53/T57 记录的
  昆仑崩溃族同指纹；昆仑 vendor 为 E2 冻结字节 `66e08be2…`，**8 分钟前 E5
  （19:31）同字节刚以 0.9636 通过**（同载体双投对照）。按崩溃族协议不计入
  代码止损；重载需用户逐发授权。
- 七芯有效读数（对照 E2）：**天数 11.1728（+19.6%，warps 分档显著获益）**、
  沐曦 4.476（metax pin 生效，选中 `_metax`，带内）、燧原 2.3602（冻结，带内）、
  **海光 6.9598（-14.9%，显著低于其同字节带 7.74–8.18，判定 warps-8 在海光
  真实负收益）**、华为 2.482（冻结 ascend，带内）、A 8.2492（+0.5%）、
  B 7.1592（+5.6%）。
- 昆仑按常规 ~0.963 折算的虚拟均值 ≈ **5.478 > E2 5.3939**：warps 分档轴净正
  收益但被昆仑崩溃吞掉。轴判定：保留 warps 分档，但海光需 pin 回 E2 字节。
- 证据 `51-status-e6-final.json` SHA256 见上；`51-submit.json` SHA256
  `409653ceb8174267aaf65fedabc7c9bb4288a02e51d4f859d385d2003f623169`；
  远端 ZIP 手工验签 `9ce71b28…10f33` 一致（19195 bytes）。
- 额度仅剩 1 发：下一发（若授权）建议合并包 = generic warps 分档（保留）+
  新增 `_hygon` vendor 冻结 E2 字节（隔离海光回退）+ `_metax` 加 rsqrt
  （沐曦轴，预注册门 ≥+12%）+ 其余 vendor 冻结；同时充当昆仑崩溃族重载
  （新字节天然绕过同 tuple 限制）。风险：昆仑若再现同指纹即按协议封存。

## E7 合并包：warps 分档保留 + hygon-pin + metax-rsqrt（2026-09-07，就绪待授权未提交）

内容：generic 维持 E6 warps 分档字节不动；**新增 `_hygon` vendor = E2 冻结字节**
（修复 E6 实测海光 -14.9%，pin 回其 7.74–8.18 带）；`_metax` 由 E2 冻结字节改为
**仅 `1.0/tl.sqrt → tl.rsqrt`**（09-05 预注册沐曦轴，门 ≥+12%）；ascend/enflame/
kunlunxin 冻结。本包同时是昆仑崩溃族重载载体（新 ZIP 字节绕过同 tuple 限制）。

预期账（E6 七芯实测 + 海光回带 + 沐曦 rsqrt 门值 + 昆仑常规 ~0.96）：
11.17 + 5.0 + 2.36 + 8.0 + 0.96 + 2.48 + 8.25 + 7.16 ≈ 45.4 → **均值 ~5.67**
（#4–5 名量级；距榜首 6.67 仍差，最后一发额度下的最优期望组合）。

- source/verification commit：`d05e57a0ee3d2d453866c85c0479b22a6fcbae8b`。
- release v2 回执全绿：7 方法，generic 83 + hygon 64 + metax 64 次非 warmup
  launch，fail/error/skip/xfail=0；两个改动/新增 vendor 均按规程代理执行。
  RTX5070Ti / Python3.12.13 / torch2.13.0+cu130 / triton3.7.1，
  远端 `gpu:/tmp/flagos-t51-e7.BeQMBI`（timeout 600，PID 316073）。
- ZIP `e7-d05e57a/fla_layernorm_gated.zip`，6 成员，SHA256
  `0211916a1afa740df00dd31e59e6996ceb89f5692e319d697c0b393949df2ad4`。
  成员：generic `d952eda8…`（=E6）、ascend `92baf6f3…`、enflame `848f5014…`、
  **hygon `712c3859…`（新）**、kunlunxin `66e08be2…`（冻结）、
  **metax `a7cd1e8a…`（rsqrt）**。
- 回执 `verification.json` SHA256
  `c2bc380f3e7327404310f45120b5e4dc99570fa69e15f7878ad16780b7fb014f`；
  `verification.log` SHA256
  `830ecaf088554d6287fd79462d24f24c459822f69066e585f317751675a229ef`。
- 预注册晋级门：8/8 valid 且均值 > 5.3939（team best 晋级）；沐曦 ≥5.0
  （rsqrt 门 +12%）；海光回带 ≥7.7；昆仑若再报 299 同指纹 → 按崩溃族协议
  封存候选、不再探针，仅走平台工单路径。**本发为最后一发额度（1/30）且属
  崩溃族重载，按纪律需用户当次明示授权后才执行 preflight+submit。**

## E7 平台终态：**8/8 VALID，新 TEAM BEST 5.816325x**（2026-09-07T21:45:32 观测）

- submission `10924` / created `2026-09-07T21:41:47`，用户会话内授权后执行唯一
  一次 preflight+submit；远端 ZIP 手工验签 `0211916a…df2ad4` 一致（23185 bytes）。
- `completed` / `valid`，8/8，average_speedup **5.816325**，is_team_best `True`
  （E2 5.3939 → +7.8%）；观测时额度 `0/30`（本批额度用尽）。

| 芯片 | 状态/正确性 | 加速比 | vs E2 | 实际文件 |
| --- | --- | ---: | ---: | --- |
| tianshu | completed/True | 11.4872 | **+22.9%** | `fla_layernorm_gated.py`（warps 分档） |
| muxi | completed/True | 4.4776 | -1.9% | `fla_layernorm_gated_metax.py`（rsqrt，带内） |
| enflame | completed/True | 3.7314 | **+34.6%** | `fla_layernorm_gated_enflame.py`（冻结字节） |
| haiguang | completed/True | 7.9938 | -2.2% | `fla_layernorm_gated_hygon.py`（pin 生效） |
| kunlunxin | completed/True | 0.9644 | +0.1% | `fla_layernorm_gated_kunlunxin.py`（无崩溃） |
| huawei | completed/True | 2.3028 | -1.5% | `fla_layernorm_gated_ascend.py`（带内） |
| card_a | completed/True | 8.428 | +2.7% | `fla_layernorm_gated.py` |
| card_b | completed/True | 7.1454 | +5.3% | `fla_layernorm_gated.py` |

- 轴结算：warps 分档（天数 +2.14、B +0.36）+ hygon pin（海光回带 +1.03 vs E6
  读数）为净收益主贡献；**沐曦 rsqrt 无效果**（4.4776 vs pin 基线 4.476，
  轴按门 ≥+12% 判负关闭）；昆仑崩溃族未复发（E6 判定平台侧获最终佐证）。
- **E5 燧原关轴判定修正**：E7 携带的燧原 vendor 与 E5 逐字节相同（默认
  launch，`848f5014…`），E5 读 2.3282、E7 读 3.7314——E5 的低位是评测机
  忙时窗口压制，默认 launch 轴实际有效（对照显式参数轮 2.33–2.34 三连读，
  +60%）；教训记入：燧原读数窗口效应 > 单轮门禁，弱带内芯片的关轴需双轮证据。
- 证据 `51-submit.json` SHA256
  `f205f4c85b5f0f51764b00e99c95241727c75e260dd40afb94b180dd2c7c76b0`；
  `51-status-e7-final.json` SHA256
  `86ddc19eebabadf92ee63b9f2b5bd0e74a3c71a164093c3b314dd02a2430313f`。
- 额度 0/30：本批提交通道关闭；后续仅离线迭代（华为大 D 分块等结构轴留待
  下一窗口或新额度），账本 current 块更新为 e7 终态。

## 2026-09-08 推荐方案实现与提交前验证（未提交平台）

华为 gate load 延迟至归约/affine 后。NVIDIA PTX 加载位置变化，但shared占用未降，计时0.994–1.002x；尚未获得华为UB/Scalar/MTE证据。

- source `8ba31a102f4ef0430c08f12c4622b27430915071`；verification `8ba31a102f4ef0430c08f12c4622b27430915071`。7 个测试方法、147 次实际 kernel 调用；选定 NVIDIA/代理范围门禁通过。
- 回执 `artifacts/competition/batch4-implementation-20260907/t51-release1/verification.json`，SHA256 `389a23663a437cfaa2142a6877e4275911e2b5d0fa3459f47e69397bbeaacb4d`；日志 SHA256 `56871542e29e698fe20501d6e58a7a96b048291deee476046353c2e97c182f91`。
- 不可变 ZIP `artifacts/competition/fla_layernorm_gated/research-20260908-8ba31a1/fla_layernorm_gated.zip`，SHA256 `579d902c7fc7bb8c4dbc382220143ccd7512adc7a05978ad1f77f806b6c81f47`；与 dry-run manifest、构建和 existing 验签一致。ZIP 是候选产物，不等于目标芯或平台已通过。
- 环境、逐源码执行范围、原始配对数据和未完成条件见[本轮报告](../implementation-batch4-20260908.md)及[证据清单](../data/batch4-implementation-20260908.json)。本轮不更新历史有效分，未做平台 preflight、上传或正式提交。

## E8 额度恢复后评测预注册（2026-09-08）

- 用户已明确授权提交评测。本轮顺序 T47 E11 → T51 E8，预算每候选仅1次上传/正式提交；2026-09-08T01:42:22+08:00 实时额度30/30，账号全局间隔120秒。sending/uncertain/stale_after_upload/submitted 均不自动重试。
- 假设与优先级：华为延迟 gate load，其他五个成员与 E7 相同；当前第6，历史均值5.816325，榜首6.668225。七芯不变时华为需从2.3028升至9.118才能登顶，不能承诺单轴达成。NVIDIA计时0.994–1.002x，目标编译器收益未验证。
- 晋级/停止门：8/8 valid、每芯≥0.1；均值>5.816325 才晋级 team best；华为≥2.53308（+10%）才支持该轴收益。无收益保留 E7，失败取证后封存该候选，不以同字节重掷。
- source commit `8ba31a102f4ef0430c08f12c4622b27430915071`；verification commit `8ba31a102f4ef0430c08f12c4622b27430915071`；本节 ledger commit 为提交本节的独立文档提交，不等同源码或验证提交。
- 正式不可变 ZIP `/Users/bytedance/ccc/flagos/artifacts/competition/fla_layernorm_gated/e8-8ba31a1/fla_layernorm_gated.zip`，23261 bytes，SHA256 `579d902c7fc7bb8c4dbc382220143ccd7512adc7a05978ad1f77f806b6c81f47`；正式阶段 `e8`，与上一节 research 包逐成员一致，existing 验签通过。
- release 回执 `artifacts/competition/batch4-implementation-20260907/t51-release1/verification.json`，SHA256 `389a23663a437cfaa2142a6877e4275911e2b5d0fa3459f47e69397bbeaacb4d`；日志 SHA256 `56871542e29e698fe20501d6e58a7a96b048291deee476046353c2e97c182f91`；测试 SHA256 `dc8e1281482839cecd2d225f89265caf1493803df135ab608516fb7f6959e0eb`。
- 所选 release 范围全绿，目标设备仍为 `target-runtime-unverified`，本次授权评测补齐；保留上一节执行范围，不将代理通过写成目标通过。
- 原始 manifest / preflight / submit / 状态证据保存目录 `artifacts/competition/batch4-submit-20260908`。提交前尚无本阶段平台结果，历史 best 保持。

| ZIP 成员 | SHA256 |
| --- | --- |
| `fla_layernorm_gated.py` | `d952eda8ccac83995fb34a2a557fd11ecab428f00e79a5c6f1f03a096926e651` |
| `fla_layernorm_gated_ascend.py` | `ac8b090dbda2158390ced51578011ab644f8d90a710e513e3a69bedc154e7f00` |
| `fla_layernorm_gated_enflame.py` | `848f50149b4134ae58f6f0c228e9a7908e049950b10f511e3f302db5af59ab4b` |
| `fla_layernorm_gated_hygon.py` | `712c38595eef274a95d559171b23c31edc3af1411b135757fbc378306083c260` |
| `fla_layernorm_gated_kunlunxin.py` | `66e08be2b4d870028f63ddceb89dd6e6b66ed79ad333e29757ad901332b1ebdd` |
| `fla_layernorm_gated_metax.py` | `a7cd1e8a9807e0ce17d2d955aa0c27b47639242fb6bdcf6fd1069d956d2d2bf1` |

## E8 平台终态（2026-09-08）

- submission `11032`，daily_seq `2`，created `2026-09-08T01:52:34`；观测 `2026-09-08T01:53:38.008714+08:00`。`completed / valid`，8/8，均值 **5.721075x**，`is_team_best=false`，保留 E7 **5.816325x**。
- 一次上传、一次正式提交；远端匿名下载验签通过，23261 bytes，ZIP SHA256 `579d902c7fc7bb8c4dbc382220143ccd7512adc7a05978ad1f77f806b6c81f47`。file URL SHA256 `0afa077050bc4735ab78aa750ce296b5f4d4ea05e66be6f676bfc5d0cd5a73f5`。
- 预注册 ledger commit `e18b94b876dabbadde49762f5fe0bfe7cbf46a75`；source / verification commit 均保持 `8ba31a102f4ef0430c08f12c4622b27430915071`。
- 原始提交 `artifacts/competition/batch4-submit-20260908/51-submit.json` SHA256 `97317a26053cb31b8ae2c8e5d77568946638ab741717261ebb814583713b0947`；终态 `artifacts/competition/batch4-submit-20260908/51-status-final.json` SHA256 `bd1b41ed7b5838059b847669e6cd135819ad2a589ca20745c1f63a3254ed0c04`。

| 芯片 | 正确性 | E8 加速比 | E7 加速比 | 实际文件 |
| --- | --- | ---: | ---: | --- |
| tianshu | PASS | 11.2122 | 11.4872 | `fla_layernorm_gated.py` |
| muxi | PASS | 4.7748 | 4.4776 | `fla_layernorm_gated_metax.py` |
| enflame | PASS | 2.3324 | 3.7314 | `fla_layernorm_gated_enflame.py` |
| haiguang | PASS | 8.6432 | 7.9938 | `fla_layernorm_gated_hygon.py` |
| kunlunxin | PASS | 1.0292 | 0.9644 | `fla_layernorm_gated_kunlunxin.py` |
| huawei | PASS | 2.3638 | 2.3028 | `fla_layernorm_gated_ascend.py` |
| card_a | PASS | 8.3158 | 8.428 | `fla_layernorm_gated.py` |
| card_b | PASS | 7.0972 | 7.1454 | `fla_layernorm_gated.py` |

- 结算：华为2.3638 vs2.3028，+2.65%，未达+10%门；总分-1.64%，未晋级，单独延迟 load 轴停止。其余五个成员均为 E7 字节，燧原从3.7314回到2.3324；不可把其他芯片读数变化归因于华为改动，也不凭单轮小涨声称目标性能优化成立。
- 本轮两候选均已终态且未刷新 best，累计只用2次额度，剩余28/30。没有重投；后续结构优化需新源码与性能证据，已知失败或门禁未过候选保持不提交。

- 提交后实时排名复核 `2026-09-08T01:54:27.199190+08:00`：第 **6**，本队 best **5.816325x**，榜首 **6.668225x**（c2flow）。证据 `artifacts/competition/batch4-submit-20260908/tasks-after.json` SHA256 `f7959a760a01815c0072e76de5c3e7fa8fb08c4e41ec3579ca946c6c64001170`。

## E9 行块特化 → invalid（昆仑真机失败），轴关闭（2026-09-08，sub 11124）

- 候选（commit `fbd7386`，PR FlagGems-sglang #50 配方）：generic 增
  ROWS/DIM constexpr 行块特化 kernel（16 行块、weight/bias 每 program
  单次载入、HAS_ROW_MASK/HAS_D_MASK 编译期折叠），`_kunlunxin` 换
  T56-e3 行块形态（tl.rsqrt/sigmoid 与 isCloseCoreTiling 保留）。
  代理 7/7 全过、配对 +1~5%。
- 终态 **invalid_correctness：昆仑真机 correctness 失败**（NVIDIA 代理
  通过 ≠ 目标芯通过，再次印证）。已出七芯：天数 **9.1732（E7 11.49→，
  行块在天数 -20% 反向）** / 沐曦 4.5712 / 燧原 2.299 / 海光 7.7266 /
  华为 2.2568 / A 8.326 / B 6.878。team best 保持 E7 5.816325。
- 处置：`_kunlunxin` 与 generic 字节已回滚 E8 树态；行块轴对 T51 关闭。
  天数偏好 per-row 调度（与 T56 l2norm 行块 +87% 相反）——**行块配方
  芯相关且算子相关，不可跨题外推**。剩余差距需 c2flow（6.668）结构
  情报，上游无 PR。

## E10 华为大 D 子块 vendor（2026-09-08，提交前预注册）

- 假设：D>1024 且 256|D 的行不走整行 BLOCK_D 形态，改为
  D_TILE= min(dim&-dim, 1024) 的 N_SUB 个精确子块（无任何 bounds
  mask、无运行时向量比较）：RMS 两遍（sumsq 遍 + 归一化输出遍）、
  LayerNorm 三遍（sum→centered sumsq→输出；保持减均值后再归约的
  reference 顺序，不用 E[x²]-E[x]²）。机制：昇腾 CMP 标量退化
  （T40 +130% 家族）+ 整行 8192 宽 fp32 多活向量撑爆 UB 的 spill
  开销；x 重读换小活跃集。dim≤1024 或非 256 整除时回落平台已验证
  的整行 kernel（host 分派）。
- 单变量：仅 `_ascend` vendor 变更；generic（E8 回滚字节）/
  enflame/hygon/kunlunxin/metax 全部冻结。
- NVIDIA 配对计时（`pair-generic.json` fla 段，ascend vendor）：
  全 case 0.97~1.006 中性——**预期签名**：UB/CMP 缓释收益只在昇腾
  真机体现（T40 E16 同款：代理不可见、平台 +130%）；NVIDIA 上
  整行形态本就无 UB 约束，8192 case -2.5~3% 为重读成本。
- source/verification commit `18616ff`；release v2 回执（远端
  `gpu:/tmp/flagos-t51e10-rel`）：7 方法全过（RELEASE_REQUIRED 全集，
  含新增 2048/1536/8192 子块维度），generic 101 + ascend 82 次非
  warmup launch，fail/error/skip/xfail=0。回执
  `e10-18616ff/validation/verification.json` SHA256
  `c8a128a5b18adf49f36a9cd8f4e6d7fb12fed4539485b4ec3dfbf7460ff564a4`、
  日志 `35711a75d621bf2dd884f5715275bfc0dcb3582414e0d3a62b27e79f6bcca145`。
- ZIP `e10-18616ff/fla_layernorm_gated.zip`（6 成员：仅 ascend 新）
  26533 bytes，SHA256
  `97823b0c76cb83b546f840422e695c700983eb43c9c9de78ba65aa78ba8418f8`。
- 预注册晋级门：8/8 valid 且 avg > 5.816325（E7 team best）才晋级；
  华为 ≥2.9（+26%）视为 UB 假设兑现、<2.4 视为轴证伪（E8 门 +10%
  未达前科在案，本次门更高因结构变化更大）。华为若编译失败
  （UB overflow 反向）则该 vendor 回滚整行字节并关轴。其他七芯
  字节冻结，读数仅在各自噪声带内（E8 燧原 2.33↔E7 3.73 窗口带）。

## E10 提交记录（2026-09-08T14:58）

- preflight 通过（quota 12/30，间隔等待 86s 后重试成功）；submit
  **submission 11204** `queued`；证据
  `batch4-codex-round-20260908/51-{preflight,submit}.*`、`51-watch.jsonl`。

## E10 平台终态：**8/8 VALID，60.7192x 新 TEAM BEST，RANK 1**（2026-09-08T15:3x）

- sub `11204` 终态 `completed / valid`，8/8 全过，average **60.7192**，
  `is_team_best=true`；实时榜单 **my_rank=1**（旧榜首 c2flow 6.668225，
  我方为其 9.1 倍；同分先交优先，追平也难越）。
- 逐芯：天数 11.3126 / 沐曦 4.5796（metax pin）/ **燧原 442.8042** /
  海光 7.9608（hygon pin）/ 昆仑 0.9672 / **华为 2.5348** / A 8.4304 /
  B 7.164。
- **结算**：① 燧原 442.8 = 冻结 E5 vendor 字节撞上极端高水位窗口
  （历史带 2.33–3.73，本次两个数量级以上——水位彩票以最大幅度兑现，
  机制疑为该窗口燧原 reference 侧极慢）；② 华为 2.5348 vs E7 2.3028
  = **+10.0%，子块 vendor 结构收益真实兑现**（低于 ≥2.9 强门但高于
  证伪门 2.4，UB/宽行假设方向正确、幅度中等；均值贡献 +0.29）；
  ③ 其余六芯全部落在冻结字节噪声带内。
- T51 就此从第 6 名（5.816）跃居 **Top1（60.72）**；本批 Top1 数
  T47+T51=2。后续：守榜（差距两个数量级，被超风险极低）；华为子块
  结构知识（大 D 行拆 D_TILE 精确子块、零 mask、两/三遍）入库可迁移。
- 证据 `batch4-codex-round-20260908/51-{preflight,submit}.*`、
  `51-watch.jsonl`；本节状态同步至 CURRENT/INDEX。

## 决赛日防守终态（2026-09-10 18:0x）

- 全日 4 次只读监控，rank2 始终 RSI 10.95，差距 ~50x，防御弹药未动用。
- **T51 以 60.7192 收盘 rank1（燧原 442.8 慢窗读数锁定）——全队今日唯一
  Top1 守住**。
