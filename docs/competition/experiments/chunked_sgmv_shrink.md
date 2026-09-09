# Task 48 `chunked_sgmv_shrink` 实验记录

```current
task: 48
operator: chunked_sgmv_shrink
batch: 4
validity: valid
platform: E9/11775八芯valid4.8135625，新团队最佳
team_best_stage: e9
team_best_commit: cea2a0c10878b39c251a36857d97311d1ab4cd73
team_best_speedup: 4.8135625
sealed: no
next: 保留E9同adapter合并；本轮两弱收益兑现，等待新目标瓶颈证据
updated: 2026-09-09
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

## E8 FP32 窄输出 Split-K（2026-09-09，提交预注册）

- 明确形状分支：仅 FP32、K>=1024、N<=128、输出 tile 数<128 时四路 K 并行；分界按32对齐，FP32 workspace 后按 part 顺序归约。FP32窄 N 用16/32行 tile；FP16/BF16 保留原 tile 和单路点积。保留长段多 tile、负 adapter 跳过、部分覆盖输出零。
- source/verification commit：`179fe7ccff681d4f99e4c4f31f0753aea6b6a7ad`，核心代码提交 `2bbdb4b`；ledger commit 为本节独立文档提交。沐曦 cpasync 原型没有固定目标执行证据，已从本隔离分支候选排除；历史源仍在 `01d736b`，本地主分支未改。
- 门：八芯正确且每芯>=0.1；整题 avg>4.7489375 才晋级团队最佳；>=15% 视为有意义结构收益。一次提交后先分析 raw_result，不以新注释/新ZIP重投相同计算。
- NVIDIA RTX5070Ti 最终 release 7 方法通过，全部3个打包成员实际入口和 kernel 均被测试，0 fail/error/skip/xfail。新增1023/1024/1025/4095/4096/4097 K边界、3dtype、部分段覆盖。
- 六轮 AB/BA、每次30次完整调用对 `01d736b` generic：FP32速度约1.14–4.36倍；BF16约0.91–0.97倍（增加参数/分支有约3%–9%开销，平台实测决定取舍）。原始配对在 validation/perf.json，不作为目标八芯证据。
- 最终验证远端 `/tmp/flagos-t48-e8-final.pBUxSd`，PID333442，timeout600；运行 verify_release 后才运行 bench。未同时运行其他GPU基准。
- validation/verification.json SHA256：`f49da1ad13ebfac68d6945962bc257a46c1b5d54ce11abade60ceefc44cf9c04`。
- validation/verification.log SHA256：`166f95440f85d93f6fdd516276526c3d6e0c5c449bd45a552ac756c3200c9fa7`。
- validation/bench.py SHA256：`9147fa85386d7ef961782bceaa085fb16b47830331787bd76c0ed3ffe4be13ab`。
- validation/perf.json SHA256：`737d86b998f9f3cdf2e5e8b5dbd5959eb0d51bec8ed0803bcb44fce22f499612`。
- test SHA256：`0a44058d12400e8d802b6f707053556dd0427ad9d1b1f897ae5179b0c91aaad9`。
- ZIP：`/private/tmp/flagos-batch4-structural-20260909/artifacts/competition/chunked_sgmv_shrink/e8-179fe7c/chunked_sgmv_shrink.zip`；15784 bytes；SHA256 `8f2323df05c07561a4fae6def68e40d7d5493b089c105364cedffbc378de2120`；dry-run/build/verify-existing 一致。

|成员|SHA256|
|---|---|
|chunked_sgmv_shrink.py|`55ddc5fdad8df3c1109b85c2ff50871eea39d713095faeba90a1748ddf684c0e`|
|chunked_sgmv_shrink_enflame.py|`a2d53ce449daf52df5fddec49305350b654d79a2bbb9320d96048b121a46dce4`|
|chunked_sgmv_shrink_kunlunxin.py|`af7a413ef50feb2f76d936feb3155e5d2b237f325c06748e0c02758d41d0112b`|

- E8 已于2026-09-09 12:22:36提交 `11767`，当日第20次，上传后远端SHA256一致；nonce已消费，不得重试。完整响应在 `e8-179fe7c/submit.json`。

### E8 八芯终态（2026-09-09 12:23 CST）

`11767` 为8/8 valid，均值 **4.460625**，较团队最佳4.7489375低6.1%，未晋级。天数3.728、沐曦4.972、燧原0.326、海光5.8125、昆仑1.752、华为5.5255、A6.7085、B6.8605。新 FP32 子域的代理收益未转化为整题收益；也存在未动 vendor 的水位差，无法仅凭均值精确归因。原始响应 `e8-179fe7c/raw-status-1.json`。停止该候选，不消耗额度重投。

## E9 同adapter段合并（2026-09-09，提交预注册）

- 不改变GEMM及其launcher：读取既有host段元数据，按weight index把多个段组成同一行列表，每个adapter一次gather、GEMM、scatter。只有一个段时沿用原行视图，不做cat；重复adapter仅合并行，权重、scale和slice语义不变，输出行独立。空段先跳过再读取哨兵adapter，保留既有rank0/负adapter规则。
- 两弱vendor增加路由合并；generic恢复`01d736b`已通过平台的基线，上一结构候选未晋级，不继续混入。kernel/launcher AST与`f9cb247`逐函数相同，新增互相穿插的重复adapter、int32 permutation、空段哨兵、3dtype测试。
- 预注册：八芯正确且每芯>=0.1；均值>4.7489375才晋级；两弱合计提高至少15%视为合并有效。只提交一次；若平台没有重复adapter收益则关轴。
- source/verification commit：`cea2a0c10878b39c251a36857d97311d1ab4cd73`；ledger commit 为本节独立文档提交。NVIDIA RTX5070Ti release 8方法，0fail/error/skip/xfail，3打包成员均实际执行。远端`/tmp/flagos-coalesce-release.6I3BMw/t48`，PID333966，timeout900，T47和T48按顺序验证/计时。
- 完整enflame wrapper六轮AB/BA、每轮10次，基线是`f9cb247`原逐段vendor。重复adapter代理速度范围2.66–12.16倍；各段adapter独立对照1.00–1.00倍。未获目标同源计时，平台负责补齐；不把代理倍率记作目标速度。
- validation/verification.json SHA256 `d76d2bcec99bd2279610b0c0077cb39879dbf5ef37c63fb26e74f290246a3f74`。
- validation/verification.log SHA256 `bbc366152c3a4e3d36274f838b0da1cb01e06b956b17aecb81b2d29aaf8159d7`。
- validation/perf.json SHA256 `8c0eb96c9a5d4ddc4004595df918222b1506adf1cc854696e3b45856655903a6`。
- validation/bench.py SHA256 `2a201fbe4a46651effae808a90dba83a6990baa9c6d6f6339f4f4a9f4aec6498`。
- validation/baseline.py SHA256 `a2d53ce449daf52df5fddec49305350b654d79a2bbb9320d96048b121a46dce4`。
- test SHA256 `9a865241e27b637b967c8ee609ca54d2276fb24d8ae6e77774b2b0c817d17540`。
- ZIP `/private/tmp/flagos-batch4-structural-20260909/artifacts/competition/chunked_sgmv_shrink/e9-cea2a0c/chunked_sgmv_shrink.zip`；15144 bytes；SHA256 `2696c03c0a287c5409dc58fb49a41c129554bdaa0231d374ed4d384e1ae10743`；dry-run/build/verify-existing一致。

|成员|SHA256|
|---|---|
|chunked_sgmv_shrink.py|`f8cf4e66072448689f4a57e25120428371c821536d048754d74c8bfcda22b780`|
|chunked_sgmv_shrink_enflame.py|`1a05db5c432e55181d40dde4f10cd81f8c1470ccddee505093774adb65540352`|
|chunked_sgmv_shrink_kunlunxin.py|`2a9e8182734f4936cdab4f04e61b7706792355c84973c54e0c7b77dadb9c08e2`|

### E9 终态：同adapter合并兑现（2026-09-09 12:43 CST）

- `11775` 于12:41:15正式提交，当日第23次；一次上传/提交，远端ZIP SHA256一致。8/8 valid，**4.8135625新团队最佳**，相对E7的4.7489375约+1.36%。
- 天数3.828、沐曦5.3525、**燧原0.897、昆仑2.424**、海光6.306、华为5.893、A7.008、B6.8。两弱合计3.321，相对E7的2.313增加43.58%，超过15%结构门；总体仍包含未变芯水位，不把全部分差归因于代码。
- 同源响应 `e9-cea2a0c/submit.json`、`raw-status-1.json`；保留E9，Split-K E8不再混入。
