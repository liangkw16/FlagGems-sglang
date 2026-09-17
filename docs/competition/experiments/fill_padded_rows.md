# Task 67 `fill_padded_rows` 实验记录

```current
task: 67
operator: fill_padded_rows
batch: 5
validity: valid
platform: completed(16579,e8,8/8,4.43975x新TB;华为+30.5%)
candidate_stage: e8
team_best_stage: e8
team_best_speedup: 4.43975
sealed: no
next: 用户要求不再新提交；E7已过发布门但预检被间隔拦下，无intent/上传/提交，保留验签包
updated: 2026-09-17
```

## 契约与范围

- 完整题面：[Task 67](../tasks/batch-5/67-fill_padded_rows.md)（2026-09-11 平台新增六题之一）。
- 接口 `fill_padded_rows(x, num_token_non_padded, fill_value)`；exact；
  返回**新张量**（reference 为 `x.clone()` 后填充），`num_token_non_padded`
  为 device 单元素整数张量，须在 kernel 内读取、静态 grid（每行一个
  program），保持 CUDA-graph 可捕获语义。
- 核心计算 Triton；无 fallback。八芯每芯 0.1x；截止 2026-09-17 19:59:59。

## 实现（S0）

- 上游：SGLang 8014d9d `kernels/ops/moe/fill_padded_rows.py`（上游为原地
  版本）；本仓改为 out-of-place：wrapper `out = x.clone()` 后单 kernel
  填充 `row >= n` 的行。`n` 由 `tl.load` 在 kernel 内读取，无 `.item()`
  同步，grid `(rows,)` 静态。
- `BLOCK_COLS = next_power_of_2(n_cols)` 单块覆盖整行；比较与寻址走 int64。

## 不可变身份

- source / verification commit：`b4727f1`（2026-09-11）。
- source SHA-256：`92caf5e002dd60f5c4ce80c523159fd150a79ea87a44e15bfdc46312f63196df`。
- test SHA-256：`07929937c007383c78d597842ed66df875eedc452e1c1d64431d6e465288fe6e`。
- ZIP：`artifacts/competition/fill_padded_rows/s0-b4727f1/fill_padded_rows.zip`。
- ZIP SHA-256：`9319d9f06783f314706b635067bf456521a22007858f73a23658bba869c78a27`。

## 验证状态

- py_compile、black/isort/flake8 全部通过（本地）。
- 测试：4 方法 / 覆盖 dtype×fill 值、n 边界（0/部分/rows/越界）、列宽
  1~7168、行距 stride、int32/int64 计数、空输入与 0 列；断言输入不变、
  返回新张量。
- **远端 GPU（192.168.5.204 / gpu-et）2026-09-11 晚全程不可达**（物理口与
  EasyTier 均超时，ARP 无表项）；release 回执待补，目标芯与代理执行均为
  `target-runtime-unverified`。KernelGen MCP 本会话不可用。

## 风险

- 结构极简（clone + 条件填充），跨芯风险最低；唯一注意点是
  `next_power_of_2(n_cols)` 在超宽行时的寄存器压力（生产场景 n_cols 为
  hidden 量级，可控）。

## 优化方向（按把握）

1. S0 直投（预期显著快于 reference 的 clone+index_put 链）。
2. E1 候选（未开发）：若平台形状列数巨大，改 grid-stride 列循环降低
   BLOCK_COLS；先取平台逐芯数据再决定。

## 2026-09-12 平台结果（submission 13300，daily_seq 5）

- **8/8 valid，均值 3.43285x**（首发即过）。逐芯：天数 9.7498 / 沐曦
  2.2450 / 燧原 1.3660 / 海光 4.8460 / 昆仑 0.5952 / 华为 1.3910 /
  A 3.9498 / B 3.3200。
- 回执（b4727f1，RTX 5070 Ti 代理）：4 方法 0 失败、25 次真实 launch、
  14 组非空 shape；`artifacts/competition/batch5-ext6-validate-20260912/fill_padded_rows/`
  （verification.json SHA-256 `ee29bb4668e9d52fcebf8587d1651242d1da0dfc77f76b21b562e005e0e8987c`）。

## 2026-09-12 E1：单写融合（候选就绪后提交）

- 结构改写（单变量）：去 wrapper `x.clone()`（读 N + 写 N，pad 行随后被
  第二次覆写），改 `torch.empty` 出参 + 单 kernel **每元素只写一次**
  （`row < n` 拷贝、否则填充，运行时标量分支 = T64 已证形态）。省
  pad 行读 + pad 行双写 + 一次 launch。
- 跨芯纪律：掩码地址用算术钳位 `cols * mask`（规避燧原无先例的整型
  `tl.where` 与昇腾 masked-lane 越界地址求值 507035 族）；拷贝路径的
  masked 向量 load 为全仓 8/8 已证形态。
- source commit：`ff5c4aba4f06af8985d1b0d07fb95d8247058675`。
- ZIP：`artifacts/competition/fill_padded_rows/e1-ff5c4ab/fill_padded_rows.zip`，
  SHA-256 `96effba431c6e4037376968dfe8859269972eb7a0c8b42f7c7ad11dede4e8b6b`；
  单成员 `fill_padded_rows.py` `59e21e9c…`。
- release 回执（v2，绑定 ff5c4ab）：
  `artifacts/competition/batch5-t67e1-validate-20260912/fill_padded_rows/verification.json`，
  SHA-256 `302c3520b6a0917e9c971193555e4184705b66e3e14fc587495daada70280bde`；
  日志 SHA-256 `3f71ba92598f6335e39151d1343b41169a68bed0d0ca5f3e419579fe59d1e20c`；
  4 方法 0 失败，25 次 launch。
- 预注册：正确性 8/8 保持；均值目标 > S0 3.4329（pad 占比大时结构收益
  接近减半流量；榜首 8.3163 的结构推断即单写形态）。

## 2026-09-12 E1 平台提交（submission 13364）

- 上传与正式 POST 各一次；state submitted。file_url SHA-256：
  `63eee149c7b9b342…`（完整值见 status 快照）。额度：发后 14/30。

## 2026-09-12 E1 平台中间判决与 E2 修复

- E1（13364）六芯已过且**全面快于 S0**：沐曦 2.6518（+18%）/ 海光
  6.4892（+34%）/ 昆仑 0.5930（持平）/ 华为 1.6868（+21%）/ A 5.4350
  （+38%）/ B 4.2746（+29%）；天数回调中。**燧原 PassManager 失败**
  （exec 6523ms，vendor=generic 被选中）。
- 根因定位：E1 相对 S0（燧原已过 1.3660）的独新构造 = **运行时标量分支
  内嵌 masked 向量 load**。本仓燧原已证形态里 masked 向量 load 全部在
  顶层（deepep_permute），分支内只有 store（fill S0/deepep_permute）。
- E2（单变量）：copy load 提到分支外，行有效性并入 load mask
  （`mask & (row < n_valid)`）；去掉算术钳位（裸 masked load 尾部小越界
  在昇腾已证可过——T64 huawei 3.76x；且 i64 乘法钳位本身是第二未证构造）。
- source commit：`a9c06b9f5e0bca8df598ad07e6e95d9cd7662962`。
- ZIP：`e2-a9c06b9`，SHA-256 `4d4d356a20054af2049a064d0b32c5008196daa11e55b7ca08d4216dca5ed064`，
  单成员 `ac9097fd…`。
- release 回执：`batch5-t67e2-validate-20260912/fill_padded_rows/verification.json`，
  SHA-256 `7d93c4a6d18c84d735b7f8636a3daaca1e0871a3def93779f299a75aa1337f15`；
  日志 `5a97ffd409160759d5196902a18bba228921e9ee2a2292d517091419254e27c6`；
  4 方法 0 失败，25 launch。

## 2026-09-12 E2 平台提交（submission 13367）

- 上传与正式 POST 各一次；state submitted。额度：发后 13/30。
- 裁决点：燧原分支内 load 毒点假设（编译）；其余七芯应保持 e1 的
  +18~38% 水位。

## 2026-09-12 E2 平台终态：8/8 VALID（submission 13367，daily_seq 17）

- **八芯全过，均值 4.2969x = 新 team best（S0 3.4329，+25%）**：
  天数 11.8956 / 沐曦 2.7324 / 燧原 **1.5106（分支内 load 毒点假设证实，
  S0 1.3660 → +11%）** / 海光 6.1164 / 昆仑 0.5854 / 华为 1.9286 /
  A 5.4502 / B 4.1560。
- 沉淀（GCU 规则集补充）：**运行时标量分支内嵌 masked 向量 load = 燧原
  PassManager 编译毒点**；load 提到顶层（行有效性并入 mask）即解除。
  可迁移：燧原 vendor 的 load 一律顶层、分支只包 store。
- 对榜首（EvokeAgent 8.3163）仍差 3.9；后续轴：天数/华为 exec 偏长
  （84s/30s）提示平台 shape 大，launch/tile 仍有空间。

## 2026-09-12 E3：列分块 grid（冲榜轴，候选就绪后提交）

- 逐芯情报（14:1x）：榜首 EvokeAgent/#2 zhaxi123 在宽 shape 芯全面
  3-4x 领先——天数 18.4/22.3 vs 我 11.9、华为 17.5/7.0 vs 我 1.93、
  燧原 5.7/5.6 vs 我 1.51；窄 shape 持平。结构性判读：e2 整行单
  program（最多 8192 lane）串行+寄存器重，天数 exec 84s 佐证。
- E3：grid 改 (rows, col_tiles)，BLOCK=min(next_pow2(n_cols),1024)，
  每 program ≤1024 lane；load 保持分支外（燧原规则）。
- source commit：`e27a0426574b65cabee1e45906eb07412f4b339a`。
- ZIP：`e3-e27a042`，SHA-256 `956b6b4751e10877ddebd1e749a504e2abdc24f6f7bebf436f95ce1a08aeed64`，
  单成员 `6eac741c…`。
- release 回执：`batch5-t67e3-validate-20260912/fill_padded_rows/verification.json`，
  SHA-256 `a7e0e26196fc57da173a63e8d0df2ea8e930a3a08adde6b6a38a54d8cd003fbe`；
  日志 `7e4e132778103c37d9fe48da8fef36b1eaf3def2ff040606136515b24562ca52`；
  4 方法 0 失败，25 launch。
- 预注册：天数/华为/燧原中位 ≥1.3x；其余五芯无回退 >5%。

## 2026-09-12 E3 平台终态：8/8 valid 但未过门（submission 13386）

- 八芯全过：天数 11.3554（-4.5%）/ 沐曦 **3.3884（+24%）** / 燧原
  1.5014（0%）/ 海光 5.9458 / 昆仑 0.6020 / 华为 1.6030（-17%）/
  A 5.2440 / B 4.2660。均值 4.23825 < e2 4.2969，team best 保留 e2。
- **预注册门（天数/华为/燧原中位 ≥1.3x）三芯全负，列分块轴关闭**：
  榜首宽 shape 优势（天数 18-22/华为 17.5/燧原 5.6）不是列并行性；
  沐曦 +24% 是唯一正信号（不同后端偏好）。宽 shape 结构待新证据。

## 2026-09-15 E4 候选就绪：固定 1024 列块（验证通道中断）

- commit `51b2030a`。单变量：`BLOCK_COLS` 由 `next_power_of_2(n_cols)`
  （cap 1024，最多 7 档编译变体）改为固定 1024 掩码块——单一编译变体
  服务所有列宽；kernel 本体不动。依据：e2 整行形态 tianshu 84s 为
  编译主导；华为 1.93 vs 榜首 17.5 的缺口形态与重编译一致。
- ZIP：`artifacts/competition/fill_padded_rows/e4-51b2030/`，2723 bytes，
  仅 generic `fill_padded_rows.py`（`0f8c9b98…`）。
- ZIP SHA-256：`018b18f8bec25ca536c3f189a0a6a3f4942742599a262e250a9a0ebfb8fea3c5`。
- 预注册门：华为 ≥3.0 或 燧原 ≥3.0 或 天数 ≥15；ΔS 判优防窄行回退。
- 阻塞：GPU 通道中断，release 回执待补；未 preflight、未耗额度。

## 2026-09-15 16:30 E4 验证回执就绪（待明早首窗发射）

- release 回执：`artifacts/competition/batch5-verify-20260915/fill_padded_rows/`
  （exit 0，generic 25 次 launch，0 skip）。门不变：华为 ≥3.0 或
  燧原 ≥3.0 或 天数 ≥15。

## 2026-09-15 16:55 E4 平台终态：8/8 VALID 4.1965x —— 固定块轴零增益

- submission（e4，51b2030a）：天数 11.6308 / 沐曦 2.656 / 燧原 1.4618 /
  海光 5.928 / 昆仑 0.615（+0.02）/ 华为 1.7632 / A 5.3884 / B 4.1284。
- 与 TB e2 逐芯全在 ±0.3 噪声带内（华为 1.93→1.76、天数 11.9→11.63）；
  预注册门（华为/燧原 ≥3.0 或天数 ≥15）远未触及。
- **判定：next_power_of_2 七档编译变体不是 T67 缺口的驱动**——
  固定单变体后读数无变化。重编译风暴假说在本题证伪；华为 1.9 vs
  榜首 17.5 的缺口属于其他形态（PR 扫描的 num_warps/过特化轴待试）。
  TB 保持 e2 4.2969。

## 2026-09-15 17:10 E5 平台终态：8/8 VALID 4.223x —— n_cols constexpr 探针无效

- submission（e5，94bdc498，_ascend vendor n_cols constexpr，基线 e2）。
  华为 1.5764（e2 1.93，-0.35 噪声~回退带）；天数 11.37 / 燧原 1.37 /
  海光 6.57（+0.45）。ΔS≈-0.07：**TB 保持 e2 4.2969**。
- 判定：形状静态化（尾 mask 折叠+静态寻址）不是华为 9 倍缺口的形态；
  第二探针=移除 load 行谓词（Codex 候选，允许读 padding 行），明日。

## 2026-09-15 21:30 E6 平台终态：行谓词探针阴性，T67 便宜假说池耗尽

- 4.295 ≈ TB 4.297：华为 1.86（噪声带）。固定块/n_cols constexpr/
  行谓词三连阴——华为 9 倍缺口（金狐狸 45.6 证明可及）需要结构级
  新证据（对照其 tianshu 20.7/燧原 4.7 同步领先，疑整 kernel 形态
  不同），本季不再投便宜探针。TB 保持 e2。

## 2026-09-16 23:40 负计数修复与跨行布局首轮

- 题面只有单元素整数约束，无非负下界；reference负数遵循Python切片。固定E2 `a9c06b9f5e0bca8df598ad07e6e95d9cd7662962` 真实GPU复现：n=-1在连续/stride2输入分别67/68元素错；n=-rows/-rows-1四个对照通过，六次非空真实入口/JIT。
- 必要修复单独commit `600ef72ab7489100317a725ee96ad0a3a918ecb3`：generic与Ascend都恢复E2结构、device内scalar if归一化负起点；不host读取计数，maskedload仍在分支外。5方法双源码screen全过，各63入口/61JIT；源码SHA `b33d17b997e3072cb29029a965a1ee7424ba9f2fb45aa859b436b464380aa1f1`，测试SHA `1ed81b0d7224061d16a944b577e69f0b9d11fb17044c247c453caa4f66bbba0e`。generic改变属于契约修复，不能描述为仅vendor性能改动。
- 新结构依据本地官方FlagGems a7620cc1 Ascend fill/masked_fill的全局offset填充/选择，将输出展平跨行合并。以修正版600ef72a为性能对照，Ascend BLOCK1024、全局gridstride、i64地址、N_COLS常除数；不是E5旧逐行constexpr轴。
- 首轮全域flat：8方法正确性全过，包括负计数、跨行tail、强制grid1/2及CUDA graph内device-count重放。冻结18主桶+6宽行对照、五轮；主mean **1.8817546725**、GM **1.4365878134**，但最差主 **0.6097161030**、控 **0.8677507526**，未达0.97/0.95门，明确NO-GO，不能拿mean晋级。
- 仅一轮后续双路径候选：rows≥4096且cols≤64走flat，其余走修复E2。保留全部24个旧桶，包括所有失败点；另加4095/4096/4097×63/64/65和31/48列边界。此轮门在计时前冻结，若仍回退就停止结构轴，仅保留必要正确性修复。Ascend目标runtime未验证。
- `artifacts/competition/t67-flat-20260916/old-regression/reproduction.json` SHA-256 `c8ec704298c78d036590e3b4228c7bdfbae3c5519ad517e56b7c2a05b8a0fad3`。
- `artifacts/competition/t67-flat-20260916/screening/benchmark.json` SHA-256 `cacfca637e9e27be9069eae65d2748ed27262cb1f43cc92f9d74da51197a3b97`。
- `artifacts/competition/t67-flat-20260916/screening/ir-decision.json` SHA-256 `7dc3f7964296b21a44e213a007b12d78e4b082931e1d5ad1ea9c9eedc9d97338`。

## 2026-09-16 23:48 保护版仍失败，仅发布必要正确性修复

- 保留原24桶并补13边界，共37桶×5轮；8旧方法加边界方法共9，正确性全过。主mean **1.95461**、最差主 **0.99095**、控 **0.97454**，但新边界 **4097×63=0.90850<0.97**（邻接4096×63=1.10348），未过冻结门。停止本轮flat结构轴，不调门、不删case。
- 正式vendor/test已恢复独立正确性commit `600ef72ab7489100317a725ee96ad0a3a918ecb3` 的5方法字节；E7 negative-safe只验证负切片语义修复，不引用任何flat速度作为提交依据。平台需八芯正确并各≥0.1；团队最佳分独立记账，必要修复不以代理提速为前提。
- 同轮最新历史状态E6/sub15376均分4.295475，TB E2仍4.2969；状态 `artifacts/competition/pair-grouped-platform-20260916/t67-before-parallel-status.json` SHA-256 `16ab1c8cc5560bdf1df0e9595f3b32e4ef80057606141065ca484a6e7460aeca`。

## 2026-09-16 23:52 E7 negative-safe 发布门通过

- source/verification commit `600ef72ab7489100317a725ee96ad0a3a918ecb3`，仅E2结构+必要device负切片归一化，所有flat/guard实现撤出正式源码。generic及Ascend成员同SHA `b33d17b997e3072cb29029a965a1ee7424ba9f2fb45aa859b436b464380aa1f1`；测试SHA `1ed81b0d7224061d16a944b577e69f0b9d11fb17044c247c453caa4f66bbba0e`。
- exact release **5/5**，0fail/error/skip/xfail；两源码各63入口/61真实JIT（合126/122）。远端 `/tmp/flagos-t67-negative-release.53ApWy` PID398199、EXIT0；NVIDIA RTX5070Ti / Torch2.13.0+cu130 / Triton3.7.1。主任务再次verify_receipt、CRC、Git/成员及ZIP验签通过，源测试已恢复commit字节。
- ZIP `artifacts/competition/fill_padded_rows/e7-negative-safe-600ef72/fill_padded_rows.zip`，5106bytes，SHA-256 `6a6fa9fb51da18a0659f5021fa3e817f846593b0c7374e1b9c2b548e26707997`，成员fill_padded_rows.py/fill_padded_rows_ascend.py。目标芯未验证，KernelGen本轮共享服务无完整回报仅作通道限制，不记Ascend失败或成功。
- `artifacts/competition/t67-flat-20260916/release/verification.json` SHA-256 `72aee23ed6f4ffb661d05508a574a9d152d40b7f32d4b419a6b48c7de8b62a71`。
- `artifacts/competition/t67-flat-20260916/release/verification.log` SHA-256 `b6a6191ad051c115c40f12da1864792f644b548128e8475dc0d9ce1c1ae15389`。
- `artifacts/competition/t67-flat-20260916/release-audit.json` SHA-256 `a5018d9779595b09b17b1806d3cb28b002c49e362a79237137c3b54e0d573b8a`。
- 本次修复晋级要求八芯正确并各≥0.1，修复本身不要求比E2提速；团队最佳只有均分超过4.2969才更新。只提交一次，若目标编译/正确性失败按实际根因处理，不携带任何被淘汰的flat性能结论。


## 2026-09-17 00:05 停止提交，保留 E7

- E7预检仅GET，返回 `error: submission interval has 67s remaining`（退出码2），未生成nonce/intent、未上传、未正式提交。随后用户明确“查看进度，不再新提交”，不再重跑preflight，也不因跨日额度恢复而自动发射。
- `artifacts/competition/pair-grouped-platform-20260916/t67-e7-preflight.stderr` SHA-256 `0fe93921dd1f1a0bbd77aedf0adf5b3c1a35e4e0f611b6922dcad15ef8f20b72`；对应JSON输出为空，不能当成功预检回执。
- 00:02:31只读记录仍以E6/sub15376为最新，4.295475x；00:05榜单TB E2为4.2969x、第11，Top1金狐狸12.463525x，追平需190.06%。E7源/测试/ZIP与5方法回执保留，目标芯未验证；两版flat均未晋级。

## 2026-09-17 E8：Ascend persistent vendor，已提交

- 结构（`f8357985`，第二轮收割）：新增 `_ascend` persistent vendor（行 stride 循环 + count 提升，逐元素家族，对照 T73 +97%）。其余成员字节冻结；本题为已证配方家族单变量投放。
- screening（RTX 5070 Ti）：unittest 5 项全绿；black/flake8 过。
- release v2（commit `f83579853cc0163321fd11e6301a1ad61d46eb58`）：5 项全过 0F/E/S/X，changed-vendor 真实 launch 61 次，exit 0。回执 `artifacts/competition/round2-20260917/fill_padded_rows/verification.json` SHA-256 `29751e8cb0bf5ff4f8ae357eb02bb876fd284bf2abf73ba8f8f574d796f6479b`。目标芯 target-runtime-unverified。
- ZIP：`artifacts/competition/fill_padded_rows/e8-f835798/fill_padded_rows.zip（5677B）`，SHA-256 `e028168913e8e0f2172c07e9b9a7d2a7524dc9e0e948d74654f4934984deaba1`，2 成员。
- 预注册门：8/8 有效且均值 > 4.2969；华为（1.93 起） ≥ 3.9 为正信号。一次候选一次判决。

## 2026-09-17 E8 单次平台提交

submission **16579**，evaluating；八芯终态另节记录。

## 2026-09-17 E8 平台终态：8/8 有效，新 TB 4.43975x

submission 16579 completed/valid，均值 **4.43975x > 4.2969 换 TB**（+3.33%）。逐芯（vs E2 TB）：huawei 1.929→**2.518（+30.5%，方向正但未过 3.9 门）**；card_a 5.489(+0.7%)/kunlun 0.603(+3.2%)/tianshu 11.526(-3.1%)/muxi 2.657(-2.7%)/enflame 1.298(-14.1%，generic 未动，窗口)。他队华为 17.5-48.9 缺口仍大。
