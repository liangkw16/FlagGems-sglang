# Task 52 `fused_dual_residual_rmsnorm` 实验记录

```current
task: 52
operator: fused_dual_residual_rmsnorm
batch: 4
validity: invalid
platform: e6/10743六芯通过,昆仑失败,燧原pending;已无有效分可能
team_best_stage: -
team_best_speedup: -
sealed: no
next: 阶段探针和大形状代理验证已完成;昆仑4元素问题未复现,仍需目标原输入首分歧,不重投
updated: 2026-09-08
```

## S0（reciprocal: v*(1/rms)*w）: 5/8
- 天数✗(1/16.7M) 燧原✗(1/33.5M) 昆仑✗(4/33.5M) card_b✓

## E1（div_rn: div_rn(v,rms)*w）: 5/8（tianshu✓ card_b✗）
- 天数✓ 燧原✗ 昆仑✗ card_b✗(NEW!)

## E2（plain /: v/rms*w）: 5/8（tianshu✓ card_b仍✗）
- 天数✓ 燧原✗ 昆仑✗ card_b✗
- 结论：三种除法形式各自修一芯破另一芯，1/33M 元素在 bf16 舍入边界

## E3 AMD reciprocal vendor → card_b 翻绿（submission 10143）
- **card_b PASS 8.09x**（reciprocal 形式恢复 S0 通过路径）
- tianshu PASS 8.62（direct / generic 继续生效）
- **6/8**：天数/沐曦/海光/华为/A/card_b 全过
- 仅燧原+昆仑败（同款 1-4/33M 精度边界，两种除法形式均不过）

## E4 燧原两核拆分（submission 10151）
- 燧原仍败——memory 物化中间 bf16 也没修复 1/33M 边界翻转
- **判定：T52 6/8 是当前方法的天花板**（燧原+昆仑均不可修）

## 2026-09-05 Codex 会诊方案与 E5 预注册

Codex 配方矩阵部分兑现：E3 `_amd` reciprocal 修好 card_b（8.09x）与
预测一致。剩余两芯（燧原/昆仑）的**未试首选是 `tl.rsqrt` 配方**——
T19 fused_rmsnorm 同平台用 tl.rsqrt 在燧原/昆仑均通过（最强直接正
证据）；E1 div_rn ≠ rsqrt，该轴尚未消费。

**E5（预注册，1 发）**：`_enflame`/`_kunlunxin` 两 vendor 的两处 rms
均改 `v * tl.rsqrt(var+eps) * w`，其余芯字节冻结 E4 载体。
次选：`tl.sqrt_rn + tl.math.div_rn`（最贴 eager 源码顺序；sqrt_rn 在
厂商 fork 的可编译性先离线验证）。
已证伪勿重复：plain /（E2）、两核拆分 mid 物化（E4）、fp64 兜底
（燧原无原生 fp64）。E5 仍败 → 只读拉 raw_result 分诊 mid vs out
首个分叉点；无新证据则封 6/8，额度转 T42/T53。

## E5 rsqrt 配方（2026-09-06 00:12，submission 10322，daily_seq 1）

- 载体：generic(E2 plain /) + `_amd`(reciprocal) + `_enflame`/`_kunlunxin`
  （单核 `v * tl.rsqrt(var+eps) * w`）；source `a0298a2`，
  ZIP `e5-a0298a2` SHA-256 `5b1d42b1…c8a`（4 成员）
- screening：远端 `/tmp/flagos-t52e5.6EODJa`（NVIDIA 代理 unittest 5/5 OK
  含 vendor 矩阵；日志 SHA `fc188f5d…`；black diff 仅既有字节工具漂移）
- **终态 6/8 invalid_correctness**：天数 8.5836 / 沐曦 5.2425 / 海光
  8.7104 / 华为 3.7427 / A 8.1713 / B(_amd) 7.9668 过；燧原/昆仑各 1 case 败
- **失败情报（raw_result）**：两芯同败 `test[18]`（33.5M 元素大 case）；
  燧原 1 元素 abs 0.0171（容差 0.015）、昆仑 4 元素 abs 0.0195 + 1 处
  expected=0 的 inf 相对差——第五种数学形态（plain//、div_rn、reciprocal、
  两核拆分、rsqrt）同指纹边界失配
- **判定**：分叉点不在除法形式（T19 单 norm rsqrt 正证据未迁移到
  双 norm + residual 链；疑燧原/昆仑 torch 的 bf16 add/mid 舍入路径
  差异，无目标芯探测通道）。**T52 按 E5 预注册止损封存 6/8**；仅
  `sqrt_rn+div_rn`、torch 归约树复刻等全新结构证据可重开。额度 29/30

## E6 Top1 冲刺候选（2026-09-07，提交前）

T52 以新的可复现数值证据重开。generic 和 enflame/kunlunxin 两 vendor 的两次归一化均使用 `tl.sqrt_rn` + `tl.div_rn`，AMD reciprocal 字节保留。此前 plain `/`、rsqrt、仅 div_rn、物化中间值的失败不作为本次通过证明。

NVIDIA seed42、BF16 `[4096,8192]` 在 `(689,410)` 的第一层均方：旧 1.0039855241775513/reference 1.0039856433868408；rms 1.0019956827163696/1.0019958019256592；y1 舍入前 2.507812738418579/2.507812261581421，BF16 变成 2.515625/2.5。加 residual -2.53125 后 mid 为 -0.015625/-0.03125，最终 out 为 -0.0311279296875/-0.062255859375。诊断 kernel 与原 kernel 在三个 seed 的 mid/out 逐位一致。RN 写法修复 seed0/7/42；仅禁用 FMA 仍失败 seed42。新增大 BF16 回归保留在正式测试中。这是代理机根因证据，不能断言目标两芯根因相同。

目标芯探索：昆仑 KernelGen 返回 502/All backends failed；燧原 job f4a4aeb5-9800-4b32-b2a3-54d2d592d723 测试 0，报 `NameError: torch is not defined`，生成的 plain sqrt/div + mid 重载方案未采用。目标运行时仍未验证。

预注册：本轴首投一次，八芯正确且各芯≥0.1 才能计有效分。以六个旧过芯分数不变推算，剩两芯合计需 >0.98373333 才超当时榜首 5.425125，需≥2.28576333 才有 3% 余量；实际以本次八芯结果为准。

- source/verification commit：`b290b6a34e368a37b5c37351a4fd3cedb0c8e2a1`；ledger commit 为本节所属提交。
- 本地 py_compile、Black、isort、flake8 通过。NVIDIA release：6 方法、26 次 kernel launch，fail/error/skip/xfail 均为0。执行源：`src/flaggems_sglang/ops/fused_dual_residual_rmsnorm.py`, `src/flaggems_sglang/runtime/backend/_enflame/ops/fused_dual_residual_rmsnorm.py`, `src/flaggems_sglang/runtime/backend/_kunlunxin/ops/fused_dual_residual_rmsnorm.py`。
- 测试源码 SHA256：`54b9d711532f224970ae7ab7874b9e0d3aeedbeeb25f3a31da30c04c108c43f1`；各输入文件 SHA 见 verification-input.json。
- 远端 `gpu:/tmp/flagos-b4-top1.8nsvBC/t52-release`，RTX5070Ti / driver610.57.04 / Python3.12.13 / torch2.13.0+cu130 / triton3.7.1。串行后台执行：`timeout 600 /home/kevin/notebook/.venv/bin/python .agents/skills/flagos-operator-race/scripts/verify_release.py run --directory /tmp/flagos-b4-top1.8nsvBC/t52-release`。
- ZIP `/Users/bytedance/ccc/flagos/artifacts/competition/fused_dual_residual_rmsnorm/e6-b290b6a/fused_dual_residual_rmsnorm.zip`，12615 bytes，SHA256 `735a8efd32f70ca87e61c705bed89854f0a9bc8a0f9f8a517f0fd533cfb6d795`；dry-run 与最终 manifest 五项恒等字段全部匹配。
- ZIP member `fused_dual_residual_rmsnorm.py` SHA256 `692a5a1769b9bb4bc7a2c018a8c0ac16e73e26b7d2f46827df5bc5cc123427be`。
- ZIP member `fused_dual_residual_rmsnorm_amd.py` SHA256 `f254c779bd3200671f94645929a5a1040c8f85888cca329143fb203f6bc1b832`。
- ZIP member `fused_dual_residual_rmsnorm_enflame.py` SHA256 `1e1582905c142c7578321f610e3c8233f3a138f36d8c3ab626b4f81fb1b5dc78`。
- ZIP member `fused_dual_residual_rmsnorm_kunlunxin.py` SHA256 `257947698a394cbb541dea5f57c0cd8b3aac400cd86b28190b505de60003a287`。
- 证据 `/Users/bytedance/ccc/flagos/artifacts/competition/fused_dual_residual_rmsnorm/e6-b290b6a/validation/verification.json` SHA256 `5b1463670b31f82d405c9b14a4769e30cbbd8c70acbb92b92c564cf1d9e3530e`。
- 证据 `/Users/bytedance/ccc/flagos/artifacts/competition/fused_dual_residual_rmsnorm/e6-b290b6a/validation/verification.log` SHA256 `7e1d2ea4da5770779de8149601242eb0b329ae7d2c9b78a8773793de7a98f0ac`。
- 证据 `/Users/bytedance/ccc/flagos/artifacts/competition/fused_dual_residual_rmsnorm/e6-b290b6a/validation/verification-input.json` SHA256 `6a1a8de77d358d0b86c1136b7dce7438f50dff8022b54575c241f9c2243cb6cd`。
- 附加基准、诊断及 MCP 证据清单 `/Users/bytedance/ccc/flagos/artifacts/competition/fused_dual_residual_rmsnorm/e6-b290b6a/validation/evidence-sha256.json` SHA256 `fb67922ae0f359c12fe7f6bb2952eb9a361a4209bf45be43670f51249083bf26`。
- 首轮每题最多1次正式上传/提交；本次预算上限沿用批准的 T43/T51/T52 各4、T57 3、储备6，须有新证据才继续消耗。sending/uncertain/stale_after_upload 不自动重试。

## 本轮平台结果与止损（2026-09-07T14:55:36+08:00）

- submission `10743` / daily_seq `10` / created `2026-09-07T14:42:51`；preflight与上传/提交均只执行一次，远端ZIP验签 `verified`。
- 平台原始状态 `evaluating` / `pending`，通过6/8、终态7/8；average_speedup `None`，is_team_best `False`；观测时额度 `18/30`。
- 昆仑 case18仍4/33554432元素失败，最大abs0.01953125坐标(3969,4831)、expected0的inf相对差坐标(1005,6909)，与E5指纹相同。RN未解决目标芯卡点；平台聚合仍pending（燧原waiting_callback），但本候选已经不可能获得八芯有效分。保留代理可复现诊断和回归，不再以该证据外推目标芯。

| 芯片 | 状态/正确性 | 加速比 | 实际文件 |
| --- | --- | ---: | --- |
| tianshu | completed/True | 8.37006667 | `fused_dual_residual_rmsnorm.py` |
| muxi | completed/True | 3.762 | `fused_dual_residual_rmsnorm.py` |
| enflame | waiting_callback/None | None | `fused_dual_residual_rmsnorm_enflame.py` |
| haiguang | completed/True | 9.29186667 | `fused_dual_residual_rmsnorm.py` |
| kunlunxin | completed/False | None | `fused_dual_residual_rmsnorm_kunlunxin.py` |
| huawei | completed/True | 3.22833333 | `fused_dual_residual_rmsnorm.py` |
| card_a | completed/True | 6.99346667 | `fused_dual_residual_rmsnorm.py` |
| card_b | completed/True | 8.1058 | `fused_dual_residual_rmsnorm_amd.py` |

- 证据 `/Users/bytedance/ccc/flagos/artifacts/competition/fused_dual_residual_rmsnorm/e6-b290b6a/validation/52-submit.json` SHA256 `fb90f2eadb4bb1409b1c56f2e36a7c2d3da6f691d13dbc575d2bcfaa2ee78436`。

- 证据 `/Users/bytedance/ccc/flagos/artifacts/competition/fused_dual_residual_rmsnorm/e6-b290b6a/validation/52-status-first-round.json` SHA256 `df19fd00e8ad1a3e02d943df3d03a13b7774a9c273c19e026e4a159eb60417fb`。

## 2026-09-08 推荐方案实现与提交前验证（未提交平台）

保持RN生产公式；新增固定输入、reference/candidate/trace独立进程和逐阶段位值打印。4096x8192 BF16代理容差通过，out/mid分别254/75个位值差异，插桩不改最终输出；不是平台失败输入。

- source `c73f6c3f83ec38d5a2c40cfdef996e64e50ecd67`；verification `c73f6c3f83ec38d5a2c40cfdef996e64e50ecd67`。6 个测试方法、20 次实际 kernel 调用；选定 NVIDIA/代理范围门禁通过。
- 回执 `artifacts/competition/batch4-implementation-20260907/t52-release1/verification.json`，SHA256 `d578cd3f99afc053f1fbc3a3c4dde5ec4ab235e1d22e92fcd7a71a8175ad06f5`；日志 SHA256 `32871ab5032ced9c53d47900a351582a7958cd9ca09b3aa66acd62ace063d5ca`。
- 环境、逐源码执行范围、原始配对数据和未完成条件见[本轮报告](../implementation-batch4-20260908.md)及[证据清单](../data/batch4-implementation-20260908.json)。本轮不更新历史有效分，未做平台 preflight、上传或正式提交。
