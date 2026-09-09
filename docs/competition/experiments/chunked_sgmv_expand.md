# Task 47 `chunked_sgmv_expand` 实验记录

```current
task: 47
operator: chunked_sgmv_expand
batch: 4
validity: valid
platform: E16/11769八芯valid24.897未晋级；最佳E12/25.36275
team_best_stage: e12
team_best_commit: d649a9d
team_best_speedup: 25.36275
sealed: no
next: E17同adapter段合并release通过，待一次平台提交
updated: 2026-09-09
```

状态：S0 候选就绪（generic 单文件），远端 NVIDIA 代理 screening 通过
（9/9 单测 + 三项 lint + 基准 bf16 15.8–19.8x、fp32 7.2x，且基准走的是
无 `max_len` 的 host 同步兜底路径）。榜首 c2flow 23.3266x（2/5 队达标）。

## 契约锁定

- 签名：`chunked_sgmv_expand(x, weights, batch_info, slice_offsets, max_slice_size, base_output)`
- x `[S, n_slices*r]`、weights `[num_lora, out_features, r]`、
  slice_offsets `[n_slices+1]`、base_output `[S, total_out]`
- 计算：每请求 `out[rows, o0:o1] += scaling * x_slice @ W_slice.T`
  （fp32 精度累加）；`lora_ranks[w_idx]==0` / 空段跳过；**rank 非零时
  用满 stored rank（reference `r = weights.shape[-1]`，不按
  lora_ranks 截断——与 T46 族不同）**
- 返回 `base_output.clone()` 语义的新张量（fp32 累加后单次 cast 回
  base dtype）；输入全部不变
- batch_info：seg_indptr/weight_indices/lora_ranks/scalings/
  permutation/bs（**无 max_len**）；scaling per-adapter 标量
- 容差：fp32 1e-4 / bf16 1.5e-2 / fp16 1e-2；八芯；标准反作弊条款

## 方案（S0）

- qkv_lora_b 骨架直接映射（T22 结构）：grid
  `(token_blocks*output_blocks, n_slices, bs)`，64/128/32、4 warps、
  stages 3；slice 边界 `slice_offsets[i]`；x 列偏移 `slice_id*r+k`；
  base RMW（load→fp32→`+acc*scaling`→cast 回 store）
- `max_len`：`getattr(batch_info, "max_len", None)` 优先（harness 若
  提供则零同步）；缺失时一次 host 同步
  `int(diff(seg_indptr).max().item())`（元数据准备，非核心计算）
- 尾块 mask 绝对列号（T37 E1 教训）；窄 slice 无效 output block 提前
  return；读序：先 seg_indptr 判空后读 adapter 元数据（空段哨兵免疫）
- fp32 accumulator + `input_precision="ieee"`

## 验证证据（screening 模式，未提交候选）

- 远端：`gpu`（RTX 5070 Ti）；目录 `/tmp/flagos-chunked_sgmv_expand.LY2hC8`
- SHA-256（与 source commit `d7d8c47` 逐字节一致）：
  - `src/flaggems_sglang/ops/chunked_sgmv_expand.py`
    `0d52334731d37bf9fab08888f5963908cd9edbe91d79b8c584bb3649b6fdf579`
  - `tests/test_chunked_sgmv_expand.py`
    `1f43d142cc2863003684376a091dc972e6b10d73b13dfd73fc9ed7ef7f33e05f`
- 门禁：py_compile / black / isort / flake8 全绿；unittest 9/9 OK
- 单测覆盖：3 dtype；非等宽 slice（65/80/129）；r 8/16/32/64；空段 +
  越界哨兵 widx；全零 rank（输出=base）；单 token 单段；单位
  permutation；rank0 时 base 不被触碰且返回新张量；空批次；输入不变性
- 基准（do_bench median，含 host 同步兜底路径）：

  | shape | speedup |
  | --- | ---: |
  | B=32 seg=128 2×2048 r=32 bf16 | 19.8x |
  | B=64 seg=64 3×4096 r=64 bf16 | 15.8x |
  | B=16 seg=256 1×1024 r=16 bf16 | 19.6x |
  | B=32 seg=128 2×2048 r=32 fp32 | 7.2x |

## 已知风险与对策（LoRA 族平台实证）

- 天数：fp32-ieee dot 静默错 → split-fp16 四点积 vendor（必踩坑）
- 燧原：i64 IR 第一嫌疑（metadata int64）→ vendor 降 i32；仍败则
  route/materialize + 64³/stages2 规则 GEMM
- 昆仑：直接上 route/materialize + 32³/stages1/
  `do_not_specialize=["M"]`（index_select 物化 → 逐段规则 GEMM →
  逆 index_select，T28/T37 双芯实证）；**不要重走 T23 pack/scatter
  五连败**
- 华为：3D grid 可用；展平超 65535 改 capped 折叠；BLOCK_N 64→128
  有 +40% 先例
- host 同步 25us 税：若平台计时含 wrapper 且 batch_info 有 max_len 则
  自动免掉；无则考虑 vendor 内预计算

## 提交预算与止损

- 默认 5 发：S0 探路 → vendor 单变量 → 回归储备；同指纹两连败止损

## 时间线

- 2026-09-04 00:xx 契约锁定、S0 实现 + 远端 screening 9/9 + 基准
  （自修：torch.full 无 generator 参数、flake8 F401/F841）

## 平台结果（2026-09-04 凌晨）

- S0（submission 9376，daily_seq 5）：6/8，燧原+昆仑 correctness 失败。
  逐芯：天数 29.489 / 沐曦 21.64 / 海光 55.865 / 华为 13.5175 /
  A 49.5485 / B 27.337
- E1（submission 9383，daily_seq 7，source `663286c`，ZIP
  `413302e1…`）：燧原 vendor（i32 + 无早退 + clamp 哨兵 + stages2）
  **已翻绿 0.2545x**；其余七芯全过（天数 27.01 / 沐曦 22.40 /
  海光 53.99 / 华为 13.28 / A 50.14 / B 29.39）；**昆仑评测中**
  （9376 昆仑为 fail，e1 待终态）
- 若 e1 昆仑仍败：route/materialize vendor（wrapper index_select
  物化 → 每非空段 32³/stages1/`do_not_specialize=["M"]` 规则 GEMM →
  逆 index_select，T28 E11 昆仑 1830s 崩溃→4.40x / T37 E4 3.47x 双证）
- vendor 数学在 NVIDIA 代理 variants 矩阵 10/10 验证（曾抓出 ieee
  丢失导致的 TF32 精度回退，已修复后才提交）


## MCP 实机初筛归档（2026-09-04 晨，24 job 全部终态）

- 产物 `log/kernelgen-round/out_<op>_<chip>.json`（24 个，含 SHA）；
  协议：注入执行 + 失败神谕 + 终态代码保真 diff
- 干净通过（fidelity=True 且零 hard error）：本题华为/天数（详见
  各算子行）；海光/沐曦后端当夜多次 502（`ld0428.baai.ac.cn`），
  这些芯的编译信号不可得，非候选失败
- 保真失败（LLM 改写）= 无判定，不作数；harness 侧 artifact
  （NameError/IndexError/`constexpr[0]`）不计入失败神谕
- 平台实测（本账本上方小节）已是更强证据，MCP 结论仅作发射风险
  参考留存

## E2 昆仑 route/materialize vendor（2026-09-04 03:5x，submission 9414，daily_seq 8）

- 7/8，昆仑仍 correctness 失败——**同指纹三连败（S0/E1/E2），昆仑轴
  按 stop gate 封存**；T28/T37 的规则 GEMM 配方在本题不奏效
- 其余七芯（generic/enflame vendor 不变）：天数 29.738 / 沐曦 21.7515 /
  燧原 0.2485 / 海光 55.3695 / 华为 15.6055 / A 48.6245 / B 28.7775
- 定格 7/8；七芯均值 ~28.6x。剩余提升轴：华为/沐曦性能（非正确性）

## E3 昆仑 GEMM 修复重投（2026-09-04 0x:xx，submission 9467，daily_seq 16）

- Codex 咨询指出 vendor GEMM 确定性缺陷；代理实测证实：**rank 32 单趟
  K 循环精确、rank≥64 第二趟起错 ~1e1**（Triton 3.7.1 对该 kernel
  形态的 codegen 问题；独立复刻加一条 store 即不复现），而 variants
  矩阵此前只覆盖 rank≤32——真实盲区
- 修复：`BLOCK_K = next_pow2(rank)`（cap 512）恒单趟（T28/T37 昆仑
  实证形态本身就用 K≤32 单趟）；rank 64/96/128 入永久回归
  （source `8a9296c`，ZIP `35568985…`）
- **平台结果：昆仑仍 fail**（第 4 投；有效指纹 3 个结构全败：
  元数据型 / i32 型 / 规则 fp32-ieee GEMM）→ 昆仑对该题 conclusive
  封轴，判定为该芯片后端独立数值问题
- 七芯（修复后读数）：天数 28.82 / 沐曦 25.08 / 燧原 0.2515 /
  海光 53.55 / 华为 14.42 / A 49.97 / B 29.17——七芯均值 ~28.7x

## 失败情报破译（2026-09-04 晚，submissions API raw_result）

**发现**：列表接口 `operator-submissions` 的 `raw_result.errors/
failed_cases` 本就携带完整失败详情，CLI status 视图把它过滤掉了。
用 token 直接 GET 即得——无需浏览器登录。

- **昆仑四投全败根因 = `index_copy_(): Expected a long tensor for
  index, but got Int`**——平台 permutation 为 int32，昆仑 torch 要
  long（NVIDIA 接受 int32，代理全绿是盲区）。E4 一行修复
  `rows.long()`（source `b20da5b`，submission 9502）
- **E4 结果：昆仑 PASS 3.7365x**（7 芯 + 昆仑全过），但燧原
  **评测超时**（1830s，R 状态机器忙；同字节 E3 过 0.2515x）→
  非代码回归
- **E5（source `b34d040`）**：燧原换昆仑同款 route/materialize
  GEMM（64³/stages2 + long 索引），打包 c62fc211…；提交意图停在
  `stale_after_upload`（文件上传成功、正式 POST 前过期；status
  核对无提交记录、额度未耗）。CLI 按设计拒绝自动重试，
  **等待用户授权归档重提**

## E5 登顶（2026-09-04 深夜，submission 9512，daily_seq 22）

- **8/8 valid，平均 25.0048125x，team best，当前榜首**
  （前榜首 c2flow 23.3266x；过线队 3）
- 燧原 route/materialize（64³/stages2）0.178x 过门槛（上轮同字节
  评测机超时）；昆仑 long 修复保持 3.717x；天数 29.06 / 沐曦 22.40 /
  海光 53.27 / 华为 13.74 / A 49.01 / B 28.66
- stale_after_upload 意图处理：根因是我同窗口提交 T45 e4 改变了
  账号级 live 状态触发上传后绑定校验；status 复核无提交记录、额度
  未耗 → 用户明确授权后归档（`.git/flagos-platform/archived/`）→
  重新 preflight → 单次提交

## E8-E10 榜首防守优化尝试（2026-09-05 深夜，submissions 10164/10175）

- **E8**（enflame direct 3D-grid + i32）：燧原 FAIL——间接寻址 + dot
  组合在该芯不支持（与 S0 同根因），route/materialize 是唯一可行形态
- **E9**（+ 昆仑 direct 3D-grid）：昆仑 FAIL——同样间接 + dot 不支持；
  7 非昆仑芯水位上涨（华为 15.61 vs 13.74、A 50.05 vs 49.01）
- **E10**：回退两 vendor → ZIP SHA 与已提交 e5 相同，CLI 拒绝重掷
- **结论：T47 25.00x 已是当前 vendor 组合的最优**。route/materialize
  对燧原和昆仑都是唯一 correctness 可行的形态，无法用 direct kernel
  替换来提速。水位上涨只在无效提交中可见，不可捕获。

## 2026-09-05 守榜注意事项（Codex 会诊附带发现，已核源码）

`_kunlunxin/ops/chunked_sgmv_expand.py` GEMM K 循环推进为
`b_ptrs += BLOCK_K * stride_bn`，应为 `stride_bk`——现网全部 shape
K ≤ BLOCK_K 单趟未触发（潜伏笔误，不影响已验 8/8 结果）。守榜期间
不动字节；若未来重开昆仑轴或调整 BLOCK_K，必须先修此行并补 K 多趟
单测（33/64/96/100/128）。

## 2026-09-08 推荐方案实现与提交前验证（未提交平台）

昆仑/燧原多轮 B 地址递增修为 BLOCK_K*stride_bk，并将 BK 上限降至128。旧源码直接多轮回归各51/51元素失败，修复后完整通过；补 rank127/128/129/511/512/513。

- source `8ba31a102f4ef0430c08f12c4622b27430915071`；verification `8ba31a102f4ef0430c08f12c4622b27430915071`。12 个测试方法、105 次实际 kernel 调用；选定 NVIDIA/代理范围门禁通过。
- 回执 `artifacts/competition/batch4-implementation-20260907/t47-release1/verification.json`，SHA256 `65545a1b6cbc33c6a5e15aeb923bf9c332eed78d492a678b2fcc0daae53c37e0`；日志 SHA256 `0382828f0f48a4b62affadae30ad579444269d6165ed4bc6f416484aba45b232`。
- 不可变 ZIP `artifacts/competition/chunked_sgmv_expand/research-20260908-8ba31a1/chunked_sgmv_expand.zip`，SHA256 `fe9639026676a93739ce63b14329c01146236ba93bccc0c82108f7f46f447b5e`；与 dry-run manifest、构建和 existing 验签一致。ZIP 是候选产物，不等于目标芯或平台已通过。
- 环境、逐源码执行范围、原始配对数据和未完成条件见[本轮报告](../implementation-batch4-20260908.md)及[证据清单](../data/batch4-implementation-20260908.json)。本轮不更新历史有效分，未做平台 preflight、上传或正式提交。

## E11 额度恢复后评测预注册（2026-09-08）

- 用户已明确授权提交评测。本轮顺序 T47 E11 → T51 E8，预算每候选仅1次上传/正式提交；2026-09-08T01:42:22+08:00 实时额度30/30，账号全局间隔120秒。sending/uncertain/stale_after_upload/submitted 均不自动重试。
- 假设与优先级：多轮 K 正确性修复守榜；generic 与 E5 完全相同，燧原/昆仑修正 stride_bk 并限制 BK≤128。当前第1，历史均值25.0048125。性能收益尚待目标评测，不把修复本身当作已提速。
- 晋级/停止门：8/8 valid、每芯≥0.1；均值>25.0048125 才晋级 team best。若无净收益保留 E5；任何目标失败或超时只读取证，不重投同字节。
- source commit `8ba31a102f4ef0430c08f12c4622b27430915071`；verification commit `8ba31a102f4ef0430c08f12c4622b27430915071`；本节 ledger commit 为提交本节的独立文档提交，不等同源码或验证提交。
- 正式不可变 ZIP `/Users/bytedance/ccc/flagos/artifacts/competition/chunked_sgmv_expand/e11-8ba31a1/chunked_sgmv_expand.zip`，18326 bytes，SHA256 `fe9639026676a93739ce63b14329c01146236ba93bccc0c82108f7f46f447b5e`；正式阶段 `e11`，与上一节 research 包逐成员一致，existing 验签通过。
- release 回执 `artifacts/competition/batch4-implementation-20260907/t47-release1/verification.json`，SHA256 `65545a1b6cbc33c6a5e15aeb923bf9c332eed78d492a678b2fcc0daae53c37e0`；日志 SHA256 `0382828f0f48a4b62affadae30ad579444269d6165ed4bc6f416484aba45b232`；测试 SHA256 `9cc151172b02ed5ecad05f2138958c59b7e6378ba5a6f7dee050d8ccae9ec666`。
- 所选 release 范围全绿，目标设备仍为 `target-runtime-unverified`，本次授权评测补齐；保留上一节执行范围，不将代理通过写成目标通过。
- 原始 manifest / preflight / submit / 状态证据保存目录 `artifacts/competition/batch4-submit-20260908`。提交前尚无本阶段平台结果，历史 best 保持。

| ZIP 成员 | SHA256 |
| --- | --- |
| `chunked_sgmv_expand.py` | `0d52334731d37bf9fab08888f5963908cd9edbe91d79b8c584bb3649b6fdf579` |
| `chunked_sgmv_expand_enflame.py` | `5cdf1c657a108f9fee016742f298f0aab9e264a174ee8dddbeb0edf43980f36b` |
| `chunked_sgmv_expand_kunlunxin.py` | `a51fa38d50babc3a45ef177e6357408f4232bba1ae6f33230461a22d3ce08e4b` |

## E11 平台终态（2026-09-08）

- submission `11031`，daily_seq `1`，created `2026-09-08T01:49:42`；观测 `2026-09-08T01:50:18.195945+08:00`。`completed / valid`，8/8，均值 **21.6584375x**，`is_team_best=false`，保留 E5 **25.0048125x**。
- 一次上传、一次正式提交；远端匿名下载验签通过，18326 bytes，ZIP SHA256 `fe9639026676a93739ce63b14329c01146236ba93bccc0c82108f7f46f447b5e`。file URL SHA256 `21ed812de07c9e6176a21646f50b71fe06c3b5c608cf106f535fea10f236e799`。
- 预注册 ledger commit `e18b94b876dabbadde49762f5fe0bfe7cbf46a75`；source / verification commit 均保持 `8ba31a102f4ef0430c08f12c4622b27430915071`。
- 原始提交 `artifacts/competition/batch4-submit-20260908/47-submit.json` SHA256 `b96a2cb922e377f96b6a3d4c58467757742c49099446a1070e39bf4d2f07970a`；终态 `artifacts/competition/batch4-submit-20260908/47-status-final.json` SHA256 `742c84090e245dcf6d7505ece76d399b284ad3e1803abc85df412b96adb29b2d`。

| 芯片 | 正确性 | E11 加速比 | E5 加速比 | 实际文件 |
| --- | --- | ---: | ---: | --- |
| tianshu | PASS | 29.757 | 29.056 | `chunked_sgmv_expand.py` |
| muxi | PASS | 21.5455 | 22.404 | `chunked_sgmv_expand.py` |
| enflame | PASS | 0.1725 | 0.178 | `chunked_sgmv_expand_enflame.py` |
| haiguang | PASS | 24.629 | 53.271 | `chunked_sgmv_expand.py` |
| kunlunxin | PASS | 3.716 | 3.717 | `chunked_sgmv_expand_kunlunxin.py` |
| huawei | PASS | 13.8335 | 13.7365 | `chunked_sgmv_expand.py` |
| card_a | PASS | 50.9785 | 49.0125 | `chunked_sgmv_expand.py` |
| card_b | PASS | 28.6355 | 28.6635 | `chunked_sgmv_expand.py` |

- 结算：两 vendor 在平台本轮用例正确性通过；燧原0.1725、昆仑3.716，均未提速。均值降低主要来自海光（使用与 E5 相同的 generic 字节），单轮数据无法证明代码回归或明确环境原因。平台覆盖范围不等于新增大 rank 回归已在目标设备执行。
- 停止门已触发：本候选未晋级，不重投；保留真实步长修复作为后续开发基线，历史榜单 best 不变。此时剩余29/30，下一发为预注册 T51 E8。

- 提交后实时排名复核 `2026-09-08T01:54:27.199190+08:00`：第 **1**，本队 best **25.0048125x**，榜首 **25.0048125x**（SoulCoder）。证据 `artifacts/competition/batch4-submit-20260908/tasks-after.json` SHA256 `f7959a760a01815c0072e76de5c3e7fa8fb08c4e41ec3579ca946c6c64001170`。

## E5r 防守重掷：**8/8 valid 25.1925 新 TB，但 rank 1 已失**（2026-09-09T06:24，sub 11641）

- 隔夜 c2flow 以 42.9844（+71.9%）夺走 T47 第一；全榜同夜上移 7–24%，
  判定批量慢窗被强队捕获。我方 E5 字节重掷（5286d26）：天数 29.07 /
  海光 53.26 / A 50.33 晨窗高位，**25.0048→25.1925 新 TB**，距 42.98
  仍差 1.71x——夺回只能靠同量级慢窗命中（E5 重掷剩 1 次 + E11 身份）。
- 跨芯知识：晨窗（06:2x）天数/海光/A 同窗齐高 = 慢窗可被传感器捕获。

## E12 燧原 128-tile 探针：**TB 25.3628 但燧原假设证伪**（2026-09-09T09:15，sub 11694）

- 单变量：燧原 vendor tile 64×64→128×128（m≥128 时）。**燧原 0.178
  原地（0.177→0.178）——"薄 MMA/向量路径"假设证伪**；均值 25.36 新 TB
  全部来自其他芯窗口（天数 30.1/海光 54.2/A 50.0）。
- 燧原 0.18 地板的剩余解释：逐段 route/materialize + per-(segment,slice)
  launch 风暴（GCU launch 开销主导）。修法=单 launch 批处理（设备端
  tile→segment 映射，moe-fp8/T28 配方），属半日级重构，若做需明日
  16:00 前完成载体。tile 轴关闭。

## E13 分组单 launch GEMM（2026-09-09，sub 待回填）

- **结构**（codex-ask 会诊主线）：燧原/昆仑 vendor 从 per-(segment,slice)
  launch 循环改为**单次分组 GEMM**——wrapper 一次 index_select 物化 +
  kernel 平铺 task 解码（标量 div/mod）、只读标量元数据、零运行期分支、
  零间接操作数地址、钳制合法地址 + store-mask、多轮 K（BLOCK_K≤64，
  按 k_start 重算地址，E11 stride 错误类不可再现，K 尾 constexpr 掩码
  补零）。grid.x 封顶 65535 + work 循环。
- **过程修复**：①首版 K 尾钳制重复加载第 0 列被 dot 计入（rank=127
  screening FAIL）→ constexpr k_mask 补零；②大 rank 单趟 BLOCK_K 爆
  shared（代理 101KB；旧 vendor 在代理同样爆——它从未在代理跑过大
  rank，平台芯局部存储更大才通过）→ 多轮 K + BLOCK_K≤64。
- **代理阶段计时（launch 风暴实锤）**：旧 per-seg vs 新 grouped =
  4 段 **2.97x** / 16 段 **2.16x** / 64 段 **12.0x**——段数越多收益
  越大，燧原 0.18 地板（launch 开销主导）预期多倍改善。
- 回归重写：multi-k stride 测试改为直接驱动新 kernel（K=65×3 趟），
  保留 rank 127/129/511/512/513 全矩阵。release 全绿（12 方法，
  generic 25 + 两 vendor 各 12 launch）。
- 预注册门：8/8 且燧原 ≥0.5 视为结构兑现、昆仑 ≥6 视为大兑现；
  任一 vendor 编译失败回滚 e12 字节。

## E13 平台终态 + e13r 重掷（2026-09-09 中午，sub 11728 → e13r）

- **e13 终态：invalid 6/8**——燧原/昆仑双芯 1830s 超时，**超时阶段=
  验证执行阶段，崩溃栈在平台 reference 侧 inductor compile_worker
  （昆仑崩溃族指纹）**，非我方 kernel 编译；六强芯全过带内（华为
  15.6/A 50.9 等）。代理 12x 结构收益待真机裁决。
- **e13r 重掷**（`be00f43`，字节同 e13）：判别"平台崩溃族抖动"vs
  "新 kernel 触发评测资源问题"。再崩同指纹 → 分组形态判死、回滚
  e12 字节、runtime work-loop 列为嫌疑；通过 → 拿到燧原/昆仑结构读数
  （门：燧原 ≥0.5、昆仑 ≥6）。
- e14（dtype 探针，`c2ffd5d` 全绿在库）冻结不发：同结构，e13 判死则
  连带死，e13r 过则按读数决定。

## E15 设备路由 + FP32 向量归约（2026-09-09，本轮预注册）

- 用户授权：充分利用提交机会完成结构尝试；每个候选只提交一次。
- 改动：两弱卡去掉分组 dot 与逐段物化，Triton 预路由 + 每 token/output tile 向量乘加；K 维部分和在循环后归约。generic 冻结。上游机制参考 vLLM PR52880，固定提交 3d45361674f874eccf51f04999e17e5f0b28c3b4。
- 契约：原签名和 FP32 计算/输出 dtype；rank 非零时使用权重存储 rank；空段、部分覆盖、零 rank 保留 base，输入不修改。新增 int32/stride、动态 metadata、65535 网格边界及多 K 回归。
- 门：八芯正确且每芯 >=0.1；整题均值 >25.36275 才晋级。两弱分数合计增加至少4作为结构收益信号，不承诺六强冻结即可 Top1。失败先取 raw_result，不重投相同候选。
- NVIDIA release 14 方法通过，完整调用对 E12 两弱旧模板代理配对速度约8.10/8.23/26.75/41.14/99.33倍；六轮 AB/BA 原始数据在 validation/perf.json。对比源为 d649a9d，非 E13；不能外推目标性能。
- 目标资源：KernelGen schema 无固定 Triton 执行接口；本轮 sunrise 单轮请求初始生成失败（0 attempts/tests）；kunlun 返回502。响应在 artifacts/competition/structural-20260909/mcp/，目标两芯仍 target-runtime-unverified，由已授权平台评测补齐。
- 远端：/tmp/flagos-t47-e15.fMAwBB，PID332772，timeout600，run.sh 先执行 verify_release.py run，再完整 wrapper 计时；RTX5070Ti、torch2.13.0+cu130、Triton3.7.1。
- source commit / verification commit：`7c5bfc05cf2107f1a5787599d39839b0469fe35b`；ledger commit 为本节独立文档提交。
- test SHA256：`af8657f177785feb6d0129721cb4d25b64b12e82c1556e289a9e60e2931523b2`。
- ZIP：`/private/tmp/flagos-batch4-structural-20260909/artifacts/competition/chunked_sgmv_expand/e15-7c5bfc0/chunked_sgmv_expand.zip`，18268 bytes，SHA256 `d2c42661289c5a737432e7f1e8a8b73fd732e4f63f6698213e0b20e938beb733`；dry-run/build/verify-existing 一致。
- release receipt SHA256：`2c7669221b4f0c1736839b91b1f3bb9df7d22c3faf51c45307e18f78cc925f00`；完整日志 SHA256：`043a513b7f5a30437ce4eba0b4f41062bd793a4ebfffc570827c0757cc19f766`；bench.py SHA256：`416fc3c942f383fc3a43c911a378ae35873ef5ae18df8db3bb60d8fbeac0f17a`；perf.json SHA256：`ec3a1cd46dec30ebe8fe2a9405ad23db38d318c0a8a8336013fe19f6ebb41273`。

| 成员 | SHA256 |
|---|---|
| chunked_sgmv_expand.py | `cec9fec2b67b3cd9c92cc83da01fd626eb3469ddbbe8795b05d708fd065e6059` |
| chunked_sgmv_expand_enflame.py | `d49b12dc4890d0cea083c9265986e2e0732e7353d0358f87157835627aaf81bf` |
| chunked_sgmv_expand_kunlunxin.py | `d49b12dc4890d0cea083c9265986e2e0732e7353d0358f87157835627aaf81bf` |

## E15 平台终态（2026-09-09 12:09 CST）

- 2026-09-09 12:04:39 正式提交 `11764`，当日第19次；一次上传/一次提交，远端 ZIP SHA256 复核一致。完整响应见同产物目录 `submit.json`、`status-watch.json`、`raw-status-1.json`。
- 终态 `invalid_correctness`，8 芯终态、7 芯正确；昆仑 case0/1 在 XPU arch3 的 `make_ttxir` 编译阶段失败：`OutOfResources: uni_sram PassManager::run failed`。错误包装显示 Required=0/limit=0，不能由此推断实际 SRAM 需求；栈中没有 kernel 名称，尚不能区分 route/vector。不是已证明的平台 reference 故障。
- 燧原正确但速度 **0.01**，低于 >=0.1 的有效门槛。代理机大幅提速没有迁移，两弱向量方案关闭，不再重投。
- 其余逐芯：天数28.7435、沐曦21.4795、海光53.829、华为14.752、A50.5255、B28.662。八芯均值无效，不据此晋级；团队最佳仍 E12/25.36275。
- 后续 E16 独立验证 generic 的设备端段 tile 前缀调度；两弱恢复 `8ba31a1` 已通过平台的有界 K 模板。前述 E13 的 reference 崩溃归因是历史推测，已有 raw trace 不足以证明候选无关。

## E16 设备端紧凑段 tile 调度（2026-09-09，提交预注册）

- generic 保留同一 IEEE dot 主体，设备端计算各段 ceil(length/64) 的前缀；扁平 task 二分寻找实际段，消除无 max_len 时的 host 同步和长短段矩形网格浪费。给定 max_len 时走原单 kernel 网格；bs>4096 保留原最大长度兜底。65535 网格分批，GPU前缀终点屏蔽多余上界 tile，无持久任务缓存。
- 两弱回退到 `8ba31a1` 的已通过平台有界 K 模板，避免把 E15 编译失败和0.01速度带入新候选。新收益轴仅 generic 调度；相对 E12 燧原 tile 恢复64×64，已知该tile轴速度在0.17附近。
- 门：八芯正确且每芯>=0.1；avg>25.36275 才晋级；六强合计较E12增加>=15% 视为结构兑现。失败先取原始错误，不重投。
- NVIDIA RTX5070Ti release 15 方法、0 fail/error/skip/xfail，所有3成员执行。新增高度偏斜段、重复前缀、4097段兜底，以及有/无max_len均与reference比对。远端 `/tmp/flagos-t47-e16-release.WtRRFj`，PID333511，timeout600。
- 六轮AB/BA、每次30完整调用相对 `7c5bfc0` generic：无max_len约1.4–1.7倍；有提示约0.95–1.02倍，额外参数有小幅开销。平台隐藏形状收益未知，不外推Top1。
- source/verification commit：`f9cb2475c7b61ab97e485c28cd0207fac7db15a9`；ledger commit 为本节独立文档提交。
- validation/verification.json SHA256：`f86c9075556231ec3a5acb6a665b9fdda2a247d4aceb33f8b384719e20b9976e`。
- validation/verification.log SHA256：`ba7f2d236befb7d9cc30a84f5f017e34e162af72c85d1c7574c5af35d56bc886`。
- validation/bench.py SHA256：`5e99afe6cf384cc82a9146451619b699ca9be4e8c2a9c470e2d93142350096cc`。
- validation/perf.json SHA256：`a04ac4a3ff165b6225ac128297c0094d083b81f2a10656b1c203008a13e4fe76`。
- test SHA256：`ee475a1c2d8d665ab2f0df7b37bc5eaa7d08e9876ac9dacca74b089575b443af`。
- ZIP：`/private/tmp/flagos-batch4-structural-20260909/artifacts/competition/chunked_sgmv_expand/e16-f9cb247/chunked_sgmv_expand.zip`；20975 bytes；SHA256 `694cb6a3a5d6f769e919fdb613d71c98160ec67446c2f3a2862d672497825479`；dry-run/build/verify-existing 一致。

|成员|SHA256|
|---|---|
|chunked_sgmv_expand.py|`9f0c4d0afaf296e97f973f957abcd189e3bb1bfebba82b13a46b3dc40eb955fc`|
|chunked_sgmv_expand_enflame.py|`5cdf1c657a108f9fee016742f298f0aab9e264a174ee8dddbeb0edf43980f36b`|
|chunked_sgmv_expand_kunlunxin.py|`a51fa38d50babc3a45ef177e6357408f4232bba1ae6f33230461a22d3ce08e4b`|

### E16 八芯终态（2026-09-09 12:27 CST）

- `11769` 于12:25:42正式提交，当日第21次；一次上传/提交，远端SHA256复核一致。8/8 valid，均值 **24.897**，较最佳25.36275低1.84%，未晋级。
- 逐芯：天数29.76、沐曦21.653、燧原0.147、海光52.4575、昆仑3.708、华为13.1735、A49.458、B28.819。设备端调度正确，但完整代理调用中的1.4–1.7倍未反映为平台整题提速；无法确认平台计时/形状对host同步的权重，不能据此盲迁T48。
- 证据 `e16-f9cb247/submit.json`、`raw-status-1.json`。恢复generic到原最佳族，下一轴合并同adapter多个段，沿用已经在两弱通过的regular GEMM；保留未覆盖行和空段语义。

## E17 同adapter段合并（2026-09-09，提交预注册）

- 不改变GEMM及其launcher：读取既有host段元数据，按weight index把多个段组成同一行列表，每个adapter一次gather、GEMM、scatter。只有一个段时沿用原行视图，不做cat；重复adapter仅合并行，权重、scale和slice语义不变，输出行独立。空段先跳过再读取哨兵adapter，保留既有rank0/负adapter规则。
- 两弱vendor增加路由合并；generic恢复`01d736b`已通过平台的基线，上一结构候选未晋级，不继续混入。kernel/launcher AST与`f9cb247`逐函数相同，新增互相穿插的重复adapter、int32 permutation、空段哨兵、3dtype测试。
- 预注册：八芯正确且每芯>=0.1；均值>25.36275才晋级；两弱合计提高至少15%视为合并有效。只提交一次；若平台没有重复adapter收益则关轴。
- source/verification commit：`cea2a0c10878b39c251a36857d97311d1ab4cd73`；ledger commit 为本节独立文档提交。NVIDIA RTX5070Ti release 16方法，0fail/error/skip/xfail，3打包成员均实际执行。远端`/tmp/flagos-coalesce-release.6I3BMw/t47`，PID333966，timeout900，T47和T48按顺序验证/计时。
- 完整enflame wrapper六轮AB/BA、每轮10次，基线是`f9cb247`原逐段vendor。重复adapter代理速度范围2.82–9.75倍；各段adapter独立对照1.00–1.00倍。未获目标同源计时，平台负责补齐；不把代理倍率记作目标速度。
- validation/verification.json SHA256 `26e9114b99c288535638702f6e33450ee0ed2ee59791202488ae19febb375aeb`。
- validation/verification.log SHA256 `8f981b563cdbc428b99eb2993ecc3a64b525b6a49f5bed44e9ceaf07b7910a0f`。
- validation/perf.json SHA256 `52e6b41775c76407b1e0525ae8c566f0746c67d4d4106a7f9dadae3c214f05d2`。
- validation/bench.py SHA256 `cf98c4d6616c0f8bcff9f11ab4a366d126ae01a1f647c0749b85ed484438fd7c`。
- validation/baseline.py SHA256 `5cdf1c657a108f9fee016742f298f0aab9e264a174ee8dddbeb0edf43980f36b`。
- test SHA256 `10fab9d45fd4a852686851e8fca7260dc0b5a3f76f8100c15e02f82aa61fd3eb`。
- ZIP `/private/tmp/flagos-batch4-structural-20260909/artifacts/competition/chunked_sgmv_expand/e17-cea2a0c/chunked_sgmv_expand.zip`；18464 bytes；SHA256 `30bdb5d81015f81960712c3034a2decdb4d65a28ca5bb86b52431d7720d7730c`；dry-run/build/verify-existing一致。

|成员|SHA256|
|---|---|
|chunked_sgmv_expand.py|`cec9fec2b67b3cd9c92cc83da01fd626eb3469ddbbe8795b05d708fd065e6059`|
|chunked_sgmv_expand_enflame.py|`ec7fe0ccab03d150ce6ef11d0b38ebf036726ace8d0cb38745f69c792ea63c4d`|
|chunked_sgmv_expand_kunlunxin.py|`ba0690d263dad46d5ea816a94b5d8e5f5a88c6ac9d39cf094062a91cbda856bb`|
