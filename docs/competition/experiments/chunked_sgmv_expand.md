# Task 47 `chunked_sgmv_expand` 实验记录

```current
task: 47
operator: chunked_sgmv_expand
batch: 4
validity: invalid
platform: 7/8(e3,昆仑四投 conclusive 封轴;七芯均值~29x)
team_best_stage: e1
team_best_commit: 663286c2399cac11ece86c7fa74fc2cd638143c5
team_best_speedup: -
sealed: no
next: 昆仑 conclusive 封轴(元数据型/i32型/规则GEMM三结构均败,且GEMM bug修复后仍败=芯片后端独立问题);守榜;可选华为/沐曦冲分
updated: 2026-09-04
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
