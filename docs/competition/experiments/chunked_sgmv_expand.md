# Task 47 `chunked_sgmv_expand` 实验记录

```current
task: 47
operator: chunked_sgmv_expand
batch: 4
validity: candidate
platform: none(未提交)
team_best_stage: s0
team_best_commit: d7d8c4793278062f55617693585ae2ab89c8fbcc
team_best_speedup: -
sealed: no
next: 额度可用时打包 preflight 首投;观察燧原/昆仑/天数 vendor 信号(预案已备)
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
