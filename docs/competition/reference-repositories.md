# 参考仓库与本地 Git 引用

## 已落到当前工作区

| 本地引用 | 来源 | 固定版本 | 用途 |
| --- | --- | --- | --- |
| `master` / `origin/master` | [`flagos-ai/FlagGems-sglang`](https://github.com/flagos-ai/FlagGems-sglang) | `3946b9a` | 官方当前开发基线 |
| `origin/flagos-sglang-batch1` | 同上 | `28d7eb7` | 第一批 7 题的题面、reference、正确性和 benchmark harness |
| `origin/pr31` | [官方 PR #31](https://github.com/flagos-ai/FlagGems-sglang/pull/31) | `3f5aa55` | 第二批 Task 14 `context_attention` 的公开候选、测试和 benchmark |
| `community/master` | [`AizanSousuke/FlagGems-sglang`](https://github.com/AizanSousuke/FlagGems-sglang) | `0e8023d` | Task 10/14/15/16 的公开实验实现及多芯片结果笔记 |

直接检索任意引用，不必切换当前工作树：

```bash
git grep -n "CORRECTNESS_CASES\|BENCH_CASES" origin/flagos-sglang-batch1
git show origin/pr31:src/flaggems_sglang/ops/context_attention.py
git show community/master:whd2/solution_notes.md
git show community/master:whd3/decode_attention_opt.py
git grep -n "def chunk_cumsum" community/master
```

如果重新克隆仓库，可恢复两个额外引用：

```bash
git fetch origin pull/31/head:refs/remotes/origin/pr31
git remote add community https://github.com/AizanSousuke/FlagGems-sglang.git
git remote set-url --push community DISABLED
git fetch community '+refs/heads/*:refs/remotes/community/*'
```

## 上游精确实现（固定 commit）

为避免把两个大型仓库重复克隆进工作区，以下使用 immutable 链接；确定主攻算子后再按需复制单个实现。

### SGLang `8014d9d`

- [`softcap_out`](https://github.com/sgl-project/sglang/blob/8014d9d062c3cc5d393596ecdf2f7009191965df/python/sglang/kernels/ops/activation/softcap.py#L30-L68)
- [`moe_sum_reduce`](https://github.com/sgl-project/sglang/blob/8014d9d062c3cc5d393596ecdf2f7009191965df/python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py#L1163-L1249)
- [`fused_rmsnorm`](https://github.com/sgl-project/sglang/blob/8014d9d062c3cc5d393596ecdf2f7009191965df/python/sglang/kernels/ops/elementwise/elementwise.py#L139-L188)
- [`chunk_local_cumsum_vector`](https://github.com/sgl-project/sglang/blob/8014d9d062c3cc5d393596ecdf2f7009191965df/python/sglang/kernels/ops/attention/fla/cumsum.py#L81-L248)

### FlagGems `ed2508b`

- [跨后端稳定 tanh 写法](https://github.com/flagos-ai/FlagGems/blob/ed2508bcb5a03000e9774734201d840ba362cd11/src/flag_gems/utils/triton_lang_helper.py#L72-L74)
- [通用 `moe_sum`](https://github.com/flagos-ai/FlagGems/blob/ed2508bcb5a03000e9774734201d840ba362cd11/src/flag_gems/fused/moe_sum.py#L24-L90)
- [通用 `_fused_rms_norm`](https://github.com/flagos-ai/FlagGems/blob/ed2508bcb5a03000e9774734201d840ba362cd11/src/flag_gems/ops/_fused_rms_norm.py#L31-L67)
- [通用 vector cumsum](https://github.com/flagos-ai/FlagGems/blob/ed2508bcb5a03000e9774734201d840ba362cd11/src/flag_gems/fused/FLA/cumsum.py#L83-L240)
- [Ascend scalar cumsum 特化](https://github.com/flagos-ai/FlagGems/blob/ed2508bcb5a03000e9774734201d840ba362cd11/src/flag_gems/runtime/backend/_ascend/fla/cumsum.py)

## 重要基线提醒

比赛页给出的 [示例 commit `9642557`](https://github.com/flagos-ai/FlagGems-sglang/commit/9642557dabcd277dabdb8abd09d1bb42e0af3b6b) 从 `49e6ec3` 分叉，不是当前 `master` 的祖先。它适合参考目录与 PR 写法，不适合直接作为第二批分支 base，否则 PR 会夹带无关提交。

## 第三批参考检索(2026-09-02)

- 官方仓库尚无第三批 PR(最新 #39/#40/#41 为第一批 Track1,c2flowDS,
  2026-09-01;已落盘 `origin/public/pr39/40/41`)。截止前每小时复查
  `gh api repos/flagos-ai/FlagGems-sglang/pulls`。
- c2flow 惯用法摘要(pr40 `mrope_fused`,rope 家族):
  - generic 每 token 一个 program + [BLOCK_HEADS, BLOCK_PAIRS] 2D 瓦片,
    cos/sin 每行一次载入广播(= 我方 T35 的 +24.8% 经验);
  - `_kunlunxin`:**3D grid (tokens, head_groups, pairs) + num_warps=1
    极小 program**——昆仑上微 program 大 grid 可行,未触发 1830s 崩溃族;
  - `_ascend`:直接 2D 广播瓦片可通过(我方 T35 的"2D 广播瓦片 NaN"
    应表述为特定 tile 形态/lowering 触发,非 2D 本身);
  - pr41 `_kunlunxin` fused_moe_gemm:FlagTree XPU 规则 GEMM +
    `do_not_specialize=["M"]` + `tl.max_contiguous(tl.multiple_of(...))`
    对齐提示——与 T28 route/materialize 范式互证。
- 上游 SGLang/vLLM 对第三批弱项:
  - T33 quant:sgl#33533(乘数 1 ULP 舍入改变 fp8 tie-breaking,与我方
    div_rn 教训同源;1.23x 主体在 EP 融合路径)、sgl#32296(消毒 clamp
    减半,CUDA 层)、vllm#46541(Hopper 特定)——无可直接迁移结构;
    EvokeAgent T33/T34 同分之谜上游无解。
  - T39:vllm#32735 masked-m fused silu+mul+quant,program=(expert,group)
    + counts 早退——印证我方 E10 块跳过为业界正统;sgl#29643 DSv4 masked
    布局参考。
  - T30:上游仅 diffusion/NPU fused rope,无通用参考。
- 社区仓库(AizanSousuke)停在 batch-2,无三批内容。
