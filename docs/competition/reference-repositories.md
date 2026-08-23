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
