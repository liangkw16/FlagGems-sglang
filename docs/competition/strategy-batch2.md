# 第二批算子快速开发与提交策略

> 结论基于 2026-08-23 的题面、榜单、官方仓库和公开上游实现。实时数字见 [task-index.md](task-index.md)。

## 先做什么

当前工作区没有现成的第一批 scalar cumsum 实现，只有 reference/harness。因此从“现在到首次 8 芯片正确提交”的时间看，建议顺序是：

| 优先级 | 题目 | 首次正确提交预估 | 跨芯片风险 | 理由 |
| ---: | --- | ---: | --- | --- |
| 1 | Task 24 `softcap_out` | 30–60 分钟 | 低 | 单个逐元素 kernel；输出固定 fp32；有 SGLang 精确同源 |
| 2 | Task 21 `moe_sum_reduce` | 45–90 分钟 | 低–中 | token/hidden block reduction；固定小 `top_k` |
| 3 | Task 19 `fused_rmsnorm` | 1–2 小时 | 中 | 一行一个 program；需关注大 hidden size 的本地存储 |
| 4 | Task 11 `chunk_local_cumsum_vector` | 2–4 小时 | 中–高 | 可把 `H*S` flatten 复用 scalar 思路；`tl.dot` 版本移植风险较高 |

若手头另有第一批 `chunk_local_cumsum_scalar` 的已验证参赛实现，Task 11 可升到第一：把 `[B,T,H,S]` 视为 `[B,T,H*S]`，kernel 逻辑、`reverse`、`scale` 和容差均不变，只改 wrapper 和输出 reshape。

第一波拿到平台反馈后，再按收益做：

1. Task 17 `embedding_lora_a`：上游签名/字段完全一致，当前仅 6 队、第一名约 23.06x，投入产出比最高。
2. Task 23 `sgemm_lora_b`，再复用 segment/permutation 逻辑做 Task 22 `qkv_lora_b`。
3. Task 15/16：签名和计算完全相同，一套 MHA/GQA kernel 改入口名可交两题。
4. Task 13：上游精确同源且性能空间大，但必须修正 `chunk_states` 语义。

暂避 Task 14 `context_attention` 和 Task 18 `fused_recurrent_gdn`：截至快照均没有方案在 8 款芯片上全部过门槛。官方 PR #31 的 context 候选也暴露了 MetaX shared-memory、昆仑超时和华为 grid 上限问题。

## 最短首交流程：Task 24

1. 从[官方题面](tasks/batch-2/24-softcap_out.md)锁定签名 `softcap_out(x, softcap_const)`、fp32 计算和 fp32 输出。
2. 只写一个 flat 1D Triton kernel；首版固定 `BLOCK_SIZE=256`、`num_warps=4`、`num_stages=1`。
3. 使用稳定公式 `cap * (2 * sigmoid(2 * x.float() / cap) - 1)`。不要直接复制上游 `(exp(2u)-1)/(exp(2u)+1)`，大正值会出现 `inf/inf`。
4. 不引入 autotune、`libdevice`、cache hint、PDL 或设备分支；首版只追求 8 芯片正确和 `>=0.1` 的最低门槛。
5. 本地做语法/风格检查；有任一目标芯片时加随机、极大正负值和三种 dtype 对照。
6. ZIP 中先只放 `softcap_out.py`。平台返回按芯片结果后，再决定是否增加 vendor 文件。

先按项目 skill 提交已选中的 source/test、完成 release 门禁，再把 manifest 中的完整
source commit 写入下方变量：

```bash
source_commit="SOURCE_COMMIT_FULL_SHA"
python .agents/skills/flagos-operator-race/scripts/build_submission.py \
  softcap_out --stage s0 --commit "$source_commit"
```

网页上传会消耗每日额度并产生外部提交；本地资料整理没有执行上传。实际提交遵循
[完整平台门禁](../../.agents/skills/flagos-operator-race/references/platform-workflow.md)。

## 17 题上游映射与语义陷阱

SGLang 链接固定在 commit `8014d9d062c3cc5d393596ecdf2f7009191965df`，避免后续主分支漂移。

| Task | 算子 | 最短复用源 | 必改/风险 |
| ---: | --- | --- | --- |
| 08 | `apply_token_bitmask` | [SGLang bitmask_ops.py](https://github.com/sgl-project/sglang/blob/8014d9d062c3cc5d393596ecdf2f7009191965df/python/sglang/kernels/ops/grammar/bitmask_ops.py) | 上游是 inplace 且有 indices；赛题要 out-of-place、无 indices |
| 09 | `bmm_chunk` | [Mamba v2.2.4 ssd_bmm.py](https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_bmm.py) | 限 4D grouped；忽略 `causal`；输出 fp32；fp32 dot 禁 TF32 |
| 10 | `chunk_cumsum` | [Mamba v2.2.4 ssd_chunk_state.py](https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_chunk_state.py) | 上游返回 `(dA_cumsum, dt_out)`，赛题顺序相反；本地 `community/master` 有多芯片实验 |
| 11 | `chunk_local_cumsum_vector` | [SGLang FLA cumsum.py](https://github.com/sgl-project/sglang/blob/8014d9d062c3cc5d393596ecdf2f7009191965df/python/sglang/kernels/ops/attention/fla/cumsum.py) | 优先 flatten `H*S`；题面未保证 `chunk_size` 是 2 的幂；generic vector 路径跨芯验证较少 |
| 12 | `chunk_state` | [Mamba v2.2.4 ssd_chunk_state.py](https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_chunk_state.py) | 固定保守 tile；输出 fp32 |
| 13 | `chunk_state_varlen` | 同上 `_chunk_state_varlen_kernel` | 必须删除上游 `acc += chunk_states * exp(...)`：赛题中 `chunk_states` 只决定输出 dtype |
| 14 | `context_attention` | [SGLang prefill_attention.py](https://github.com/sgl-project/sglang/blob/8014d9d062c3cc5d393596ecdf2f7009191965df/python/sglang/kernels/ops/attention/prefill_attention.py)、本地 `origin/pr31` | 0 个 8 芯片有效方案；先避开 |
| 15 | `decode_attention` | [公开 self-contained 候选](https://github.com/AizanSousuke/FlagGems-sglang/blob/0e8023da851c1a2917b628d5296d4f9e68b6ca56/whd3/decode_attention.py)、[SGLang production](https://github.com/sgl-project/sglang/blob/8014d9d062c3cc5d393596ecdf2f7009191965df/python/sglang/kernels/ops/attention/decode_attention.py) | 从公开候选删除 model/reference/demo；输出 fp32 |
| 16 | `decode_grouped_attention` | 同 Task 15 | 用 `kv_head = q_head // (H_Q // H_KV)` 同时覆盖 MHA/GQA |
| 17 | `embedding_lora_a` | [SGLang embedding_lora_a.py](https://github.com/sgl-project/sglang/blob/8014d9d062c3cc5d393596ecdf2f7009191965df/python/sglang/kernels/ops/gemm/embedding_lora_a.py) | 几乎精确同源；保留 rank=0、empty segment、extra embedding 语义 |
| 18 | `fused_recurrent_gdn` | [SGLang FLA fused_recurrent.py](https://github.com/sgl-project/sglang/blob/8014d9d062c3cc5d393596ecdf2f7009191965df/python/sglang/kernels/ops/attention/fla/fused_recurrent.py) | 去 autograd/varlen；动态 T 循环和大寄存器状态跨芯风险高；0 个 8 芯方案 |
| 19 | `fused_rmsnorm` | [SGLang elementwise.py](https://github.com/sgl-project/sglang/blob/8014d9d062c3cc5d393596ecdf2f7009191965df/python/sglang/kernels/ops/elementwise/elementwise.py#L139-L188) | 官方现有 Gemma RMSNorm 使用 `1+weight`，不能原样复用；权重乘法保持 fp32 |
| 20 | `mamba_layernorm_gated` | [SGLang FLA layernorm_gated.py](https://github.com/sgl-project/sglang/blob/8014d9d062c3cc5d393596ecdf2f7009191965df/python/sglang/kernels/ops/attention/fla/layernorm_gated.py) | 只取 1-pass forward；删 backward/SM-count/PDL/NPU import |
| 21 | `moe_sum_reduce` | [SGLang fused_moe kernel](https://github.com/sgl-project/sglang/blob/8014d9d062c3cc5d393596ecdf2f7009191965df/python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py#L1163-L1249) | 用真实 stride；TOPK constexpr；4/8 warps，首版不 autotune |
| 22 | `qkv_lora_b` | [SGLang qkv_lora_b.py](https://github.com/sgl-project/sglang/blob/8014d9d062c3cc5d393596ecdf2f7009191965df/python/sglang/kernels/ops/gemm/qkv_lora_b.py) | `n_slices=output_offset.numel()-1`；base_output 是 clone 后增量，rank0 仍保留 base |
| 23 | `sgemm_lora_b` | [SGLang sgemm_lora_b.py](https://github.com/sgl-project/sglang/blob/8014d9d062c3cc5d393596ecdf2f7009191965df/python/sglang/kernels/ops/gemm/sgemm_lora_b.py) | 先做 23，再复用 segment/permutation helper 做 22；不要原地污染 base_output |
| 24 | `softcap_out` | [SGLang softcap.py](https://github.com/sgl-project/sglang/blob/8014d9d062c3cc5d393596ecdf2f7009191965df/python/sglang/kernels/ops/activation/softcap.py#L30-L68) | 换稳定 sigmoid 公式；任何输入 dtype 都返回 fp32 |

## 跨 8 芯片首版统一约束

- `num_warps <= 8`，首版 `num_stages=1`；不复制面向 NVIDIA 的 16/32 warp autotune 表。
- 不使用 `try/except` 或设备判断回退 PyTorch。
- 先不用 `libdevice`、PDL、cache hint 和后端私有 API。
- 严格遵守题面输出 dtype；尤其 Task 9/10/11/12/14/15/16/24 为 fp32。
- fp32 GEMM/dot 路径显式禁 TF32。
- LoRA 题保留 `base_output`，不得原地污染输入。
- 隐藏 shape 未公开；先用保守固定 tile 拿到一次 8 芯反馈，再用有限提交额度做 vendor 特化。

## 提交额度的用法

每天默认 15 次，不足以对 17 题盲试。建议：

1. 首日只用 3–5 次打通一个低风险算子的全流程。
2. 一次提交只改变一个变量，保存 ZIP、commit 和逐芯片结果。
3. generic 正确后再按失败芯片补 `<operator>_<gpu>.py`，不要一开始维护 8 份实现。
4. 默认建议预留 2 次给截止日前的最终回归；用户可在看到实时额度和完整提交 tuple
   后，通过当次确认明确使用保留额度。按 2026-08-27 19:59:59 截止执行。
