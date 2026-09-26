# 第七批六题契约审查 Spec（供 review 对照，来源=平台题面 docs/competition/tasks/batch-7/）

评分口径：ranking_score = Σ_芯片(1/逐芯名次)；正确性合格且全芯 S_i>=0.1 才有效。
反作弊（六题通用）：核心计算必须完全基于 Triton/Triton-TLE 真实执行；严禁 try/except、
条件分支、设备判断等方式 fallback 到 PyTorch 内置算子。

## T93 add_constant（elementwise/add_constant）
- `def reference(src, constant)`：src 为 1D 连续 CUDA int32 张量，非空；`out = src + constant`。
- 输出与输入同 shape 同 dtype；正确性判别：精确（整数运算，int32 回绕语义必须与 torch 一致）。
- baseline：C++ 模板，常量为编译期模板参数（JIT 折叠），>2^20 元素走 kMaxVecBytes 宽向量路径。

## T94 concat_mla_absorb_q（attention/concat_mla_absorb_q）
- `def reference(a, b)`：`out = torch.cat([a, b], dim=-1)`。
- a：`[dim0, dim1, a_last]` bf16；b：`[dim0, dim1, b_last]` bf16；输出 `[dim0, dim1, a_last+b_last]` bf16。
- 题面明示："两个不同 stride 的源行"——不得假设任一输入连续；正确性：精确（纯数据搬移）。

## T95 create_chunked_prefix_cache_kv_indices（kvcache/...）
- 签名 `(req_to_token, req_pool_indices, chunk_start_idx, chunk_seq_lens, chunk_cu_seq_lens, chunk_kv_indices)`。
- `chunk_kv_indices[cu[i] : cu[i]+n_i] = req_to_token[req_pool_indices[i], start_i : start_i+n_i]`；
  **reference 与 baseline 都先 `chunk_kv_indices.clone()` 再写 clone 并返回**（入参不被修改；
  clone 基底里未写入的尾部哨兵必须原样保留）。正确性：exact 整数。

## T96 fused_pack_qkv（diffusion/fused_pack_qkv）
- `(q, k, v, indices)`：q/k/v `[B,S,H,D]` 同 shape 同 dtype；indices `[total_valid]` int32/int64，
  flat B*S 位置；`q_unpad[i] = q_flat[indices[i]]` 三个输出均 `[total_valid,H,D]`。
- 正确性：exact（纯 gather 逐元素相等）；输出为新张量。

## T97 fused_sigmoid_mul（activation_norm/fused_sigmoid_mul）
- `(attn_output, gate)`：`g = gate.reshape(attn_output.shape)`；`out = attn.float() * sigmoid(g.float())`
  再 `.to(attn_output.dtype)`。两条路径：flat（同形 2D）；strided（gate 3D `[N,heads,dim]`
  可能非连续，attn 2D 连续）——内核需按 stride 读 gate，不做 contiguous 拷贝。
- 正确性：per-dtype 标准容差（fp32 计算口径）；结果 dtype=输入 dtype；非 in-place。

## T98 get_mla_kv_buffer（kvcache/get_mla_kv_buffer）
- `(kv_buffer, loc, cache_k_nope, cache_k_rope)`：`nope[i]=kv_buffer[loc[i],:nope_dim]`、
  `rope[i]=kv_buffer[loc[i],nope_dim:]`；nope_dim=cache_k_nope.shape[-1]；
  **cache_k_nope/rope 只用作 dtype/shape 模板，返回两个新张量**；目标 dtype 可与 kv_buffer 不同
  （store 隐式转换，torch `.to` 的 RTNE 语义）。正确性：exact（纯数据搬运）。

## 审查重点
1. 六个 kernel 的语义与上述契约逐条对照（含边界：空输入/零宽半边/非连续/int32 索引/i64 溢出/回绕）。
2. 反作弊红线：是否存在任何 fallback 路径或非 Triton 计算路径。
3. wrapper 契约断言是否可能与平台传入形态冲突（如平台传非连续 attn、int64 loc 等）。
4. 测试矩阵是否覆盖上述全部边界且 RELEASE_REQUIRED_TESTS 与实际方法一致。

## T100 pad_draft_extend_query（diffusion/pad_draft_extend_query）
- `(q, padded_q, seq_lens_q, cu_seqlens_q)`：q `[total,H,D]`；padded_q `[bs,max_seq,H,D]`；
  `padded_q[b,:s] = q[cu[b]:cu[b]+s]`；**reference 先 clone 再写 clone 并返回**（越 s 的行保持基底）；
  exact 纯搬移。题面明示 SGLang baseline 3D grid 早退浪费是优化目标。

## T101 post_reorder_deepgemm（moe/post_reorder_deepgemm）
- `(down_output, output, src2dst, topk_ids, topk_weights, topk, num_tokens, hidden_size, routed_scaling_factor)`：
  `output[t,:] = scaling * Σ_{i: ids[t,i]>=0} down[src2dst[t,i],:] * w[t,i]`；
  **-1=padding 跳过，ids==num_experts（fused shared expert）有效**；reference 对无效槽 dst 取 clamp(min=0)；
  返回**新张量**（output 仅作 dtype/device 模板）；per-dtype tolerance。

## T102 pre_reorder_cutlass（moe/pre_reorder_cutlass）
- `(input, gateup_input, src2dst, topk_ids, a1_scales, num_local_experts, topk, num_tokens, hidden_size)`：
  对每个 `ids[t,i] != num_local_experts` 的 slot：`gateup[dst[t,i],:] = (input[t,:] * (1/a1_scales)).to(dtype)`；
  **a1_scales 可为 None（scale=1.0）或 1 元素 fp32**；clone-first 语义（未写入行保持基底）；per-dtype tolerance。

## T104 rmsnorm_hf（norm/rmsnorm_hf）
- `(input, weight, eps)`：`y=(x.float()*rsqrt(mean(x²)+eps)).to(input.dtype)`；**`out = weight * y`**
  ——归一化结果先 cast 回输入 dtype 再乘 weight（HF 语义，区别于全 fp32 fused_rmsnorm）；
  input 2D fp16/bf16；per-dtype tolerance。

## 审查重点补充（第二批六题）
- T100/T102 的 clone-first：未写入区域必须等于基底字节（含 gap/乱序/零长度情形）。
- T101 无效槽（ids<0）：不得产生越界读（dst 可能 -1）；ids==num_experts 是有效槽。
- T102 的 a1_scales=None 路径与标量读取；slot→(t,i) 的 2D stride 寻址正确性。
- T104 的 cast 顺序（先 cast 再乘 weight）与行归约精度。

## T105 set_mla_kv_buffer（kvcache/set_mla_kv_buffer）
- `(kv_buffer, loc, cache_k_nope, cache_k_rope)`：`out[loc[i], :nd]=nope[i]`、`out[loc[i], nd:]=rope[i]`；
  **clone 后写 clone 返回**（未被写的 slot 保持基底）；loc int64；kv bf16，源 dtype 可不同（cat 后 .to(out.dtype)）；exact。

## T106 tiny_n_gemm（gemm/tiny_n_gemm）
- `(x, w, out_dtype)`：`out = (x.float() @ w.float().t()).to(out_dtype)`；x `[m,k]` bf16 **m<=16**；w `[n,k]` bf16；
  out_dtype ∈ {bf16, fp32}；per-dtype tolerance。注意 tl.dot M>=16 下限与共享内存约束。

## T107 tma_align_input_scale（quant/tma_align_input_scale）
- `(input_scale)`：2D fp32，返回值等于输入（exact），但为 column-major——pad M 到 (16/elem_size) 倍数后 buffer 的转置视图。

## T108 topk_sigmoid（moe/topk_sigmoid）
- `(topk_weights, topk_ids, gating_output, renormalize, routed_scaling_factor)`：
  **destination-passing**（原地写 fp32 weights / int32 ids 并返回）；scores=sigmoid(logits.float())，topk 降序；
  renormalize 时 `w*scale/(sum+1e-20)`；tie 不在契约内（fp32 logits 无同分行）；weights per-dtol、ids exact。

## T109 zero_experts_identity（moe/zero_experts_identity）
- `(expert_indices, expert_scales, num_experts, zero_expert_type, hidden_states)`：
  `out[t,:] = hidden[t,:] * Σ where(idx>=num_experts, scales, 0)`；hidden_dim 为 256 倍数；per-dtype tolerance。
