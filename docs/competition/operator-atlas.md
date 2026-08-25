<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

# 算子功能与原理图谱（已发布 24 题）

> 整理时间：2026-08-25。全部签名、公式、shape、容差均逐条取自
> `docs/competition/tasks/` 下的本地题面（同步于
> `2026-08-25T00:19:05+08:00`），不是记忆或推测。
>
> **发布范围说明**：赛制宣传"约 200 道算子、分 13 批"是**计划总量**；
> `data/race-overview.json` 的实时统计为 `total_task_count = 24`、
> `released_task_count = 24`。后续批次题面尚未公开，其算子名、签名和语义
> 无法获取，本文只覆盖已发布的 24 题。
>
> **芯片数差异**：第一批 7 题为 **7 芯**，第二批 17 题为 **8 芯**。
> 全部 24 题的最低加速比门槛均为 `0.1x`。

## 目录

- [一、Pointwise 与位掩码](#一pointwise-与位掩码)
- [二、归一化与 Reduction](#二归一化与-reduction)
- [三、Scan / 前缀和](#三scan--前缀和)
- [四、SSM 状态与递推](#四ssm-状态与递推)
- [五、GEMM 与分段矩阵乘](#五gemm-与分段矩阵乘)
- [六、Attention](#六attention)
- [七、位置编码与布局变换](#七位置编码与布局变换)
- [八、题型与跨芯风险对照](#八题型与跨芯风险对照)
- [九、易误读的契约点](#九易误读的契约点首投前逐条核对)
- [十、学习与复用路径](#十学习与复用路径)

---

## 一、Pointwise 与位掩码

结构最简单的一类：输出元素只依赖同位置输入，无跨元素依赖。骨架为 flat 1D
indexing + tail mask，纯 memory-bound，瓶颈在访存带宽与向量化宽度。

### T24 `softcap_out`（activation_norm）

```python
def reference(x, softcap_const)
```

`output = tanh(x / softcap_const) * softcap_const`，把无界 logits 压进
`[-cap, cap]`。全程 float32 计算，**输出保持 float32**（不 cast 回输入
dtype，这点与多数题不同）。

**原理**：Gemma-2 等模型用它替代硬 clip 防止 softmax 前数值溢出，处处可导。

**难点**：`softcap_const` 极小时 `x / cap` 会溢出——本仓库 generic 用
`CAP_RECIPROCAL_OVERFLOWS` 常量参数在编译期分支处理（阈值
`0x1p-128`）。`tanh` 需用跨后端稳定写法（见 FlagGems 固定源码）。

容差：fp32 `1e-4`、bf16 `1.5e-2`、fp16 `1e-2`。

### T08 `apply_token_bitmask`（sampling_grammar）

```python
def reference(logits, bitmask)
```

- `logits`：`[B, V]` 浮点
- `bitmask`：`[B, ceil(V/32)]` **int32**，每 bit 对应一个 token
- 第 `v` 个 token：`word_idx = v // 32`、`bit_idx = v % 32`
- 若 `(bitmask[:, word_idx] >> bit_idx) & 1 == 0` 则 `logits[:, v] = -inf`
- 输出与 `logits` 同 shape 同 dtype

**原理**：约束解码 / 语法采样。强制模型只能输出合法 token（如合法 JSON、
符合 EBNF 的串），做法是把非法 token 的 logits 设为 `-inf`，softmax 后概率
归零。bitmask 压位是为了让 `V=152064` 的掩码只占 `V/32` 个 int32。

**难点**：计算量极低（仅位运算），完全 memory-bound；vocab 维极大导致
`grid.x` 可达 2,433,024，**这正是燧原/华为 65535 启动上限的触发点**。

### T07 `silu_and_mul`（activation_norm，批次 1）

```python
def reference(hidden_states)
```

`d = shape[-1] // 2`，`out = silu(x[..., :d]) * x[..., d:]`。fp32 计算后
cast 回输入 dtype。要求最后一维为偶数。量化路径（`quantize=`/`scales=`）
明确不在范围内。

**原理**：SwiGLU 的核心。FFN 把输入投影成 gate 和 up 两半，`silu(gate)`
作为门控乘到 up 上。读两份写一份，算术强度低。

---

## 二、归一化与 Reduction

一行（或一组）元素归约成少数统计量再回写。核心是 FP32 累加 + 行内
reduction；寄存器压力随 hidden size 增长，是 BLOCK 选择的主约束。

### T19 `fused_rmsnorm`（activation_norm）

```python
def reference(x, weight, eps)
```

- `rms = sqrt(mean(x^2, dim=-1) + eps)`
- `out = (x / rms) * weight`
- 中间 float32，输出 cast 回输入 dtype

**原理**：RMSNorm 相比 LayerNorm 省掉减均值（不做中心化），只用二阶矩缩放，
是 LLaMA 系列标准归一化。融合点在于平方和、rsqrt、乘权重一趟完成，避免中间
张量落显存。

**难点**：hidden size 决定 BLOCK 与寄存器压力。本仓库 generic 用
`next_power_of_2(hidden_size)` 作 BLOCK、`num_warps=8`。

### T20 `mamba_layernorm_gated`（mamba）

```python
def reference(x, weight, bias, eps, z=None, group_size=None,
              norm_before_gate=True, is_rms_norm=True)
```

- 按 `group_size` 分组：`x.view(M, ngroups, group_size)`
- `norm_before_gate=False` → **先**门控：`x = x * z * sigmoid(z)`
- `is_rms_norm=True` → `x_hat = x * rsqrt(mean(x^2) + eps)`；
  否则 LayerNorm：`x_hat = (x - mean) * rsqrt(var + eps)`
- 应用 `weight`（及可选 `bias`）
- `norm_before_gate=True` → **后**门控：`y = y * z * sigmoid(z)`
- 输出 `[M, N]`，cast 回输入 dtype

**原理**：`z * sigmoid(z)` 即 SiLU。Mamba/SSM 用门控控制信息流；分组归一化
让统计量只在 `group_size` 内计算，不跨整个 hidden，等价于多头结构。

**难点**：四种组合（RMS/LN × 前门控/后门控）都必须正确，`bias` 和 `z` 均可
为 None。BLOCK 由 `group_size` 数据驱动（`next_power_of_2(group_size)`），
不能简单硬设常量——这是燧原大 BLOCK 配方在此题不能直接套用的原因。

### T21 `moe_sum_reduce`（moe）

```python
def reference(input, routed_scaling_factor)
```

- `input`：`[num_tokens, top_k, hidden_dim]`
- `output = input.sum(dim=1) * routed_scaling_factor`
- float32 累加，cast 回输入 dtype

**原理**：MoE 中每个 token 被路由到 top-k 个专家，此步把 k 份专家输出求和
并乘全局缩放因子。注意 `routed_scaling_factor` 是**标量**，不是 per-token
权重（与 T03 不同）。

**难点**：纯 bandwidth-bound，读 k 倍数据写 1 份，算术强度极低。
`num_tokens × cdiv(hidden, BLOCK)` 的 2D grid 在昆仑展平后可达 114,688，
超 65535 上限——本仓库靠 BLOCK 256→1024 压回。

### T04 `merge_state`（attention，批次 1）

```python
def reference(prefix_output, prefix_lse, suffix_output, suffix_lse)
```

- `max_lse = max(prefix_lse, suffix_lse)`（逐元素，`inf` 按 `-inf` 处理）
- `p = exp(prefix_lse - max_lse)`、`s = exp(suffix_lse - max_lse)`
- `output = (prefix_output * p + suffix_output * s) / (p + s)`
- `output_lse = log(p + s) + max_lse`

**原理**：flash-decoding 把长 KV 切成多段并行计算，每段得到局部输出 + 局部
log-sum-exp，此算子按 online-softmax 规则合并。数学上等价于对完整序列做一次
softmax。

**难点**：`inf` 要按 `-inf` 处理（空段语义），减 `max_lse` 是数值稳定必需。

---

## 三、Scan / 前缀和

存在跨时间步依赖，天然反并行。解法是分块：块内并行 scan，块间不传递（本组
三题都是 **chunk-local**，不跨块累积）。

### T02 `chunk_local_cumsum_scalar`（fla，批次 1）

```python
def reference(g, chunk_size, reverse=False, scale=None)
```

- reshape 为 `(B, NT, BT, H)`，`NT = T // BT`、`BT = chunk_size`
- 在块内时间轴做 cumsum；`reverse=True` 时先 flip、cumsum、再 flip 回
- `scale` 非 None 则乘该标量
- 范围：`head_first=False` 布局、`cu_seqlens=None`（定长 batching），
  `T` 恒为 `chunk_size` 的整数倍且 chunk_size 为 2 的幂

**原理**：门控线性注意力（GLA/FLA 家族）需要把逐 token 的 log-decay 值累积成
块内累计衰减，供后续 `exp(A_i - A_j)` 计算衰减权重。

### T11 `chunk_local_cumsum_vector`（fla）

```python
def reference(g, chunk_size, reverse=False, scale=None)
```

与 scalar 版唯一区别：输入 `[B, T, H, S]`，最后一维是**向量**（`S > 1`），
cumsum 在 `dim=2`（块内时间维）。输出同 shape。

**难点**：多一个 `S` 维意味着每个 program 的工作量放大 S 倍，tile 形状选择
与 scalar 版不同。本仓库候选发现 tiny chunk 用 2 warps 有 1.032–1.162x 收益。

### T10 `chunk_cumsum`（mamba）

```python
def reference(dt, A, chunk_size, dt_bias=None, dt_softplus=False)
```

- `dt`：`[batch, seqlen, nheads]` 时间步长
- `A`：`[nheads]` 衰减系数（**负值**）
- 可选 `dt_bias` 相加、可选 `dt_softplus` 做 softplus
- `dt` 经 `clamp(min=0)` 后 reshape 为 `[batch, nheads, nchunks, chunk_size]`
- `dA = dt * A`，在 chunk_size 维 cumsum
- **返回两个张量** `(dt_out, dA_cumsum)`，shape 均为
  `[batch, nheads, nchunks, chunk_size]`

**原理**：Mamba 的离散化步骤。连续 SSM 参数 `A` 经时间步 `dt` 离散化为
`exp(dt·A)`，此算子算出指数的累积形式。`softplus` 保证 `dt > 0`，
`clamp(min=0)` 是数值保护。

**难点**：**双输出**、参数组合多（bias × softplus 四种），且有 reshape 带来的
非连续访存。本仓库 E1–E4 四个性能候选全部被拒绝，是已知的难优化题。

---

## 四、SSM 状态与递推

在 scan 之上引入状态矩阵累积，用 `tl.dot` 做外积。片上存储需求最高。

### T12 `chunk_state`（mamba）

```python
def reference(B, x, dt, dA_cumsum)
```

- `B`（SSM 矩阵）：`[batch, seqlen, ngroups, dstate]`
- `x`：`[batch, seqlen, nheads, headdim]`
- `dt`、`dA_cumsum`：`[batch, nheads, nchunks, chunk_size]`
- `decay = exp(dA_cumsum[..., -1:] - dA_cumsum)`
- `scale = decay * dt`
- `states = einsum("bcthp,bcthn->bchpn", x_c, B_scaled)`
- 输出 `[batch, nchunks, nheads, headdim, dstate]`

**原理**：Mamba 块内隐状态。在 chunk 内把每个时刻的 `x_t ⊗ B_t` 外积按衰减
权重累加，得到该 chunk 结束时的状态。`dA_last - dA_t` 保证衰减对齐到块尾
（越早的 token 衰减越多）。

**难点**：题面显式警告参数 `B` 是 SSM 状态投影矩阵、**不是 batch size**。
`ngroups` 与 `nheads` 是独立维度（题面未声明二者相等，Mamba 结构上 group 可被
多个 head 共享，但这属于结构推断、非题面事实），head→group 映射需按实际
shape 推导。容差 **`3e-2`（宽）**，为低精度 dot 留出空间。

### T13 `chunk_state_varlen`（mamba）

```python
def reference(B, x, dt, dA_cumsum, cu_seqlens, chunk_states)
```

- packed 变长格式（无 batch 维，序列拼接）
- 对每条序列，**只计算其最后一个 chunk** 的 SSM 隐状态
- `scale = exp(dA_last - dA_seg) * dt_seg`
- `states[bidx, h] = x_seg.T @ (B_seg * scale)`
- `chunk_states` 参数**仅用于确定输出 dtype**
- 输出 `[batch, nheads, headdim, dstate]`

**难点**：变长边界（`cu_seqlens` 索引）、"仅最后一个 chunk"的语义容易实现成
全部 chunk。`chunk_states` 是纯 dtype 载体这一点极易误读为参与计算。容差
`3e-2`，但本仓库低精度 dot 候选曾因确定性反例 `0.0625 > 0.03` 被拒绝——
说明误差来自**累加**而非输入精度。

### T18 `fused_recurrent_gdn`（fla）

```python
def reference(q, k, v, g, beta, scale, initial_state,
              output_final_state, use_qk_l2norm_in_kernel=False)
```

逐时间步 `t = 0..T-1` 串行递推：

1. `state = state * exp(gate)` — 指数衰减
2. `pred = einsum("bhvk,bhk->bhv", state, k_t)` — 读出预测
3. `correction = (v_t - pred) * beta` — delta 修正
4. `state = state + correction[:,:,None] * k_t[:,None,:]` — 外积更新
5. `output_t = einsum("bhvk,bhk->bhv", state, q_t)` — 读出输出

可选 q/k 的 L2 norm、可选 `initial_state`。输出
`(output [B,T,HV,V], final_state or None)`。

**原理**：Gated DeltaNet。DeltaNet 的核心是"先预测再修正"——用当前 key 查询
状态得到预测值，与真实 value 的差乘 `beta` 作为修正量写回状态。这是线性
注意力的 delta rule 形式，等价于在线学习一个关联记忆。

**难点**：**全赛道最难**。真串行递推（无法分块并行），状态矩阵 `[V, K]` 常驻
片上，需求随 head_dim 平方增长。截至 08-25 **19 队全部失败、0 队达标**，
状态为 `pending_challenge`。燧原 gcu500 的 `max_dsm` 仅 312 KiB，此题极易
资源超限。

### T01 `causal_conv1d_fn`（mamba，批次 1）

```python
def reference(x, weight, bias, query_start_loc, seq_lens_cpu,
              activation="silu")
```

- 序列 `i` 占 `x` 的列 `query_start_loc[i]:query_start_loc[i+1]`
  （左右拼接，**不跨序列串味**）
- 每序列、每通道 `d`、每位置 `t`：
  `conv[d,t] = sum_{k=0}^{width-1} weight[d,k] * x_pad[d, t+k]`，
  `x_pad` 是该序列该通道行**左填 `width-1` 个零**（因果，不看未来也不回看
  上一条序列）
- `out = conv + bias`（若给），再 `silu` 若 `activation in {"silu","swish"}`
- 范围：仅 fresh prefill（无 cache、无 `conv_states`、无 `cache_indices`）

**原理**：Mamba 在 SSM 前用深度可分离因果卷积做局部 token 混合，
`width` 通常为 4。深度可分离 = 每通道独立卷积，无跨通道混合。

**难点**：continuous batching 的序列边界必须严格隔离；左 padding 与因果性。

---

## 五、GEMM 与分段矩阵乘

用 `tl.dot` 的 M/N/K tiling + FP32 累加器。本仓库全部 dot 调用都显式写了
`input_precision="ieee"`（`ieee` 是唯一八芯全支持的精度）。

### T09 `bmm_chunk`（mamba）

```python
def reference(a, b, chunk_size, causal=False)
```

- 输入 `a`、`b`：`[batch, seqlen, ngroups, k]`
- 切块：`a_c = a.reshape(batch, nchunks, chunk_size, ngroups, k)`
- `out = einsum("bcigk,bcjgk->bcgij", a_c, b_c)`
- 输出 `[batch, nchunks, ngroups, chunk_size, chunk_size]`
- 全程 float32

**原理**：Mamba chunk 内的 `C @ B^T`，产出 chunk-local 的 token×token 注意力
式矩阵。`causal` 参数控制是否只保留下三角。

**难点**：输出是 `chunk_size × chunk_size` 方阵，随 chunk_size 平方膨胀。
容差 fp32 `1e-4` 严、低精度 `1.5e-2` 宽——**dtype 分支化精度的机会**。

### T03 `fused_moe_gemm`（moe，批次 1）

```python
def reference(A, B, topk_weights, topk_ids, top_k)
```

`out[t, j] = (A[t] @ B[topk_ids[t,j]].T) * topk_weights[t, j]`

题面明确说明：baseline 的实际 kernel 还需要
`sorted_token_ids`/`expert_ids`/`num_tokens_post_padded`（来自
`moe_align_block_size`）来按专家分组建 grid，但那是**调度元数据、不影响输出
布局**（kernel 通过 `offs_token` 写回各 token 自己的行，不是排序后的位置），
所以 `baseline.py` 内部自算，不属于逻辑契约。

容差 **`atol=0.5, rtol=1e-2`**（全赛道最宽的 atol）。

### T17 `embedding_lora_a`（lora）

```python
def reference(input_ids, weights, batch_info, vocab_size,
              extra_embeddings=None)
```

对每个 segment（`batch_info` 指定的连续 token 区间）：

1. 取 adapter weight index 和 rank
2. embedding lookup：`out[rows, :r] = weights[w_idx, :r, tokens].T`
3. 若有 `extra_embeddings` 且 `token >= vocab_size`，用 extra embedding 替换

输出 `[S, rank]`，与 weights 同 dtype。

**原理**：LoRA 把权重更新分解为低秩两矩阵 `A(d×r)`、`B(r×d)`，`r` 通常
8–64。多适配器服务场景下按 segment 路由到不同 adapter。这是 A 阶段
（降维 + gather）。

**难点**：**空 segment**（本仓库曾因空段 metadata 越界修 bug）、
`token >= vocab_size` 的 extra embedding 分支、注意 weights 布局是
`[w_idx, r, vocab]` 需转置。

### T23 `sgemm_lora_b`（lora）

```python
def reference(x, weights, batch_info, base_output)
```

对每个 segment：取 x 对应行（**支持 permutation 重排**），
`out[rows] += scaling * (x_seg @ W.T)`。`base_output` 是已有 dense 输出，
LoRA 做**增量叠加**。float32 计算，cast 回 `base_output` 的 dtype。

### T22 `qkv_lora_b`（lora）

```python
def reference(x, qkv_lora_b, batch_info, output_offset,
              max_qkv_out_dim, base_output)
```

比 T23 多一层 slice：按 `output_offset` 分 `n_slices`（对应 Q/K/V 三个
投影），每 slice 做
`out[rows, o_start:o_end] += scaling * (x_slice @ W_slice.T)`。

**难点**：Q/K/V 三个 slice 的输出宽度可能不同（GQA 下 K/V 更窄），
`max_qkv_out_dim` 用于统一分配。本仓库曾修复空段越界并跳过窄 slice 的无效
GEMM。ragged M（每段行数不同）导致 tile 利用率波动。

---

## 六、Attention

### T14 `context_attention`（attention）

```python
def reference(q, k, v, b_start_loc, b_seq_len, max_input_len, is_causal)
```

- layout：`[total_tokens, num_heads, head_dim]`（packed）
- 每条序列 `i` 取 `q/k/v[start:end]`
- 调 scaled dot-product attention（含可选 causal mask）
- 输出 shape 同 `q`，**dtype 为 float32**
- `max_input_len` 仅为 kernel 实现预留（分配临时空间），reference 未使用

**原理**：prefill 阶段注意力。用 online softmax（Flash Attention）分块遍历
KV，维护 running max 与 running sum，避免物化 `S×S` 矩阵。

**难点**：shared memory 随 head_dim 增长。本仓库记录 `D=513/1024` 需
132,160 bytes shared，**超 NVIDIA 101,376 上限**；燧原 gcu500 DSM 仅
312 KiB，风险更高。全赛道仅 1 队达标。

### T15 `decode_attention`（attention）

```python
def reference(q, k_buffer, v_buffer, kv_indptr, kv_indices, sm_scale)
```

对每个 batch 元素 `b`：

1. 用 `kv_indptr[b]:kv_indptr[b+1]` 和 `kv_indices` 索引取 KV 序列
2. `logits = q[b] @ K.T * sm_scale`（float32）
3. `logits -= logits.max()`（数值稳定）
4. `p = softmax(logits)`
5. `o[b] = p @ V`

输出 `[B, H_Q, D_v]`，dtype float32。本题 `H_Q == H_KV`（MHA）。

**原理**：decode 阶段 query 长度恒为 1，计算特征与 prefill 完全不同——极度
memory-bound，瓶颈是读整个 KV cache。`kv_indices` 是 paged KV 的间址表
（KV 分页存储，避免变长带来的显存碎片）。

### T16 `decode_grouped_attention`（attention）

签名与 T15 完全相同。区别：`H_KV < H_Q`，KV heads 通过
`repeat_interleave(group_size)` 扩展对齐，`group_size = H_Q // H_KV`。

**优化点**：共享同一 KV head 的多个 Q head 可打包进同一 program 复用 KV
载入——本仓库该手段拿到 1.326–1.845x。

---

## 七、位置编码与布局变换

### T05 `mrope_fused`（rope，批次 1）

```python
def reference(q, k, cos_sin_cache, positions, mrope_section,
              head_size, rotary_dim)
```

- `cos_sin_cache[p, :rd//2]` 是 cos，`cos_sin_cache[p, rd//2:rd]` 是 sin
- 每 token `i` 拼一对 `(rd//2,)` 的 cos/sin：
  索引 `< mrope_section[0]` 取 `positions[0,i]`；
  `mrope_section[0] <= idx < mrope_section[0]+mrope_section[1]` 取
  `positions[1,i]`；其余取 `positions[2,i]`
- 每 head（Q、K 各自独立）把前 `rotary_dim` 个通道分半 `x1,x2`，旋转为
  `out1 = x1*cos - x2*sin`、`out2 = x2*cos + x1*sin`；
  超出 `rotary_dim` 的通道（partial rotary）原样透传
- 范围：non-interleaved、neox 风格（rotate-half）、无 axis map
- **in-place** 修改 Q 和 K

**原理**：多模态 RoPE。RoPE 把位置编码为旋转相位；mrope 为
时间/高度/宽度三个维度分配不同频率段（Qwen-VL 系列），使图像/视频的
二维空间位置和时间位置能共存于同一 rotary 维度。

**难点**：三段式 position 选择、partial rotary 透传、in-place 语义。

### T06 `per_group_transpose`（quantization，批次 1）

```python
def reference(a, expert_offsets, m_alignment=1)
```

- `expert_offsets` 是累积行数数组（`[0] == 0`、`[E] == M`），
  组 `e` 拥有行 `expert_offsets[e]:expert_offsets[e+1]`
- 每组：
  `out.view(-1)[start*K : start*K + n_e*K] == a[start:end].T.contiguous().view(-1)`
- 输出与 `a` **总 shape/size 相同**——不是全局转置，只是逐组转置后
  逐组排布
- `m_alignment` 只是 baseline kernel 的编译对齐提示，**不影响输出值**

容差：**精确相等（`atol=0, rtol=0`）**。题面明确说明这是整数/字节搬运操作，
无浮点计算，必须 bit-exact。榜首 631.88x（reference 实现极差）。

---

## 八、题型与跨芯风险对照

结合 [`data/vendor-backends/`](data/vendor-backends/README.md) 缓存源码：

| 题型 | 题号 | 主瓶颈 | 弱三芯关键风险 |
| --- | --- | --- | --- |
| Pointwise | 07、08、24 | 纯带宽 | 燧原需大 BLOCK 对齐 `vector_length=2048`；大 vocab 触发 grid 上限 |
| Reduction | 04、19、20、21 | 行内 FP32 累加 | 华为 UB 容量 + 尾轴需 32B 对齐；2D grid 展平超限 |
| Scan | 02、10、11 | 块内 scan + 非连续 reshape | 双输出（T10）增加写带宽；tile 形状难调 |
| SSM 状态 | 01、12、13、18 | 状态矩阵片上存储 | 燧原 gcu500 `max_dsm` 仅 312 KiB；T18 无人达标 |
| GEMM | 03、09、17、22、23 | tile + 累加精度 | **昆仑走独立 SDNN 路径**，pointwise 经验不可外推 |
| Attention | 14、15、16 | shared/UB 容量 | 燧原**仅支持 `ieee`**，无降精度余地；大 D 易超 shared |
| 布局变换 | 05、06 | 访存重排 | T06 要求 bit-exact，无精度腾挪空间 |

**容差分布**（决定精度优化空间）：

| 容差 | 题号 | 是否容得下 tf32（~1e-3） |
| --- | --- | --- |
| `atol=0` 精确 | 06 | 否 |
| fp32 `1e-4` | 01、02、04、05、07、08、09、17、19、20、21、22、23、24 | 否（fp32 路径） |
| 低精度 `1.5e-2` / `1e-2` | 同上各题的 bf16/fp16 路径 | **是（约 15 倍余量）** |
| `1e-2` 统一 | 14、18 | 是 |
| `3e-2` 统一 | 12、13 | **是（约 30 倍余量）** |
| `3e-2`/`1e-2` | 15、16 | 是 |
| `atol=0.5` | 03 | 是（最宽） |

---

## 九、易误读的契约点（首投前逐条核对）

以下每条都是**只看算子名或凭经验推测会写错、必须回到题面才能确认**的语义。
整理来源是逐条比对题面原文与直觉描述后发现的实际偏差，不是假想风险。

| 题号 | 直觉推测 | 题面实际要求 |
| --- | --- | --- |
| T21 | top-k 加权求和，权重 per-token | `routed_scaling_factor` 是**标量**；带 per-token 权重的是 T03 |
| T13 | `chunk_states` 参与状态累积 | **仅用于确定输出 dtype**，不参与任何计算 |
| T10 | 单输出 cumsum | **返回两个张量** `(dt_out, dA_cumsum)` |
| T24 | 输出 cast 回输入 dtype | 输出**保持 float32** |
| T06 | 全局转置 | 逐组转置，**写回原扁平字节偏移**，输出总 shape 同输入 |
| T06 | `m_alignment` 影响布局 | 仅编译对齐提示，**不影响输出值** |
| T03 | 需要 `sorted_token_ids` 等入参 | 那是调度元数据，`baseline.py` 内部自算，**不属于逻辑契约** |
| T14 | 输出 dtype 跟随 `q` | 输出 shape 同 `q`，但 **dtype 为 float32** |
| T14 | `max_input_len` 参与计算 | 仅为 kernel 预留临时空间，**reference 未使用** |
| T04 | `inf` 按正无穷处理 | `inf` **按 `-inf` 处理**（空段语义） |
| T05 | 返回新张量 | **in-place** 修改 Q 和 K |
| T05 | 全部通道旋转 | 超出 `rotary_dim` 的通道**原样透传**（partial rotary） |
| T01 | 可跨序列回看 | 因果且序列边界隔离，**不跨序列**；左填 `width-1` 个零 |
| T02 | 全局 cumsum | **chunk-local**，块间不累积 |
| T20 | 只有一种门控顺序 | `norm_before_gate` 决定**前门控或后门控**，四种组合都要对 |
| T17 | weights 布局为 `[w_idx, vocab, r]` | 实际 `[w_idx, r, vocab]`，**需转置** |
| T12 | 参数 `B` 是 batch size | 题面显式警告：`B` 是 **SSM 状态投影矩阵** `[batch, seqlen, ngroups, dstate]` |

**范围限定**（题面显式声明"不在范围内"，实现时不要多做）：

- T07：量化路径（`quantize=`/`scales=`）不在范围
- T01：仅 fresh prefill，无 cache / `conv_states` / `cache_indices`
- T02：`head_first=False` 布局、`cu_seqlens=None` 定长 batching
- T05：non-interleaved、neox 风格（rotate-half）、无 axis map

---

## 十、学习与复用路径

按骨架复用度递增，每步只新增一个概念：

1. **`softcap_out`**（T24）— flat 1D、mask、fp32 特殊函数
2. **`silu_and_mul`**（T07）— 同骨架，加双输入
3. **`apply_token_bitmask`**（T08）— 同骨架，加位运算与大 vocab grid
4. **`moe_sum_reduce`**（T21）— 加小 reduction
5. **`fused_rmsnorm`**（T19）— 加行归约、rsqrt、寄存器压力
6. **`mamba_layernorm_gated`**（T20）— 加分组与参数组合分支
7. 二选一分支：
   - **LoRA 线**：`embedding_lora_a`（T17，gather+空段）→
     `sgemm_lora_b`（T23，分段 GEMM）→ `qkv_lora_b`（T22，多 slice）
   - **SSM 线**：`chunk_local_cumsum_vector`（T11）→ `chunk_cumsum`（T10，
     双输出）→ `chunk_state`（T12，加外积 dot）
8. **`bmm_chunk`**（T09）— 纯 tiling GEMM
9. **Attention**：`decode_attention`（T15）→ `decode_grouped_attention`
   （T16，加 GQA 复用）→ `context_attention`（T14，加 online softmax +
   causal）
10. **最后**：`chunk_state_varlen`（T13，变长边界）、
    `fused_recurrent_gdn`（T18，真串行递推）

T18 同时放大地址计算、数值稳定、片上存储和跨芯 launch 差异，应最后攻。
