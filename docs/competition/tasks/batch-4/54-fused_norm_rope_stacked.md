<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/fused_norm_rope_stacked -->
<!-- synced_at: 2026-09-07T11:09:08+08:00 -->

# fused_norm_rope_stacked (speculative/fused_norm_rope_stacked)

## 任务描述

融合 RMSNorm 与旋转位置编码（RoPE）的堆叠 KV 计算：对多层堆叠的 KV 张量，先对 K 部分逐层进行 RMSNorm 归一化，再对归一化后的 K 施加 RoPE 旋转编码，最后将 K 和 V 分别从 `[T, L, H, D]` 转置为 `[L, T, H, D]` 格式输出。

## 接口签名

```python
def fused_norm_rope_stacked(kv, k_norm_weight, eps, cos_sin_cache, positions, num_kv_heads, head_dim, rotary_dim)
```

> 选手实现的函数签名需与上述 `fused_norm_rope_stacked(...)` 完全一致。

## 计算定义

输入说明：
- `kv`: `[T, L, H*D*2]` 张量，`T` 为序列长度，`L` 为层数，最后一维前半为 K、后半为 V（各占 `H*D`）
- `k_norm_weight`: `[L, D]` 或可广播为 `[1, L, 1, D]` 的 float 张量，RMSNorm 可学习权重
- `eps`: `[L]` 或可广播为 `[1, L, 1, 1]` 的 float 张量，RMSNorm 各层的稳定项 ε
- `cos_sin_cache`: `[max_pos, rotary_dim]` float 张量，前半列为 cos 值，后半列为 sin 值
- `positions`: `[T]` int 张量，每个 token 的位置索引
- `num_kv_heads`: int，KV 头数 `H`
- `head_dim`: int，每头维度 `D`
- `rotary_dim`: int，参与旋转的维度数（`rotary_dim <= D`，取 K 的前 `rotary_dim` 维做旋转）

输出：
- `k_out`: `[L, T, H, D]` 张量，经 RMSNorm + RoPE 后的 K，dtype 与 `kv` 一致
- `v_out`: `[L, T, H, D]` 张量，直接转置后的 V，dtype 与 `kv` 一致

计算步骤（以 float32 精度进行）：

**步骤 1：分离 K 和 V**

$$K = kv[ldots,; :H cdot D].reshape(T, L, H, D)$$
$$V = kv[ldots,; H cdot D:].reshape(T, L, H, D)$$

**步骤 2：对 K 做逐层 RMSNorm**

对每层 $l$，每头 $h$，每个 token $t$：

$$	ext{inv\_rms}_{t,l,h} = left(frac{1}{D}sum_{d=0}^{D-1} K_{t,l,h,d}^2 + varepsilon_light)^{-1/2}$$

$$K^{	ext{norm}}_{t,l,h,d} = K_{t,l,h,d} cdot 	ext{inv\_rms}_{t,l,h} cdot w_{l,d}$$

其中 $w_{l,d}$ 为 `k_norm_weight[l, d]`。

**步骤 3：对归一化后的 K 施加 RoPE**

设 $r = 	ext{rotary\_dim} / 2$，对每个 token $t$ 取位置 $p = 	ext{positions}[t]$：

$$cos_d = 	ext{cos\_sin\_cache}[p,; d], quad d in [0, r)$$
$$sin_d = 	ext{cos\_sin\_cache}[p,; r + d], quad d in [0, r)$$

旋转（仅作用于每头前 `rotary_dim` 维）：

$$k^{	ext{rot}}_{t,l,h,d} = K^{	ext{norm}}_{t,l,h,d} cdot cos_d - K^{	ext{norm}}_{t,l,h,r+d} cdot sin_d, quad d in [0, r)$$
$$k^{	ext{rot}}_{t,l,h,r+d} = K^{	ext{norm}}_{t,l,h,r+d} cdot cos_d + K^{	ext{norm}}_{t,l,h,d} cdot sin_d, quad d in [0, r)$$

`rotary_dim` 之后的维度保持不变。

**步骤 4：转置输出**

$$k\_out = K^{	ext{rot}}.permute(1, 0, 2, 3).contiguous().to(	ext{kv.dtype})$$
$$v\_out = V.permute(1, 0, 2, 3).contiguous()$$

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`

## 参考实现

```python
def reference(kv, k_norm_weight, eps, cos_sin_cache, positions, num_kv_heads, head_dim, rotary_dim):
    T, L, _ = kv.shape
    H, D = num_kv_heads, head_dim
    kv_size = H * D
    half_rd = rotary_dim // 2

    k_all = kv[..., :kv_size].float().view(T, L, H, D)
    v_all = kv[..., kv_size:].view(T, L, H, D)

    w = k_norm_weight.float().view(1, L, 1, D)
    eps_l = eps.float().view(1, L, 1, 1)
    inv_rms = (k_all.pow(2).mean(dim=-1, keepdim=True) + eps_l).rsqrt()
    k_normed = k_all * inv_rms * w

    pos = positions.long()
    cos = cos_sin_cache[pos, :half_rd].float().view(T, 1, 1, half_rd)
    sin = cos_sin_cache[pos, half_rd:rotary_dim].float().view(T, 1, 1, half_rd)

    k1 = k_normed[..., :half_rd]
    k2 = k_normed[..., half_rd:rotary_dim]
    rot1 = k1 * cos - k2 * sin
    rot2 = k2 * cos + k1 * sin

    k_out = k_normed.clone()
    k_out[..., :half_rd] = rot1
    k_out[..., half_rd:rotary_dim] = rot2

    k_out = k_out.permute(1, 0, 2, 3).contiguous().to(kv.dtype)
    v_out = v_all.permute(1, 0, 2, 3).contiguous()
    return k_out, v_out
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
