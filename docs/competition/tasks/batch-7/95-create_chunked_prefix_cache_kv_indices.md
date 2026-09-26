<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/create_chunked_prefix_cache_kv_indices -->
<!-- synced_at: 2026-09-26T22:45:04+08:00 -->

# create_chunked_prefix_cache_kv_indices (kvcache/create_chunked_prefix_cache_kv_indices)

## 任务描述

`create_flashinfer_kv_indices` 的 chunked-prefill 变体：每个请求只贡献其 token pool 的 `[chunk_start_idx, chunk_start_idx + chunk_seq_len)` 窗口，打包在 `chunk_cu_seq_lens[i]`。

## 接口签名

```python
def reference(req_to_token, req_pool_indices, chunk_start_idx, chunk_seq_lens, chunk_cu_seq_lens, chunk_kv_indices,)
```

> 选手实现的函数签名需与上述完全一致。

## 计算定义

- `chunk_kv_indices[cu[i] : cu[i] + n_i] = req_to_token[req_pool_indices[i], start_i : start_i + n_i]`
- `chunk_kv_indices` 原地写入；baseline 与 reference clone 后返回

## 正确性判别标准

exact（整数运算）

## 参考实现

```python
def reference(
    req_to_token,
    req_pool_indices,
    chunk_start_idx,
    chunk_seq_lens,
    chunk_cu_seq_lens,
    chunk_kv_indices,
):
    out = chunk_kv_indices.clone()
    for i in range(req_pool_indices.shape[0]):
        beg = int(chunk_cu_seq_lens[i])
        n = int(chunk_seq_lens[i])
        start = int(chunk_start_idx[i])
        pool = int(req_pool_indices[i])
        out[beg : beg + n] = req_to_token[pool, start : start + n]
    return out
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
