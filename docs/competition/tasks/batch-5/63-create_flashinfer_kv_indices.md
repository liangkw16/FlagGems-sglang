<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/create_flashinfer_kv_indices -->
<!-- synced_at: 2026-09-10T23:30:38+08:00 -->

# create_flashinfer_kv_indices (kvcache/create_flashinfer_kv_indices)

## 任务描述

将每个请求的 token slots 从 `req_to_token` pool 扁平化到 FlashInfer paged attention 消费的 ragged `kv_indices` buffer。

## 接口签名

```python
def reference(req_to_token, req_pool_indices, page_kernel_lens, kv_indptr, kv_start_idx, kv_indices)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- 对请求 i：`beg = kv_indptr[i]`，`kv_indices[beg : beg + len_i] = req_to_token[req_pool_indices[i], kv_start_i : kv_start_i + len_i]`
- `len_i = page_kernel_lens[i]`，`kv_start_i = kv_start_idx[i]`（未提供时为 0）
- `req_to_token`: `[max_batch, max_context_len]` int32；`kv_indptr`: `[bs + 1]` int32 前缀偏移
- `kv_start_idx` 可为 `None`

## 正确性判别标准

exact（整数运算）

## 参考实现

```python
def reference(
    req_to_token, req_pool_indices, page_kernel_lens, kv_indptr, kv_start_idx, kv_indices
):
    out = kv_indices.clone()
    for i in range(req_pool_indices.shape[0]):
        beg = int(kv_indptr[i])
        n = int(page_kernel_lens[i])
        start = int(kv_start_idx[i]) if kv_start_idx is not None else 0
        pool = int(req_pool_indices[i])
        out[beg : beg + n] = req_to_token[pool, start : start + n]
    return out
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
