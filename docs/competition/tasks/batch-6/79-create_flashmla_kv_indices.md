<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/create_flashmla_kv_indices -->
<!-- synced_at: 2026-09-19T23:20:17+08:00 -->

# create_flashmla_kv_indices (kvcache/create_flashmla_kv_indices)

## 任务描述

构建 FlashMLA block table：不按 token 输出，而是每 page 一个 entry，读每个 page 边界处的 token slot 并除以 page size。

## 接口签名

```python
def reference(req_to_token, req_pool_indices, page_kernel_lens, kv_start_idx, kv_indices, page_size)
```

> 选手实现的函数签名需与上述完全一致。

## 计算定义

- 对 page p（`p < ceil(len_i / page_size)`）：`slot = req_to_token[req_pool_indices[i], kv_start_i + p * page_size]`，`kv_indices[i, p] = slot // page_size`
- `kv_indices` 是 2D `[bs, max_pages]` int32 block table，原地写入

## 正确性判别标准

exact（整数运算）

## 参考实现

```python
def reference(
    req_to_token, req_pool_indices, page_kernel_lens, kv_start_idx, kv_indices, page_size
):
    out = kv_indices.clone()
    for i in range(req_pool_indices.shape[0]):
        n = int(page_kernel_lens[i])
        start = int(kv_start_idx[i]) if kv_start_idx is not None else 0
        pool = int(req_pool_indices[i])
        num_pages = (n + page_size - 1) // page_size
        slots = req_to_token[pool, start : start + num_pages * page_size : page_size]
        out[i, :num_pages] = slots // page_size
    return out
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
