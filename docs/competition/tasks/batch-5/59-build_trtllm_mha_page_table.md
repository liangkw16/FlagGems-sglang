<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/build_trtllm_mha_page_table -->
<!-- synced_at: 2026-09-10T23:30:38+08:00 -->

# build_trtllm_mha_page_table (kvcache/build_trtllm_mha_page_table)

## 任务描述

在设备端填充 TRT-LLM MHA page table，无 D2H sync：对每个请求，从 `req_to_token` 读每个 page 边界处的 KV token slot，存 `slot // page_size` 作为 block id。

## 接口签名

```python
def reference(req_to_token, req_pool_indices, cache_seqlens, page_table, page_size)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- `page_table[i, p] = req_to_token[req_pool_indices[i], p * page_size] // page_size`，`p < ceil(cache_seqlens[i] / page_size)`
- `page_table`: `[bs, max_num_pages]` int32，原地写入；baseline 与 reference clone 后返回
- `page_size` 必须整除 4096

## 正确性判别标准

exact（整数运算，逐元素相等）

## 参考实现

```python
import torch


def reference(req_to_token, req_pool_indices, cache_seqlens, page_table, page_size):
    # page_table[i, p] = req_to_token[pool_i, p * page_size] // page_size
    #   for p < ceil(cache_seqlens[i] / page_size)
    # Vectorized: build the (bs, max_pages) index matrix once and gather.
    out = page_table.clone()
    bs = req_pool_indices.shape[0]
    max_pages = out.shape[1]
    device = out.device

    n_pages = (cache_seqlens + page_size - 1) // page_size  # [bs]
    page_idx = torch.arange(max_pages, device=device).unsqueeze(0)  # [1, max_pages]
    mask = page_idx < n_pages.unsqueeze(1)  # [bs, max_pages]

    pool = req_pool_indices.unsqueeze(1)  # [bs, 1]
    tok = req_to_token[pool, (page_idx * page_size).expand(bs, max_pages)]
    out[mask] = (tok // page_size)[mask].to(out.dtype)
    return out
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
