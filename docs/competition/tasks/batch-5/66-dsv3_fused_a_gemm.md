<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/dsv3_fused_a_gemm -->
<!-- synced_at: 2026-09-12T20:28:08+08:00 -->

# dsv3_fused_a_gemm (gemm/dsv3_fused_a_gemm)

## 任务描述

DeepSeek-V3 fused QKV-A 下投影的 decode 形状：`out = mat_a @ mat_b`，num_tokens <= 16。
这是 "fused A" GEMM —— 拼接的 `q_a_proj` + `kv_a_proj` 权重。在 min-latency batch
下它是 attention 前处理中单次最重的权重读取，因此用手写的 skinny-M kernel 而非 cuBLAS。

## 接口签名

```python
def reference(mat_a, mat_b)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- `mat_a`：`[num_tokens, hd_in]` bf16 row-major；hd_in 为 256 的倍数，num_tokens ∈ [1, 16]。
- `mat_b`：`[hd_in, hd_out]` bf16 **列主序**（即 row-major `[hd_out, hd_in]` 权重的 `.t()`）；
  hd_out 为 16 的倍数。
- 计算流程：

  ```
  out = mat_a.float() @ mat_b.float()
  ```

  要求 SM90+。

## 正确性判别标准

标准 per-dtype tolerance。

## 参考实现

```python
def reference(mat_a, mat_b):
    return (mat_a.float() @ mat_b.float()).to(mat_a.dtype)
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
