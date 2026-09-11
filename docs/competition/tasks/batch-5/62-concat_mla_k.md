<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/concat_mla_k -->
<!-- synced_at: 2026-09-12T00:03:19+08:00 -->

# concat_mla_k (attention/concat_mla_k)

## 任务描述

构建 MLA key 张量：逐 head 拷贝 NoPE 部分，并把 single-head 的 RoPE 部分广播到所有 head。
SGLang 内核原地写 `k`；baseline/reference 克隆 `k` 并返回，按返回张量评分。

## 接口签名

```python
def reference(k, k_nope, k_rope)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- `k`：`[num_tokens, num_heads, nope_dim + rope_dim]` bf16（生产形状 `[T, 128, 192]`）。
- `k_nope`：`[num_tokens, num_heads, nope_dim]` bf16（nope_dim=128）。
- `k_rope`：`[num_tokens, 1, rope_dim]` bf16（rope_dim=64）。
- 计算流程：

  ```
  rope = k_rope.expand(-1, num_heads, -1)
  out  = cat([k_nope, rope], dim=-1)
  ```

## 正确性判别标准

精确（纯数据搬移）。

## 参考实现

```python
import torch


def reference(k, k_nope, k_rope):
    num_heads = k.shape[1]
    rope = k_rope.expand(-1, num_heads, -1)
    return torch.cat([k_nope, rope], dim=-1).to(k.dtype)
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
