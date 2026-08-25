<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/chunk_local_cumsum_vector -->
<!-- synced_at: 2026-08-25T00:19:05+08:00 -->

# chunk_local_cumsum_vector (fla/chunk_local_cumsum_vector)

## 任务描述

Vector-mode local cumulative sum within chunks: 将输入 `[B, T, H, S]` 按 `chunk_size` 切块后，在 chunk 的时间维度上做 cumsum（前缀和）。支持反向 cumsum 和可选缩放。与 scalar 版本的区别在于最后一维是向量（`S` > 1）。

## 接口签名

```python
def reference(g, chunk_size, reverse=False, scale=None)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- 输入 `g`: `[B, T, H, S]`（T 必须被 chunk_size 整除）
- 按 chunk_size 切块: `g_c = g.view(B, NT, BT, H, S)`
- 在 chunk 内的时间维（dim=2）上做 cumsum
- `reverse=True` 时先 flip 再 cumsum 再 flip 回来
- `scale` 非 None 时乘以该标量
- 输出与输入同 shape `[B, T, H, S]`

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`


## 参考实现

```python
def reference(g, chunk_size, reverse=False, scale=None):
    B, T, H, S = g.shape
    BT = chunk_size
    NT = T // BT

    g_c = g.float().view(B, NT, BT, H, S)
    if reverse:
        g_c = g_c.flip(2)
    out = g_c.cumsum(dim=2)
    if scale is not None:
        out = out * scale
    if reverse:
        out = out.flip(2)

    return out.reshape(B, T, H, S)
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
