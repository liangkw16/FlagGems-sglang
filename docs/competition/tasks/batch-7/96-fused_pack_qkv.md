<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/fused_pack_qkv -->
<!-- synced_at: 2026-09-24T20:58:34+08:00 -->

# fused_pack_qkv (diffusion/fused_pack_qkv)

## 任务描述

将 padding 后的 `[B, S, H, D]` Q/K/V 通过共享的 gather 索引打包为 varlen `[total_valid, H, D]`——一次 kernel launch 完成三个张量的 gather（而不是三次独立的 advanced-index gather）。

## 接口签名

```python
def reference(q, k, v, indices)
```

> 选手实现的函数签名需与上述完全一致。

## 计算定义

- `q`/`k`/`v`: `[B, S, H, D]`，同 shape 同 dtype
- `indices`: `[total_valid]` int32/int64，flat `B*S` 位置中保留 token 的下标
- 逐行计算：`q_unpad[i] = q_flat[indices[i]]`，`k_unpad[i] = k_flat[indices[i]]`，`v_unpad[i] = v_flat[indices[i]]`
- 输出 `(q_unpad, k_unpad, v_unpad)`，均为 `[total_valid, H, D]`

## 正确性判别标准

exact（纯 gather，逐元素相等）

## 参考实现

```python
def reference(q, k, v, indices):
    bs, seq, h, d = q.shape
    idx = indices.long()
    return (
        q.reshape(bs * seq, h, d)[idx],
        k.reshape(bs * seq, h, d)[idx],
        v.reshape(bs * seq, h, d)[idx],
    )
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
