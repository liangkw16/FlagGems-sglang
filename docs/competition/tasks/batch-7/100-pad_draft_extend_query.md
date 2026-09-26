<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/pad_draft_extend_query -->
<!-- synced_at: 2026-09-26T22:45:04+08:00 -->

# pad_draft_extend_query (attention/pad_draft_extend_query)

## 任务描述

把 varlen（ragged）的 draft query 按 `cu_seqlens_q` 作为源偏移散入 padded 的
`[bs, max_seq_len, H, D]` buffer。超过 `seq_lens_q[b]` 的行保持不变。
SGLang 内核用 3D grid `(bs * max_seq_len, head_blocks, dim_blocks)`，短序列时大部分
program 立即退出 —— 该 early-exit 浪费正是优化目标。

## 接口签名

```python
def reference(q, padded_q, seq_lens_q, cu_seqlens_q)
```

> 选手实现的函数签名需与上述完全一致。

## 计算定义

- `q`：`[total_tokens, H, D]`；`padded_q`：`[bs, max_seq_len, H, D]`。
- `seq_lens_q`：`[bs]` int32；`cu_seqlens_q`：`[bs + 1]` int32。
- 计算流程：

  ```
  padded_q[b, :s] = q[cu_seqlens_q[b] : cu_seqlens_q[b] + s]   for s = seq_lens_q[b]
  ```

  先克隆 padded_q 再写回并返回。

## 正确性判别标准

精确（纯数据搬移）。

## 参考实现

```python
def reference(q, padded_q, seq_lens_q, cu_seqlens_q):
    out = padded_q.clone()
    bs = cu_seqlens_q.shape[0] - 1
    for b in range(bs):
        s = int(seq_lens_q[b])
        beg = int(cu_seqlens_q[b])
        out[b, :s] = q[beg : beg + s]
    return out
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
