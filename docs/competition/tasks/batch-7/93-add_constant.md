<!-- source: https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/add_constant -->
<!-- synced_at: 2026-09-24T20:58:34+08:00 -->

# add_constant (elementwise/add_constant)

## 任务描述

给每个元素加一个编译期整型常量：`dst = src + constant`。
常量是 C++ 模板参数，每个不同值触发一次 JIT 编译并折叠进 kernel 体内；
超过 2^20 元素时切换到 `kMaxVecBytes` 宽的向量化路径。

## 接口签名

```python
def reference(src, constant)
```

> 选手实现的函数签名需与上述完全一致。

## 计算定义

- `src`：1D 连续 CUDA int32 张量，非空。
- 计算流程：

  ```
  out = src + constant
  ```

- 输出与输入同 shape 同 dtype。

## 正确性判别标准

精确（整数运算）。

## 参考实现

```python
def reference(src, constant):
    return src + constant
```

## 评分标准

本题评分标准仅展示赛题级补充信息；全赛道统一的正确性、加速比、性能门槛与排名规则请参阅「赛制规则 - 评分规则」。

**本题支持芯片：** 天数、沐曦、海光、昆仑芯、华为、国际通用芯片A、国际通用芯片B。不同赛题支持芯片可能不同，以该题的题目说明为准。

**反作弊规则：**

- 参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 **try/except** 异常捕获、条件分支、设备判断或其他方式，在 Triton 执行失败时 fallback 到 PyTorch 内置算子。
- 若代码实际执行路径未运行 Triton 自定义 kernel，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名。
- 若采用异常捕获、条件分支等手段规避 Triton 执行并 fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格。
