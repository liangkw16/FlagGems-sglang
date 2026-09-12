# Task 67 `fill_padded_rows` 实验记录

```current
task: 67
operator: fill_padded_rows
batch: 5
validity: valid
platform: completed(13300,s0,8/8,3.4329x)
candidate_stage: s0
team_best_stage: s0
team_best_speedup: 3.43285
sealed: no
next: S0 首发 8/8 valid（seq 5）；昆仑 0.5952 为最薄芯；E1 方向（列分块 grid-stride）仅在需要抬昆仑时开发
updated: 2026-09-12
```

## 契约与范围

- 完整题面：[Task 67](../tasks/batch-5/67-fill_padded_rows.md)（2026-09-11 平台新增六题之一）。
- 接口 `fill_padded_rows(x, num_token_non_padded, fill_value)`；exact；
  返回**新张量**（reference 为 `x.clone()` 后填充），`num_token_non_padded`
  为 device 单元素整数张量，须在 kernel 内读取、静态 grid（每行一个
  program），保持 CUDA-graph 可捕获语义。
- 核心计算 Triton；无 fallback。八芯每芯 0.1x；截止 2026-09-17 19:59:59。

## 实现（S0）

- 上游：SGLang 8014d9d `kernels/ops/moe/fill_padded_rows.py`（上游为原地
  版本）；本仓改为 out-of-place：wrapper `out = x.clone()` 后单 kernel
  填充 `row >= n` 的行。`n` 由 `tl.load` 在 kernel 内读取，无 `.item()`
  同步，grid `(rows,)` 静态。
- `BLOCK_COLS = next_power_of_2(n_cols)` 单块覆盖整行；比较与寻址走 int64。

## 不可变身份

- source / verification commit：`b4727f1`（2026-09-11）。
- source SHA-256：`92caf5e002dd60f5c4ce80c523159fd150a79ea87a44e15bfdc46312f63196df`。
- test SHA-256：`07929937c007383c78d597842ed66df875eedc452e1c1d64431d6e465288fe6e`。
- ZIP：`artifacts/competition/fill_padded_rows/s0-b4727f1/fill_padded_rows.zip`。
- ZIP SHA-256：`9319d9f06783f314706b635067bf456521a22007858f73a23658bba869c78a27`。

## 验证状态

- py_compile、black/isort/flake8 全部通过（本地）。
- 测试：4 方法 / 覆盖 dtype×fill 值、n 边界（0/部分/rows/越界）、列宽
  1~7168、行距 stride、int32/int64 计数、空输入与 0 列；断言输入不变、
  返回新张量。
- **远端 GPU（192.168.5.204 / gpu-et）2026-09-11 晚全程不可达**（物理口与
  EasyTier 均超时，ARP 无表项）；release 回执待补，目标芯与代理执行均为
  `target-runtime-unverified`。KernelGen MCP 本会话不可用。

## 风险

- 结构极简（clone + 条件填充），跨芯风险最低；唯一注意点是
  `next_power_of_2(n_cols)` 在超宽行时的寄存器压力（生产场景 n_cols 为
  hidden 量级，可控）。

## 优化方向（按把握）

1. S0 直投（预期显著快于 reference 的 clone+index_put 链）。
2. E1 候选（未开发）：若平台形状列数巨大，改 grid-stride 列循环降低
   BLOCK_COLS；先取平台逐芯数据再决定。

## 2026-09-12 平台结果（submission 13300，daily_seq 5）

- **8/8 valid，均值 3.43285x**（首发即过）。逐芯：天数 9.7498 / 沐曦
  2.2450 / 燧原 1.3660 / 海光 4.8460 / 昆仑 0.5952 / 华为 1.3910 /
  A 3.9498 / B 3.3200。
- 回执（b4727f1，RTX 5070 Ti 代理）：4 方法 0 失败、25 次真实 launch、
  14 组非空 shape；`artifacts/competition/batch5-ext6-validate-20260912/fill_padded_rows/`
  （verification.json SHA-256 `ee29bb4668e9d52fcebf8587d1651242d1da0dfc77f76b21b562e005e0e8987c`）。
