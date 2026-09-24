# Task 97 `fused_sigmoid_mul` 实验记录

```current
task: 97
operator: fused_sigmoid_mul
batch: 7
validity: candidate-wip
platform: 未提交（窗口开时 quota 0/30，等重置）
candidate_stage: s0
team_best_stage: -
team_best_speedup: -
sealed: no
next: 复筛+codex-review 通过后 seal；E1 轴=昆仑/华为专配（EA 独占两芯第一）；quota>0 投 s0
updated: 2026-09-24
```

## 契约

- `(attn_output, gate)`：`out = attn * sigmoid(gate)`，fp32 计算、按输入 dtype 存储；
  flat 同形路径 + strided 路径（gate 3D 非连续，需按 stride 读）；per-dtype 容差。

## 榜单靶标

EvokeAgent 5.00 分：tianshu 5.301(c2flow) / muxi 3.033 / haiguang 4.102 /
**kunlun 2.007** / **huawei 1.486**（EA 独占两芯第一，他队 0.60-0.96）/ A 3.776 / B 3.150。
T90/T81 华为经验（子块结构 1.23→1.5 轴）可迁移。

## S0 结构（commit b6641097）

- 单 kernel flat map BLOCK=2048/w8；ATTN_CONT/GATE_CONT 双 constexpr 分支：
  连续时纯 flat 偏移；非连续时 t=offs//hidden 分解 + gate 3D stride 直读（无 contiguous 拷贝）；
  `tl.sigmoid` fp32（T90 已验证跨芯形式）。
- 代理 sweep：4096×4096 torch 825.75µs vs 我方 115.3µs（**7.1x**，torch 5 kernel）；
  strided3d 66.1µs。

## 证据

- NVIDIA 代理 screening（BLOCK=4096 旧配置）6 测试 0 失败 18 launch；b2048 复筛中。

## 情报

- 快照 SHA-256 `061475d31505bb81f16a5a212fa7cfd6e4b1142188cb02da2c202962efedadb2`。
