# Task 93 `add_constant` 实验记录

```current
task: 93
operator: add_constant
batch: 7
validity: candidate-wip
platform: 未提交（窗口开时 quota 0/30，等重置）
candidate_stage: s0
team_best_stage: -
team_best_speedup: -
sealed: no
next: codex-review + 复筛通过后 seal s0；quota>0 即 preflight+提交；E1 轴=BLOCK/warps 扫描与华为专配
updated: 2026-09-24
```

## 契约

- `def reference(src, constant)`：1D 连续 int32，`out = src + constant`，精确。
- baseline 是 C++ 模板（常量编译期折叠 + kMaxVecBytes 向量化）；>2^20 元素走向量路径。

## 榜单靶标（逐芯 #1 读数，score=Σ1/rank 口径，见 plan 文档）

wangteam 4.15 分领跑但全场均值 <1.13：tianshu 1.129 / muxi 1.012 / haiguang 0.971 /
kunlun 0.910 / huawei 0.988 / A 1.005 / B 1.008。全芯 ~1.0+ 即可拿 5.5+ 分。
关键分化：华为其他队 0.14-0.34（wangteam 0.988 独家配方），昆仑次优 0.48-0.72。

## S0 结构（commit b6641097）

- `CONSTANT: tl.constexpr` 编译期折叠（对齐 baseline 模板 trick）；
- flat map，BLOCK=1024 / num_warps=4（4M 元素代理 11.69µs ≈ torch 11.70µs）；
- i64 仅寻址；mask 全覆盖；`if numel` 防 0 元素。

## 证据

- NVIDIA 代理 screening（BLOCK=4096 旧配置）4 测试 0 失败；1024/4 新配置复筛中。
- 代理 launch 开销实测：Triton ~4.8µs vs torch.add ~2.6µs（小 shape 全场 <1.0 的成因假设）。
- KernelGen 华为 autotune job `6e09dda3`（探华为专配）；昆仑后端仍 502。

## 情报

- 快照 `docs/competition/data/batch7-intel-20260924.json`
  SHA-256 `061475d31505bb81f16a5a212fa7cfd6e4b1142188cb02da2c202962efedadb2`。
