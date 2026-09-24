# Task 98 `get_mla_kv_buffer` 实验记录

```current
task: 98
operator: get_mla_kv_buffer
batch: 7
validity: candidate-wip
platform: 未提交（窗口开时 quota 0/30，等重置）
candidate_stage: s0
team_best_stage: -
team_best_speedup: -
sealed: no
next: 复筛+codex-review 通过后 seal；quota>0 首发投 s0（kunlun/huawei #1 均 <0.51，直接争分）
updated: 2026-09-24
```

## 契约

- `(kv_buffer, loc, cache_k_nope, cache_k_rope)`：按 `loc[i]` gather 行并拆 NoPE/RoPE 两半；
  `cache_k_nope/rope` 仅作 dtype/shape 模板（返回两个新张量，store 隐式转换 RTNE）；exact。

## 榜单靶标

EvokeAgent 4.59 分：muxi 1.418 / haiguang 1.886 / **kunlun 0.131** / **huawei 0.508** 全芯第一
（昆仑四队全部 ~0.13 = 结构性踩坑）；tianshu 第一 2.911（EA 自己仅 2.412 排 #7，幽灵池有界）。
任意正常行 gather 即可昆仑 #1（>0.131）+ 华为 #1（>0.508）。

## S0 结构（commit b6641097）

- 单 kernel：BLOCK_R=8 行瓦片，loc 一次 load；行内 BLOCK_C=512 列向量两段
  （NoPE 段 + RoPE 段偏移 nope_dim）；store 时 `.to(dtype)` 转换；i64 寻址；num_warps=8。
- 代理 sweep @n=16384：torch 基线 56.69µs vs 我方 13.65µs（**4.1x**）；r8c512w8 最优。

## 证据

- NVIDIA 代理 screening（r16 旧配置）6 测试 0 失败 19 launch；r8 新配置复筛中。
- 昆仑 0.131 之谜假设：torch 参考在昆仑走 vendor gather+cast 4-5 kernel 全速，
  Triton gather 在昆仑（XPU 端口）慢；单 kernel 减 launch 数是正确方向，昆仑专配
  待 KernelGen kunlun 后端恢复（请求已归档 `artifacts/competition/batch7-kernelgen-20260924/t98-kunlun-tune-req.json`）。

## 情报

- 快照 SHA-256 `061475d31505bb81f16a5a212fa7cfd6e4b1142188cb02da2c202962efedadb2`。
