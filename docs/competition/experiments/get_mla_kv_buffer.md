# Task 98 `get_mla_kv_buffer` 实验记录

```current
task: 98
operator: get_mla_kv_buffer
batch: 7
validity: candidate-sealed
platform: 未提交（quota 0/30 未重置；release 门禁全过，候选封存待投）
candidate_stage: s0
team_best_stage: -
team_best_speedup: -
sealed: yes
next: s0 已封存待投：kunlun/huawei #1 均 <0.51 首发争分；E1 轴=nope_dim constexpr 特化/昆仑专配
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

## 不可变身份（s0，2026-09-24）

- source commit = verification commit = `90d732fabdfe198b82bbfd5be662cc4ee77904b4`。
- ZIP：`artifacts/competition/get_mla_kv_buffer/s0-90d732f/get_mla_kv_buffer.zip`，SHA-256 `a3643b550f444e117ca19075d1084fdf71551bcc4766bc129e84f2df0ac74906`（单成员 `get_mla_kv_buffer.py`，generic-only）。
- release 回执：`artifacts/competition/b7-s0-release-20260924/get_mla_kv_buffer/verification.json`（mode=release，exit 0，绑定该 commit 字节），SHA-256 `7fe288110ae159099458d60bbd6398c7eff42578447536fa73a2833bffe068d4`；
  日志 SHA-256 `a66f5499902fc5c088a1e065303a90763bd7dec0545e0b15578980fa1cabbd1b`。NVIDIA RTX 5070 Ti / torch 2.13.0+cu130 / triton 3.7.1。
- codex-review（commit 级，gpt-6-astra）：4 项发现（1×P1 grid/块失配、3×P2）全部修复后复筛/release 全绿。
