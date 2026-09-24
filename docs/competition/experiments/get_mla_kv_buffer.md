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

## 不可变身份（s0 v2，2026-09-24，契约 spec 高阶复审后）

- source commit = verification commit = `daf77923f6913447b65a8c226f34297ce13b4440`（review v1 零宽半边已修；review v2：补 kv_buffer.stride(1) 与 loc.stride(0)）。
- ZIP：`artifacts/competition/get_mla_kv_buffer/s0-daf7792/get_mla_kv_buffer.zip`，SHA-256 `1d6e5fe55092e2a1081283b1448c1b918480af1c22b229c9bfa07a728f017002`（单成员 `get_mla_kv_buffer.py`，generic-only）。
- release 回执：`artifacts/competition/b7-s0-release2-20260924/get_mla_kv_buffer/verification.json`（mode=release，exit 0，绑定该 commit 字节），SHA-256 `64ff8b794bf824abcab0e096a9d5eb6848008f6391972532756373d89fd5bcce`；
  日志 SHA-256 `bc5213f527c79040273dd9e5c225fde3ccc97d24a10c4493f105e8f3df74fa16`。NVIDIA RTX 5070 Ti / torch 2.13.0+cu130 / triton 3.7.1。
- 提交脚本：`artifacts/competition/b7-s0-release2-20260924/submit-batch7-s0.sh`（v2，全部参数预烘焙）。
- review 历史：v1（commit 级，gpt-6-astra medium）4 项全修；v2（--base 对照契约 spec，high）5 项 P2 全修；
  三轮修复均经 screening + release 双门禁复跑全绿。
