# Task 96 `fused_pack_qkv` 实验记录

```current
task: 96
operator: fused_pack_qkv
batch: 7
validity: candidate-sealed
platform: 未提交（quota 0/30 未重置；release 门禁全过，候选封存待投）
candidate_stage: s0
team_best_stage: -
team_best_speedup: -
sealed: yes
next: s0 已封存待投：kunlun/huawei #1 <0.62 首发争分；E1 轴=BLOCK_R/C 扫描与 muxi 调优
updated: 2026-09-24
```

## 契约

- `(q,k,v,indices)`：`[B,S,H,D]` 三张量按 flat `B*S` 索引 gather 成三个
  `[total_valid,H,D]`；indices int32/int64；exact 纯 gather。

## 榜单靶标

EvokeAgent 6.00 分（仅 2 队可见）：tianshu 5.512 / muxi 2.446 / haiguang 3.266 /
**kunlun 0.606** / **huawei 0.617** / A 1.481 / B 1.392。两队长芯 kunlun/huawei 均 <1：
单 kernel 行 gather 结构大概率直接拿下这两芯 #1。

## S0 结构（commit b6641097）

- 单 kernel：BLOCK_R=4 行 × BLOCK_C=1024 列 2D 瓦片；`indices` 每行 load 一次按列广播；
  一个 program 内完成 q/k/v 三份拷贝（1 launch vs baseline 3 gather + long cast）；
  `idx.to(tl.int64)*row_elems` 防溢出；num_warps=4（sweep @HD2048 26.46µs vs torch 28.38µs）。

## 证据

- NVIDIA 代理 screening（r8 旧配置）5 测试 0 失败 11 launch；r4 新配置复筛中。
- proxy L2 常驻下 HD512 1.65x / HD2048 1.07x；平台冷数据口径下结构优势更大。

## 情报

- 快照 SHA-256 `061475d31505bb81f16a5a212fa7cfd6e4b1142188cb02da2c202962efedadb2`。

## 不可变身份（s0，2026-09-24）

- source commit = verification commit = `90d732fabdfe198b82bbfd5be662cc4ee77904b4`。
- ZIP：`artifacts/competition/fused_pack_qkv/s0-90d732f/fused_pack_qkv.zip`，SHA-256 `81767f230ba87ff3a2bd29a741909879a90b85c2ecaa9523fd770907aee27db3`（单成员 `fused_pack_qkv.py`，generic-only）。
- release 回执：`artifacts/competition/b7-s0-release-20260924/fused_pack_qkv/verification.json`（mode=release，exit 0，绑定该 commit 字节），SHA-256 `8e22c4884a819d64f008d1ac6e517ba72f44da2f4249e6aa42c65459c4c46469`；
  日志 SHA-256 `0d3477592fe48b177789cb1da5124102b8c281b7bd611e6022941832570bc682`。NVIDIA RTX 5070 Ti / torch 2.13.0+cu130 / triton 3.7.1。
- codex-review（commit 级，gpt-6-astra）：4 项发现（1×P1 grid/块失配、3×P2）全部修复后复筛/release 全绿。
