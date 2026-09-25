# Task 96 `fused_pack_qkv` 实验记录

```current
task: 96
operator: fused_pack_qkv
batch: 7
validity: valid(7/7,e1)
platform: s0(21124)invalid_threshold昆仑0.010;e1(21146)valid 7/7:昆仑0.010→0.583(57x,vendor 1D结构兑现)/华为0.950/天数2.95/海光3.15
candidate_stage: s0
team_best_stage: e1
team_best_speedup: -
sealed: yes
next: Q/K/V三程序分工代理测无增益(0.94-0.97)跳过;mask消除代理无增益(Triton已向量化);天数2.95->5.52的1.9x均匀差未定位,需EA结构证据
updated: 2026-09-25
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

## 不可变身份（s0 v2，2026-09-24，契约 spec 高阶复审后）

- source commit = verification commit = `daf77923f6913447b65a8c226f34297ce13b4440`（review v2：补 indices.stride(0) + 非连续 q/k/v 归一化（替代断言））。
- ZIP：`artifacts/competition/fused_pack_qkv/s0-daf7792/fused_pack_qkv.zip`，SHA-256 `4edde6d3aacc05c8dd63dee86dfd0d782bc707ae3fa217a463701dcb80705da0`（单成员 `fused_pack_qkv.py`，generic-only）。
- release 回执：`artifacts/competition/b7-s0-release2-20260924/fused_pack_qkv/verification.json`（mode=release，exit 0，绑定该 commit 字节），SHA-256 `98b7f6f9c603979336da0f3a66b9f9dab8696dd65b2fa289491b20ee761e3f94`；
  日志 SHA-256 `7e5058c05673e57ab61e2f9c9dc6563feee7d090443a8b9048d37f8a56b3f578`。NVIDIA RTX 5070 Ti / torch 2.13.0+cu130 / triton 3.7.1。
- 提交脚本：`artifacts/competition/b7-s0-release2-20260924/submit-batch7-s0.sh`（v2，全部参数预烘焙）。
- review 历史：v1（commit 级，gpt-6-astra medium）4 项全修；v2（--base 对照契约 spec，high）5 项 P2 全修；
  三轮修复均经 screening + release 双门禁复跑全绿。
