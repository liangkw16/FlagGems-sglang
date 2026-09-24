# Task 94 `concat_mla_absorb_q` 实验记录

```current
task: 94
operator: concat_mla_absorb_q
batch: 7
validity: candidate-sealed
platform: 未提交（quota 0/30 未重置；release 门禁全过，候选封存待投）
candidate_stage: s0
team_best_stage: -
team_best_speedup: -
sealed: yes
next: s0 已封存待投：华为 #1=0.42 低垂果实；E1 轴=大 shape 宽度调优
updated: 2026-09-24
```

## 契约

- `(a, b)`：沿最后一维 cat，`[d0,d1,a_last]+[d0,d1,b_last] → [d0,d1,a_last+b_last]` bf16；
  题面明示两源行 stride 不同（不得假设连续）；exact 纯搬移。

## 榜单靶标

EvokeAgent 5.50 分（仅 2 队）：tianshu 1.700 / muxi 1.216 / haiguang 2.258 / kunlun 1.409 /
**huawei 0.420** / A 1.101 / B 1.420。华为第一仅 0.42 = 低垂果实；海光 2.26 显示上行空间。

## S0 结构（commit b6641097）

- 单 kernel：BLOCK_R=4 行瓦片；行 id 分解 (i0,i1)=divmod(row,d1)，a/b 各自
  stride 算基址；先 a_last 后 b_last 两段列循环（BLOCK_C=512 mask），一次写出行；
  i64 寻址；num_warps=8。
- 代理 sweep：小 shape 全配置 launch-bound（~6.1µs vs torch cat 2.7µs）；平台大 shape
  下单 kernel 结构 vs torch.cat 多 kernel 占优；config 保持 r4c512w8。

## 证据

- NVIDIA 代理 screening（配置未变）4 测试 0 失败 9 launch（21:15 回执有效）。

## 情报

- 快照 SHA-256 `061475d31505bb81f16a5a212fa7cfd6e4b1142188cb02da2c202962efedadb2`。

## 不可变身份（s0，2026-09-24）

- source commit = verification commit = `90d732fabdfe198b82bbfd5be662cc4ee77904b4`。
- ZIP：`artifacts/competition/concat_mla_absorb_q/s0-90d732f/concat_mla_absorb_q.zip`，SHA-256 `924da41ce714a9b230a904025e44e633c6c314a39000c50a9e0142d54326ae3f`（单成员 `concat_mla_absorb_q.py`，generic-only）。
- release 回执：`artifacts/competition/b7-s0-release-20260924/concat_mla_absorb_q/verification.json`（mode=release，exit 0，绑定该 commit 字节），SHA-256 `039a181b3cd3073f55ee70259a1466d1299d68e31b7930c498f6b39e8788690a`；
  日志 SHA-256 `c9917f2ff8081d1142bd55d61c889d737a713968c5e7f5d41def35cef3ff9cf1`。NVIDIA RTX 5070 Ti / torch 2.13.0+cu130 / triton 3.7.1。
- codex-review（commit 级，gpt-6-astra）：4 项发现（1×P1 grid/块失配、3×P2）全部修复后复筛/release 全绿。
