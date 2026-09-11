# Task 69 `fused_moe_dispatch_index` 实验记录

```current
task: 69
operator: fused_moe_dispatch_index
batch: 5
validity: candidate-wip
platform: not-submitted
candidate_stage: s0
team_best_stage: -
sealed: no
next: 远端 GPU 恢复后补 release 回执；重点观察昆仑/燧原对 masked tl.atomic_add 的编译与执行；回执齐全进入发射队列（把握序第 4）
updated: 2026-09-11
```

## 契约与范围

- 完整题面：[Task 69](../tasks/batch-5/69-fused_moe_dispatch_index.md)（2026-09-11 新增）。
- 接口 `fused_moe_dispatch_index(topk_ids, num_local_experts, m_max)`；
  返回 `(masked_m, src2dst)`：atomic cursor 按 expert 分桶，
  `src2dst[i] = e*m_max + offset`；`masked_m[e]` 精确比较，
  `src2dst` 按桶内 multiset 比较（桶内顺序来自 atomics）。
- `-1` padding 跳过（reference 对应写 0）；int32；核心计算 Triton；八芯 0.1x。

## 实现（S0）

- 上游：SGLang 8014d9d `fused_moe_dispatch_index_triton_kernel`。本仓 S0
  简化为双 `torch.zeros` 初始化（masked_m 与 src2dst 的 padding 位语义
  都与 reference 对齐）+ 单 kernel masked `tl.atomic_add`；
  舍弃上游 single-block `ZERO_INIT + debug_barrier` 分支（弱芯 barrier
  语义风险 > 省 1 次 memset 的收益；reference 本身含 `.tolist()` CPU
  往返，多一次 memset 不改变量级）。
- grid-stride（cap 65535）× BLOCK=256。

## 不可变身份

- source / verification commit：`b4727f1`。
- source SHA-256：`de7fa1488ed40608b95ed23894942883f89a0ea472268b86fda59bac224e4d4d`。
- test SHA-256：`aac8b512245e0c3817b9a01e5379ba8577cd126950e1f3579f038ccc362712d3`。
- ZIP：`artifacts/competition/fused_moe_dispatch_index/s0-b4727f1/fused_moe_dispatch_index.zip`。
- ZIP SHA-256：`4c133b19523f74a3ac46ab19c26d7412566071790d03e0870f10cacaa20ed20a`。

## 验证状态

- py_compile、格式与 lint 通过（本地）。
- 测试：3 方法 / 常规与 25% padding、形状族（E=1/256、T*K 跨单块与
  grid-stride 边界 8193×8）、空输入；`masked_m` 精确、`src2dst` 全局
  排序 multiset（合法 dst 全局唯一，桶错位必被检出）、padding 位精确 0。
- **远端 GPU 不可达（2026-09-11 晚）**：release 回执待补，
  `target-runtime-unverified`。

## 风险

- **`tl.atomic_add` 跨芯证据缺口**：本仓已交付算子无 atomic 内核先例；
  FlagTree #1019 仅确认昆仑不支持 `tl.atomic_cas`，未涉及 atomic_add。
  昆仑 XMLIR / 燧原 GCU 的 masked atomic_add 需平台首轮反馈；若失败，
  备选是双 kernel 无原子方案（Triton 直方图计数 + 前缀和重写），代价是
  复杂度与两次启动。
- `expert_safe * m_max + offset` 为 int32：按题面约束
  （每专家计数 < m_max、E×m_max 受显存约束）不越界。

## 优化方向（按把握）

1. S0 直投，首轮重点读昆仑/燧原编译结果。
2. 若 atomic 芯失败：E1 无原子双 kernel（直方图 + 独占前缀和 + scatter），
   仅覆盖失败芯做 vendor 文件。
