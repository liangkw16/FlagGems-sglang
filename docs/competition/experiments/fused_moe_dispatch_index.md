# Task 69 `fused_moe_dispatch_index` 实验记录

```current
task: 69
operator: fused_moe_dispatch_index
batch: 5
validity: invalid_correctness
platform: completed(13303,s0,5/8)
candidate_stage: s0
team_best_stage: -
sealed: no
next: 三芯失败各归其类：燧原=PassManager 编译失败（masked tl.atomic_add 合法化缺口）、华为=582s 后 reference 侧 RuntimeError、昆仑=崩溃族（服务线程卡死，非内核裁决）；五芯读数 36~120x 证明 atomic 形态跑通处极快；下一轴=无原子三段式 vendor（直方图+独占前缀和+确定性 scatter）覆盖燧原/华为（可试带昆仑），generic 保留
updated: 2026-09-12
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

## 2026-09-12 平台提交（submission 13303，daily_seq 8）

- 回执（b4727f1，含 6 次 launch 的 atomic 路径）：
  `batch5-ext6-validate-20260912/fused_moe_dispatch_index/`
  （SHA-256 `c788579079acc54e32b8195bc66dd266d48ce0b8da557d23f4e7f8161a00828d`）。
- **燧原失败**：`RuntimeError: Pipeline run failed: PassManager execution
  failed`——masked `tl.atomic_add` 在 GCU300 无法合法化（编译层，与
  FlagTree#1019 确认的 atomic_cas 缺口同族）。
- **华为失败**：exec 582678ms 后 speedup 0.0，错误栈落在平台 reference
  文件 `flaggems_reference/fused_moe_dispatch_index.py:23` 与 torch_npu
  utils——atomic 形态拖垮 NPU 运行时后 reference 崩溃的嫌疑最大
  （c2flow 8/8 通过证明 reference 本身可在华为运行）。
- **昆仑失败（崩溃族）**：`服务线程卡死自动恢复，请重新提交`，
  exec 0ms——昆仑评测器平台侧故障，非内核裁决，按崩溃族协议不计代码
  止损、重掷需用户当次明示授权。
- 已过 5 芯读数（atomic 路径跑通处非常快，reference 的 CPU tolist 循环
  极慢）：海光 **119.737** / A 71.1100 / 天数 62.0734 / B 54.3222 /
  沐曦 36.1954。
- 结论：atomic 路线在燧原（编译）/华为（reference 侧崩溃）双双落定失败；
  下一候选为**无原子三段式 vendor**（Triton 直方图计数 → 独占前缀和 →
  确定性 scatter），覆盖燧原/华为（可试带昆仑，若跑通则一并绕开崩溃族
  重掷问题），generic 保留五芯已证路径。
