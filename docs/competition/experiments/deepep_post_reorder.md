# Task 65 `deepep_post_reorder` 实验记录

```current
task: 65
operator: deepep_post_reorder
batch: 5
validity: candidate-wip
platform: not-submitted
candidate_stage: s0
team_best_stage: -
sealed: no
next: 远端 GPU 恢复后补 release 回执；回执齐全即进入 09-12 窗口发射队列（把握序第 3）
updated: 2026-09-11
```

## 契约与范围

- 完整题面：[Task 65](../tasks/batch-5/65-deepep_post_reorder.md)（2026-09-11 新增）。
- 接口 `deepep_post_reorder(down_output, output, src2dst, topk_ids, topk_weights, topk, hidden_size, routed_scaling_factor)`；
  从置换布局按 `src2dst[t,i]`（-1 跳过）gather topk 个专家下行输出并按
  `topk_weights * routed_scaling_factor` 加权求和；reference 返回**新张量**
  （`output` 仅提供 shape/dtype/device，不得改写）。
- 正确性为 mismatch-ratio（3e-2/3e-2，占比 <1e-2）：fp32 累加恒合法。
- 核心计算 Triton；八芯 0.1x。

## 实现（S0）

- 上游：SGLang 8014d9d `deepep_post_reorder_triton_kernel`（同为 Triton，
  bf16 累加形态）。本仓 S0 采用 fp32 累加（合法性更强、无性能代价），
  结构沿用 T64 已 8/8 的 house 形态：token 维 grid-stride（cap 65535），
  program 内按 BLOCK=512 分块扫 hidden，slot 循环 gather。
- 与 T64 的家族共性意味着 GCU 规则集（load 值不得直接作 store 寻址等）
  已在本形态上平台验证过；本 kernel 的 `dst` 同样先 `.to(tl.int64)` 再
  乘 stride，不落入已知毒点。

## 不可变身份

- source / verification commit：`b4727f1`。
- source SHA-256：`caf26efe7d040cc04584e4d0de0afaa597b00f2ca2ef36c6156a3a7735191ce9`。
- test SHA-256：`9da8cf303dd9962108c540ee7eccc24fb8ac054042eb182c6771769f1ead35f1`。
- ZIP：`artifacts/competition/deepep_post_reorder/s0-b4727f1/deepep_post_reorder.zip`。
- ZIP SHA-256：`fa59aba75434cc1857a2e08adf115dc64de7a5d149dc3cdf8211e1cff5f3bb31`。

## 验证状态

- py_compile、格式与 lint 通过（本地）。
- 测试：4 方法 / dtype（fp16/bf16/fp32）× 输出 dtype 转换 × 权重 dtype、
  topk 1~8 × hidden 1~7168（含非整除）、8193 token、scaling 1.0/0.0625、
  strided 路由与 down、空维度与全 -1；输入不变 + 新张量断言；公差 3e-2。
- **远端 GPU 不可达（2026-09-11 晚）**：release 回执待补，
  `target-runtime-unverified`。

## 风险

- 结构与 T64 同族，风险最低的一题之一。slot 循环内标量 `tl.load`（路由/
  权重逐个取）沿用上游；若弱芯上标量循环开销显著，E1 方向是把路由整行
  载入后向量化（参考 T64 深度优化轮的形态演化）。

## 优化方向（按把握）

1. S0 直投。
2. E1（未开发）：路由行向量化 + slot 常量化小循环展开。
3. 与 T64 共享燧原 grid 封顶经验（若大 grid 形状出现）。
