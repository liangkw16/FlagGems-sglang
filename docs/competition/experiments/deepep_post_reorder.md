# Task 65 `deepep_post_reorder` 实验记录

```current
task: 65
operator: deepep_post_reorder
batch: 5
validity: invalid_correctness
platform: submitted(e3,评测中;e2=7/8 昆仑间歇崩溃)
candidate_stage: e2
team_best_stage: -
sealed: no
next: 昆仑=间歇崩溃+需小 BLOCK 双条件；新 ZIP 重掷待健康窗口（同晨 T62 e5 正常判）
updated: 2026-09-12
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

## 2026-09-12 平台提交（submission 13302，daily_seq 7）

- 源 5573ffc（含 topk=0 退化分支 zeros 修复）；ZIP `s0-5573ffc`，SHA-256
  `af3f13539969f058bad1b9c93c977875c7e4e23e8112cc050ae3627f6009f67a`；
  回执 `batch5-ext6-validate-20260912/deepep_post_reorder/`
  （SHA-256 `09ee196654f1d1f232258ff672fcbad8fd51f6e84cb213960f41a9221a017f9e`，
  4 方法 0 失败、33 launch）。
- 七芯真实通过：天数 14.1316 / 沐曦 6.2860 / 燧原 1.3252 / 海光 16.6416 /
  华为 16.8990 / A 13.8796 / B 7.7532。
- **昆仑失败（崩溃族）**：exec 0ms，`服务线程卡死自动恢复，请重新提交`
  ——平台侧故障非内核裁决，按崩溃族协议不计代码止损；注释载体重掷需
  用户当次明示授权（≤2 发），首选平台工单健康 worker rerun。

## 2026-09-13 E1：路由行向量化（候选就绪后提交）

- S0 的路由/权重标量在每个 (hidden块, slot) 重读 → E1 提为每 token
  一次 masked 向量载入（昆仑可靠形态），slot 值算术提取（i32 lane
  乘法，避整型 where/i64 向量乘两大毒点）。全新 ZIP 重掷昆仑窗口。
- source commit：`8b9f4820ae661e2417eeeda057bba0efa6a0ba38`；ZIP `e1-8b9f482`，
  SHA-256 `70ccf24ec8426effa6e50ad8ec2267bc2a3de5654784fdbc8015121cbe8b1a7a`；
  release 回执 SHA-256 `e0a272cf2ffde7a8fdc21d701fcd71ac516d5a3b3cbf6cc9082397daa237c9e0`。
- submission 13790；裁决点=昆仑窗口 + 七芯读数（部分和 76.9 基准）。

## 2026-09-13 E1 平台终态：7/8（昆仑=uni_sram，真实执行非崩溃）

- 七芯：天数 12.0866 / 沐曦 **6.7288（+7%）** / 燧原 **1.6032（+21%）** /
  海光 **18.7770（+13%）** / 华为 10.5298（-38% 窗口）/ A 11.98 / B 7.59。
- **昆仑 exec 10503ms 真实执行后 `OutOfResources: uni_sram`**——向量
  载入形态脱离秒崩族；与 T72 同类，修法=昆仑 vendor 缩 BLOCK。
- E2：`_kunlunxin` vendor（BLOCK=512→128、num_warps=1）。

## 2026-09-13 E2：昆仑小 BLOCK vendor（候选就绪后提交）

- source commit：`c3d3af0bbe48056a1efdee9f3faea262ac5aa877`；ZIP `e2-c3d3af0`，
  SHA-256 `80775244042c0ce30f8d2ba025dcc41b5250925a6bd449d188aeb528becf2fde`；
  release 回执 SHA-256 `d8ffe65cd688a0e3c0976ec406a8de95a6dbb6037a397246b823831cb9483843`；
  4 方法 0 失败。
- submission 13792；裁决点=昆仑 uni_sram 解除（过即 8/8：七芯 e1 水位
  部分和 69.4 基准）。

## 2026-09-13 E2 平台终态：7/8（昆仑=崩溃族，间歇性）

- 昆仑终态 exec 0ms 服务线程卡死——**间歇性崩溃**：e1（BLOCK=512）
  真实执行 10.5s 报 uni_sram，e2（BLOCK=128）秒崩。同晨 T62 e5 昆仑
  vendor 正常判决 ⇒ 按提交闪断，非 kernel 确定性。
- 结论：T65 昆仑需（a）小 BLOCK 过 uni_sram +（b）健康评测窗口双条件。
  e2 字节已备好（BLOCK=128/num_warps=1），新 ZIP 身份重掷即可。
- 七芯 e2 水位：天数 12.07 / 沐曦 **9.24** / 燧原 1.56 / 海光 18.34 /
  华为 10.84 / A 11.93 / B 7.66（部分和 71.6）。

## 2026-09-13 E3：BLOCK=256 + num_stages=1 重掷载体（已发射）

- e2 字节（BLOCK=128）未获裁决即被间歇崩溃击中。E3 载体：BLOCK=256 +
  num_stages=1（介于 uni_sram 上限 512 与 e2 之间）。
- source commit：`e56175ab16644a1d271b49f5ff152106a16da61c`；ZIP `e3-e56175a`，
  SHA-256 `b477288091e52bddc5c0972a0546af228176604bfe89c9a55908e86e7b5e5435`；
  release 回执前缀 `09c3c36f`；4 方法 0 失败。
