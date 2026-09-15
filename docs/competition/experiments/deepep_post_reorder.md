# Task 65 `deepep_post_reorder` 实验记录

```current
task: 65
operator: deepep_post_reorder
batch: 5
validity: valid
platform: completed(e8,8/8,13.4538x新TB;BLOCK1024全芯+38%)
candidate_stage: e8
team_best_stage: e8
sealed: no
next: e9 BLOCK2048明确追发(阶梯广谱兑现中);燧原3.0/沐曦10.1,榜首58.6
updated: 2026-09-16
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

## 2026-09-13 E3 平台终态：7/8（昆仑 uni_sram，BLOCK=256 仍超）

- 昆仑 exec 7571ms 真实执行后 uni_sram——512(e1)/256(e3) 均超。
  E4 方向：BLOCK=64。

## 2026-09-13 E4：BLOCK=64（已发射）

- 256 仍超 ⇒ E4 降到 64。
- source commit：`ef6270e3229672dfdabd3399d793fab4c35b9b85`；ZIP `e4-ef6270e`，
  SHA-256 `1ba76ed7b231559f732f263967db8edb0afb26e7c783ac79e460763987d737d8`；
  release 回执 SHA-256 `61decb1e55cdcaaae002f77c1cc400f0a192f330ed64de5275bd05b6af234a90`。

## 2026-09-13 E4 平台终态：7/8（昆仑间歇崩溃,BLOCK=64 未获裁决）

- exec 0ms 服务线程卡死——BLOCK=64 的字节没被检验。T65 昆仑=e3
  真实执行 uni_sram@256 + e4 秒崩,双条件窗口问题维持。

## 2026-09-15 E5 候选就绪：注释载体（e4 BLOCK=64 字节从未被裁决）

- commit `14417e2e`。e4 的昆仑发射撞崩溃族窗口（exec 0ms 服务线程
  卡死），BLOCK=64 字节从未被检验；本次为注释载体重掷（崩溃族协议
  1/1），执行零变化。
- 诊断补充（09-15）：平台 API 对 e1/e3 两发真实执行后失败不返回任何
  底层错误文本；本仓缓存昆仑 compiler.py:363 证实 `uni_sram` 标签包装
  任意 pm.run 异常——该标签无诊断价值，重掷是剩余唯一动作。
- ZIP：`artifacts/competition/deepep_post_reorder/e5-14417e2/`，7789
  bytes，成员 generic `aa643d92…`（=e4）/ kunlunxin `62dae23e…`
  （仅注释差异）。
- ZIP SHA-256：`b35d54e83c0ac65aff6dda5be7b5329d981ebe697c76fd7fe3347b5dede48554`。
- 门：昆仑产生有效判决；崩溃族再中即终封（同指纹两次）。
- 阻塞：GPU 通道中断，release 回执待补；未 preflight、未耗额度。

## 2026-09-15 16:20 E5 平台终态：7/8 —— 昆仑为编译期 SIGABRT（全新指纹，诊断突破）

- submission 15213（15:50 发射，GPU 通道恢复当日）。七芯全部健康：
  天数 11.773 / 沐曦 7.1714 / 燧原 1.5834 / 海光 18.42 /
  华为 11.2014 / A 12.1396 / B 7.6418——与 e2 水位一致，七芯部分和
  待命值不变。
- **昆仑失败带完整堆栈（首次）**：`评测进程异常崩溃（运行 4s，退出码 1），
  Fatal Python error: Aborted（SIGABRT，编译器内部错误）`，栈顶
  `triton/backends/xpu/compiler.py:439 make_llir` → compile 全链。
  非"服务线程卡死"崩溃族，非 uni_sram 资源超限。
- **历史归因订正**：本仓缓存 compiler.py:363 证实 `uni_sram` 标签包装
  pm.run 任意异常，而 make_llir 在 pm.run 内——e1（512）/e3（256）的
  "真实执行 10.5s/7.5s 后 uni_sram"大概率同为编译期 abort（编译耗时
  被计入 exec），BLOCK 大小自始不是病根。
- 处置：崩溃族重掷 1/1 已消耗；BLOCK 轴正式关闭。剩余唯一轴=结构性
  改写（更换触发 make_llir 断言的操作组合），无 assert 消息文本，
  需按操作族逐一隔离；暂列明日评估，不再盲发。
- 回执/日志：`artifacts/competition/batch5-verify-20260915/deepep_post_reorder/`
  （release exit 0，generic+昆仑 vendor 各 33 launch）。

## 2026-09-15 17:10 E6 平台终态：8/8 VALID 8.7338x —— 昆仑解锁，向量提取链确证为编译器 bug 根因

- submission（e6，0f883760）。**昆仑 vendor 被选中、passed 0.3592**
  （exec 14457ms 真实执行）——六轮失败后首次有效判决。
- 七芯：天数 11.8174 / 沐曦 7.0206 / 燧原 1.6698 / 海光 18.0998 /
  昆仑 **0.3592** / 华为 11.5022 / A 11.7876 / B 7.6136；均值 **8.7338**。
- **根因闭环**：E1 引入"masked 向量载入 + one-hot(lane==slot)乘 +
  tl.sum 归约提取标量"后昆仑连续编译期 SIGABRT（make_llir）；e6 恢复
  S0 形态标量 slot 读（BLOCK64/warps1/stages1/循环结构/累加序全保）
  即通过。Codex r6 咨询的差分嫌疑被平台直接证实。
- 后续：燧原 1.67 为最弱轴（榜首 15+）；昆仑 0.36 健康窗水位。

## 2026-09-15 21:30 E7 平台终态：8/8 VALID 9.7016x 新 TB；E8 就绪待回执（GPU 二次中断）

- **E7（6ff7c868，generic 标量 slot 形态）**：华为 **11.50→16.45（+43%）**、
  天数 11.8→14.2、A 11.8→13.3、B 7.76；沐曦 6.2 / 海光 17.7（-0.4）/
  燧原 1.60（持平）。均值 **8.734→9.702 新 TB**（ΔS=+0.97）。门（≥15）
  未达：标量化方向正确但 34-207 的榜首形态还有其他结构差。
- **E8（29f76413，generic BLOCK 512→1024，同燧原阶梯族）**：commit/ZIP
  就绪（SHA `29a49fb7…`），远端验证执行中 GPU 主机再次不可达
  （192.168.5.204 超时，今日第二次），回执未取回；恢复后补验即发。
- 额度：今日 18/30 已用。

## 2026-09-16 00:40 E8 平台终态：8/8 VALID 13.4538x 新 TB —— BLOCK 1024 全芯广谱兑现

- 判决：天数 16.46 / 沐曦 **10.07（+62%）** / 燧原 **3.01（+88%）** /
  海光 **26.87（+52%）** / 昆仑 0.36 / 华为 **18.41（+60%）** /
  A **20.95（+58%）** / B **11.49**；均值 **9.70→13.45（+38%）**。
- 标量形态（e7）+ BLOCK 1024（e8）复合阶梯成立；**下一档 2048 是
  明确追发项**（T63/T64 同族曲线仍在升）。
- 当日 T65 轨迹：0 → 8.73（解锁）→ 9.70（标量）→ **13.45（BLOCK）**。
