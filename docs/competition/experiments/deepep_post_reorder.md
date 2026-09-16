# Task 65 `deepep_post_reorder` 实验记录

```current
task: 65
operator: deepep_post_reorder
batch: 5
validity: valid
platform: release-ready(e10;TB e9 8/8,16.10245x)
candidate_stage: e10
team_best_stage: e9
team_best: e9 16.10245x
team_best_speedup: 16.10245
sealed: no
next: e10 bounded hidden-grid已过screening与exact release，live preflight后一次提交；不重投e9
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

## 2026-09-16 实时对账：E9 已提交且终态有效，禁止重投

- 本轮只读核对既有 submission **15824**（10:39:45），8/8、valid、均值 **16.10245x**，较 E8 13.4538 **+19.69%**；10:46 榜单第 **11**，榜首 **69.07705x**。此前提交事实不归功于本轮新发射。
- 逐芯：tianshu 17.4662 / muxi 12.5834 / enflame 4.362 / haiguang 36.0698 / kunlunxin 0.353 / huawei 19.019 / card_a 26.2802 / card_b 12.686。
- source / verification commit：`aa06d46d4b278e585795997b4e3e6a406e909dbe`；ledger commit 为本节所属提交。
- ZIP `artifacts/competition/deepep_post_reorder/e9-aa06d46/deepep_post_reorder.zip`，SHA-256 `614240444e83066136fa20dff3fe33f243f44e3f8dfe8215ce0ad7c452d55b12`；成员 `deepep_post_reorder.py` / `deepep_post_reorder_kunlunxin.py`。
- release 回执 `artifacts/competition/batch5-verify-20260916/deepep_post_reorder/verification.json`，SHA-256 `cf46ee647098f116a055ef440df72bd55a32d192f98585fe3be5b5be0e67c819`；4 tests、零失败/skip，generic 与昆仑代理实际执行；昆仑真机证据由本次平台8/8补齐。日志 SHA-256 `9c4b30917d8f5ff37ad5857cbc68b91d1c4d0b4c8ab46cdfa5d3a4d51ed1784e`。
- 源码成员哈希：generic `ff8f63752563125523f71c7736449a2f642baccdc209608e758aabb349260db9`；昆仑 `80a428c5c005b54b3fe7019b091c413438a462ba2b68f08253cafb5a083ae5ab`。
- 原始状态 `artifacts/competition/top1-20260916/t65-e9-status.json`，SHA-256 `c2e925fa9a364b7cf87670ef07afab6ad0fd02997458bd823110099ab8b4e8aa`；一次性 intent 状态 submitted。旧远端 ZIP 回读 unavailable，不重发上传。
- 当前剩余额度 28/30（全账号共享）；宽度仍可能提分，但距离新榜首 +328.98%，按本轮Top1优先级后移。

## 2026-09-16 E10：有界 hidden-grid 并行，发布就绪

- 成熟来源：SGLang `5f6dd44edc96779d4a15331637e26e73265ff6eb` 的 `post_reorder_deepgemm_triton_kernel`，见[固定上游报告](../research-top1-upstream-refresh-20260916.md)。只将hidden块调度到grid.y，保留BLOCK2048、token/hidden双轴stride、slot顺序、fp32加权累加、scalar路由、所有stride和输出语义；昆仑逐字节冻结TB E9。
- 首个独立轴cap候选虽筛选1.31996x，发布审查用T21 submission4274的昇腾coreDim114688失败证据阻断。新版本 `gy=min(cdiv(hidden,2048),255)`、`gx=min(tokens,max(1,65535//gy))` 保证总program≤65535；原字节不晋级、不上传。
- 新screening `artifacts/competition/t65-hidden-grid-bounded-screening-20260916/`：9/9（8数值+1CPU元数据捕获）、0F0E0S，generic/冻结昆仑各56入口/50实际launch；TB同9方法通过。14affected×5轮geomean **1.323555**，轮次1.317674–1.329518，最差0.955736，6control为0.996791–1.019597，zero spill/shared0。T1024/H4097/K2保留fp16约4.4%、bf16约2.7%回退；T1/H2049/K1两桶全无效路由，不能当有效gather收益。
- 新测试覆盖hidden2048/4096/255块边界、token65535/36/37、strided权重/topk16；T32768/H2049/K1实际数值测试验证grid(32767,2)下末token的stride覆盖。gy255巨型组合只做CPU wrapper元数据捕获，不宣称数值执行。screening receipt SHA-256 `4a57fb63411e9ee89f0a6038ce7d56e95ddee88720c8c3ee5e56cb4667d7cca8`；benchmark `7aab491ab97f94c492768e8be84d18fd8c8de6faca56cd3f09d86bb8854ac049`。
- source/verification commit **`4713e8d6029dfbe953060999e573ce8b5afd9983`**。generic SHA-256 `efb06c8262bfacc1380686cffa33c5bb2629e3e1e38e43fc90c53cb2d1729549`；昆仑 `80a428c5c005b54b3fe7019b091c413438a462ba2b68f08253cafb5a083ae5ab`；test `cfc15209dd1cfecce6f75f86254453420ece2fceb84695f1d47d447d7cb943d0`。与新screening逐字节一致；py_compile/Black/isort/flake8与独立静态审查通过。
- exact release `artifacts/competition/t65e10-release-20260916/`：9/9、0F0E0S，generic/昆仑各56入口/50实际launch。回执SHA-256 `1f09cb9b7a4ac90a420015eec3d962b54e26f585a86bcd71ee5290328ed4001a`；完整日志 `d3722b34e51af6cc2c4a4505e58462710774afc5811e7c4ee40514ddf89aa76b`。RTX5070Ti/Python3.12.13/Torch2.13.0+cu130/Triton3.7.1；远端 `/tmp/flagos-t65e10-release.NVcdmH`、PID388094、660秒上限、EXIT0，输入/输出双端验签通过。NVIDIA proxy，非NVIDIA目标runtime仍未验证，需八芯平台裁决。
- 不可变ZIP `artifacts/competition/deepep_post_reorder/e10-4713e8d/deepep_post_reorder.zip`，7436bytes，SHA-256 **`d1afa670187f6a24cca33c630a9df01fdc58bd04ce69d109e8ef9abec0fcf42b`**；成员 `deepep_post_reorder.py` / `deepep_post_reorder_kunlunxin.py`，member hash如上。
- **平台预注册**：8/8、每芯≥0.1、均值>16.10245才换TB；≥20.0为继续投入信号；榜首69.07705不能由代理1.32x推得。一次候选一次判决，不按结果改样本或门槛。
