# Task 65 `deepep_post_reorder` 实验记录

```current
task: 65
operator: deepep_post_reorder
batch: 5
validity: valid
platform: completed(16572,e13,8/8,26.54165x<TB;保E10)
candidate_stage: e13
team_best_stage: e10
team_best: e10 26.917125x
team_best_speedup: 26.917125
sealed: no
next: E12终态7/8，昆仑收集测试失败；保留权重契约修复，需目标执行新证据后再迭代，不重发15913
updated: 2026-09-17
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

## 2026-09-16 E10 单次平台提交

11:52:57，submission **15876**、daily_seq8，nonce `c1df0069070a982ee6e25dd671d4102f`。live preflight身份、题号、完整commit、test/receipt/ZIP哈希、成员和额度全部匹配后执行返回的一次性confirm命令；upload/POST各一次，state=submitted。远端ZIP回读7436bytes、SHA-256 `d1afa670187f6a24cca33c630a9df01fdc58bd04ce69d109e8ef9abec0fcf42b` verified。发前23/30，预计发后22/30，实际以status为准。TB仍E9，等待八芯结果；证据 `artifacts/competition/top1-20260916/t65-e10-{preflight,submit}.json`。

## 2026-09-16 E10 平台终态：8/8，26.917125x，新TB +67.16%

11:54:58只读核对submission15876：completed/valid，8/8全部通过，平均 **26.917125x**，比E9 **16.10245x提升67.16%**，超过20.0继续投入门。平台改善覆盖六个generic芯片；昆仑源码冻结，读数变化不归为代码收益。华为17.5128较E9 19.019回退约7.9%，仍为对榜首的最大缺口。禁止重复提交本候选。

|芯片|E9|E10|
|---|---:|---:|
|tianshu|17.4662|31.4906|
|muxi|12.5834|18.7902|
|enflame|4.362|8.1668|
|haiguang|36.0698|70.5854|
|kunlunxin|0.353|0.376|
|huawei|19.019|17.5128|
|card_a|26.2802|48.852|
|card_b|12.686|19.5632|

状态证据 `artifacts/competition/top1-20260916/t65-e10-status.json`；剩余额度22/30。榜首69.07705仍需约156.63%，不能报告已经Top1；后续只能依据新结构证据筛选，原有BLOCK阶梯/同字节不重试。

## 2026-09-16 E10 后续独立根因：普通 hidden 的单次动态循环

- exact E10四形状IR探针保留了hidden回边：例如T32/H8192/K8、grid.y=4，优化后TTGIR仍有`scf.for %start`，LLVM与PTX都有索引推进/比较/向后分支；40regs、0spill、0shared。上游SGLang固定SHA的DeepGemm版在918–919行直接以pid1*BLOCK取块，提供成熟来源。不是重试BLOCK/warps，也不删除token stride。
- 原始证据 `artifacts/competition/t65-e10-hidden-ir-probe-20260916/`：resources.json SHA-256 `966e557e3890133bb6444a4c8286389ca06df4a4f87c5f80f647d3ff0c14d232`；日志 `99698512fd4ae467cf8992fb3f15f612a8c87236c99f218faf75ef92863951cd`。4形状各一次实际launch/数值通过；E10 kernelbody与先前候选相同，缓存asm的source-location仍可能指向旧路径，实际wrapper/source由输入双端hash另行绑定。NVIDIA证据，非昇腾收益证据。
- 拟E11仅加constexpr `DIRECT_HIDDEN=(cdiv(hidden,2048)<=255)`：普通hidden直接执行同一数学，大hidden完整保留E10循环；grid总乘积限制、token stride、slot if、累加序、fp32、昆仑字节均冻结。slot predication因成熟主实现仍用if、无目标瓶颈且有NaN/Inf契约风险，本轮不做。
- **筛选预注册**：原20桶全部是affected（含hidden≤2048，不按结果缩域），另加direct边界及hidden>522240 fallback control；5轮AB/BA，affected GM≥1.05、每轮≥1.03、每桶≥0.95，fallback control在0.97–1.03，0spill。完整9方法加E10逐位对比/最后有效slot/stride；未通过不发布。平台须8/8、每芯≥0.1、均值>26.917125才换TB，≥30为进一步投入信号。

## 2026-09-16 E11 hidden-direct 筛选终态：NO-GO，保留E10

- 原20桶全部纳入primary，4个direct边界secondary、4个大hidden fallback controls；28桶×5轮AB/BA完整结束。primary GM **0.9930499114** <1.05，5轮0.991333–0.996207均<1.03；最差T3/H4097/K2 fp16 **0.902643** <0.95。fallback controls0.995542–1.000959稳定，secondary均≥1.00166，0spill/0shared。按原门关闭，不挑域、不提交。
- **10/10完整正确性**、baseline10/10；generic/昆仑各68公开入口、62实际launch。13补充逐字节对照+28性能形状逐字节对照全过；compiled TTGIR的direct从3层loop减为2，fallback双方仍3层，说明机制真实改变但未提供收益。尾形状寄存器从95/94降至48也未转化成性能改善。
- 源码SHA-256 `a7b91f72a8531bca03be3d661cbb00c89b37f57944f0465d9ae548154c07c40e`，test `2f7147a3367551331cbf893d9a3384ca05af226aa99b69cf86aed13845b3de40`，benchmark脚本 `9d0d6d2126fb7a4c61fc6235a8090a75a0d9f09e6386b7f5ec6cd4d802df211e`，与预注册冻结字节一致；数学helper/旧fallback/旧token stride的AST逐节点一致，昆仑exact E10未变。未改主树、未commit候选、未打ZIP/建立intent/上传/POST。
- 完整证据 `artifacts/competition/t65-e11-hidden-direct-screening-20260916/`：receipt SHA-256 `984e107f79165ac444664244232c669b27288b67f30c99a98d06afb1cd845694`，verification.log `af24b1a15633181a19c5093bfcf3c1525e1a7699272901d39151a3740a77bb85`，benchmark.json `1f198b7eef25d5d261d5e3f01f7577e8132944f05f6c1194bd44729a45718799`，screening-summary.json `13c523819d106678f578a62ae13aac81089909b4544e8613bd3f519bef0b131e`。140对原始样本CSV、16IR、全部双端hash保留。远端 `/tmp/flagos-t65e11-screen.AvE3F9`、PID388623、timeout600、EXIT0，GPU已释放。
- 本轮保留平台E10 **26.917125x**；12:13榜单第10、榜首69.07705，仍需156.63%。余22/30，不为剩余预算强投失败候选。

## 2026-09-16 12:26 华为诊断：缺成功用例明细与开发执行入口

- [诊断报告](../t65-huawei-diagnosis-20260916.md)及[新快照](../data/t65-huawei-diagnosis-20260916.json)（SHA-256 `20d6220582728897291288c3c234e3eafd955e2ddde92482b4cdad068aa3d02b`）。实时T65仍第10/26.917125，榜首69.07705，额度22/30；本次只读查询，无新候选/提交。
- 华为前三为255.2878/207.1470/122.7448，非单队孤例；我方17.5128同芯第9。剔除华为后七芯均值28.2606仍第10，第一42.475514，还需50.30%；即华为追到255.2878，八芯反事实均值也仅56.639。这是分数算术，不是可达性预测或运行时占比。
- E9/E10与11条历史华为raw_result仅有空errors/failed_cases，无成功shape/dtype/stride/T_base/T_opt。华为分数-7.91945%贡献均值-0.188275，不能将execution_time_ms当kernel耗时或断言T_opt+8.60%。已检查官方前端实际调用链及CLI，范围内未找到额外只读明细入口；历史高值仍无法区分算法与窗口。
- exact E9/E10只改变有界hidden-grid调度，逻辑hidden块/slot迭代数不变；实际目标访存与调度成本未知。E11筛选失败维持关闭。实时KernelGen鉴权正常，但公开schema无固定源码执行/IR/profile契约；项目未登记昇腾主机，现有runner仅CUDA/HIP。
- 下一步取得已有授权昇腾入口或配对case导出，绑定上述不可变源码在同设备做reference/E9/E10对照与IR/profile归因，再决定是否新改代码。本轮未连接GPU、未修改源码、未preflight/上传/POST。

## 2026-09-16 E12：修复权重舍入契约，发布就绪

- 继续复核公开reference时发现确定性错误：reference要求权重先转`down_output.dtype`再转fp32，generic与昆仑E10直接转fp32。合法fp32权重配低精度down，在抵消输入上输出错误；与华为分数回退归因无关，不宣称提速。
- 最小回归：T3/H65/K2，bf16 down两行±256、权重`[0.5005,0.4995]`；fp16 down±1024、权重`[0.5001,0.4999]`。权重和为1、各值非负且有限，先转down dtype均舍入0.5，reference全0；exact E10残差0.2048–0.2560，**每组195/195元素超0.03**，违反题面1%比例门。两dtype×两输出dtype×generic/昆仑共8条旧版失败、修复后全过；down/routes/weights均覆盖stride、输入不变和新输出。未改reference或放松公差。
- 两实现只在唯一weight load后增加`.to(down.dtype.element_ty)`，随后仍转fp32、按原slot顺序累加；AST验证其他代码一致。固定SGLang `5f6dd44edc96779d4a15331637e26e73265ff6eb` 原版post-reorder也使用`.to(InDtype)`，Ascend caller在dispatch前cast到hidden dtype；不能用caller的预处理缩窄赛题输入。
- 源码/验证commit **`134adc8e0f70f84bb5c3bc6aebc7b8f87d0cf0bc`**；generic SHA `4e13bc9262a11cf3f4006fd6cf6ecc69b1671a768e07eb68e9d2e4205bd4173d`，昆仑 SHA `e13a7000b3c034acc1a5b6b797e74855bb0b268818fa6fd0549ad2bdb40bf43d`，test SHA `db064bccc74773d00081b69a05660b2f76cfd0836885e8f4018bf2e0c128c6bc`。新增回归列入RELEASE_REQUIRED_TESTS，Black/isort/flake8/py_compile和独立审查通过。
- 复现 `artifacts/competition/t65-weight-rounding-repro-20260916/`，result SHA `d05294ef756b17165a828c2a330e1c5a9fd4adeb0e9b05b5da87fbed1d4a9170`；旧版日志 `7d591fa11a647f1e0af083d324a4d7c86e33ac9064a4ddbf915144d1270cf9e8`，新版 `388da4fb68d118df11c2f18daaf1f34e3aec44742066da2ced844051a87f6052`。远端`/tmp/flagos-t65-weight-repro.5gowjO`、PID388985、timeout180、EXIT0。
- exact release `artifacts/competition/t65e12-release-20260916/`：**10/10，0失败/错误/skip**，generic和昆仑代理均实际执行；源码/test与复现字节一致。回执SHA `1029d409db192911b5ca898680cf84142eb41e632b38537471a751131d5e6bea`，完整日志 `a5c4390fca4f02ba42d49ecd6ff51bb18f48103e39ca4633a9270d9cbb453252`。RTX5070Ti/Torch2.13.0+cu130/Triton3.7.1，远端`/tmp/flagos-t65e12-release.cBJvRB`、PID389061、timeout330、EXIT0。其他目标runtime仍未验证。
- 不可变ZIP `artifacts/competition/deepep_post_reorder/e12-134adc8/deepep_post_reorder.zip`，7858bytes，成员`deepep_post_reorder.py`/`deepep_post_reorder_kunlunxin.py`；SHA **`1fa28320977280c8bcc9ff85d89642baabb6f9556694bd1685ce459897a8f569`**，与release前dry-run一致。
- 本轮为必要正确性修复，完整release后进入一次平台判决，不以性能筛选否决契约修复。8/8且各芯≥0.1方为有效；TB按实际平台分数登记，E10保留历史成绩但不得再次发布已知有缺陷的字节。发前只读额度22/30；不把修复当作华为性能根因已解。

## 2026-09-16 13:18 E12 单次提交与后续结构筛选

- E12 submission **15913**，daily_seq9；nonce `4001ec8cf4d029ac9e9ef9236206ef2a`。live preflight tuple与上述source/test/receipt/ZIP完全匹配，upload/POST各一次，state=submitted。上传后回读7858bytes、SHA `1fa28320977280c8bcc9ff85d89642baabb6f9556694bd1685ce459897a8f569` verified；13:18:59只读状态仍queued，实际剩余额度 **21/30**。未把入队计为通过，E10仍历史平台TB。
- 证据位于 `artifacts/competition/t65e12-release-20260916/{preflight,submit,status-watch}.json`；status-watch为连续只读JSON记录流，以对应submission终态为准。exact release每路径60入口/54实际launch。
- 独立性能方向只留slot route分支→masked数据流，基于修正dtype后的E12，保持hidden/token/slot循环、grid、BLOCK、launch参数、权重舍入与累加顺序；昆仑冻结E12。来源为SGLang未合并PR22426固定head `8ec0c4987bacdf8ad8d38006707add8c5acda207`，只提取掩码路由机制，不照搬Gluon/CUDA-only路径，也不借用其14x复合收益。
- 预注册：先核对baseline IR保留路由分支且candidate真消分支，未成立即停；再保留原20primary桶及额外drop率/抵消/stride回归，5轮AB/BA，primary GM≥1.05、每轮≥1.03、每affected桶≥0.95、0spill才晋级。不是E11 hidden-loop重试；目标华为仍无同源开发通道。

## 2026-09-16 13:32 slot predication 终态：机制成立，性能 NO-GO

- 候选仅在独立目录 `artifacts/competition/t65-slot-predication-screening-20260916/`；generic SHA `be9ad2f593b4599069c9b0a4fc4b8b13b1e8e72c15a2a7b7af5138a18c0c3948`，tests SHA `3c2e40467bfd40c48ef4261c47383145b270097dbacb6a0c4017c33ec8bc8287`；昆仑冻结E12 `e13a7000b3c034acc1a5b6b797e74855bb0b268818fa6fd0549ad2bdb40bf43d`。逆变换AST证明仅route if→masked loads+select变化；plan SHA `e1be5f8701ea67f71eb9fd2bf0f47effb2274358c28e07f732ab5c6385c55073`，benchmark脚本 `d75793f1a6c8dc24850980a61f298b0ed66428e4946fab48e44564f385c023d0`。
- **13/13回归通过**，generic/昆仑各91入口/85实际launch；baseline E12同13方法通过，88次输出逐位对照一致。原10方法/reference未变；新有限值18路由case与1顺序抵消case走原reference；另12非有限值case专门验证现有invalid-slot skip语义，明确记录literal reference会传播NaN，**这12项不计为literal-reference通过**。没有删失败样本或放松公差。
- 三dtype T32/H4096/K4实际编译：TTGIR scf.if从1→0，三层动态循环不变；PTX中route load后的`setp.lt`→条件`bra`改成predicated weight/row load与`selp`保留acc。fp16/bf16总bra15→14、寄存器40不变；fp32 bra15→13、寄存器38→56，全部0spill。root按原始PTX逐项裁决后才开始计时，不能外推昇腾收益。
- 完整 **32桶×5轮AB/BA**、160对原始计时，原20primary GM **0.9815065795**，5轮GM **0.980148–0.982863**，均未达到1.05/1.03门；最差T40/H4097/K4 fp16 drop100 **0.7618450742**，bf16 drop100 **0.7792973810**，drop30也约0.823/0.829。全部32桶与E12逐位一致、0spill；按原门关闭，不挑域，不提交。
- receipt SHA `9dbb7a6ed624a26e558bc313966339c7bff510bab759962ab3c7b45d9a652b98`，verification.log `01a964803cd8eca6965eb9bb1e371ca6c5009f0ffabb99e1c92120e9ab849832`；IR probe `2bd215dab6fca1a978933fd3b640ce57299d1d4f533f0d0bf40fb3e8f80e0da7`，IR裁决 `7c23dc3b3de6ab2f275e4161bba06f1737a8333580b458a1bea34346746e291b`；benchmark `2625f31246389167fae9716639f2674eeaa7b1f6056cdaab9372f03ae43a62d4`。全部raw asm、JSON与CSV保留。远端`/tmp/flagos-t65-predication.ZcaUFk`，首段PID389223、次段PID389430，各timeout300/EXIT0，GPU已释放；主树仍E12，未建立性能候选intent/ZIP/上传。

## 2026-09-16 KernelGen 与资源续查

- 单次`generate_kernel(device=huawei)`于13:10:06–13:17:23返回，客户端未重试，服务内部报告attempts3；`passed=false`、`total_tests=0`、`NameError: torch is not defined`、全部计时null。实际覆盖未知，不把传输EXIT0当数值通过，也不把错误归到E10源码。
- 返回torch_code与triton_code逐字节相同，独立reference丢失；返回结构只有已独立修复的weight cast，没有新性能机制。生成测试只含6shape×3dtype同dtype权重，遗漏mixed dtype/stride/关键边界；benchmark变量未定义。停止生成，不晋级、不额外GPU重测这个重复修正。
- 原件 `artifacts/competition/t65-kernelgen-huawei-20260916-continue/`：request SHA `313e230a76078ebb593027f5b99a512db1cdd41161549fac18ffcf8a4a964c08`；raw response `ac9d22de54e1f935dc12d90f6ac8524c4311fcfc2ed72914d0c54683ba2deeee`；emitted source `5c307b64d21fac46fcbce091fa6e4141673269adcaa4b7e821d9f3b274a84777`，static review `c2c791226a24d59135cbe377487d35a1db80a8c069f6fd06b6319f49d62eb9c2`。
- 已只读检查现有SSH主配置及3个显式Include，无遗漏的Ascend/Huawei/NPU别名或授权入口；未探测未知主机。增量资源审计 `artifacts/competition/t65-resource-followup-20260916/resource-followup.json` SHA `8dfe2c320ce9a4ed62211cd47c45ddc1b5aa3f6a7e5ecf572dc4047b6335d5d5`。仍缺昇腾同源诊断入口，不重新申请已正常的Token。

## 2026-09-16 13:33:40 E12 等待状态：八芯 queued，尚无判决

按submit返回的URL哈希绑定watch连续只读等待900秒后退出124，共49份快照；最后一份submission15913仍queued、0/8终态、validity=pending、均值null，额度 **21/30**。这不是算子失败或通过，不能报告完整八芯闭环完成；一次性intent保持submitted，禁止重新上传/提交。`artifacts/competition/t65e12-release-20260916/platform-latest.json` SHA `20479cd9a97afdef5262cb050bec1917d29538e931190602699be5e98060d06f`，完整watch流 SHA `d5757c1d04d871927c590c62e7424218cdb69ec34029ffb30aaba795796098a7`。本地GPU作业均已结束，后续只需取该submission终态；主树E12保持契约修复，不回滚至E10缺陷字节。

## 2026-09-16 14:20 E12 部分结果：5芯通过，3芯待终态

- submission15913从queued进入evaluating；**尚非八芯有效提交**，average_speedup仍null。天数30.9596、沐曦17.885、华为15.8694、A48.967、B19.616均completed/passed；燧原与海光retry_wait，昆仑waiting_callback。它们是平台原提交内部状态，不是本任务重投；上传/正式提交仍各一次，余21/30。
- 昆仑已分配validation_id，平台显示next_status_query_at=2026-09-16T16:10:45；回调何时完成未知，不把该时间当完成承诺。未手动触发评测重试或查询未知validation端点。
- 实时榜单仍第10，历史TB E10 **26.917125**，榜首69.07705，差距未变。E12华为当前读数低于E10，但无同窗Tbase/Topt，不据此判定weight cast性能因果；主树继续保留已证实的正确性修复。
- 最新单次只读快照 `artifacts/competition/top1-followup-20260916-1349/status-15913-closeout.json`，SHA `e662ee0c52c374738bbe0012b3bf2fda5e7d904926aa08109606fe92bb8294ce`；此前240秒有界watch共11个完整snapshot，超时124，SHA `94e0c542cf0255be94e702f86e099d82290eb08353428b8b8c5c44a69ca6af3c`。watch已结束，无后台监控承诺；不重复建立intent。

## 2026-09-16 14:47 E12 更新：昆仑收集测试失败，另两芯待终态

- 原 submission **15913** 的文件 URL 哈希仍为 `3023e148ad28af32c098092ad8d9065e764f00aeaa72f8b46d1f76047295a10a`，未上传/提交新候选。平台 status=dispatching、validity=pending、average_speedup=null，6芯终态中5芯passed；燧原retry_wait、海光dispatching。历史TB E10仍26.917125、名次10，实时榜首69.07705；余21/30。
- 昆仑 validation `25d3ad022b03` 已 completed/passed=false。raw_result 是 `No test results found (empty report)`；pytest在构造 `_case(1)`、将输入搬到设备时返回 `RuntimeError -299`，日志有 `A kernel exception has occurred`、收集0项并以exit2中止。此处没有候选公开入口执行证据，不能归因于本次weight cast，也不能把NVIDIA数学通过替代目标芯结果；设备错误的责任归属仍未知。
- 天数30.9596、沐曦17.885、华为15.8694、A48.967、B19.616与上次相同。尚未取得剩余两芯终态，不把这条pending提交计为有效或最终无效，不重复提交或主动重启平台评测。
- 原始快照 `artifacts/competition/t74-output-search-screening-20260916/platform-start.json`，SHA `b021e7dedb86ee3f2f1bd2c5b415efb69f9294fb68f2295a36d53e8d67c1dfdb`；绑定文件身份的昆仑原始错误 `t65-kunlun-result.json`，SHA `ad9b84d2ccf749abd35afe3101ac8de7cdcb17ec29afbdd0b27dac21127a60cd`。本节仅更新既有提交观察，不建立新intent。

## 2026-09-16 15:05 E12 平台终态：7/8，invalid_correctness

- 同一submission15913现为completed，8/8终态，7芯passed；平台正式判定 **invalid_correctness**，average_speedup=null。新完成燧原9.0242、海光79.1944均通过，其余5通过芯片读数未变；昆仑仍为上节的测试收集阶段设备错误，0项测试，不伪称是算子数值失配。
- 快照 `artifacts/competition/t74-output-search-screening-20260916/platform-closeout.json`，observed_at=2026-09-16T15:05:50.512258+08:00，SHA `4b7cb56f0451d924d6c2c0296145d8af0edbe3fbf52be5a242b2fe594deb4ee7`；已再次绑定原file_url哈希。上传/正式提交仍各一次，余21/30，未重试。
- 历史平台TB仍E10 26.917125；主树保留E12已证明必要的weight舍入修复，不回滚有缺陷字节。此次八芯判决已收齐，但未达到8芯合格闭环；新的昆仑迭代须先获得目标运行/设备健康证据，不用同字节重发掩盖失败。

## 2026-09-17 E13：Ascend persistent vendor，已提交

- 结构（`842a8169`）：仅新增/重写 `_ascend` vendor = generic kernel 字节不变 + persistent launch（`num_vectorcore`，fallback 40；T64 E7 已证载体形态，华为 +44%）。其余成员字节冻结。host 侧 `_worker_count` 仅对 TensorMetadata mock 缺 device 做默认值护栏（T65 grid 测试需要），非计算 fallback。
- screening（RTX 5070 Ti）：unittest 10 项全绿；black/isort/flake8 过。
- release v2（source=verification commit `842a81694a9fed9530ea5ada88343eea3415ccfc`）：10 项全过 0F/E/S/X，generic/ascend 各 54 真实 launch，exit 0。回执 `artifacts/competition/persist-batch-20260917/deepep_post_reorder-verification.json` SHA-256 `ff68b95f84475bb5ecc535cecaa8df4e4d56ea9640a21b0c1efb819b08468fd3`。ascend target-runtime-unverified。
- ZIP：`artifacts/competition/deepep_post_reorder/e13-842a816/deepep_post_reorder.zip（12169 bytes）`，SHA-256 `1cc19c5a5924dc1e155fd32310a24b4cd1ec14465076d87282b40f651a8f70be`，3（generic=E12语义字节/ascend/kunlunxin=E10冻结） 成员。
- 预注册门：8/8 有效且均值 > 26.917125；华为 ≥ 35 为 persistent 轴正信号。一次候选一次判决。

## 2026-09-17 E13 单次平台提交

12:22，submission **16572**（daily_seq 2），evaluating；发后额度 28/30。八芯终态另节记录。
## 2026-09-17 E13 平台终态：8/8 有效但低于 TB，保留 E10

submission 16572 completed/valid，8/8，均值 **26.54165x < TB 26.917125**，不换 TB。逐芯（vs E10 TB）：huawei 17.513→**14.1384（-19.3%，persistent 轴负向）**；其余七芯与 E10/E12 同字节量级（haiguang 70.584/tianshu 30.8036/muxi 18.7734/card_a 49.2028/enflame 8.5384(+4.5%)/card_b 19.9366/kunlun 0.356 冻结漂移）。判读：persistent 对本题"slot 循环归约"形态无效（对照 T64 散布 +44%/T73 逐元素 +97%），40-program 双轴切分反而劣化；华为 245-279x 他队形态仍未破译。轴关闭，需全新结构证据（官方 gather/scatter best-practice 或目标芯 IR）。
