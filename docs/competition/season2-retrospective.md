# FlagOS 第二季算子赛收官复盘(2026-08-27)

> 范围:第二批 Task 08–24 全部 17 题(第一批未参与)。数据来源:17 份实验
> 账本、平台实时 API 终榜(2026-08-27 14:30 CST)、全部本地 commit 与不可变
> ZIP 证据链。本文档是跨阶段资产,供 review/PR 阶段与下一季直接复用。

## 1. 终局战绩

**9/17 题 8/8 有效,8 题尝试后按规则停止,1 题(T18)平台侧死锁未解。**
当日额度 30 次全部用尽,提交窗口 19:59:59 截止。

| Task | 算子 | 我队最佳 | 排名/达标队 | 榜首 | 关键胜负手 |
| ---: | --- | ---: | --- | ---: | --- |
| 13 | chunk_state_varlen | **166.4583x** | 5/6 | 707.0x | host 解析变长边界+逐段规整 dot,昆仑起死回生 |
| 17 | embedding_lora_a | **14.1618x** | 6/7 | 25.6x | GCU i32 metadata;route+gather 双 kernel;海光 warps |
| 20 | mamba_layernorm_gated | **4.3783x** | 7/8 | 7.4x | Ascend capped grid-stride;GCU 多行 persistent |
| 12 | chunk_state | **4.0372x** | 9/12 | 15.3x | Ascend Cube 低精度 dot(6.4x);GCU fold-64 |
| 21 | moe_sum_reduce | **3.0637x** | 10/11 | 3.7x | 六 vendor 全家桶;GCU reduction tile 峰值 8192 |
| 24 | softcap_out | **2.2400x** | 11/14 | 58.6x | GCU 32K pointwise tile;昆仑 BLOCK 曲线 |
| 09 | bmm_chunk | **1.2830x** | 12/13 | 4.0x | 天数 split-fp16;GCU 64/64/128+stages2 模板 |
| 08 | apply_token_bitmask | **4.6869x** | 15/17 | 709.4x | 单 pass 融合即天花板;grid 折叠恢复两芯 |
| 19 | fused_rmsnorm | **4.5566x** | 15/17 | 5.6x | GCU 默认 launch +38%;昆仑 multi-row |

无效终态:T10/T11/T16(8 芯正确但燧原/昆仑低于 0.1x 门槛)、T14/T15/T22/T23
(7/8,单芯结构性失败)、T18(pending_challenge,燧原评测停摆+昆仑 reference
侧崩溃,0/28 队解出)。

**相对排名规律**:凡"参考实现带 Python 循环/同步"的题(T13/T22/T23)榜首
都是百倍级——高倍数来自打掉串行结构,而非 kernel 微调;我们在 T13 吃到了
这类红利(第 5 名),在 T22/T23 倒在昆仑编译侧。

## 2. 跨芯片知识库(核心资产)

以下全部是平台验证事实,附任务出处。**这是下一季最先要读的一节。**

### 2.1 燧原 Enflame/GCU(投入产出比最高的芯)

| 事实 | 证据 |
| --- | --- |
| **i64 IR 是编译失败第一根因**:int64 metadata/显式 i64 stride cast 触发 `Pipeline run failed`;wrapper 有界降 int32 + 删 i64 cast 可把全 case 编译失败恢复为正确 | T17 E2a-i32(0→0.39x)、T22 E3-i32(0→4.75x) |
| **pointwise 大 tile 单调上升至官方上限 32K**:256→0.35x、4096→1.19x、32768→2.98x | T24 S2/S8(2.15 倍) |
| **reduction(保活累加器)tile 峰值 8192**:256→0.21、2048→0.39、8192→**0.99**、16384→0.74 | T21 E8/E14/E15 |
| **dot kernel 病理与解法固定**:stages=1 千倍慢(0.001x);`64/64/128 + stages2 + fold cap64` 三次平台验证(+26%~6.4 倍) | T09、T12 E3/E4 |
| **物理并行度≈12 CTA**:grid 放开到 65535 反而 -24.5%;cap 12 贴近硬件 | T24 S7(反证)、T08 S1 |
| **官方默认 launch(不传 warps/stages)+38%**:kernel 与 generic 逐字一致即可 | T19 E5(1.50→2.08x) |
| 运行时分支/cumsum/标量 masked load 触发 Pipeline 失败;grid.x≤65535、grid.y≤255 | T14/T18/T21 多处 |
| 32×32+fp16-dot 组合编译失败;大 tile 迁移不到间接 load(bitmask gather -5.8%) | T09/T08 E5 |

### 2.2 昆仑芯 Kunlunxin/XPU

| 事实 | 证据 |
| --- | --- |
| **2D grid 展平总数 ≤65535(编译期)**;超限走折叠或加大 BLOCK | T21 S0c→S3 四组证据链 |
| **BLOCK 是唯一有效调参轴**(`num_warps/num_stages` 为 invalid 参数);elementwise 曲线 256→1024→4096 未饱和 | T24 S2d/S2e、T21 |
| **ragged/varlen/cumsum 结构固有慢或编译爆炸**(0.012–0.016x);**解药=host 解析边界+规整 direct dot**:T13 从五 case 全数值失败到 55.37x | T13 E2/E3/E4、T10/T11 |
| fp16 操作数 `tl.dot` 数值失败,fp32-ieee 通过(与天数互为镜像) | T12 E5 |
| dot 走 SDNN 路径可过(规整结构);masked-memory 两种官方模式(mask-zero/legacy env)都修不了 pack/scatter 数值 | T23 E10/E11 |
| 编译期 1830s 超时+`Fatal Python error: Aborted`(栈在 Inductor compile worker)与提交字节无关——平台侧,四题复现 | T15/T16/T18/T22 |

### 2.3 华为 Ascend

| 事实 | 证据 |
| --- | --- |
| **grid 展平 coreDim ≤65535**;解法 capped grid-stride(`min(total,4096)`,或 vendor 特化 48/12/65535),**七次平台验证** | T08/20/21/12/09/13/17 |
| **flash 型 kernel 整行重复 bug**(launcher 钳到物理 Vector Core 数);解法 `num_vectorcore` 物理 worker + kernel 内 grid-stride | T15 E4、T16 E2 |
| UB 溢出报 `ub overflow ... bits`:tile 收缩 `block_h ≤ 512//block_size` | T10 S2 |
| **官方低精度 Cube dot 可迁移**:64×64×64 使 chunk_state 华为 0.33→2.12x(6.4 倍) | T12 E7 |
| BLOCK 512 收益不普适(T24 +19.9% 成功,T08 -0.7%、T21 +35% 成功)——逐题验证 | T24 S4、T08 E4、T21 E5 |

### 2.4 天数智芯 Iluvatar

- **fp32 操作数 `tl.dot` 静默不可执行**(kernel 不跑,输出=未写内存,错误指纹恒定):T12 三假设同指纹定位。解法 fp16 操作数+fp32 累加,或 split-fp16 三点积(fp32 容差)。
- 低精度 dot 收益大:T12 E5 +282%(1.98x)、T22 47.8x。

### 2.5 沐曦 MetaX

- **官方 launch policy 有效**:T21 E6 BLOCK1024/8warps(Qwen shape 门控)+10.25%;但 T19 默认 launch 中性——收益来自"官方为该算子调过的形状档",不是默认值本身。

### 2.6 海光 Hygon

- `@triton.autotune` 与显式 `BLOCK_SIZE=` kwarg 双重绑定 → 平台 `TypeError`(FlagTree AABS 会把 kwarg 写回 config);**修法:BLOCK 走位置参数**(T19 E3)。
- gather 类 warps 2 档(T17 E4)中性偏正;reduction autotune 四档(T21 E9)中性。
- 串行 FMA 归约树敏感:T18 海光 740→3 错(T09 家族知识)。

### 2.7 国际通用 A/B(card_a/card_b)

- card_a:fp16 dot 解锁张量核心 +282%(T12);选 `_nvidia` 后缀(T18 证明)。
- card_b:按 `_amd` 建 fallback 且不得加 `_nvidia`;串行归约族 serial<halves<evenodd(T18 E10–E12);split-FMA 反而恶化。

### 2.8 dot 操作数 dtype 兼容矩阵(最重要的一张表)

| 芯片 | fp32-ieee dot | fp16 操作数 dot |
| --- | --- | --- |
| 天数 | **静默不可执行** | ✅(或 split-fp16) |
| 昆仑 | ✅ | **数值失败** |
| 燧原 | 病理慢 | ✅(需 64 tile+stages2) |
| card_a | 慢 | ✅(+282%) |
| 沐曦/国际B | ✅ | 性能回退(-30%/-56%) |

→ generic 低精度 dot 必须搭配昆仑 fp32 回退 vendor。

## 3. 算法形态 × 解法模板

| 形态 | 模板 | 出处 |
| --- | --- | --- |
| 纯 pointwise | GCU 32K tile;昆仑 BLOCK 4096+;单 pass 融合即上限 | T24、T08 |
| 行归一化/reduction | GCU 峰值 8192 tile 或官方默认 launch;昆仑 multi-row 仅 shape 命中时 | T21、T19、T20 |
| dot-GEMM 规整 | 按 2.8 矩阵逐芯选操作数精度;GCU 64/64/128+stages2+fold64;Ascend Cube | T12、T09 |
| **varlen/ragged** | **host `tolist()` 解析边界 → 每段规整 direct dot,metadata 全部出 kernel** | T13 E4(166x) |
| gather/indirect | i32 化 metadata;无分支化;route+gather 双 kernel | T17、T22 |
| flash attention | Ascend 物理 worker 必配;昆仑标量 sum+64 对齐;GCU 不稳定 | T15 E4、T16 E2 |
| 串行递归 | 逐字复现 Torch 归约序(k%4 FMA 链)+libdevice exp | T18 E4(仅 card_a 清零) |
| cumsum 家族 | 昆仑/燧原 lowering 固有瓶颈,**需两阶段分块扫描算法重写**(未验证) | T10/T11 四轮 |

## 4. 有效方法论(流程层)

1. **闭环纪律**:锁契约 → generic 单变量保守版 → 平台反馈 → 单芯 vendor →
   one-shot 预注册晋级门 → 触发即停。17 题全部走完,无额度浪费在伪新候选。
2. **team-best 计分 = 无下行风险**:激进单芯候选只赚不赔(T21 E14/E15 连打、
   T24 S 系列连打全部成立)。
3. **历史提交是资产**:E14 把 E8 被噪声淹没的燧原信号从历史 commit 里捡回来
   +4.7 倍。被"冻结芯噪声"否决≠假设被否决,要区分芯片级信号与平均值信号。
4. **跨任务根因迁移是最廉价的重开依据**:T17 i32 → T15/T22;T24 BLOCK 曲线 →
   T08;T15 结构修复 → T16。但迁移收益不保证(T08 E4 Ascend512 失败)。
5. **错误指纹诊断法**:恒定 mismatch 计数跨三假设不变=kernel 静默未执行;
   相对差 1e7 量级=读到无关内存;同字节同判定可作对照(T18 E11)。
6. **多芯机会合并一次提交**:单芯 +26% 会被另一芯 -9% 噪声淹没(T12 E4 教训)。
7. **远端门禁三件套**(screening/release/resource audit)+ 静态 AST 等价检查,
   大量候选在消耗额度前被拦截。

## 5. 反模式清单(代价最贵的排前)

1. **NVIDIA 代理外推调度类信号**:T24 S7 CUDA 5x → GCU -24.5%;T16 代理
   9.6–19.1x → 实际 0.035x(差两个数量级)。代理只做语法/数值/资源灾难门。
2. **AST/静态测试不加载模块**:T19 E1 海光 autotune 缺陷平台爆炸。之后规
   定:每个 vendor 必须真实 `importlib` 加载+reference 数值回归。
3. **autotune + kwargs 双重绑定**(2.6)。
4. **无条件 grid 折叠**:benchmark 未超限时折叠循环是纯开销(T11 E1a 华为
   0.0255x);要 hybrid 门控或先确认规模。
5. **大 tile 迁移到间接 load**(T08 E5)与非 pointwise 形态(T21 E15 见顶)。
6. **数学等价≠逐字节一致**:T12 E6 被"等价改写"拖累,E6r 回退精确字节 +82%。
7. **一手 GitHub 证据不能替代平台 harness**(T14:上游作者 12/12 我们仍超时)。
8. 平台侧故障(reference OOM/评测器停摆/Inductor 崩溃)与提交字节无关,不为
   其重传相同字节(T08 E3、T18、T19 E6)。

## 6. 平台元规则

- 计分按团队历史最佳(`is_team_best`),平均=8 芯算术平均,**任一芯 <0.1x 或
  错误即 invalid**(T16 展示 14.59x 仍无效)。
- 每日额度 30 次(季初 15);平台评测队列拥堵时单提交可 >1h;评测器停摆指纹
  =`retry_wait` 且 `last_progress_at` 冻结。
- ZIP 只认 UTF-8 `.py`、generic `<operator>.py`、vendor `<operator>_<后缀>.py`;
  平台按后缀路由,card_a 可被 `_nvidia` 命中。
- vendor 文件必须自包含(self-contained)、同签名同 `__all__`。

## 7. 下一阶段行动(review 08-28~09-03,PR 09-04~09-10)

1. **证据链归档核对**:9 个有效任务 × (source commit、ZIP SHA、平台
   submission ID、逐芯表)已在账本;review 前逐项复验一遍 ZIP 可再生成。
2. **PR 候选优先级**(上游 FlagGems-sglang,Draft PR #32 softcap 已开):
   - 高价值:T13 host-resolved varlen(Torch 循环→单 kernel,通用性强)、
     T17 route+gather 双 kernel、T24 GCU 32K tile 策略。
   - 配套:第 2 节知识库中**通用性结论**(i32 化、grid 折叠阈值、autotune
     位置参数)适合作为 backend 公共修复 PR,逐芯 vendor 则归属 FlagGems
     各 backend 目录。
   - 注意上游化清理:去掉竞赛保守配置(如无条件小 BLOCK)、补齐非竞赛 shape
     的门控。
3. **T19 5343 终态补录**:18:16 自动检查已挂;若停摆未恢复,review 期首日
   人工再查一次并关闭账本。
4. **T18 遗产**:E4 的 Torch 归约序复现 + 按芯归约树分发是独立可发表的工程
   结论,值得在 PR 阶段以 issue 形式反馈给 FlagTree/FlagGms。

## 8. 若有下一季:前置清单

1. **开赛前直接建好**:平台 CLI+token 流、确定性打包器、远端 GPU 门禁环境、
   per-chip vendor 模板库(第 2/3 节即为初版)。
2. **首批提交策略**:generic 保守版全覆盖 → 用一次提交换八芯指纹 → 按第 2
   节对号入座上 vendor;参考实现含 Python 循环的题优先(倍数天花板高)。
3. **预算分配**:每题硬上限 2 次+全局 20% 额度留给"已证伪但芯片级信号为正"
   的重开(第 4.3 条);每日预留 2 次截止前回归。
4. **需要在下一季前补的能力**:昆仑/燧原 cumsum 的两阶段分块扫描实现(本季
   四轮未解);GCU flash 类 kernel 的稳定性根因(平台侧线索已归档)。

## 9. 外部资料与情报(2026-08-27 agent-reach 全网检索)

### 9.1 官方赛制与规范(必读)

- [FlagOS X SGLang 算子优化挑战赛官宣(SegmentFault)](https://segmentfault.com/a/1190000048185827):
  完整赛制——200+ 题分 13 批次、每日 15→30 次额度、批次周期 7 天;
  **PR 规范**:攻占团队须在窗口期提交 PR 至 FlagGems-sglang,命名
  `[FlagOS Competition-Track1] Add [Kernel Name] Triton Kernel for sglang`,
  签 CLA,平台提交与 PR 字节一致性是评奖必要条件。三大奖项:全域攻占/
  攻克突破/单题极致性能(单题最高万元)。
- 比赛页:<https://flagos.io/race-detail-season2?id=782kzq4m>;
  KernelGen 上海站(历史赛):<https://kernelgen.flagos.io/challenge>。

### 9.2 最值得精读的技术分享(与本队经验互证)

- **[sinpeyw/flagos-kernel-challenge-shanghai-2026](https://github.com/sinpeyw/flagos-kernel-challenge-shanghai-2026)**:
  上海站 72h 三题获奖方案全开源,含技术报告与获奖 PDF。方法论比我们高阶的
  三点值得吸收:
  1. **算法级 Roofline 先行**:Task01(融合 RMSNorm+量化,4.38x rank1)先算
     AI=0.666 FLOP/Byte 对比 A100 ridge 12.54,推出"带宽主导→必须单遍四输出
     直写,任何第二遍/scratch 都必然劣化"的设计约束。
  2. **按 shape 分层路由**:小/中/大 M 分别走 launch 敏感/过渡/带宽路径,
     不同芯片不同方案(Ascend 串行小 grid scan、GPGPU 并行 scan)。
  3. **数学专化降复杂度**:Task03(MLA backward,54x rank2)用各向同性近似
     +suffix scan 把 O(S²) 主路径降到 O(S)——这是"串改并行"之上的
     "改数学"层,本季我们从未触及。
  - 其"昇腾双 AIV 独立行映射"即本队 T15/T16 发现的 `num_vectorcore` 物理
    worker 问题——我们的经验与冠军方案在这一点互证。
- **[iamsuperfly/Flag-Operators](https://github.com/iamsuperfly/Flag-Operators)**
  (第一季 Track1):static `@triton.jit` 替代 codegen、原生 `tl.atomic_*`、
  5 档 autotune;`chunk_gated_delta_rule` 等新算子实现可参考。

### 9.3 编译器/架构背景(理解跨芯行为的底层解释)

- [FlagTree/FLIR 架构文(SegmentFault)](https://segmentfault.com/a/1190000047725977):
  16 后端单仓库;GPGPU 走 TritonGPU IR→LLVM,NPU(AIPP/可重构)走
  FLIR→硬件 IR;**非结构化访存(sgather/scatter)只能收敛到三条有限降级
  路径**——这解释了我们 varlen/ragged 类 kernel 在昆仑的系统性困难;
  多输出规约初值与离散 mask 是官方承认的难点。
- [Triton-TLE 与 AscendNPU IR 适配](https://segmentfault.com/a/1190000047702113);
  [FlagTree 官方文档](https://docs.flagos.io/projects/FlagTree/en/latest/user_guide/user-guide.html)
  (目录级,兼容性细节需查子页/源码)。
- [量子位:国产 GPU 开源局报道](https://www.qbitai.com/2026/05/417791.html)(生态背景)。

### 9.4 官方仓库与工具

- [flagos-ai/FlagGems](https://github.com/flagos-ai/FlagGems)(497 个 Triton 算子,
  vendor 策略的事实标准)、[FlagGems-sglang](https://github.com/flagos-ai/FlagGems-sglang)
  (本季 PR 目标;当前竞赛 PR 仅我队 #32 softcap 与官方 #31 context-attention,
  **PR 窗口 09-04 开后才会有各队情报,届时应系统性扫描**)、
  FlagGems-Experimental/vllm、[FlagTree](https://github.com/flagos-ai/FlagTree)。
- [flagos-ai/skills](https://github.com/flagos-ai/skills):官方 kernel dev/
  perf tuning skills,下一季前值得对照本队工作流查漏。

### 9.5 AI 写 Kernel 官方验证(CSDN 预告)

[FlagOS 210 算子验证预告](https://flagos.csdn.net/6a0d44c7662f9a54cb75d141.html):
KernelGenBench(210 题 × 6 芯 × 150B token)的结论在直播回放中,待追。

### 9.6 对下一阶段的直接启示

1. 上海站报告的 Roofline→shape 路由→数学专化三层法,应并入第 8 节的下一季
   前置能力清单;特别是"改数学"层,是超越 vendor 调参的下一级杠杆。
2. PR 窗口(09-04~09-10)开启后第一时间扫描 FlagGems-sglang 的
   `[FlagOS Competition-Track1]` PR——那是各队最强解的免费情报,直接对照
   本队 vendor 策略找差距。
3. 非结构化访存的 FLIR 三条降级路径是昆仑 ragged 慢的官方级解释,cumsum
   两阶段重写方案设计时应避开 scatter/gather 形态。

## 10. OPSEC 与痕迹管理教训(2026-08-27 事故复盘)

**事故**:竞赛研究分支 `research/season2-batch2`(含全部 17 份账本、所有
vendor 解法、跨芯策略)于 08-26 03:48 被推到**公开 fork**
`liangkw16/FlagGems-sglang`,暴露约 1.5 天后才被发现并删除。同日确认
Draft PR #32(S0 baseline)全程公开可见。

**核查与清理结果**:fork 的 fork 数为 0(无持久副本);已删除两个分支与
PR 留言;fork 现仅剩上游镜像分支。**不可清除**:PR 记录本身、上游
`refs/pull/32/head` 提交引用(GitHub 在基仓库永久保留)、外部爬虫存档、
commit 中的企业邮箱。

**下一批起强制执行的三条规则**:

1. **赛中零公开 push**:竞赛期间研究分支/账本只留本地,严禁 push 到任何
   public remote(包括自己的 fork);远端验证一律走 SSH 主机。push 只发生在
   PR 窗口期、且只推干净的单算子分支。
2. **开赛前 remote 体检**:`git remote -v` 确认无 public fork 指向;fork 若
   保留仅作 PR 跳板,开赛前清空非镜像分支。
3. **提前开 PR = 提前泄题**:官方 PR 窗口(评审后)才提交;Early draft PR
   会把解法公开给全部对手(#32 的教训)。

## 11. PR 阶段作战手册(09-04~09-10 窗口)

1. **命名必须合规**:`[FlagOS Competition-Track1] Add [Kernel Name] Triton
   Kernel for sglang`(#32 未按此命名,已关闭,正式 PR 从新分支重开)。
2. **字节一致性是评奖必要条件**:PR 内容必须与平台最终提交 ZIP 逐字节一致
   ——以账本中的 source commit 为准(如 T24=S8 `b2a249b`)重建干净分支,
   不夹带 docs/竞赛资料。
3. **结构参考官方样例**:维护者 silu_and_mul 示例 PR #22/#26(competition
   stubs 三层结构),以及 FlagGems 主仓库另一赛事的十几个
   `[FlagGems Operator Development Competition]` PR。
4. **CLA 先签**:CLAassistant 在 PR 上检查签署,提前完成避免阻塞。
5. **09-04 当天侦察**:批量扫描 `[FlagOS Competition-Track1]` PR——第一、
   二批全部获奖实现集中涌现,对照本队 vendor 策略找差距,直接反哺第三批。
6. **fork 卫生**:正式 PR 从干净 fork/分支发出;赛前遗留的 fork 若不需要,
   `gh auth refresh -s delete_repo && gh repo delete` 清除。

## 12. 下一批(batch 3+)开赛日 checklist

1. **读题分类**(半天内):每题按第 3 节形态矩阵归类,标出"参考实现含
   Python 循环/`.item()` 同步"的机制 A 题——倍数天花板最高,优先。
2. **模板直接上身**:按第 2 节知识库对号入座——燧原 i32 化+大 tile、昆仑
   host 解析边界、华为 grid 折叠、天数 fp16 dot、card_a `_nvidia`/card_b
   `_amd` 后缀,首轮 ZIP 即带预防性 vendor(T12/T13 先例证明可省 1-2 轮)。
3. **额度纪律**:每题 2 次硬上限+全局 20% 留给"均值门失败但芯片信号为正"
   的重开;每日 2 次截止回归储备;team-best 计分下激进单芯候选无下行风险。
4. **门禁流水线就绪**:打包器/平台 CLI/token/远端 GPU 环境已建成;新算子
   只需 30 分钟接入 screening 模板。
5. **上海站三层法**(第 9.2 节)作为性能设计默认流程:Roofline 先行 →
   shape 分层路由 → 最后才考虑数学专化。
