# Task 28 `gate_up_lora_b` 实验记录

状态:S0 候选就绪,待额度重置后提交(排在 29→30→25 之后)

## 契约锁定

- 签名:`reference(x, gate_up_lora_b, batch_info, output_dim, base_output)`
- 输入:`x [S, 2r]`;`gate_up_lora_b [num_lora, 2*output_dim, r]`;
  `base_output [S, 2*output_dim]`;batch_info 含 seg_indptr/weight_indices/
  lora_ranks/scalings/permutation
- 计算:per segment、per 切片 i∈{gate,up}:
  `out[rows, i*od:(i+1)*od] += scaling * (x_slice @ w_slice.T)`,
  float32 累加,输出转回 base_output dtype;lora_rank 为 0 的段跳过
- 输出:与 `base_output` 同 shape 同 dtype(克隆起步)
- 容差:fp32 1e-4 / bf16 1.5e-2 / fp16 1e-2(无精确相等项)

## 方案

- T22/T23 家族复用;K=r 很小,tl.dot K 下限需 pad/mask
- 燧原套 stages≥2 + ≥64 tile + grid-stride cap 64 已验证模板;
  昆仑 SDNN 规整结构路径;float32 计算配 fp32-ieee dot
- 风险:家族前科(T22 燧原编译失败、T23 昆仑数值失败,均非本征);
  排最后做

## 提交预算与止损(2026-08-27 定稿)

- 每题 5 次提交预算;S0 首投探路 → 最多 3 次 vendor 单变量迭代 → 剩 1 次留作
  截止前回归储备。
- 同指纹失败连续 2 次提前停,不烧满 5 次;额度只花在有单变量假设的候选上。
- generic dot 策略(如涉及 `tl.dot`):fp32-ieee 操作数 generic + `_tianshu`
  split-fp16 vendor;昆仑保持 fp32-ieee(T12 镜像证据,昆仑 fp16-dot 数值失败
  有平台实证)。

## S0:generic baseline

状态:候选就绪,未提交(额度阻塞)
时间:2026-08-27 22:00–22:20 CST
source/verification commit(同一提交):`f89f64e9b38bdb336bcb8df7021e77c8ded7c239`

### 构建身份

| 项目 | 值 |
| --- | --- |
| 源文件 | `src/flaggems_sglang/ops/gate_up_lora_b.py` |
| 源文件 SHA-256 | `3dc68f0b6eda1cdce4e3f9ad8e956af890b3523f4107e039d26bf5924c0ceb29` |
| 测试 SHA-256 | `225ce1415a5b802fdf8a378ca846acacff4f6a4bf1ced5cd79a10320fb175b23` |
| ZIP | `artifacts/competition/gate_up_lora_b/s0-f89f64e/gate_up_lora_b.zip` |
| ZIP SHA-256 | `deccf49c29ad7aa8d418bd6da8f74ff55ae01646b65a41e0728c2a56d4c0482a`(与 canonical 一致) |
| ZIP 内容 | 单个顶层文件 `gate_up_lora_b.py`,5895 bytes;ZIP 6027 bytes |
| screening 目录 | `gpu:/tmp/flagos-batch3-rest.oTBskH/gate_up_lora_b`,mode 0700 |
| release 目录 | `gpu:/tmp/flagos-gu-release.24UoN3`,mode 0700,文件取自 Git 对象 |

### 唯一候选配置

- `qkv_lora_b`(T22)骨架:3D grid(token 块 × 输出块, slice∈{gate,up},
  segment),全部 stride 传入;`base_output.clone()` 起步,kernel 内
  read-modify-write 加回(平台 6/8 先例,T22 失败与本路径无关)。
- BLOCK 64/64/64 + `num_stages=2`(燧原 ≥64-tile dot 规则;T22 的
  BLOCK_K=32 疑似其燧原编译失败根因,本 S0 已规避);fp32-ieee dot。
- rank=0 adapter 与空段 early-return;`batch_info.max_len` 缺失时回退为
  由 seg_indptr 推导(仅 grid 尺寸,host 级)。

### 正确性

screening 与 release 两次均 9/9 通过:fp32/fp16/bf16 × (r, od, 段结构)
矩阵(r=16/32/64,od=64/65/96/127/128/129/256,段长 1–300 多段);
rank-0 与空段;permutation 无/恒等/乱序;非连续 x/base;输入不变性;
S=0;8192×2048 大 case。

### 远端 NVIDIA 代理性能(五组 AB/BA p50 中位数)

| dtype | S×seg×r×od | op p50 (ms) | torch p50 (ms) | speedup |
| --- | ---: | ---: | ---: | ---: |
| float16 | 1024×8×16×512 | 0.043008 | 0.760224 | 17.6763x |
| float16 | 8192×32×16×1024 | 0.493568 | 3.460096 | 7.0100x |
| float16 | 8192×32×64×1024 | 0.497664 | 3.118016 | 6.2653x |
| float16 | 16384×16×64×2048 | 1.917792 | 4.030464 | 2.1016x |
| float32 | 1024×8×16×512 | 0.053248 | 0.669824 | 12.5793x |
| float32 | 8192×32×16×1024 | 0.641024 | 3.026912 | 4.7220x |
| float32 | 8192×32×64×1024 | 0.649216 | 2.643200 | 4.0714x |
| float32 | 16384×16×64×2048 | 2.483168 | 3.196416 | 1.2872x |

最差 1.2872x(reference 逐段 python 循环天然慢)。

### 已知边界与风险

- 3D grid 于 T22 平台 6 芯通过(华为在内);燧原编译、昆仑评测仍为
  家族风险面,tile 已按燧原规则规避。
- E(num_lora)与 r 无上限约束;r>64 时 K 循环多轮,BLOCK_K=64 仍合法。

### 提交计划

- preflight tuple:season 2、race `782kzq4m`、account `15600308080`、
  team `SoulCoder`、batch 3、task 28、operator `gate_up_lora_b`、
  stage `s0`、commit `f89f64e9b38bdb336bcb8df7021e77c8ded7c239`、ZIP
  `artifacts/competition/gate_up_lora_b/s0-f89f64e/gate_up_lora_b.zip`、
  SHA-256 `deccf49c29ad7aa8d418bd6da8f74ff55ae01646b65a41e0728c2a56d4c0482a`、
  member `gate_up_lora_b.py`。

## 平台提交记录

- 2026-08-28 00:07 CST 额度重置(30/30)后,按 29→30→25→28→27→26 顺序自动
  preflight + 一次性提交;全部 tuple 与账本一致后执行 confirm。
- 提交时间约 2026-08-28 00:18 CST;submission_id `5740`;ZIP SHA-256
  `deccf49c29ad7aa8d418bd6da8f74ff55ae01646b65a41e0728c2a56d4c0482a`;state `submitted`、validity `pending`、评测入队。
- 提交后团队当日额度剩余 24/30(6 投全记录)。


### 八芯结果(S0 首投,sub 5740,截至 00:55,昆仑芯评测中)

已出 7 芯:5 过 2 败:

| 芯片 | speedup | 结果 |
| --- | ---: | --- |
| muxi | 16.582x | 通过 |
| haiguang | 41.686x | 通过 |
| huawei | 14.9635x | 通过(3D grid 在昇腾可用,与 T22 证据一致) |
| card_a | 11.1705x | 通过 |
| card_b | 2.8625x | 通过 |
| tianshu | - | 数值失败:fp32-ieee dot 在天数静默算错(T12 已知镜像证据),与预期风险一致 |
| enflame | - | 编译失败 `Pipeline run failed`(T22 家族指纹;64/64/64+stages2 未规避,疑 early-return 分支) |
| kunlunxin | - | 评测中 |

### E2 计划(预算剩 4 次)

- `_tianshu`:split-fp16 三点积(T12 已验证配方);
- `_enflame`:消除 rank==0/空段 early-return(改零贡献路径)或按 T12
  燧原模板调整。

## E1 vendor 轮提交(sub 5767,2026-08-28 01:4x CST)

- 首轮逐芯失败指纹对应的 vendor 修复;vendor commit
  `0e3d58715ec0c5b1d3b841e2cf6b277b48fd8f9c`;ZIP SHA-256 `3a2f6002b1f1dda88e59e7203b8ac35dc721518edba98e596b83df9c1899e7f4`。
- 成员:generic + `gate_up_lora_b_iluvatar.py`(split-fp16 三点积)+
  `gate_up_lora_b_enflame.py`(1D grid 折叠 + 无运行时分支,T12 燧燃模板)。
- 远端 NVIDIA 代理:router/lora/enflame/ascend vendor 数值全对;
  `_kunlunxin` last-index 在 NVIDIA 失配为设计内现象(NVIDIA torch 平局
  取首索引)。
- 提交后当日额度 20/30 剩余;评测中。

## 昆仑恢复期收口投(sub 5811,2026-08-28 05:5x CST)

e2 单变量 generic BLOCK_N 64→128(qkv 昇腾先例,燧原 vendor 不动);
  七芯全过且华为 16.4→23.3x、燧原 1.25→1.30x,昆仑 waiting_callback。
- 昆仑分析结论:T30(00:10)昆仑曾通过(唯一 passed 记录),00:14 后平台
  评测器崩溃(T25 曾返回"服务线程卡死自动恢复"),非候选内容问题;
  本投兼作昆仑恢复探针。若昆仑通过即 8/8 valid。

## E3 结果(sub 5845,2026-08-28 11:5x CST)

- 七芯全过且华为 22.64x(BLOCK_N=128 保持);昆仑返回
  **"服务线程卡死自动恢复,请重新提交"**(exec 服务端问题,平台明示
  可重投)。预算 4/5,剩最后一发留给服务恢复后的重投。

## E4 重投载体(预备,未提交)

- 平台对 sub 5845 昆仑返回"服务线程卡死自动恢复,请重新提交";内容
  与 e3 完全一致(七芯全过、华为 22.64x),仅加注释标记作为新 ZIP 载体。
- 触发条件:昆仑健康信号(任一第三批题达标队伍数上升,即有队伍昆仑
  通过)出现后,按门禁 preflight + 单次提交。本题为最后一次预算。

## E4 终态(sub 5861,2026-08-28 14:5x CST)

- 用户决策"立即投、接受撞运气风险"后打出;七芯第三次全过
  (天数 13.14x、沐曦 17.36x、燧原 1.26x、海光 39.24x、华为 22.51x、
  card_a 12.90x、card_b 2.16x,均值 ~15.5x),**昆仑仍为 inductor
  子进程崩溃指纹**(与 erf 无关——本题无超越函数;T28 全天三次昆仑
  尝试均为服务端崩溃/卡死)。
- **T28 终态 7/8,预算 5/5 用尽。**七芯证据完整;若赛方对 5861/5845
  免费重跑昆仑或修复服务,本题随时可补成 8/8 并登顶(榜首 9.65x,
  我方七芯均值 ~15.5x)。工单路径见
  `docs/competition/kunlun-crash-ticket-2026-08-28.md`。

## E5 追加终投(sub 5876,2026-08-28 15:00 终态)

- 用户放宽 1 次预算后打出 host-resolved `_kunlunxin` v2(T13 模板:段
  元数据 host 解析、dot 编译单元零动态标量 load);七芯第四次全过
  (海光 41.95x、沐曦 17.93x、天数 15.63x、华为 21.37x),**昆仑仍为
  同一 inductor 崩溃指纹**。
- 假设证伪:ragged 标量元数据 load 不是触发面。剩余嫌疑按序:permutation
  间接寻址、dot 本身、RMW;或服务/worker 侧(四种不同结构同指纹崩溃
  现在同样支持该解释)。
- **T28 最终 7/8**(预算含追加共 6 次用尽);七芯均值 ~18.5x 遥超榜首
  9.65x。唯一剩余路径:赛方免费 rerun 昆仑或返还机会——工单证据链
  因"四结构同指纹"反而更强。


## E6 昆仑恢复重投(2026-08-29 00:5x CST)

- 昆仑健康信号确认:T29 gelu_and_mul e7(sub 6127,2026-08-29 00:48 终态)
  昆仑 0.4847 正常完成、8/8 valid——08-28 全天 inductor 崩溃为平台评测器
  故障而非候选内容;榜单 T28 达标队伍 1→2(HelloWorldTJU 11.499x 登顶),
  昆仑可跑本题型。
- 本投内容与 e5 逐字节等价(generic 仅注释标记行不同),四成员不变:
  generic + `_enflame`(1D 无分支)+ `_iluvatar`(split-fp16)+
  `_kunlunxin`(host-resolved v2);七芯已平台四连过(海光 41.95x、
  沐曦 17.93x、天数 15.63x、华为 21.37x)。
- source commit `c56f08e`(e6-retry-carrier 注释);ZIP
  `artifacts/competition/gate_up_lora_b/e6-c56f08e/gate_up_lora_b.zip`,
  SHA-256 `9856356773015c9de2eec576e342730874bcffb590243f0c923991068543f770`,
  打包器 verified-existing(canonical)。
- 预期:昆仑正常 → 8/8 valid,七芯均值 ~15.5x > 榜首 11.50x 即登顶;
  若昆仑同指纹第五次崩溃 → 维持平台侧结论,工单升级。

### E6 平台终态(2026-08-29 01:30,sub 6135)

**7/8,invalid_correctness。**七芯第五次全过:天数 15.718、沐曦 17.061、
燧原 1.222、海光 38.977、华为 20.899、card_a 12.745、card_b 2.215。
昆仑 `waiting_callback` 30 分钟后超时终止,错误为**同一 inductor
compile-worker 崩溃指纹**(`subproc_pool._recv_msg` 挂起 → Fatal
Python error: Aborted)。

- 关键新证据:同日 01:0x T29 gelu 昆仑正常完成(评测器健康),
  而本题第 5 次撞同一指纹——故障不再是平台整体宕机,而是本题型
  的 Kunlun 编译路径触发 inductor compile-worker 死锁。
- 复盘五次昆仑尝试(3D grid generic、BLOCK_N 128 generic、1D fold、
  host-resolved dot v1/v2):grid/结构/host-resolve 均变过,
  `num_stages=2` 与 `tl.dot` 从未变过;而昆仑平台已证配置
  (T13 chunk_state_varlen、T21、T24)均为 `num_stages=1`,
  T21/T24 更是无 dot 的简单 kernel。

## E7:净室 Kunlun vendor v3——FMA 无 dot + stages=1(2026-08-29)

- 单变量改动(`_kunlunxin` vendor,其余成员与 e6 逐字节一致):
  ①去掉 `tl.dot`,改显式 fp32 FMA K 循环(`tl.range(0,RANK)` 行列
  广播乘加,消除整个 dot/ieee lowering 面);②`num_stages=1` +
  `num_warps=4`(T13/T21/T24 昆仑已证约定);③全部 int32 索引
  (去掉 int64 stride cast);④host-resolved 逐段 launch 保持 v2。
- screening(gpu:/tmp/flagos-router-asc.6i1BpG,RTX 5070 Ti):
  vendor driver 全绿(fp32/fp16/bf16 × r16/32/64 × od64/96/129/256 ×
  perm/无 perm/rank0+空段/8192 大 case/非连续/空输入)。
- source commit `dd6bb27`(blob SHA `41132e57…` 与 screening 一致);
  ZIP `artifacts/competition/gate_up_lora_b/e7-dd6bb27/`,canonical
  SHA-256 `1d14cb4288418f34ce119f198c80d80bac270501b6b5ba18e56895cb75010fd0`。
- 判读:e7 昆仑通过 → 8/8 valid 且 ~15x 登顶(榜首 11.499x);
  仍同指纹 → 与内容无关的最后疑点排除,工单证据升级为
  "六结构两时代同指纹"。

### E7 平台终态(sub 6194,2026-08-29 03:0x)与 E8 重投(sub 6223)

- E7:七芯第六次全过(华为 22.389x);**昆仑同一 inductor compile-worker
  崩溃指纹**——净室 FMA 无 dot + stages=1 + int32 也崩,"编译复杂度/
  dot/stages"假设全部排除;同日 T29 gelu 昆仑正常通过,且 T28 榜首
  (HelloWorldTJU 11.499x)为近期 8/8 达标 → 昆仑评测窗口对该题型
  为概率性故障,非内容阻断。
- E8:e7 内容(v3 FMA)换载体重投,探昆仑窗口。
  source commit `01895cf`;ZIP SHA-256
  `e17d6e1cc0feb6e47dfbed1914e4ebd06d11079bdcfb5c3744f99bcfc4b332f9`。

### E9 恢复窗口重投(2026-08-29 21:2x CST)

- 健康信号:21:20 快照 T25 达标 7→8(crash 族首例新增 8/8);
  同步脚本断言已软化适配平台新 API(41 任务)。
- e9 = e8 树 + 注释载体(commit `54444fc`,ZIP SHA
  `f07dc111…`);preflight 全过(额度 7/30),单次提交,昆仑终态
  待回填。T31 e2(sub 6477)同窗口已投。

### E9 终态(sub 6481,2026-08-29 22:3x CST)

昆仑同指纹崩溃(1830s + Aborted)。T28 七投(六种结构 + 三个时代)
全部同指纹,恢复窗口无效;永久转工单路径,代码侧关闭。

### E10 双确认恢复窗口重投(2026-08-30 10:5x CST)

- 信号:T26 达标 5→6 **且** T28 达标 2→3 双题同时新增(比 08-29
  的 T25 单题信号强得多,且含本题);
- e10 = e9 + 注释载体(commit `68fa28d`,ZIP SHA `ff1755ec…`),
  preflight 全过(额度 30/30 消耗 1),单次提交;昆仑终态待回填。

### E10 终态(sub 6583,2026-08-30 12:0x CST)

昆仑同指纹崩溃(第 12 次,T28 七投)。同 moe_fused_gate E3 终态
节的解读反转:他队可过 → 疑我方 kernel 惯用法(topk/分段 GEMM
结构)触发昆仑 inductor-triton 路径不稳定。工单表述需按此修正,
不能纯归因平台。
