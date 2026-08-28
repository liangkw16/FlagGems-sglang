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

