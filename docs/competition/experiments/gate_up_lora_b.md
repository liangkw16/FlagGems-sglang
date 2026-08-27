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
