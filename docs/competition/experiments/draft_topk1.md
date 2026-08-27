# Task 25 `draft_topk1` 实验记录

状态:E1 候选就绪(两阶段并行 argmax,取代 S0 成为拟提交版本);待额度
重置后按 29 → 30 → 25(e1) 顺序提交

## 契约锁定

- 签名:`reference(next_token_logits, positions, draft_tokens=None, draft_token_column=0)`
- 输入:`next_token_logits [B, V]`;`positions [B]` int64;`draft_tokens [B, D]`
  int 或 None;`draft_token_column` int(默认 0)
- 输出:`(topk_p, topk_index, out_positions, out_draft_tokens)`;
  `topk_p` = ones `[B,1]` float32;`topk_index` = argmax int64 `[B,1]`;
  `out_positions` = positions+1;`draft_tokens` 为 None 时第四项为 None,
  否则 clone 后第 `draft_token_column` 列写入 topk_index
- 容差:fp32 1e-4 / bf16 1.5e-2 / fp16 1e-2;`topk_index` 与
  `out_draft_tokens` 要求精确相等
- 关键语义:argmax 不经 softmax;平局取首索引(torch.argmax 语义),
  fp16/bf16 大 V 下平局常见,严格大于更新模式必须复刻

## 方案

- 每行一 program,V 分块循环归约 argmax(严格大于保首索引);
  `ones`/`positions+1` 小 kernel;draft 缓冲区 copy kernel + 列覆盖
- 纯归约无 dot;昆仑 BLOCK 轴、燧原无分支风险

## 提交预算与止损(2026-08-27 定稿)

- 每题 5 次提交预算;S0 首投探路 → 最多 3 次 vendor 单变量迭代 → 剩 1 次留作
  截止前回归储备。
- 同指纹失败连续 2 次提前停,不烧满 5 次;额度只花在有单变量假设的候选上。
- generic dot 策略(如涉及 `tl.dot`):fp32-ieee 操作数 generic + `_tianshu`
  split-fp16 vendor;昆仑保持 fp32-ieee(T12 镜像证据,昆仑 fp16-dot 数值失败
  有平台实证)。

## S0:generic baseline

状态:候选就绪,未提交(额度阻塞)
时间:2026-08-27 21:40–22:00 CST
source/verification commit(同一提交):`c4edba73be9e17375b83720f9d53187b1976e854`

### 构建身份

| 项目 | 值 |
| --- | --- |
| 源文件 | `src/flaggems_sglang/ops/draft_topk1.py` |
| 源文件 SHA-256 | `1d95a67cee636db175b7cd8c9ead088cf8d2ec7d9d8e4b9ce63b74cf4a780672` |
| 测试 SHA-256 | `652c1f9cac23e6f994f9b7a4684b87492c895188c1ce1ea92136c3b71e27817f` |
| ZIP | `artifacts/competition/draft_topk1/s0-c4edba7/draft_topk1.zip` |
| ZIP SHA-256 | `9d010d3e54786d4593f36c12d4c743e847fd38389791dce828caf74aec4f5662`(与 canonical 一致) |
| ZIP 内容 | 单个顶层文件 `draft_topk1.py`,4591 bytes;ZIP 4717 bytes |
| screening 目录 | `gpu:/tmp/flagos-draft-topk1.J1v1yk`,mode 0700 |
| release 目录 | `gpu:/tmp/flagos-draft-release.eBPfz5`,mode 0700,文件取自 Git 对象 |

### 唯一候选配置

- 三 kernel:①每行一 program,BLOCK_V 1024 分块串扫 V,块内
  `max`+首索引(严格大于跨块更新保首索引),fp16/bf16 平局语义与
  torch.argmax 一致(代理回归证实);②flat meta kernel 写
  `positions+1` 与 ones;③draft copy kernel 平铺 B×D,列等于
  `draft_token_column` 处写 topk_index(cast 到 draft dtype,镜像 torch
  赋值语义)。
- 全部 grid `min(cdiv, 65535)` + stride 折叠;BLOCK 1024;无显式
  num_warps/num_stages;无 try/except、设备判断或 PyTorch fallback。

### 正确性

远端环境同 T29。lint:isort/flake8 远端通过,black 25.12.0 本地通过。
screening 与 release 两次均 10/10 通过,覆盖:三 dtype × (bs, vocab) 边界
(含 V=1、1023/1024/1025、50000、128256);fp16 重复值与显式平局回归
(900 与 3 同为最大 → 3);draft int32/int64 × 列 0/中/尾;draft=None;
非连续 logits/positions/draft;输入不变性;B=0;70000 行折叠路径。

### 远端 NVIDIA 代理性能(五组 AB/BA p50 中位数)

| dtype | B×V | op p50 (ms) | torch p50 (ms) | speedup |
| --- | ---: | ---: | ---: | ---: |
| float16 | 4×128256 | 0.081952 | 0.036896 | 0.4502x |
| float16 | 64×128256 | 0.096160 | 0.052320 | 0.5441x |
| float16 | 256×128256 | 0.133056 | 0.119648 | 0.8992x |
| float16 | 1024×32000 | 0.106480 | 0.103456 | 0.9716x |
| float16 | 4096×32000 | 0.342048 | 0.349312 | 1.0212x |
| float32 | 4×128256 | 0.096192 | 0.039904 | 0.4148x |
| float32 | 64×128256 | 0.104416 | 0.063488 | 0.6080x |
| float32 | 256×128256 | 0.178112 | 0.195584 | 1.0981x |
| float32 | 1024×32000 | 0.182240 | 0.187392 | 1.0283x |
| float32 | 4096×32000 | 0.658400 | 0.664608 | 1.0094x |

### 已知边界与 E1 假设

- 小 B(V 大)下每行单 program 串行扫 V,SM 占用不足,加速比 0.41–0.90x:
  高于 0.1x 门槛但拖累均分。E1 单变量假设:两阶段 argmax(kernel1 按
  (行, V 块) 并行写部分 max/idx 工作区,kernel2 归约),预期小 B 恢复
  ≥1x。S0 先按现状投,以平台逐芯结果决定 E1 优先级。
- 大 B(≥1024)已 ≥1x,不受 E1 影响(声明 affected:B<512 的 case)。
- topk_index/out_draft 精确相等已由平局回归背书;torch 平局语义若在
  某芯的 torch 版本不同,以平台失败详情为准再修。
- int32 索引上限 2^31 输出元素。

## E1:两阶段并行 argmax

状态:候选就绪,未提交(额度阻塞)
时间:2026-08-27 23:00–23:55 CST
source commit:`b24781fdbfb8b938e7461fb3bf3046cd6a541e25`
verification commit:test 沿用 `c4edba73be9e17375b83720f9d53187b1976e854` 中
已提交字节(SHA 未变:测试与 S0 相同)

### 构建身份

| 项目 | 值 |
| --- | --- |
| 源文件 | `src/flaggems_sglang/ops/draft_topk1.py` |
| 源文件 SHA-256 | `4ad8c2418f891e61fd38cf0a023bf45536d9b920c1f3efdeed6c659c86575849` |
| 测试 SHA-256 | `652c1f9cac23e6f994f9b7a4684b87492c895188c1ce1ea92136c3b71e27817f`(同 S0) |
| ZIP | `artifacts/competition/draft_topk1/e1-b24781f/draft_topk1.zip` |
| ZIP SHA-256 | `db5f111ce4a50d28b6566ab803f8de380056a27ea6b176e7de62d0a076e7f9c8`(与 canonical 一致) |
| screening 目录 | `gpu:/tmp/flagos-batch3-rest.oTBskH/draft_topk1`(round2) |
| release 目录 | `gpu:/tmp/flagos-rel3.Fp3vo7/draft_topk1`,文件取自 Git 对象 |

### 单变量改动(相对 s0-c4edba7)

- 仅 argmax 阶段:行内串行 chunk 扫描 → scan kernel 按 (行, V 块) 全并行
  写 `[B, n_chunks]` (max, first-idx) 工作区 + finalize kernel 逐行向量化
  归约 chunk(BLOCK_C=pow2(n_chunks) 单次 axis-0 归约)。
- meta/draft kernel 与 wrapper 其余部分逐字节不变;测试不变
  (10/10 两轮均过,含平局/折叠/非连续回归)。

### 远端 NVIDIA 代理性能(五组 AB/BA p50 中位数)

| dtype | B×V | op p50 (ms) | torch p50 (ms) | speedup | S0 speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| float16 | 4×128256 | 0.010240 | 0.036864 | 3.6000x | 0.4502x |
| float16 | 64×128256 | 0.030720 | 0.052368 | 1.7047x | 0.5441x |
| float16 | 256×128256 | 0.098304 | 0.118784 | 1.2083x | 0.8992x |
| float16 | 1024×32000 | 0.098304 | 0.103424 | 1.0521x | 0.9716x |
| float16 | 4096×32000 | 0.344064 | 0.350208 | 1.0179x | 1.0212x |
| float32 | 4×128256 | 0.012288 | 0.039936 | 3.2500x | 0.4148x |
| float32 | 64×128256 | 0.053248 | 0.063488 | 1.1923x | 0.6080x |
| float32 | 256×128256 | 0.182272 | 0.195584 | 1.0730x | 1.0981x |
| float32 | 1024×32000 | 0.182304 | 0.187392 | 1.0279x | 1.0283x |
| float32 | 4096×32000 | 0.660480 | 0.664608 | 1.0062x | 1.0094x |

最差 1.0062x(S0 为 0.4148x);小 B 8 倍提升,大 B 不回归。

### 已知边界

- 同 S0:torch 平局语义设备相关;int32 索引 <2^31;V 极大(n_chunks>65536)
  的 pow2 归约块未设防,超出合理词表规模。

### 提交计划

- preflight tuple:season 2、race `782kzq4m`、account `15600308080`、
  team `SoulCoder`、batch 3、task 25、operator `draft_topk1`、
  stage `e1`、commit `b24781fdbfb8b938e7461fb3bf3046cd6a541e25`、ZIP
  `artifacts/competition/draft_topk1/e1-b24781f/draft_topk1.zip`、
  SHA-256 `db5f111ce4a50d28b6566ab803f8de380056a27ea6b176e7de62d0a076e7f9c8`、
  member `draft_topk1.py`。

## 平台提交记录

- 2026-08-28 00:07 CST 额度重置(30/30)后,按 29→30→25→28→27→26 顺序自动
  preflight + 一次性提交;全部 tuple 与账本一致后执行 confirm。
- 提交时间约 2026-08-28 00:13 CST;submission_id `5737`;ZIP SHA-256
  `db5f111ce4a50d28b6566ab803f8de380056a27ea6b176e7de62d0a076e7f9c8`;state `submitted`、validity `pending`、评测入队。
- 提交后团队当日额度剩余 24/30(6 投全记录)。


### 八芯结果(E1 首投,sub 5737,终态)

5/8,`invalid_correctness`,三芯独立失败:

| 芯片 | speedup | 结果 |
| --- | ---: | --- |
| tianshu | 3.987x | 通过 |
| muxi | 0.94025x | 通过 |
| haiguang | 0.75075x | 通过 |
| card_a | 1.383x | 通过 |
| card_b | 1.44075x | 通过 |
| enflame | - | 编译失败:`Pipeline run failed: PassManager`,定位在 `_draft_topk1_finalize_kernel`(动态标量 load + pow2 跨 chunk 向量归约),已知燧原指纹 |
| kunlunxin | - | `argmax index mismatch`:topk_index 精确失配;首索引假设与昆仑 torch.argmax 平局语义差异为一等假设 |
| huawei | - | `draft_tokens mismatch`:argmax 正确、draft 列写入失配;`tl.where` int64/int32 混合后隐式 cast 在昇腾疑似错误 lowering |

### E2 计划(单芯 vendor,预算剩 4 次)

- `_enflame`:finalize 换行内串行 chunk 合并(消除 pow2 块归约与动态标量 load);
- `_kunlunxin`:平局改取后索引(对齐昆仑 torch.argmax 行为)——先用最小
  平局回归在代理复现假设再定;
- `_huawei`:topk 载入后显式 cast 到 draft dtype 再 where,或拆成
  copy + 列散布两个无 where 的 kernel。

## E2 vendor 轮提交(sub 5769,2026-08-28 01:4x CST)

- 首轮逐芯失败指纹对应的 vendor 修复;vendor commit
  `0e3d58715ec0c5b1d3b841e2cf6b277b48fd8f9c`;ZIP SHA-256 `90f1001efe74c1dde11a0df22d8e4174eae902b94efe01509531bc91f5c6790e`。
- 成员:generic + `draft_topk1_ascend.py`(双 store 散布)+
  `draft_topk1_enflame.py`(finalize 串行合并)+
  `draft_topk1_kunlunxin.py`(平局取后索引假设)。
- 远端 NVIDIA 代理:router/lora/enflame/ascend vendor 数值全对;
  `_kunlunxin` last-index 在 NVIDIA 失配为设计内现象(NVIDIA torch 平局
  取首索引)。
- 提交后当日额度 20/30 剩余;评测中。

## E3 第二轮 vendor(sub 5804,2026-08-28 03:0x CST)

- E2/e2 结果回填:T28 天数+燧原已修(split-fp16 与 1D 无分支模板均有效);
  T27/T26 天数已修;华为失败根因确认为 **UB overflow(2.89M/1.57M bits)**,
  即 batch-2 已知昇腾 UB tile 上限,非 input_precision;T25 昆仑 actual 为
  未初始化垃圾(标量访存静默失效),燧原编译失败落在 finalize i64 标量 store。
- 本轮修复:router `_ascend` reduce BLOCK_R 32→8;draft `_ascend` 改纯拷贝
  + finalize 融合列写入;draft `_enflame`/`_kunlunxin` 改行内串行 argmax,
  lane 向量 store、无工作区、无标量访存(kunlunxin 回归首索引)。
- vendor commit `e62a27eb7f41819461fb981c44adf397f25b8729`;远端代理 9/9 绿。
- 评测中;额度 17/30。

