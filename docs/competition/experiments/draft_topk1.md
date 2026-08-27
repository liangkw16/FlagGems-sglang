# Task 25 `draft_topk1` 实验记录

状态:S0 候选就绪,待额度重置后提交(顺序 29 → 30 → 25;2026-08-27 团队
当日 30/30 额度已耗尽,2026-08-28 00:00 重置)

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
