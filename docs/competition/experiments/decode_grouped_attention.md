# Task 16 `decode_grouped_attention` 实验记录

## S0：generic GQA baseline

状态：S0 已打包并通过本地门禁；等待当次上传确认

验证时间：2026-08-24 01:28–01:37 CST

源码 commit：`f431ba4`

### 契约

| 项目 | 值 |
| --- | --- |
| 公开接口 | `decode_grouped_attention(q, k_buffer, v_buffer, kv_indptr, kv_indices, sm_scale)` |
| GQA 约束 | `H_KV < H_Q` 且 `H_Q % H_KV == 0` |
| 头映射 | `kv_head = query_head // (H_Q // H_KV)` |
| 输入 | q `[B,H_Q,D]`；K `[P,H_KV,D]`；V `[P,H_KV,D_v]`；CSR page 索引 |
| 计算 | `softmax(q @ K.T * sm_scale) @ V`，logits/softmax/累加均 FP32 |
| 输出 | `[B,H_Q,D_v]`，FP32，out-of-place；输入不变 |
| 支持芯片 | 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用 A/B，共 8 款 |
| 截止 / 门槛 | 2026-08-27 19:59:59；`speedup_threshold=0.1` |

固定来源与 Task 15 相同：题面、SGLang `8014d9d` production 实现和
community `0e8023d:whd3/decode_attention.py`。S0 仅保留在线 softmax 与正确
GQA 头映射；无 reference/demo、Torch 计算、连续化、host/设备识别、fallback、
split 临时张量或私有 API。

2026-08-24 01:31 CST 公开 API 状态：41 次提交、9 支队伍、1 支达到门槛；
榜首 c2flow 为 8/8、76.45365x。动态值仅用于当时决策。

### 唯一候选

- 与 Task 15 相同的自包含单 kernel 保守策略；每个 Task ZIP 不依赖另一文件。
- 每个 `(batch, query_head)` 一个 program；通过整数除法映射到 KV head。
- 所有输入按真实 stride 读取；int32/int64 CSR 均可；输出 FP32。
- `BLOCK_D/BLOCK_DV` 下一 2 次幂，sequence tile 受 8192-element tile 预算
  限制；4 warps、1 stage、FP32 online softmax、完整变长 mask。

### 验证

| 项目 | 值 |
| --- | --- |
| 源文件 SHA-256 | `d14bde1140cec961690d940da8b3d8e8c89a45e78a466bdeb88e6eb5be0cd7c9` |
| 测试 SHA-256 | `8e7a714a79a3487d4c4ebb4fc4abbebad92fc12f1afbdd5beda750344019f43b` |
| ZIP | `artifacts/competition/decode_grouped_attention/s0-f431ba4/decode_grouped_attention.zip` |
| ZIP SHA-256 | `4ed5e04d8453e100a38feff3d8986801fab9a13c4d77481e070a3260855136ef` |
| ZIP manifest | 顶层 `decode_grouped_attention.py`，5054 bytes |
| 远端证据目录 | `gpu:/tmp/flagos-batch2.SQaIX2` |
| 远端环境 | RTX 5070 Ti 16 GB；PyTorch 2.13.0+cu130；Triton 3.7.1；CUDA 13.0 |

- 共享 unittest 5/5 通过；Task 16 覆盖 BF16/FP32/FP16，`H_Q/H_KV` 为
  `8/2`、`4/1`，真实非连续 Q/K/V，int32 CSR，变长 `3/70`，
  `D=40,D_v=24`，以及 `D=64,D_v=257` 和空 batch。
- `py_compile`、`git diff --check`、Black 79、isort、flake8 均通过；本地与
  远端源码/测试哈希一致。
- wrapper-inclusive FP16 benchmark：`B=8,H_Q=32,H_KV=8,D=128,D_v=64,
  L=256`，S0 `0.137529 ms`，题面 reference `0.726847 ms`，代理 speedup
  `5.285x`。

### 风险与下一步

- NVIDIA 代理不能证明其余七款芯片；平台前不增加 vendor 文件。
- 公开 reference 对单个 `L=0` 序列未定义；只承诺空 batch。CSR tensor 必须
  与 Q/K/V 位于同一设备。
- GQA 当前每个 Q head 重复读取共享 KV head。平台正确后，第一性能候选是单个
  program 合并同组 Q heads；长序列再评估 split-KV，不同时改变两项。
- ZIP 由 commit `f431ba4` 直接生成；`unzip -t`、UTF-8、单一 `.py`、10 MB、
  basename 和 ZIP 内源码哈希门禁均通过。尚未上传或消耗额度；上述 ZIP 需要
  用户当次确认。
