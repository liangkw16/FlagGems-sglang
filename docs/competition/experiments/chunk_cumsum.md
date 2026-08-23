# Task 10 `chunk_cumsum` 实验记录

## S0：generic baseline

状态：S0 已打包并通过本地门禁；等待当次上传确认

验证时间：2026-08-24 01:20–01:28 CST

源码 commit：`3fac516`

### 契约与固定来源

- 接口：`chunk_cumsum(dt, A, chunk_size, dt_bias=None,
  dt_softplus=False)`。
- 输入 `dt=[B,T,H]`；可选 FP32 bias/softplus 后 clamp 到非负，生成
  `[B,H,nchunks,chunk_size]` 的 FP32 `dt_out`，再按 head 乘 `A` 并沿
  chunk 维做 FP32 cumsum；返回顺序必须是 `(dt_out, dA_cumsum)`。
- 题面支持 FP16/BF16/FP32 输入和八款芯片；核心计算必须走 Triton。
- 固定来源：Mamba v2.2.4 `ssd_chunk_state.py`，以及公开本地引用
  `community/master` 的多芯实验。S0 没有复制后者的设备识别、cache hint、
  异常重试或 vendor 分支。

### 构建身份

| 项目 | 值 |
| --- | --- |
| 源文件 SHA-256 | `ce9ac83b61fe67c684060d7aaa1aac9238995b21179ddf80f48f019944c55a8d` |
| 测试 SHA-256 | `379396fb34b4eb27f3bed7196b5b7c508da1c61d5b37df0f0b42fed4d0738a1b` |
| ZIP | `artifacts/competition/chunk_cumsum/s0-3fac516/chunk_cumsum.zip` |
| ZIP SHA-256 | `81a1cff508d5ca8a7eb921d8644e4061b40382ea2ab9e4ce12a231118e48c607` |
| ZIP manifest | 顶层 `chunk_cumsum.py`，3974 bytes |
| 远端证据目录 | `gpu:/tmp/flagos-batch2.SQaIX2` |

### 候选与代理验证

- 单个标准 Triton kernel，3D grid `[head tile, chunk, batch]`；每 program
  处理最多 8 个 head 和一个 chunk，按真实 input stride 读取。
- `BLOCK_SIZE=next_power_of_2(chunk_size)`，tile 上限启发式 4096 元素，
  4 warps、1 stage；无 autotune、设备判断、fallback 或 vendor 文件。
- RTX 5070 Ti 16 GB 上 unittest 1/1 通过；循环覆盖三 dtype、chunk
  `5/16/64`、bias/softplus 开关、非连续输入、tail 和输入不变性。
- wrapper-inclusive 代表 shape 的代理加速比为 `3.184x–29.044x`；这只用于
  筛除明显慢候选，不能替代平台逐芯成绩。
- ZIP 由 commit `3fac516` 直接生成，`unzip -t`、UTF-8、单一 `.py`、
  10 MB、basename 和 ZIP 内源码哈希门禁均通过。

### 已知风险

- 公开 reference 的 `ceil(T/chunk_size)` 与后续 reshape 暗含 `T` 可整除；
  S0 对尾 chunk 做安全掩码和零初始化，但平台语义仍以真实 harness 为准。
- `tl.cumsum` 和二维 tile 尚未由其余七类编译器验证；单芯失败后才增加对应
  vendor，不预先推断 A/B 映射。
- 尚未上传或消耗额度；上述 ZIP 需要用户当次确认。
