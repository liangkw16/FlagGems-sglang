# Task 11 `chunk_local_cumsum_vector` 实验记录

## S0：generic baseline

状态：S0 已打包并通过本地门禁；等待当次上传确认

验证时间：2026-08-24 01:23–01:28 CST

源码 commit：`3fac516`

### 契约与固定来源

- 接口：`chunk_local_cumsum_vector(g, chunk_size, reverse=False,
  scale=None)`。
- `g=[B,T,H,S]` 且 `T` 可被 chunk_size 整除；按每个 chunk 的时间维做
  FP32 cumsum，支持 reverse 和可选 scalar scale，输出 FP32、shape 不变。
- 固定来源：SGLang `8014d9d` 的 FLA cumsum 与 FlagGems `ed2508b` 的
  vector cumsum；S0 采用更短的标准 Triton generic，不复制 dot/设备策略。

### 构建身份

| 项目 | 值 |
| --- | --- |
| 源文件 SHA-256 | `7e3aefc98ccd0cfcadcd4a4120b47b84e748a5d460754aa47f71777a9fe70292` |
| 测试 SHA-256 | `f050289a0abfffea54fdabb99aee659463dc40fa8284910142e6e58a5e983d3e` |
| ZIP | `artifacts/competition/chunk_local_cumsum_vector/s0-3fac516/chunk_local_cumsum_vector.zip` |
| ZIP SHA-256 | `b4ab7b21ecd5a4f23b0d53aab00e8ef504c2e2f329c27b1bbf77306db5daab3a` |
| ZIP manifest | 顶层 `chunk_local_cumsum_vector.py`，3680 bytes |
| 远端证据目录 | `gpu:/tmp/flagos-batch2.SQaIX2` |

### 候选与代理验证

- 把 `H*S` flatten 成 feature 维；3D grid `[feature tile, chunk, batch]`，
  按真实四维 stride 读取和写回，reverse 只改变时间索引。
- `BLOCK_SIZE=next_power_of_2(chunk_size)`，每 program 最多 8 个 feature，
  4 warps、1 stage；无 autotune、设备判断、fallback 或 vendor 文件。
- RTX 5070 Ti 上 unittest 2/2 通过；覆盖三 dtype、chunk `5/64`、非 2 次幂、
  reverse、scale `None/-0.25/2.0`、非连续输入、空输入和输入不变性。
- wrapper-inclusive 代表 shape 的代理加速比为 `1.605x–2.919x`。
- ZIP 由 commit `3fac516` 直接生成，`unzip -t`、UTF-8、单一 `.py`、
  10 MB、basename 和 ZIP 内源码哈希门禁均通过。

### 已知风险

- `tl.cumsum`、非 2 次幂 mask 和 reverse 索引尚未被其余七类编译器验证。
- 若单芯 grid/tile 编译失败，只为该芯添加固定来源的 grid-stride 或 scalar
  vendor；不预制八份实现。
- 尚未上传或消耗额度；上述 ZIP 需要用户当次确认。
