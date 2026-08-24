# Task 11 `chunk_local_cumsum_vector` 实验记录

## S0：generic baseline

状态：已被 E1 的 tiny-chunk warp 配置替代；历史 ZIP 保持不可变，未提交平台

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

## E1：`chunk_size <= 8` 使用 2 warps

状态：候选就绪，尚未提交平台

验证时间：2026-08-24 06:29–06:39 CST

### 单变量筛选

S0 对所有 tile 固定 4 warps；当 `chunk_size=5` 时，每个 program 只扫描
`BLOCK_F × BLOCK_SIZE = 8 × 8 = 64` 个元素，4 warps 的调度成本偏高。固定
SGLang/FlagGems 来源都把 2 warps 纳入合法 autotune 集，但没有提供跨芯固定阈值，
因此只做 NVIDIA 代理单变量筛选，kernel、grid、数学、mask 和内存地址均不变。

第一候选 `chunk_size <= 16` 使用 2 warps 被明确否决：affected 几何均值仅
`1.0329x`，最差 `0.9326x`；其中 `chunk=5` 提升约 `5.8–8.8%`，但
`chunk=16` 下降约 `4.9–6.7%`。对应日志为
`gpu:/tmp/flagos-chunk-local-cumsum-vector-e1.opvWoR/ab.log`，SHA-256
`1e23ab7cfda1e2eb34c5e07d4d2b412d90ef2ffa0344d7d2200dfd7332cb8c3a`。

E1 据此把唯一阈值收窄为 `chunk_size <= 8`。这是一行 host launch 配置；
`chunk=5` 走 2 warps，`chunk=16/17/64/128` 保持 S0 的 4 warps。

### Release 门禁

- fresh release 目录：
  `gpu:/tmp/flagos-chunk-local-cumsum-vector-release.4wzl84`，mode 0700；
  静态/单测 PID/PGID `82327`，A/B PID/PGID `82408`。
- Black 79、isort、flake8、py_compile 和现有 GPU unittest 2/2 通过。额外
  correctness 覆盖三 dtype、阈值边界 `1/7/8/9`、非连续输入，以及 9 组
  affected/control 规格；base/candidate 共 78/78 次 module-case 检查通过。
- 五轮交替 AB/BA，`warmup=25, rep=100`，每次 wrapper 批量 20 次。12 个
  `chunk=5` affected 点几何均值 `1.0811x`；FP16/BF16/FP32 分别为
  `1.0864x`、`1.0862x`、`1.0708x`，最差 `1.0578x`。
- 15 个 `chunk=16/17/64/128` controls 几何均值 `0.9996x`，范围
  `0.9968–1.0015x`。affected 总体 `>=1.05x`、逐 dtype `>=1.02x`、任一点
  `>=0.98x`、controls `0.98–1.02x` 的预设门禁全部通过。
- 独立审计指出首轮性能只覆盖了阈值内的 `chunk=5`，随后用同一 commit 补跑
  `chunk=1/7/8` 的正向与 reverse+scale 共 18 个 affected 点：几何均值
  `1.1081x`，FP16/BF16/FP32 分别为 `1.1139x`、`1.1129x`、`1.0975x`，
  最差 `1.0318x`；`chunk=9/16` controls 为 `0.9983–1.0000x`。
  补充任务 PID/PGID `82560`，72/72 次 module-case 检查通过；日志
  `release-threshold.log` SHA-256 为
  `e9c40fabe270e9a12a54a74841ea9a4dbbebc581685701e69ddb457b091b14a6`。
- 为覆盖阈值内每个整数，第二个补充探针再测 `chunk=2/3/4/6` 的正向与
  reverse+scale 共 24 个 affected 点：几何均值 `1.0936x`，三 dtype 分别为
  `1.0938x`、`1.0933x`、`1.0936x`，最差 `1.0428x`；`chunk=9/16`
  controls 为 `0.9989–1.0000x`。任务 PID/PGID `82700`，84/84 次
  module-case 检查通过；`release-threshold2.log` SHA-256 为
  `fe957b00e989c2461d977ffbd0f57f9bc1c620d295ccf80f6495268fdc639b42`。
  至此 E1 实际影响的每个 `chunk_size=1..8` 都有 correctness 和性能证据。
- base/candidate 各编译 39 个变体，最大均为 39 registers/thread、2,048 bytes
  shared、1 stage；candidate 仅 affected 变体从 4 改为 2 warps。spill、global
  scratch、local load/store 全为 0。
- `release.log` 与 `release-ab.log` SHA-256 分别为
  `22ced87c5eff51a125b84e964911b3b1d2956db123dbb7d42efae1f0802416b5`、
  `4a450df0b6611579e7b09f0109f326aa7878ffaa11ce2204db95894607a9e0d0`。

验证环境：NVIDIA GeForce RTX 5070 Ti 16 GB，driver 610.57.04，Python 3.12.13，
PyTorch 2.13.0+cu130，Triton 3.7.1，CUDA 13.0。该结果仅是 NVIDIA 代理证据。

### E1 构建身份

| 项目 | 值 |
| --- | --- |
| source commit | `528a2bbc31d4f298966b404c376b3421efc3fe84` |
| verification commit | `528a2bbc31d4f298966b404c376b3421efc3fe84` |
| ledger commit | 本节所在 commit |
| 源文件 SHA-256 | `3b151d953ac283e436600ee7aa5e9db6e2e6718c135d1916f138cee3d3f14e88` |
| 测试 SHA-256 | `f050289a0abfffea54fdabb99aee659463dc40fa8284910142e6e58a5e983d3e` |
| ZIP | `artifacts/competition/chunk_local_cumsum_vector/e1-528a2bb/chunk_local_cumsum_vector.zip` |
| ZIP SHA-256 | `7f0484b9b2ae078bf284e4fda1c5a1a0ffb0c8545b907e801d9fa21200fde7d8` |
| ZIP manifest | 顶层 `chunk_local_cumsum_vector.py`，3706 bytes；ZIP 3860 bytes |

`unzip -t`、UTF-8/语法、唯一普通 `.py`、basename、大小和成员逐字节哈希均已
复验；打包器第二次运行状态为 `verified-existing`。

### 剩余风险与停止点

- 2/4 warps 在不同后端可能映射为不同 wave/线程组织；RTX 的阈值性能不能外推
  其余七芯。若平台只在某芯回退，保留 generic E1，仅为该芯回到 4 warps。
- `tl.cumsum`、非 2 次幂 mask 和 reverse 索引仍未被八芯全部实测；E1 没有改变
  这些路径。
- E1 为“候选就绪、未提交”。未打开浏览器、未读取或消耗平台额度；上传前必须
  重新验签 ZIP、读取平台实时 tuple，并取得用户针对该精确产物的一次性确认。

## E1a：折叠 grid Ascend/Kunlun vendor（首投候选，≤2 次预算）

状态：release 门禁通过，候选就绪

grid 审计：generic 3D `(feature_blocks, nchunks, batch)` 展平总数在大 shape
可超 65535（如 features 4096/block_f 8=512 块 × 128 chunk × 2 batch =
131072），触发华为 launch 越界与昆仑编译失败。vendor（ascend/kunlunxin 同
字节）：1D `min(total, 4096)` + 逻辑 id 按 batch→chunk→feature 分解（div/mod
形式，Task 08 generic 在两芯平台通过的同款结构）；kernel 数学、BLOCK、
warps、stages=1 与 E1 generic 逐行一致。天数无 dot、燧原无分支均不需要
vendor。新增回归 `test_vendors_cover_folded_grid`（2×8192×32×16、chunk 64，
总 program >4096，三 dtype × 正反向 × 三模块对 reference）。screening
`gpu:/tmp/flagos-clcv-vend.URFrLw`（首跑 Black 折行经远端格式化回拷修正），
最终 PID/PGID `112894`，3/3 unittest（1.063s），`screening.log` SHA-256
`48fc39bbf4365cdb14897f72e995d67d83c10afb387ad5bce9573d477ac01034`。
release `gpu:/tmp/flagos-clcv-release.*`，source/verification commit
`6b4794050ef6c3ebd3e247c2c6540ee61390ab8f`，`RELEASE_OK`，`release.log`
SHA-256
`a38432f3401fe37fd5868fe6e495ec43130b84e5a37ea2531834821527a61c39`。
vendor blob
`62fbabc03c2ef1521a2f75243ef61135b42c5b21ba3865c887dfe5ab9a3d508e`，测试
`331fdb9834c7794dfd476e8496a5834494b25da2240cd37e2ce817211bfe25df`。
canonical ZIP
`artifacts/competition/chunk_local_cumsum_vector/e1a-6b47940/chunk_local_cumsum_vector.zip`，
SHA-256
`3ef95e909fa8d4018ccb7ea0569776e24bf61cfa15fcd4523d5c390bd589bf4c`，成员
generic + ascend/kunlunxin，`unzip -t` 通过。
