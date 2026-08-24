# Task 19 `fused_rmsnorm` 实验记录

## 契约

- 接口：`fused_rmsnorm(x, weight, eps)`。
- 公式：`x * rsqrt(mean(x * x, dim=-1) + eps) * weight`。
- `x`、平方和、均值、`rsqrt` 和权重乘法均按 FP32 计算，输出转换回
  `x.dtype`；输入保持不变。
- 题面支持 FP16、BF16、FP32，容差分别为 `1e-2`、`1.5e-2`、`1e-4`。
- 支持八类芯片；最低加速比为 0.1x。核心路径必须使用 Triton/TLE，禁止
  设备判断、异常 fallback 和纯 PyTorch 实现。

固定参考：SGLang `8014d9d` 的 `elementwise.py` 第 139–188 行，以及
FlagGems `ed2508b` 的 `_fused_rms_norm.py` 第 31–67 行。SGLang 的设备分支、
autotune 和最高 32 warps 未进入 generic 首版。

## S0：generic single-row baseline

状态：canonical S0c 平台首投完成，8/8 通过；平均加速比 4.53x，当前单题第 6 名

验证时间：2026-08-24 01:23–01:28 CST

源码 commit：`3fac516`

### 构建身份

| 项目 | 值 |
| --- | --- |
| 源文件 | `src/flaggems_sglang/ops/fused_rmsnorm.py` |
| 源文件 SHA-256 | `02bed1a5cb28b583c343892569d9e25d1ef3d888e124fdd066d1155a0b964997` |
| 测试文件 | `tests/test_fused_rmsnorm.py` |
| 测试 SHA-256 | `e77dd1a17f6df51b310eb145fec621a49af5e8dcf7f0dbca97f2a8032b608f91` |
| ZIP | `artifacts/competition/fused_rmsnorm/s0-3fac516/fused_rmsnorm.zip` |
| ZIP SHA-256 | `93780caf704341737ddfe5925cfacdcd7115ccefc2f38edf3c7ff006716d1820` |
| ZIP manifest | 顶层 `fused_rmsnorm.py`，2337 bytes |

### 唯一候选配置

- 一行一个 Triton program，`BLOCK_SIZE=next_power_of_2(hidden_size)`，
  8 warps、1 stage。
- 显式使用输入行/列 stride 和权重 stride；输出为连续、同 shape、同 dtype
  tensor。高维非连续输入在 wrapper 中 reshape，必要时由 PyTorch 生成布局副本，
  核心归一化仍只由 Triton kernel 完成。
- 尾部完整 mask；空输入直接返回空输出。
- 无 autotune、vendor 文件、设备判断、异常捕获或 PyTorch 计算 fallback。

### 验证

- `py_compile`、AST 解析、Black 79、isort、flake8 和 `git diff --check`
  已通过。
- 公开接口测试先于实现落盘；覆盖 FP16/BF16/FP32、hidden 513/8193、
  非连续 `x/weight`、输入不变性、输出 shape/dtype 和空输入。
- 本机没有 PyTorch/Triton/GPU，不能执行数值测试。远端 `gpu` 使用 RTX 5070 Ti
  16 GB、driver 610.57.04、Python 3.12.13、PyTorch 2.13.0+cu130、
  Triton 3.7.1、CUDA 13.0、compute capability 12.0。
- 远端源码与测试 SHA-256 和本地一致。执行 `tests/test_fused_rmsnorm.py -v`
  为 2/2 unittest 方法通过；循环内覆盖 6 个 dtype/hidden 组合，运行 1.291 秒。
- wrapper-inclusive 代理 benchmark 覆盖 `(rows, hidden)=(128,4096)、
  (32,8193)、(512,1024)` 的三 dtype；相对 Torch reference 为
  `1.830x–4.720x`。

### 已知风险

- NVIDIA 代理只验证语法、数值与候选性能，不能证明其余七类后端正确或达标。
- 大 hidden 会把整行放入一个 power-of-two block；8193 已纳入回归，但更大隐藏维
  可能出现寄存器/本地内存压力，应先看隐藏 harness 或平台结果再决定分块两阶段方案。
- 当前只验证题面常见的一维 weight（元素数等于 hidden）；没有为未公开的广播形状
  增加推测性分支。
- ZIP 由 commit `3fac516` 直接生成；`unzip -t`、UTF-8、单一 `.py`、10 MB、
  basename 和 ZIP 内源码哈希门禁均通过。没有平台提交授权，也未消耗额度。

## S0 发布复核：保留单文件候选

状态：平台首投完成，8/8 通过；平均加速比 4.53x，当前单题第 6 名

复核时间：2026-08-24 02:42–03:03 CST

| 项目 | 值 |
| --- | --- |
| source commit | `3fac516a8d64c88b183801668a7857d969a05e37` |
| verification commit | `2bdd8efba29fcdf36c44831f8e26585598cc7ee6` |
| 当前测试 SHA-256 | `e8f275f23912cafcf258df664499d34c70a05fb698d124bdbb073bd8daa59f59` |
| 源文件 SHA-256 | `02bed1a5cb28b583c343892569d9e25d1ef3d888e124fdd066d1155a0b964997`，与 S0 不变 |
| ZIP SHA-256 | `93780caf704341737ddfe5925cfacdcd7115ccefc2f38edf3c7ff006716d1820`，`verified-existing-legacy` |
| canonical ZIP | `artifacts/competition/fused_rmsnorm/s0c-3fac516/fused_rmsnorm.zip`；2026-08-24 11:59 CST 生成并以 `verified-existing` 复核，未覆盖旧产物 |
| canonical ZIP SHA-256 | `442f480b91396829f3ea81501a01e5134aa42acf2fb0b5636434247341a3ce4b` |
| 远端证据目录 | `gpu:/tmp/flagos-rmsnorm-s1.giHJqZ`，mode 0700 |
| 发布门禁任务 | PID `71212`；`release-gates.log`；2026-08-24 02:47:26 CST |
| 高行数探针 | PID `71556`；`high-m-probe.log`；2026-08-24 03:03:25 CST |
| 平台结果 | 2026-08-24 14:08:35 CST 完成；8/8，平均 4.53x，第 6 名；提交后剩余 12/15 |

- 远端 py_compile、Black 79、isort、flake8 全部通过；unittest 3/3 方法通过。
  除原有三 dtype × `H=513/8193` 的非连续回归外，新增三 dtype ×
  `(rows,H)=(1,512)/(4,4096)/(4,5120)/(4,8192)`，覆盖单行 decode、常见
  power-of-two hidden、确定性零输入、`eps=1e-5` 和 BLOCK 512–8192。
- 远端源码、测试 SHA-256 分别为 `02bed1a5...b964997`、
  `e8f275f2...a59f59`，与 source/verification commit 完全一致。高行数探针脚本
  SHA-256 为 `67de2055...5752`，日志 SHA-256 为 `9f070c48...9c40`；完整文件均
  保留在上述 mode 0700 证据目录。
- 固定 FlagGems `ed2508b` 的同类 kernel 使用默认 4 warps；当前 generic 固定
  8 warps。固定 FlagTree `c1ea8285` 的 Enflame 编译器对 GCU400/410 会在
  options 初始化阶段把 `num_warps > 4` 自动降为 4，GCU300/500 则允许 8，
  所以没有证据证明当前 S0 必须增加 vendor 文件。
- NVIDIA 单变量探针显示全局改成 4 warps 会让 FP16/BF16 `H=8193` 慢约 16%，
  并出现 12 个 spills；因此不为未知平台版本牺牲其余芯片，保留 generic 8
  warps。若平台 Enflame 明确报 warp 编译错误，再只增加 self-contained
  `_enflame.py` 4-warps override。
- 固定 FlagGems 的 GCU400 RMSNorm 在 `rows > 65535` 时压缩 launch grid，并让
  每个 program 处理多行；当前 generic 仍以 `grid=(rows,)` 启动。题面没有公开
  rows 范围，而且 GCU300 实现仍使用 `grid=rows`，所以这只是条件性平台风险；若
  燧原实际报 grid 超限，再单独加入 grid-stride vendor。
- 固定 Kunlun RMSNorm 明确记录单行方案在 `[10000,256]` 只有 `0.006x`，并在
  `hidden <= 256, rows >= 4096` 时切换 multi-row kernel。题面没有公开该 shape，
  NVIDIA 代理也不能证明昆仑芯表现，因此当前不预加 vendor；若平台出现对应低速，
  再以该固定实现做单变量恢复。

wrapper-inclusive NVIDIA decode 代理；五组交替顺序，组内
`warmup=25, rep=100`，表中为五组 p50 的中位数：

| dtype | `(rows,H)` | S0 ms | reference ms | speedup |
| --- | --- | ---: | ---: | ---: |
| FP16 | `(1,512)` | 0.004212 | 0.015835 | 3.7595x |
| FP16 | `(4,4096)` | 0.004651 | 0.018574 | 3.9938x |
| FP16 | `(4,5120)` | 0.004911 | 0.018931 | 3.8548x |
| FP16 | `(4,8192)` | 0.005020 | 0.018454 | 3.6759x |
| BF16 | `(1,512)` | 0.004299 | 0.015924 | 3.7045x |
| BF16 | `(4,4096)` | 0.004750 | 0.018569 | 3.9092x |
| BF16 | `(4,5120)` | 0.004840 | 0.018928 | 3.9106x |
| BF16 | `(4,8192)` | 0.005020 | 0.018436 | 3.6725x |
| FP32 | `(1,512)` | 0.004237 | 0.010682 | 2.5209x |
| FP32 | `(4,4096)` | 0.004835 | 0.012490 | 2.5831x |
| FP32 | `(4,5120)` | 0.005094 | 0.013198 | 2.5906x |
| FP32 | `(4,8192)` | 0.005894 | 0.011966 | 2.0301x |

额外 high-row 代理使用 `hidden=256`，同样五组交替、`warmup=25, rep=100`；
三 dtype 的 12 个组合全部通过 reference：

| dtype | rows | S0 ms | reference ms | speedup |
| --- | ---: | ---: | ---: | ---: |
| FP16 | 4096 | 0.014469 | 0.038324 | 2.6487x |
| FP16 | 10000 | 0.028419 | 0.071434 | 2.5136x |
| FP16 | 65535 | 0.142190 | 0.829491 | 5.8337x |
| FP16 | 65536 | 0.142178 | 0.829646 | 5.8353x |
| BF16 | 4096 | 0.014403 | 0.038885 | 2.6999x |
| BF16 | 10000 | 0.028423 | 0.071270 | 2.5075x |
| BF16 | 65535 | 0.142197 | 0.829429 | 5.8330x |
| BF16 | 65536 | 0.142132 | 0.829581 | 5.8367x |
| FP32 | 4096 | 0.014821 | 0.028019 | 1.8906x |
| FP32 | 10000 | 0.031104 | 0.057709 | 1.8553x |
| FP32 | 65535 | 0.186803 | 0.573734 | 3.0713x |
| FP32 | 65536 | 0.186873 | 0.573941 | 3.0713x |

结论：24 个新增代理点全部远高于 0.1x 门槛，未发现值得替换源码的本地证据。
`rows=65536` 在 NVIDIA 正确运行不能消除 Enflame 的物理 grid 风险。
打包器确认旧 ZIP 的唯一成员与 source commit 逐字节一致；旧容器字节保持不可变，
不以规范 ZIP 覆盖。S0 单文件 canonical ZIP 已用于 Task 19 首投并由平台验证为
8/8；NVIDIA 代理结论与平台结果仍分别保留。

### 平台首投

- 提交时间：2026-08-24 14:08:35 CST
- 登录账号 / 团队：`8080_apiqhow` / `SoulCoder`
- 提交文件：`fused_rmsnorm.zip`；ZIP SHA-256：
  `442f480b91396829f3ea81501a01e5134aa42acf2fb0b5636434247341a3ce4b`；
  2467 bytes
- 平台远端产物验签：文件大小和 SHA-256 均与本地 canonical ZIP 完全一致。
  上传页显示的 `18.4 KB` 是当前前端 bundle 的固定展示字符串，不是文件实测大小。
- 当前状态：已完成，8/8 通过；平均加速比 4.53x，当前单题第 6 名
- 本次提交后今日剩余额度：12/15
- 页面未展示独立提交 ID，以 Task19、文件名和提交时间联合定位

| 芯片 | 正确性 | speedup |
| --- | --- | ---: |
| 天数智芯 | 通过 | 7.73x |
| 沐曦 | 通过 | 5.39x |
| 燧原 | 通过 | 1.52x |
| 海光 | 通过 | 7.51x |
| 昆仑芯 | 通过 | 0.94x |
| 华为 | 通过 | 1.66x |
| 国际通用 A | 通过 | 5.85x |
| 国际通用 B | 通过 | 5.67x |

## E1：小 hidden multi-row（否决）

状态：未晋升、未提交平台；源码与测试均已恢复，S0 ZIP 不变

验证时间：2026-08-24 06:50–06:52 CST

### 假设与实现

E1 复用仓库 `gemma_rms_norm.py` 的 multi-row 模式，保持 8 warps 和原数值路径：
`BLOCK_SIZE<=512/1024/2048` 时分别让一个 program 处理 `16/8/4` 行，并把
weight load 移到静态行循环外；行数不足对应分组时仍走单行 S0。grid 改为
`ceil(rows/ROWS_PER_PROGRAM)`，尾 program 用 `row < rows` 保护。没有引入 vendor、
autotune、fallback 或额外张量。

永久回归候选曾覆盖 RPP16 exact/tail、RPP8、RPP4；独立 harness 还覆盖三 dtype、
hidden 边界 `255/256/257`、`511/512/513`、`1023/1024/1025`、
`2047/2048/2049`，以及非连续 x/weight。候选 3/3 unittest 和 138/138 次
base/candidate module-case correctness 均通过。

### A/B 与停止结论

- 证据目录 `gpu:/tmp/flagos-fused-rmsnorm-multirow.RX6luz`，mode 0700；
  静态/单测 PID/PGID `83413`，A/B PID/PGID `83521`。
- 五轮交替 AB/BA，`warmup=25, rep=100`，每次 wrapper 批量 20 次；每点取
  5 个配对 speedup 的中位数。15 个 affected 点几何均值仅 `0.5715x`，
  FP16/BF16/FP32 分别为 `0.5570x`、`0.5659x`、`0.5922x`，范围
  `0.1951–0.9809x`。
- RPP1 controls 也因统一 kernel 多出的 static loop/row guard 和 weight-load 顺序
  变化而落在 `0.9268–1.0769x`，超出 `0.98–1.02x` 门禁。
- base/candidate 最大分别为 186/194 registers/thread，shared 最大从 32 bytes
  增至 8,192 bytes；两者均无 spill、global scratch 或 local load/store。候选
  同时未通过预设寄存器/资源门禁。
- 候选源码、测试、harness SHA-256 分别为
  `281ca1c76fd8e93b378ad9093b683e769ed7beddd7ffb829a03446e1bfeef409`、
  `4159b1f41b0eea69085a5861181f5a1d8bc092b1160420ee892f9db4a2e24eef`、
  `0660b8393278b15d6191f0e75425f63506111507df9b2c80826396c63f902661`；
  `screening.log` 与 `ab.log` SHA-256 分别为
  `c8783713b302def2d0663e900e3d82f2509f3164ba5a96b71b321d6d5494bd4f`、
  `5305fa1c2e03be6377436b7dc3b5f1f2aa24c790703c69ed437ed2f9f99e5958`。

验证环境为 NVIDIA GeForce RTX 5070 Ti 16 GB、driver 610.57.04、Python
3.12.13、PyTorch 2.13.0+cu130、Triton 3.7.1、CUDA 13.0。E1 正确但性能和
资源均明确失败，已按停止门禁删除候选 diff，不再为 NVIDIA 细分 RPP；当前源码
SHA-256 恢复为 `02bed1a5...b964997`，不可变首投仍为 S0 `3fac516`。

未打开浏览器、未读取或消耗平台额度；旧确认不授权新产物。
