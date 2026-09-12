# Task 67 `fill_padded_rows` 实验记录

```current
task: 67
operator: fill_padded_rows
batch: 5
validity: valid
platform: completed(13386,e3,8/8,4.23825x;三宽芯门全负,轴关闭;TB e2 4.2969x)
candidate_stage: e3
team_best_stage: e2
team_best_speedup: 4.2969
sealed: no
next: e3 列分块轴关闭(唯沐曦+24%);TB e2 4.2969 守榜(#8,榜首 8.32);宽 shape 结构待新证据
updated: 2026-09-12
```

## 契约与范围

- 完整题面：[Task 67](../tasks/batch-5/67-fill_padded_rows.md)（2026-09-11 平台新增六题之一）。
- 接口 `fill_padded_rows(x, num_token_non_padded, fill_value)`；exact；
  返回**新张量**（reference 为 `x.clone()` 后填充），`num_token_non_padded`
  为 device 单元素整数张量，须在 kernel 内读取、静态 grid（每行一个
  program），保持 CUDA-graph 可捕获语义。
- 核心计算 Triton；无 fallback。八芯每芯 0.1x；截止 2026-09-17 19:59:59。

## 实现（S0）

- 上游：SGLang 8014d9d `kernels/ops/moe/fill_padded_rows.py`（上游为原地
  版本）；本仓改为 out-of-place：wrapper `out = x.clone()` 后单 kernel
  填充 `row >= n` 的行。`n` 由 `tl.load` 在 kernel 内读取，无 `.item()`
  同步，grid `(rows,)` 静态。
- `BLOCK_COLS = next_power_of_2(n_cols)` 单块覆盖整行；比较与寻址走 int64。

## 不可变身份

- source / verification commit：`b4727f1`（2026-09-11）。
- source SHA-256：`92caf5e002dd60f5c4ce80c523159fd150a79ea87a44e15bfdc46312f63196df`。
- test SHA-256：`07929937c007383c78d597842ed66df875eedc452e1c1d64431d6e465288fe6e`。
- ZIP：`artifacts/competition/fill_padded_rows/s0-b4727f1/fill_padded_rows.zip`。
- ZIP SHA-256：`9319d9f06783f314706b635067bf456521a22007858f73a23658bba869c78a27`。

## 验证状态

- py_compile、black/isort/flake8 全部通过（本地）。
- 测试：4 方法 / 覆盖 dtype×fill 值、n 边界（0/部分/rows/越界）、列宽
  1~7168、行距 stride、int32/int64 计数、空输入与 0 列；断言输入不变、
  返回新张量。
- **远端 GPU（192.168.5.204 / gpu-et）2026-09-11 晚全程不可达**（物理口与
  EasyTier 均超时，ARP 无表项）；release 回执待补，目标芯与代理执行均为
  `target-runtime-unverified`。KernelGen MCP 本会话不可用。

## 风险

- 结构极简（clone + 条件填充），跨芯风险最低；唯一注意点是
  `next_power_of_2(n_cols)` 在超宽行时的寄存器压力（生产场景 n_cols 为
  hidden 量级，可控）。

## 优化方向（按把握）

1. S0 直投（预期显著快于 reference 的 clone+index_put 链）。
2. E1 候选（未开发）：若平台形状列数巨大，改 grid-stride 列循环降低
   BLOCK_COLS；先取平台逐芯数据再决定。

## 2026-09-12 平台结果（submission 13300，daily_seq 5）

- **8/8 valid，均值 3.43285x**（首发即过）。逐芯：天数 9.7498 / 沐曦
  2.2450 / 燧原 1.3660 / 海光 4.8460 / 昆仑 0.5952 / 华为 1.3910 /
  A 3.9498 / B 3.3200。
- 回执（b4727f1，RTX 5070 Ti 代理）：4 方法 0 失败、25 次真实 launch、
  14 组非空 shape；`artifacts/competition/batch5-ext6-validate-20260912/fill_padded_rows/`
  （verification.json SHA-256 `ee29bb4668e9d52fcebf8587d1651242d1da0dfc77f76b21b562e005e0e8987c`）。

## 2026-09-12 E1：单写融合（候选就绪后提交）

- 结构改写（单变量）：去 wrapper `x.clone()`（读 N + 写 N，pad 行随后被
  第二次覆写），改 `torch.empty` 出参 + 单 kernel **每元素只写一次**
  （`row < n` 拷贝、否则填充，运行时标量分支 = T64 已证形态）。省
  pad 行读 + pad 行双写 + 一次 launch。
- 跨芯纪律：掩码地址用算术钳位 `cols * mask`（规避燧原无先例的整型
  `tl.where` 与昇腾 masked-lane 越界地址求值 507035 族）；拷贝路径的
  masked 向量 load 为全仓 8/8 已证形态。
- source commit：`ff5c4aba4f06af8985d1b0d07fb95d8247058675`。
- ZIP：`artifacts/competition/fill_padded_rows/e1-ff5c4ab/fill_padded_rows.zip`，
  SHA-256 `96effba431c6e4037376968dfe8859269972eb7a0c8b42f7c7ad11dede4e8b6b`；
  单成员 `fill_padded_rows.py` `59e21e9c…`。
- release 回执（v2，绑定 ff5c4ab）：
  `artifacts/competition/batch5-t67e1-validate-20260912/fill_padded_rows/verification.json`，
  SHA-256 `302c3520b6a0917e9c971193555e4184705b66e3e14fc587495daada70280bde`；
  日志 SHA-256 `3f71ba92598f6335e39151d1343b41169a68bed0d0ca5f3e419579fe59d1e20c`；
  4 方法 0 失败，25 次 launch。
- 预注册：正确性 8/8 保持；均值目标 > S0 3.4329（pad 占比大时结构收益
  接近减半流量；榜首 8.3163 的结构推断即单写形态）。

## 2026-09-12 E1 平台提交（submission 13364）

- 上传与正式 POST 各一次；state submitted。file_url SHA-256：
  `63eee149c7b9b342…`（完整值见 status 快照）。额度：发后 14/30。

## 2026-09-12 E1 平台中间判决与 E2 修复

- E1（13364）六芯已过且**全面快于 S0**：沐曦 2.6518（+18%）/ 海光
  6.4892（+34%）/ 昆仑 0.5930（持平）/ 华为 1.6868（+21%）/ A 5.4350
  （+38%）/ B 4.2746（+29%）；天数回调中。**燧原 PassManager 失败**
  （exec 6523ms，vendor=generic 被选中）。
- 根因定位：E1 相对 S0（燧原已过 1.3660）的独新构造 = **运行时标量分支
  内嵌 masked 向量 load**。本仓燧原已证形态里 masked 向量 load 全部在
  顶层（deepep_permute），分支内只有 store（fill S0/deepep_permute）。
- E2（单变量）：copy load 提到分支外，行有效性并入 load mask
  （`mask & (row < n_valid)`）；去掉算术钳位（裸 masked load 尾部小越界
  在昇腾已证可过——T64 huawei 3.76x；且 i64 乘法钳位本身是第二未证构造）。
- source commit：`a9c06b9f5e0bca8df598ad07e6e95d9cd7662962`。
- ZIP：`e2-a9c06b9`，SHA-256 `4d4d356a20054af2049a064d0b32c5008196daa11e55b7ca08d4216dca5ed064`，
  单成员 `ac9097fd…`。
- release 回执：`batch5-t67e2-validate-20260912/fill_padded_rows/verification.json`，
  SHA-256 `7d93c4a6d18c84d735b7f8636a3daaca1e0871a3def93779f299a75aa1337f15`；
  日志 `5a97ffd409160759d5196902a18bba228921e9ee2a2292d517091419254e27c6`；
  4 方法 0 失败，25 launch。

## 2026-09-12 E2 平台提交（submission 13367）

- 上传与正式 POST 各一次；state submitted。额度：发后 13/30。
- 裁决点：燧原分支内 load 毒点假设（编译）；其余七芯应保持 e1 的
  +18~38% 水位。

## 2026-09-12 E2 平台终态：8/8 VALID（submission 13367，daily_seq 17）

- **八芯全过，均值 4.2969x = 新 team best（S0 3.4329，+25%）**：
  天数 11.8956 / 沐曦 2.7324 / 燧原 **1.5106（分支内 load 毒点假设证实，
  S0 1.3660 → +11%）** / 海光 6.1164 / 昆仑 0.5854 / 华为 1.9286 /
  A 5.4502 / B 4.1560。
- 沉淀（GCU 规则集补充）：**运行时标量分支内嵌 masked 向量 load = 燧原
  PassManager 编译毒点**；load 提到顶层（行有效性并入 mask）即解除。
  可迁移：燧原 vendor 的 load 一律顶层、分支只包 store。
- 对榜首（EvokeAgent 8.3163）仍差 3.9；后续轴：天数/华为 exec 偏长
  （84s/30s）提示平台 shape 大，launch/tile 仍有空间。

## 2026-09-12 E3：列分块 grid（冲榜轴，候选就绪后提交）

- 逐芯情报（14:1x）：榜首 EvokeAgent/#2 zhaxi123 在宽 shape 芯全面
  3-4x 领先——天数 18.4/22.3 vs 我 11.9、华为 17.5/7.0 vs 我 1.93、
  燧原 5.7/5.6 vs 我 1.51；窄 shape 持平。结构性判读：e2 整行单
  program（最多 8192 lane）串行+寄存器重，天数 exec 84s 佐证。
- E3：grid 改 (rows, col_tiles)，BLOCK=min(next_pow2(n_cols),1024)，
  每 program ≤1024 lane；load 保持分支外（燧原规则）。
- source commit：`e27a0426574b65cabee1e45906eb07412f4b339a`。
- ZIP：`e3-e27a042`，SHA-256 `956b6b4751e10877ddebd1e749a504e2abdc24f6f7bebf436f95ce1a08aeed64`，
  单成员 `6eac741c…`。
- release 回执：`batch5-t67e3-validate-20260912/fill_padded_rows/verification.json`，
  SHA-256 `a7e0e26196fc57da173a63e8d0df2ea8e930a3a08adde6b6a38a54d8cd003fbe`；
  日志 `7e4e132778103c37d9fe48da8fef36b1eaf3def2ff040606136515b24562ca52`；
  4 方法 0 失败，25 launch。
- 预注册：天数/华为/燧原中位 ≥1.3x；其余五芯无回退 >5%。

## 2026-09-12 E3 平台终态：8/8 valid 但未过门（submission 13386）

- 八芯全过：天数 11.3554（-4.5%）/ 沐曦 **3.3884（+24%）** / 燧原
  1.5014（0%）/ 海光 5.9458 / 昆仑 0.6020 / 华为 1.6030（-17%）/
  A 5.2440 / B 4.2660。均值 4.23825 < e2 4.2969，team best 保留 e2。
- **预注册门（天数/华为/燧原中位 ≥1.3x）三芯全负，列分块轴关闭**：
  榜首宽 shape 优势（天数 18-22/华为 17.5/燧原 5.6）不是列并行性；
  沐曦 +24% 是唯一正信号（不同后端偏好）。宽 shape 结构待新证据。
