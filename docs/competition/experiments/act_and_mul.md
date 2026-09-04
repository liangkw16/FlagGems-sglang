# Task 42 `act_and_mul` 实验记录

```current
task: 42
operator: act_and_mul
batch: 4
validity: valid
platform: 8/8(e1,3.1647x)
team_best_stage: e1
team_best_commit: 1273a60095dd1db08435ac89d25fc840eed85266
team_best_speedup: 3.1647
sealed: no
next: 榜首Warmhearted 3.3698x(7队过线);剩余拖累:昆仑0.455/华为1.668;华为BLOCK轴或slab结构可再试一发
updated: 2026-09-03
```

状态：S0 候选就绪（generic 单文件），远端 NVIDIA 代理 screening 通过
（11/11 单测 + 三项 lint + 基准水位健康），未打包、未提交。2026-09-03
当日额度 30/30 已耗在第 3 批（最后 16:06），本题为第 4 批开题首投候选。

## 契约锁定

- 签名：`act_and_mul(gateup_output, activation="silu", swiglu_limit=None)`
- 输入：`gateup_output [M, 2H]`（**严格 2D**，参考实现 `[:, :half]` 切
  dim 1，>2D 无定义语义）；`gate = [:, :H]`，`up = [:, H:]`
- 计算：gate/up 先 `.float()`；可选 clamp（gate 只 max，up 对称）；
  activation ∈ {"silu","gelu(tanh)"}，其他值 `ValueError`；
  **激活在 fp32 算完后 cast 回输入 dtype，与 up（fp32 clamp 后同样 cast
  回）在输入 dtype 下相乘**——精度切换点与 T7/T29/T39 的 fp32 乘不同
- 输出：`[M, H]`，dtype 同输入
- 容差：fp32 1e-4 / bf16 1.5e-2 / fp16 1e-2
- 支持八芯；反作弊：核心计算必须 Triton，禁 try/except / 设备判断 /
  PyTorch fallback

## 方案（S0）

- 骨架复用 T29 `gelu_and_mul.py`（row/col-block 映射 + capped 65535
  grid-stride + BLOCK_COL=1024，两题八芯验证的结构）：
  每 program 一行 × 1024 列，gate/up 双载，块级除法（每块一次
  `row_id = block_id // num_col_blocks`，燧原逐元素除法贵的教训）
- SiLU 逐字题面公式 `gate / (1 + exp(-gate))`（fp32，禁稳定化改写）
- GELU-tanh：`0.5*x*(1+tanh(0.7978845608*(x+0.044715x³)))`，tanh 用
  稳定形式 `s·(1-e^{-2|y|})/(1+e^{-2|y|})`（饱和精确 ±1、无溢出、
  无 libdevice；避开昆仑 erf 崩溃族）。`tl.math.tanh` 留作 vendor 轴
- `HAS_LIMIT`/`ACT_IS_GELU` 均为 host 已知量 → constexpr 特化，非运行期
  分支（燧原约束）
- store 行：`act.to(elem_ty) * up.to(elem_ty)`（输入 dtype 乘法）
- 不显式设 num_warps/num_stages；wrapper `contiguous()` + 空输入 guard

## 验证证据（screening 模式，未提交候选）

- 远端：`gpu`（kkgpu/RTX 5070 Ti，driver 610.57.04），
  torch 2.13.0+cu130，triton 3.7.1，CUDA 可用
- 目录：`/tmp/flagos-act_and_mul.1zQ1I9`（0700）；两文件 SHA-256 与
  上述 source commit 逐字节一致：
  - `src/flaggems_sglang/ops/act_and_mul.py`
    `c5b7a981ee79694664a77ff266ba6143866afa292858e5395d6de495bba15acb`
  - `tests/test_act_and_mul.py`
    `a894bd19eeea52ed850c2938c974006e85d6e57d419325d74c684600afef574c`
- 门禁：py_compile 通过；black/isort(--profile black --line-length 80)/
  flake8(--max-line-length=120) 全绿（字节为远端 black 重排后取回，
  hash recheck 一致）；`python -m unittest -v` 11/11 OK（1.5s）
- 单测覆盖：3 dtype × 2 activation、边界 rows/half（1/7/63/64/65/511/
  512/513/1023/1024/1025/4096）、非连续、输入不变性、空输入/零宽、
  特殊值（±inf/±1e4/±92/±90/NaN，equal_nan）、limit∈{0.5,7,1e4}、
  **limit=0.0 仍生效**（falsy 非 None）、不对称 clamp 两侧探针、
  非法 activation ValueError、2D 契约（4097×8192 大形状）
- 基准（`triton.testing.do_bench` warmup=25 rep=100 median，AB 同机）：

  | shape | dtype | silu | gelu+limit7 |
  | --- | --- | ---: | ---: |
  | 4096×2048 | bf16 | 5.07x | 7.57x |
  | 16384×2048 | bf16 | 6.40x | 9.06x |
  | 4096×7168 | bf16 | 6.44x | 9.05x |
  | 1024×512 | bf16 | 3.14x | 3.71x |
  | 16384×2048 | fp32 | 1.68x | 3.04x |

  NVIDIA 代理证据，不能外推八芯。当前公开榜首 c2flow 3.1941x
  （八芯均值，含弱芯拖累），S0 水位判断有余量。

## 已知风险与对策

- 昆仑：无 erf/libdevice 依赖，无运行期分支；若读数弱，唯一有效轴
  BLOCK=2048（T29/T39 双实证）
- 燧原：整行 BLOCK 4096 四证强假设（T24/T33/T39/T29）；kernel 内无
  运行期分支已满足
- 华为：BLOCK 512 -42% 勿碰（T39）
- 沐曦：flat 大 BLOCK 2048（+65%/+11% 双证）
- 国际 B：四档列 tile autotune（128/2w、256/4w、512/8w、1024/8w，
  key=half_width）+3.83% 可搬；国际 A autotune 证伪不投
- 海光水位波动 34–56x，单轮高值不当结构收益
- gelu tanh 分支 fp32 下我们的稳定 tanh 与 torch tanh 有 ulp 级差，
  容差 1e-4 内（单测已含三 dtype 特殊值对齐）

## MCP 实机初筛（注入执行协议，2026-09-04 凌晨）

- 协议：`autotune_kernel` description 注入 VERBATIM 候选全文，
  `operator_name=<算子名>`（**不得加前缀**——首轮 `screen_` 前缀使
  harness 按前缀名调用、全部 NameError，失败神谕被污染，已废弃重交，
  污染产物存 `log/kernelgen-round/screen-prefix-polluted/`）
- **华为**：completed；iteration 5 轮零 error（失败神谕未触发）；
  终态代码与候选 kernel 逻辑逐字一致（diff 仅注释/license 头）；
  total_tests=0 → 按可信度阶梯记
  `mcp-compile-screened(fidelity)`，不采信 passed/自测 speedup；
  产物 `log/kernelgen-round/out_act_and_mul_huawei.json`
  SHA-256 `7f5981c381650ea4…`（完整哈希见文件）
- **天数**：completed；同样零 error + 逻辑保真 →
  `mcp-compile-screened(fidelity)`；产物
  `out_act_and_mul_tianshu.json` SHA-256 `d8e2621c5f8aab27…`
- 海光/沐曦：后台队列进行中（结果落地后续记）
- 结论：发射前编译风险最高的两家（华为/天数）无编译失败信号

## 提交预算与止损（2026-09-03 定稿）

- 默认 5 发：S0 探路 → 最多 3 次 vendor 单变量 → 1 发回归储备；
  同指纹失败连 2 次提前停
- 首投排程：09-04 00:00 额度重置后第 1 发（打包 → preflight → 一次性
  提交，门禁全过即自动执行）
- 每轮只改一个 vendor，其余字节逐字节冻结；冻结芯分数变化按水位处理

## 时间线

- 2026-09-03 20:42 题面同步、契约锁定、S0 实现 + 远端 screening
  11/11 通过 + 基准；commit `2652a4e`；未提交（额度 0/30）

## 平台首投结果（2026-09-04 01:09，submission 9370，daily_seq 1）

- **8/8 valid，平均 3.066825x**（ZIP `3d78528e…`，source `2652a4e`）
- 逐芯：天数 6.2234 / 沐曦 2.2476 / 燧原 0.969 / 海光 4.9658 /
  昆仑 0.279 / 华为 1.6714 / 国际A 4.538 / 国际B 3.6404
- 全芯正确性通过；最低昆仑 0.279x 仍过 0.1 门槛
- 榜首 c2flow 3.1941x，差距 4.2%；下一轴按预案：昆仑 BLOCK 2048 /
  燧原整行 4096 / 华为块结构（T39 E10 +246% 同构题）


## MCP 实机初筛归档（2026-09-04 晨，24 job 全部终态）

- 产物 `log/kernelgen-round/out_<op>_<chip>.json`（24 个，含 SHA）；
  协议：注入执行 + 失败神谕 + 终态代码保真 diff
- 干净通过（fidelity=True 且零 hard error）：本题华为/天数（详见
  各算子行）；海光/沐曦后端当夜多次 502（`ld0428.baai.ac.cn`），
  这些芯的编译信号不可得，非候选失败
- 保真失败（LLM 改写）= 无判定，不作数；harness 侧 artifact
  （NameError/IndexError/`constexpr[0]`）不计入失败神谕
- 平台实测（本账本上方小节）已是更强证据，MCP 结论仅作发射风险
  参考留存

## E1 三 vendor 冲分（2026-09-04 04:3x，submission 9416，daily_seq 10）

- **8/8 valid，平均 3.1647x**（3.0668 → +3.2%；source `1273a60`，
  ZIP `bfe4389f…`）
- 燧原 4096 整行：0.969→**1.682（+74%）**；昆仑 flat 2048：
  0.279→**0.4548（+63%）**；沐曦 flat 2048：2.2476→2.4552（+9%）
- 未动芯：天数 6.214 / 海光 4.9688 / 华为 1.668 / A 4.238（水位波动）
  / B 3.6368
- 榜首已升至 Warmhearted 3.369775x（7 队过线）
