# Task 45 `chunk_scaled_dot_kkt` 实验记录

```current
task: 45
operator: chunk_scaled_dot_kkt
batch: 4
validity: invalid
platform: 6/8(s0,enflame+kunlun correctness失败;huawei 0.031x低于门槛)
team_best_stage: s0
team_best_commit: d7d8c4793278062f55617693585ae2ab89c8fbcc
team_best_speedup: -
sealed: no
next: 昆仑fp32-ieee dot vendor + 燧原64/64/128+stages2+fold vendor + 华为性能轴
updated: 2026-09-04
```

状态：S0 候选就绪（generic 单文件），远端 NVIDIA 代理 screening 通过
（7/7 单测 + 三项 lint + 基准 bf16 5.4–14x）。榜首 EvokeAgent 15.0261x
（2/7 队达标）。

## 契约锁定

- 签名：`chunk_scaled_dot_kkt(k, beta, g_cumsum=None, chunk_size=64)`
- k `[B,T,Hg,K]`、beta `[B,T,H]`、g `[B,T,H]` 或 None；`ratio = H//Hg`
  GQA 共享；`T % BT == 0`（不整除 ValueError）
- 计算：k 按 head repeat ratio 倍后逐 chunk
  `A[i,j] = k[i]·k[j]`；可选 safe-exp 衰减（`d=g_i-g_j`，`d<=0` 取
  `exp(d)` 否则精确 0）；`A[i,j] *= beta[i]`；严格下三角（对角线精确
  0）；输出 `[B,T,H,BT]` **float32**
- 容差：fp32 1e-4 / bf16 1.5e-2 / fp16 1e-2；八芯；标准反作弊条款

## 方案（S0）

- bmm_chunk 骨架直接映射（T09 平台 8/8 验证结构）：grid
  `(BT M/N tiles, B, NT*H)`；kernel 内 `chunk*BT + offsets` 定位、
  k 的两次转置加载供 `tl.dot(a, b)` 两侧（K@K.T 形态）
- **输出直接写最终布局**：四维 stride 显式传入，store 偏移
  `b*sb + (n*BT+i)*st + h*sh + j*sl`，跳过 reference 的
  permute/reshape（多题验证的唯一可靠解法）
- GQA 零物化：kernel 内 `head // ratio` 索引（chunk_state.py L85）
- 合成顺序：fp32 dot 累加 → `* beta[i]` → safe-exp → `where(i>j, ., 0)`
  → 全边界 store（上三角写精确 0，不用 store mask 屏蔽）
- dtype 分派：fp16/bf16 低精度直送 dot + fp32 累加
  （USE_INPUT_DTYPE）；fp32 输入 `input_precision="ieee"` 禁 TF32
- 32/32/32、4 warps、1 stage 起步

## 验证证据（screening 模式，未提交候选）

- 远端：`gpu`（RTX 5070 Ti），torch 2.13.0+cu130，triton 3.7.1；
  目录 `/tmp/flagos-chunk_scaled_dot_kkt.fUqVW9`
  目录 `/tmp/flagos-chunk_scaled_dot_kkt.fUqVW9`
- SHA-256（与 source commit `d7d8c47` 逐字节一致）：
  - `src/flaggems_sglang/ops/chunk_scaled_dot_kkt.py`
    `20a3c83ceb77bd43ba110377750e078b034e9da1467c72295c4b96763d7c150c`
  - `tests/test_chunk_scaled_dot_kkt.py`
    `352171886685196333c68b60a90f723e03c7caf6a028df5b8ea0d7e60f86bb9d`
- 门禁：py_compile / black / isort / flake8 全绿（远端 black 重排取回，
  hash recheck 一致）；unittest 7/7 OK
- 单测覆盖：3 dtype；GQA ratio 1/2/4；g 有无（混合符号触发 `d>0`
  置 0 分支）；chunk_size 32/64/128 × K 16/32/64/100/128；严格下三角
  含对角线精确 0（permute 后 mask 断言）；B=1；非连续 k；非法 shape
  ValueError ×2；输入不变性
- 基准（do_bench median）：

  | shape | bf16 | fp32 |
  | --- | ---: | ---: |
  | B2 T1024 H8 K128 g-/g+ | 5.4x / 7.6x | 1.4x / 1.9x |
  | B4 T2048 H16 K64 | 8.6x / 14.0x | 2.7x / 4.4x |
  | B1 T4096 H16 K128 | 9.2x / 11.8x | 1.7x / 2.3x |

## 已知风险与对策（fla 族平台实证，勿重复试错）

- 天数：fp32 操作数 dot **静默算错**（T12/T13/T37 三证）→ fp32 路径
  必须 split-fp16 三点积 vendor
- 昆仑：fp16 操作数 dot 数值失败（T12 镜像）→ 只许 fp32-ieee dot
- 燧原：64/64/128 + stages≥2 + capped grid-stride fold（cap 64）
  起步配置（chunk_state/bmm_chunk 实证）
- 华为：`NT*H` 维大 shape 易超 65535 → capped grid-stride
  （`min(total,4096)`）；UB 溢出收缩 tile
- 沐曦/国际 B：generic 低精度回退时保留 ieee 字节回退 vendor；国际 A
  低精度预期大收益（chunk_state +282% 先例）

## 提交预算与止损

- 默认 5 发：S0 探路 → vendor 单变量 → 回归储备；同指纹两连败止损

## 时间线

- 2026-09-04 00:xx 契约锁定、S0 实现 + 远端 screening 7/7 + 基准
  （测试自修三轮：g 切片尺寸、整除用例、三角 view permute——均为测试
  问题，kernel 字节未动）

## 平台首投结果（2026-09-04 01:17，submission 9375，daily_seq 4）

- 6/8 正确、`invalid_correctness`（燧原、昆仑 fail；case 细节未透出）
- 逐芯：天数 5.8115 / 沐曦 5.4355 / 燧原 FAIL / 海光 14.777 /
  昆仑 FAIL / 华为 0.031（低于门槛）/ A 19.246 / B 7.9035
- 与预案一致：昆仑 fp16 操作数 dot 数值失败镜像（T12）→ fp32-ieee
  dot vendor；燧原按 64/64/128 + stages2 + capped fold 配置 vendor；
  华为 0.031x 需 Cube 低精度/结构轴


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
