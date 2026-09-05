# Task 45 `chunk_scaled_dot_kkt` 实验记录

```current
task: 45
operator: chunk_scaled_dot_kkt
batch: 4
validity: invalid
platform: 7/8(e7,仅昆仑失败;华为0.0455→0.2535x过门槛!)
team_best_stage: e7
team_best_commit: 4d16e0701b054247b83cf09c2e39664cd750235f
team_best_speedup: -
sealed: no
next: E10输入精度dot证伪(0.0105x);E11 flat epilogue已提交待裁决;再败则昆仑轴转入深度分诊
updated: 2026-09-06
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

## E1 双 vendor（2026-09-04 03:5x，submission 9415，daily_seq 9）

- 燧原 64/64/128+stages2+fold 与 昆仑 fp32-ieee 双 vendor 同发：
  **两芯仍 correctness 失败，均两连败触发止损**
- 其余六芯读数与 S0 一致（华为 0.031 仍低于门槛）
- 结论：本题失败非配置/精度分派能解，需结构性改写（换 dot 形态或
  tile 族）才值得再投；优先级降后

## E2 无 dot 结构 + 华为 64³（2026-09-04，submission 9484，daily_seq 19，评测中）

- P1（Codex 处方）：燧原/昆仑逐 k 外积 FMA（零 dot/零转置 operand/
  零 2D 归约，K_CHUNK=128 静态展开）；华为 64³/stages1（chunk_state
  E7 Cube 形态）；source `23b1c17`，ZIP `4a0f95a1…`
- 中间态：**华为 64³ correctness 失败**（E7 形态未迁移，华为对该题
  dot 的正确性问题独立于 tile）；燧原/昆仑评测中（无 dot kernel 慢，
  接近该芯历史 1830s 超时线）；其余六芯正常（A 21.41 / 海光 14.77 /
  天数 5.66 / 沐曦 5.45 / B 7.76）
- 待终态后判定

## E3 vllm-ascend 惯用法 vendor（2026-09-04，submission 9491，daily_seq 20）

- **来源：GitHub 扫描挖到 `vllm-project/vllm-ascend#7576`**——该算子的
  昇腾生产 kernel。vendor 镜像其结构：k 以 [BT,K] 连续 block_ptr 加载
  + `tl.dot(b_k, tl.trans(b_k))`，**dot 与下三角掩码每个 k-group 只算
  一次、组内 HPG 个 head 共享**（generic 每 head 重算 dot，ratio 倍
  浪费——华为 0.031x 主嫌疑），逐 head 仅 beta 缩放 + 题面 safe-exp +
  strided block_ptr 存储
- 开发中抓到两处 bug：block_shape 需 2 幂（K 填充 next_pow2）；
  beta/g 行偏移漏 `pid_t*BT`（chunk-local 行号错用——字符串补丁在
  black 折行字节上静默未命中，改为 assert 后命中）
- 平台中间态：**燧原 correctness 翻绿且 1.6225x**（此前该芯最高
  0.031x/全部失败——第三个结构家族命中）；**华为仍败**（第三种结构
  correctness 失败，该芯对本题的毒点独立于 dot/无 dot/block_ptr 形态）
  ；昆仑评测中；其余六芯正常（A 19.25 / 海光 16.01 / 天数 5.82 /
  沐曦 5.43 / B 7.56）
- 若昆仑终态翻绿则 7/8（只差华为）；昆仑/华为双败则 6/8 但燧原
  1.62x 的大幅改善已固化为 team 资产

## E3 终态（2026-09-04 晚）

- 昆仑 **FAIL**（第 4 个独立结构：dot 配置 / 无 dot FMA / 强制 ieee /
  vllm-ascend block_ptr 惯用法全部 correctness 失败）；华为 FAIL
  （第 3 结构）→ 本题维持 6/8 invalid，但**失败芯组合从
  「燧原+昆仑」变为「华为+昆仑」**——燧原 1.6225x 翻绿是固化资产，
  后续若破译华为/昆仑任一即 7/8
- 判定：华为/昆仑均需失败 case 细节（与 T43 同一情报墙），
  无情报不再盲投

## 失败情报破译与 E4（2026-09-04 晚，submission 9504）

- **华为 E2/E3 失败根因 = BiShengHIR `ub overflow`**（64 tile 需
  2146304–3694592 bits，预算 1572864）——纯编译资源问题非数值
- **昆仑 E2/E3 = 评测器崩溃族**（1830s 超时 + compile_worker
  Aborted）——平台侧故障，按协议不计代码止损、封存等健康窗口
- E4（source `03469f9`）：华为 vendor = S0 已证可编译的 32³ K-loop
  形态 + vllm-ascend GQA 共享（每 k-group 的 dot 只算一次、组内
  HPG head 共享，计算量降 ratio 倍）
- **E4 结果：华为 correctness 翻绿，0.031→0.045x**（仍低于 0.1
  门槛）；燧原 1.6355x 维持；昆仑 waiting_callback（崩溃族观察中）
- 华为下一轴：GQA 后仍需 ~2.2x（BLOCK_K 128 单趟/更少冗余 load）；
  若昆仑健康窗口通过则 7/8（只余华为门槛）

## E5/E6 华为 persistent 形态（2026-09-04 深夜，submissions 9564/9565）

- E5（9564）：vllm-ascend 完整 64×64 tile + BK=128 + persistent batch
  loop → **华为 UB 又爆**（2646016 > 1572864 bits）
- E6（9565，source `51bea2b`）：回退 32×32 tile + BK=128 单趟 K +
  persistent batch loop → **华为 correctness 翻绿 0.0455x**（与 e4
  的 0.045x 持平——K-loop 和 launch 开销不是瓶颈，Ascend 对该形状
  的上限即在此）
- 燧原 e6 转超时（同字节 e5 过 1.63x，评测机忙）；昆仑崩溃族持续
- **判定**：华为 0.0455x 离 0.1x 门槛仍差 2.2x，persistent/BK128
  均非答案；Ascend UB 预算限制了 tile 大小（64×64 必爆），32×32 +
  GQA 共享已是该预算内最优形态。需全新的执行策略（如双核拆分或
  CANN 原生接口）才能突破

## E7 FLA persistent 大突破（2026-09-05 凌晨，submission 9850，daily_seq 1）

- **华为 0.0455→0.2535x（5.6x 跃升，首次过 0.1x 门槛！）**
  FLA PR #1023 结构：物理核 persistent + 单 head task + UB-aware BK
  + beta/g 预转 [H,B,T] + 直接指针——正是"短 accumulator 生命周期"
  带来的飞跃（source `4d16e07` 含 Codex review P1/P2 修复）
- 逐芯：天数 5.66 / 沐曦 5.66 / 燧原 1.63 / 海光 14.72 /
  **华为 0.2535** / A 19.24 / B 7.74；昆仑崩溃族（compile_worker Aborted）
- **7/8**——华为+燧原均过门槛，唯一阻挡是昆仑平台侧崩溃
- Codex review P1 修复（非连续 k contiguous）+ P2（BT cap 64）在提交前完成

## 2026-09-05 Codex 会诊作战方案（预注册）

平台证据：3 队达标 → 昆仑可过。T28（昆仑 4.40x）/T37（3.47-3.73x）的
route/materialize + 规则 GEMM 是同类 dot 题已验证昆仑解法。

**首选（1-2 发）：昆仑三段拆分**
1. wrapper 一次性布局物化：k→`[Q=B*NT*Hg, BT, K]` 连续 + 转置副本；
   beta/g→`[QH, BT]`。Gram 只按 Hg 算一次，ratio 扩展进 epilogue
2. Stage1 纯规则 GEMM：32×32×32 / GROUP_M=8 / input_precision="ieee" /
   1D flattened grid / `do_not_specialize=["M"]` /
   `tl.max_contiguous(tl.multiple_of(...))`；无 block_ptr、无 tl.trans、
   无 head 循环、无 exp
3. Stage2 epilogue：16×32 小 tile，按 reference 顺序 g_decay→beta→
   因果 mask→直接写最终连续输出偏移（上三角/对角精确零）

均值账：昆仑 0.1x 即 8/8（均值 6.88x），0.5x 仅 +0.05x——**validity
优先，不追昆仑性能**。降级阶梯：flattened 批量仍崩 → host 逐 q 单发
（T28 原样）；明确 uni_sram → 16³ 或 split-K；规则 tl.dot 仍崩 →
8×8 Gram tile + runtime `tl.range` 逐 k 外积（禁 static_range(128)）。

⚠️ 实现注意（已核源码）：T47 `_kunlunxin/chunked_sgmv_expand.py` 的
K 循环推进 `b_ptrs += BLOCK_K * stride_bn` 是潜伏笔误（应为 stride_bk，
现网 shape 单趟未触发）——勿照抄；K 多趟（33/64/96/100/128）必须单测。

## E8 昆仑 route/materialize + 规则 GEMM 三段拆分（2026-09-06 00:31，submission 10328，daily_seq 3）

- 载体：`_kunlunxin` 整体替换为三段式——wrapper 物化
  k→[Q,BT,K]/beta,g→[QH,BT] + Stage1 规则 32×32 单趟 ieee fp32 GEMM
  （1D flattened grid + GROUP_M=8 + do_not_specialize M）+ Stage2
  16×32 epilogue（beta→decay→严格下三角→直写连续输出）；
  source `016bd9c`，ZIP `e8-016bd9c` SHA-256 `5c173dc8…bfd1`（4 成员）
- screening：远端 `/tmp/flagos-t45b.*`（unittest 9/9 OK；首轮发现批量化
  步长映射 OOB——矩阵步长误作行步长，修复后全绿）
- **终态 8 芯 correctness 全过、昆仑首次翻绿——崩溃族击穿！**
  但昆仑 speedup 0.0095x < 0.1x → `invalid_threshold`
- 逐芯：天数 5.807 / 沐曦 5.648 / 燧原 1.6225 / 海光 16.0555 /
  昆仑 0.0095 / 华为 0.252 / A 19.241 / B 7.52（avg 7.0194）
- **判定**：validity 只差昆仑性能（105 倍差距，疑 wrapper torch 物化
  + 微 GEMM 开销主导）。E9 单变量 = 去物化：两个 kernel 直接按
  k/beta/g 原生 strides 寻址（全部规整 stride，无间接索引），
  fp32 上抛在 kernel 内完成，消除全部 permute/contiguous/float 拷贝
- 跨题知识沉淀：**route/materialize + 规则 GEMM 同样击穿 KKT 类
  compile-worker 崩溃族**（T28/T37/T45 三题互证，昆仑 dot 题通用解）

## E9 去物化直读 strides（2026-09-06 00:41，submission 10331，daily_seq 4）

- 单变量：GEMM/epilogue 直接按 k/beta/g 原生规整 strides 寻址，
  wrapper 只留 gram 缓冲，全部 permute/contiguous/float 拷贝消除；
  source `4e410b1`，ZIP `e9-4e410b1` SHA-256 `730b686b…7a77`
- screening：unittest 9/9 OK（首轮 `b` 标量与 tile 重名 loop-carried
  编译错，改名修复）
- **终态 8 芯 correctness 全过，昆仑 0.01x 仍 < 0.1x（去物化仅 +5%）**
  ——物化不是瓶颈；瓶颈锁定 ieee fp32 tl.dot 本身（参考是昆仑
  SDNN 快速 einsum，fp32-ieee dot 走非矩阵单元路径，~100x 差距）
- 逐芯：天数 5.7955 / 沐曦 5.3295 / 燧原 1.6365 / 海光 14.736 /
  昆仑 0.01 / 华为 0.252 / A 21.4315 / B 7.6885（avg 7.1099）
- E10 假设：k 为 fp16/bf16 时 dot 用原生输入精度（SDNN 路径；
  bf16 输入 + fp32 accumulate 与 fp32-ieee 数学等价），fp32 输入
  保持 ieee——即 generic 的 USE_INPUT_DTYPE 模式。风险：T12 曾有
  昆仑 fp16 dot 正确性失败前科，一发可决

## E10 输入精度 dot 终态（2026-09-06，submission 10333）

- fp16/bf16 输入走原生精度 dot（SDNN 路径假设）——**昆仑 0.0105x，
  与 E9 的 0.01x 持平：假设证伪**（隐藏 case 疑为 fp32，ieee 路径
  未被绕开；或昆仑低精度 dot 亦不走矩阵单元）
- 八芯 correctness 全过（天数 5.809/沐曦 5.1775/海光 14.84/华为
  0.253/A 21.138/B 7.748/昆仑 0.0105；燧原在评但已无悬念）
- E11（已 screening 9/9）：epilogue 改 flat 1024-lane（T53 已证调度
  形态），替换 QH×8 个 [16,32] 微 program；source `13bf303`
