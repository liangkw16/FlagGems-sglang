# Task 46 `chunked_embedding_lora_a` 实验记录

```current
task: 46
operator: chunked_embedding_lora_a
batch: 4
validity: valid
platform: 8/8(e2,13.998125x)
team_best_stage: e2
team_best_commit: 63557eb68f8bb8823cca377a089c1930c7ccfcb8
team_best_speedup: 13.998125
sealed: no
next: 榜首18.75x差距25%;剩余大拖累:昆仑0.227/燧原0.334/华为2.127/沐曦5.44;华为折叠轴仅+10%,下轴待新证据
updated: 2026-09-03
```

状态：S0 候选就绪（generic 单文件），远端 NVIDIA 代理 screening 通过
（10/10 单测 + 三项 lint + 基准 12.5–123.9x）。榜首 EvokeAgent 18.7483x
（1/2 队达标），天花板高。

## 契约锁定

- 签名：`chunked_embedding_lora_a(input_ids, weights, batch_info, vocab_size)`
- 输入：input_ids `[S]` int；weights `[num_lora, max_rank, vocab_size]`；
  batch_info 含 seg_indptr `[B+1]`、weight_indices `[B]`、lora_ranks
  `[num_lora]`、permutation `[S]`、bs（host int，**无 max_len 字段**）
- 计算：第 b 条请求取 rows = permutation[start:end]、
  tokens = input_ids[rows]，`out[rows, :r] = weights[w_idx, :r, tokens].T`；
  r==0 / 空段跳过
- 输出：`[S, max_rank]`，未覆盖行与 rank 右侧列保持 0
- 容差：fp32 1e-4 / bf16 1.5e-2 / fp16 1e-2；支持八芯；标准反作弊条款

## 方案（S0）

- T17 `embedding_lora_a.py` 骨架（BLOCK_RANK=128 + runtime rank loop
  尾 mask、`torch.zeros` 起步、4 warps/1 stage）适配两处契约差异：
  1. **无 max_len**：改为 capped 65535 flat grid-stride 逐全局位置
     处理，segment 用 kernel 内 **branchless 二分**（LOG2_SEGMENTS
     constexpr 静态展开，全部非 masked 标量 load；空段永不被选中，
     其越界 adapter 哨兵不会被读——T17 S0/S1 读序陷阱结构性免疫）
  2. **permutation 间接**：row = permutation[pos]，token =
     input_ids[row]，输出写 row 行
- **r==0 不用 early-return**：`rank_mask = rank_offsets < rank` 全假
  自然不写（燧原 early-return 分支嫌疑的规避）；kernel 全程零分支
- int64 地址分量沿用 T17 generic（跨芯正确性实证）；weights 走显式
  stride 支持非连续

## 验证证据（screening 模式，未提交候选）

- 远端：`gpu`（RTX 5070 Ti），torch 2.13.0+cu130，triton 3.7.1；
  目录 `/tmp/flagos-cela.FKXSVH`（0700）
- SHA-256（与 source commit `abe446c` 逐字节一致）：
  - `src/flaggems_sglang/ops/chunked_embedding_lora_a.py`
    `ad54c22cd6750e28ff0f54224b34038ee45805232652ad8a0ea1e76115bfe6ea`
  - `tests/test_chunked_embedding_lora_a.py`
    `4f901cd868f841db8818a3d3e40e82c01b366a613843ca1739fdda742256531b`
- 门禁：py_compile / black / isort / flake8 全绿；unittest 10/10 OK
- 单测覆盖：3 dtype；id dtype int32/int64；rank ∈ {0,1,127,128,129,200,256}
  （跨 BLOCK_RANK 边界）；空段 + 越界 widx 哨兵；全零 rank；单 token/
  单段；40 段不等长；rank 右侧列保持 0（确定性 adapter 映射）；非连续
  weights；空批次；输入不变性
- 基准（do_bench median，bf16、全 rank）：

  | shape | speedup |
  | --- | ---: |
  | B=32 seg=128 rank=128 vocab=32k | 13.4x |
  | B=64 seg=64 rank=256 vocab=152k | 14.9x |
  | B=16 seg=512 rank=64 vocab=32k | 12.5x |
  | B=128 seg=32 rank=32 vocab=152k | 123.9x |

  reference 是逐段 Python 循环 + `.item()`，天花板高；NVIDIA 代理
  证据不能外推八芯。

## 已知风险与对策

- 燧原：int64 地址分量 + 标量 load 组合是历史坑（T17 E2a-i32 修复）；
  generic 首投观察，弱则上 i32 route+gather vendor（wrapper 降 i32、
  删 int64 cast）
- 昆仑/华为：grid cap 已做；若 2D/折叠问题重现，按 T17 实证上
  `_ascend`/`_kunlunxin` token 折叠 vendor（token_cap=65535//bs）
- 海光：`_hygon` warps 4→2 低风险加分轴（T17 E4 +3.34%）
- 榜首 18.7x 主要看弱芯不拖垮均值

## 提交预算与止损

- 默认 5 发：S0 探路 → vendor 单变量 → 回归储备；同指纹两连败止损
- 首投排程：09-04 重置后第 3 发（T42、T43 之后）

## 时间线

- 2026-09-03 21:59 契约锁定、S0 实现 + 远端 screening 10/10 + 基准；
  commit `abe446c`；未提交（额度 0/30）

## 平台首投结果（2026-09-04 01:14，submission 9374，daily_seq 3）

- 7/8 正确、`invalid_correctness`（**仅燧原 fail**；其余全过 0.1 门槛）
- 逐芯：天数 26.778 / 沐曦 4.919 / 燧原 FAIL / 海光 20.354 /
  昆仑 0.2325 / 华为 2.0755 / A 30.2205 / B 17.439
- 燧原指纹与 T17 S0/S1 同族（int64 地址分量 + 标量控制流）；
  修复路径 T17 E2a-i32 已实证：metadata 降 i32、route 预计算 + gather
  零标量分支；修复后 7 芯均值 ~14.6x（榜首 18.7x）

## E1 燧原 vendor（2026-09-04 01:52，submission 9381，daily_seq 6）

- **8/8 valid，平均 12.5165625x**（ZIP `57699a9e…`，source `7cb4558`）
- vendor：`chunked_embedding_lora_a_enflame.py`（T17 E2a-i32 配方：metadata
  全 i32、零 `tl.cast(int64)`、searchsorted 设备端预路由替代 kernel 内
  二分、直线 gather 无早退）
- 燧原 FAIL→0.246x（过门槛）；其余七芯读数与 S0 水位一致
  （天数 26.704 / 沐曦 4.8065 / 海光 19.9505 / 昆仑 0.2315 /
  华为 1.9265 / A 29.349 / B 16.9185）
- 榜首 EvokeAgent 18.7483x；差距集中在弱芯（昆仑/华为/沐曦）


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

## E2 华为折叠 + 海光 warps2（2026-09-04 06:4x，submission 9447，daily_seq 13）

- **8/8 valid，平均 13.998125x**（12.5166→+11.8%；source `63557eb`，
  ZIP `1eece864…`）
- 海光 warps2：19.9505→**27.505（+38%）**；沐曦水位上移
  4.8065→5.4445；燧原 0.246→0.334；国际A 29.349→31.8545；华为折叠
  vendor：1.9265→2.127（+10%，幅度有限）；昆仑 0.2315→0.227（水位）
- 榜首 EvokeAgent 18.7483x，差距收窄至 25%
