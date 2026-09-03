# Task 44 `chain_speculative_sampling` 实验记录

```current
task: 44
operator: chain_speculative_sampling
batch: 4
validity: candidate-limited
platform: none(未提交)
team_best_stage: s0
team_best_commit: d7d8c4793278062f55617693585ae2ab89c8fbcc
team_best_speedup: -
sealed: no
next: 半精度逆CDF bit-exact 判定为不可行(NVIDIA代理实证);提交与否需用户门控——fp32全对,fp16/bf16最终token有~20-30%/请求失配;pending_challenge 0/6 队达标,一发探针或可换平台dtype口径情报
updated: 2026-09-04
```

状态：S0 候选就绪但**带已量化 limitation**。接受链（predicts 链 /
accept_index / accept_token_num）在全部 dtype 精确匹配；最终逆 CDF 采样
在 fp32 精确、**fp16/bf16 存在固有 bit-exact 缺口**（见下）。远端
screening 8/8（含 1 个如实标注的 expectedFailure）。题目 atol=0 +
0/6 队达标，是全批最高确定性风险题。

## 契约锁定

- 签名：`chain_speculative_sampling(candidates, retrive_index,
  uniform_samples, uniform_samples_for_final_sampling, target_probs,
  draft_probs, num_slots)`
- 逐步判据：`coin*q(t) < p(t)`（coin/q/p 同 dtype，乘积按 dtype 舍入
  后比较）；首 False 终止；最终 `val = p`（全接受）或
  `max(p - nan_to_num(q), 0)`；归一化逆 CDF
  （`cumsum > coin*sum` 首索引，无则 V-1）
- 输出：predicts `[num_slots]`（candidates dtype）、accept_index
  `[B,S]`（-1 填充，retrive dtype）、accept_token_num `[B]` int32；
  **三 dtype 全部 atol=0 精确匹配**

## 方案（S0）

- **接受链并行化**：`cur_row` 恒为 `s-1`（参考实现的隐藏不变量，
  实测确认），接受判定各步独立：接受数 = 前缀连续 True 长度。kernel 1
  每 request 一 program：标量循环 + `k += (active & accepted)`
  branchless 前缀计数；`predicts[retrive[b,s-1]] = c_s` masked store；
  accept_index 行按 `j <= k` select 写
- `coin*q` 按 `(f32*f32).to(dtype).to(f32)` 复刻 torch 标量乘舍入
- kernel 2 最终采样：val 元素 fp32 计算后 round 到输入 dtype（复刻
  torch elementwise 语义）；norm/prefix 用 fp32 串行块累加，prefix 每
  元素 round 回 dtype 后与 target_u 比较（target_u 同样按
  `coin*norm` → dtype 舍入）
- draft_probs 行基址 **`b*(S-1)+...`**（每批 S-1 行——初版错用
  `b*S`，差分 harness 定位修复；b0 碰巧掩盖）
- seqlen=1 时 draft 空：传 target 占位保地址安全（all_accepted 恒真，
  值不用）
- 全程零运行期分支（燧原约束）

## go/no-go 确定性实验（2026-09-04，NVIDIA 代理，结论 NEGATIVE-半精度）

1. `val.sum()`：fp16/bf16 与「fp32 串行 + round」**逐位一致**（20/20）；
   fp32 偶有归约序差（~2/5）
2. `torch.cumsum`（半精度）：与所有可构造候选**均不一致**——
   fp32 串行 round（0/20）、dtype 逐步累加（0/20）、
   `round(cumsum(f32))`（0/20）、blocked+Hillis 结构仿真
   （group 2–32 全 0/10）→ torch 内部为半精度累加的树形扫描，
   maxdiff 2.9e-3（fp16）/1.2e-2（bf16）
3. 端到端差分：fp32 0/10 失配；fp16 2/10、bf16 3/10 请求级最终
   token 失配（接受链全对，仅最终采样 token 偏移 1–11 个词表位）
4. **跨芯加成风险**：八芯各自 torch 构建的 cumsum 实现可能互不相同，
   即使完美复刻 NVIDIA 侧也无法保证迁移

## 验证证据（screening 模式，未提交候选）

- 远端：`gpu`（RTX 5070 Ti）；目录
  `/tmp/flagos-chain_speculative_sampling.jQQyPF`
- SHA-256（与 source commit `d7d8c47` 逐字节一致）：
  - `src/flaggems_sglang/ops/chain_speculative_sampling.py`
    `dc68f24aad83219af392d94fa58c0bad272dc11165d2ecc7bddae541e7e09718`
  - `tests/test_chain_speculative_sampling.py`
    `f251b8c14ef03fb49a62246d2c6af1112a10c5c545ebf32c0d11af4520e1b982`
  `/tmp/flagos-chain_speculative_sampling.jQQyPF`
- 门禁：py_compile / black / isort / flake8 全绿；unittest 8/8 OK
  （1 expected failure = 半精度缺口，注释含实证数据）
- 单测：fp32 精确（单例 + 40 轮随机 V=4096 全对）；接受率 0/0.3/0.7/1；
  batch×seqlen 矩阵（含 seqlen=1）；draft NaN（nan_to_num 语义）；
  全零 val 回退 V-1；空批次
- 性能：kernel 0.2–0.3us（B 16–64 × V 128k–152k）；reference 是逐
  请求 Python 循环 + `.item()`，加速比必然 T06 级（631x 先例），
  0.1x 门槛零风险

## 风险与决策门

- 平台隐藏 case 若测 fp16/bf16：约每请求 20–30% 失配概率 → 芯片级
  正确性几乎必挂 → 8/8 无望
- 若平台只测 fp32（或 dtype 集合不同）：全对
- **提交决策留给用户**（一发探针可换平台 dtype 口径情报，题目
  pending_challenge 0/6 队达标，信息价值可能值回额度）
- 昆仑采样族崩溃墙（T25/26/27/31/38 六题前科）独立存在：即使数值
  全对也可能 7/8 封顶

## 提交预算（若用户放行）

- 1 发探针 + 同指纹 2 次止损；昆仑崩溃指纹出现即封存

## 时间线

- 2026-09-04 00:xx 契约锁定、S0 实现；draft 行索引 bug 差分定位修复；
  cumsum 语义四组实验定论；screening 8/8 + 基准
