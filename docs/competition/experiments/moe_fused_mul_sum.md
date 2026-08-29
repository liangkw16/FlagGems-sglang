# Task 32 `moe_fused_mul_sum` 实验记录

## S0：generic baseline

状态：screening 进行中（远端 NVIDIA 代理）

生成时间：2026-08-29

### 契约

- 接口：`moe_fused_mul_sum(inputs, topk_weights, topk_ids=None,
  expert_map=None, routed_scaling_factor=None, is_ep=False)`，与题面
  reference 完全一致。
- `inputs` `[T, top_k, D]`（fp16/bf16/fp32）；`topk_weights` `[T, top_k]`；
  `topk_ids` `[T, top_k]` int32 或 None；`expert_map` `[num_experts]` int32
  或 None；`routed_scaling_factor` float 或 None；`is_ep` bool。
- 语义：`w = topk_weights.float() * (scale or 1.0)`；有 `expert_map` 时
  `w *= (expert_map[topk_ids] >= 0)`，否则 `is_ep` 时 `w *= (topk_ids >= 0)`；
  `out[t,d] = Σ_k inputs[t,k,d] * w[t,k]`，FP32 累加，输出 `[T, D]` 与
  inputs 同 dtype。
- 容差：FP32 `1e-4/1e-4`，BF16 `1.5e-2/1.5e-2`，FP16 `1e-2/1e-2`。
- 支持芯片：天数、沐曦、燧原、海光、昆仑芯、华为、国际通用 A/B（八芯）。
- 核心计算只走 Triton；无 try/except、无 PyTorch fallback、无设备分支。

### 生成路径

- 按仓库规则走 `kernelgen-mcp generate_kernel`（服务端 KernelGen 2.0.0，
  streamable HTTP JSON-RPC）；生成后由服务端验证：
  `passed=true`，`speedup=2.35x`（服务端 shape (64,8,7168) 口径）。
- 服务端生成版两处已知问题，整理进仓库时修正：
  1. 输出先建 FP32 tensor 再 `.to(dtype)`，多一轮全量读写 —— 改为直接
     以 inputs dtype 分配输出，`tl.store` 自动 cast；
  2. expert_map 索引用 `.to(tl.int64)` —— 改为 int32（num_experts 小，
     国产后端 int64 惩罚更重，与 T21 经验一致）。

### 唯一候选配置

- 2D grid `(num_tokens, ceil(D/256))`，BLOCK 256，`TOP_K` constexpr 展开。
- 每个 program 处理一个 token 的一个 hidden block：先逐 k load 标量权重
  （乘 scale 并按 expert_map/is_ep `tl.where` 置零），再 load 输入块 FP32
  累加；单次读 inputs、单次写 output。
- inputs 三个 stride 与 weights/ids stride 显式传入；input stride 在 kernel
  内 cast int64（大 shape 地址溢出保护），token/hidden 偏移 int32 起算。
- `HAS_EXPERT_MAP`/`IS_EP` 为 constexpr 分支；topk_ids 为 None 时传空
  int32 tensor 占位，两个分支均为编译期消除。
- 空 T/D 直接返回合法空输出；top_k=0 时循环体为空、输出为零，语义与
  reference 的空维求和一致。
- 显式 4 warps、1 stage；无 autotune、无 vendor 文件。

### 上游参考

- vLLM `fused_moe.py` 的 `moe_sum` 与 Fused MoE Modular Kernel 的
  TopKWeightAndReduce（mul+reduce 融合 epilogue 思想）。
- SGLang fused_moe_triton_kernels（一 token 一 hidden tile、FP32 累加）。
- 本仓库 T21 `moe_sum_reduce` 的地址公式与 launch 纪律。
- 引入的外部最佳实践参考：
  `docs/competition/reference/kernel-skills/patterns-fuse-elementwise-ops`。

### Screening（进行中）

- 模式：screening（未提交候选快速筛选）。
- base commit：`9714d53`（工作树另含用户未提交改动，未触碰）。
- 本地/远端 SHA-256 一致：
  - `src/flaggems_sglang/ops/moe_fused_mul_sum.py`
    `097009c6771e74bb2c67418f16e7d78bcea22d28b69f84be7b677c6c6c2c537e`
  - `tests/test_moe_fused_mul_sum.py`
    `5f3cabe245d20322b8965d994d2a8f540063d890585d202eb1f52576a3cb68e0`
- 远端证据目录：`gpu:/tmp/flagos-moe_fused_mul_sum.9hfSsb`（mode 0700，
  PID 191334，日志 `run.log`）。
- 远端环境：NVIDIA CUDA，torch 2.13.0+cu130，triton 3.7.1。
- 测试矩阵：三 dtype 主 shape、scale None、expert_map 掩码（含 -1 槽位）、
  is_ep 负 id 掩码、非连续输入、块边界（255/256/257/511/512/513）、
  空维（T=0/D=0/top_k=0）、平台规模 (4096,8,7168)。

### Screening 结果（第一轮，未通过 → 已定位）

- 远端 job 于 09:35 完成：`Ran 8 tests, FAILED (errors=26)`。
- 错误数完全吻合的根因：**测试 harness 传参 bug，非 kernel 缺陷**。
  5 个用例把 scale 写成第 3 个位置参数（该位是 `topk_ids`），
  wrapper 里 `topk_ids.stride(0)` 抛 AttributeError。错误分布
  3(dtype)+1(非连续)+18(块边界)+1(空维仅 (2,0,17) 到达 stride,
  另两 shape 被空维早退拦住)+3(平台规模)=26，与日志完全一致。
  修复：全部改为 `routed_scaling_factor=` 关键字传参。

### Screening 结果（第二轮，通过）

- 测试修复后同目录重跑（远端字节 = 本地字节）：
  - 源码 `f586b45b68172075ef2bbf12a215df1a8c789bb9d829b366c5cebc95a54ca238`
    （与 ff09392 commit 相同，未改动）；
  - 测试 `7895539ed8de370098fa1128046af178e5e5554a9e6eed74afee91093546c798`；
  - `python -m unittest -v`：**8/8 全部通过（UT=0）**；
  - isort/flake8 通过。
- black 口径澄清：远端临时目录无 `pyproject.toml`，black 回退默认
  line-length 88 才报 BLACK=1；仓库 `[tool.black] line-length = 79`
  下两文件均合规（本地 black 26.5.1 + py3.12 验证 unchanged）。
  远端证据命令应显式 `-l 79` 或携带 pyproject，已在 run3/run4 采用。
- 远端环境：NVIDIA CUDA（gpu:/tmp/flagos-moe_fused_mul_sum.9hfSsb，
  mode 0700），torch 2.13.0+cu130，triton 3.7.1；GPU host 间歇性
  outage 多次（与 T33 记录同型），日志轮询带退避。

（release 重验、benchmark 与 ZIP 打包待补；未提交平台）

## S0 定稿(2026-08-29 18:2x CST,接续会话记录)

- benchmark 假象澄清:本会话首轮 bench 的 AB/BA 标签反转,导致
  "0.11–0.55x" 假读数;修正后 kernel 实为 **2.56–8.75x**。
- 采用重写版(commit `63e2550`):flat 1D capped grid + 块级除法 + int32,
  消除原 2D grid 超华为 65535 上限、int64 逐元素、显式 launch 参数
  三处跨芯风险;K 循环权重/掩码全寄存器。
- unittest 8/8(gpu:/tmp/t32.qa34FX);bench 修正后:
  4096×8×7168 **8.75x**、65536×4×1024 8.50x、256×8×4096 5.96x、
  1024×16×512 5.61x、16×4×2048 2.56x、128×8×7168 fp32 1.85x。
- ZIP `s0-63e2550/moe_fused_mul_sum.zip`,SHA `0706e14647c36c26c785812bd79281a6e68b969efd56e7387d66f8e3124a915e`。
