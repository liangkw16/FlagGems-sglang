# Task 18 `fused_recurrent_gdn` 实验记录

## S0：experimental generic recurrent kernel

状态：RTX 5070 Ti 代理验证和不可变 ZIP 门禁通过；**八芯高风险，未提交平台**
验证时间：2026-08-24 01:45–01:52 CST
源码 commit：`de1530b`

### 契约

| 项目 | 值 |
| --- | --- |
| 公开接口 | `fused_recurrent_gdn(q, k, v, g, beta, scale, initial_state, output_final_state, use_qk_l2norm_in_kernel=False)` |
| 输入 | q/k `[B,T,H,K]`；v `[B,T,HV,V]`；g `[B,T,HV]` |
| beta | scalar/head `[B,T,HV]` 或 vector/head `[B,T,HV,V]` |
| 头映射 | `q_head = value_head // (HV // H)`，要求 `HV % H == 0` |
| 状态 | FP32 `[B,HV,V,K]`；每步依次 decay、delta correction、外积更新、q readout |
| 可选项 | q/k FP32 L2 norm；initial state；FP32 final state |
| 输出 | `[B,T,HV,V]`，dtype=`v.dtype`；final state 或 `None` |
| 容差 | `atol=1e-2, rtol=1e-2` |
| 支持芯片 | 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用 A/B，共 8 款 |
| 截止 / 门槛 | 2026-08-27 19:59:59；`speedup_threshold=0.1` |

固定来源为本地 Task 18 题面，以及 SGLang commit
`8014d9d062c3cc5d393596ecdf2f7009191965df` 的
`python/sglang/kernels/ops/attention/fla/fused_recurrent.py`。S0 保留固定源码的
前向递推和 `(V tile, B*HV)` 所有权，删除 autograd、varlen/KDA、input guard、
连续存储假设和 SGLang 私有 `exp` helper。

2026-08-24 01:51 CST 公开 API 状态：49 次提交、13 支队伍，仍为
`pending_challenge`，0 支达到门槛且无榜首。动态值仅用于当时决策。

### 唯一候选

- 一个 self-contained Triton kernel；每个 program 持有一个
  `(batch, value_head, V tile)` 的完整 `[BLOCK_V,BLOCK_K]` FP32 状态，并在
  device 端按真实 `T` 顺序循环。
- `BLOCK_K=next_power_of_2(K)`，`BLOCK_V=min(next_power_of_2(V),8)`，4 warps、
  1 stage。相较固定 SGLang 的 `BLOCK_V<=32`，S0 主动压低状态 tile，优先降低
  跨芯寄存器/本地内存风险。
- q/k/v/g/beta、initial state、output 和 final state 全部使用真实 stride；
  地址 stride 转 int64。MHA/GQA、两种 beta 均在同一 kernel 内用 constexpr
  选择，不按设备分支。
- decay、L2 norm、prediction、correction、状态更新和输出 readout 全部 FP32；
  output 最后由 store cast 到 `v.dtype`，final state 保持 FP32。
- 仅用 Torch 分配输出；非空计算路径没有 reference/demo、Torch 计算、
  `.contiguous()`、autograd、autotune、vendor、设备识别或 fallback。

### 验证

| 项目 | 值 |
| --- | --- |
| 源文件 SHA-256 | `5fbad8f892ae00b9331731a5a18135929227dfbb6cfeecc331d27690025dda73` |
| 测试 SHA-256 | `1f7ff9ea725e73940dedeedb86e2810ecf2c5ef26d71fb514fa6d77e4a200a83` |
| ZIP | `artifacts/competition/fused_recurrent_gdn/s0-de1530b/fused_recurrent_gdn.zip` |
| ZIP SHA-256 | `cf27e0e48f41fc1948075cd3bc22864e45d2387d8e61b5b6371fe1147fe9ce7f` |
| 远端证据目录 | `gpu:/tmp/flagos-task18-tdd`、`gpu:/tmp/flagos-batch2.SQaIX2` |
| 远端环境 | RTX 5070 Ti 16 GB；PyTorch 2.13.0+cu130；Triton 3.7.1；CUDA 13.0 |

- TDD 红灯先于实现：仅同步测试时，公开入口文件不存在，unittest 以
  `FileNotFoundError: .../fused_recurrent_gdn.py` 失败；实现后 unittest 2/2
  通过。
- 测试覆盖 FP16/BF16/FP32、MHA/GQA、scalar/vector beta、L2 开关、无/非零
  initial state、final state 开关、真实非连续 strides、`T=0` 和全部输入不变性。
- 额外大维度正确性：`K=128,V=127,T=8` 与
  `K=256,V=129,T=4,L2+vector beta` 均通过题面容差；最大 output 绝对误差分别
  为 `0`、`4.768e-7`，最大 final-state 误差分别为 `1.304e-8`、`1.118e-8`。
- 编译压力探针通过 `K/V/T` 为 `64/64/16`、`128/128/16`、`256/128/8`、
  `256/256/8`、`512/512/2` 的五组 BF16 GQA；该结果只证明 NVIDIA Triton
  3.7.1 能编译运行。

wrapper-inclusive BF16 benchmark；均为 `H=4,HV=8`、有初态和 final state，
`triton.testing.do_bench(warmup=25, rep=100)`：

| `B,T,K,V` | S0 (ms) | 题面 Torch reference (ms) | 代理 speedup |
| --- | ---: | ---: | ---: |
| `2,32,64,64` | 0.0206 | 2.2779 | 110.36x |
| `1,16,128,128` | 0.0245 | 1.0197 | 41.56x |
| `1,8,256,256` | 0.0171 | 0.3756 | 21.92x |

Torch reference 含 Python 时间循环，代理 speedup 不能外推为官方八芯成绩。

ZIP 由 commit `de1530b` 的算子子树直接生成，仅含顶层 UTF-8
`fused_recurrent_gdn.py`。`unzip -t`、10 MB、成员名和逐字节 SHA-256 门禁
均通过。

### 假设、风险与下一步

- 平台未公开 shape 集。S0 假设题面有效域：q/k shape 一致、所有 tensor 同设备、
  `K/V/H/HV > 0`、`HV % H == 0`、beta 只取题面两种 rank、initial state shape
  匹配。没有为未定义的非法输入发明语义。
- **首要风险是跨芯资源模型。** 状态必须跨动态 `T` 存活；即使 `BLOCK_V=8`，
  每个 program 仍持有 `8 * next_power_of_2(K)` 个 FP32 元素。NVIDIA 上
  `K=512,V=512` 可编译不代表天数/沐曦/燧原/海光/昆仑/华为可编译或不落慢速
  local memory。当前公开 49 次提交仍无任何 8 芯有效方案，与该风险一致。
- 每个 V tile 重复读取 q/k。压低 BLOCK_V 换取资源安全的代价是在大 V 上增加
  读流量；首次平台结果正确但低于门槛时，下一单变量才尝试 BLOCK_V 16。
- `tl.exp` / `tl.rsqrt` 的跨编译器精度尚未实测；题面容差为 `1e-2`，NVIDIA
  三 dtype 与大维度探针均有充分余量，但不能替代八芯结果。
- 该 S0 只适合作为一次受控平台实验；尚未消耗额度。网页上传必须取得用户对
  明确 Task、上述 ZIP 路径、SHA-256 和实时额度的当次确认。
