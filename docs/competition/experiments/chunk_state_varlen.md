# Task 13 `chunk_state_varlen` 实验记录

## S0：generic baseline

状态：本地静态门禁、RTX 5070 Ti 代理验证和不可变 ZIP 门禁通过；未提交平台
验证时间：2026-08-24 01:35–01:40 CST
源码 commit：`b05bfeb`

### 契约

| 项目 | 值 |
| --- | --- |
| 公开接口 | `chunk_state_varlen(B, x, dt, dA_cumsum, cu_seqlens, chunk_states)` |
| packed 输入 | x `[total,nheads,headdim]`；B `[total,ngroups,dstate]` |
| chunk 参数 | dt / dA `[nheads,nchunks,chunk_size]`；cu_seqlens `[batch+1]` |
| 头映射 | `group = head // (nheads // ngroups)` |
| 计算 | 每条序列末 chunk 上 `x.T @ (B * exp(dA_last-dA) * dt)` |
| `chunk_states` | **只读取 dtype；数值、shape 和存储均不参与计算** |
| 输出 | `[batch,nheads,headdim,dstate]`，dtype=`chunk_states.dtype` |
| 支持芯片 | 天数、沐曦、燧原、海光、昆仑芯、华为、国际通用 A/B，共 8 款 |
| 截止 / 门槛 | 2026-08-27 19:59:59；`speedup_threshold=0.1` |

固定来源为本地 Task 13 题面及 Mamba v2.2.4
`mamba_ssm/ops/triton/ssd_chunk_state.py::_chunk_state_varlen_kernel`。上游在序列
跨入末 chunk 时会累加 `chunk_states * exp(dA_last)`；题面 reference 明确没有
该项，因此 S0 连 kernel 参数都不传 `chunk_states` 指针，只在 wrapper 读取 dtype。

2026-08-24 01:39 CST 公开 API 状态：65 次提交、11 支队伍、2 支达到门槛；
榜首 MakeYUNAGreatAgain 为 8/8、477.784875x。动态值仅用于当时决策。

> **红线：公开 reference 的跨 chunk 定义自相矛盾。** 它用整段
> `x[start:end]` / `B[start:end]`，却只从最后一个 chunk 对 dt/dA 做 Python
> 负切片。`chunk_size=8`、单序列长度 10 时分别得到长度 10 和 2，直接广播
> 报错；长度 9/17 时 scale 长度 1 又会被静默广播到整段。S0 只承诺每条序列
> 完全位于其最后一个全局 chunk 内的可定义范围，不擅自补上游语义。

### 唯一候选

- 一个 self-contained Triton kernel；grid 为 output 的 32×32 矩阵 tile、batch、
  head，K tile 32，4 warps、1 stage，无 autotune/vendor。
- packed B/x、dt、dA、cu_seqlens 和 output 全部使用真实 stride；cu_seqlens
  支持 int32/int64，并转 int64 做地址计算。
- decay、scale、X/B 和 accumulator 全为 FP32；`tl.dot(...,
  input_precision="ieee")` 明确禁止 FP32 TF32。
- 输出由 `torch.empty` 按 `chunk_states.dtype` 分配；非空路径核心计算全在
  Triton，无设备判断、连续化或 fallback。

### 验证

| 项目 | 值 |
| --- | --- |
| 源文件 SHA-256 | `29da5027f726c9976b3251f21a4dea59efd45ec8d41e10dffe801f554af97707` |
| 测试 SHA-256 | `e678acd4cdc7390e52f1971becdd05a366ccb235847e8ade33e0c12b7b6ba907` |
| ZIP | `artifacts/competition/chunk_state_varlen/s0-b05bfeb/chunk_state_varlen.zip` |
| ZIP SHA-256 | `bd23ddad1c833c8f9ba2c8e0e551fa5e4c3d7ad446351d74a346af14c850603b` |
| 远端证据目录 | `gpu:/tmp/flagos-batch2.SQaIX2`，mode 0700 |
| 远端环境 | RTX 5070 Ti 16 GB；PyTorch 2.13.0+cu130；Triton 3.7.1；CUDA 13.0 |

- unittest 3/3 通过；覆盖 FP16/BF16/FP32 输入和输出，GQA `6 heads / 2
  groups`，多序列 `[3,5,4]` / `[1,7,2,6]`，两个全局 chunk，所有输入
  非连续、int32/int64 cu_seqlens、空 batch 和输入不变性。
- 使用两组不同且非零的 `chunk_states` 内容重复调用，输出逐位一致；同时输出
  dtype 严格跟随它。
- `py_compile`、`git diff --check`、Black 79、isort、flake8 均通过；本地与
  远端源码/测试 SHA-256 完全一致。
- wrapper-inclusive FP16 benchmark：`batch=16,nheads=8,ngroups=2,
  headdim=dstate=64,chunk_size=64`，S0 `0.010662 ms`，题面 reference
  `6.539520 ms`，代理 speedup `613.340x`。reference 含 Python/head 循环与
  `.item()` 同步，不能外推为八芯成绩。
- ZIP 由 commit `b05bfeb` 的算子子树直接生成，仅含顶层 UTF-8
  `chunk_state_varlen.py`；`unzip -t`、10 MB、成员名和逐字节 SHA-256 门禁
  均通过。

### 风险与下一步

- NVIDIA 代理不能证明其余七款芯片；平台前不增加 vendor 文件。
- 只覆盖上面标明的 reference 可定义域。若平台实际给跨 chunk 用例，必须先
  由赛方澄清是“仅末 chunk token”还是“前序 chunk_states 延续”，不能猜测。
- 若平台正确但矩阵维度尾块性能不足，下一单变量候选才调整 32×32 tile。
- 未提交平台或消耗额度；上传前仍需用户确认上述路径、SHA-256 和实时额度。
