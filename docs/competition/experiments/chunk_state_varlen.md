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

## S1：匹配公开 reference 的 Python slice 与广播语义

状态：确定性正确性修复已通过提交字节发布门禁并生成不可变 ZIP；未提交平台

验证时间：2026-08-24 07:02–07:09 CST

源码 commit：`791193061b115177052b365441baee52c1f3a81e`

### 根因与修复

S0 不是公开 reference 的实现。`chunk_size=8`、`cu_seqlens=[0,9]` 时，公开
reference 将最后 chunk 的单个 scale 广播到整段；S0 却从最后 chunk 指针向前
线性读取前一 chunk，导致 `x=B=1`、前一 chunk 的 `dt=0`、最后 scale 为 1 时
reference 返回 9、S0 返回 1。现有用例中的每条序列都恰好位于其结束 chunk
内，未覆盖该反例。空序列还会让 S0 构造 `dA[-1]` 地址。

S1 只修正这一根因，不调整 tile 或并行策略：

- 按长度为 `chunk_size` 的 Python 一维 slice 规则归一化负 `start_relative`，
  计算 `slice_start`、`slice_stop` 和 `scale_length`。
- `scale_length == 1` 时把唯一 dt/dA scale 广播到整段；
  `scale_length == sequence_length` 时逐 token 读取。
- `dA_last` 对空序列使用 masked load；K 循环执行零次并写出全零，消除负地址
  读取。
- 对公开 reference 自身会因 shape 不匹配而报错的
  `scale_length not in {1, sequence_length}` 保持未定义，不猜测 Mamba 的
  `chunk_states` 延续语义。

### 发布验证

| 项目 | 值 |
| --- | --- |
| 源文件 SHA-256 | `025ac23b0026a423e381f8fffe3715c4e21ce64558e79ad4841145351f11d4f1` |
| 测试 SHA-256 | `c827c21d3dc44c3ae344cf93d680b2ba2fef44e1670a741427fe47ae0c85b27c` |
| ZIP | `artifacts/competition/chunk_state_varlen/s1-7911930/chunk_state_varlen.zip` |
| ZIP SHA-256 | `fcc17df06adf338578402f315e4dab75bf2361e641885bb99f94e23be46efd49` |
| ZIP 大小 / 成员 | 7,516 bytes；顶层 `chunk_state_varlen.py` 7,376 bytes |
| 远端发布目录 | `gpu:/tmp/flagos-chunk-state-release.KKAmMf`，mode 0700 |
| 远端环境 | RTX 5070 Ti 16 GB；PyTorch 2.13.0+cu130；Triton 3.7.1；CUDA 13.0 |

- 从 source commit 导出的发布字节执行 unittest 5/5 通过。新增永久回归精确
  覆盖 `[0,9]` 的单 scale 广播，以及 `[0,0,1]` 的 leading-empty 序列；原有
  三 dtype、GQA、非连续 stride、int32/int64 和输入不变回归全部保留。
- Black 79、isort、flake8、`py_compile` 和 `git diff --check` 通过；远端
  source/test 与 commit SHA-256 一致。
- 仅在 S0/S1 都正确的两个 control shape 上做 FP16/BF16/FP32 五轮交替配对：
  六点中位 speedup 为 `1.0020/0.9997`、`1.0004/0.9972`、
  `0.9965/1.0182`，几何均值 `1.0023x`。编译变体为 40–56
  registers/thread、8,192-byte shared，0 stack、0 local、0 scratch。
- 确定性打包器从 commit 直接生成 ZIP；重复验签得到相同 canonical SHA-256，
  `unzip -t`、UTF-8、成员名、10 MB 和逐字节来源门禁均通过。

S1 只解决公开 reference 能正常返回的域，未做性能 E1，也未获得八芯平台证据。
上传前必须重新读取实时额度，并取得用户针对 Task 13、上述绝对 ZIP 路径和完整
SHA-256 的当次确认。

## E1：低精度原生 dot（拒绝）

状态：screening 正确性失败，源码已逐字节恢复 S1；未生成 ZIP、未提交平台

验证时间：2026-08-24 08:27–08:35 CST

固定 Mamba v2.2.4 上游会在 FP32 计算 decay/scale 后，把缩放后的 `B` 转回
`x` dtype 再进入 `tl.dot`。screening 先验证了宽矩阵上的性能空间：K=64、
`headdim/dstate` 为 128×256、256×128 或 256×256 时，RTX 5070 Ti 代理约为
`1.13–1.15x`；较小 K 或矩阵则可能回退，因此后续候选已收窄到宽矩阵。

但该 downcast 不满足题面 `atol=rtol=3e-2` 的完整数值契约。确定性 FP16
反例使用 K=2、`x=[64,-48]`、`B=[1,1]`、
`dt=[2.2734375,2.86328125]`、`dA=[0,-0.0570068359375]`，其 FP32
reference 和 S1 都精确为 0，低精度 scaled-B dot 输出 `0.0625`，绝对误差
超过 `0.03`。因此不以随机样本上的小误差或平均性能替代契约门禁，E1 直接拒绝。

screening base commit 为 `3c38f1c25db250e1a290067a610ae77170b89b6a`；最终
候选源码 SHA-256 为
`08e6c3f3edf4e80091a30a1811f16f8055115b94b8787650657731eb241d7fea`。
远端目录为 `gpu:/tmp/flagos-chunk-state-varlen.sJaYzJ`（mode 0700）；静态与
原有 5/5 单测日志 SHA-256 为
`f647610d779661a65a67ef4d7fc52dcbf2ea15d68c59de5d9066775c9c76e5a1`，反例
日志 SHA-256 为
`208cd018d28fe8af01e855b299a6b02a9cca8d5a0d8c416ecaf32c537c916786`。
