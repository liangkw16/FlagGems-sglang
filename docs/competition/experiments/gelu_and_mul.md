# Task 29 `gelu_and_mul` 实验记录

状态:S0 候选就绪,待额度重置后首投。2026-08-27 团队当日额度 30/30 已耗尽
(最近一次提交 14:15:01 CST,非本会话所为);计划 2026-08-28 00:00 额度
重置后立即 preflight + 一次性提交。

## 契约锁定

- 签名:`reference(hidden_states)`
- 输入:`hidden_states [bs, 2*d]` 任意浮点 dtype;输出 `[bs, d]` 同 dtype
- 计算:`out = gelu(x1.float(), approximate="none") * x3.float()` 转回输入
  dtype;精确 erf 公式 `x * (1 + erf(x / sqrt(2))) / 2`,禁 tanh 近似
- 容差:fp32 1e-4 / bf16 1.5e-2 / fp16 1e-2(无精确相等项)
- 支持八芯;反作弊:核心计算必须 Triton,禁 try/except / 设备判断 /
  PyTorch fallback

## 方案(S0)

- 单 pass 逐元素:每 program 处理一行内 BLOCK 列,读 gate 列与 up 列,
  fp32 计算 erf GELU × up,写输出;完整 tail mask
- erf 用 `tl.math.erf`;某芯 lowering 缺失时备选 Abramowitz-Stegun
  7.1.26 有理逼近(误差 ~1.5e-7,fp32 容差内)——仅在代理/平台证据
  显示必要时启用
- 不显式设置 num_warps/num_stages;支持非连续输入

## 提交预算与止损(2026-08-27 定稿)

- 每题 5 次提交预算;S0 首投探路 → 最多 3 次 vendor 单变量迭代 → 剩 1 次留作
  截止前回归储备。
- 同指纹失败连续 2 次提前停,不烧满 5 次;额度只花在有单变量假设的候选上。
- generic dot 策略(如涉及 `tl.dot`):fp32-ieee 操作数 generic + `_tianshu`
  split-fp16 vendor;昆仑保持 fp32-ieee(T12 镜像证据,昆仑 fp16-dot 数值失败
  有平台实证)。

## S0:generic baseline

状态:候选就绪,未提交(额度阻塞,非门禁失败)
时间:2026-08-27 21:20–21:45 CST
source/verification commit(同一提交):`67364b316ebd6039f2e7e07207ee7fec1910280c`

### 构建身份

| 项目 | 值 |
| --- | --- |
| 源文件 | `src/flaggems_sglang/ops/gelu_and_mul.py` |
| 源文件 SHA-256 | `679a7d2c702ffe84fa675448e283df39cdc924d8dd40d214e0cf709da0dc6d66` |
| 测试 SHA-256 | `9605dba61e1637004143e109f36e7c5a4e00d9b48f6c51ff396030f9c6e97bd2` |
| ZIP | `artifacts/competition/gelu_and_mul/s0-67364b3/gelu_and_mul.zip` |
| ZIP SHA-256 | `b26205dbd5a6ca2230ef67377082d974a1167d79f825db546c655047a1733185`(与 canonical 一致) |
| ZIP 内容 | 单个顶层文件 `gelu_and_mul.py`,2184 bytes;ZIP 2312 bytes,10 MB 门禁通过 |
| screening 目录 | `gpu:/tmp/flagos-gelu-and-mul.rej1pH`,mode 0700 |
| release 目录 | `gpu:/tmp/flagos-gelu-release.lODllF`,mode 0700,文件取自 Git 对象 |

ZIP 成员字节、commit blob、远端 screening/release 目录三方 SHA-256 逐项
一致;无 vendor 文件、测试、缓存或目录前缀。

### 唯一候选配置

- generic Triton kernel;flat 输出索引,`row = offs // d`、`col = offs - row*d`,
  gate 读 `base`、up 读 `base + d`,fp32 计算精确 erf GELU × up,store 处
  cast 回输入 dtype。
- BLOCK 1024;1D grid `min(cdiv(n, 1024), 65535)` + kernel 内 grid-stride
  折叠循环(T12 已平台验证的跨芯安全模式);不显式设置 `num_warps`/
  `num_stages`。
- wrapper:`contiguous()` 支持非连续输入;`last_dim == 0` 与空输入提前返回;
  任意前导维按行展平;无 try/except、设备判断或 PyTorch fallback。
- 常量内联(Triton 3.7 禁止 jit 内访问模块级普通全局,首轮 screening 实证
  修复)。

### 正确性

远端环境:RTX 5070 Ti 16 GB、Python 3.12.13、PyTorch 2.13.0+cu130、
Triton 3.7.1。lint 门禁:isort/flake8 远端通过;black 以仓库基线 25.12.0
本地通过(远端 26.5.1 hug_parens 漂移按 T24 先例记录不拦截,
`black26-diff-exit=1` 仅报告)。

screening 与 release(取自 Git 对象)两次均 8/8 通过,覆盖:

- fp32/fp16/bf16 三 dtype 容差矩阵;
- (rows, d) 边界 1×1 至 129×1025(含 63/64/65、511/512/513、1023/1024/1025
  尾块);
- 多前导维 `[2,3,5,256]`;非连续列切片输入;输入不变性;
- 空行 `[0,64]`、零宽 `[8,0]`;
- ±Inf/±0/NaN/大幅值特殊值(equal_nan);
- 65536×1024 强制 grid 折叠路径(n = 67,108,864 > 65535×1024)。

### 远端 NVIDIA 代理性能(wrapper-inclusive,do_bench warmup=25 rep=100,
五组 AB/BA 交替 p50 中位数)

| dtype | rows×d | op p50 (ms) | torch p50 (ms) | speedup |
| --- | ---: | ---: | ---: | ---: |
| float16 | 256×512 | 0.006208 | 0.014336 | 2.3093x |
| float16 | 4096×2048 | 0.067776 | 0.316416 | 4.6686x |
| float16 | 8192×4096 | 0.261088 | 1.675296 | 6.4166x |
| float16 | 65536×4096 | 2.055136 | 13.367264 | 6.5043x |
| bfloat16 | 256×512 | 0.006208 | 0.014336 | 2.3093x |
| bfloat16 | 4096×2048 | 0.067760 | 0.316384 | 4.6692x |
| bfloat16 | 8192×4096 | 0.262016 | 1.674320 | 6.3901x |
| bfloat16 | 65536×4096 | 2.055168 | 13.381632 | 6.5112x |
| float32 | 256×512 | 0.006208 | 0.008256 | 1.3299x |
| float32 | 4096×2048 | 0.132928 | 0.200768 | 1.5104x |
| float32 | 8192×4096 | 0.516000 | 0.864352 | 1.6751x |
| float32 | 65536×4096 | 4.096000 | 6.876224 | 1.6788x |

最差 1.3299x(小 shape fp32)。编译产物:4 warps、3 stages;n_regs/n_spills
metadata 未暴露。

### 已知边界与风险

- `tl.math.erf` 仅在 NVIDIA 代理验证过 lowering;tanh(T24)之外的其他
  超越函数跨芯 lowering 未知——若平台某芯编译失败,首修复假设为
  Abramowitz-Stegun 7.1.26 有理逼近替代 erf(纯算术,fp32 容差内)。
- NVIDIA 代理结论不能外推八芯;加速比以平台为准。
- int32 索引上限 2^31 元素,超出题面合理范围。

### 提交计划

- 额度重置后 preflight tuple:season 2、race `782kzq4m`、account
  `15600308080`、team `SoulCoder`、batch 3、task 29、tid `s2t1op029`、
  operator `gelu_and_mul`、stage `s0`、commit
  `67364b316ebd6039f2e7e07207ee7fec1910280c`、ZIP
  `artifacts/competition/gelu_and_mul/s0-67364b3/gelu_and_mul.zip`、
  SHA-256 `b26205dbd5a6ca2230ef67377082d974a1167d79f825db546c655047a1733185`、
  member `gelu_and_mul.py`。门禁全过即自动执行 confirm 命令。
