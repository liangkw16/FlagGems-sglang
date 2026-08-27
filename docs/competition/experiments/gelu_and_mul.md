# Task 29 `gelu_and_mul` 实验记录

状态:进行中(2026-08-27 第 3 批开闸当晚启动,本批首攻)

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
