# Task 28 `gate_up_lora_b` 实验记录

状态:未开始(2026-08-27 第 3 批开闸)

## 契约锁定

- 签名:`reference(x, gate_up_lora_b, batch_info, output_dim, base_output)`
- 输入:`x [S, 2r]`;`gate_up_lora_b [num_lora, 2*output_dim, r]`;
  `base_output [S, 2*output_dim]`;batch_info 含 seg_indptr/weight_indices/
  lora_ranks/scalings/permutation
- 计算:per segment、per 切片 i∈{gate,up}:
  `out[rows, i*od:(i+1)*od] += scaling * (x_slice @ w_slice.T)`,
  float32 累加,输出转回 base_output dtype;lora_rank 为 0 的段跳过
- 输出:与 `base_output` 同 shape 同 dtype(克隆起步)
- 容差:fp32 1e-4 / bf16 1.5e-2 / fp16 1e-2(无精确相等项)

## 方案

- T22/T23 家族复用;K=r 很小,tl.dot K 下限需 pad/mask
- 燧原套 stages≥2 + ≥64 tile + grid-stride cap 64 已验证模板;
  昆仑 SDNN 规整结构路径;float32 计算配 fp32-ieee dot
- 风险:家族前科(T22 燧原编译失败、T23 昆仑数值失败,均非本征);
  排最后做

## 提交预算与止损(2026-08-27 定稿)

- 每题 5 次提交预算;S0 首投探路 → 最多 3 次 vendor 单变量迭代 → 剩 1 次留作
  截止前回归储备。
- 同指纹失败连续 2 次提前停,不烧满 5 次;额度只花在有单变量假设的候选上。
- generic dot 策略(如涉及 `tl.dot`):fp32-ieee 操作数 generic + `_tianshu`
  split-fp16 vendor;昆仑保持 fp32-ieee(T12 镜像证据,昆仑 fp16-dot 数值失败
  有平台实证)。
