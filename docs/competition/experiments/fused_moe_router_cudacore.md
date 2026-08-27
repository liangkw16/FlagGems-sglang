# Task 26 `fused_moe_router_cudacore` 实验记录

状态:未开始(2026-08-27 第 3 批开闸)

## 契约锁定

- 签名:`reference(x, router_weight, topk, moe_softcapping, correction_bias=None)`
- 输入:`x [B, H]`;`router_weight [E, H]`;`topk` int;`moe_softcapping`
  float(0 不启用);`correction_bias [E]` float32 或 None
- 计算:fp32 GEMM logit → 可选 tanh 软封顶 → 可选 bias → 全 E softmax →
  top-k(`argsort(logits, descending=True)[:, :topk]`)→ gather 权重
- 输出:`(topk_weights [B, topk] fp32, topk_ids [B, topk] int32)`
- 容差:fp32 1e-4 / bf16 1.5e-2 / fp16 1e-2;`topk_ids` 精确相等
- 关键语义:权重来自全 E softmax,不对 topk 子集重归一化;topk 可大于 2

## 方案

- 不用 tl.dot(cudacore 变体):H 分块 FMA 归约,累加顺序贴近 torch fp32 GEMM
- E 整块加载,行内全 E softmax(数值稳定 max 减);topk 迭代抽取
- 风险:topk>2 argsort 精确匹配、近平局翻转;排 T27 之后做

## 提交预算与止损(2026-08-27 定稿)

- 每题 5 次提交预算;S0 首投探路 → 最多 3 次 vendor 单变量迭代 → 剩 1 次留作
  截止前回归储备。
- 同指纹失败连续 2 次提前停,不烧满 5 次;额度只花在有单变量假设的候选上。
- generic dot 策略(如涉及 `tl.dot`):fp32-ieee 操作数 generic + `_tianshu`
  split-fp16 vendor;昆仑保持 fp32-ieee(T12 镜像证据,昆仑 fp16-dot 数值失败
  有平台实证)。
