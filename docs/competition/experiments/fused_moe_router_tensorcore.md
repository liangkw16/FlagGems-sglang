# Task 27 `fused_moe_router_tensorcore` 实验记录

状态:未开始(2026-08-27 第 3 批开闸)

## 契约锁定

- 签名与计算语义与 cudacore 变体完全相同,区别:底层用 `tl.dot`,
  且 `topk <= 2`、`H` 须为 64 的倍数
- 输出:`(topk_weights [B, topk] fp32, topk_ids [B, topk] int32)`
- 容差:fp32 1e-4 / bf16 1.5e-2 / fp16 1e-2;`topk_ids` 精确相等

## 方案

- generic fp32-ieee `tl.dot`(昆仑可过、天数静默失败 → `_tianshu`
  split-fp16 三点积 vendor,fp32 1e-4 容差,T12 已验证镜像规则)
- topk≤2 → 行内两遍扫 max/second-max;近平局次序需与 torch.topk 一致,
  代理验证先测 tie 行为
- H%64 对齐 BLOCK_SIZE_K

## 提交预算与止损(2026-08-27 定稿)

- 每题 5 次提交预算;S0 首投探路 → 最多 3 次 vendor 单变量迭代 → 剩 1 次留作
  截止前回归储备。
- 同指纹失败连续 2 次提前停,不烧满 5 次;额度只花在有单变量假设的候选上。
- generic dot 策略(如涉及 `tl.dot`):fp32-ieee 操作数 generic + `_tianshu`
  split-fp16 vendor;昆仑保持 fp32-ieee(T12 镜像证据,昆仑 fp16-dot 数值失败
  有平台实证)。
