# Task 25 `draft_topk1` 实验记录

状态:未开始(2026-08-27 第 3 批开闸)

## 契约锁定

- 签名:`reference(next_token_logits, positions, draft_tokens=None, draft_token_column=0)`
- 输入:`next_token_logits [B, V]`;`positions [B]` int64;`draft_tokens [B, D]`
  int 或 None;`draft_token_column` int(默认 0)
- 输出:`(topk_p, topk_index, out_positions, out_draft_tokens)`;
  `topk_p` = ones `[B,1]` float32;`topk_index` = argmax int64 `[B,1]`;
  `out_positions` = positions+1;`draft_tokens` 为 None 时第四项为 None,
  否则 clone 后第 `draft_token_column` 列写入 topk_index
- 容差:fp32 1e-4 / bf16 1.5e-2 / fp16 1e-2;`topk_index` 与
  `out_draft_tokens` 要求精确相等
- 关键语义:argmax 不经 softmax;平局取首索引(torch.argmax 语义),
  fp16/bf16 大 V 下平局常见,严格大于更新模式必须复刻

## 方案

- 每行一 program,V 分块循环归约 argmax(严格大于保首索引);
  `ones`/`positions+1` 小 kernel;draft 缓冲区 copy kernel + 列覆盖
- 纯归约无 dot;昆仑 BLOCK 轴、燧原无分支风险

## 提交预算与止损(2026-08-27 定稿)

- 每题 5 次提交预算;S0 首投探路 → 最多 3 次 vendor 单变量迭代 → 剩 1 次留作
  截止前回归储备。
- 同指纹失败连续 2 次提前停,不烧满 5 次;额度只花在有单变量假设的候选上。
- generic dot 策略(如涉及 `tl.dot`):fp32-ieee 操作数 generic + `_tianshu`
  split-fp16 vendor;昆仑保持 fp32-ieee(T12 镜像证据,昆仑 fp16-dot 数值失败
  有平台实证)。
