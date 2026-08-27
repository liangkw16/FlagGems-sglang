# Task 30 `interleaved_rope` 实验记录

状态:未开始(2026-08-27 第 3 批开闸;排在 T29 之后)

## 契约锁定

- 签名:`reference(x, mrope_section)`
- 输入:`x [3, S, D]` 三路 RoPE 流(t/h/w);`mrope_section [s0,s1,s2]`
  python int 列表,s0+s1+s2 = D//3
- 输出:`[S, D]`,dtype 同 x;纯选择无浮点运算:
  d%3==1 且 d < s1*3 取高度流;d%3==2 且 d < s2*3 取宽度流;
  其余(d%3==0 或超段界)取时间流
- 容差:同 dtype 逐元素,选择型天然精确

## 方案(S0)

- 每列来源仅由 d 决定;s1、s2 作为标量参数传入 kernel,列向量上
  `tl.where` 选择源指针,单 pass 读源写目的
- 纯字节搬运,内存主导;无 dot、无分支结构风险

## 提交预算与止损(2026-08-27 定稿)

- 每题 5 次提交预算;S0 首投探路 → 最多 3 次 vendor 单变量迭代 → 剩 1 次留作
  截止前回归储备。
- 同指纹失败连续 2 次提前停,不烧满 5 次;额度只花在有单变量假设的候选上。
- generic dot 策略(如涉及 `tl.dot`):fp32-ieee 操作数 generic + `_tianshu`
  split-fp16 vendor;昆仑保持 fp32-ieee(T12 镜像证据,昆仑 fp16-dot 数值失败
  有平台实证)。
