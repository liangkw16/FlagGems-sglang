# 前批任务上游 PR 扫描（2026-09-15）

来源：flagos-ai/FlagGems-sglang 全部 76 个 PR（git 协议拉取
`refs/pull/*/head`，gh/API 限流不影响）。本地引用 `origin/pr/<n>`。
覆盖范围：第二、三批算子；**无任何第四/五批算子**（仅 pr64 覆盖 T38
sigmoid_gate_topk_renorm）。

## 主要队伍与形态

| PR | 作者 | 日期 | 内容 | vendor 文件 |
| --- | --- | --- | --- | --- |
| 74/75 | dongsheng.wang | 09-14/15 | 第二批 ~15 题整包 | 73（六芯全覆盖） |
| 76 | liuhycs | 09-15 | 第二批 ~15 题整包 | 67 |
| 71/72 | xuanzhengdu-eng | 09-13 | 5 题 | 22-23 |
| 42 | ABan12 | 09-12 | chunk 族 5 题 | 24 |
| 47-53 | 杜选政 | 09-06 | 第二批单题 ×7 | 各 6 |
| 45/65-69 | yzw1128 | 09-06~11 | 单题 ×6 | 4-6 |
| 64 | sitraliqui | 09-10 | T38 | 全六芯 |

## 可学模式（按对我们第五批的价值排序）

### 1. 形状特化被拥抱而非回避（直接反转我们的去特化直觉）

pr75 `moe_sum_reduce` 把 `num_tokens/top_k/hidden_dim` 全部声明为
`tl.constexpr`；`per_token_quant_int8` 在 kernel 里放 `if N == BLOCK_N`
精确匹配快路径（免 mask）。头部队伍默认编译成本被 warmup 摊销，
按 shape 拿最优代码生成——与 triton-ascend #1673（constexpr stride
快 5.4x）同向，也与我们 T63 e9 实测"去特化反伤华为 72.99→23.41"
互证。**映射**：弱芯（尤其华为）开一条"过特化"轴——vendor 里把
runtime 维度升 constexpr、精确匹配分支——与 T67 e4（固定 BLOCK）
方向一致但更激进，可作为 e5 候选。

### 2. vendor 是薄补丁，不是重写：num_warps/num_stages 是最常用旋钮

pr75 `moe_sum_reduce` 昇腾 vendor 与 generic 逐字节相同，唯一差异
`num_warps 4→8`；pr64 T38 昇腾 vendor 唯一差异是去掉 `num_stages=2`；
metax vendor 是 warp 启发式调整。深重写只发生在昆仑（pr64 T38 的
两 kernel tile-topk 重写，sigmoid 用 `1/(1+exp(-x))` 恒等式——与我们
纪律相同）。**映射**：我们第五批候选中只有昆仑 T65 钉过 num_warps=1，
其余全部未扫过 num_warps/num_stages——这是覆盖 10+ 题的未试轴，
每芯一发即可测。明日第二发群加"launch 参数阶梯"轴。

### 3. 燧原小归约轴：整块 2D load + tl.sum 替代 runtime 循环

pr75 `moe_sum_reduce` 燧原 vendor：每 program 一个标量 token，
topk 轴一次 load 成 [BLOCK_K, BLOCK_N] 2D tile 后 `tl.sum(axis=0)`
树归约，替代 generic 的 `for topk_index in range(top_k)` 循环。
**映射**：T64（topk 循环写多目的地）/T65（topk 加权求和）的燧原
vendor 可套同一形态；与我们 e11 的预偏移轴正交可叠加。

### 4. 昆仑是唯一需要结构性重写的芯

pr64 昆仑 vendor 的两 kernel 分解（候选收集 + tile 归约）再次确认：
昆仑的问题不是参数能解决的。与我们 T69/T70/T65 昆仑经验一致，
无新增动作，仅佐证"昆仑结构轴"的投入判断。

### 5. 泄露面核查（OPSEC）

无任何队伍推送第四/五批算子；我们自己的第五批工作也从未推送上游。
公开 fork 已清理（09-15 01:00 处置），private 为唯一 push 目标，维持。

## 并入明日发射序的增量

1. launch 参数阶梯轴（模式 2）：T63/T64/T67/T74 各芯 vendor 加
   num_warps/num_stages 档位，单变量，门沿用各题既有门。
2. 华为过特化轴（模式 1）：T67 e5/T59 候选把 runtime 维度升
   constexpr，与 e4 固定 BLOCK 形成对照实验。
3. 燧原 K-tile 归约（模式 3）：T64/T65 vendor 的 topk 循环改
   2D tile + tl.sum。
