# T108 topk_sigmoid 实验账本

```current
task: 108
operator: topk_sigmoid
validity: invalid_correctness(昆仑编译墙未破)
platform: e2/sub21856 昆仑选vendor仍TritonXPULegalize失败(9case);华为8case bishengir失败(21834同generic字节曾过,疑编译抖动或ZIP内vendor成员被连带编译);muxi 0.0324异常低;card_a 1.31/card_b 0.55/海光0.78
team_best: 无有效提交(榜首EvokeAgent 4.4711,仅1队达标)
sealed: no
next: 昆仑vendor需去chunked-rank新结构(2D broadcast+axis=1 sum仍撞legalize);先拉21856华为raw_result确认失败文件身份再定huawei轴
updated: 2026-09-27
```

## 2026-09-26/27 昆仑 vendor 三连击（今日窗口）

- 根因链：generic `BLOCK_E²` rank 矩阵 + 提取循环内标量 store → 昆仑 `TritonXPULegalize` + uni_sram OOR（sub 21834 全 9 case 编译失败）。
- e1/e2 vendor（commit 4e003841）：chunked-J rank（BLOCK_J=16, axis=1）+ 单次 2D masked scatter 提取，无标量 store。NVIDIA 代理 release 4/4 过（回执 /tmp/t108-verification.json sha 6f987367…，verification_commit 4e003841）。中途修 2 个 P1/P2：codex-review 抓到 rank 方向反转（选成最低分）；代理实测抓到 scatter 指针 (1,K) vs 值 (E,K) 形状不匹配；renorm 跨 lane store→load 竞态改为 1D rank<k 归约直算+重 scatter。
- 平台 e2/sub21856：昆仑**仍** PassManager 失败（loc :18）→ chunked 2D 结构本身在 XPU legalize 不可行；另发华为 8 case bishengir 编译失败（同 generic 字节 21834 曾全过，存疑）。
- ZIP：artifacts/competition/topk_sigmoid/e2-4e00384 sha256=bf7af47b…（含 generic+topk_sigmoid_kunlunxin.py）。
- 下一步：昆仑改纯 1D 形态（e.g. 按 slot 展开的 flat kernel 或 k 轮全向量 where-max 无 2D）；华为先取 raw_result 确认失败源文件。
