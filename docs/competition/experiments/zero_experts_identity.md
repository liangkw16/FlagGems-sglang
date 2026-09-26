# T109 zero_experts_identity 实验账本

```current
task: 109
operator: zero_experts_identity
validity: invalid_threshold(昆仑正确但0.0614x<0.1)
platform: e3/sub21861 昆仑vendor正确0失败但0.0614x(<0.1门);其余6芯generic健康(天数5.0/沐曦3.06/海光5.34/华为1.41/A3.24/B3.51);e1/sub21854昆仑正确0.0146x(平铺div/mod骨架已证数值正确)
team_best: 无有效提交(榜首EvokeAgent 3.9279,4队达标)
sealed: no
next: 昆仑差1.7x:e1平铺骨架+BLOCK 2048/4096+warp调优,或2D网格(pid0行,pid1列块)+标量基址+向量偏移的已证安全族;e2证据=per-token for循环store形态在XPU仍错编译(85.6%垃圾)
updated: 2026-09-27
```

## 2026-09-26/27 昆仑 vendor 迭代（今日窗口）

- sub 21838（generic）：昆仑 9 case 数值败，3e38 垃圾 → 输出未写全，XPU 错编译。
- e1（commit a6bd5ad9，sub 21854）：flat int32 元素网格 + div/mod 行推导 + runtime-k gather。**昆仑 0 失败（数值正确）**，但每元素 2k 次 gather → 0.0146x。NVIDIA 代理 3/3 过。ZIP e1-a6bd5ad sha=51023e3b…。
- e2（commit 936be572，sub 21858）：per-token 行向量寻址（generic 速度型）→ 昆仑又现 85.6% 垃圾 → **for 循环内 store 或行×stride 形态是墙，非仅标量行**。
- e3（commit ecadcf4b，sub 21861）：两 kernel 拆分——rowsum 归约到 fp32 临时（flat offs，lane0 向量 store）+ 平铺 scale（每元素 1 次 temp load）。NVIDIA 代理 3/3 过（回执 /tmp/t109d-verification.json，verification_commit ecadcf4b）。中途修 rowsum store 落址 bug（pid*top_k→row，OOB）。平台：昆仑正确、0.0614x（4.2x 提升，仍差 1.7x）。
- ZIP：e3-ecadcf4 sha256=cb8dc8f8…。额度已重置 29/30。
