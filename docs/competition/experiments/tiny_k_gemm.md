# Task 91 `tiny_k_gemm` 实验记录

```current
task: 91
operator: tiny_k_gemm
batch: 6
validity: valid
platform: e8-3337be6(18750)valid TB1.973925; e8-14437ad(19380)valid 1.8200625; e9(20170)valid 1.885275; e10/e10b 本地筛选负例未提交
candidate_stage: e10b-screen-rejected
team_best_stage: e8-3337be6
team_best_speedup: 1.973925
sealed: yes
next: 待同版本 GCU 逐 shape 计时和 lowered IR，才考虑反转 dot 操作数；剩余 1 次额度不用于未过筛选门的 T91
updated: 2026-09-23
```

## 2026-09-20 S0/E1 首发记录

- 开发+验证+codex-review 门后 00:12-00:27 发射（详见
  `artifacts/competition/nx-s0-20260920/`）。

## 2026-09-20 E2 平台终态：双 vendor 中性，TB 保 1.953

- submission **18308** completed/valid，8/8，均值 1.951 ≈ TB 1.953（保 e1）。
  沐曦 1.8（warps8 无增益）、燧原 0.6→0.7（微）；B 2.8。距榜首 2.016 差
  3.3%，缺口在沐曦 1.8 vs 2.4 与 B 2.8 vs 3.0——无已知配方，关闭本轴。

## 2026-09-20 E7 平台终态：BLOCK_N/warps 阶梯到顶，TB 保 1.953

- submission **18345** completed/valid，8/8，均值 1.953 = TB（持平）。沐曦
  1.7（16 档无增益）、昆仑 1.1（warps8 无增益）、华为 0.8→0.9；BLOCK_N
  128 因 smem 149KB 不可发（K=256 tile 物理上限）。T91 全部已知轴关闭。

## 2026-09-21 夜 e8 候选就绪（午夜首发第 2 弹，Top1 冲刺）

- 结构（`3337be6b`）：metax vendor BLOCK_N 16→**64 + num_stages=1**——
  s0 时 BLOCK_N=64 需 86KB smem（默认 3 阶段）超沐曦 64KB 上限；
  单阶段把 tile 降到 ~40KB 物理可行，**榜首沐曦 2.3 vs 我方 1.8 是
  唯一有效缺口**（+0.5 芯分 ≈ +0.06 均值即够反超 2.032）。预注册门：
  沐曦 ≥2.1；均值 >1.953 换 TB，>2.032 冲 #1。
- 五元组：commit `3337be6b`；ZIP `e8-3337be6` SHA
  `1b0a69f1189478081b21ee253571dba938f701531773bf22c6dace9bc1ac5960`；
  回执 `top1-wave-20260921/tiny_k_gemm/`（四路径）。

## 2026-09-21 00:08 E8 平台终态：1.9739 新 TB（+0.021），沐曦 1.9 门未过

- submission completed/valid。逐芯：天数 2.73 / 沐曦 1.9（门 ≥2.1 未过，
  BLOCK_N64+单阶段仅 +0.1）/ 燧原 **0.67** / 海光 2.76 / 昆仑 1.09 /
  华为 0.83 / A 2.93 / B 2.88。A/B/海光已超榜首同芯。
- **榜首 2.032 → 2.2448**（EvokeAgent，天数 4.49 拉动）。Δ 归因：天数
  -1.76、燧原 -1.23 两洞合计 -0.37 均值 > 总差 0.27——**下一轴 = generic
  GCU 结构 + 燧原 vendor**。


## 2026-09-21 E8 候选就绪（燧原官方 gcu300 规则集，待 09-22 发射）

- 官方源码调研（FlagGems _enflame gcu300 codegen / FlagTree enflame
  backend）：max_grid_size=(12,1,1)（grid-stride 吸收超额）、
  enflame_heuristics_for_num_warps 钉 2、stride 需编译期互整除才走 DMA。
  本题 vendor 应用：grid 24→12+ num_warps=2 + 运行时 stride 参数 constexpr 化（DMA 判定编译期可证）。
- 五元组：commit `14437ad3`；ZIP
  `artifacts/competition/tiny_k_gemm/e8-14437ad/tiny_k_gemm.zip`
  SHA `14379b21bbf057d40c34581aaae829fabb0a9b67dd493ac3a955dfcdc5f69994`；成员 generic/enflame/kunlunxin/metax（generic 与非燧原 vendor 字节
  不变，账本明确列全）；回执 `day5prep-20260921/tiny_k_gemm/verification.json`
  SHA `8bb2c26447e6d7d4…`。
- 预注册门：燧原 0.67→≥1.34(×2)；其余七芯不动（vendor-only 单变量）。codex-review
  零发现（grid-stride 边界模拟无漏算）。

## 2026-09-22 e9 候选就绪（燧原 w natural-layout DMA 流形态，上膛待发射）

- 结构（`30386020`，enflame vendor 单变量）：launch 字节冻结 e8（12 CTA
  grid-stride / constexpr stride / num_warps=2 / num_stages=3），仅改 w 的
  访问形态——tile 按 w 自然 [BLOCK_N, K] 行主序加载（k 连续轴落在 tile
  末轴 = GCU DMA streaming 向量形态），寄存器内 `tl.trans` 进 dot，替代
  连续轴落在首轴的 stride 转置加载。测试新增非连续 stride / ragged-tail
  回归并纳入 `RELEASE_REQUIRED_TESTS`。
- 五元组：
  - source/verification commit `30386020`（回执 source_commit =
    verification_commit = `30386020e1c86347b4c23869141327b106b3dce8`，
    = 上膛时 HEAD）；
  - ZIP `artifacts/competition/tiny_k_gemm/e9-3038602/tiny_k_gemm.zip`
    （8868 字节）SHA
    `70656f2d75c785c70a0a067d5e135d3389bd3db98f647f5e8c860858268fe035`，
    与 e8 `14379b21…` 不同字节（平台元组去重键 zip_sha256，新候选
    确为新 ZIP）；
  - 成员逐一（zipfile 实读核对一致，与回执 manifest 哈希逐项相等，
    unzip -t/-l 无夹带）：`tiny_k_gemm.py`
    `0aa069c9b04adc0c…474cdeda`（generic，字节同 e8）、
    `tiny_k_gemm_enflame.py`
    `818b59ce61d01a76…695e12f7`（唯一改动成员）、
    `tiny_k_gemm_kunlunxin.py`
    `8563c40cda846de4…0caa28d`（同 e8）、
    `tiny_k_gemm_metax.py`
    `a55fd82f1f763159…e799fc3`（同 e8）；
  - 回执 `artifacts/competition/day5prep-20260921/tiny_k_gemm_e9_enflame_natural_layout-wf/verification.json`
    SHA `b26d668cddfcc3aa5c4589c14a552d7f597c10d1f3cc4a204fe16d034c055dc6`；
  - 远端 gpu（RTX 5070 Ti，torch 2.13.0+cu130 / triton 3.7.1）
    timeout 900 run exit 0：四源路径（generic/enflame/kunlunxin/metax）
    各 60 次真实 kernel launch、272 子用例 0 fail / 0 skip / 0 xfail、
    RELEASE_REQUIRED_TESTS 两方法（test_matrix /
    test_w_natural_layout_strides）在列通过、非空 bf16 shape 12 组
    （m∈{1,7,16,64,1000,4096}×K∈{128,256}）；三 vendor 路径为 NVIDIA
    代理执行，目标芯 unverified（交平台补齐）。
- 预注册门：燧原 e8 平台 1.18 → ≥1.5；其余七芯不动（vendor-only
  单变量）；均值 >1.9739 换 TB（取正文 e8-3337be6 平台记录值；
  彼时 CURRENT 块沿用 1.953@s0，已在下文依平台状态修正）。

## 2026-09-23 平台终态与 e10 结构筛选

- 实时平台状态：e8-3337be6 submission **18750** 8/8 有效，均分
  **1.973925**，是我方有效最佳；e8-14437ad submission **19380**
  8/8、1.8200625，燧原 1.1760 / 昆仑 0.1488；e9 submission **20170**
  8/8、1.885275，燧原 1.0870 / 昆仑 0.1501。后两次昆仑源码字节未变，
  相比 e8-3337be6 的 1.0891 出现低水位。e9 燧原优于团队最佳的
  0.6745，故保留当前 e9 源码；总分最佳仍为 e8-3337be6。
- 榜首 HAiWORLD 3.1755875、燧原 10.5105；第二名燧原 1.6158。
  若其他七芯固定为我方最佳 18750 的逐芯分数，超越榜首要求燧原
  **>10.2878**。10.5105 是相对其他队伍的离群读数，不能据此判定错误。
- e10 依 [SGLang CUDA tiny-K](https://github.com/sgl-project/sglang/blob/2d257677598134fbb091ecf5f023d1d678ee8c26/python/sglang/kernels/jit/csrc/gemm/tiny_gemm.cuh) 的“每列归约 K、逐行 M”工作划分，
  将 `tl.dot` 换成 `[BLOCK_N,K]` 权重一次加载和逐 M 行 `tl.sum`；
  `tl.static_range` 与运行时 `range` 两版各通过 generic/燧原代理路径
  2/2 正确性。回执分别为
  `artifacts/competition/tiny_k_gemm/e10-screen-20260923/verification.json`
  （SHA `4513eda5cf6418598f858224fe1a62531e876fa7c73a318b8c81bfbecf74693f`）
  与 `e10b-screen-20260923/verification.json`
  （SHA `895f1f75544ca98d472ed77546f2f16c368b3ea87a9160007fb05fde6ef03b22`）。
- RTX 5070 Ti 五组 AB/BA wrapper 计时：相对 e9，静态版在
  `(M,K,N)=(1,128,4096)/(7,128,4096)/(16,256,4096)/(16,256,1000)`
  的速度比为 **1.016/0.595/0.106/0.281**；运行时版
  **1.016/0.575/0.101/0.247**。两版 0 spill；主要风险是按 M
  串行归约并失去矩阵乘吞吐。筛选负例，**未制作 ZIP、未调用 codex-review、
  未提交平台**。e10 源码和测试工作区差异已撤销，当前燧原源码 SHA
  `818b59ce61d01a761ef740405f8cc1df6090f57914238f3a9c33a980695e12f7`
  与不可变 e9 ZIP 成员一致。
- 遇阻后使用 `codex-ask`（gpt-6-sol / max，退出码 0；输出 SHA
  `c75ddebfccb4be3e0c2f70453ca2dd33305d74d59e4a7b0b77e96c6030d8e91e`）。
  独立意见同意封存：二维同时归约所有 M 会形成巨大 FP32 中间张量且
  仍放弃 `tl.dot`。只在取得 GCU 逐 shape 时间和 lowered layout/dot IR
  后，考虑 `W[64,K] @ X[K,16]` 再转置存储，以较小 X 转置替代 W 转置；
  需先看到 GCU 大 shape 收益且 M=1、N=64 不显著倒退，才重启完整
  release → codex-review → preflight → 单次提交闭环。
