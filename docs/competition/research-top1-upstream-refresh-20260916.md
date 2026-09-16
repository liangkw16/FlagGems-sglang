# 第五批 Top1：vLLM / SGLang 上游源码复核（2026-09-16）

> 本文是盘中固定源码研究快照。后续T65 E10平台+67.16%，T64/T68/T66已完成一次八芯裁决，T75/T73/T65 E11筛选已关闭；最新榜差和终态见[最终报告](top1-opportunities-20260916.md)。

本报告只新增研究记录，不修改算子、不提交平台。榜差以 `data/top1-intel-20260916-refresh2.json` 的 **11:18:32 +08:00** 快照为准；不是重新抓取榜单。已读取竞赛 skill、今日 Top1 报告及相关完整题面、源码、测试和账本。下述速度预测均是待验证假设，非平台成绩。

## 结论与排序

1. **现有 T66 窄 N split-K 候选继续既定门禁**，本扫描不另开重复实现。
2. **现有 T68 cap24 path-split 候选继续单独筛选**。改动只有 e/h 路径在同一次 launch 中分别调度；不能混同历史 E6 两次 launch，也不能把 Hygon 整行版本直接覆盖到燧原。
3. **新增 T65：复用 SGLang 已合入的 `post_reorder_deepgemm` hidden 分块并行结构。** 当前本仓仍在每个 token program 内串行扫描 hidden；这是本次最明确的成熟上游结构差。优先做固定 BLOCK2048 的映射单变量筛选，保留昆仑已过线的 vendor。
4. T62 的 warp 内流式复用可作次级研究，暂无目标芯性能证据；不再试已失败的 BH、mask 或 warps 梯度。其他检索结果未满足“成熟、契约相符、未重复失败轴”三项，不为用掉额度而启动。

## 固定一手源码

通过 `gh api repos/<repo>/commits/main`、递归 Git tree、contents 和 PR API 读取；没有连接 GPU 做这些查询，没有下载或修改上游工作树。

| 仓库 | 本次固定 main SHA | HEAD 提交时间 UTC |
| --- | --- | --- |
| SGLang | `5f6dd44edc96779d4a15331637e26e73265ff6eb` | 2026-09-16 03:02:40 |
| vLLM | `c16bb6068f70878fb8a2f7c4d6cda95cd03a778b` | 2026-09-16 03:10:58 |

这两个 SHA 是查询时取得的 main，不代表之后 main 不会变化。PR 的开闭、merge 状态均读 API，未把 closed 等同 merged。

## T65：把 hidden 串行循环移到 grid.y

### 成熟实现及本仓差异

SGLang main 的 [`post_reorder_deepgemm_triton_kernel`:902–945](https://github.com/sgl-project/sglang/blob/5f6dd44edc96779d4a15331637e26e73265ff6eb/python/sglang/kernels/ops/moe/ep_moe_kernels.py#L902-L945) 已采用：

- `offset = BLOCK_SIZE * program_id(1) + arange(BLOCK_SIZE)`，hidden chunks 独立并行。
- token 维 `program_id(0)` 起步、`num_programs(0)` 步长，保留超大 token 数覆盖。
- `src2dst >= 0` 才 gather；标量路由转 int64；fp32 顺序累加 topk，最后写回。
- [wrapper:948–972](https://github.com/sgl-project/sglang/blob/5f6dd44edc96779d4a15331637e26e73265ff6eb/python/sglang/kernels/ops/moe/ep_moe_kernels.py#L948-L972) 调用同文件 `_get_launch_config_2d`。相邻 CUTLASS 路径也使用同构映射。

同 SHA 的 [专用 CUDA CI 测试](https://github.com/sgl-project/sglang/blob/5f6dd44edc96779d4a15331637e26e73265ff6eb/test/registered/kernels/ops/moe/test_post_reorder_deepgemm.py#L64-L84) 覆盖 tokens=1/8/128/1024/4096、hidden6144、drop=0/30%、scale=1/2；这证明上游有持久化回归，**不表示本报告执行过这些测试**。

本仓 E9 `aa06d46d4b278e585795997b4e3e6a406e909dbe` generic 仍是 token grid-stride 内嵌 `for start in tl.range(0, hidden, 2048)`。完整账本 E1–E9 试过路由向量化、标量恢复、BLOCK 梯度；未试 hidden 维 program 并行。E1 昆仑 one-hot + `tl.sum` 提取标量已导致编译器 abort，不得重新引入。

### 最小适配

只改 generic 的 hidden 映射：令 `gy=min(cdiv(hidden,2048),255)`、`gx=min(tokens,max(1,65535//gy))`，grid为`(gx,gy)`，保留token与hidden双轴步进。hidden≤2048是同工作量控制组；更大hidden是affected。**11:40发布审查订正**：只分别封顶x/y不足；T21 submission4274的grid(4096,28)在昇腾报coreDim114688>65535，须将总program数控制在65535内。原独立封顶候选仅有screening，已阻断发布并保留负向安全审计事实，不能将其1.32x用于新字节背书。

保留当前 BLOCK2048、slot 顺序、fp32 累加、`row * (w * scaling)` 顺序、所有 stride、输出新分配、空维度、topk=0 和全 -1 语义；昆仑 vendor 字节不动。不同时加入上游的资源探测、NUM_STAGES3、scale 后移或 Gluon。题面 reference 将权重先转换成 down dtype；新测试继续按题面 reference 对比，不能用上游 float32 权重 reference 替换。

预期受益是小 tokens / hidden 多块、当前只有少数 program 的输入。T/H 已大到带宽饱和时不应假定收益；多 program 还可能重复加载元数据并产生调度成本。海光、天数、沐曦、A/B 可做跨芯性能假设；华为与燧原占榜差较大，但目标芯 lowering 未验证。不能凭 NVIDIA proxy 推断其收益。

### Top1 差距及最小验证

11:18 我方 **16.10245**，榜首 **69.07705**，需要整体 **4.2898 倍**。逐芯主要差额为华为19.019→255.2878、海光36.0698→91.6252、A26.2802→66.7566。即只把华为追到当前榜首读数，均值也只有 **45.63605**，仍非 Top1。

隐藏形状分布未知。以 hidden7168、BLOCK2048 为例只增加4个独立块；即假设除昆仑外七芯**全部**提高4倍，均值也仅 **64.277425**。这不是算法的普遍上限，而是提醒：本最小映射候选有明确根因，却不应被描述为单步必胜。

建议首轮筛选：完整4个现有方法；追加 hidden2047/2048/2049/4095/4096/4097/7168、tokens1/3/8/32/256/1024、topk1/2/8/16；覆盖 fp16/bf16/fp32、权重fp32、scale1/0.0625/2.5、非连续 down/route/weight、全 -1、输入不变及新张量。以同一 frozen source 做5轮 wrapper-inclusive AB/BA，affected 几何均值≥1.15、每轮≥1.10、无shape回退>5%、control≥0.95、无spill才晋级；再进入 commit 字节 release。阈值为本报告建议，尚未运行或注册成平台结果。

### 两个容易误用的旁证

[SGLang PR #22426](https://github.com/sgl-project/sglang/pull/22426) 当前 **open、未合并**，head=`8ec0c4987bacdf8ad8d38006707add8c5acda207`。它把旧 deepep post-reorder 改成2D Gluon、predicated零填充、bf16向量FMA；作者报告H200小batch大收益，但这是作者测量且同时改了多项。这里采用已合入的 Triton deepgemm 结构作为主证，不照搬 Gluon、32-lane布局或bf16累加。若下一轮才研究分支消除，须单独处理无效路由和权重NaN/Inf的契约，不把 `0 * weight` 自动视为无害。

vLLM main 的 [`finalizeMoeRoutingKernel`:93–165](https://github.com/vllm-project/vllm/blob/c16bb6068f70878fb8a2f7c4d6cda95cd03a778b/csrc/libtorch_stable/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.inl#L93-L165) 采用128-bit读写、fp32累加，但仍是一block一row、串行topk；CUDA/CUTLASS实现也不满足赛题必须Triton/TLE的直接复用条件，故不当作另一条成熟跨芯方案。

## T68：已有候选继续，最新同名 vLLM 不能照搬

我方7.08945833，榜首10.40585833。仅把燧原1.56226667追至榜首21.41346667，均值为9.57085833；单靠燧原夺第一需超过28.09346667。21.41可能含参考耗时窗口因素，无法由排名反推出对方源码。

冻结 artifact `artifacts/competition/t68-path-split-cap24-screening-20260916/` 从 TB E4 `220aa32d18a1c3a4aca829b09b79e2e906471e43` 改成 `(min(tokens,24),2)`；每program处理单path，保留cap24、row-stride、CHUNK4096、两次读取、warps8/stages1。AST已证明撤回path分支及grid.y后与E4一致。候选并非Hygon逐字节复制；原始Hygon `(tokens,2)` 探针因大token grid风险已取消且未运行。

- candidate SHA256 `d5f8a4995d5cbb28ab385f169051be9e6bac6e0c8c27dadfb60e58e0ce77e515`
- test SHA256 `f9c02d2f392fdf3cfb99c16edbf93d5a4808973c77247f8af916a0ac314267f9`
- 7个必需方法：原5方法 + input/weight列stride + tokens65535/65536/65537；24桶5轮，tokens≤24为primary，>24为secondary，后者不是不变control。

最新 vLLM [同名 fused_eh_norm](https://github.com/vllm-project/vllm/blob/c16bb6068f70878fb8a2f7c4d6cda95cd03a778b/vllm/models/glm5next/nvidia/ops/fused_eh_norm.py#L18-L76) 增加 positions 输入，并在position0清零嵌入；仍顺序处理e/h。这与赛题签名/公式不同，没有提供可直接替换的性能突破。

## T62：warp流式复用只有次级价值

SGLang [CUDA concat_mla.cuh:58–116](https://github.com/sgl-project/sglang/blob/5f6dd44edc96779d4a15331637e26e73265ff6eb/python/sglang/kernels/jit/csrc/elementwise/concat_mla.cuh#L58-L116) 每warp持有一份rope，按16 heads流式读写NoPE，使用8B/thread NoPE、4B/thread RoPE向量访存。当前generic一次构建 `[BH,BN]` tile而非按head流式，寄存器驻留形态确有差异；“rope复用”本身已经实现，不能算新机会。

若后续能看到寄存器/访存宽度瓶颈，可独立研究保持BH16、把head搬运改成静态循环的Triton实现。不得搬NVIDIA asm/cache hint/32warps配置，不能缩小stride或尾部支持；昆仑保持两段独立2幂块形态。需要生产128/128/64及heads15/16/17、非2幂维度、stride、纯拷贝精确测试。当前没有目标芯证据，不排在T65/T68前，不重开已证伪BH4、去mask、warps4轴。

## 其余源码与PR筛除

| 对应题 | 本次一手检查 | 结论 |
| --- | --- | --- |
| T59 | SGLang `ops/kvcache/trtllm_mha_page_table.py`，按request/page-block二维调度、设备seq_len提前退出 | 其wrapper原地写预分配缓冲，赛题返回新张量需适配；不足以重开已有row-pack/地址/整除失败族 |
| T61/T64/T69 | vLLM [PR #30280](https://github.com/vllm-project/vllm/pull/30280)，head `976bf1d1408d2f3fe8188e3d5278b7e0d9326813` | closed但**未merged**；no-permute通过每个route分配独立padded block改变排序输出协议，不能替换固定src2dst/dispatch契约 |
| T64 | SGLang main `deepep_permute_triton_kernel` / `pre_reorder_for_cutlass_moe` | 前者仍token内串行hidden；后者hidden拆分本仓T64已有。去clone需逆映射，历史destination-gather代理0.35x，不能用新名字复投 |
| T66 | SGLang `fused_a_gemm.py` / JIT `dsv3_fused_a_gemm.cuh`；vLLM同族 | 成熟CUDA依赖SM90 MMA、cp.async、mbarrier及K分块，不能直接复制为跨芯Triton；当前split-K工作已有负责人，不另开参数副本 |
| T67 | SGLang `ops/moe/fill_padded_rows.py` | 每row读取设备n_valid并条件填充；上游原地写，赛题clone返回。不是消除clone的合法新证据 |
| T72 | SGLang `group_norm_silu_twopass_triton.py` | 名为two-pass但实际partial/finalize/apply三次launch；主要优势来自channels-last避免布局往返。使用E[x²]−mean²且限定C为2幂≤2048，无法直接替换本题大均值中心化方差保障。小group驻留E9平台已失败，不重开 |
| T60/T70 | 当前未发现适配其既有编译/运行失败根因的成熟vLLM/SGLang修复 | 仍需对应vendor根因或目标执行证据；不把普通上游重构当作重掷理由 |
| T63/T71/T73/T74/T75 | 搜索同名/相邻实现与PR后，无本次已核实、契约相符且可直接晋级的新结构 | 维持账本封闭的宽度、warps、no-loop、索引/prefix轴；不宣称遍历了所有历史PR |

11:18 T72榜首已升至 **4.78075**（华为10.82666667），不是旧报告3.944625。E9平台8/8但华为1.1215、均值2.48429167，低于TB2.66266667；NVIDIA小group收益没有迁移，不能用新榜差触发同轴重试。

## 执行边界与后续证据

本报告新增项T65尚未实现、未GPU、未提交。T68已单独获得主代理串行GPU令牌，按冻结plan做screening；其回执及性能应以artifact原始JSON为准，再更新实验账本，不能用本报告代替release。

KernelGen最新schema保存在 `artifacts/competition/top1-20260916/kernelgen-tools-refresh2.json`，无固定候选源码执行工具；当前无登记的独立目标芯主机。重新生成代码不能记为同源验证。所有未取得目标运行证据的迁移继续标记 **target-runtime-unverified**。
