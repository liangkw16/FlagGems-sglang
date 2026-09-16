# Task 74 `seqlens_expand` 实验记录

```current
task: 74
operator: seqlens_expand
batch: 5
validity: valid
platform: completed(e4,8/8,20.1778x 新TB;海光28.78门兑现,沐曦+2.09)
candidate_stage: e12-hint-safe-ready
team_best_stage: e4
team_best_speedup: 20.177775
sealed: no
next: E12已修复全部fallback低报hint漏写，16方法release与新性能筛选通过，等待实时preflight单次提交
updated: 2026-09-16
```

## 契约与实现（S0）

- 完整题面：[Task 74](../tasks/batch-5/74-seqlens_expand.md)（2026-09-12 晚新增五题之一）。
- attention pad.py kernel; per-request program, clamped base+arange; cumsum offsets in wrapper。
- 核心计算 Triton，无 fallback；八芯 0.1x；截止 2026-09-17 19:59:59。

## 不可变身份

- source / verification commit：`e6b450fd4beb001255079ade929d3b2de51796d6`（五题同批提交）。
- source SHA-256：`884da15e0f885cc01f31ee53198fbd61cbd045d4c0b71c3ee0106e6644c9de35`。
- test SHA-256：`83da626a4e5922220d9283748353bea7d7e1286a7cc00fc5b0a49561402152ce`。
- ZIP：`artifacts/competition/seqlens_expand/s0-e6b450f/seqlens_expand.zip`，SHA-256 `2a6ecb1cea7019b6154ef9d78aa5969797d0b4b5d60a6598cd77ee9d111fc9bc`（单成员 `seqlens_expand.py`）。
- release 回执：`artifacts/competition/batch5-new5-validate-20260913/seqlens_expand/verification.json`，
  SHA-256 `3a84097f235d313a00adc7e357ab7b1fb90c6d87f63cf7c26542acb0522356c6`；日志 SHA-256 `a022f73c47816e01f24df117e3e2e09ae12545e4915c7ad8373192b0017fcf3d`。

## 验证状态

- screening 多轮门禁拦下并修复的缺陷已记录于提交说明；
  最终 release：0 失败/错误/skip，非空 kernel launch，NVIDIA 代理范围。
- 所有八芯目标 `target-runtime-unverified`；裁决权在平台。

## 2026-09-13 E1：请求维 tile（候选就绪后提交）

- S0 整请求单 program（BLOCK=next_pow2(max_q_len)）在请求少而 q_len 大
  的 shape 欠填充（对榜首差距全芯均匀 +5~+48）。E1：kv_indices splits
  同款 2D tile——固定 1024-lane，(request, tile) 每 program。
- source commit：`114be7818888c9d8bfd1b36c7bb42845bc2c7fd4`；ZIP `e1-114be78`，
  SHA-256 `85ae694a999dca3b9657d0815391102a4a69cf0e47f859f1ba7c3989ee0dc884`；
  release 回执 SHA-256 `38662ae336ddab30deca33cca2fc719fd3d2b475601aee26c3818404ae1801a6`；
  3 方法 0 失败。
- submission 13796；裁决点=全芯均匀差距是否收窄。

## 2026-09-13 E1 终态与 E2 重掷（submission 13804）

- E1（13796）7/8:七芯全过且 tile 正收益（天数 27.0782 +6% / 沐曦
  8.2582 +8% / 其余持平），昆仑 exec 0ms 间歇崩溃（s0 时昆仑正常判过
  1.8150 ⇒ 按提交闪断）。
- E2：tile 维 255 封顶 + grid-stride（燧原 grid.y 硬限的真实加固）=
  新 ZIP 身份重掷。
- source commit：`40f82daea4780bdb9e99b4474e5546748ef7c117`；ZIP `e2-40f82da`，
  SHA-256 `7c57ef034849d4c7b39d0beb19bf29bd84085f9da5209c6f6be613c5730db008`；
  release 回执 SHA-256 `941cc373a8481dc0d5e8e9ac4e75ddd8231f6bef00035e84c7ccbe341b1df9df`。
- submission 13804；裁决点=昆仑窗口 + 七芯 tile 增益兑现（均值预期
  ≈11.2）。

## 2026-09-13 E2 平台终态：8/8 valid 10.5267x（低于 TB，轴收口）

- 昆仑窗口重掷成功（exec 8786ms 真实执行，1.5952 过线）——**8/8 valid
  但均值 10.5267 < s0 TB 10.6444**：e1 的天数/沐曦 tile 增益被本窗口
  沐曦（8.26→6.82）/海光（14.66→13.82）回落吃掉。
- 处置：team best 保留 s0；tile 轴收益确认存在但量级 < 窗口方差，
  不再追加同轴发射。T74 保持 valid 在榜。

## 2026-09-14 E3 候选就绪：前缀和进 kernel，3 launch → 1（待发射）

- 发射前用逐芯榜单快照（`data/batch5-intel-20260914-afternoon.json`
  SHA-256 `fa5706cf094eb0b6...`，observed 14:48）对账修正旧记录：
  榜首 c2flow 26.356 vs 我方 10.644 并非"+5~48%"，而是**除昆仑外
  七芯均匀 1.9~3.3x 落后**（天数 25.5/83.1、沐曦 7.6/16.1、燧原
  6.8/18.3、海光 14.6/30.4、华为 5.8/12.9、A 12.5/28.3、B 10.6/20.1；
  昆仑我方 1.815 反超 1.753）。昆仑例外 + 全芯等比 ⇒ 与"zeros+cumsum+
  主核 3 次 launch vs 1 次"的 launch 开销假说相容（E1 已证 tile 轴
  无关）；并行度假说与"昆仑不例外"矛盾，排除。
- 载体 = 独占前缀和进 kernel（1024-lane 定宽 masked 块累加，无按
  N 变化的 constexpr，单一编译产物），弃 torch.zeros+torch.cumsum，
  wrapper 只剩一次 launch。grid/掩码/clamp 语义与 TB s0 字节一致。
- source / verification commit：`ea248247…`；ZIP `e3-ea24824`，单成员，
  SHA-256 `2f2edf06c5a4680a4a188f9cc7e5f39c8a9024914751e2b7b9f65e314fab78df`。
- release 回执 `batch5-submit-20260914/seqlens_expand/verification.json`
  SHA-256 `0f5c774de2e003ae161913431eb732555993450cb177f080c4416a495880e337`
  （3 tests 0F0E0S，6 次真实 launch，无未执行源）。
- 预注册晋级门：**均值 > TB s0 10.644**；单芯波动只在窗口方差内判读。

## 2026-09-14 E3 平台终态：8/8 VALID 19.3101x 新 TB（submission 14535，+81%）

- **预注册门大过**：均值 19.3101 >> TB s0 10.6444。前缀进 kernel
  （3 launch→1）的 launch 开销假说兑现——七芯全部大涨：
  天数 25.52→60.00（2.35x）/ 沐曦 7.62→9.78 / 燧原 6.78→9.20 /
  海光 14.57→23.71 / 华为 5.75→7.68 / A 12.50→24.67（1.97x）/
  B 10.59→17.80；昆仑 1.82→1.64（窗口内小落，唯一例外与"昆仑
  launch 占比小"的预判一致）。
- 榜单位置：越 Nectar 19.306，预计 #5；榜首 c2flow 26.356，差 7.05。
- 残余轴：flat-grid 并行度（请求少时的 (n,tiles) 欠填充）尚未动，
  留作下一单变量候选；本发先收盘守 TB。
- 额度：发后 21/30（watch 实测待 T70 e3 后刷新）。

## 2026-09-15 E4 候选就绪：es 漏乘修复 + 大 batch 两段式（验证通道中断）

- commit `45662b8c`。两个成分：
  1. **正确性修复（载体）**：融合 kernel 前缀累积 `extend + ridx` 漏乘
     stride，与非连续 extend 视图交互时基址读错（当前行读 `pid*es` 一直
     正确；隐藏评测用连续输入故在榜字节未败）。补回归
     `test_strided_inputs`（stride-2 视图，5 元素走融合路径 + 1500 元素
     走 scan 路径）。
  2. **性能轴**：n>1024 走 `_seqlens_prefix`（单 program 分块
     tl.cumsum+标量 carry，平台已证原语）+ `_seqlens_expand_p`
     （base 改单标量 load），消除每 program O(pid) 前缀重算；
     n≤1024 路径与 e3 字节级行为一致。
- 新增 `test_large_batch_two_path`（n=2050 混合 qos/kvs）；
  RELEASE_REQUIRED_TESTS 同步扩充。
- ZIP：`artifacts/competition/seqlens_expand/e4-45662b8/`，5325 bytes，
  仅 generic `seqlens_expand.py`（`c6612828…`）。
- ZIP SHA-256：`c0e2356ced2230411573f2b0ccfbf3812ce75ed8ee9379835518e7f9fd3d47c9`。
- 预注册门：天数 ≥66 或 海光 ≥27；8/8 且均值 ≥19.0 才替换 TB e3 19.31。
- 阻塞：GPU 通道中断，release 回执待补；未 preflight、未耗额度。

## 2026-09-15 16:30 E4 验证回执就绪（含测试修复重跑）

- 首轮 release 抓出两个**测试代码 bug**（kernel 未证伪）：strided 构造
  误用 `stack(dim=0).flatten()`（拼接而非交错，-1 填充泄漏进断言）与
  大 batch 用例 list+Tensor 相加；修复 commit `82b1eacd`（kernel 字节
  不动，ZIP e4-45662b8 仍有效）。
- 重跑回执：`artifacts/competition/batch5-verify-20260915/seqlens_expand/`
  （exit 0，generic 11 次 launch，0 skip，5 方法含新增两回归）。
  verification commit=`82b1eacd`。门不变：天数 ≥66 或 海光 ≥27。

## 2026-09-15 16:55 E4 平台终态：8/8 VALID 20.1778x 新 TB（海光门兑现 +21%）

- submission（e4，45662b8c）：天数 59.5166 / 沐曦 **11.8654（+2.09）** /
  燧原 9.2032 / 海光 **28.7834（+5.07，门 ≥27 兑现）** / 昆仑 1.6826 /
  华为 7.6296 / A **26.0222（+1.35）** / B 16.7192（-1.08）。
- ΔS=+6.94/8：**均值 19.3101 → 20.1778，新 team best（e3→e4）**。
  大 batch 两段式在沐曦/海光/A 三芯同向 +21% 量级——隐藏评测确含
  大 n shape，scan 路径真实兑现；天数 59.5 未及 ≥66 门（-0.5 噪声级，
  该芯未从两段式获益，后续 flat 并行度轴仍开放）。
- es 漏乘修复随载体上线（非连续输入回归已在 suite 内）。

## 2026-09-15 17:10 E5 平台终态：8/8 VALID 20.052x ≈ TB（flat 映射零增益）

- submission（e5，bbef5e08，tile 优先 flat 工作映射）。天数 61.17
  （e4 59.52，+1.7 但门 ≥66 未及）；沐曦 10.47（-1.4）/ 海光 27.12
  （-1.7）/ 其余噪声带。ΔS≈-0.13：**TB 保持 e4 20.178**。
- 判定：相邻 program 同 request 连续 tile 的局部性不是天数缺口形态；
  天数轴剩余=qo_len constexpr 静态化（Codex 候选 B，明日）。

## 2026-09-15 21:30 E6 平台终态：tiles constexpr 阴性关闭

- 19.797 < TB 20.178：天数 62.3（未及 66 门）、沐曦 -1.7、海光 -4.1。
  静态 work 解码不是天数缺口形态；TB 保持 e4。天数轴剩余=255 上限
  取消（需 flat 已过门，条件不满足，关闭）。

## 2026-09-16 E7 小batch前缀候选：代理未过门，不发射

- 从TB e4（45662b8c）单变量改写n≤1024前缀：动态循环改单次masked reduce，BLOCK_N按next_power_of_2(n)裁剪；大n scan/expand函数保持TB字节。源码与完整6项边界回归已留在隔离screening包。
- RTX5070Ti：6/6、0失败/skip，16桶×5轮AB/BA，零spill；寄存器40（scan48），共享内存≤32B。受影响桶geomean **1.01099<1.05**，五轮aggregate1.00748–1.01465，controls0.99604–1.00214。只有n1024/qmax1025稳定+7.70%，不能当全域突破。
- **未过预注册门，不生成release/平台intent**；主树恢复本轮前字节。原始源/测试/脚本/manifest/verification/benchmark/PID在 `artifacts/competition/t74e7-screening-20260916/`。
- 本轮性能JSON SHA-256 `3a30a7d3dfc7b8c957fb4296ebeac3ca37f14520f40bf3e1b4df0e002462269d`。

## 2026-09-16 E8 空 tile 提前退出：机制成立，筛选 NO-GO

- 固定真实TB E4 `45662b8c403778e4b93b96ec90b4ca3306c6a03a`，generic SHA `c6612828e9344df00fbb49ed50bbbf6be7a5c18a8b099128ee622ac786a32eda`；当前主树E6不是TB，本轮未修改主树。SGLang固定 `5f6dd44edc96779d4a15331637e26e73265ff6eb` 的 `python/sglang/kernels/ops/attention/pad.py` 已在inactive位置提前return，再读取cumsum，作为成熟机制来源。
- 单变量候选仅在small-n kernel前增加带stride的qo读取与 `pid.y*1024>=qo` 退出；其余源码逆变换逐字节等于E4，大n路径冻结。候选 SHA `6eb944844ca9bf2ed1fd7683d85eed86b208e773e5b84080ba9fc153709ea24d`；tests `d2fd334454d20cdb83b15f6c752b499dbf7693b7fa87dc3f1915f65ee5496d33`，plan `894e5f78db7cbd7b35635471c934a66214cb62bd0cbdb3fdf20394dac7ca8cc9`。
- baseline与candidate同8方法全部通过；candidate 28入口/29实际launch，0失败/skip。保留原6方法，新增1024/2048空tile、stride2/storage_offset及qmax261119/261120/261121/522241，验证grid-stride后续tile255/510完整。回执 SHA `8a5d17fc2531337f8809aa7fa3ba9b267e4818ae7bcac944053c8f16c8894afa`，日志 `01c6d91f1b43974cf82007238c8a9ac3b426fd0833aa121e6b058b792a6bcbf4`。
- n1024/qmax1025及n4/qmax522241的IR均证实guard支配prefix循环及load，inactive直接ret；两种runtime shape共用同一编译字节。寄存器40→38，0spill；IR报告 SHA `aad9456c47d2affbe27cf33d2b6a7158baf2be8ae572574955c0065136cd1f3d`，人工判决绑定 `d7e5685825b2b17145bfe306f7b549b0b8915add60a442131047f7b0b003ed9b`。
- 原12主桶+4大n对照+8不均匀/全填充附加桶，5轮AB/BA共120对，未删样本。primary GM **1.0302765483<1.05**；五轮 **1.03555/1.00712/1.04273/1.00621/1.03737** 未全部≥1.03；最差主桶0.99540、附加0.96316，对照0.99643–1.00516，零spill。附加n1024/q8193高度不均匀桶1.70161不能替代整体门。**关闭，不生成ZIP、intent或平台提交。** 性能JSON SHA `61354f61bee9af336bb209e3491fdbe63d87407439e11ac3bff16822cba138e5`，日志 `fc158cd8c706320106752ad7ff8b468fbf93a1eaa5330cc7c8dbe243bcc47e2d`。
- 新审查发现E4既有发布隐患：grid `(n,min(cdiv(qmax,1024),255))` 没有总数上限；258×255=65790、1024×65=66560超过同族已证实的Ascend65535。提前退出不减少launch coreDim，本轮无混合修复，后续候选须另改并重新验证，不能拿代理数学通过宣称目标launch安全。
- 产物 `artifacts/competition/t74-empty-tile-screening-20260916/`，含完整源/测试/IR/原始CSV/hash。首次目录 `/tmp/flagos-t74-empty.rcLdnj`、PID389619在附属static文件哈希变化时退出1，尚未执行GPU，原包保存在precheck-attempt1；冻结后新目录 `/tmp/flagos-t74-empty.XLWzAB`，correctness/IR PID389639上限400秒、计时PID389752上限330秒，均EXIT0、作业结束、前后compute列表为空。RTX5070Ti/Python3.12.13/Torch2.13.0+cu130/Triton3.7.1；仅NVIDIA代理结论。

## 2026-09-16 E9 输出位置二分：完整筛选 NO-GO

- 14:47实时榜单：我方第6、E4 **20.177775**，榜首26.355675，仍需+30.62%。总均值差6.1779中天数贡献2.95345、燧原1.133175、华为0.654425；这些是榜单分数差，不能反推设备耗时。快照 `artifacts/competition/t74-output-search-screening-20260916/platform-start.json`，SHA `b021e7dedb86ee3f2f1bd2c5b415efb69f9294fb68f2295a36d53e8d67c1dfdb`。
- 从真实TB E4 `45662b8c403778e4b93b96ec90b4ca3306c6a03a`（generic `c6612828e9344df00fbb49ed50bbbf6be7a5c18a8b099128ee622ac786a32eda`）派生。普通大N保留E4前缀核，然后按256连续输出位置分块，对exclusive prefix作right upper_bound定位请求；重复prefix正确跳过空请求。来源为 [FlagGems固定searchsorted](https://github.com/flagos-ai/FlagGems/blob/2cbcecbf0d2dcb24f0686b95b2683bb2516c072e/src/flag_gems/ops/searchsorted.py#L57-L96)。E4普通smallN/prefix两个函数逐字节不变；未重试E5/E6映射或E8提前退出。
- 独立安全改动把旧2D grid总数约束为65535，并为fallback增加row/tile grid-stride。总长、请求计数、stride乘法或cdiv加法超int32安全域时，使用独立int64 prefix/地址，结果仍int32 wrap后clamp。它们不影响60个性能桶的工作划分。超过8GB的实际输出未分配：宽前缀用 `[INTMAX,1,1]` 与跨1024的少量输入实跑，极端wrapper路由用25项metadata捕获；目标芯宽scratch/lowering仍未验证。代码仅在隔离candidate中，主树E6未被替换。
- 首次 `/tmp/flagos-t74-output.VM9gPf`、PID390500：12方法中4个错误，根因为Triton将total_len=1特化成Python常量后不能调用`.to`。未开始IR/性能。原bundle、回执和完整日志保存在`attempt1/`；receipt SHA `7cf8125fbc84b2c86973ea6f3bb2767cdb7788395c6a6acf77223d9c72f46a12`，log `fac0fa70cd1584603d5c8902a9a6d687b595729676ff476230d533810f549db1`。仅把3个新kernel的4处metadata cast改成`tl.cast`，保留失败用例并追加wide n1/q1回归后，重新冻结全部身份。
- 最终source SHA **`c0760c43d9c158e3dc390a34c1a3eb5eda21f2a172855aea46d2cfe4274f7083`**，test **`8510e06c8102d210a27881241c615514e5bfcb5873229c5d20b67695469229b2`**；plan `fda50aca03ab7f64d9030d3de07746e23e4d9c5e628cf68844a8b87a3207f0eb`。Black79/isort80/flake8/py_compile通过；原5测试AST保留。**12/12，0失败/错误/skip，74入口记录、73真实launch**；包含N1025/2048/4096/65535/65536/65537、单输出、256±1、stride0/2/3和storage offset、整数极值、双维尾步进。baseline只运行原5方法与全部性能形状，不宣称通过新安全回归。
- receipt `588a74ca20c58d2de2b05ad153f31fc331da237cc9b6d071c044b767ae0c8211`，verification.log `2e7efff4c7a84efc513b99e6c8e9afab38e6e8992901f09a27dedb2e10af2d5a`。q8/q8193编译产物确认输出一维grid-stride、13轮masked upper_bound及连续store；expand寄存器40→32、0spill，prefix40regs/32Bshared不变。probe SHA `222df13ce84cbe4a009f2e04f586bfd29e1b001cd1835cff4203c7b8544c4893`，root逐项IR裁决 `681b9cf537cedaada1af4b005911b5a74dca69bed6affdf3249d0c067c567a4a`，之后才开始计时。缓存的baseline asm source-location可指向历史目录，实际baseline.py由双端SHA绑定。
- **60桶×5轮、300对原始样本完整结束**，未删旧24桶；40个所有largeN桶为primary，20smallN为control。primary GM **0.9051021966<1.05**，五轮0.905234/0.904805/0.906419/0.906296/0.905326均<1.03；最差balanced N8192/qcap8193 **0.3516096084<0.95**。controls0.976434–1.027871在门内，200条资源记录均0spill。
- 同total三分布追加矩阵中，qcap1/8/1025/8193各9桶GM依次1.08577/1.18338/0.95406/0.57282；短输出最高1.50024不能抵消长输出退化。逐输出二分增加的访存/比较随输出长度增长，与观察方向相容，但没有profile证明其是唯一瓶颈。**按原门关闭，不挑域、不改阈值、不生成release/ZIP/intent或提交。**
- 性能JSON SHA `95432659a8cd1a7009ec7f438b3a2be38e4e475e1ef43886390dc90abc06ef85`，log `c1a2d7e5ee9e8f29d9c8b7fd202eb93c2a1f3e211a12b4e647ed62d7d28ae912`，CSV `881fe8fdee1998d04ce97d9a1f25bb0e0d6951c18da3007ecdfead5bee5a1387`。最终远端`/tmp/flagos-t74-output.9Splfl`，correctness/IR PID390584上限400秒、benchmark PID390697上限630秒，均EXIT0；前后GPU无其他compute进程，已释放。RTX5070Ti/driver610.57.04/Python3.12.13/Torch2.13.0+cu130/Triton3.7.1。全部源/输入哈希/IR/日志在上述artifact目录；仅NVIDIA代理证据，未消费平台额度。

## 2026-09-16 E10 块级 merge-path：完整筛选未过门，不晋级

- 本轮只读榜单快照时间 **16:23:18 CST**：我方第 **8**，E4
  **20.177775**；榜首 EvokeAgent **26.821525**，均分差 **6.64375**
  （需 +32.93%）。天数与燧原分别贡献 3.4805、1.703175 的均分差。
  快照为 `artifacts/competition/contract-fixes-20260916/platform-t74.json`，
  SHA `b05be2a52a2badd2c71f2e24fa19e02312d119b85067dc207d461385f2bf92e8`。
  分数差只用于确定研究重点，不能反推隐藏 shape、耗时或瓶颈。
- 精确基线仍是有效 TB E4
  `45662b8c403778e4b93b96ec90b4ca3306c6a03a`，已核对 Git blob 与原 E4
  ZIP 成员。隔离候选将 exclusive prefix 与隐式输出位置稳定合并，
  每个 256 项对角线块做两次标量 co-rank，块内最多载入 256 个请求前缀；
  0/1 边界直接展开，多边界才做 9 轮局部 gather 搜索。总任务数为
  `ceil((N+total_len)/256)`，封顶 65535 后 grid-stride；重复前缀保持
  right-bound 语义。普通小 N 核和大 N prefix 保留 E4，独立继承 E9 的
  grid/宽地址安全路径，结果仍先 int32 wrap 再 clamp。未分配超过 8GB
  输出，宽路径实际覆盖仍限于既有 metadata 与小规模执行回归。
- 所有产物位于 `artifacts/competition/t74-block-expand-screening-20260916/`。
  本轮未修改主树算子或正式测试，主树 E6 未被候选覆盖；候选由以下文件
  SHA 绑定，不能冒称为已发布 source commit 或正式 release。

| 输入身份 | SHA-256 |
| --- | --- |
| `baseline.py` | `c6612828e9344df00fbb49ed50bbbf6be7a5c18a8b099128ee622ac786a32eda` |
| `candidate.py` | `5850ace3eb04838099b03e4c9b576673a6e7f14ca718381419651c5870cea517` |
| `test_candidate.py` | `561f735f2ef235cb10547416954caf5cd43933a850abea11e7366ff7d9620d54` |
| `screen.py` | `1ced8094b3b83400ab68f12187cd0430f6d1b670c85d6f4c1ce667b363e6c494` |
| `plan.json` | `bacca18966cb83a07eb8c4bfc6a0d4ea00e5b1e2aa964b378b45f439b0158dfd` |
| `fixture_basis.py` | `5f9382e0416f0ba97196f35578915c6129b06f479bfc15df496bf114bc9b79fb` |

- **首轮失败完整保留**：`attempt1/` 对应远端
  `/tmp/flagos-t74-block-merge.ePfPqk`、PID **392777**。baseline 5 法、
  candidate 12 法均通过，short probe 已运行；long probe 的原始
  N4096/q8193 fixture 超过未改变的 8,389,632 元素上限，在生成 fixture
  时断言退出，`precheck.exit=1`，未开始长 probe 或性能计时。这是探针
  构造错误，不是候选数值或性能裁决。
  原计划 SHA `4dd65d324d622342a3081f842282211f8e544c78cfc27cdd4a7f94b17246d388`；
  原 correctness 回执 `43ab3e0c0bee1c2eab19183cac1263275d398746e57075254c2fa425ac048bea`；
  原失败 probe log `c4a3b1e7faf6d231d19800f6a57d64af9bd36b22e6cfbb4bd2c993f182c0bc85`。
  修正仅让 long probe 精确复用已登记的
  `concentrated-n4096-qcap8193`（total **4,194,816**），并在 check 阶段
  校验两项 probe；candidate、测试及完整 60 个性能桶不变。旧回执只
  证明旧 harness，新计划重新冻结并完整重跑。
- **最终正确性**：baseline 原 **5/5**、candidate 完整 **12/12**，
  均 0 failure/error/skip。保留原五法 AST 与 E9 全部 12 法，涵盖重复
  前缀/零 q、长零串、256±1、N65535±1、非连续与 stride0、整数回绕、
  尾部 grid-stride 和宽前缀；全部性能桶另做精确整数校验。Black/isort/
  flake8/py_compile 通过；CPU 模型 87,400 例只属算法模型证据。
  `correctness.json` 与 `probe.json` 明确是 **screening 回执**，不是
  官方或项目正式 release 回执，NVIDIA 通过不代表其他芯片通过。
- **IR 判别先于计时**：root 人工裁决绑定 source、baseline、probe SHA。
  N4096 的两条标量 co-rank 链各 13 轮，全局 midpoint load 为标量；
  多边界路径一次载入 256 项局部前缀，再做固定 9 轮 gather。对应 PTX
  每轮重新写共享内存并使用两个 `bar.sync`，此分支共 **18 个 barrier**。
  无输出块跳过 gather/store，0/1 边界分支绕过 gather；长 q 的该分支
  仍支付每块两次 co-rank。merge 为 **38 registers、0 spill、1024 B
  shared**；prefix 为 40 registers、0 spill、32 B shared，与 baseline
  汇编相同。两项 probe 的 32 个汇编文件 SHA 已逐项复核；短 probe
  缓存中的历史 debug 路径不影响已绑定的实际源字节。
- **完整 60 桶 × 5 轮，300 对原始样本**，40 个大 N 主桶、20 个小 N
  对照，未删桶或改阈值。完整 wrapper 计入分配与 prefix，warmup20ms/
  rep50ms，按 AB/BA/AB/BA/AB 采样。主桶中位比值的算术均值
  **0.8090173429862597**，GM **0.7224616817620629**；五轮算术均值
  **0.807958 / 0.810288 / 0.810306 / 0.806205 / 0.809824**。
  均未达到预注册的总体 ≥1.03、每轮 ≥1.01。最差主桶
  `balanced-n8192-qcap8193` 为 **0.2095481045**；对照中位比值
  **0.975941–1.024399**，均在 [0.97,1.03] 内，200 条资源记录全部
  0 spill。算术均值、GM、300 行 CSV 和逐轮样本已独立重算对账。
- 同 total 的 qcap1/8/1025/8193 四组各 9 桶，算术均值依次
  **1.06271 / 1.15462 / 0.69085 / 0.36438**。短输出局部收益未能抵消
  长输出退化；额外 co-rank、局部 gather 的共享存储/同步与测量方向
  相容，但未做 profile，不能断言它们是唯一瓶颈。**本候选不晋级，
  不生成 ZIP、平台 intent 或正式提交，不消费额度**；保留 E4 TB。
  本结论只否决已测候选，不永久关闭整题，也不把代理均值当平台分数。

| 最终证据 | SHA-256 |
| --- | --- |
| `correctness.json` | `baaa57d6ae02286a0153fac8084ac915b3e559bd2c4eeba0e29ce8b65fd5f877` |
| `correctness.log` | `d0810fc4d9b9036856d3a702858f1cad70b827bcb8c687a6084ca61f7d5fcebd` |
| `probe.json` | `53563b4beddf02d941fca7b187752106db21a0b71127ee2e629a8505d322a18c` |
| `probe.log` | `3e2b2beca28a8ff8773dc668102d64a76f8e6abcbeaf84df0bcc23e2e14f781a` |
| `ir-decision.json` | `831997223e7b3bba684de459583abb42e79d6f37e53f10ce9005e4d92a518033` |
| `benchmark.json` | `f531da6d3ec0d1b527cfb17c97d467a3134c2b73eeee1f31b675cba009e8b19f` |
| `benchmark.log` | `fee81a64f4133456261bde37b7d7a15b6b8232db9ef2660d352b0230c6eccf2d` |
| `raw-samples.csv` | `e60704a06772df8c9c780b8d37c9c4dbe7dff2ac73451603d9d5d34325526e20` |

- 最终远端 `/tmp/flagos-t74-block-merge.3c5uVG`：正确性/IR PID
  **392888**（外层550秒，内层正确性400秒、probe120秒），benchmark PID
  **392970**（外层660秒、计时总上限630秒）。预检与 benchmark wrapper
  均 **EXIT0**，最终正确性、probe、完整性能三个阶段均成功结束。
  `launch.json` SHA `a3d61d62dfcfc68a297172981764a23715394bd94ab86d408bce8249061f288e`，
  `benchmark-launch.json` SHA `8bf60df81b303a81172013c942e123307bfbdfde3722d65e11382ad4dabba97d`。
  阶段前后 GPU 快照均无其他 compute 进程；RTX5070Ti、driver610.57.04、
  Python3.12.13、Torch2.13.0+cu130、Triton3.7.1。所有回执身份与源/测试/
  计划/CSV、IR 裁决之间的 SHA 链均已复核，目标 runtime 仍未验证。

## 2026-09-16 晚间 E11：短请求四行分组，开发验证完成、未提交

- 固定团队最佳E4 `45662b8c403778e4b93b96ec90b4ca3306c6a03a` 为基线。`N>1024 && max_q_len<=32` 走4请求×32位置展开，一组按真实q最大值循环；hint不限制本路径写入。保留E4 prefix、小N/长q普通核；相对主树E6恢复E4二维调度，同时接入既有E9/E10宽prefix/安全grid路径，不把旧E6字节当团队最佳。
- 新地址始终i64，结果值int32 wrap后signed clamp；大grid按请求/组步进。wide域包括total或stride物理地址超过int32等条件；该安全域小N可能由单launch变两次。普通fallback仍依赖原有max_q_len提示，不宣称修复所有既有hint契约缺口。
- 冻结18个短q主桶+8个小N/长q/one-long控制，5轮AB/BA，wrapper含分配与prefix；门槛主算术均值≥1.03、每轮≥1.01、每控制中位数在[0.97,1.03]。实测主算术均值 **1.261561825**、GM **1.253914198**，每轮1.258086–1.266228，最差主桶1.055275，零回退、零control drift、零spill，过门。这是NVIDIA短q代理结果，不能解释成平台总分涨26.16%。
- 15项候选回归与5项原E4回归均过；48份TTIR/TTGIR/LLVM/PTX绑定。短q group为34寄存器、0shared/0barrier、2次shuffle；前缀及长q控制四格式汇编逐字相同。没有二分、gather或merge co-rank。计时证据 `artifacts/competition/t74-grouped-20260916/`。
- source/verification commit `6a516b2838857087ac657fd624d6fb31bbcdd895`，源码与筛选候选逐字一致，SHA `264b700cb67c1c287b4a494c8bca6745acd6e979f62293e134dfa41bf0565791`。正式测试SHA `5a6a6219f845d525c0ce0123c40c0b153a764f608953f63f49a14dfa85e747df`：保留15方法，补充非零输出的低报hint断言，Black格式化不改变AST。
- 完整release **15/15**，0失败/错误/skip/xfail，入口136次、实际kernel launch189次；NVIDIA RTX5070Ti / driver610.57.04 / Python3.12.13 / Torch2.13.0+cu130 / Triton3.7.1。最大请求数65537已运行；超过8GB实际输出未执行，metadata mock仅为分派证明，不算设备数值执行。
- 远端release `/tmp/flagos-t74-grouped-release.AoXLZH`，PID396344、600s上限；本地完整证据 `artifacts/competition/t74-grouped-release-20260916/`。初次screen传输文件名错误在启动Python入口前退出，未运行kernel；新隔离目录完成全部筛选，不将该传输错误计入候选正确性。
- ZIP `artifacts/competition/seqlens_expand/e11-grouped-6a516b2/seqlens_expand.zip`，9181bytes、单generic成员，SHA-256 `725be3800805576c9ce8c99fc4744f62bfdb036ac3e64316585a6372027d1c5f`。dry-run/final manifest、Git字节、成员及release回执/相邻日志验签通过。
- `verification.json` SHA-256 `92b5ba5fa82d2f3737cd8ee28cadb2f396ac40c3d4014e38c3a7cbd54eb771d1`。
- `verification.log` SHA-256 `e04a25a6a68da9a6575c960360d8538234d9c18c3be57a13bb5dff89c8231bdb`。
- `release-audit.json` SHA-256 `3b9aca64fb0fd904545e62512318c6f0e466e0397a61bf5fda1f7397a26a4543`。
- 开发、验证和打包完成，**未运行平台preflight、上传或提交**；所有目标芯仍未取得本候选的实际运行证据，保留E4团队最佳记录。

## 2026-09-16 22:21 提交前复核：E11暂停，修复hint边界

- 公开reference不读取max_q_len，题面没有为其规定下界。E11虽修复grouped路径，旧smallN/long-hint路径仍以hint限定迭代，静态反例N=1、q=1025、kv=1042、total=1025、hint=1会漏写最后一项；N=1025且hint=33也存在同类缺口。该问题为既有路径缺陷，已执行15方法通过不能覆盖此域。
- E11不运行preflight/上传/提交；E12将三条fallback循环改由实际qo限定，hint只保留启动调度用途，补非零预期/poison回归并重新验证性能与release。旧ZIP、回执和筛选成绩保留历史身份，不为E12背书。
- 22:21实时榜单我方E4为20.177775、第8，榜首26.821525；与T69共用上述逐芯快照。完成修复后正式晋级门仍为八芯正确、每芯≥0.1、均值>20.177775，未过门保留E4且不重发同一候选。

## 2026-09-16 22:37 E12：实际长度边界修复，验证与打包完成

- 最小根因修复：三个非group kernel按读取的实际qo计算i64 tile数并循环，hint只用于启动调度；地址保持i64、输出按int32 wrap后clamp。group、prefix和host wrapper算法未变。source/verification commit `ecda8d3775bbf8875f5ba8b4c5d42ec82de3863e`；源码SHA `58a135c85b6b8970eee9cbca9a03f68af83cd38d3cb848c26d373efee9dfdb59`，测试SHA `f7c5e2d6b7440c2f9a817954e034196bed5a72e453e6b0e5803f0b06abe70a05`。
- 原15方法保留，新增公开入口test_actual_lengths_ignore_hint_all_paths：N=1/5/1025/65537、hint=0/1/33/1024、跨tile真实长度、非零预期和真实分配poison。旧E11逐字源码运行该方法，16子组合中12失败、0error/skip、16入口/24JIT；新版本全部通过。不是仅靠静态猜测，也未替换实际kernel执行。
- 新预注册保留E4基线、原26桶及5轮AB/BA。18主桶mean≥1.03、每轮≥1.01；8个旧控制因fallback实际循环边界改变，预先改为各median≥0.97的语义对照，不再限制合法加速上限。实际mean **1.2597481395**、GM **1.2527069942**，最差主桶1.062019866，每轮最低1.257395498；对照范围0.988173733–1.383845286，无主桶回退/无spill。130原始配对样本由主任务独立复算一致；这些是NVIDIA代理结果，不外推平台均分。
- 72份编译产物验签，三条改核有actual qo→i64 loop/address及int32值wrap证据；screening完整16/16（152入口/225JIT）和E4原5方法均通过。精确commit release另行 **16/16**、0fail/error/skip/xfail、152入口/213真实JIT；筛选/发布JIT计数分别记账。
- NVIDIA RTX5070Ti / driver610.57.04 / Python3.12.13 / Torch2.13.0+cu130 / Triton3.7.1。远端旧复现 `/tmp/flagos-t74-e12-old.XYB8qC` PID396466，screen `/tmp/flagos-t74-e12-screen.pExTaH` PID396514、benchmark PID396623，release `/tmp/flagos-t74-e12-release.kZB4ng` PID396674；GPU阶段串行、前后无其他compute进程。所有非NVIDIA目标runtime仍未验证。
- 不可变ZIP `artifacts/competition/seqlens_expand/e12-hint-safe-ecda8d3/seqlens_expand.zip`，9266bytes、单成员seqlens_expand.py（9134bytes），SHA-256 `529ff7c76c82bb6ea364b2ebc5249469f61d5daa5f1f977207e7161221053055`。dry-run/final、Git逐字、CRC、回执及完整日志均验签。E11未提交，其ZIP仍封存。
- `artifacts/competition/t74-hint-safe-20260916/old-regression/reproduction.json` SHA-256 `2e91b7bf1ab6fc716bbfad9d02f316e2953ddb1576c24c1c4005df5b771c364b`。
- `artifacts/competition/t74-hint-safe-20260916/screening/plan.json` SHA-256 `26dbe33ec0bcc2099b63fb5988bec7fd9a71ceaa70cf7d3ae5bb14fd24bf0aca`。
- `artifacts/competition/t74-hint-safe-20260916/screening/benchmark.json` SHA-256 `63126753cbcd1cdd99f29b9c726714da5060f8bd02699d56fed11b1450a643cf`。
- `artifacts/competition/t74-hint-safe-20260916/screening/raw-samples.csv` SHA-256 `5f986e73a062e50643dfd30fee21b509aef565501bf211d54c5fbce4b1332be8`。
- `artifacts/competition/t74-hint-safe-20260916/release/verification.json` SHA-256 `e73d37d9cbca786b133ef524a5afb761eaba0ccad0e627259226587830adc3f4`。
- `artifacts/competition/t74-hint-safe-20260916/release/verification.log` SHA-256 `a13c65e0b572573ff953bff59dba3752271a1e6cfb938181777ad0bf8f9a636c`。
- `artifacts/competition/t74-hint-safe-20260916/release-audit.json` SHA-256 `1a6505e45947ad263bec5e4d553dedefff8b81d217b3c059d916832277593970`。
- 正式实验仍以八芯正确、每芯≥0.1及均分>团队最佳E4 **20.177775** 为晋级门；假设短请求减少CTA、长hint路径减少空tile可提高总均分，影响范围及目标芯增幅待平台。只发一次，未过门保留E4、不重投相同ZIP。
