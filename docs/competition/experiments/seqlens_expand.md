# Task 74 `seqlens_expand` 实验记录

```current
task: 74
operator: seqlens_expand
batch: 5
validity: valid
platform: completed(e4,8/8,20.1778x 新TB;海光28.78门兑现,沐曦+2.09)
candidate_stage: e4
team_best_stage: e4
team_best_speedup: 20.177775
sealed: no
next: E8空tile提前退出GM1.03028未达门，关闭；保留e4 TB，后续发布另须修复二维grid乘积上限
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
