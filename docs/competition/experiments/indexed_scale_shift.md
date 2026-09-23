# Task 83 `indexed_scale_shift` 实验记录

```current
task: 83
operator: indexed_scale_shift
batch: 6
validity: candidate(旧e2 7/8，新e3rr待平台)
platform: completed(17729,e2,7/8;昆仑多case失败，含1元素刀刃及大shape失配)
candidate_stage: e3rr
team_best_stage: -
team_best_speedup: -
sealed: no
next: e3rr复用输出缓冲区跨三kernel落实bf16边界；release 5/5、ZIP验签、codex-review第三轮无缺陷；实时preflight后单次提交，门=8/8且每芯>=0.1，否则回旧e2源码
updated: 2026-09-23
```

## 过程摘要（2026-09-19 凌晨，题面 09-18 晚随批 6 扩容上线）

- 开发：s0 generic + 全 unittest 矩阵（commit `540b574a` 家族），代理
  release 多轮修复后 8/8 验证；按回执逐芯定位失败根因，vendor 修复弹
  按 codex-review 门后发射（review 抓出 moe_align_single_token 填充轮
  越界读 P1 并修复）。
- 回执与产物：见 `artifacts/competition/b6ext-*-20260919/`（verification
  + log 逐弹归档），ZIP 在各 `artifacts/competition/indexed_scale_shift/` 下按 stage。
- 今日新增跨芯硬事实：XPU 拒 f32→i16 位转换（经 i32 高半字绕过）；
  XPU 拒 tt.scan/tl.cumsum 与 atomic_rmw；XPU 上张量索引 gather 与
  runtime 宽度寻址产生垃圾（hash_topk 三形态同指纹，标量串行为唯一
  可用形态）；GCU make_gcuir 拒张量索引 gather（torch 预 gather 先例
  再证）；XPU bf16 downcast 刀刃值与 eager 差 1 ulp（rtne 显式钉仍差）。

## 2026-09-20 测试修正：真中点 tie 覆盖（咨询第六轮纠错落地）

- 旧 `test_double_round_boundary` 用 2^-9 —— bf16 在 [1,2) ULP=2^-7，
  2^-9 只是 1/4 间距，从未触及 tie。改为三个真中点：+2^-8（1.0 偶尾数
  vs 1+2^-7 奇）、+3·2^-8（1+2^-7 奇 vs 1+2^-6 偶）、-2^-9（1-2^-8 vs
  1.0，1 下方 ULP 减半）；x=1/shift=0 使第二/三轮舍入无 tie，用例纯隔离
  one_plus 轮。
- 核对内核（当前 generic）三处显式舍入链与 reference 逐轮一致；昆仑
  刀刃容差的舍入定位诊断（构造 scaled/final 轮 tie）留待下一 GPU 窗口。
- 无新候选：仅测试字节变化，不影响既有回执绑定。

## 2026-09-23 E3：跨 kernel 落实 bf16 舍入边界

- **重新核对平台原始回执**：e2 提交 `17729` 确实选择
  `indexed_scale_shift_kunlunxin.py`，全平台 7/8 芯通过。昆仑失败并非
  只有一元素：case 1 是 1/75776 元素差 0.015625（阈值 0.015），
  case 2/5/6/7/8 有少量失配，case 3 有 4154643/12582912 元素
  失配（33.0%）。原 CURRENT 的“一元素容差”只描述首个 case，不能
  概括该提交。大 shape 失配使同核写后读/编译重排成为可检验假设。
- `codex-ask` 建议将 `bf16(1+scale)`、`bf16(x*one_plus)`、最终加法
  分成三个独立 Triton kernel，前两段各真实写入 bf16 张量，再由下一段
  读取。e3 仅修改昆仑 vendor；generic 字节冻结。新增一个 exact-bit
  测试，分别把三次 bf16 舍入放到真中点，含非整 17 维尾块及 int32 索引。
- 源码/测试 commit `ee60c4139df462e0582291f1c4abb801428db168`；
  generic SHA-256 `efd3fd554a953e8fe65b5299079d63d62f1dca4b830d99958215d72f8d5a49c0`，
  kunlun SHA-256 `8f3e4d3743eac744872defb550ab7273c2c43685b06ecee2c908dfd592006ddb`，
  test SHA-256 `a73b15dfdbb5b271fd2dde17b15d4a2ee2cd4df5d6a35faf94a3fd350d31c48b`。
  ZIP `artifacts/competition/indexed_scale_shift/e3-ee60c41/indexed_scale_shift.zip`
  6706 B、SHA-256
  `3bb05c9d9142361d5227f6a3e4048464a4b89c9291725ccb9b1f5221901ab01d`，
  两成员 `indexed_scale_shift.py` / `indexed_scale_shift_kunlunxin.py`；
  `--verify-existing` 和 `unzip -t` 通过。
- 代理 release：同 commit 的
  `artifacts/competition/indexed_scale_shift/e3-ee60c41/verification.json`
  SHA-256 `c22f23170cb263a097e7d0e95b2c8973e73e2cf52be39a15fd345380f5adf24e`，
  日志 SHA-256 `68eb59df1f55691e976bb7af0bb41e1d11faac2fc2ede071aec67caf61fcc730`；
  4 测试 0 失败/错误/skip，generic 11、kunlun 33 次真实 launch。
  py_compile、Black、isort、flake8 均通过。NVIDIA 仅为数学代理，
  昆仑目标仍标 `target-runtime-unverified`。
- **预注册门**：平台 8/8 正确、每芯 speedup ≥0.1 才建立首个有效成绩；
  其他七芯与旧 e2 相同 generic 字节，若出现异常回退核对平台水位。
  若昆仑编译失败、数值失败或低于 0.1，e3 判负并恢复 e2 vendor 源码
  字节，不重试同一候选；速度优先级低于首次有效性。

## E3r：首轮 codex-review P2/P3 修复

- 首轮评审指出 e3 为稀疏索引预计算整张 `(num_variants, hidden)` 表会在
  `rows=1、num_variants` 极大时无谓分配和发射（P2）；新二维 grid
  缺少第二块尾部测试（P3）。e3 **未提交平台**。e3r 将第一段改为仅按
  `indices[row]` 计算被使用的行，临时张量 `(rows, hidden)`；仍保留
  两次真实 bf16 中间存取和三个独立 kernel。测试新增 `hidden=1025`
  第二块尾部、多 variant 索引，以及 stride0 的百万逻辑行稀疏表，
  新旧路径精确比对。
- 源码/测试 commit `61016b826a2a3694eaee254024cf0d47350b4cec`；
  kunlun SHA-256 `36caaeeb63b0519dbbdc41bb465b7d077e5cd3a2f3d47b53a3031b9ce6081a05`，
  test SHA-256 `16fd801e5191c6722497fa1fe60b333521622eb600e3ff06036517f3537468af`；
  generic 仍是 `efd3fd554a953e8fe65b5299079d63d62f1dca4b830d99958215d72f8d5a49c0`。
- ZIP `artifacts/competition/indexed_scale_shift/e3r-61016b8/indexed_scale_shift.zip`
  6892 B，SHA-256
  `e980627e17be0724d7dd239d8ce928e4f5ac02326cc0edf5425802aa83baf176`，
  两成员同 e3；打包器 existing 验签及 `unzip -t` 通过。
- 同 commit release：
  `artifacts/competition/indexed_scale_shift/e3r-61016b8/verification.json`
  SHA-256 `746a76d0a12d55594ac5664b5b9474a523857d069374a41c907100bb20a2715d`，
  日志 SHA-256 `968ac43ab2eab19b75b93846d1dcf1785dfd694b603ad57d0c3fa5caab020b72`；
  5 测试 0 失败/错误/skip，generic 13、kunlun 39 次真实 launch；
  py_compile、Black、isort、flake8 通过。昆仑目标仍待正式平台执行。
  预注册晋级门与上节一致；第二轮 codex-review 审完整 e3+e3r 差异。

## E3rr：第二轮 codex-review P2 修复，零额外张量

- 第二轮评审指出 e3r 在 `rows≫num_variants` 时按行临时表仍会占据
  与输出同量级的额外显存；e3r **未提交平台**。e3rr 直接复用契约输出
  `out`：第一 kernel 写 bf16 one_plus，第二 kernel 读它并原位写 bf16
  scaled，第三 kernel 读 scaled 原位写最终结果。三个独立 launch 保证
  两次中间舍入各跨真实内存边界，wrapper 仅一次输出分配，且不依赖
  `rows/num_variants` 比值。既有 1025 尾块、稀疏表及行间隔输入用例
  全部通过。
- 源码/测试 commit `3015a0cdf64282d39344a7bbf83a60e3b290ed8e`；
  kunlun SHA-256 `f68061f6ed293494adf6a427de4e2b25c4a713a5d9e8b7d34b787bffa9cf4949`，
  test 与 e3r 同 SHA-256
  `16fd801e5191c6722497fa1fe60b333521622eb600e3ff06036517f3537468af`；
  generic 仍是 `efd3fd554a953e8fe65b5299079d63d62f1dca4b830d99958215d72f8d5a49c0`。
- ZIP `artifacts/competition/indexed_scale_shift/e3rr-3015a0c/indexed_scale_shift.zip`
  6793 B、SHA-256
  `05705b45c1abba10068f0de026db6743c7c8b4bca1d55bcbdf31cd55684286cf`；
  两成员同前，existing 验签与 `unzip -t` 均通过。
- 同 commit release：
  `artifacts/competition/indexed_scale_shift/e3rr-3015a0c/verification.json`
  SHA-256 `cb9c7a7df9e0ff16c3e9826ed543d2e29832622c779391b481c32b9c9c4ee310`，
  日志 SHA-256 `6702c7b59e307af4359cda0eb82cbf555ea9e1da75c9db545dc95e0f25a21e15`；
  5 测试 0 失败/错误/skip，generic 13、kunlun 39 次真实 launch，
  py_compile、Black、isort、flake8 通过。昆仑目标仍待平台执行。
- **最终预注册门**：8/8 正确且每芯 ≥0.1 获得首个有效成绩；若昆仑
  编译/数值或速度门槛失败，恢复 e2 vendor 原字节，e3rr 不自动重试；
  其他七芯复用旧 e2 generic 原字节，其波动逐芯对照，不误归因本改动。
  第三轮 codex-review 审 e3 至 e3rr 累积 diff，通过后才 preflight。
- 第三轮 `codex-review --base d915a42e --spec .../83-indexed_scale_shift.md`
  审 e3→e3r→e3rr 累积差异：Spec 与 Standards 均未发现符合报告门槛的
  缺陷；其结论明确不替代昆仑目标运行结果。评审门已通过。
