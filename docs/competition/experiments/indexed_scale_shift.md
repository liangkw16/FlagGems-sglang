# Task 83 `indexed_scale_shift` 实验记录

```current
task: 83
operator: indexed_scale_shift
batch: 6
validity: invalid(7/8,e6昆仑XPU编译失败)
platform: e6(20448)7/8 invalid_correctness;arith.select类型校验失败;源码回滚e2
candidate_stage: -
team_best_stage: -
team_best_speedup: -
sealed: yes
next: 昆仑XPU目标编译通道可用后再审e4稀疏舍入；当前额度转T90/T92/T81/T76/T91
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

## 2026-09-23 E3rr 平台终态（20438）：7/8，三段边界假说证伪

- 实时 preflight 绑定 SoulCoder/T83/e3rr、源码与测试
  `3015a0cdf64282d39344a7bbf83a60e3b290ed8e`、ZIP SHA
  `05705b45c1abba10068f0de026db6743c7c8b4bca1d55bcbdf31cd55684286cf`、
  release SHA
  `cb9c7a7df9e0ff16c3e9826ed543d2e29832622c779391b481c32b9c9c4ee310`；
  一次性提交 **20438**。远端已上传 ZIP 哈希/大小双验签通过，昆仑选中
  `indexed_scale_shift_kunlunxin.py` 并完成编译运行。终态
  `invalid_correctness`、7/8、均分无效；额度余 8/30。
- 逐芯通过且 speedup：天数 14.5882、沐曦 8.8402、燧原 0.9842、
  海光 13.1402、华为 3.7456、A 11.2778、B 9.6418；昆仑失败无速度。
  昆仑 case 1 为 7/75776（最大差 0.015625，阈值 0.015），
  case 3 为 **4,154,953/12,582,912（33.0%）**；e2 同 case 是
  4,154,643/12,582,912，且最大差同在 `(2508,793)`。
  三个真实 kernel 边界仍保持这一主指纹，故“中间 bf16 未物化”为
  主因的假设判负。
- 12,582,912 可写成 3072×4096，约三分之一失配与 grid
  `min(rows,2048)` 的第二次迭代 1024 行吻合；**这只是推断**，
  平台没有公开 case 3 的完整 shape/输入。已用 codex-ask 询问是否
  应优先改为每行一 program、去 grid-stride，再考虑位级 bf16 舍入。
- 按预注册门，仓库昆仑 vendor 从 e2 ZIP 原样恢复，SHA-256
  `44dc00e2e8a7816a8405ebf6bf02f4b5ce23f1c07ae501b3d01f1bf1c8d8c6a4`，
  与 e3 前源码 diff 为空；回滚 commit `fc5f7b3e`。e3rr ZIP/回执
  留档且不重传，下一候选必须另立假设与完整门禁。

## E4：每行一 program 排除 XPU 第二轮 grid-stride

- 第二次 `codex-ask` 独立判断：e2/e3rr case 3 错数均约 415 万，
  12,582,912 **可能**为 3072×4096；若如此，`grid=min(rows,2048)`
  后的 1024 行恰占三分之一，两个回执最大差均在 `(2508,793)`。
  证据支持优先测行调度，却未公开真实 shape/失配位置分布，不能宣称
  已定位 XPU 编译器错误。建议仅把三段 kernel 的外层 grid-stride
  row 循环改为 `row=tl.program_id(0)`、`grid=(rows,)`；列块、bf16
  舍入、out 原位复用与三次 launch 全保留，位级舍入后置。
- 候选从 e3rr **不可变 ZIP 字节**重新取源，不从回滚后的 e2 拼接；
  code/test commit `4d12ecf004eb23dc5d1f3ff4aed9205da274a409`。
  kunlun SHA-256 `b5ee7f10b3b789fefcc9a63a681ddc200125f8ad93a1ae8e49582388f67ed4bf`，
  generic 仍 `efd3fd554a953e8fe65b5299079d63d62f1dca4b830d99958215d72f8d5a49c0`，
  test SHA-256 `67344c61d3ec1a3db3503357fe4c7d3a12d822d8f37dc3c87380433c1430a32e`。
  新 exact-bit 行边界回归覆盖 rows 2047/2048/2049 与 3072×4096；
  输入每行数值与 variant 均变化，第二轮漏算不能靠同值蒙混。
- ZIP `artifacts/competition/indexed_scale_shift/e4-4d12ecf/indexed_scale_shift.zip`
  6475 B、SHA-256
  `ef37e9df56adc21105ad232bddea7fe4a9406fc85eae971f7d0bcd9e03b57c15`，
  两成员 `indexed_scale_shift.py`/`indexed_scale_shift_kunlunxin.py`，
  existing 验签与 `unzip -t` 均通过。
- 同 commit release：
  `artifacts/competition/indexed_scale_shift/e4-4d12ecf/verification.json`
  SHA-256 `9d77c36bd65b8ced9a00b3f8018b1d38475cda357575e07e7ec960686d94b458`，
  日志 SHA-256 `6a47a89d7575b5b6ead194c6122c3324c6e36f432ac29f3af681c37819f07fab`；
  6 测试 0 失败/错误/skip，generic 17、kunlun 51 次真实 launch；
  py_compile、Black、isort、flake8 通过，仍是 NVIDIA 数学代理。
- **预注册门**：8/8 正确且每芯 speedup ≥0.1 才保留为有效成绩。
  若昆仑仍失败，回滚仓库至 e2 vendor 原字节，不重投 e4；比较
  case 3 失配数与 e3rr 的 4,154,953：若下降 >99%（<41,550），
  grid 调度假说获得强支持，后续新候选可从 e4 ZIP 固定字节继续解决
  残余舍入；若基本不变，转查索引/地址与计算。平台若编译/超时或
  速度破 0.1，亦回滚。其他七芯复用旧 e2 generic 成员字节。
- `codex-review --commit 4d12ecf0 --spec .../83-indexed_scale_shift.md`
  已完成：Spec 和 Standards 均无可靠缺陷；评审明确 NVIDIA release
  不等于昆仑目标验证。评审门通过。

## 2026-09-23 E4 平台终态（20440）：行调度主因得到强支持，数值尾差待解

- 实时 preflight 与 ZIP SHA `ef37e9df56adc21105ad232bddea7fe4a9406fc85eae971f7d0bcd9e03b57c15`
  验签通过，远端上传字节哈希/大小复核一致；一次性提交 **20440**，昆仑选择
  vendor 文件并运行。平台终态 `invalid_correctness`、7/8，通过七芯分别为
  天数 14.6696、沐曦 8.8630、燧原 0.9666、海光 12.8904、华为 3.6368、
  A 11.0744、B 9.7056。昆仑无速度，额度余 **7/30**。
- 昆仑 case 3 失配从 e3rr 的 **4,154,953/12,582,912** 降至
  **899/12,582,912**，下降 99.978%；case 1/2/4/5/6/7/8 分别失配
  7/247/1/10/207/1859/6973 元素。case 3 最大绝对差 0.046875
  位于 `(88,608)`，最大相对差为 inf 位于 `(4,1469)`。逐行调度修复了
  主要错误的证据很强；平台未给原始张量，尚不能把根因定性为 XPU
  grid-stride 编译器错误。剩余低密度误差需单独诊断。
- 按预注册门，仓库昆仑源码从 e2 ZIP 原字节恢复，SHA
  `44dc00e2e8a7816a8405ebf6bf02f4b5ce23f1c07ae501b3d01f1bf1c8d8c6a4`；
  与 e3 前源码 diff 为空，回滚 commit `8776374c`。e4 ZIP/回执保留，
  不重投 e4；下一候选从 e4 不可变 ZIP 取源，并另做验证和评审。

## E5：逐位钉有限值 RTNE，保持 e4 结构

- 第三次 `codex-ask`（`gpt-6-sol`、`max`）审 e4 平台稀疏失配、三段真实
  bf16 写入和代码后，建议三处有限值 f32→bf16 舍入统一改为整数位级 RTNE：
  f32 位重解释为 u32，加 `0x7fff + ((bits>>16)&1)` 后清低 16 位，再
  重解释为 f32 供原 bf16 store；Inf/NaN 仍交给原转换。该方案是推断，
  昆仑目标无中间值，不能排除稀疏地址或算术错误。e5 从 e4 ZIP 原字节取源，
  保留逐行 grid、三 kernel、out 复用、索引和算式不变。
- 代码/测试 commit `7928c166b56cc3fe8ad82680ac0da40b444654f2`；
  generic SHA-256 `efd3fd554a953e8fe65b5299079d63d62f1dca4b830d99958215d72f8d5a49c0`，
  kunlun SHA-256 `caae63a45932e6cf31b1aeda13d8e63e2e5f8c9b2c9dbf9fe83ff6fdd1e4de60`，
  test SHA-256 `8ffaff25cf4c4121dc15f6b4f12ddcd0a993dae9f01e35dbe621d893f09104e4`。
  测试加正负取消项与 Inf/NaN 分类，保留行边界 exact-bit 回归。
- ZIP `artifacts/competition/indexed_scale_shift/e5-7928c16/indexed_scale_shift.zip`
  6749 B、SHA-256 `b81636590ffb77708e00d5894944b4147b231adeb4483343f067a20993fb6ae6`，
  两成员 `indexed_scale_shift.py`/`indexed_scale_shift_kunlunxin.py`；
  `--verify-existing`、`unzip -t` 通过。
- 同 commit release：
  `artifacts/competition/indexed_scale_shift/e5-7928c16/verification.json`
  SHA-256 `5519414f35b88cd8cce812fe4848f653d9f33ef843e721634db4aebaf5db7ec3`，
  日志 SHA-256 `25bc702f5de19bd8afc8857526abdf93177501e8b3b6dd5cbeba11c30b234e95`；
  NVIDIA RTX 5070 Ti 代理 7 测试 0 失败/错误/skip，generic 22、kunlun
  66 次实际 kernel launch。py_compile、Black、isort、flake8 均通过。
  代理 PTX 见 u32 取位、加偏置、掩码与最终 bf16 store；昆仑目标仍是
  `target-runtime-unverified`。
- **预注册门**：codex-review 无可靠缺陷后才 preflight；平台 8/8 正确且
  每芯 speedup ≥0.1 才建立首个有效成绩。昆仑编译、数值或速度失败则
  从 e2 ZIP 恢复 vendor 源码，不重投 e5；若数值仍失败，逐 case 对照 e4
  的 899/12,582,912 等错数。其他七芯 generic 成员字节保持不变。
- `codex-review --commit 7928c166 --spec .../83-indexed_scale_shift.md`
  完成：Spec 和 Standards 均未发现可确认的新增缺陷；评审明确昆仑目标
  尚未验证。审查门通过。

## 2026-09-23 E5 平台终态（20445）：XPU 位转换编译墙

- 实时 preflight 将 SoulCoder/T83/e5、源码/测试 `7928c166`、ZIP SHA
  `b81636590ffb77708e00d5894944b4147b231adeb4483343f067a20993fb6ae6`
  与 release SHA `5519414f35b88cd8cce812fe4848f653d9f33ef843e721634db4aebaf5db7ec3`
  逐项绑定；一次性提交 **20445**，远端 ZIP 哈希/大小均验证一致。
  平台终态 `invalid_correctness`、7/8，额度余 **6/30**。
- 七芯通过且 speedup：天数 14.5266、沐曦 8.8072、燧原 0.9694、
  海光 12.9590、华为 3.6436、A 11.3984、B 9.3922。昆仑选中
  `indexed_scale_shift_kunlunxin.py`，九个 case 全部在 `_bf16_rtne_fp32`
  的 `value.to(tl.uint32, bitcast=True)` 编译失败：
  `triton_xpu.unpack reached lowering without a bufPtr; tritonxpu-alloca must attach one`
  （`ConvertTritonXPUToLLVM`）。因此本弹**没有**检验位级 RTNE 能否消除
  e4 的稀疏数值失配，不能据此判定该数值假说真假。
- 按预注册门，从 e2 ZIP 还原 vendor 原字节 SHA
  `44dc00e2e8a7816a8405ebf6bf02f4b5ce23f1c07ae501b3d01f1bf1c8d8c6a4`，
  与 e3 前源码 diff 为空，回滚 commit `e1c44da8`。e5 不重投；已用
  `codex-ask` 携本次编译错误与 T87 已成功的 f32→i32 位读先例，咨询
  下一独立候选。目标芯可编译性仍是首门，数值门次之。

## E6：有符号位读与原始 uint16 存储

- 新 `codex-ask`（`gpt-6-sol`、`max`）对 e5 的 XPU `unpack` 编译错误
  与 T87 的 f32→i32 已通过先例做区分：建议从 e4 不可变 ZIP 取源，
  只把三处 bf16 转换/存储改为有符号 i32 位读、RTNE 加偏置、取高
  16 位后经重类型化指针存原始半字。这样避开 e5 的 u32 位读、
  i32→f32 反向 bitcast 和最终有限值 f32→bf16 转换。保持三段物化、
  逐行 grid、索引和算式不变。XPU 是否接受 i32 位读的算术结果及
  uint16 指针重类型化仍未知，不把 T87 的成功外推为本题通过。
- 代码 commit `57cefa93a2b99c413e80f6e1ec760318039c1b34`；测试沿用
  `7928c166` 的正负取消项、Inf/NaN 分类与全部行边界用例，测试 SHA-256
  `8ffaff25cf4c4121dc15f6b4f12ddcd0a993dae9f01e35dbe621d893f09104e4`。
  generic SHA-256 `efd3fd554a953e8fe65b5299079d63d62f1dca4b830d99958215d72f8d5a49c0`，
  kunlun SHA-256 `db6af7089766ebd98c8b06cc3fe63ec413833f9d1d4baa3205d388d5294bdc09`。
- ZIP `artifacts/competition/indexed_scale_shift/e6-57cefa9/indexed_scale_shift.zip`
  6769 B、SHA-256 `eda724dda6782f1f9a6d1f03f6a36578b80801185b270df3bd3d952d749c35a4`，
  两成员 `indexed_scale_shift.py`/`indexed_scale_shift_kunlunxin.py`；
  existing 验签与 `unzip -t` 通过。
- 同 commit release：
  `artifacts/competition/indexed_scale_shift/e6-57cefa9/verification.json`
  SHA-256 `8f89ef0459cbb9065fbe0ebdc5b318c96ead6f31fc332968f3ff932816275c54`，
  日志 SHA-256 `6a2ec2c1f0ae7588b3088842b4c3e38c0dfe51dbff38856c53be37cec25cef04`；
  NVIDIA RTX 5070 Ti 代理 7 测试 0 失败/错误/skip，generic 22、kunlun
  66 次实际 kernel launch。py_compile、Black、isort、flake8 通过；三段
  代理 PTX 均有 `st.global.b16`，无有限值 `cvt.rn.bf16.f32`。昆仑目标
  仍 `target-runtime-unverified`。
- **预注册门**：codex-review 无可靠缺陷后再 preflight；8/8 正确且每芯
  speedup ≥0.1 才成为首个有效成绩。昆仑编译、数值或速度失败则
  恢复 e2 vendor 原字节，不重投 e6；若能运行但数值失败，逐 case
  与 e4 的 899/12,582,912 等稀疏失配对比。其他七芯 generic 成员
  字节不变。
- `codex-review --commit 57cefa93 --spec .../83-indexed_scale_shift.md`
  完成：Spec 和 Standards 均未发现可确认的新增缺陷；评审明确 NVIDIA
  回执不证明昆仑编译或性能。审查门通过。

## 2026-09-23 E6 平台终态（20448）：XPU `arith.select` 编译墙，回滚

- 实时 preflight 绑定 SoulCoder/T83/e6、源码/测试 `57cefa93`、ZIP SHA
  `eda724dda6782f1f9a6d1f03f6a36578b80801185b270df3bd3d952d749c35a4`
  与代理回执 SHA `8f89ef0459cbb9065fbe0ebdc5b318c96ead6f31fc332968f3ff932816275c54`；
  一次性提交 **20448**，远端 ZIP 哈希/大小 `verified`。平台终态
  `invalid_correctness`、7/8，观测于 `2026-09-23T17:48:48+08:00`，
  额度余 **5/30**。
- 七芯通过且 speedup：天数 14.3794、沐曦 8.8504、燧原 0.9848、
  海光 12.936、华为 3.951、A 11.3984、B 9.6582。昆仑选中
  `indexed_scale_shift_kunlunxin.py`，8 个 case 在 `_store_bf16_rtne`
  的 `tl.where(special, 0, bits)` 触发 `TritonXPUUnrollControl`：
  `'arith.select' op failed to verify that all of {true_value, false_value,
  result} have same type`。因此没有数值读数，e4 的 899 个 case 3 尾差
  仍未被位级 RTNE 假说检验。
- 按预注册门，从 e2 不可变 ZIP 恢复 Kunlun vendor SHA
  `44dc00e2e8a7816a8405ebf6bf02f4b5ce23f1c07ae501b3d01f1bf1c8d8c6a4`，
  与 e3 前源码 diff 为空，回滚 commit `c39eecf7`。e6 不重投；没有可
  绑定源码的 XPU 编译通道前，暂停 T83 平台试错，把额度转向已有效题目。
