# Task 36 `selective_state_update` 实验记录

```current
task: 36
operator: selective_state_update
batch: 3
validity: invalid
platform: 7/8
team_best_stage: e8
team_best_commit: 7414c69
team_best_speedup: 七芯~5.8
blockers: e15昆仑case2-4同E13数值指纹
sealed: no
next: e16关闭stage2 Vectorize
updated: 2026-09-01
```

状态:E11 昆仑 7.219s 明确落到 stage1 `8×16` uni_sram;
E12 保留 P=8/N=16,仅关闭 XPU stage1 vectorization pass。

## 契约锁定

- 签名:`reference(state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False)`
- `state [B,H,dim,N]`;`x/dt [B,H,dim]`;可执行平台契约
  `A [H,dim,N]`(负);`B/C [B,G,N]`
  广播 g=h//(H//G);可选 D/z(silu 门)/dt_bias/softplus;fp32 计算
- 输出 `(y.to(x.dtype), state_new.to(state.dtype))`;输入 state 不变
- 容差:fp32 1e-4 / bf16 1.5e-2 / fp16 1e-2;八芯

## S0(2026-08-29,commit `a14d36b`)

- kernelgen 生成 + 契约重写(生成版 state 双写、while 循环、C 借用
  B stride 三处缺陷):flat 1D capped grid + 块级 div/mod,[64, N]
  tile,fp32 全程,溢出安全 softplus(max+log1p(exp(-|t|))),
  constexpr 旗标,int32,显式输出 cast。
- screening(gpu:/tmp/t36.n5vM5A,字节与 commit blob 逐项一致):
  unittest 4/4(含 D/z/bias/softplus 全组合 × 3 dtype × 6 形状、
  非连续、softplus 极值、2048 大 batch);bench 7/7,代理
  **7.51–8.34x**(fp32 全旗标 3.65x)。
- ZIP `s0-a14d36b`,SHA `744ab4ea66a23f42cc818f7201d48018aecc112788b18b82c943dc53296352f5`,单成员。

### 跨芯风险

- `tl.exp` 平台实证安全(T24/T29);`tl.log`(softplus 路径)与
  `tl.sigmoid` 未有昆仑实证,若昆仑崩溃则 A&S 类多项式替换;
- 归约仅 axis-1 sum(T21 先例,燧原可过)。

### S0 平台终态(sub 6356,2026-08-29 16:4x)

**8/8 全部数值失败**(逐芯同指纹):5 个 case 各 ~80% 元素超差。
- 失败签名:y 首元素精确、其余中等无规偏差(最大绝对差 1.19、
  最大相对差 42x)——与"平台传入 vllm 式 1-D [nheads] D/dt_bias、
  kernel 按 [nheads,dim] 索引越界读垃圾(H=1 时 p=0 恰好读对)"
  完全吻合;题面快照摘录的 reference 本身不可运行(dA 广播错),
  已证其与真实 harness 有出入。
- 代理无法复现(本地测试均按题面 2-D 形状构造)。

### E1:1-D 形状归一(commit `a5b6f04`)

- wrapper 将 1-D A/D/dt_bias 广播展开为 2-D(纯形状处理);
  测试参照同步归一;unittest 5/5(新增 1-D 变体用例)。
- ZIP `e1-a5b6f04`,SHA `f35e70c2e843820943ec6042f86cafc405c28ccec1cb575a36faaa39f2d9eee1`。额度 12→11/30。

### E1 平台终态(sub 6358,2026-08-29 16:5x)

**仍 8/8 数值失败,失败指纹与 S0 逐位相同**(51/64、同索引)——
1-D 形状假设证伪(D/dt_bias/A 本就是 2-D)。

- **同指纹两连败,按止损规则 T36 停止**(额度 10/30)。
- 未定位的真实语义差异线索:case 2 最大绝对差 1212(量级爆炸型,
  提示结构性差异——疑 state 布局转置、A 符号或 dt 变换分支),
  题面 reference 摘录不可运行、与真实 harness 有已证出入;
  代理无法复现(本地构造恒绿)。恢复条件:赛方澄清、公开真实
  reference,或他队通过后的结构证据。

## 契约校正与 generic 收敛

后续从官方 Mamba/vLLM 实现和平台差分重新锁定可执行契约：`A` 实际为
`[H,dim,N]`，更新式使用 `A[h,p,n]`。E1 的阶段性止损因此解除。

### E2:`A[p,n]` 中间假设(commit `06d957c`)

- 仅修正 scoring broadcast；screening
  `gpu:/tmp/flagos-selective_state_update.IScko8`，日志 SHA
  `8256cb221c6ba0873a95413254d27e24c16b64ebda6f510db98e80fc11084535`；
  release `gpu:/tmp/flagos-selective_state_update-release.lV9C99`，日志 SHA
  `35c20c4c5bbef52dae6e2204bcf4a164cd5a78a2f12099d05f81744ee0130f14`。
- ZIP `artifacts/competition/selective_state_update/e2-06d957c/selective_state_update.zip`，
  SHA `ffbe5c7a2767be46a55a75e04a1267bffe72619205eab3114f125904ffbaab60`；
  generic SHA `8f9e36d65fb52be1a15c0f34c14f696e342a4ff5b629503765987171deaafd20`。
- submission `6885`，file URL SHA
  `a99e3344d9b24d45fbefe3bda1c7bc442acdd4271dd8fc1eb17961a422fc6b11`：
  **0/8**，但 card_b 五例超差比例由
  `79.7/75.6/90.9/84.7/84.6%` 降至
  `53.1/65.9/85.6/82.0/83.3%`，证明已接近但仍缺 `h` 维。

### E3:`A[h,p,n]` generic 终版(commit `f143f65`)

- screening `gpu:/tmp/flagos-selective_state_update-e3.ahMcNG`，日志 SHA
  `d8feac490c75fa96b50dec8d010c942f65ecab4fb56a5a902738e058c03a75a1`；
  release `gpu:/tmp/flagos-selective_state_update-e3-release.0X4ZNI`，日志 SHA
  `2eeeecab4ec467759fb34f10e48b68c26b59e64af58bcf88411d8bd6a781c147`。
- ZIP `artifacts/competition/selective_state_update/e3-f143f65/selective_state_update.zip`，
  SHA `18c0ddafbf5dd697fa8aa6cae9c46be0304ec82b43d23e5067908e9572981563`；
  generic SHA `c1e1801200a3f56c7827714d86932defdd19ee40dab34d1300b4a29d1f7eac4c`，
  tests SHA `83a8715d3f22eac8a39b9c4df6d983046687686d43831e78a17b8c07f494a99b`。
- submission `6889`，file URL SHA
  `883e5f7f790376dfcb0c0f7f3a53b740bce7bedd141aec4872b1bd6e3f7de3b0`：
  **7/8**；天数 `3.993x`、沐曦 `9.084x`、燧原 `0.515x`、海光
  `8.440x`、华为 `3.639x`、card_a `6.434x`、card_b `8.2625x`；
  仅昆仑 `uni_sram PassManager::run failed`。此后冻结 generic 与 tests。

## 昆仑单变量 vendor 迭代

### E4:`BLOCK_P 64→4`(commit `0b12c69`)

- 新增自包含 `selective_state_update_kunlunxin.py`，仅缩小 P tile；vendor SHA
  `4df40c9aaa3332b7a629b4e12abb0538db73aa1294883bc48bd3701f39a2813b`。
  KernelGen 的有效建议与独立审计均支持该单变量，未采纳无证据 launch 参数。
- screening `gpu:/tmp/flagos-selective_state_update-e4.pwtBhH`，日志 SHA
  `5de024c15f933f10a5369d2a5c30151f28cfa0c78bd92cc088cfcfc75e5bfb30`；
  release `gpu:/tmp/flagos-selective_state_update-e4-release.dQjXO9`，日志 SHA
  `1547685d260d29ff354cd59a8805b1716072c221d3e2ac5d4abdaa12f976716b`。
- ZIP `artifacts/competition/selective_state_update/e4-0b12c69/selective_state_update.zip`，
  SHA `025877602ba15e915a64b3b7d9ec1ee48c706ac0ee6c1109d0d531267d300c65`；
  submission `6892`，file URL SHA
  `a122f9fee1bd4042d54ab95cdedab99c3a285ee3df1371c85e4671a137422e2f`。
- 终态仍 **7/8**；昆仑已选择 vendor，但五例均同一
  `uni_sram PassManager::run failed`。其余七芯 generic 正确。

### E5:direct 3-D grid(commit `77ee33e`)

- 保留 `BLOCK_P=4`，仅把扁平 grid/device div-mod 循环改为
  `(tiles_per_head,segment_batch,nheads)`，并在 host 端按 grid 上限分段；vendor SHA
  `37b8dc2d8daffa02dfd0ee4c4dcb45a35a7555546bf5da2b2bdeb55e0ef90843`。
- screening `gpu:/tmp/flagos-selective_state_update-e5d.PwvVnW`，日志 SHA
  `28c87d8f9048f63a034b90525468931175de2ec93007d22277f5ef6d6dc1c460`；
  release `gpu:/tmp/flagos-selective_state_update-e5-release.uJn1Qa`，日志 SHA
  `a059220824c05a305e921a4fc92905248bbdbc72245da4c92ed988b198b11614`。
- ZIP `artifacts/competition/selective_state_update/e5-77ee33e/selective_state_update.zip`，
  SHA `7c5ac08147b98f541bd59a304d304f0287e06a286e986b3c3b4a4069f94a6bae`；
  submission `6897`，file URL SHA
  `60db491f0325a36489a4a5ffe83fa05318396650230fc9a2e114ed0ae21b66cc`。
- 2026-08-30 18:43 实时复核：终态 **7/8 invalid_correctness**；天数
  `3.994x`、沐曦 `9.094x`、燧原 `0.517x`、海光 `8.450x`、华为
  `3.6555x`、card_a `6.424x`、card_b `8.257x`。昆仑选择 vendor，五例仍为
  `uni_sram PassManager::run failed`，故 direct-grid 假设证伪。

## E6 本地筛选检查点(未 commit、未打包、未提交)

目标是继续把 `[BLOCK_P,DSTATE]=[4,128]` 活跃矩阵按 `BLOCK_N=16` 分块，
冻结 E5 的 direct 3-D grid 与 host 分段。三种最小 lowering 均在 NVIDIA 代理筛选中
出现极稀疏但幅度很大的 `y` 错值，不能晋级：

- `tl.static_range + [4,16] y_lanes + 循环外归约`，source SHA
  `c33d7ed6a6e474d9252b04b9d6e81ed42aa5a11b49751768665fe0432b34d237`；
  `gpu:/tmp/flagos-selective_state_update-e6.xaKpON`，8 个 subcase 失败，日志 SHA
  `693a33d40e0f633eef87ea0cddec7d2ff91f33b139d4ef264f605dbcd987d5b3`。
- `range + [4,16] y_lanes + 循环外归约`，source SHA
  `84688844de1166a91be534c496638c6149278363d05f994a378b5a901b71a8e8`；
  `gpu:/tmp/flagos-selective_state_update-e6b.gCigF5`，10 个 subcase 失败，日志 SHA
  `9c951d61a6a2f608fe093ef05d0e335f908b89ff342f8f37376c6597e8f91fc3`。
- `range + 块内 tl.sum + [4] y_val`，source SHA
  `18afc837df95914d7acba5443445cbd9b0f63b1159ee7fdc058b582c7fff5b01`；
  `gpu:/tmp/flagos-selective_state_update-e6d.4LBKpX`，23 个 subcase 失败，日志 SHA
  `712e3059d68a62ce41335399e3b496e285c3530cf382ad73a29c1f3387658043`。

这些不是容差边缘的求和次序差：最多只有个位数元素超差，但最大误差可达数个单位，
属于 backend/compiler silent corruption 风险。当前工作树保留第三种未通过候选供后续
诊断；它没有 source commit、release、ZIP、preflight 或平台提交，不消耗额度。
平台快照：`used=25/30`、`remaining=5`。下一候选必须改用能避免 loop-carried
状态的资源收缩方式(优先两阶段 FP32 partial workspace)，并重新走完整门禁。

### E7 两阶段 vendor:开发中受阻(2026-08-31 05:3x CST)

- 设计已落盘(kunlunxin vendor,commit 未提交):stage1 [16,16] 切片
  写 state + FP32 partial_y 工作区,stage2 归约 + D/z;目标为
  uni_sram 编译失败的最小活跃矩阵假设;
- 本地代理:unittest 63 失败(softplus 极值 1/8 元素 0.108 abs
  差 + 全矩阵系统性失配)——**存在正确性 bug 待修**,初步排查
  非求和顺序(9e-4 相对差过大),嫌疑 partial 布局或 softplus
  下溢路径;
- GPU 代理主机失联(ping 100%,VPN 链路旧疾复发),迭代受阻;
  E6d 工作区改动已 stash 保全。
- 状态:候选不完整,**未提交未耗额度**;恢复条件=链路恢复后
  修 bug + 全量门禁。额度 20/30。

### E7:两阶段 vendor 破案与发射(2026-08-31 07:0x CST)

- **根因破案**:前日 63 失配的真凶是 **A 契约**——真实布局为
  `[nheads, dim, dstate]` 三维(E3/f143f65 夜间会话已改契约),我方
  vendor 仍按旧 `[H,N]` 索引;诊断脚本两侧同错导致假绿;
- 修复后两阶段 vendor(stage1 [16,16] 切片 + partial_y 工作区 /
  stage2 归约 + D/z)unittest **4/4 全绿**;screening 字节与 commit
  blob 逐项一致(gpu:/tmp/t36e7.S5CWNR);
- commit `7f7f2c1`,ZIP `9eeda4cb…`,2 成员;preflight 全过
  (额度 20/30),单次提交——**uni_sram 最小活跃矩阵假设的最终
  验证**。

### E7 昆仑终态 + E8 重载(sub 7108 → 2026-08-31 07:5x CST)

- 七芯全过(沐曦 9.09/card_b 8.30/海光 8.47/card_a 6.42/华为
  3.65/天数 3.99/燧原 0.51);
- 昆仑返回"**服务线程卡死自动恢复,请重新提交**"——**非
  uni_sram 编译失败**!两阶段最小活跃矩阵结构疑已通过编译,
  仅服务线程卡死(T28 E3 先例,平台明示可重投);
- E8 = E7 + 注释载体(commit `7414c69`,ZIP `b23663e6…`);
  preflight 全过(额度 19/30),单次提交,昆仑终态待回填。

### E8 终态(sub 7135):T36 封存于 7/8

- 昆仑返回 inductor 崩溃指纹(1830s + Aborted,failed_cases=0);
  与 E7 的服务线程卡死交替——**两种均为平台侧服务故障,非
  uni_sram 编译失败**(两阶段结构的编译假设未被证伪,但无法验证);
- T36 reference 含 einsum(matmul 族,6v6 崩溃相关性),与
  T26/T28/T31 同墙;三投(含夜间会话)终态 **7/8,候选
  `7414c69` 封存**;额度 18/30。
- 本题破案收获:A 三维契约 + 两阶段可编译结构,若平台修复即可
  一发转正。

## E10:split-matrix + direct 3D(2026-09-01 11:1x–11:3x CST)

状态:screening、source/test commit、commit-bound release 与 canonical ZIP
均通过;平台结果待实测。

### 根因矩阵与单变量

历史四象限中,full `[4,128]` + flat loop(E4)和 full + direct 3D(E5)
均报 `uni_sram`;split `[16,16]` + 多层 device loop(E7–E9)则落入服务线程
卡死/compile-worker 1830s crash。唯一未试组合是 **split `[16,16]` + direct
3D**。

- 冻结 E7 的 `_BLOCK_P=16`、`_N_SLICE=16`、FP32 partial workspace、
  softplus、状态更新数学和 stage2 归约顺序。
- stage1 的 N slice 改为 host 展开;每次 launch 只处理一个 slice,
  direct grid 为 `(P tile,batch,head)`,kernel 内无 tile/slice 循环。
- stage2 同样改为 direct `(P tile,batch,head)`,删除 row/P 两层循环。
- host 按 `tiles_per_head * nheads` 计算 batch chunk,保证每次 3D launch
  总 workgroups 不超过 65535;host offsets 标记 `do_not_specialize`,避免
  Triton 3.7 编译扇出。
- 测试从固定 generic 改为 `load_operator_modules`,首次把 Kunlun vendor
  纳入正式全矩阵;新增 N=65 尾 slice 与 B=70000 grid-fold。

### 构建身份

| 项目 | 值 |
| --- | --- |
| source / verification commit | `e3e40d7c785793d6042d9fbd441c32d7fb480c02` |
| generic SHA-256 | `c1e1801200a3f56c7827714d86932defdd19ee40dab34d1300b4a29d1f7eac4c` |
| Kunlun SHA-256 | `14e8c23e198409416bc0dc734172934feef8fbe6e9ef1ed4bc5ace944f73112a` |
| test SHA-256 | `a6cc8c509960f82c69e4124eef8c6b927879ebc789c044ec0fd75fbde638aaf0` |
| `_op_variants.py` SHA-256 | `cdc5fe3e4cb5a85976f0a3414cd194bb53c79f6f2830be01f685f996b97ca0d7` |

### Screening 与代理性能

- 目录:`gpu-et:/tmp/flagos-selective_state_update-e10-final-screening.WhuCWb`;
  冻结 payload SHA-256
  `b403911d9d6338194535e039624ed749805186ff7d3c664f70d271f0b97eeb72`。
- pycompile、Black79、isort80、flake8、前后哈希均通过;完整 generic +
  Kunlun variants unittest **5/5**,22.791s;gate log SHA-256
  `b2a2b55cd08d204cc2df32be2ba960a653b092a85b4194fc9746e917efad50f2`。
- 覆盖三 dtype、flags 全组合、N=1/64/65/128、P 尾块、softplus 极值、
  noncontiguous、B=2048 大 batch、B=70000 grid-fold 和输入 state 不变性。
- 五轮 AB/BA 代理中,Kunlun vendor 的 fp16 full / bf16 large / fp32 tail
  speedup 分别为 **1.4340x / 5.0208x / 1.1986x**;五轮最小值仍为
  1.3860x / 4.9718x / 1.1419x。benchmark log SHA-256
  `c37105b9dd91b9f92d171d4a9e4f4101e29f5ca79d6340991fef81e9a0d6e1a1`;
  peak allocated/reserved 1,043,333,120 / 1,218,445,312 bytes,无 OOM/竞争进程。

### Commit-bound release 与不可变 ZIP

- release 目录:`gpu-et:/tmp/flagos-selective_state_update-e10-release.wEEoGe`
  (0700,保留);PID/PGID/SID `235437`;五文件均从 commit Git objects 导出。
- 完整 variants unittest 5/5,22.884s;静态门禁及前后哈希全过;release log
  SHA-256 `963397d7b6f447968abfc060de25c81fc675e927683705677b1b218af6967ae3`。
- canonical ZIP:
  `artifacts/competition/selective_state_update/e10-e3e40d7/selective_state_update.zip`,
  13543 bytes,SHA-256
  `9ae684ffe320f30b7ecb146d319374125506f16e1349c55b9cb8266dcbee8cb7`;
  实际构建与 `--verify-existing` 一致。成员为 generic + `_kunlunxin`,
  member SHA 与上述 commit blob 一致。

### 实时登顶账(提交前)

- 19 队/147 投中仅 2 队有效;榜首 c2flow `8.4960x`,第二 Fields
  `6.4056875x`;SoulCoder 因 7/8 尚未排名。E8 七芯和 `40.4035`。
- 第一门是昆仑转正;若仅过 0.1x,投影均值约 5.063x、成为第 3。若七芯
  冻结,登顶需昆仑 `>27.5645x`;转正后按收益优先优化 card_a、沐曦、
  燧原、华为、card_b、天数,保护已领先榜首的海光路径。

### E10 平台结果(sub 7584,2026-09-01 11:36 CST)

- preflight 精确绑定 commit/test/release/two members/ZIP SHA,实时额度
  25/30、时间窗和最小间隔均通过;一次性提交后额度 24/30。平台文件 URL
  SHA-256 `cd210a1c8b17a459a0baace2280a812c1a30d238de0aec1886291479241f588f`;
  远端 ZIP 回读因未配置受信主机为 `unavailable`,未重试。
- 七芯 generic 全过:天数 3.995x、沐曦 9.104x、燧原 0.5095x、海光
  8.469x、华为 3.668x、card_a 6.4185x、card_b 8.248x;七芯和
  `40.412x`。
- 昆仑选择 `_kunlunxin`,执行 **7274ms** 后五例全部明确失败:
  `_ssu_stage1_kernel`,grid `(1,1,4)`,`num_stages=1`,错误为
  `uni_sram PassManager::run failed`;不再出现服务卡死或 1830s
  compile-worker crash。
- 根因收敛:direct grid/device-loop removal 已修复崩溃族,但 stage1
  `[BLOCK_P,N_SLICE]=[16,16]` 活跃矩阵仍超过 XPU lowering 能力。
  下一候选只改 `_BLOCK_P 16→8`;若同指纹,再只改 `N_SLICE 16→8`。

## E11:`BLOCK_P 16→8`(2026-09-01 11:3x–11:4x CST)

- 唯一源码变量:`_BLOCK_P=16→8`;direct 3D、`N_SLICE=16`、workspace、
  数学、归约、generic 与 tests 全冻结。source/verification commit
  `6e0bc65f8601b011110a6ed20ea4f7847c09cb20`;Kunlun SHA-256
  `279f4bfc76201ec28584636a84ccfba24be593fc5a07d1c98fb32d93c1b59c7d`,
  test SHA 仍为 `a6cc8c509960f82c69e4124eef8c6b927879ebc789c044ec0fd75fbde638aaf0`。
- screening:`gpu-et:/tmp/flagos-selective_state_update-e11-screening.b53hUf`;
  payload SHA `4540cdaf9d59a0378d3ddd002ba10962e7d6580b8ce1f3465b895989d32021d8`;
  static + variants unittest **5/5**,22.579s;gate log SHA
  `3932da89c68266f24a47daf70f5a9164ca9f09495aa09d756dd927eb8684ae59`。
- vendor 代理 fp16 full / bf16 large / fp32 tail 为 **1.4065x / 5.5724x /
  1.18395x**;相对 E10 分别约 +0.12% / +10.11% / -0.94%,无 >1%
  回退;benchmark log SHA
  `58dba502a7b00d95d6a4aaa6a3581c491f4f4e127458c427aa7de5ce2ef75b15`。
- commit-bound release:
  `gpu-et:/tmp/flagos-selective_state_update-e11-release.QANOf5`;PID/PGID/SID
  `236406`;static + variants unittest 5/5,23.276s;manifest 前后一致;release log
  SHA `5eaece03ffc89a5fef38a7da44de665c72f667fa7b10ba59dd9a41df713c3da0`。
- canonical ZIP:
  `artifacts/competition/selective_state_update/e11-6e0bc65/selective_state_update.zip`,
  13542 bytes,SHA-256
  `e70ec56985d01a006076063faa249594e72873117893440908293e5ffea19d81`;
  actual/`--verify-existing` 一致,成员 generic + `_kunlunxin`。

### E11 平台结果(sub 7600,2026-09-01 11:47 CST)

- preflight 全过后一次性提交;平台文件 URL SHA-256
  `5a66ecc738432ffe657947b7d7ee7f5f6621a9aac0b3010965de2beb796c2218`;
  提交后额度 23/30,远端 ZIP 回读 `unavailable`,未重试。
- 七芯 generic 全过;昆仑执行 **7219ms**,五例均在 stage1、grid
  `(2,1,4)`,`num_stages=1` 返回相同 `uni_sram PassManager::run failed`。
- `BLOCK_P 16→8` 未跨过编译阈值;下一候选只改 `N_SLICE 16→8`,形成
  stage1 8×8 活跃矩阵。在线源码复核后撤销该计划:官方已记录 8×8/16×16
  方块 tile 的 Legalize verifier 失败,不能把 8×8 当作可靠的降 SRAM 手段。

## `uni_sram` 在线根因校正(2026-09-01 12:0x CST)

- FlagTree XPU backend 将 `make_ttxir` 的整条非 SDNN pass pipeline 包在同一个
  `try/except` 中,任意 pass 异常都被重写成
  `OutOfResources(0,0,"uni_sram ...")`;因此 E10/E11 的
  `required=0/limit=0` **不是 SRAM 容量测量值**。官方源码见
  [compiler.py L271-L367](https://github.com/flagos-ai/FlagTree/blob/2e6258114a79f14440e6f1134e5daca67d332925/third_party/xpu/backend/compiler.py#L271-L367)。
- 当前 stage1 同时含二维 masked load/broadcast、`exp`/可选 `log` 和 axis-1
  reduce。官方验证报告记录同一 `TritonXPUVectorize` pass 会令复杂 masked
  kernel 编译失败,关闭 vectorization 后可编译运行;见
  [validation L239-L247](https://github.com/flagos-ai/FlagTree/blob/2e6258114a79f14440e6f1134e5daca67d332925/third_party/xpu/docs/triton-3.6-validation.md#L239-L247)。
  FlagGems 的 Kunlun `exp`、`log1p`、`sigmoid`、`logsumexp` 也使用
  `isCloseVectorization=True`;其中
  [logsumexp](https://github.com/flagos-ai/FlagGems/blob/2822a8067ca3f1f6278a58599fd1c4b88bb5bac5/src/flag_gems/runtime/backend/_kunlunxin/ops/logsumexp.py#L107-L133)
  与本题同为超越函数 + reduction。
- 默认 `buffer_size_limit=512` bytes,FP32 折算为 128 elements,恰好等于当前
  `8×16`;它只说明 buffer 边界,不能定位是哪一个 pass 失败。官方计算见
  [triton_xpu.cc L513-L543](https://github.com/flagos-ai/FlagTree/blob/2e6258114a79f14440e6f1134e5daca67d332925/third_party/xpu/triton_xpu.cc#L513-L543)。
- 预注册后续单变量顺序:E12 只关 Vectorize;若仍同指纹则关闭该轴,依次试
  `isCloseCoreTiling=True`、`buffer_size_limit=2048`、
  `isCloseUnrollControl=True`;全部失败才拆分 state update 与 C-reduce。
  不再扫描无效的 `num_warps/num_ctas/num_stages`,也不提交方块 8×8。

## E12:关闭 stage1 Vectorize pass(commit `370ca66`)

- 唯一执行变量:保持 E11 的 `_BLOCK_P=8`、`_N_SLICE=16` 和全部数学/布局,
  仅给 stage1 launch 传 `isCloseVectorization=True`。为让同一 vendor 在 CUDA
  代理可执行,沿用官方
  [mv.py constexpr 模式](https://github.com/flagos-ai/FlagGems/blob/2822a8067ca3f1f6278a58599fd1c4b88bb5bac5/src/flag_gems/runtime/backend/_kunlunxin/ops/mv.py#L54-L110):
  同名 unused `tl.constexpr` 既是 XPU backend option,也是其他 backend 的合法
  kernel 参数,无设备判断或 fallback。
- source/verification commit
  `370ca66cfb0319f4eca3f999113d07272269735d`;generic SHA-256
  `c1e1801200a3f56c7827714d86932defdd19ee40dab34d1300b4a29d1f7eac4c`;
  Kunlun SHA-256
  `fde957889fa2e889fb06c3934568efd0071967a821d1105d681679b7c52719ea`;
  test SHA-256
  `a6cc8c509960f82c69e4124eef8c6b927879ebc789c044ec0fd75fbde638aaf0`。
- screening:`gpu-et:/tmp/flagos-selective_state_update-e12-screening.FB9Upc`;
  PID/PGID/SID `237166`;static + 完整 generic/Kunlun variants **5/5 PASS**,
  23.279s;gate log SHA-256
  `c6d634707717c8fa7e27980f61babdd009b026a3d2e3a32c77b05e339e5edf5e`。
  CUDA JIT/执行确认同名 kwarg 不会报 unknown parameter;因变量仅影响 XPU
  lowering,不重复无区分力的 CUDA benchmark。
- commit-bound release:
  `gpu-et:/tmp/flagos-selective_state_update-e12-release.qE8m0Y`;PID/PGID/SID
  `237638`;Git-object 五文件前后 manifest 完全一致,static + variants
  **5/5 PASS**,23.227s;release log SHA-256
  `b8735bc66d05663977b2bc57feee91263a8c2916966a2fe064c7a873bcf430ce`。
- canonical ZIP:
  `artifacts/competition/selective_state_update/e12-370ca66/selective_state_update.zip`,
  13625 bytes,SHA-256
  `09e2080f1295134689cc86a044deb3d56a6353b228c17357e5ad0291451067df`;
  `created`/`--verify-existing` 一致,仅 generic + `_kunlunxin` 两成员。
- 2026-09-01 12:05:07 CST 只读状态:Task competing/can_submit,额度
  `23/30`,最小间隔已满足。晋级门为昆仑五例全过;其余七芯成员逐字节冻结。
  若仍为 stage1 `uni_sram PassManager::run failed`,不重投 E12,直接转
  CoreTiling 单变量。

### E12 平台结果(sub 7618,2026-09-01 12:11 CST)

- preflight 精确绑定 source/test/release/two members/ZIP SHA,额度 `23/30`;
  一次性提交成功后为 `22/30`。平台文件 URL SHA-256
  `0bb00c9d5643b8f9735d77bf85e40150480089c79dcb78e0468443c65a8109e6`;
  远端 ZIP 回读因未配置可信 hostname 为 `unavailable`,未重试。
- 终态仍 **7/8 invalid_correctness**:天数 `3.885x`、沐曦 `9.104x`、
  燧原 `0.5115x`、海光 `8.4725x`、华为 `3.649x`、card_a `6.416x`、
  card_b `8.269x`;七芯 generic 均通过。
- 昆仑选择 `_kunlunxin`,执行 **7319ms**,五例仍在 stage1、grid
  `(2,1,4)` 返回同一 `uni_sram PassManager::run failed`。Vectorize workaround
  没有跨过失败点,不重投同候选。
- 下一步先核对平台旧版 `default_run` 对“同名 constexpr + backend option”的绑定:
  若 E12 flag 未进入 XPUOptions,改为真实 launch metadata 重新归因;若已进入,按
  预注册顺序转 `isCloseCoreTiling=True`。不同时叠加 buffer/unroll。

## E13:关闭 stage1 CoreTiling pass(commit `1443966`)

- 平台 traceback 三处行号精确匹配 FlagTree
  [`7b0370a4`](https://github.com/flagos-ai/FlagTree/commit/7b0370a4976c6fcdbab89420bf53728472d75a9e):
  该版 `default_run` 对完整 launch kwargs 调 `backend.parse_options`,再按
  `XPUOptions` 字段展开 metadata。因此 E12 flag 已真实关闭 Vectorize,失败点在
  其他 pass,不是 dummy constexpr 被吞或旧 cache 命中。
- 唯一执行变量:从 E11 P8×N16 精确分叉,不携带 E12 flag,只把 stage1 metadata
  设为 `isCloseCoreTiling=True`。官方 `min_dim` 对同一
  `uni_sram / PassManager::run failed` 用该选项解决全部 shape/dtype;
  [CoreTiling 源码](https://github.com/flagos-ai/FlagTree/blob/2e6258114a79f14440e6f1134e5daca67d332925/third_party/xpu/lib/Dialect/TritonXPU/Transforms/CoreTiling.cpp#L144-L204)
  也直接处理本题的 rank-2 axis-1 reduce、broadcast/expand-dims/store 编码。
- source/verification commit
  `1443966a146ad3c8f6d2682ade9fd407195b70b9`;Kunlun SHA-256
  `4653494410faf0b7d4060429a15079d7e2177c77795a27d5cae64198144733f3`;
  generic/test 仍为 `c1e180...` / `a6cc8...`。
- screening:`gpu-et:/tmp/flagos-selective_state_update-e13.fTmoFH`,PID/PGID/SID
  `238150`;static + variants **5/5 PASS**,15.813s;log SHA-256
  `88d4fd0d56aad742e5fb748afae8e9da867df26218421aa5cb9fa2d07dfe3e1d`。
- commit-bound release:
  `gpu-et:/tmp/flagos-selective_state_update-e13-release.bGlc8Y`,PID/PGID/SID
  `238485`;Git-object manifest 前后一致,static + variants **5/5 PASS**;
  release log SHA-256
  `7bb33975fd9e931dfef98fabf32623026aff82e7e36c2974d89dc0194a6f1aa1`。
- canonical ZIP:
  `artifacts/competition/selective_state_update/e13-1443966/selective_state_update.zip`,
  13619 bytes,SHA-256
  `f9596117d09650c1200d38f6c3f7cf5c9cd189a7189edbb48500006372151502`;
  actual/`--verify-existing` 一致,仅 generic + `_kunlunxin` 两成员。
- 晋级门仍为昆仑五例全过;若相同 stage1 pass 指纹失败,关闭 CoreTiling 轴,
  下一独立变量为 `isCloseUnrollControl=True`,不叠加 buffer。

### E13 平台结果(sub 7627,2026-09-01 12:21 CST):编译墙已破

- preflight 全过后一次性提交;平台文件 URL SHA-256
  `7df7cdf7f801189aa36633a04429d8c0566c4f2d20b8fa2972f83c4bc8a9e585`;
  提交后额度 `21/30`,远端 ZIP 回读 `unavailable`,未重试。
- 七芯 generic 全过:天数 `3.8835x`、沐曦 `9.107x`、燧原 `0.5165x`、
  海光 `8.473x`、华为 `3.656x`、card_a `6.775x`、card_b `8.192x`。
- **关键进展**:昆仑不再出现 `uni_sram`。vendor 完成编译和执行(18818ms),
  case 0/1 通过;仅 case 2/3/4 数值失败。由输出元素数和最大索引可锁定三者
  `y` shape 为 `[3,16,128]`、`[64,32,128]`、`[256,32,128]`,超差比例
  `95.0%/95.9%/96.0%`;前两例 P<=64 已过。CoreTiling 是编译阻塞 pass,
  关闭后暴露 P=128/grid.x=16 相关 lowering 错误。
- stop gate 修正:不撤销已证明必要的 CoreTiling flag,也不按旧计划直接替换成
  UnrollControl(会重新引入编译失败)。下一候选以 E13 为基线,只改变一个
  P/grid 或后置 pass 变量;先用源码证据区分 grid.x=16、双 store 和 reduce
  live-range,不盲叠 flags。

## E14:同时关闭 stage1 CoreTiling 与 Vectorize(commit `bdbb868`)

- 唯一执行变量:保留 E13 的 P8×N16 和已证明必要的
  `isCloseCoreTiling=True`,只新增 `isCloseVectorization=True`。E12 在更早的
  CoreTiling pass 即失败,没有执行到 Vectorize,因此不能证伪双关闭组合。
- 官方 `rwkv_ka_fusion` 与本题 stage1 同为二维 tile、axis-1 reduce、broadcast
  和多 store;其注释明确记录默认 XPU store vectorizer 会丢 lane,关闭后恢复
  正确且几乎无性能损失([kernel](https://github.com/flagos-ai/FlagGems/blob/d8b500b368343ac5f5ff4e01b508d9e8e03ad5c5/src/flag_gems/runtime/backend/_kunlunxin/fused/rwkv_ka_fusion.py#L61-L71),
  [workaround](https://github.com/flagos-ai/FlagGems/blob/d8b500b368343ac5f5ff4e01b508d9e8e03ad5c5/src/flag_gems/runtime/backend/_kunlunxin/fused/rwkv_ka_fusion.py#L98-L118))。
  官方 [LayerNorm](https://github.com/flagos-ai/FlagGems/blob/d8b500b368343ac5f5ff4e01b508d9e8e03ad5c5/src/flag_gems/runtime/backend/_kunlunxin/ops/layernorm.py#L566-L582)
  与 [InstanceNorm](https://github.com/flagos-ai/FlagGems/blob/d8b500b368343ac5f5ff4e01b508d9e8e03ad5c5/src/flag_gems/runtime/backend/_kunlunxin/ops/instance_norm.py#L625-L644)
  也组合使用这些关闭项。
- source/verification commit
  `bdbb868186fef47c387a4dc026af5ed188810f89`;Kunlun SHA-256
  `816448b987a7c38e0a72e635690f9fd1c864a56c9b0f06bd2201c915d14b468f`;
  generic/test 仍为 `c1e180...` / `a6cc8...`。
- screening:`gpu-et:/tmp/flagos-selective_state_update-e14-screening.y7eH67`,
  PID/PGID/SID `238807`;static + 双 constexpr CUDA JIT + variants **5/5 PASS**,
  8.356s;log SHA-256
  `dce5e4d5d86192d08c04c18cf2b3e15483a394407010fcf62f58d57a543de2e9`。
- commit-bound release:
  `gpu-et:/tmp/flagos-selective_state_update-e14-release.GozStj`,PID/PGID/SID
  `239156`;Git-object manifest 前后一致,static + 双 constexpr JIT + variants
  **5/5 PASS**;release log SHA-256
  `389928e9a685ecfc2f897f36de68f94dd00570bb2bf0b1f46ec7986296a6d917`。
- canonical ZIP:
  `artifacts/competition/selective_state_update/e14-bdbb868/selective_state_update.zip`,
  13702 bytes,SHA-256
  `70392da68c79a1c3c65d1b0f41496f729d45b02e4030dec5a573c03d53e62b90`;
  actual/`--verify-existing` 一致,仅 generic + `_kunlunxin` 两成员。
- 2026-09-01 12:31 CST 实时榜单:团队仍为 7/8、无有效排名;榜首 `c2flow`
  八芯均值 `8.4960x`。当前七芯均值 `5.8004x`;本轮门槛先让昆仑正确且
  `>=0.1x`,形成首个有效八芯成绩。若 E14 仍为相同数值指纹,E15 只新增
  `isCloseUnrollControl=True`;再失败则停止扫 flag,拆分二维 state store 与
  C-reduce/partial store。

### E14 平台结果(sub 7635,2026-09-01 12:35 CST):与 E13 同指纹

- preflight 全过后一次性提交;平台文件 URL SHA-256
  `35281ddae31139a0f1d1bf942d6c7656a6abad8cfac1b952be6c9970b321a1cd`;
  提交后额度 `20/30`,远端 ZIP 回读 `unavailable`,未重试。
- 七芯 generic 全过:天数 `3.886x`、沐曦 `9.1095x`、燧原 `0.5165x`、
  海光 `8.466x`、华为 `3.67x`、card_a `6.521x`、card_b `8.301x`。
- 昆仑编译执行完成(18468ms),case 0/1 通过,case 2/3/4 失败。三例失配数
  `5834/251361/1006386`、最大绝对误差 `44332/340/278` 及最大误差索引
  `(1,12,126)/(22,30,84)/(151,51,28)` 均与 E13 完全相同。
- 结论:在平台当前编译器上,额外关闭 Vectorize 未改变错误路径。E15 保留
  CoreTiling + Vectorize 关闭,只新增官方 norm kernel 同用的
  `isCloseUnrollControl=True`;若仍同指纹,停止继续扫描 metadata flag。

## E15:再关闭 stage1 UnrollControl(commit `4d8f796`)

- 唯一执行变量:保留 P8×N16 及 CoreTiling/Vectorize 两关闭项,只新增
  `isCloseUnrollControl=True`;这是官方 LayerNorm/InstanceNorm 对二维归约和
  多 store 使用的完整三开关组合。若平台仍为 E13 指纹,metadata flag 轴封存。
- source/verification commit
  `4d8f796ae229e79e7d350c5eae4eb612eb3e8699`;Kunlun SHA-256
  `cb23962e1180feafd796d673636fb28b76bc39d3cd25e4625c3f2cb2e5ebbe04`;
  generic/test 仍为 `c1e180...` / `a6cc8...`。
- screening:`gpu-et:/tmp/flagos-selective_state_update-e15-screening.UqsrA6`,
  PID/PGID/SID `239390`;static + 三 constexpr CUDA JIT + variants **5/5 PASS**,
  8.313s;log SHA-256
  `6aa33d6392982fa7596fc6e53b91131c6e2b869aa073be78dc176283d318cd2b`。
- commit-bound release:
  `gpu-et:/tmp/flagos-selective_state_update-e15-release.eWYVJg`,PID/PGID/SID
  `239731`;Git-object manifest 前后一致,static + 三 constexpr JIT + variants
  **5/5 PASS**;release log SHA-256
  `759f4b50e3aec1d5c924ab4cb7df5aac9d46d071aca621c283490e379f1f6460`。
- canonical ZIP:
  `artifacts/competition/selective_state_update/e15-4d8f796/selective_state_update.zip`,
  13785 bytes,SHA-256
  `3668676581cb45dbf15c24e43e3f24b43bd865b5894f84a5e161a876a6f509b8`;
  actual/`--verify-existing` 一致,仅 generic + `_kunlunxin` 两成员。

### E15 平台结果(sub 7642,2026-09-01 12:40 CST):仍与 E13 同指纹

- preflight 全过后一次性提交;平台文件 URL SHA-256
  `d1b78bfa1460db92ca9997095661ec89cc3dc94168f45db438aadf4c63678a78`;
  提交后额度 `19/30`,远端 ZIP 回读 `unavailable`,未重试。
- 七芯 generic 全过:天数 `3.8825x`、沐曦 `9.0875x`、燧原 `0.516x`、
  海光 `8.462x`、华为 `3.6855x`、card_a `6.412x`、card_b `8.2005x`。
- 昆仑编译执行完成(18494ms),case 0/1 通过;case 2/3/4 的失配数、最大
  绝对误差和索引继续与 E13/E14 完全相同。stage1 metadata flag 轴封存。
- 新证据修正后续顺序:平台先断言 `y`,不能证明 `new_state` 同时错误;而 E13-E15
  的所有关闭项只传给 stage1。stage2 仍是 `[8,8]` axis-1 reduce + P 向量 store,
  未带任何 XPU workaround。因此 E16 只关闭 stage2 Vectorize;若仍同指纹再拆分
  state update 与 y reduction。
