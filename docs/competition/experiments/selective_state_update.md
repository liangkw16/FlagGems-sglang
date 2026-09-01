# Task 36 `selective_state_update` 实验记录

```current
task: 36
operator: selective_state_update
batch: 3
validity: invalid_threshold
platform: 8/8
team_best_stage: e22(correctness)
team_best_commit: f1a12f7
team_best_speedup: 5.1200625x;昆仑0.0025x
blockers: e28昆仑回退至0.008x,exact-2D轴关闭;有效门仍差7.41倍
sealed: no
next: 从e27分叉P-major源码级N-unroll8单变量
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
  case 0/1 通过;仅 case 2/3/4 数值失败,超差比例 `95.0%/95.9%/96.0%`。
  CoreTiling 是编译阻塞 pass,关闭后暴露后置 lowering 错误。旧版曾把三例
  `y` shape 锁定为 `[3,16,128]`、`[64,32,128]`、`[256,32,128]`;该结论撤销:
  case 4 最大误差索引 `(151,51,28)` 与第二维 32 矛盾。元素数和索引只能给
  shape 约束,不能证明错误仅与 P=128/grid.x=16 有关。
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

## E16:关闭 stage2 Vectorize(commit `b280817`)

- 唯一执行变量:stage1、P8×N16 和三关闭项完全不变;只给 stage2 的 `[8,8]`
  axis-1 reduce + 行向量 store 传 `isCloseVectorization=True`。官方 Kunlun
  [logsumexp kernel](https://github.com/flagos-ai/FlagGems/blob/d8b500b368343ac5f5ff4e01b508d9e8e03ad5c5/src/flag_gems/runtime/backend/_kunlunxin/ops/logsumexp.py#L43-L69)
  是同构路径,其 [launch](https://github.com/flagos-ai/FlagGems/blob/d8b500b368343ac5f5ff4e01b508d9e8e03ad5c5/src/flag_gems/runtime/backend/_kunlunxin/ops/logsumexp.py#L107-L121)
  明确关闭 Vectorize。
- source/verification commit
  `b2808170a3d4e9684646f554d9b1ff5bb2b1671c`;Kunlun SHA-256
  `9355d2af3701ca060bae92d036696d51246706938920ca9faf04d5a4857af7a7`;
  generic/test 仍为 `c1e180...` / `a6cc8...`。
- screening:`gpu-et:/tmp/flagos-selective_state_update-e16-screening.SkkBrZ`,
  PID/PGID/SID `239963`;static + stage1/stage2 独立 constexpr JIT + variants
  **5/5 PASS**,5.730s;log SHA-256
  `fe11997bb2bab2e8185253b7fb31f39c1ae60bc31673feb9e245f132f80c808d`。
- commit-bound release:
  `gpu-et:/tmp/flagos-selective_state_update-e16-release.wXYeRh`,PID/PGID/SID
  `240277`;Git-object manifest 前后一致,static + 双 JIT probe + variants
  **5/5 PASS**;release log SHA-256
  `feb8d87dea338f2301d25c9fd9ddee0c6c4b54b0099b1aef47756f2f0bdc5515`。
- canonical ZIP:
  `artifacts/competition/selective_state_update/e16-b280817/selective_state_update.zip`,
  13864 bytes,SHA-256
  `ab59679823740df8196b105f88267fa3af3d0d4d818c8de2cac78ed8aa276b47`;
  actual/`--verify-existing` 一致,仅 generic + `_kunlunxin` 两成员。

### E16 平台结果(sub 7647,2026-09-01 12:46 CST):仍与 E13 同指纹

- preflight 全过后一次性提交;平台文件 URL SHA-256
  `10032b10a804117833bfbd8437ec0c03444c408315c338df4250eb395c53034f`;
  提交后额度 `18/30`,远端 ZIP 回读 `unavailable`,未重试。
- 七芯 generic 全过:天数 `3.884x`、沐曦 `9.126x`、燧原 `0.517x`、
  海光 `8.4705x`、华为 `3.6575x`、card_a `6.403x`、card_b `8.315x`。
- 昆仑编译执行完成(17594ms),case 0/1 通过;case 2/3/4 的失配数、最大
  绝对误差和索引仍与 E13-E15 完全相同。单独关闭 stage2 Vectorize 被证伪。
- E12 编译 metadata 与后续输出共同锁定五例均为 BF16,`y` shape/grid 依次为
  `[1,4,16]/(2,1,4)`、`[5,8,64]/(8,5,8)`、
  `[3,16,128]/(16,3,16)`、`[64,32,128]/(16,64,32)`、
  `[256,64,64]/(8,127,64)+batch chunks`。case 1/4 同为 P64/grid.x=8
  却一过一败,因此 P128/grid.x=16 假设正式否定;更可能与 B×H、总 program
  数、dstate/num_slices 或多行归约 specialization 相关。
- E17 保留该关闭项,只新增 stage2 `isCloseCoreTiling=True`;CoreTiling 在
  Vectorize 之前处理 reduction/broadcast 编码,且已在 stage1 证明能实质改变
  编译行为。若仍同指纹,再比较 stage2 Unroll 与 1D per-output reduction。

## E17:再关闭 stage2 CoreTiling(commit `4f21397`)

- 唯一执行变量:保留 E16 的 stage2 Vectorize 关闭项,只新增
  `isCloseCoreTiling=True`;stage1、P8×N16 均不变。
- source/verification commit
  `4f21397987fc8bddc8aadaffa930a9b92b330a25`;Kunlun SHA-256
  `8e51ceec2f0009dffac5ab31aaf6ffe1faa4c163a5706c80f70b0f78d8df94e9`;
  generic/test 仍为 `c1e180...` / `a6cc8...`。
- screening:`gpu-et:/tmp/flagos-selective_state_update-e17-screening.t62RhI`,
  PID/PGID/SID `240538`;static + stage1/stage2 独立 constexpr JIT + variants
  **5/5 PASS**,5.731s;log SHA-256
  `fd8c7c2e1d5f92d4aaa53dbd24718cb7bd3a4abba19662c467cb15a02152f9f3`。
- commit-bound release:
  `gpu-et:/tmp/flagos-selective_state_update-e17-release.QN1R4n`,PID/PGID/SID
  `240856`;Git-object manifest 前后一致,static + 双 JIT probe + variants
  **5/5 PASS**;release log SHA-256
  `064cb9e8edaa8b51e286196a39ca921016eb8cf50e3e3e31bb007970fc0f6d8a`。
- canonical ZIP:
  `artifacts/competition/selective_state_update/e17-4f21397/selective_state_update.zip`,
  13937 bytes,SHA-256
  `c87aace83674c1fecf1ce93659b2ece3a290d6fd85ece04d69496e32ad780c64`;
  actual/`--verify-existing` 一致,仅 generic + `_kunlunxin` 两成员。

### E17 平台结果(sub 7654,2026-09-01 12:54 CST):仍与 E13 同指纹

- preflight 全过后一次性提交;平台文件 URL SHA-256
  `594c67539edf164d1329b4a3dd861d64bebd46c3b7b38f998161dc5b561c1f93`;
  提交后额度 `17/30`,远端 ZIP 回读 `unavailable`,未重试;首次 watch 遇网络
  timeout,随后只读 status 取得终态。
- 七芯 generic 全过:天数 `3.8835x`、沐曦 `9.1155x`、燧原 `0.5155x`、
  海光 `8.4605x`、华为 `3.6295x`、card_a `6.772x`、card_b `8.3145x`。
- 昆仑编译执行完成(18005ms),case 0/1 通过;case 2/3/4 指纹继续与 E13
  完全相同。stage2 CoreTiling/Vectorize metadata 轴封存。
- E18 冻结 stage1 和 `partial_y` 布局,只把 stage2 改为每个输出一个 program:
  连续载入最多 8 个 FP32 partial、1D reduce、标量 epilogue/store。全局输出按
  65535 分块;若 y 转正则旧二维 stage2 lowering 为根因,若仍同指纹则强指向
  stage1 partial 生成。

## E18:stage2 改为 1D 一输出一 program(commit `35ce353`)

- 唯一执行变量:stage1 及 `partial_y[(row*P+p)*num_slices+s]` 布局逐字节
  不变;stage2 从 `[8,num_slices]` 二维 tile 改为每个 program 处理一个全局
  `(b,h,p)` 输出。连续载入最多 8 个 FP32 partial,做 1D reduction 与标量
  D/z epilogue/store;host 按 65535 个输出切 launch,无 device loop 和额外内存。
- 官方 Kunlun [logsumexp 1D per-row kernel](https://github.com/flagos-ai/FlagGems/blob/d8b500b368343ac5f5ff4e01b508d9e8e03ad5c5/src/flag_gems/runtime/backend/_kunlunxin/ops/logsumexp.py#L72-L104)
  和 [sum 1D reduction](https://github.com/flagos-ai/FlagGems/blob/d8b500b368343ac5f5ff4e01b508d9e8e03ad5c5/src/flag_gems/runtime/backend/_kunlunxin/ops/sum.py#L32-L73)
  均采用同类一维归约/标量 store;本题向量长度远低于其已验证上限。
- source/verification commit
  `35ce3536bd021b7f46e6f84875686ac48686517c`;Kunlun SHA-256
  `463ac58517046614e3a1131f4408fcb4510b0e9f7a8adde0d927ba51521cbfdf`;
  generic/test 仍为 `c1e180...` / `a6cc8...`。
- screening:`gpu-et:/tmp/flagos-selective_state_update-e18-screening.T28kj7`,
  PID/PGID/SID `241122`;static + stage1 probe + 非零 output offset probe + variants
  **5/5 PASS**,5.189s;large-batch/grid-fold 实际覆盖 17/2 个 stage2 chunk;
  log SHA-256
  `0489df61514923464543764325763fdb39cb8265918ad10bb9a66ad7fd67f6f5`。
- commit-bound release:
  `gpu-et:/tmp/flagos-selective_state_update-e18-release.N0NJRR`,PID/PGID/SID
  `241458`;Git-object manifest 前后一致,同组 probes + variants **5/5 PASS**;
  release log SHA-256
  `11c0609945304eea1f1790659b1d3e454f57477446fbbacb3e0da019ea082dee`。
- canonical ZIP:
  `artifacts/competition/selective_state_update/e18-35ce353/selective_state_update.zip`,
  13562 bytes,SHA-256
  `093ed90ceaf6c7e694f03e52c9a6ca1cfc33b5636d8ec322d838ae2e13381b50`;
  actual/`--verify-existing` 一致,仅 generic + `_kunlunxin` 两成员。

### E18 平台结果(sub 7657,2026-09-01 12:59 CST):坏值来自 stage1

- preflight 全过后一次性提交;平台文件 URL SHA-256
  `f456eeb21039992c5fcf3712a09d045606c8509e27d917cce1d8a6ec6ae2eaff`;
  提交后额度 `16/30`,远端 ZIP 回读 `unavailable`,未重试。
- 七芯 generic 全过:天数 `3.885x`、沐曦 `9.1145x`、燧原 `0.5155x`、
  海光 `8.4695x`、华为 `3.674x`、card_a `6.422x`、card_b `8.203x`。
- 昆仑编译执行完成(19648ms),case 0/1 通过;case 2/3/4 的失配数、最大
  绝对误差和索引仍与 E13-E17 完全相同。旧二维 stage2 已完全被一维标量
  reduction 替代仍读到相同坏值,强证据指向 stage1 `partial_y` 生成。
- E19 保留 E18 stage2,把 stage1 同一计算体静态特化为 `WRITE_STATE=False/True`
  两核,严格先 partial 后 state。前者从 `s_old` 生成 FP32 partial 且不写 state;
  后者仍从 `s_old` 重算并仅写降精度 state。无新 buffer;反序会 double-update,
  因而禁止。

## E19:拆分 stage1 partial/state 特化(commit `6be54f6`)

- 唯一执行变量:保留 E18 flat stage2、P8×N16 与所有 lowering flag;stage1
  同一源码新增 `WRITE_STATE` constexpr,每个 slice/batch 按 `(False,True)`
  双 launch。False 从 `s_old` 计算 FP32 `new_s` 后只做 C-reduce/partial store;
  True 仍从同一 `s_old` 重算,只做降精度 state store。由此切断同一 DAG 的
  二维 state store 与 reduction-derived store,不新增 buffer。
- 顺序是数学硬约束:若 True 先执行,False 会从已更新 state 再递推一次而
  double-update。vLLM 上游同样先用 FP32 update 计算输出、再降精度写 state
  ([FP32 update/output](https://github.com/vllm-project/vllm/blob/ce2e343be1f757a92b3c990023f87bdd87a579ac/vllm/model_executor/layers/mamba/ops/mamba_ssm.py#L385-L455),
  [state store](https://github.com/vllm-project/vllm/blob/ce2e343be1f757a92b3c990023f87bdd87a579ac/vllm/model_executor/layers/mamba/ops/mamba_ssm.py#L465-L494))。
- source/verification commit
  `6be54f6e43fd5f89c9ab17ea4b1a779a2ed234c7`;Kunlun SHA-256
  `57e0c2990682030b38238fcc67e44c1b96d50c7156143656ab2c0faad1ecd6b6`;
  generic/test 仍为 `c1e180...` / `a6cc8...`。
- screening:`gpu-et:/tmp/flagos-selective_state_update-e19-screening.plFAA9`,
  PID/PGID/SID `241741`;False state 不变/partial 正确、True state 正确/partial
  哨兵不变、False→True→flat 完整数值链和 offset probes 全过;variants
  **5/5 PASS**,9.910s;large/grid-fold 覆盖 stage1 48/4 launches 与 stage2
  17/2 chunks;log SHA-256
  `1f3029c6d2211e06d18b6fc44c3e47705b57af763e3756080e6e443be0c49679`。
- commit-bound release:
  `gpu-et:/tmp/flagos-selective_state_update-e19-release.KOoiG8`,PID/PGID/SID
  `242242`;Git-object manifest 前后一致,同组语义 probes + variants **5/5 PASS**;
  release log SHA-256
  `77f854229819df32f9d0a6070443f1bb1dfb65a17e347ceed2aef9fadf0a8053`。
- canonical ZIP:
  `artifacts/competition/selective_state_update/e19-6be54f6/selective_state_update.zip`,
  13874 bytes,SHA-256
  `c640f4af73fc75e229d2fa5d00a1415ac19c7d58a42df3084952d4befb53b90e`;
  actual/`--verify-existing` 一致,仅 generic + `_kunlunxin` 两成员。

### E19 平台结果(sub 7660,2026-09-01 13:06 CST):二维 partial 归约仍错

- preflight 全过后一次性提交;平台文件 URL SHA-256
  `3c23087d68ccab46667a42d3386a749f0badbb6eaf6175b526a4d87d40919093`;
  提交后额度 `15/30`,远端 ZIP 回读 `unavailable`,未重试。
- 七芯 generic 全过:天数 `3.9945x`、沐曦 `9.079x`、燧原 `0.513x`、
  海光 `8.472x`、华为 `3.635x`、card_a `6.786x`、card_b `8.2505x`。
- 昆仑编译执行完成(14050ms),case 0/1 通过;case 2/3/4 指纹仍与 E13-E18
  完全相同。多 store/reduce 同 DAG 被证伪,但拆分改变了执行时间,说明候选
  确实改变生成代码而非 cache 假象。
- E20 保留 flat stage2 和独立 state 写回,把 partial 计算改成每个
  `(b,h,p,slice)` 一个 program、沿 N_SLICE=16 的真正 1D update+C dot 和标量
  store。官方 [Kunlun dot direct kernel](https://github.com/flagos-ai/FlagGems/blob/a7620cc191a0b42e040194622c5758b22a7a25dc/src/flag_gems/runtime/backend/_kunlunxin/ops/dot.py#L27-L47)
  是同构安全路径;官方 [mean 注释](https://github.com/flagos-ai/FlagGems/blob/a7620cc191a0b42e040194622c5758b22a7a25dc/src/flag_gems/runtime/backend/_kunlunxin/ops/mean.py#L124-L141)
  还记录 converted BF16 二维 axis-1 reduction 可产生约 97% mismatch,与本题
  `95%-96%` 指纹高度一致。

## E20:stage1 partial 改为 1D dot(commit `8b295a3`)

- 唯一执行变量:保留 E18 flat stage2;E19 state-only 语义不变并去掉 dead
  partial 参数;partial 改为每个全局 `(b,h,p,slice)` 一个 program,从 immutable
  原 state 重算 FP32 update,沿 N_SLICE=16 做 1D `new_s*C` reduction 后标量
  store。每个 slice 先按 65535 输出分块跑完 partial,再运行旧 state-only;
  无新 buffer,避免从已降 BF16 state 回读产生额外误差。
- 首次 screening 仅在 Black79 decorator 换行失败即停止,未跑 JIT/数值,
  log SHA-256 `75f6083a...`;按格式修正产生新字节后从头筛选,失败证据未复用。
- source/verification commit
  `8b295a3254c80764687d819a2753e9ea65ac90c3`;Kunlun SHA-256
  `63be0e3c80be853b5b387f7f9ca0e22ff26f79077bc4c3fe5d2350c422724705`;
  generic/test 仍为 `c1e180...` / `a6cc8...`。
- corrected screening:
  `gpu-et:/tmp/flagos-selective_state_update-e20-screening-corrected.T6Xf6Q`,
  PID/PGID/SID `242739`;1D partial 非零 offset + N21 尾片 + immutable state、
  state-only、组合 flat stage2 probes 全过;variants **5/5 PASS**,9.083s;
  large 几何为 partial/state/stage2 `136/24/17`,grid-fold 为 `2/2/2`;
  log SHA-256
  `69916216ea9ec3dae9087e3f3b3e3717c95086b3e5dd90e0dee938f472e50bb8`。
- commit-bound release:
  `gpu-et:/tmp/flagos-selective_state_update-e20-release.fmYpaD`,PID/PGID/SID
  `243252`;Git-object manifest 前后一致,同组语义/几何 probes + variants
  **5/5 PASS**;release log SHA-256
  `b52778dbef7052137a0809def126f0d474282e1f2929d280d1e4f77abec2c9c7`。
- canonical ZIP:
  `artifacts/competition/selective_state_update/e20-8b295a3/selective_state_update.zip`,
  15717 bytes,SHA-256
  `8dd9aeef873afa1ea7c2d838248b1fa88ef006647113797d0c2af0c996b0e279`;
  actual/`--verify-existing` 一致,仅 generic + `_kunlunxin` 两成员。

### E20 平台结果(sub 7661,2026-09-01 13:15 CST):y 全部修复,仅 state 错

- preflight 全过后一次性提交;平台文件 URL SHA-256
  `f4ef89e607c9abcd24759c6a1413dc5fbccc5d19b407e7e043d4d49057d5fef4`;
  提交后额度 `14/30`,远端 ZIP 回读 `unavailable`,未重试。
- 七芯 generic 全过:天数 `3.887x`、沐曦 `9.0965x`、燧原 `0.512x`、
  海光 `8.444x`、华为 `3.643x`、card_a `6.418x`、card_b `8.2925x`。
- 昆仑编译执行完成(14839ms),case 0/1 全过。case 2/3/4 不再出现
  `y` 的 `[B,H,P]` 失配,唯一失败分母分别为 state 的 `[B,H,P,N]`
  元素数:`90991/196608`(46.3%,最大绝对差 31275)、
  `27822035/33554432`(82.9%,54)、`111235478/134217728`(82.9%,51.5)。
  因此 E20 的 1D partial + 1D stage2 已完整修复输出路径,剩余根因被隔离到
  旧 `[8,16]` state-only tile/store。
- E21 仅把 state-only 改为每个 `(b,h,p,slice)` 一个 program、沿 N=16
  单向量计算和写回;复用 E20 已通过的全局输出分块,冻结 partial/stage2。

## E21:state 改为 1D 单向量写回(commit `1313907`)

- 唯一执行路径变量:冻结 E20 已由平台证明正确的 partial/stage2;state-only
  从 `[P=8,N=16]` 二维 tile 改成每个全局 `(b,h,p,slice)` 一个 program,
  只保留 N=16 连续向量。删除二维 broadcast/mask/store 和旧 3D batch grid,
  复用 E20 已通过的 `output_start + program_id(0)` 解码与 65535 host 分块。
- state source/destination 显式分离:始终从 immutable 原 `state` 读、向
  `new_state` clone 写,消除原地 load/store RAW alias;数学、FP32 更新、尾片 mask
  和最终 dtype 不变。E20 失败比例在 N32/N128 上接近后续 slice 损坏,因此这比
  继续调 pass 参数更直接。
- source/verification commit
  `1313907126acb1f32b66a656989e338a299cace6`;Kunlun SHA-256
  `ca02c06095a31cf8caab2328cbf25ae86672c9cd6f2e8671f6b5821f28e050aa`;
  generic/test 仍为 `c1e180...` / `a6cc8...`。
- screening:`gpu-et:/tmp/flagos-selective_state_update-e21-screening.O9h2vb`,
  PID/PGID/SID `243671`;static + partial/state/stage2 独立 probes + 完整组合链
  + variants **5/5 PASS**,8.807s;state probe 覆盖非零 output offset、N21 尾片、
  destination sentinel 和 source bitwise immutable;large/grid-fold 几何为
  partial/state/stage2 `136/136/17` 与 `2/2/2`;log SHA-256
  `dc65a2e6d9110739e56267900daf478dde664ad55719131c8bcafd4c457b51c6`。
- commit-bound release:
  `gpu-et:/tmp/flagos-selective_state_update-e21-release.NZV8GV`,PID/PGID/SID
  `244191`;Git-object 五文件 manifest 前后完全一致,同组门禁 + variants
  **5/5 PASS**,4.064s;release log SHA-256
  `7c3911b552000876b01f805803c123de4e27c525f9c92d5c01f032f378a3ac99`。
- canonical ZIP:
  `artifacts/competition/selective_state_update/e21-1313907/selective_state_update.zip`,
  15097 bytes,SHA-256
  `fad579e5d6794fab3208a3f5d3e6d7e60efd3f2d9b118de4237d91c56497298c`;
  actual/`--verify-existing` 一致,仅 generic + `_kunlunxin` 两成员。

### E21 平台结果(sub 7664,2026-09-01 13:24 CST):首次 8/8,性能门槛未过

- preflight 全过后一次性提交;平台文件 URL SHA-256
  `ea54b85739cd9504e8786ef137348800ef3b5f7ddafca7ed5ecc1e15d0bf48c8`;
  提交后额度 `13/30`,远端 ZIP 回读 `unavailable`,未重试。
- **八芯正确性全部通过**:天数 `3.9915x`、沐曦 `9.0935x`、燧原
  `0.514x`、海光 `8.4405x`、昆仑 `0.001x`、华为 `3.6465x`、
  card_a `6.7635x`、card_b `8.2575x`;八芯均值 `5.0885x`。平台终态为
  `invalid_threshold`,唯一原因是昆仑低于逐芯 `0.1x` 门槛,数值失败为零。
- 昆仑 validation 用时 `144366ms`,符合保守结构在最大例上产生
  partial/state/stage2 `136/136/17`、合计 289 次 launch 的代价。E20/E21 已将
  1D N16 reduction 和 vector store 分别平台验真,后续不再调正确性 pass。
- 对照 12:31 已验实时榜首快照 c2flow `8.4960x`:当前均值绝对差
  `3.4075x`(-40.1%)。冻结七芯 `40.707x` 总和时,恢复有效只需昆仑
  `>=0.1x`;若要单靠昆仑追平榜首则需 `27.261x`,故先恢复门槛再逐层合并。
- E22 优先删除 workspace 和 slice 级 host launch:每个 `(b,h,p)` program
  直接处理完整 N32/N128,同一 1D DAG 计算 FP32 new_state、C reduction、D/z,
  并一次写 state/y。保留 E21 的 flat grid、immutable source 和三个 close flag。

## E22:full-N 1D state/y 融合(commit `f1a12f7`)

- 删除 E21 的 partial workspace、N-slice host loop 和 state/partial/stage2 三核;
  每个全局 `(b,h,p)` program 以 `N_BLOCK=next_power_of_2(dstate)` 一次处理
  完整 N32/N128,FP32 `new_s` 同时供 C reduction 与 state store,再融合 D/z 和
  y store。max case 从 `136+136+17=289` launches 降为 17,且 new_s/exp 不再
  重算,删除约 32 MiB partial workspace。
- 延续 E21 的 immutable `state`→独立 `new_state`、flat 65535 chunk 和三个
  close flags;输出改为 `empty_like` 后由所有 program 唯一覆盖,省去最大约
  256 MiB clone。load 按 state/A→B→C 依赖顺序排列,先完成 C reduction 再
  vector store,缩短活跃向量周期。FlagTree 的
  [XPUOptions](https://github.com/flagos-ai/FlagTree/blob/7b0370a4976c6fcdbab89420bf53728472d75a9e/third_party/xpu/backend/compiler.py#L83-L114)
  和 [pass 顺序](https://github.com/flagos-ai/FlagTree/blob/7b0370a4976c6fcdbab89420bf53728472d75a9e/third_party/xpu/backend/compiler.py#L303-L320)
  证明不存在可伪造的 StoreControl close flag,故不再增加无效参数。
- 首次 screening 使用 source SHA
  `eb22b9660db38f65268475783418e845561d7535697d25e62a7e7eff905a0e4b`,
  static/probe/variants 5/5
  全过,目录 `gpu-et:/tmp/flagos-selective_state_update-e22-screening.cfLzao`,
  log SHA
  `0af210c7b6d0cb2a9c205349f8857084fc7ff99d83db1b13159bb06b7bdfeba1`;
  随后删除全 state clone并收缩 live range,源码字节改变,
  此结果仅作诊断,没有升级为 release 证据。
- source/verification commit
  `f1a12f7588b9e23937df66a54bb4a142263187c9`;Kunlun SHA-256
  `9237a91f04b1fcb5573822bc92b6aeda14a673b5229df8fb9621cbc8b84fb619`;
  generic/test 仍为 `c1e180...` / `a6cc8...`。
- corrected screening:
  `gpu-et:/tmp/flagos-selective_state_update-e22-screening-corrected.xzOwh3`,
  PID/PGID/SID `244881`;full-N direct probe 覆盖非零 offset、跨 batch、N21
  尾片和 D/z/bias/softplus 全分支;large/grid-fold 实际 17/2 launches;
  variants **5/5 PASS**,6.457s;log SHA-256
  `009bbf3e54a6d483369abd8cab59e441ce83bc483cb8e82006cb9e2bfb76d08c`。
- commit-bound release:
  `gpu-et:/tmp/flagos-selective_state_update-e22-release.NNPaxO`,PID/PGID/SID
  `245194`;Git-object 五文件 manifest 前后完全一致,同组门禁 + variants
  **5/5 PASS**,3.996s;release log SHA-256
  `c2bba216b5f9e3b3d4a81307bfea6cd6480b7e1b3e28c6ecaf6dccf74af5c479`。
- canonical ZIP:
  `artifacts/competition/selective_state_update/e22-f1a12f7/selective_state_update.zip`,
  10684 bytes,SHA-256
  `9bc4981215c14bc90263c1383b4ee0a047e1777a0367e5fa47383a84590d6fc1`;
  actual/`--verify-existing` 一致,仅 generic + `_kunlunxin` 两成员。

### E22 平台结果(sub 7669,2026-09-01 13:37 CST):8/8,仍低于门槛

- preflight 全过后一次性提交;平台文件 URL SHA-256
  `538ed1a42fb58463b1072a94889a95b7eb95476e6c1a9214c640b3115aaed6c3`;
  提交后额度 `12/30`,远端 ZIP 回读 `unavailable`,未重试。
- 八芯全部正确:天数 `3.999x`、沐曦 `9.1005x`、燧原 `0.7655x`、
  海光 `8.467x`、昆仑 `0.0025x`、华为 `3.649x`、card_a `6.768x`、
  card_b `8.209x`;均值 `5.1200625x`,终态仍 `invalid_threshold`。
- 昆仑 validation `52428ms`,相较 E21 `144366ms` 缩短 63.7%,speedup
  提升 2.5 倍;full-N 融合方向被平台实证,但逻辑 program 数仍为 1,048,576。
- E23 删除 65535 host chunk,直接 `grid=(total_outputs,)`;FlagTree
  [LoopGrid](https://github.com/flagos-ai/FlagTree/blob/7b0370a4976c6fcdbab89420bf53728472d75a9e/third_party/xpu/lib/Dialect/TritonXPU/Transforms/LoopGrid.cpp#L35-L102)
  会在 12 clusters 上重建完整逻辑 pid,本题最大 grid 仍在 i32 范围。并为
  power-of-two N32/N128 增加 constexpr 无 mask 快路;官方 Kunlun
  [fused RMSNorm](https://github.com/flagos-ai/FlagGems/blob/5d281c8f9073bf9547351b0e4c835465586d327f/src/flag_gems/runtime/backend/_kunlunxin/fused/fused_add_rms_norm.py#L81-L108)
  记录恒真 mask 可带来接近 2 倍损失。

## E23:单 logical grid + power-of-two N 无 mask(commit `b3dc8bd`)

- 唯一性能变量:冻结 E22 已经八芯验证的 full-N fused 数学、immutable source、
  独立 destination 和三个 close flag;删除 `_MAX_GRID`、`output_start` 与最大
  17 次 host chunk launch,改为直接 `grid=(total_outputs,)`。kernel 内
  `out_idx=tl.program_id(0)`,由 FlagTree LoopGrid 在 12 clusters 上重建逻辑
  program id。
- `N_BLOCK=next_power_of_2(dstate)`,增加 constexpr `NEED_N_MASK`;平台主形状
  N32/N128 走完全无 mask 的 load/store/reduction 快路,N21/N65 等代理尾部继续走
  masked 路径。没有引入二维 tensor、workspace 或额外 kernel。
- 首次 screening 在 Black79 格式门禁即失败并停止,没有执行 JIT/数值;目录
  `gpu-et:/tmp/flagos-selective_state_update-e23-screening.ngAV4d`,该结果不作候选
  证据。格式修正改变源码字节后按完整门禁重新执行。
- source/verification commit
  `b3dc8bdb4f39c7cecbcb69193419c475594d08bb`;Kunlun SHA-256
  `b1872e5fcef57ba975ca63ae7e20adc22153948f8dfe2b6f4dc918484fd89595`;
  generic/test 仍为 `c1e180...` / `a6cc8...`。
- corrected screening:
  `gpu-et:/tmp/flagos-selective_state_update-e23-screening-corrected.lc6Fcg`,
  PID/PGID/SID `245615`;static、direct N21 all-flags probe、最大/折叠 logical
  grid 和 variants **5/5 PASS**,6.397s;log SHA-256
  `343ecb5b1dc7b5ff3e0c22e7058226673f0610971de7e773a03c4ac8cc5d316a`,
  gate SHA-256
  `22fa41ff5c1cd8bb42d80b4799cbaee90ee74570bb8e617e1114083178ce90db`。
- commit-bound release:
  `gpu-et:/tmp/flagos-selective_state_update-e23-release.bNL4G4`,PID/PGID/SID
  `245904`;Git-object 五文件 manifest 前后一致,同组门禁 + variants
  **5/5 PASS**,3.938s;release log SHA-256
  `7eaa3a0ac4998dd52726b4caf8bc5c5922895d640b1c9db45dc06def5c31bdd5`,
  gate SHA-256
  `a3d100ff5c59d78dd384e452c386dc36f9f3425188b4f5d4719790938fa13cbb`。
- canonical ZIP:
  `artifacts/competition/selective_state_update/e23-b3dc8bd/selective_state_update.zip`,
  11035 bytes,SHA-256
  `0bed9c681d675767184274a1cc392c0e096feaacb7ebcd64ddf8d44e3b6eb1b0`;
  actual/`--verify-existing` 一致,仅 generic + `_kunlunxin` 两成员。

### E23 平台结果(sub 7672,2026-09-01 13:46 CST):无 mask 快路数值破坏

- preflight 全过后一次性提交;平台 file URL SHA-256
  `fb3b0ff0802cfbacf51c395ff9eba649156465b7d4fd946ca291a5f874c52262`;
  提交后额度 `11/30`,远端 ZIP 回读 `unavailable`,未重试。
- 七芯 generic 全过:天数 `3.8825x`、沐曦 `9.114x`、燧原 `0.514x`、
  海光 `8.4925x`、华为 `3.637x`、card_a `6.405x`、card_b `8.2685x`。
  昆仑 10402ms 后在 pytest 的前三例上停止,终态 `invalid_correctness`。
- 三例都只报 state:case 0 `360/512`(70.3%,含 `inf`),case 1
  `25621/40960`(62.6%,含 `inf`),case 2 `104548/196608`(53.2%,最大绝对差
  `4.673e34`)。E22 同三例全过;case 0/1 的 logical grid 本就低于旧 65535
  host cap,因此超大 grid 不是这些失败的必要条件,首要嫌疑是新加的 power-of-two
  N 无 mask load/store。
- E24 做最小二分:恢复 E22 已平台验真的恒真 `n_mask` load/store,只保留 direct
  logical grid。若八芯恢复正确,再评估单 grid 收益;在此之前不叠加 P_TILE,避免把
  compiler 数值问题和多输出结构混在同一候选。

## E24:恢复恒真 mask,单独验证 direct grid(commit `895e4e1`)

- 唯一执行变量:从 E23 删除 `NEED_N_MASK` 及全部 unmasked load/store 分支,
  恢复 E22 已八芯验真的 4 个 masked load + 1 个 masked state store;继续保留
  `grid=(total_outputs,)`、`out_idx=tl.program_id(0)` 和单 host launch。相对 E22
  的源码差异仅为删除 `_MAX_GRID/output_start/host chunk loop`,数学 IR 不变。
- source/verification commit
  `895e4e1f2d92fd1a70f7aa93ab53fa5bfb8a68e2`;Kunlun SHA-256
  `83bfbb8edbe274ba2b83fa2809d5d4264e76cfc21ba8b6ca6298638aa0422dc4`;
  generic/test 仍为 `c1e180...` / `a6cc8...`。
- screening:
  `gpu-et:/tmp/flagos-selective_state_update-e24-screening.vo9mAK`,mode 0700,
  PID/PGID/SID `246216`;Black79/isort/flake8/static audit 全过;N21 尾部、N32/N128
  恒真 mask、跨 batch/head、state immutable 和最大 `1,048,576`/fold `70,000`
  direct grids 全过;variants **5/5 PASS**,6.440s;log SHA-256
  `be34dfaba685fb8110d4e8465b23e92ebf9b319d6db0193a1f0c2574c0ace812`,
  gate SHA-256
  `1c88f5c745b77a5f47ba817249a5c015d24be57a6d666e152062d343742973f0`。
- commit-bound release:
  `gpu-et:/tmp/flagos-selective_state_update-e24-release.5Z3dH0`,mode 0700,
  PID/PGID/SID `246520`;Git-object 五文件 manifest 三次一致,同组门禁 + variants
  **5/5 PASS**,3.932s;release log SHA-256
  `b293b76d4e773c61721ac7326da59f874839f58128439e2ea5d92e2fb21e45f9`,
  gate SHA-256
  `83f8e38dd5b172e4b8133aa5e2ef4d406b0a0ae9ff5d7d5d2a7408c07dd10e8e`。
- canonical ZIP:
  `artifacts/competition/selective_state_update/e24-895e4e1/selective_state_update.zip`,
  10343 bytes,SHA-256
  `fdb4b213ff2003043ea0514f78d29ceeefb1eaf041c3c09ecd8f1bbf8cad2341`;
  dry-run/created/`--verify-existing` 一致,仅 generic + `_kunlunxin` 两成员。
- 13:47 CST 实时榜单:162 投/20 队,仅 c2flow `8.4960x` 与 Fields
  `6.4056875x` 有效;c2flow/Fields 的昆仑分别为 `0.7145x/0.113x`。SoulCoder
  当前仍以 E22 的昆仑 `0.0025x` 为正确基线;E24 首门是恢复 8/8 并判断 direct
  grid 能否逼近 `0.1x`,而不是凭代理耗时预测名次。

### E24 平台结果(sub 7678,2026-09-01 13:56 CST):8/8,单 grid 无收益

- preflight 全过后一次性提交;平台 file URL SHA-256
  `ed654c0e6178c5f1b84617f966c789a9a01cf90c3a7707ceadac59dbae89a40b`;
  提交后额度 `10/30`,远端 ZIP 回读 `unavailable`,未重试。
- 八芯全部正确:天数 `3.883x`、沐曦 `9.1075x`、燧原 `0.5165x`、海光
  `8.4685x`、昆仑 `0.0025x`、华为 `3.642x`、card_a `6.7745x`、card_b
  `8.2755x`;均值 `5.08375x`,终态 `invalid_threshold`。
- 昆仑 validation `52510ms`,与 E22 的 `52428ms/0.0025x` 等价。E23 的
  silent corruption 因恢复恒真 mask 完全消失,证明旧 XPU 对本核不能使用官方
  RMSNorm 的 unmasked memory 快路;但 17→1 host launch 没有可测收益,说明
  LoopGrid 最终仍执行同样的 `1,048,576` logical programs。
- E25 才改变主瓶颈:每个 program 在同一 `(b,h)` 内顺序处理 4 个 scalar p,
  grid 缩小 4 倍;每个 lane 内仍只形成独立 1D N 向量、axis-0 reduction 和立即
  state/y store,禁止重新形成 E13-E19 已证伪的 `[4,N]` tensor。N load/store
  继续无条件使用 E24 mask。

### E23/E24 数值根因:逻辑恒真 mask 是物理 idle-core guard

- 平台 FlagTree 默认 XPU `core_num=64`;关闭 CoreTiling 后,无 encoding 的 N
  向量仍按 64 cores 建 ClusterLayout,N8/N16/N32 分别产生 56/48/32 个 idle
  cores。官方 [layout 定义与 idle-core 示例](https://github.com/flagos-ai/FlagTree/blob/7b0370a4976c6fcdbab89420bf53728472d75a9e/third_party/xpu/include/triton/Dialect/TritonXPU/IR/TritonXPUAttrDefs.td#L75-L186)
  和 [make_range lowering](https://github.com/flagos-ai/FlagTree/blob/7b0370a4976c6fcdbab89420bf53728472d75a9e/third_party/xpu/lib/Conversion/TritonXPUToLLVM/MakeRangeOpToLLVM.cpp#L141-L180)
  证明这些 cores 会得到 `n_off>=N`,并不存在隐式 bounds guard。
- E22/E24 的 runtime `n_off < dstate` 被
  [Mask pass](https://github.com/flagos-ai/FlagTree/blob/7b0370a4976c6fcdbab89420bf53728472d75a9e/third_party/xpu/lib/Dialect/TritonXPU/Transforms/Mask.cpp#L537-L817)
  降成每核 GM2LM/LM2GM 的 DMA length 与 `scf.if`;idle core 得到 length 0 并
  跳过。E23 删除 mask 后,unmasked LM2GM 使用完整 buffer length,越过当前 row
  写入后续 state 并与其他 program 竞态,解释 `inf`/巨大值与非确定性损坏。
- y 先保持正确而 state 损坏也与官方
  [StoreControl](https://github.com/flagos-ai/FlagTree/blob/7b0370a4976c6fcdbab89420bf53728472d75a9e/third_party/xpu/lib/Dialect/TritonXPU/Transforms/StoreControl.cpp#L103-L176)
  一致:它只为定义链包含 ReduceOp 的 scalar store 加 used-core/core0 guard;
  `new_s` state store 不在保护范围。故这不是 Vectorize/Unroll 参数或真实 SRAM
  容量问题。
- 结论:本核的 runtime N mask 不得因“逻辑 N 等于 N_BLOCK”删除。理论上只有
  `N_BLOCK>=64` 才没有物理 idle core,但 E23 在前三例失败后停止,N128 未形成独立
  平台证据,当前不启用该猜测快路;普通 tensor-of-pointers 的 `boundary_check` 也
  不是合法替代。

## E25:每 program 顺序处理 4 个 scalar p(commit `6ffc75d`)

- 唯一性能变量:每个 logical program 从一个 `(b,h,p)` 改为一个
  `(b,h,p_tile)` owner,通过 `tl.static_range(P_TILE=4)` 顺序完成四次独立的
  scalar-p + 1D-N update/reduce/store;最大 grid `1,048,576→262,144`。每次
  lane 内立即写 state/y,B/C 不跨 lane hoist,源码只有唯一 N `tl.arange`,没有
  P arange、`[:,None]`、axis-1 reduction 或 `[4,N]` tensor。
- P 尾块用 constexpr `NEED_P_MASK`;tile 解码按
  `row=pid//ceil(P/4), p_tile=pid%ceil(P/4)`,因此尾 lane 不会跨 head/batch。
  N load/store 延续 E24 的 runtime `n_mask`,包括 power-of-two 主形状;输入 state
  与 `new_state` 继续分离。官方 Kunlun
  [fused RMSNorm](https://github.com/flagos-ai/FlagGems/blob/5d281c8f9073bf9547351b0e4c835465586d327f/src/flag_gems/runtime/backend/_kunlunxin/fused/fused_add_rms_norm.py#L28-L59)
  同样把 small-N/many-row 归因为 per-program launch latency,但其二维 multirow
  路径与 E13-E19 的平台反例冲突,故本候选保持顺序 1D DAG。
- 首次 source commit `7c3e7ec8aa948231ee1c9e711e485b2679e421cc`,Kunlun
  SHA `1b058bda...`;screening
  `gpu-et:/tmp/flagos-selective_state_update-e25-screening.9BvWgJ` 在 Black79
  格式门禁即停止,未运行 JIT/数值;gate/log SHA-256 分别为
  `2ee4106854202e7420c02079454db36e8806a21bb3d3b31169516976d9abcbd7` /
  `a7f71fc3247e3ea50460bbab0c7944013733d63820f8ccd8c6cad8fccb85b874`,
  不作候选证据。按冻结副本 Black diff 修正后产生新 commit/字节并从头筛选。
- source/verification commit
  `6ffc75d4a00ee652ca2f3f58fbd53947ad51fb45`;Kunlun SHA-256
  `774347e4f17dbad17e522d2fd7921d568bebe06cb74d400308e3a234e5aca9fe`;
  generic/test 仍为 `c1e180...` / `a6cc8...`。
- corrected screening:
  `gpu-et:/tmp/flagos-selective_state_update-e25-screening-corrected.PFbwjQ`,
  mode 0700,PID/PGID/SID `247163`;P1/P5/P33 × N21/N32/N65/N128 共
  **12/12** direct probes 全过,覆盖 P/N tail、redzone、跨 head/batch、all-flags/
  no-flags 与 state immutable;最大/fold grid `262144/70000`,single launch;
  variants **5/5 PASS**,8.781s;log SHA-256
  `fd297c59b54c07024699e589c85ccb24077563a431dee08f4aa414ae68b2af9c`,
  gate SHA-256
  `2edd1f417a7df5efb26883b7d5b5d14b4ad4d75e2d161d738986c46d0b10c648`。
- commit-bound release:
  `gpu-et:/tmp/flagos-selective_state_update-e25-release.EaIbz9`,mode 0700,
  PID/PGID/SID `247511`;Git-object manifest 三次一致,同组 12 probes + geometry +
  variants **5/5 PASS**,3.916s;release log SHA-256
  `799721ebc718d219838ef37adcfb123b6d5e19ffaffc909d6136defe9c2d85de`,
  gate SHA-256
  `06bffb716893bb63b30becfe096eb28c0be71833f052c5ba6411705280ca57a2`。
- canonical ZIP:
  `artifacts/competition/selective_state_update/e25-6ffc75d/selective_state_update.zip`,
  12070 bytes,SHA-256
  `8c1426e0bc48a9de8f13df66652dbadb6b4fdb3d6d926dcd42c5e8cf07a2943a`;
  dry-run/created/`--verify-existing` 一致,仅 generic + `_kunlunxin` 两成员。

### E25 平台结果(sub 7691,2026-09-01 14:19 CST):8/8,仅小幅收益

- preflight 全过后一次性提交;平台 file URL SHA-256
  `66f997d4b3cb4711de8435e7a1367138412d917f76ebedcdc64fb5ed65f932b2`;
  提交后额度 `9/30`,远端 ZIP 回读 `unavailable`,未重试。
- 八芯全部正确:天数 `3.9895x`、沐曦 `9.1025x`、燧原 `0.516x`、海光
  `8.4715x`、昆仑 `0.003x`、华为 `3.632x`、card_a `6.422x`、card_b
  `8.237x`;均值 `5.0466875x`,终态 `invalid_threshold`。
- 昆仑 validation `50348ms`,相较 E24 `52510ms/0.0025x` 仅缩短 4.1%,平台
  speedup 量化为 +20%,远低于 grid 数 4 倍缩减。static_range 只是把四份相同
  DAG 放进一个 logical program,总 state/A/B/C 流量和 exp/reduce 次数完全未减,
  因而 launch/LoopGrid 迭代不是唯一主瓶颈。
- E26 停止继续扫 P_TILE,改为每个 `(b,h)` 一个 program、运行时 loop 精确遍历
  P16/P64/P128;把同一 `(b,g)` 的 1D B/C 在循环外各加载一次,loop 内仍保持
  state/A 的 1D masked update、axis-0 reduce 和立即 store。最大 grid 降至
  `B*H=16384`,同时主形状 B/C global load 降 16–128 倍;代价是两条 FP32 N
  向量跨 loop 常驻,需先过 XPU 编译资源门。

## E26:每 batch-head 循环 P 并复用 B/C(commit `5e731b1`)

- 唯一性能变量:删除 E25 的 `P_TILE=4`/`static_range`,每个 `(batch,head)` 由一个
  program 独占,用 `for p_idx in range(0,dim)` 精确覆盖 P;同一 `(batch,group)` 的
  B/C 各在循环外 masked load 一次。grid 从 E25 最大 `262144` 降到 `16384`,B/C
  global load 按 P 降 16–128 倍;state/A/update/reduce/store 仍是逐 P 的唯一 1D N
  tensor。E23 已证明必要的 runtime `n_mask` 覆盖 B/C、state/A load 和 state
  store,输入 state 与输出 new_state 分离。
- 固定 FlagTree 源码审计确认 runtime `range` 保留为 SCF loop,不是静态展开;
  [SCF conversion](https://github.com/flagos-ai/FlagTree/blob/7b0370a4976c6fcdbab89420bf53728472d75a9e/third_party/xpu/lib/Conversion/TritonToTritonXPU/TritonToTritonXPUPass.cpp#L386-L430)、
  [LoopGrid](https://github.com/flagos-ai/FlagTree/blob/7b0370a4976c6fcdbab89420bf53728472d75a9e/third_party/xpu/lib/Dialect/TritonXPU/Transforms/LoopGrid.cpp#L35-L102)、
  [Mask](https://github.com/flagos-ai/FlagTree/blob/7b0370a4976c6fcdbab89420bf53728472d75a9e/third_party/xpu/lib/Dialect/TritonXPU/Transforms/Mask.cpp#L537-L817)
  与 [StoreControl](https://github.com/flagos-ai/FlagTree/blob/7b0370a4976c6fcdbab89420bf53728472d75a9e/third_party/xpu/lib/Dialect/TritonXPU/Transforms/StoreControl.cpp#L103-L176)
  均能处理嵌套 loop。三项 close flag 原样保留,避免同时打开额外 XPU DAG
  unroll/vectorize 变量。该证据只证明结构合法,不能替代目标昆仑最终 lowering。
- source/verification commit
  `5e731b1bb63c75ec424b2a81ed7045f572e85d65`;Kunlun Git blob
  `77be342f7e817d9253df842c452915129ccde1bc`,SHA-256
  `0d11c87734cf10cb7c439567f4a7f813af763c97250be71da3492dc724c97689`;
  generic/test SHA-256 仍为 `c1e180...ac4c` / `a6cc8c...aaf0`。
- 首次 screening fresh dir
  `gpu-et:/tmp/flagos-selective_state_update-e26-screening.WsqlWn` 在任何 JIT 前因
  gate 自身把 `torch.full` size 写成 int 而非 tuple fail-fast;gate/log SHA-256
  为 `efb090e123f4dd7b964677eacb331cc6dfc476c16286a68d204d827313081879` /
  `27fd34acf17a4bd14e72bf020343fbb78e21bc9faad3166d81f2d8aac5a05111`。
  该失效证据不复用;只修 gate 后在新目录从 Git objects 重新冻结并全量重跑。
- corrected screening
  `gpu-et:/tmp/flagos-selective_state_update-e26-screening.8bNiXy`,mode 0700,
  PID/PGID/SID `248026`;Black79 首门、源码审计、三次 manifest 全过;最小真实
  JIT P1/N21 为 0.250577s,P={1,5,16,33,64,128}×N={21,32,65,128}×
  {no/all flags} 共 **48/48 PASS**,覆盖 state immutable、redzone、跨 batch/head;
  最大/fold grid `16384/70000` single launch;variants **5/5 PASS**,6.652s。
  gate/log SHA-256 为
  `665c68c7991fe3b7b95b874d650fe89a4cfd2fcbaa0ad610594dd37d51d04707` /
  `d7dd72203d380de1c65318bb4035483a408ced399f5cf813082b8cb8fcea93dd`。
- 独立 commit-bound release
  `gpu-et:/tmp/flagos-selective_state_update-e26-release.mvuojh`,mode 0700,
  PID/PGID/SID `248451`;重新冻结五文件,48/48 + geometry + variants
  **5/5 PASS**,三次 manifest 一致,无 compile/`uni_sram`/timeout 错误特征;
  release gate/log SHA-256 为
  `817ab53b841c4684ce44d6425876878bda80bd04da4a7c6a331db04be940fb25` /
  `c06eae6726f5d43f968f2741ae222f8506919246670859b8a45b4dbb9a43dd27`。
- canonical ZIP
  `artifacts/competition/selective_state_update/e26-5e731b1/selective_state_update.zip`,
  10511 bytes,SHA-256
  `39a25f6173f43df5689294fbf517942317e3de8180c70ca8e4d5c5bf8f5f6b43`;
  dry-run/created/`--verify-existing` 一致,仅 generic + `_kunlunxin` 两成员。

### E26 平台预注册门

- 基础门:8/8 正确且每芯 `>=0.1x`;机制门:昆仑高于 E25 `0.003x` 且 validation
  低于 50348ms;有效门即昆仑 `>=0.1x`。本候选只上传和正式提交一次,不重掷方差。
- 若目标昆仑出现 compile/`uni_sram` 或数值失败,关闭当前跨 loop live-in 形态,
  只允许用“B/C 放回 loop”作单变量归因,不得动 N mask 或三个 close flag。若八芯
  正确但昆仑 `<0.01x`,说明 row-owner+B/C 复用仍不足一个数量级,关闭该轴,不再做
  `(batch,group)` 合并;后者只再省 `H/G` 份 B/C,理论收益不足以跨 `0.1x` 门。

### E26 平台结果(sub 7696,2026-09-01 14:38 CST):8/8,复用轴无收益

- preflight tuple 全部匹配后一次性提交;平台 file URL SHA-256
  `e0fcf9fed7378717a2ba2ea3bc91aaccbad3088e4976807049cb43d6f1e073b6`;
  远端 ZIP 回读因可信 host 未配置为 `unavailable`,提交已成功且未重试。14:37:47
  CST 终态额度 `8/30`,submission `7696`/daily seq `22`。
- 八芯全部正确:天数 `3.886x`、沐曦 `9.104x`、燧原 `0.5055x`、海光
  `8.4715x`、昆仑 `0.003x`、华为 `3.6615x`、card_a `6.4185x`、card_b
  `8.281x`;均值 `5.041375x`,终态 `invalid_threshold`。
- 昆仑 validation `48495ms`,仅比 E25 的 50348ms 少 3.68%,展示 speedup 完全相同。
  因而 `P_TILE4→row-owner`、grid `262144→16384` 与 B/C load 16–128 倍下降
  都未改变主瓶颈;E26 未过预注册 `<0.01x` 机制门,该轴关闭。
- `(batch,group)` 再合并只会额外省 `H/G` 份 B/C。按每输出约三条不可共享主向量
  与两条 B/C 向量估算,相对 E26 的乐观上限为
  `(3P+2)/(3P+2/(H/G))`,P16/64/128 分别不超过约 1.0417/1.0104/1.0052x,
  且会降低并行度;远不足 `0.003→0.1x`,不提交该猜测。下一步必须把 48.5s 拆成
  specialization/JIT 编译与实际 kernel 执行,只尝试能消除数量级开销的结构。

## E27:P-major 64 cores,N 串行融合(commit `5e3d802`)

- E22 的 17 host launches 与 E24 的 1 launch 分别为 52428/52510ms;E25 logical
  programs 降 4 倍只到 50348ms。五个已锁定平台例共有 `Q=1,319,488` 个输出、
  `QN=168,010,240` 个状态元素,case 3/4 占 QN **99.8583%**。E26 虽把逻辑主流量
  从约 1.680GB 降到 1.018GB(-39.44%),理论带宽上限也只有 1.65x,平台无收益符合
  预期;约 48s 主因不是 JIT/host launch,而是每输出的小向量 DMA、exp 与跨核 reduce。
- E27 对 `P>=64 && P%64==0` 新增 P-major 路径:每 program owner 为
  `(batch,head,p_tile64)`,`tl.arange(0,64)` 让 64 XPU cores 各持一个 p,沿 runtime
  N loop 顺序加载 strided state/A,B/C 标量广播,立即写 new_state 并在每 core 的
  FP32 `y_acc` 累加;循环后一次写 64 个 y。全程只有 1D P tensor,没有 E13–E19
  证伪的 `[P,N]` tensor,也没有 `tl.sum`/跨核归约。P16/非 64 倍数继续走 E26
  的 masked N-major 正确路径。
- 主路径 64 lanes 恰好覆盖固定 XPU 64 cores,无 idle core,且 runtime N loop 精确
  覆盖 `[0,dstate)`,因此不需要 P/N mask,不会复现 E23 的 idle-core LM2GM 越界。
  `new_s` 在降精度写 state 前参与 FP32 y 累加;输入 state 只读、new_state 独立。
  固定 FlagTree 的
  [ClusterLayout](https://github.com/flagos-ai/FlagTree/blob/7b0370a4976c6fcdbab89420bf53728472d75a9e/third_party/xpu/include/triton/Dialect/TritonXPU/IR/TritonXPUAttrDefs.td#L75-L186)、
  [SCF conversion](https://github.com/flagos-ai/FlagTree/blob/7b0370a4976c6fcdbab89420bf53728472d75a9e/third_party/xpu/lib/Conversion/TritonToTritonXPU/TritonToTritonXPUPass.cpp#L386-L430)
  与 [LoopGrid](https://github.com/flagos-ai/FlagTree/blob/7b0370a4976c6fcdbab89420bf53728472d75a9e/third_party/xpu/lib/Dialect/TritonXPU/Transforms/LoopGrid.cpp#L35-L102)
  支持该结构;无 reduce 后 StoreControl 不介入。官方 Kunlun
  [cumsum](https://github.com/flagos-ai/FlagGems/blob/5d281c8f9073bf9547351b0e4c835465586d327f/src/flag_gems/runtime/backend/_kunlunxin/ops/cumsum.py#L360-L402)
  已有 runtime loop + scalar broadcast + vector store 的同构先例。三 close flags 保留。
- source/verification commit
  `5e3d802a10a538f850b73b433b468492d4be6177`;Kunlun Git blob
  `b126b601b133b8ee6ae50a8da2c832ceb08c404d`,SHA-256
  `0b7e148e51ba2eee2d51f67e083465da8ef04c6118e27d21d398d92e6c222e24`;
  generic/test SHA-256 仍为 `c1e180...ac4c` / `a6cc8c...aaf0`。
- screening
  `gpu-et:/tmp/flagos-selective_state_update-e27-screening.b3Q0Co`,mode 0700,
  PID/PGID/SID `248802`;Black79 首门、双路径源码审计、三次 manifest 全过;
  P64/N21 no flags 与 P128/N128 all flags 最小 JIT **2/2 PASS**;三 dtype
  fast/fallback 矩阵 **144/144 PASS**,覆盖 D/z/bias/softplus、state immutable、
  redzone、跨 batch/head/group;平台几何 fast grid `40/96/4096/16384`、fallback
  `4`,最大/fold `16384/70000` single launch;variants **5/5 PASS**。gate/log
  SHA-256 为
  `9161c8ce689d579e525a04fabc6bbf3ec8ad6751223ca4ee5874be78d7adf571` /
  `d145699a934e203aca3a839ce7c9dd54e590981a31df8ca7b9562e29787e6b84`。
- 独立 commit-bound release
  `gpu-et:/tmp/flagos-selective_state_update-e27-release.sVnawa`,mode 0700,
  PID/PGID/SID `249273`;重新冻结五文件,同组 2/2 + 144/144 + geometry +
  variants **5/5 PASS**,三次 manifest 一致,无 compile/`uni_sram`/timeout 指纹;
  release gate/log SHA-256 为
  `084d0c643c790618cd96f2293d02e88fbeb4aaf3e6447880257dd814d17a6f80` /
  `3027719fddddbd095198e48b4933259cab88518f33ce57e92d4df8ed7bce3ed3`。
- canonical ZIP
  `artifacts/competition/selective_state_update/e27-5e3d802/selective_state_update.zip`,
  13626 bytes,SHA-256
  `003daecf2ae2ab5c34edc763314d43c383250840130d5a4307d778a9c23b3611`;
  dry-run/created/`--verify-existing` 一致,仅 generic + `_kunlunxin` 两成员。

### E27 平台预注册门

- 基础门:8/8 正确且每芯 `>=0.1x`;机制门:昆仑明显高于 E26 `0.003x` 且
  validation 低于 48495ms;有效门为昆仑 `>=0.1x`。候选只提交一次,不重掷方差。
- 若主路径 compile/`uni_sram` 或数值失败,先按 strided load/store、loop-carried
  `y[64]` 分类,不得回改已验 fallback 或 N mask。若正确但昆仑 `<0.01x`,说明
  strided P-major 仍被 DMA/loop 控制吞噬,关闭该轴,下一单变量只测占 99.8583%
  工作量的 N128 N-major 无 mask 快路;E23 从未独立执行到 N128,故该边界尚未证伪。

### E27 平台结果(sub 7708,2026-09-01 14:57 CST):昆仑提升 4.5 倍

- preflight tuple 全部匹配后一次性提交;平台 file URL SHA-256
  `ea53b729a33afcdc5e606f632ef7822e79ded8bcb1d4f93453a5cbb2a81b7f28`;
  远端 ZIP 回读 `unavailable`,未重试。14:57:09 CST 终态额度 `7/30`,
  submission `7708`/daily seq `23`。
- 八芯全部正确:天数 `3.9935x`、沐曦 `9.1005x`、燧原 `0.505x`、海光
  `8.4435x`、昆仑 `0.0135x`、华为 `3.66x`、card_a `6.4225x`、card_b
  `8.286x`;均值 `5.0530625x`,终态 `invalid_threshold`。
- 昆仑 validation `17636ms`,相较 E26 的 48495ms 降 63.63%;speedup
  `0.003→0.0135x`,提升 **4.5 倍**。这同时排除了“strided P-major 必然更慢”并
  证明删除 Q 次跨核 reduce/把 P 映射到 64 cores 是当前首个数量级有效结构。
- 仍距最低门 `0.1x` 差 `7.4074x`,未达到有效门,但超过预注册的 `0.01x`
  机制继续线。剩余主路径每 core 对 N128 仍执行 128 次 scalar loop,且 E27 延续
  旧正确路径关闭 XPU Vectorize/UnrollControl;下一轮只选择一个编译器变量,
  优先判断官方同构 kernel 默认开启的 Vectorize 能否合并每 core 的连续 N 访存,
  不同时改 P_BLOCK、mask 或 fallback。

## E28:exact N128 × P64 二维 block-DMA(commit `d614385`)

- 固定 FlagTree 源码审计否定了 E27 上直接开 Vectorize:其 rank-1 P64 被默认
  64-core layout 分成每核 1 个 FP32 元素,低于 Vectorize 的 512-bit/16-FP32
  门槛,且该 pass 不会跨 SCF N-loop 合并迭代([Vectorize](https://github.com/flagos-ai/FlagTree/blob/7b0370a4976c6fcdbab89420bf53728472d75a9e/third_party/xpu/lib/Dialect/TritonXPU/Transforms/Vectorize.cpp#L526-L540),
  [门槛](https://github.com/flagos-ai/FlagTree/blob/7b0370a4976c6fcdbab89420bf53728472d75a9e/third_party/xpu/lib/Dialect/TritonXPU/Transforms/Vectorize.cpp#L960-L998))。
  static-unroll8 也只复制八份 scalar DAG,理想上限 `8x` 几乎刚好等于所需
  `7.407x`,现实仍含四次 load、exp 与 store;unroll4 理论上限不足过门。因此两者
  不消耗 E28 配额。
- 唯一结构变量:仅 `P>=64 && P%64==0 && N==128` 改走 exact constexpr-N
  `[64,128]` 二维 tile;state/A 是整块连续地址,B/C 沿 N 广播,new state 从独立
  输入以 FP32 计算并写独立输出,y 在降精度前做 axis-1 sum。没有 P/N mask、
  runtime N loop、N slice 或 workspace;N16/N32 和所有非主形状逐字节保留 E27
  两级 fallback。case3/4 占全部 QN **99.8583%**,因此该 gate 已覆盖主瓶颈而把
  风险限制在单一 specialization。
- 该 tile 恰为官方 Kunlun
  [fused RMSNorm multirow](https://github.com/flagos-ai/FlagGems/blob/5d281c8f9073bf9547351b0e4c835465586d327f/src/flag_gems/runtime/backend/_kunlunxin/fused/fused_add_rms_norm.py#L172-L240)
  的 8192-element SRAM 上限;launch 同样只关闭 Vectorize/UnrollControl,不传
  `isCloseCoreTiling`,故 CoreTiling 保持开启。固定
  [CoreTiling 公式](https://github.com/flagos-ai/FlagTree/blob/7b0370a4976c6fcdbab89420bf53728472d75a9e/third_party/xpu/lib/Dialect/TritonXPU/Transforms/CoreTiling.cpp#L136-L196)
  对 `[64,128]` 得到 `64 groups × 1 core`,每核独占一条连续 N128 行并做本核
  reduction。E13-E19 是 P8×N16 sliced、mask 且关闭 CoreTiling 的跨核路径,
  不构成本候选反例;P32 会退成 32 groups×2 cores,故不扫描更小 P tile。
- source/verification commit
  `d6143859a4244dab99f380179c0e8cafa5577a0b`;Kunlun Git blob
  `a4fc405d8d6e4ffb31c65fdf38e87126d8684fa3`,SHA-256
  `b4347ee826cd38f3238a286cd077bb2ca6f7d337f811a52b3abe1566e2fcee9f`;
  generic/test 仍为 `c1e180...ac4c` / `a6cc8c...aaf0`。
- 前两份辅助 gate 在任何 JIT 前因 `probe.py` 单文件/多文件 isort 分类相反而
  fail-fast:`gpu-et:/tmp/flagos-selective_state_update-e28-screening.TyyFLu` 与
  `...-corrected.tjPa2B`,日志 SHA-256 分别为
  `cd69dbab2099402308af7d1ce4efa7dd063dea94c8de88120e4edb02786c0706` /
  `023b5594cf7fc25eb1f1755b8e2e09e80c7efa87fe4c28e6155ae22032085869`;
  两者不作候选证据。按远端规范只让五个目标仓库文件进入 isort 后,从新目录重跑。
- 最终 screening:
  `gpu-et:/tmp/flagos-selective_state_update-e28-screening-final.SDPfPV`,mode 0700,
  PID/PGID/SID `249995`;Black79/静态审计/三次 manifest 全过;exact N128
  P64/P128 × 3 dtype × no/all flags **12/12**,连同 P-major/fused 回退共
  **42/42 PASS**,覆盖 state immutable、redzone、跨 batch/head/group;
  平台路径/grids 为 `pmajor,pmajor,exact,exact` / `40,96,4096,16384`,variants
  **5/5 PASS**。gate/probe/log SHA-256 为
  `787c6efb58bf376f06c31e8200707b083df2fd3ae87ae5dd02e1e31ca3dade60` /
  `0a055b2a562174f099cb4a5685a23c268c2ad276004dca431ed4adc1b5ea6849` /
  `5a273edcb021c4e1686ad802e96e36f327d68328939a2e1484e830e15b86a90a`。
- 独立 commit-bound release:
  `gpu-et:/tmp/flagos-selective_state_update-e28-release.UcTFSw`,mode 0700,
  PID/PGID/SID `250352`;重新从 Git objects 冻结五文件,同组 42/42 + geometry +
  variants **5/5 PASS**,三次 manifest 一致。gate/probe/log SHA-256 为
  `787c6efb58bf376f06c31e8200707b083df2fd3ae87ae5dd02e1e31ca3dade60` /
  `0a055b2a562174f099cb4a5685a23c268c2ad276004dca431ed4adc1b5ea6849` /
  `842351253c17df3627760ae74b1b7fc3bd254ae011efb4e9926ef3fb4d688202`。
- canonical ZIP
  `artifacts/competition/selective_state_update/e28-d614385/selective_state_update.zip`,
  16546 bytes,SHA-256
  `a3769a6e02fa85fda866d4bdde4696ba23b7a982d73b42601f4cb4d820c69a0a`;
  dry-run/created/`--verify-existing` 一致,仅 generic + `_kunlunxin` 两成员。

### E28 平台预注册门

- 基础门:8/8 正确且每芯 `>=0.1x`;机制门:昆仑高于 E27 `0.0135x` 且
  validation 低于 17636ms;有效门为昆仑 `>=0.1x`。候选只上传和提交一次,
  不重掷方差。
- 若出现 `uni_sram` 或任一 state/y 数值失败,关闭 exact-2D 轴,不扫描 P32 或
  metadata flags;转源码级 N-unroll8。若正确但昆仑 `<0.027x`,说明 block-DMA
  未兑现,停止 exact tile 参数扫;`0.027-0.04x` 没有安全单变量跨门;
  `0.04-0.1x` 才允许一次 exact-2D Vectorize-on 高风险验证;`>=0.1x` 达标即停。

### E28 平台结果(sub 7727,2026-09-01 15:23 CST):正确但回退

- preflight tuple 全部匹配后一次性提交;平台 file URL SHA-256
  `e277f87264f8e52ad9b968fe98c46e4ba0d0f58276d6c0d79746276852c25465`;
  远端 ZIP 回读 `unavailable`,未重试。终态额度 `6/30`,submission `7727` /
  daily seq `24`。
- 八芯全部正确:天数 `3.9915x`、沐曦 `9.0855x`、燧原 `0.5125x`、海光
  `8.4705x`、昆仑 `0.008x`、华为 `3.6605x`、card_a `6.4165x`、card_b
  `8.2615x`;均值 `5.0508125x`,终态 `invalid_threshold`。
- 昆仑 validation `24288ms`,相较 E27 的 `17636ms` 增加 **37.72%**,
  speedup `0.0135→0.008x` 回退 **40.74%**。目标 XPU 证明 exact-N block DMA
  与一核一行归约能正确 lowering,但本题 exp/多 FP32 live tensor/axis-1 reduce 的
  代价超过删除 128 次 scalar SCF 的收益;官方 RMSNorm 的同构结构不能外推到本算子。
- 结果低于预注册 `<0.027x` stop gate,exact-2D 轴正式关闭:不试 P32、N64
  分块、Vectorize-on 或 metadata 重扫。E29 从 E27 的已验 P-major 1D 结构分叉,
  只把 N-loop 主体源码静态展开 8 次并保留精确 tail;若编译失败不降 U4,后者
  理论上限 `4x` 无法从 `0.0135x` 跨过 `0.1x`。
