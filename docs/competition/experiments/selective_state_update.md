# Task 36 `selective_state_update` 实验记录

状态:E5 平台 7/8；E6 本地筛选否决，未提交

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
