# Task 79 `create_flashmla_kv_indices` 实验记录

```current
task: 79
operator: create_flashmla_kv_indices
batch: 6
validity: valid
platform: completed(17336,e5,8/8,171.59x<TB;保s0 173.36;静态页块轴判负)
candidate_stage: e5
team_best_stage: s0
team_best_speedup: 173.3643125
sealed: no
next: 静态页块轴关闭(华为69.1<动态93.7,钳位乘法+全掩空tile浪费>控制流收益);e4双失败定性=507035(华为,钳位修复)+双互补掩码store(昆仑,e3字节vendor修复);华为93→182与沐曦122→165仍未破译;后续基座=e3字节(generic动态+enflame vendor)
updated: 2026-09-18
```

## 契约与实现（S0）

- 题面：[Task 79](../tasks/batch-6/79-create_flashmla_kv_indices.md)。T63 姐妹题
  改 per-page：`slot = req_to_token[pool, start + p*page_size]`，
  `kv_indices[i,p] = slot // page_size`；输出 2D clone 基底 + 行尾保留；
  exact。上游 92d831d `create_flashmla_kv_indices_triton` 对照。
- 实现：T63 e8 族结构（grid-stride 行循环 + 页块 split、GCU 安全算术、
  i64 仅寻址）；page_size 走 constexpr（pow2 除法强度削减、昆仑向量除法
  风险规避）；out=empty + kernel 行尾回写（免 clone 全量拷贝）。
- 测试：page_size {1,3,16,64,128}、非整除长行、kv_start=None/非零、
  空 batch/零长度行、行尾哨兵保留与 base 变更重读、宽度==页数、
  split 几何边界（255/256/257、batch 257）、stride/dtype 矩阵。

## 不可变身份

- source commit：`370923bfb83a8b7e681051acb2a3b1585dde12fc`（五题同批）。
- verification commit：`370923bfb83a8b7e681051acb2a3b1585dde12fc`（含测试侧修复轮）。
- source SHA-256：`93d158ad535142e941d8a8f0000ebeeccb147df6c8ceb02edf5d78efb9d49f01`。
- test SHA-256：`9083d94d1729394865a7b4e6026472b0fc84fef3864538c544409044350a0612`。
- ZIP：`artifacts/competition/create_flashmla_kv_indices/s0-370923b/create_flashmla_kv_indices.zip`，SHA-256
  `8818616277c4bbd9b0ad408bdc164d6696ee7a5434b12eb7a2455e8da2cb235d`（单成员 `create_flashmla_kv_indices.py`，generic-only）。
- release 回执：`artifacts/competition/b6-s0-release-20260917/create_flashmla_kv_indices/verification.json`
  （7 测试 0 失败，37 次非 warmup kernel launch，
  NVIDIA RTX 5070 Ti / torch 2.13.0+cu130 / triton 3.7.1），
  SHA-256 `037423bff60d94b0b0f4624098a07fe980606f89cc26b3c3891b4a1df8d21f1d`；日志 SHA-256 `80afa45e6882833b218ea9e9090d76a4763461d00203047c139188240d8d2890`。

## 靶子与下一步

- 榜首 HAiWORLD 187.47x；我方 T63 banked 逐芯等效 ≈199x，直移争第一；昆仑轴（8 vs 17.8）是最大增量项。
- 验证三轮教训已固化在测试侧：numel%16 契约、token 边界与 kv_lens 解耦、
  归约噪声容差、行内连续 stride 构造。
- 八芯目标 `target-runtime-unverified`（NVIDIA 代理证据），裁决权在平台。

## 2026-09-18 S0 平台首回执（submission 17209，observed_at 01:0x +08）

- 状态：8/8 valid, #2/4；均值 173.3643125。
- 逐芯：天数 340.72 / 沐曦 120.46 / 燧原 34.61 / 海光 284.95 / 昆仑 15.42 / 华为 97.54 / A 309.56 / B 183.64。
- 首发即 #2（次席 cgzhou 171.25 仅高 2.1）。对榜首逐芯：华为 +5.0 / 天数 -9.0 / 海光 -19.0 / A -12.1 / B -15.0 / 沐曦 -43.7 / 燧原 -16.7 / 昆仑 -2.4。燧原与沐曦两轴合计可 +7.5 均值。

## 2026-09-18 E1 平台终态：燧原 GCU 配方兑现 51.54，均值 169.69 < TB 保 s0

- 结构（`49e844b4`）：新增 `_enflame` vendor（T63 GCU 配方页空间版：24-SIP
  封顶 launch、BLOCK_P 512、num_stages=3、无 warps 钉）。release v2 双路径
  37+37 launch 全过；ZIP `e1-49e844b` SHA-256
  `07e18fa664289e4cba2c085ac9c6af8958b93ae6d179ff6e20edc3a5cce18403`。
- submission **17225** completed/valid，8/8，均值 **169.69x < TB s0 173.36**
  （保 s0）。逐芯（vs s0）：**燧原 34.61→51.54（+49%，追平榜首 51.3）**；
  沐曦 120.5→124.3（+3%）；其余窗口回落：海光 285→270、A 309.6→287.0、
  天数 340.7→328.5、华为 97.5→93.2、B 183.6→187.6、昆仑 15.4→15.3。
- 判读：燧原轴目标达成即关闭；e1 均值回落是七芯窗口整体降温（同发 T80
  读数正常），不判 vendor 结构回归。下一轴：沐曦（124 vs 榜首 164，
  +5 avg 最大单项）。

## 2026-09-18 E2 平台终态：华为 persistent 判负，gather 族反例第三次确认

- E2（17298，`0051b001`）：`_ascend` persistent 华为 97.54→**78.99（-19%，
  预注册门 ≥100 未过，轴关闭）**；燧原 52.22（vendor 保持）；均值 167.19
  < TB s0 173.36（保）。release 双路径全过；ZIP `e2-0051b00` 3 成员。
- 判读：persistent 增益按 op 族分化——copy/broadcast 族（T62/T78）+35~97%，
  gather 族（T63/T79）零或负。**配方适用性=按访存形态分类**，入跨芯知识库。
- 榜单：榜首 c2flow 343.5 为华为 1212 慢窗彩票（次优同芯 182.6）；真实
  靶子 #2 GuanghuLab 196.7（华为 182.6 另有结构，未破译）。

## 2026-09-18 E3 平台终态：split 修正中性，ascend vendor 移除，TB 保 s0

- E3（17316，`975df79e`）：splits 分母 128→256 对齐 BLOCK_P（codex-ask
  发现半数 split 空转）+ 删除已判负 ascend vendor（华为回 generic）。
  逐芯：华为 93.68（回 generic 水位 ✓）/ 昆仑 15.4→**17.59（+14%，split
  修正小幅正）**/ 沐曦 122.5 / 燧原 50.4（vendor 保持）。均值 173.2335
  ≈ TB s0 173.3643（差 0.08%，保 s0 名义；e3 ZIP=generic+enflame 为后续
  最佳基座）。
- 判读：split 修正对昆仑有效、对总量中性；华为 182 结构缺口仍未破译。

## 2026-09-18 E4/E5 平台终态：静态页块轴判负，两起跨芯失败完成定性

- E4（17332，`504e94ea`）：静态单页块/程序（每元素单写者、动态循环消除，
  代理 +2-7%）。**6/8**：华为 `AclrtSynchronizeStreamWithTimeout 507035`
  向量核崩溃（整块掩空的 masked load 未钳位地址——T63 generic 同款纪律）；
  昆仑 8/8 case 全错（同一指针双互补掩码 store 踩 XPU 家族雷）。
- E5（17336，`6fa9f882`）：masked load 算术钳位（T63 507035 纪律）+
  `_kunlunxin` vendor 回 e3 动态循环字节——8/8 修复 ✓，但均值 171.59
  < TB 173.36，**华为 69.1 < 动态形式 93.7（-26%）**：钳位乘法与全掩空
  tile 的调度浪费超过控制流收益。**轴关闭**。
- 新增跨芯硬事实：(1) 静态 grid 的整块掩空 tile 必须钳位 masked load
  地址（华为 507035）；(2) 同指针双互补掩码 store 在昆仑产生全量垃圾
  （与 2D broadcast store 同族）。两者均已入踩坑表候选。

## 2026-09-18 晚 E6 候选就绪（提交前验证完成，待额度刷新首发）

- 结构：e3基座+metax BLOCK_P=512（round-5 轴4）。预注册门：沐曦≥145或耗时-15%;8/8且均值≥170。
- source/verification commit `d4c96bbb4327379d7ed99df89a0759b09ca69b0d`；release 回执
  `artifacts/competition/b6-e6-ready-20260918/create_flashmla_kv_indices/verification.json`（4成员,4路径x37launch,7测试）。
- ZIP `artifacts/competition/create_flashmla_kv_indices/e6-*/create_flashmla_kv_indices.zip`：
  SHA-256 `3f114c49d61d9c5ad1adb56cbaca90301d7fef506559c9a9801bd0af62bc87e8`；test SHA-256 `9083d94d1727394865a7b4e6046472b0fc84fef3864538c544409044350a0612`；回执 SHA-256 `62aa4e8fbe69f95d4233b2c73e9efcf46d7737650be377ede361c2109d0f1689`。
- 发射参数齐备（commit/zip/sha/test/receipt 五元组已核对），午夜额度
  刷新后按第五轮排序直接 preflight→submit。
