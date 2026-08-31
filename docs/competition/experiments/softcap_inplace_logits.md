# Task 40 `softcap_inplace_logits` 实验记录

## S0：T24 跨芯 winner 复用

状态：commit-bound release 与 canonical ZIP 门禁通过，待一次性平台提交。

### 契约与实现

- 签名：`softcap_inplace_logits(full_logits, final_logit_softcapping)`；原地计算
  `tanh(full_logits / cap) * cap`，返回同一 buffer，shape 与 dtype 不变。
- 连续输入整体展平；非连续输入沿用 SGLang 上游约束，只接受二维且末维连续，按
  `row_stride` 寻址。统一 1D block-id + grid-stride，避免二维物理 grid 上限。
- generic 使用 `BLOCK=1024`；Ascend 使用 T24 已验证的 `BLOCK=512/grid=48`；
  Enflame 使用 `BLOCK=32768/grid=12/native tanh/constexpr cap`；Kunlun 使用
  `BLOCK=4096/native tanh`。后三条均有 T24 同公式目标芯 8/8 平台正证据。
- 数值路径沿用 T24：小值五阶近似、其余 `exp` 形式，并保留极小 cap 的零点
  `NaN` 语义；Enflame/Kunlun 使用各自官方 libdevice `tanh`。

### 构建身份

| 项目 | 值 |
| --- | --- |
| source / verification commit | `c05692530af49a309dc3081f48842bb7a9b45290` |
| generic SHA-256 | `6c3bd08a7b2ad52d38aa23980e2b66b3719233b55b5aac69f3ccce633b856f98` |
| Ascend SHA-256 | `46faecf8f5ef853b798a072f56c03487cac37179201ecdfbfe5d756460cf907f` |
| Enflame SHA-256 | `b5d049fd12a3d90388884e393fc4f052aa7817640fb9b6c39b857e003836a7dc` |
| Kunlun SHA-256 | `7aaf803413bc64bc3115204cf83daa4a8779f88d6f641500e1b6dd519a7d6dfd` |
| test SHA-256 | `1bb5ec58c81999ad2a0a925889d2ca47b8382c8f298c35abe6d1572a48a73ec6` |
| canonical ZIP | `artifacts/competition/softcap_inplace_logits/s0-c056925/softcap_inplace_logits.zip` |
| ZIP size / SHA-256 | `13136` bytes / `74e3fbb4b9030251085e8aa77d39177cc9f03c0e601f746593c8ebe9103dd69b` |
| ZIP members | generic、`_ascend`、`_enflame`、`_kunlunxin`，共 4 个普通 UTF-8 `.py` |

### 验证证据

- screening：`gpu:/tmp/flagos-t40-screen.JZS4ay`，最终 v4 从工作树精确传入上述
  source/test 字节；`py_compile`、black、isort、flake8、5/5 unittest 与
  `SCREENING_OK` 全过；日志 SHA-256
  `67479c8ceeff924856c983a9409e3b60abf55f7586d89ab4e199b4a0fe8d9b04`。
- release：`gpu:/tmp/flagos-t40-release.Rmq7xd`，只从 commit Git 对象生成；
  前后哈希一致，静态门禁与 5/5 unittest 全过，`RELEASE_OK`；日志 SHA-256
  `c5ac92b7c7581121e0ac7739deaed98998f7269afedd910f6e3d2c5a90f6a591`。
- 回归覆盖 FP16/BF16/FP32、多维连续、二维 row-stride、四条 vendor、各 tile
  边界、空张量、NaN/Inf/正负零以及 cap=`0/2^-128/Inf/NaN`；oracle 使用题面
  原 dtype 公式，并验证返回指针未变及 padding 未被覆盖。
- RTX 5070 Ti wrapper-inclusive 代理：四个代表 shape 上相对上游 SGLang kernel
  为 `0.9896–1.0606x`，相对 Torch reference 为 `1.5434–1.8236x`；benchmark
  日志 SHA-256 `912254b54c80e905d6c1ea871f456c3af4e8e9f3387ed642348fc5c41fb21177`。
  该卡只证明语法、数值与候选未明显回退，不外推其他芯片速度。
- canonical ZIP 已通过构建器 `--verify-existing`、`unzip -t/-l`、成员哈希与
  10 MB 上限验签；ZIP 字节与 source commit 完全对应。

### 提交预注册

2026-08-30 02:18 CST 实时只读状态：账号 `15600308080`、团队 `SoulCoder`、
Task 40/`softcap_inplace_logits`、batch 3、`competing/submitting`、
`can_submit=true`、action=`challenge_operator`；本队尚无该题提交，额度 `20/30`。
公开榜首为 Warmhearted `1.71430208x`（15 次提交、4 队）。

S0 只允许上述 ZIP 提交一次。基础门为 8/8 `valid`；冲榜门为平均严格高于
`1.71430208x`。平台应为天数/沐曦/海光/card A/card B 选择 generic，燧原选择
`_enflame`、昆仑选择 `_kunlunxin`、华为选择 `_ascend`。若单芯失败，只修该
vendor；若 8/8 但未登顶，只根据逐芯结果选择一个已解释的 BLOCK/grid/math 轴，
不重传 S0。正式 preflight 必须重新读取实时额度与最小提交间隔。

### S0 平台终态（sub 6606，2026-08-30 02:20 CST）

实时 preflight tuple 全匹配，额度 `20/30`；S0 只提交一次，提交后额度
`19/30`。对象存储匿名回读 `13136` bytes，SHA-256 与 canonical ZIP 完全
一致；`file_url_sha256=b82b01724eb24e3310ee7f56c494ac8655b618ffce085682d31082846ba73d8f`。
平台选中文件与预注册完全一致。

终态 **8/8、valid、team best、平均 `1.60635417x`**，公开第 3/4，未超过
Warmhearted `1.71430208x`：

| 芯片 | S0 | 榜首同芯 | 差值 | 文件 |
| --- | ---: | ---: | ---: | --- |
| 天数 | `2.09208333x` | `2.60883333x` | `-0.51675x` | generic |
| 沐曦 | `1.36608333x` | `1.51166667x` | `-0.14558334x` | generic |
| 燧原 | `2.33000000x` | `0.81900000x` | `+1.511x` | `_enflame` |
| 海光 | `1.98975000x` | `2.20950000x` | `-0.21975x` | generic |
| 昆仑 | `0.24575000x` | `0.94533333x` | `-0.69958333x` | `_kunlunxin` |
| 华为 | `0.61441667x` | `1.31858333x` | `-0.70416666x` | `_ascend` |
| 国际 A | `2.07975000x` | `2.10391667x` | `-0.02416667x` | generic |
| 国际 B | `2.13300000x` | `2.19758333x` | `-0.06458333x` | generic |

S0 的 Enflame 为公开四队最高；两个国际芯与榜首贴近。下一轮先保持三份 vendor
逐字节冻结，只把 generic 的手写 `exp` 降低替换为 SGLang 上游
`libdevice.tanh`。该单一 math-lowering 轴同时命中天数/沐曦/海光/card A/card B；
五芯榜首合计比 S0 高 `0.97083334x`，理论均分空间 `+0.12135417x`，足以覆盖
当前 `0.10794791x` 榜首差距。若 E1 未提升平均则永久停止 generic-native 轴，
转向 Kunlun/Huawei vendor。

## E1：generic 上游 native `tanh`

状态：commit-bound release 与 canonical ZIP 门禁通过，待一次性平台提交。

E1 从 S0 分叉，只把 generic 的 Taylor/`exp` lowering 换成 SGLang 上游
`triton.language.extra.libdevice.tanh`；BLOCK1024、grid-stride、row-stride、
cap 缩放与极小 cap 保护不变。Ascend、Enflame、Kunlun 与 S0 逐字节冻结。
该 generic 预期命中天数、沐曦、海光、国际 A/B；S0 已证明国际 A/B 仅落后
榜首 `0.0242/0.0646x`，主要收益目标是前三芯。

| 项目 | 值 |
| --- | --- |
| source / verification commit | `f3d135212b126ba705898d6a6fbe5c456681fd92` |
| generic SHA-256 | `8740e8e9f6332046bbfc04a8f0f3f69e3e1067a3447dd520889a499cbe9a99c0` |
| Ascend SHA-256 | `46faecf8f5ef853b798a072f56c03487cac37179201ecdfbfe5d756460cf907f`（=S0） |
| Enflame SHA-256 | `b5d049fd12a3d90388884e393fc4f052aa7817640fb9b6c39b857e003836a7dc`（=S0） |
| Kunlun SHA-256 | `7aaf803413bc64bc3115204cf83daa4a8779f88d6f641500e1b6dd519a7d6dfd`（=S0） |
| test SHA-256 | `1bb5ec58c81999ad2a0a925889d2ca47b8382c8f298c35abe6d1572a48a73ec6` |
| screening | `gpu:/tmp/flagos-t40-screen.JZS4ay`；5/5、`SCREENING_OK`；日志 SHA-256 `05f61d9cd571186074c1c9c80c55934873662a10ce804cc86eabd28c663095f3` |
| release | `gpu:/tmp/flagos-t40-e1-release.W2gK20`；5/5、前后哈希一致、`RELEASE_OK`；日志 SHA-256 `ddaf2bff5b665f24018d3ba13f899520870d19d21c590ab4b69ec0acbe0d99d9` |
| canonical ZIP | `artifacts/competition/softcap_inplace_logits/e1-f3d1352/softcap_inplace_logits.zip`，`12961` bytes，SHA-256 `a4aeef579774d54db3f019e243ffc7e2b0718cd29a41c4c9dab292eae0acc0e1` |

RTX 5070 Ti 四个代表 shape 上 E1 相对上游同字节数学为 `0.9822–1.0120x`，
相对 Torch 为 `1.4658–1.8200x`；日志 SHA-256
`95ea88955bdd78538d10d025d521652fb632f36341d4d07e1940d04e05036cfd`。
代理未见结构回退，但不外推 native lowering 在其他芯片的收益。

E1 基础门为 8/8 valid、平均严格高于 S0 `1.60635417x`；冲榜门为严格高于
实时榜首 `1.71430208x`。若平均不升，保留 S0 并永久停止 generic-native；若
提升但未登顶，冻结新的 generic best，再单独测试 Kunlun 或 Ascend 一个 vendor
轴。E1 ZIP 只允许提交一次。

### E1 平台终态（sub 6609，2026-08-30 02:25 CST）

实时 preflight tuple 全匹配，额度 `19/30`；E1 只提交一次，提交后额度
`18/30`。对象存储匿名回读 `12961` bytes，SHA-256 与 canonical ZIP 完全
一致，remote verification=`verified`；`file_url_sha256=f87e805925607906d484371dd784c8bf5f088b2c1eb09a91bc72354185a7afbe`。

终态 **8/8、valid、team best、平均 `1.64087500x`**，较 S0 提升
`+0.03452083x`，但仍低于 Warmhearted `1.71430208x` `0.07342708x`：

| 芯片 | E1 | 相对 S0 | 榜首同芯 | 文件 |
| --- | ---: | ---: | ---: | --- |
| 天数 | `2.06925000x` | `-0.02283333x` | `2.60883333x` | generic |
| 沐曦 | `1.30958333x` | `-0.05650000x` | `1.51166667x` | generic |
| 燧原 | `2.32116667x` | `-0.00883333x` | `0.81900000x` | `_enflame` |
| 海光 | `2.28450000x` | `+0.29475000x` | `2.20950000x` | generic |
| 昆仑 | `0.24383333x` | `-0.00191667x` | `0.94533333x` | `_kunlunxin` |
| 华为 | `0.71291667x` | `+0.09850000x` | `1.31858333x` | `_ascend` |
| 国际 A | `2.03900000x` | `-0.04075000x` | `2.10391667x` | generic |
| 国际 B | `2.14675000x` | `+0.01375000x` | `2.19758333x` | generic |

E1 证明 native generic 对海光有大幅正收益，但天数、沐曦和国际 A 回退；
因为三份 vendor 与 S0 字节不变，燧原、昆仑和华为的差值只作为平台测量
波动，不归因于 E1。按预注册停止继续改 generic，下一轮只测一个 vendor
math-lowering 轴。华为当前距榜首 `0.60566666x`；若其他七芯不变且追平该单芯，
理论平均为 `1.71658333x`，足以超过当前榜首，因此 E2 优先 Ascend native
`tanh`，保持 `BLOCK=512/grid=48` 不变。

## E2：Ascend 官方 native `tanh`

状态：commit-bound release 与 canonical ZIP 门禁通过，待一次性平台提交。

E2 从 E1 分叉，只把 `_ascend` 的 Taylor/`exp` lowering 替换为项目
`tl_extra_shim.tanh`。该 shim 由 Ascend backend 按 Triton 版本动态选择
`triton.language.extra.ascend` 或 `triton.language.extra.cann`；保留已由 T24 S4
平台证明的 `softcap_const * tanh(...)` scalar 左乘、`BLOCK=512`、
physical grid cap 48、grid-stride、row-stride 和极小 cap 保护。generic、
Enflame、Kunlun 和测试与 E1 逐字节冻结。T18 已实证华为 worker 能导入
该 shim 并编译执行 `exp`，但 `tanh` symbol 仍是本轮待平台验证假设。

| 项目 | 值 |
| --- | --- |
| source / verification commit | `1f1903c83d415e7d051fa0aa34ccc0cc842d0ffe` |
| generic SHA-256 | `8740e8e9f6332046bbfc04a8f0f3f69e3e1067a3447dd520889a499cbe9a99c0`（=E1） |
| Ascend SHA-256 | `7cd35d10f63241217f3c4f2116e421db7a27da363a9a116d7a44a4702088a101` |
| Enflame SHA-256 | `b5d049fd12a3d90388884e393fc4f052aa7817640fb9b6c39b857e003836a7dc`（=E1） |
| Kunlun SHA-256 | `7aaf803413bc64bc3115204cf83daa4a8779f88d6f641500e1b6dd519a7d6dfd`（=E1） |
| test SHA-256 | `1bb5ec58c81999ad2a0a925889d2ca47b8382c8f298c35abe6d1572a48a73ec6`（=E1） |
| screening | `gpu:/tmp/flagos-softcap-inplace-e2-screen.FVh7Ul`；5/5、`SCREENING_OK`；日志 SHA-256 `0e85275d290ad2f06c99c48a1e9e34ffa68cf056d7387a3a7435348e9a67c023` |
| release | `gpu:/tmp/flagos-softcap-inplace-e2-release.5qfCIn`；5/5、commit 字节哈希一致、`RELEASE_OK`；日志 SHA-256 `d9af6a930c911464b4e69de0cbe23944448e4a8e1326776f0c95243c4a92d7db` |
| canonical ZIP | `artifacts/competition/softcap_inplace_logits/e2-1f1903c/softcap_inplace_logits.zip`，`12865` bytes，SHA-256 `a2c0097854c07581125c982c7b606f6caeff19e485b96ff68385754ee5f01ceb` |

RTX 5070 Ti 上的非目标代理将 E2 与同 `BLOCK/grid` 的 E1 Ascend 手写数学
逐项对照；四个代表 shape 上 native/manual 为 `0.8637–0.9616x`，即 CUDA
上慢约 4%–16%。candidate 与 control 日志 SHA-256 分别为
`461b199ede43002f37a051c85ce9ce958c207be16bfb5d010e157f5cbe757556` 和
`c0f31312a404dce666c9aca74ae3c3eb20babf17c01f59b1c0b22019012932ad`。这只是数值、
编译和资源负代理证据；CUDA libdevice 与 48-worker cap 时延不外推为 Ascend
收益。

2026-08-30 02:35 CST 公开实时榜首仍为 Warmhearted
`1.71430208x`，SoulCoder E1 为 `1.64087500x`。E2 基础门为 8/8 valid；
单轴晋级门为华为严格高于 E1 `0.71291667x` 且平均严格高于
`1.64087500x`；冲榜门为平均严格高于 `1.71430208x`。若华为编译失败或
不升，永久停止 Ascend native-`tanh` 轴；若升分但未登顶，冻结 E2 Ascend
字节并转向 Kunlun 的单一 BLOCK 轴。E2 ZIP 只允许提交一次。

### E2 平台终态（sub 6611，2026-08-30 02:37 CST）

实时 preflight tuple 全匹配，额度 `18/30`；E2 只提交一次，提交后额度
`17/30`。对象存储匿名回读 `12865` bytes，SHA-256 与 canonical ZIP 完全
一致，remote verification=`verified`；`file_url_sha256=31f971f62af0229f2fca71fb18c8e346d4b3f40433cc6f1199df35bf4f50ad19`。

终态 **8/8、valid、team best、平均 `1.66054167x`**，较 E1 表面提升
`+0.01966667x`，公开榜第 3/4，仍低于 Warmhearted `1.71430208x`
`0.05376041x`：

| 芯片 | E2 | 相对 E1 | 文件 |
| --- | ---: | ---: | --- |
| 天数 | `2.06483333x` | `-0.00441667x` | generic |
| 沐曦 | `1.45358333x` | `+0.14400000x` | generic |
| 燧原 | `2.33116667x` | `+0.01000000x` | `_enflame` |
| 海光 | `2.26750000x` | `-0.01700000x` | generic |
| 昆仑 | `0.24566667x` | `+0.00183334x` | `_kunlunxin` |
| 华为 | `0.67075000x` | `-0.04216667x` | `_ascend` |
| 国际 A | `2.10025000x` | `+0.06125000x` | generic |
| 国际 B | `2.15058333x` | `+0.00383333x` | generic |

本轮唯一变化的华为明确回退 `5.91%`，未过单轴晋级门；未改字节的
七芯合计测量波动盖过了该回退，不能把平均升分归因于 native `tanh`。
保留 E2 作为平台 team best，但永久停止 Ascend native-`tanh` 轴；后续候选
恢复 E1 已验证的 Ascend 手写字节，新实验只改 Kunlun 的一个 BLOCK 参数。

## E3：Kunlun 连续输入 direct fast path

状态：平台 8/8、`valid`、公开第 1/5，E3 为 team best。

E2 终态后对 T24 全曲线做了更强的机制对照，因此在任何 E3 候选建立前
替换了原先的 BLOCK1024 计划：T24 direct BLOCK1024→4096 为
`0.7645→0.8637x`，同 BLOCK4096 改成 grid-loop 后突降至 `0.24175x`，恢复
direct 并使用 native `tanh` 后为 `0.97591667x`。T40 E0–E2 的 Kunlun
`0.2438–0.2457x` 与 T24 loop 指纹几乎一致，说明更高置信根因是动态
loop/control，不是 BLOCK 或 native `tanh` 本身。

E3 只在 `_kunlunxin` 为 contiguous 且 logical blocks `<=65535` 的输入新增
一程序一 BLOCK4096 的无循环 direct kernel；非连续二维 row-stride 和超大
grid 继续走 E2 原 loop fallback。已知 T24 最大 `65,667,072` 元素只需
`16,032/65,535` programs；T40 代理最大 `9,723,904` 元素只需
`2,374` programs。E2 被否决的 Ascend native 先在独立 restore commit
`b40c21d5af8fa85734250888fc2813fb2278a8a2` 恢复为 E1 手写字节；这不是
E3 新假设。generic、Ascend、Enflame 和测试均为已验证字节，E3 新变量
仅为 Kunlun direct dispatch。

| 项目 | 值 |
| --- | --- |
| source / verification commit | `248be7f9c3484667e884c5731023de9332b5f558` |
| generic SHA-256 | `8740e8e9f6332046bbfc04a8f0f3f69e3e1067a3447dd520889a499cbe9a99c0`（=E1/E2） |
| Ascend SHA-256 | `46faecf8f5ef853b798a072f56c03487cac37179201ecdfbfe5d756460cf907f`（=E1） |
| Enflame SHA-256 | `b5d049fd12a3d90388884e393fc4f052aa7817640fb9b6c39b857e003836a7dc`（=E1/E2） |
| Kunlun SHA-256 | `56f5350374104216da658f207bf624681bc9a4c6ba774f3e49714bb5595e1ba1` |
| test SHA-256 | `1bb5ec58c81999ad2a0a925889d2ca47b8382c8f298c35abe6d1572a48a73ec6`（=E1/E2） |
| screening | `gpu:/tmp/flagos-softcap-inplace-e3-screen.HBbBVL`；5/5、`SCREENING_OK`；日志 SHA-256 `0bf26c2abf505182ea5a570fcba3247a8e8f0b9f0be1471728bcdfa7d21ec530` |
| release | `gpu:/tmp/flagos-softcap-inplace-e3-release.NZaKIQ`；5/5、commit 字节哈希一致、`RELEASE_OK`；日志 SHA-256 `6f448407e7d1ae8f2ac213de8c5027ef544ac2f86981ce3abc22abcfe044c0fc` |
| canonical ZIP | `artifacts/competition/softcap_inplace_logits/e3-248be7f/softcap_inplace_logits.zip`，`14016` bytes，SHA-256 `13be219e67dd7c2b8ecec223bd75265e9124596cb0c5a4cb1f99f9e995464af9` |

RTX 5070 Ti 上 direct/loop 四个代表 shape 为 `1.010–1.095x`；direct 相对上游
原生布局为 `0.739–1.001x`，相对 Torch 为 `1.121–1.819x`。candidate/control
日志 SHA-256 分别为
`58fa59e28e02f3e671c28a20523f51f8ff41ebb42fe02804068f1d400cb59574` 与
`7d6283f4b46c6c33732adfffc8d55cfc88f7d4020ed5b7ab59aba70cdf24508a`。九份实际
TTIR 均为 `scf.for=0`，代表编译元数据为 4 warps、1 stage、0 shared、0 global
scratch；TTIR 证据日志 SHA-256
`f4d3c134ed408cb8050730d8a640262063b41012169d6cd6be00779b4c85b82d`。CUDA 收益仍不
外推至 XPU，关键正证据是 T24 同数学、同 BLOCK 的平台路径对照。

2026-08-30 02:43 CST 实时榜首仍为 `1.71430208x`，SoulCoder 为
`1.66054167x`。E3 基础门为 8/8 valid；单轴晋级门为 Kunlun 严格高于
E2 `0.24566667x` 且平均严格高于 `1.66054167x`；机制确认门为
Kunlun `>=0.76x`，冲榜门为平均严格高于 `1.71430208x`。若 valid
但 Kunlun `<0.76x`，则 T24 direct 机制未迁移，停止 Kunlun BLOCK/direct 扫描；
若登顶，冻结 E3 字节并转新题。E3 ZIP 只允许提交一次。

### E3 平台终态（sub 6612，2026-08-30 02:45 CST）

实时 preflight tuple 全匹配，额度 `17/30`；E3 只提交一次，提交后额度
`16/30`。对象存储匿名回读 `14016` bytes，SHA-256 与 canonical ZIP 完全
一致，remote verification=`verified`；
`file_url_sha256=45496f6b7c10f11689b8bfceb744fb8bdd351a934c93bbe4a5570b6c3db8939a`。

终态 **8/8、valid、team best、平均 `1.72662500x`**，较 E2 提升
`+0.06608333x`。02:46 CST 官方公开接口显示 19 次提交、5 支队伍，当前榜首为
SoulCoder `1.72662500x`，领先原榜首 Warmhearted `1.71430208x`
`0.01232292x`，公开第 **1/5**：

| 芯片 | E3 | 相对 E2 | 文件 |
| --- | ---: | ---: | --- |
| 天数 | `2.06650000x` | `+0.00166667x` | generic |
| 沐曦 | `1.46100000x` | `+0.00741667x` | generic |
| 燧原 | `2.32900000x` | `-0.00216667x` | `_enflame` |
| 海光 | `2.29383333x` | `+0.02633333x` | generic |
| 昆仑 | `0.94450000x` | `+0.69883333x` | `_kunlunxin` |
| 华为 | `0.55666667x` | `-0.11408333x` | `_ascend`（恢复 E1 字节） |
| 国际 A | `2.02666667x` | `-0.07358333x` | generic |
| 国际 B | `2.13483333x` | `-0.01575000x` | generic |

唯一新机制所在的昆仑从 `0.24566667x` 提至 `0.94450000x`，绝对提升
`0.69883333x`、达到 E2 的 `3.844x`，同时超过预注册 `0.76x` 机制门，证明
grid-stride 动态 loop/control 是此前约 `0.245x` 指纹的主因。其余芯片没有 E3
新变量；华为虽恢复为 E1 手写源码，当前分数仍只视为平台波动，不据此重开已否决的
native-`tanh` 轴。按预注册冻结 E3 全部字节，停止 Task 40，转入下一道高收益题。

### E4:华为 direct vendor 赌注(2026-08-31 03:2x CST)

- 昆仑已证模式(0.25→0.94)推广:连续输入+网格不超限 → 无循环
  BLOCK 4096 直通 kernel,否则回退;标准 libdevice tanh;
- commit `fbb95a0`,ZIP `3b901379…`,4 成员;unittest 5/5(全 vendor
  矩阵);额度 23/30。华为需 0.557→≥1.69 才单芯追平榜首——低概率
  高回报,失败即关轴。

### E4 终态(sub 7054)与 E5 机械修复

- E4:7/8,**华为 direct kernel 编译失败**(BLOCK 4096 超昇腾 UB,
  CompilationError)——15 case 全 compile error,其余七芯正常;
- E5:direct BLOCK 降为 512(T24 softcap_out 昇腾平台已证尺寸),
  属编译错误机械修复类;unittest 5/5。

### E5 提交(2026-08-31 03:5x CST)

preflight 全过(额度 22/30),单次提交;终态待回填。
