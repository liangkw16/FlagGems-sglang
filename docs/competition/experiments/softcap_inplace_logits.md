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
