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
