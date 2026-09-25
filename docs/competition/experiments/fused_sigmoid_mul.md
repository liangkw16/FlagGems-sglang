# Task 97 `fused_sigmoid_mul` 实验记录

```current
task: 97
operator: fused_sigmoid_mul
batch: 7
validity: valid(7/7,s0)
platform: e2(21267)valid avg3.052(水位推高天数5.26/沐曦3.16):grid48中性(华为1.311≈1.340);昆仑vendor字节错误带16384(0.821)已回滚2048;TB=e2(avg口径)但昆仑rank最优在s0的1.166
candidate_stage: e3(开发员r2已commit未发射:ascend flat两段式无mask+r2 int32溢出防护(2^31域分派+i64冷备)+同包昆仑2048恢复;远端release验证未做)
team_best_stage: e2
team_best_speedup: -
sealed: yes
next: 先跑release门禁绑定本commit字节;等T93 E3平台裁决校准华为方向后发射;门:华为≥1.6保留/≥1.9判轴兑现,<1.311或数值失败回滚_ascend至5cfdb654字节(blob ab88bc50);昆仑<1.0判水位重掷(chip-rulesets:58)不归因2048
updated: 2026-09-26
```

## 契约

- `(attn_output, gate)`：`out = attn * sigmoid(gate)`，fp32 计算、按输入 dtype 存储；
  flat 同形路径 + strided 路径（gate 3D 非连续，需按 stride 读）；per-dtype 容差。

## 榜单靶标

EvokeAgent 5.00 分：tianshu 5.301(c2flow) / muxi 3.033 / haiguang 4.102 /
**kunlun 2.007** / **huawei 1.486**（EA 独占两芯第一，他队 0.60-0.96）/ A 3.776 / B 3.150。
T90/T81 华为经验（子块结构 1.23→1.5 轴）可迁移。

## S0 结构（commit b6641097）

- 单 kernel flat map BLOCK=2048/w8；ATTN_CONT/GATE_CONT 双 constexpr 分支：
  连续时纯 flat 偏移；非连续时 t=offs//hidden 分解 + gate 3D stride 直读（无 contiguous 拷贝）；
  `tl.sigmoid` fp32（T90 已验证跨芯形式）。
- 代理 sweep：4096×4096 torch 825.75µs vs 我方 115.3µs（**7.1x**，torch 5 kernel）；
  strided3d 66.1µs。

## 证据

- NVIDIA 代理 screening（BLOCK=4096 旧配置）6 测试 0 失败 18 launch；b2048 复筛中。

## 情报

- 快照 SHA-256 `061475d31505bb81f16a5a212fa7cfd6e4b1142188cb02da2c202962efedadb2`。

## 不可变身份（s0 v2，2026-09-24，契约 spec 高阶复审后）

- source commit = verification commit = `daf77923f6913447b65a8c226f34297ce13b4440`（review v1 grid/块失配已修；review v2：strided 分支与 flat offs 全面 i64 化）。
- ZIP：`artifacts/competition/fused_sigmoid_mul/s0-daf7792/fused_sigmoid_mul.zip`，SHA-256 `40003a157420b8a5598fede4bc02aea5051d149da29d451bac0043116a720b51`（单成员 `fused_sigmoid_mul.py`，generic-only）。
- release 回执：`artifacts/competition/b7-s0-release2-20260924/fused_sigmoid_mul/verification.json`（mode=release，exit 0，绑定该 commit 字节），SHA-256 `96836a78ee559aac6c11889a5b03ddcc97d0482aea0f3149f57fbeb9eb7ae122`；
  日志 SHA-256 `4cb32043bbc3d80ad5babd5a2bc8626c048a93a13a9fbb69869428ae698e8922`。NVIDIA RTX 5070 Ti / torch 2.13.0+cu130 / triton 3.7.1。
- 提交脚本：`artifacts/competition/b7-s0-release2-20260924/submit-batch7-s0.sh`（v2，全部参数预烘焙）。
- review 历史：v1（commit 级，gpt-6-astra medium）4 项全修；v2（--base 对照契约 spec，high）5 项 P2 全修；
  三轮修复均经 screening + release 双门禁复跑全绿。

## E3 候选预注册（开发员 r1，2026-09-26；已 commit 未验证未发射）

**缺口构成**（climb-loop.json s2t1op097，逐芯核对）：我方 3.05257936(#10) vs 金狐狸
3.31494444(#1)，差 0.2624 = huawei(1.311 vs 2.185 全榜#1，-0.1249) + card_b(3.122 vs
4.156 断层#1，-0.1477) + kunlunxin(0.821 vs 1.113，-0.0417) + card_a(3.705 vs 3.835，
-0.0186)；天数/沐曦/海光我方领先合计 +0.0705，四芯之和与榜均差精确相等。

**字节变更**（单 commit，`git log -1 -- 本账本` 即 source commit）：

- `_ascend/ops/fused_sigmoid_mul.py`：flat 连续热路径（attn 且 gate 连续）改两段式无
  mask——kernel 内 `pid < n_full` 标量分支、整块 2 载 1 存双无 mask 无 `other`、尾块
  masked 不带 `other`（store 同 mask，undef lane 不落存）、BLOCK=16384/w16
  （chip-rulesets.md:40 阶梯）、int32 寻址（:36 域内）；wrapper host 静态分派，两臂皆
  Triton kernel，无 try/except/设备判断。strided 路径保留 E2 persistent 字节不动
  （_TILE=1024/_PERSISTENT=48/w4）。回滚锚：改前 _ascend blob SHA-256
  `ab88bc50632db1b20f842849892881a3875a79ae49790b1572a519aae873ef8c`（git 5cfdb654）。
- `_kunlunxin/ops/fused_sigmoid_mul.py`：**本 commit 零改动**——树内已是回滚后的
  `_BLOCK=2048` 已证字节（1.166 两连 vs 16384 的 0.821），随同包恢复提交，不发新字节。
- 测试：`tests/test_fused_sigmoid_mul.py` 新增
  `FusedSigmoidMulTest.test_two_segment_block_boundaries`（numel=16383/16384/16385/
  32767/32768/32769/65536，覆盖标量分支两侧与无尾块整除格点），已列入
  RELEASE_REQUIRED_TESTS。py_compile 双文件通过（本地无 CUDA，远端 release 门禁未做）。

**依据**：三次提交差分锁定——b2db48ca(纯 generic) kunlun=1.166/huawei=1.022 →
e1e241c0(昆仑 16384) kunlun=0.821 → 9ab3c5c4(华为 persistent+grid48) huawei=1.311；
persistent 化 +28% 而 CTA 数轴中性，剩余瓶颈=每 tile 的 `offs<numel` 向量比较
（chip-rulesets.md:36 Vector CMP 降标量）+ `other=0` 预填（:39 MTE2 串行化）。迁移
模板=T93 E3 上膛字节（`_ascend/ops/add_constant.py`，树内核验同形态）；家族证据 T40
E16 零比较热路径华为+130%、T39 e10 整块标量跳过+246%（见 add_constant.py docstring
所引 experiments README:984-990/:735）。

**负证据披露**：T92 E23 同概念（整块无 mask + masked 尾块）华为 -28.8% 平台实证——
彼为 int64 + 2D grid-stride + per-base 守卫，本候选 int32 + 1D 单 tile + kernel 级单
分支，家族证据未分解哪一因子致败，华为符号开放。T93 E3（同族 1 载 1 存）armed-
unfired，**等其平台裁决校准后再发射本候选**（昆仑 2048 恢复分量不必等）。

**预注册门**：华为 ≥1.6 保留 / ≥1.9 判轴兑现；<1.311 或数值失败 → 回滚 _ascend 至
5cfdb654 字节；昆仑 <1.0 判水位重掷（chip-rulesets.md:58 昆仑 ±10-35%）不归因字节。
保守核算：昆仑恢复 +0.049 + 华为按 GuanghuLab 水位 2.0 半成功率折算 ≈+0.05，均值
3.0526 → ~3.15。

### r2 评审修复（开发员，2026-09-26）：int32 寻址溢出防护

**发现（P2-1）**：r1 的 flat 两段式 `offs = pid*BLOCK + tl.arange(0,BLOCK)` 为纯
int32 且 host 无 numel 上限检查；numel ≥ 2^31+1 时尾块 base 回绕为 -2^31，全部 lane
通过 `offs < numel` → OOB 读写。generic/kunlunxin 均有 `.to(tl.int64)` 防护，本路径
是库内唯一例外，docstring "int32 addressing stays in-domain" 无代码保证。

**修复**（同 commit）：`_INT32_NUMEL_MAX = 2^31` + host 分派函数 `_flat_cold(numel)`
+ i64 冷备 kernel `_fused_sigmoid_mul_two_segment_i64`（同两段式结构、generic/昆仑
同款 cast），超限走冷备。域推导本地解析验证（逐 program 极值 int32 回绕模拟）：
numel=2^31-1/2^31 全 program 干净（2^31 时无尾块、max offs=INT32_MAX），2^31+1 尾块
base=-2^31 全 lane 过比较（复现评审发现），2^31+BLOCK 整块 OOB——`numel ≤ 2^31` 为
BLOCK=16384 的精确无溢出域（BLOCK 整除 2^31）。

**回归**：`test_two_segment_dispatch_boundary`（边界分派真值表 + 域整除性）与
`test_two_segment_i64_cold_variant_numerics`（临时下调模块分派界强制冷备走公共
seam，尾块/无尾块两格点全数值校验，finally 恢复）均入 RELEASE_REQUIRED_TESTS。
真实 2^31 元素执行需 3×4GiB(bf16) 超释放设备显存，不作为 skip 门禁（skip 会阻断
release 晋级），执行覆盖由强制冷备测试承担。

**如实陈述**：本地无 torch/triton（远端验证仓），分派 seam 本地仅 py_compile+AST
核验（module 级无 try、分派读模块全局为动态 seam），执行证据待远端 release 回执；
strided persistent 臂 flat 寻址维持 int32 旧字节（先在，>2^31 安全性本地不可验，
触发需单张 ≥4GiB bf16，竞赛 shape 不达，不阻塞发射门禁）；热路径 int32 是
chip-rulesets.md:36 的有意权衡，防护不改变 ≤2^31 域内字节行为。
