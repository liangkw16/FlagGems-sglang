# Task 96 `fused_pack_qkv` 实验记录

```current
task: 96
operator: fused_pack_qkv
batch: 7
validity: valid(7/7,e1)
platform: s0(21124)invalid_threshold昆仑0.010;e1(21146)valid 7/7:昆仑0.010→0.583(57x,vendor 1D结构兑现)/华为0.950/天数2.95/海光3.15
candidate_stage: e2
team_best_stage: e1
team_best_speedup: -
sealed: yes
next: E2天数_iluvatar vendor(per-row 1D+int32域+单alloc,评审r1)armed-unfired,平台单发裁决;预注册门:天数≥4.0保留/≥4.6进场带/均值>1.72675换TB/数值失败或<2.95回滚删vendor文件回e1字节;同字节重掷≤2;其余六芯vendor隔离仅tianshu选中;E3-generic-i32单变量归因留后续
updated: 2026-09-26
```

## 契约

- `(q,k,v,indices)`：`[B,S,H,D]` 三张量按 flat `B*S` 索引 gather 成三个
  `[total_valid,H,D]`；indices int32/int64；exact 纯 gather。

## 榜单靶标

EvokeAgent 6.00 分（仅 2 队可见）：tianshu 5.512 / muxi 2.446 / haiguang 3.266 /
**kunlun 0.606** / **huawei 0.617** / A 1.481 / B 1.392。两队长芯 kunlun/huawei 均 <1：
单 kernel 行 gather 结构大概率直接拿下这两芯 #1。

## S0 结构（commit b6641097）

- 单 kernel：BLOCK_R=4 行 × BLOCK_C=1024 列 2D 瓦片；`indices` 每行 load 一次按列广播；
  一个 program 内完成 q/k/v 三份拷贝（1 launch vs baseline 3 gather + long cast）；
  `idx.to(tl.int64)*row_elems` 防溢出；num_warps=4（sweep @HD2048 26.46µs vs torch 28.38µs）。

## 证据

- NVIDIA 代理 screening（r8 旧配置）5 测试 0 失败 11 launch；r4 新配置复筛中。
- proxy L2 常驻下 HD512 1.65x / HD2048 1.07x；平台冷数据口径下结构优势更大。

## 情报

- 快照 SHA-256 `061475d31505bb81f16a5a212fa7cfd6e4b1142188cb02da2c202962efedadb2`。

## 不可变身份（s0 v2，2026-09-24，契约 spec 高阶复审后）

- source commit = verification commit = `daf77923f6913447b65a8c226f34297ce13b4440`（review v2：补 indices.stride(0) + 非连续 q/k/v 归一化（替代断言））。
- ZIP：`artifacts/competition/fused_pack_qkv/s0-daf7792/fused_pack_qkv.zip`，SHA-256 `4edde6d3aacc05c8dd63dee86dfd0d782bc707ae3fa217a463701dcb80705da0`（单成员 `fused_pack_qkv.py`，generic-only）。
- release 回执：`artifacts/competition/b7-s0-release2-20260924/fused_pack_qkv/verification.json`（mode=release，exit 0，绑定该 commit 字节），SHA-256 `98b7f6f9c603979336da0f3a66b9f9dab8696dd65b2fa289491b20ee761e3f94`；
  日志 SHA-256 `7e5058c05673e57ab61e2f9c9dc6563feee7d090443a8b9048d37f8a56b3f578`。NVIDIA RTX 5070 Ti / torch 2.13.0+cu130 / triton 3.7.1。
- 提交脚本：`artifacts/competition/b7-s0-release2-20260924/submit-batch7-s0.sh`（v2，全部参数预烘焙）。
- review 历史：v1（commit 级，gpt-6-astra medium）4 项全修；v2（--base 对照契约 spec，high）5 项 P2 全修；
  三轮修复均经 screening + release 双门禁复跑全绿。

## E2 候选：天数 `_iluvatar` vendor（per-row 1D + int32 寻址域 + 单 alloc，评审 r1，2026-09-26）

**假设**：用上游原型 per-row（grid=n_valid，一行一程序）1D 单段拷贝替换自研
BLOCK_R=4×1024 2D tile，叠加全链 int32 寻址（域内）与 3 alloc→1 单 buffer 三 view，
可把天数从 2.95 拉进对手群众带 4.6-6.1。

**缺口证据**（climb-loop s2t1op096，2026-09-26T01:50 快照，分诊员实跑复核）：我方天数
2.9505 全场 14 队垫底，其余 13 队 3.447-6.148（榜首 c2flow 5.5245、Sweetdeath 6.148）；
天数→5.52 = 均值 +0.366，占均值缺口 58%。platform_cli status 实查（只读）
s0(578dfe02)→e1(b25a11c2) 天数同 generic 字节读数 2.95575/2.9505（差 0.2%），排除水位、
判定纯结构差。

**机制证据**（4 条）：
1. 上游 sglang 原型 `varlen_pack_pad_triton.py::_fused_pack_qkv_kernel` 即 per-row
   （grid=(n_valid,) 每程序一行）+ BLOCK_HD=next_pow2(hd) 1D 单段 + 标量 idx load；
   对手群众带 4.6-6.15 与我方 2D tile 2.95 构成结构对照。
2. T51 E9 平台证据：行块（16 行/程序）在天数 -20%（fla_layernorm_gated.md:386-388）。
3. 本题内部同族：昆仑 2D tile 0.010→1D 行程序 0.583（57x，`_kunlunxin` vendor e1 平台兑现；
   分诊员实跑复核 s2t1op096 kunlunxin=0.5830 ✓）；天数从未测过任何非 2D 形态。
4. int64 全链（generic `fused_pack_qkv.py:30-38` 五处 int64：rows/idx/src/dst/cols）是
   三题族共同特征且天数系统性落后（T95 per-row int64 82.3 vs AICity 139.8；T98 2D int64
   2.37 vs c2flow 2.91；跨芯负资产先例：燧原 enable_i64=False、华为 Vector ADD 无 int64，
   chip-rulesets.md）。

**实现**（`src/flaggems_sglang/runtime/backend/_iluvatar/ops/fused_pack_qkv.py`，新增文件）：
- `_pack_row_i32_kernel` / `_pack_row_i64_kernel` 双 Triton 孪生：per-row grid=(n,)、
  标量 idx load、`BLOCK_C=min(65536, max(128, next_pow2(row_elems)))` 单段掩码拷贝；
  列循环仅在 H*D>65536 时兜底（宽行回归 70000 覆盖）。i32 版全链不 cast（int32 寻址），
  i64 版与 generic 五处 cast 逐位对齐（溢出域安全网）。
- host 分派 `_use_int32(q_numel, n_rows, idx_s0)`：数值域判定（非设备判断，T80 e30 同构）。
  界：q.numel() < 2^31 且 (n_rows-1)*idx_s0 < 2^31（覆盖 strided indices load 深度；
  src=idx*row_elems<numel、dst=n*row_elems≤numel 由语义 n≤B*S 保证）。
- 单 alloc：`torch.empty(3*n*row_elems)` 一块平 buffer，q/k/v 三输出为三个连续 view
  （T77 e6/e17 先例：天数 +8.6%、昆仑 +91%）。
- num_warps=4（generic 在 HD2048 的 proxy sweep 值）；纯 Triton、无 try/except、
  无模块级可变容器。
- caveat：grid=(n,) 在 i64 域需 n≥2^31 且 row_elems=1 的退化形态才触 CUDA grid-x 上限，
  公开域（B*S*H*D≤6.3M）不可达，记录不设防。

**测试**：`tests/test_fused_pack_qkv.py` 新增 `test_iluvatar_int32_domain_guard`
（2^31 边界四断言：最大合法域 True / numel==2^31 False / strided indices 越界 False /
空 gather True），列入 RELEASE_REQUIRED_TESTS；既有矩阵（shapes/tails、int64+重复索引、
fp32/bf16、sliced indices、非连续 q/k/v、70000 宽行、全量+空）自动经
`_op_variants.load_operator_modules` glob 发现并全量跑 i32 路径。

**覆盖度声明**：本候选 target-runtime-unverified（天数 lowering 本地不可验证）；
i64 孪生 kernel 在 CI 域（<2^31 元素）不可达、未被执行，仅靠与 generic 已证 i64
数学逐位同构 + 守卫单测锚定边界；NVIDIA 代理只验数学/JIT，不可外推天数性能，
平台单发裁决。本轮（评审 r1）本地验证为 py_compile + 守卫函数实跑断言，
代理 GPU 矩阵留待 release 轮。

**预注册门**（armed-unfired）：天数 ≥4.0 保留 / ≥4.6 进场带（13 队群众带下沿）/
均值 >1.72675 换 TB；数值失败或天数 <2.95（回退）即回滚删 vendor 文件、树回 e1 字节；
同字节重掷 ≤2 次；其余六芯 vendor 隔离（仅 tianshu 选中 `_iluvatar`，
映射 E9 五发实证 fused_recurrent_gdn.md:548）读数应不变。

**归因注记**：三变量打包（形态+i32+单 alloc）归因模糊，E3-generic-i32 单变量留作
后续归因补充而非本轮同题并发。单芯兑现不翻榜（均值→约 2.09 < 榜首 2.3555），
属大幅抢分 + 打开天数轴。
