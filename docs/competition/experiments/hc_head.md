# Task 55 `hc_head` 实验记录

```current
task: 55
operator: hc_head
batch: 4
validity: invalid_correctness
platform: 7/8(e1,10668已终态;昆仑compile_worker Aborted,归因未定)
team_best_stage: -
team_best_speedup: -
sealed: no
next: E2(e2-afbe602)昆仑三平铺kernel vendor已过release门禁待单次平台裁决;假设=深嵌套generic在昆仑编译超时,预注册门=昆仑跑出结果且七芯维持
updated: 2026-09-07
```

## S0 每 token 两遍融合（2026-09-06，远端 GPU 1/1 OK）

- 形态：grid=(T,)；pass1 单遍同时累积 sumsq（RMSNorm）与
  mixes_raw（x·hc_fn[j] 3D tile [HC,HC,BH] 嵌套 tl.sum）；
  sigmoid 门控；pass2 重读 x（L2 热）折叠 hc_mult 轴写 y。
- BLOCK_H 限 4096//(HC²)；m 轴全 mask 防 OOB。
- 远端 5070 Ti：1/1 OK（T∈{1,3,4,8,16,33}×hc_mult∈{2,4,8}×hidden∈
  {128,256,320,512,1024,2048}，bf16 1.5e-2）。

## S0 平台进行中（2026-09-06，seq26）

- 已过 7：天数 1.59 / 沐曦 1.23 / 燧原 0.257 / 海光 3.22 /
  华为 2.04 / A 0.97 / B 4.45；昆仑评测中。

## S0 昆仑终态（2026-09-07 核实）

- seq26 昆仑失败详情：`执行超时(1830s/1800s) + Subprocess crash:
  Fatal Python error: Aborted`（torch inductor compile_worker 栈），
  属已知昆仑评测器崩溃族，不计代码止损；七芯成绩维持
  天数 1.59 / 沐曦 1.23 / 燧原 0.257 / 海光 3.22 / 华为 2.04 /
  A 0.97 / B 4.45。

## E1 扁平累加器 + 循环外归约（2026-09-07 已提交）

- 参照官方 FlagGems `hc_head_fused_kernel` 结构：`sqr_acc[BLOCK_H]` 与
  `mix_acc[HC,BLOCK_H]` 逐 lane 累加，fn 以 `[HC,BLOCK_H]` 2D tile 按
  输入行 m 展平流式载入，全部 `tl.sum` 移出 hidden 循环；第二遍加权
  输出不变。BLOCK_H 预算从 `[HC,HC,BH]≤4096` 改为 `HC*BH≤2048`
  （HC=8→256 / HC=4→512 / HC=2→1024），保持单 kernel。
- NVIDIA 代理 wrapper 基准（RTX 5070 Ti，中位数）：
  (T=33,hc=4,h=2048) 1.46x、(T=64,hc=8,h=2048) 1.52x、
  (T=8,hc=8,h=1024) 1.39x、(T=16,hc=2) 1.05x，全 shape 无回退；
  screening 1/1、release 6 launches 0 失败。
- source `0052424`，ZIP `e1-0052424` SHA-256
  `cf34629620dd7b945a428a06ca0f022bbf7218b0e80e8cbdef77674649d45cde`；
  2026-09-07 提交评测中。燧原（0.257x 弱芯）目标 ≥20%；昆仑同时重试。

## E1 终态（2026-09-07，submission 10668）

- **7/8，昆仑第 2 次同一崩溃指纹**（`compile_worker Aborted`，与 s0
  seq26 完全一致）；同日昆仑评测器正常完成 T48/T49/T53/T56 本队提交，
  证明是 hc_head 题评测侧（reference/inductor）确定性崩溃，非我方
  Triton 内核问题。按崩溃族协议不计代码止损。
- 七芯 vs s0：天数 1.38（-13%）/ 沐曦 1.10（-11%）/ 燧原 0.2995
  （+16.5%，未过 20% 门）/ 海光 2.156（-33%）/ **华为 3.153（+54%）/
  A 2.1815（+125%）/ B 9.088（+104%）**；七芯总和 13.76→19.36（+40%）。
  若昆仑恢复，e1 字节即为大幅 team best 候选；同字节重试不需新候选。
- 结构结论：循环外归约在华为/A/B 大幅兑现，海光/天数/沐曦小回退，
  与官方 FlagGems 结构在该三芯的 lowering 差异待后续分轴。

## 2026-09-07 只读盘点校正

旧节“确定性reference崩溃、非我方kernel问题”证据不足，现撤回该归因。两个提交的compile_worker/Aborted与其他题当天成功，只证明题目相关失败，不排除候选代码、编译与设备状态作用。本轮未再次提交。
- 查询证据 `/Users/bytedance/ccc/flagos/artifacts/competition/batch4-top1-20260907/tasks-now.json` SHA256 `fc73368c3d98b228b0c7815d6ec1e9042a58af8d953337fff58990daec1474fc`。
- 查询证据 `/Users/bytedance/ccc/flagos/artifacts/competition/batch4-top1-20260907/55-submissions-now.json` SHA256 `3a954f68db1ecbadc01b3e43a704eab3d8bd6929e57e34d2f77c9053b2864085`。

## E2 昆仑三平铺 kernel vendor（2026-09-07）

- 假设：两次 compile_worker 崩溃（1830s 超时 + Aborted）的触发面是
  generic 深嵌套结构（hidden 循环内逐 h0 展开 [HC,BLOCK_H] fn tile）在
  昆仑 FlagTree 的编译耗时/崩溃；T45 E16 刚定位的"复合谓词 mask 错译"
  进一步约束 vendor 形态。新 `_kunlunxin` vendor 按 T28 E11/T37 E4/T45 E8
  配方拆三个平铺规则 kernel：①grid-stride 行平方和（BLOCK 1024）；
  ②per-(token,j) mixes GEMV（纯向量 FMA 无 tl.dot，fp32；T45 证据昆仑
  fp32 dot 本就走标量路，无损）③per-(token,h-block) 折叠（HC 标量
  unroll + 512 宽向量，sigmoid/rsqrt 逐标量重算）。全部 mask 均为简单
  边界 `<` 谓词，无复合谓词、无 early-return、无 gather/广播操作数。
- generic 字节不动（七芯 e1 读数路径不变）；单变量 = 昆仑成员替换。
  数学顺序逐元素对齐 reference（mean 后 rsqrt、sigmoid((mix*r)*scale+base)、
  fp32 加权求和后单点 cast）。
- source/verification commit `afbe602`；screening `/tmp/flagos-t55e2`
  （2/2）与 release `/tmp/flagos-t55e2-rel`（2/2 方法、0 fail/error/skip/
  xfail，generic 10 调用/10 launch、kunlunxin 12 调用/12 launch 实跑）
  双绿；远端 black/isort/flake8 全过（black 重排后取回，hash 一致）。
  新增 `HcHeadVariantsTest`（4 shape × 全 variant）+ RELEASE_REQUIRED_TESTS。
- NVIDIA 代理性能不作依据（vendor 仅昆仑用）；目标 = 昆仑跑出首个结果
  （correctness 或明确失败指纹均可推进归因）。若再次 compile_worker
  同指纹崩溃 → 编译面假设削弱，转 reference 侧取证，不再盲投。
- ZIP `e2-afbe602`，9355 bytes，SHA256
  `884d35d654b1e81e33bf7db42baa66bbf1489de28251b7c68de2db86ac32668f`；
  成员 2：`hc_head.py`（generic，e1 字节不变）、`hc_head_kunlunxin.py`
  SHA256 `7eddd591ebf252078ed3728f03ffc965d8d5ffe080f082ac40f276aaa1ac7bd9`。
- 证据 `validation/verification.json` SHA256
  `2e7c03773e2d7f1a4a5a3e3c63b40e2e41c206fffd8074adb207c77cace6f518`、
  `validation/verification.log` SHA256
  `6755facfbbb640f581e7270363bf517f3451580fb44fc6e2fba56a533eb02e1c`。
