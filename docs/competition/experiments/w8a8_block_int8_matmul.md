# Task 58 `w8a8_block_int8_matmul` 实验记录

```current
task: 58
operator: w8a8_block_int8_matmul
batch: 4
validity: valid
platform: 8/8(e1,10412,122.66158333x,排名3)
team_best_stage: e1
team_best_speedup: 122.66158333
sealed: no
next: 真正int8测试已补齐;cpasync候选待沐曦运行时识别与执行,当前门禁失败无ZIP
updated: 2026-09-08
```

## S0 fp32-ieee dot + 组内 scale（2026-09-06，远端 GPU 全过）

- 形态：单 Triton kernel；int8 load → `.to(tl.float32)` → `tl.dot(...,
  input_precision="ieee")`；每个 K 迭代后按 `k_start // block_k` 取
  As 行向量 [BLOCK_M] 与 Bs 列向量 [BLOCK_N]（`offs_n // group_n` gather），
  `acc += dot * a_s[:,None] * b_s[None,:]` — 与 reference 逐块舍入顺序一致。
- **遵守题面注意事项**"int8 须先 cast fp32、不可 int8 直接 GEMM"：不做
  int8 tensor-core dot，规避判罚风险（int8 dot 虽数学上更精确）。
- BLOCK_K = largest_pow2 ≤ min(group_k, 64)（保证 BLOCK_K | group_k，
  k 迭代不跨 scale 组）；BLOCK_N = largest_pow2 ≤ min(group_n, 64)；
  num_stages=2（3 stages + BLOCK_N=128 曾爆 shared memory 245760 > 101376）。
- 远端 5070 Ti：unittest 4/4 OK（平台契约 fp32 scales；atol 1e-2 本地线，
  平台 0.5）；bench 4 平台 shape：3.3x/6.5x/6.0x/22.4x，maxdiff ≤ 0.031。
- 教训：本地测试的 bf16/fp16 scale 子测是自加严（reference 的
  bf16 `s=a_s*b_s` 乘积自身差 |C|·2^-9≈0.04）— 已对齐平台契约改为
  fp32 scales。

## S0 平台结果（2026-09-06，submission ~08:40）

- **7/8 PASS**：天数 74.6x / 沐曦 147.5x / 燧原 4.60x / 海光 367.1x /
  华为 177.4x / A 61.7x / B 7.96x；**昆仑 PassManager::run failed（编译崩溃）**。
- e1：`_kunlunxin/ops/w8a8_block_int8_matmul.py` vendor — 唯一逐 lane
  向量整除 `offs_n // group_n` 改为标量 `(pid_n*BLOCK_N)//group_n`
  （BLOCK_N | group_n 时组索引跨 tile 恒定）；generic 字节不动。
  远端 variants 矩阵 5/5 OK（含 vendor 在 NVIDIA 代理编译+数值）。

## E1 → **8/8 VALID**（2026-09-06，submission seq27，第 8 个 8/8！）

- 逐芯：天数 74.44 / 沐曦 145.17 / 燧原 4.48 / 海光 358.86 /
  昆仑 142.03 / 华为 185.79 / A 62.84 / B 7.68 → **avg 118.16x**
- 昆仑 vendor（标量组索引替向量整除）一发修复编译崩溃，且 142x 高性能。
- 榜首 EvokeAgent 209.30x，差 43%；燧原/B 是短板轴。

## 2026-09-07 只读盘点校正

实时 team best 仍 E1/sub10412，真实八芯平均122.66158333，校正旧118.15512。榜首562.41590833；平台当前排名3，本轮未提交。
- 查询证据 `/Users/bytedance/ccc/flagos/artifacts/competition/batch4-top1-20260907/tasks-now.json` SHA256 `fc73368c3d98b228b0c7815d6ec1e9042a58af8d953337fff58990daec1474fc`。
- 查询证据 `/Users/bytedance/ccc/flagos/artifacts/competition/batch4-top1-20260907/58-submissions-now.json` SHA256 `fa822e61f388826ef2e99c7712299f89b369b40a7a2b68f3758f94944eb69d13`。

## 2026-09-08 推荐方案实现与提交前验证（未提交平台）

新增沐曦cpasync候选，保持int8先转FP32和scale顺序；修复原测试误生成int64，补负数及K/N尾部。NVIDIA拒绝pipeline参数（13条error），未打包。

- source `26a95766b179d263916e9483dfc8d2343c40406a`；verification `26a95766b179d263916e9483dfc8d2343c40406a`。6 个测试方法、23 次实际 kernel 调用；选定 NVIDIA/代理范围门禁失败。
- 回执 `artifacts/competition/batch4-implementation-20260907/t58-release1/verification.json`，SHA256 `27ad5180edb00e2702b1e4bdda7122b2a9229fa6ac02d73f260a872b70a19109`；日志 SHA256 `0d55f76f5f476e30857a93a5cd779ad0d904163619acde76a76dbde19fae26ea`。
- 环境、逐源码执行范围、原始配对数据和未完成条件见[本轮报告](../implementation-batch4-20260908.md)及[证据清单](../data/batch4-implementation-20260908.json)。本轮不更新历史有效分，未做平台 preflight、上传或正式提交。
