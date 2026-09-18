# Task 76 `add3` 实验记录

```current
task: 76
operator: add3
batch: 6
validity: valid
platform: completed(17393,e2,8/8,1.0564x新TB;燧原8192档+21%过门)
candidate_stage: e2
team_best_stage: e2
sealed: no
next: 燧原8192档兑现(0.755→0.914,门0.9刚过;纯streaming与T63 gather的8192回落成族边界对照);剩余=昆仑0.64→0.8/沐曦1.11→1.3无新假设;题边际递减,基本收官
updated: 2026-09-18
```

## 契约与实现（S0）

- 题面：[Task 76](../tasks/batch-6/76-add3.md)。`out = bf16(bf16(a+b)+c)`，
  **双重舍入是契约**（与未融合对逐位一致，禁 fp32 单舍入）；bf16 连续、
  numel%16==0；标准 bf16 容差。
- 实现：flat grid-stride kernel，`(a.f32+b.f32).bf16` 显式中间舍入再
  `+c` 出 bf16（RTNE，与 torch eager 每算子语义对齐）；BLOCK 1024、
  grid≤4096；对照 SGLang 197832b add3（PDL 为 NVIDIA 专属未引入）。
- 测试：双舍入最小反例（a=1,b=2^-8,c=-1 → 双舍入 0 / 单舍入 2^-8）、
  尾块边界、±inf 相消与 NaN 位级对照。

## 不可变身份

- source commit：`370923bfb83a8b7e681051acb2a3b1585dde12fc`（五题同批）。
- verification commit：`686092eb443f6a7b9514f8890fa9fdaa2d339a02`（含测试侧修复轮）。
- source SHA-256：`5f89fb2445a68fad815a89f298a29ce207d4e3e4c11369baeeaf4f23b820d296`。
- test SHA-256：`c4f0a92ceaf002c88bd47564eb825b09d61cd483219e3dec03c0d69d196149a6`。
- ZIP：`artifacts/competition/add3/s0-370923b/add3.zip`，SHA-256
  `6a50c36f2787094b347c45f0ed466f14309b14162971b4405cf56e924312f9c7`（单成员 `add3.py`，generic-only）。
- release 回执：`artifacts/competition/b6-s0-release-20260917/add3/verification.json`
  （3 测试 0 失败，9 次非 warmup kernel launch，
  NVIDIA RTX 5070 Ti / torch 2.13.0+cu130 / triton 3.7.1），
  SHA-256 `e7a372866f53bc1f4a7ae4612c7467f2ead8df6a6aa4790789c3136366f4eb93`；日志 SHA-256 `ed0eb9ef80b57877051e0ccf409d8c29c67867bc04a50e52b7fac2e7585720da`。

## 靶子与下一步

- 榜首 GuanghuLab 1.1611x（9/9 队有效，无彩票指纹）；主轴华为（全员<0.5）与昆仑 vendor。
- 验证三轮教训已固化在测试侧：numel%16 契约、token 边界与 kv_lens 解耦、
  归约噪声容差、行内连续 stride 构造。
- 八芯目标 `target-runtime-unverified`（NVIDIA 代理证据），裁决权在平台。

## 2026-09-18 S0 平台首回执（submission 17220，observed_at 01:0x +08）

- 状态：8/8 valid；均值 0.99170833。
- 逐芯：天数 1.4622 / 沐曦 1.1103 / 燧原 0.4382 / 海光 1.3677 / 昆仑 0.6405 / 华为 0.2956 / A 1.3511 / B 1.2681。
- 首回执与榜首 GuanghuLab 1.1611 对照：天数 1.46 vs 1.66、A 1.35 vs 1.38 接近；差距集中在华为（0.30 vs 0.42）、燧原（0.44 vs 1.20）、昆仑（0.64 vs 0.73）。

## 2026-09-18 E1 平台终态：双 vendor 兑现，TB 1.042875

- E1（17294，`8b65a43e`）：燧原 streaming vendor 0.4382→**0.7553（+72%）**、
  华为 persistent vendor 0.2956→**0.3631（+23%）**、A 1.351→1.392；均值
  **1.042875 新 TB**（vs s0 0.9917，+5.1%）。release 三路径 9+9+9 launch；
  ZIP `e1-8b65a43` 3 成员。
- 榜首 Sweetdeath 1.194（17 队过线）；我方差距集中在燧原/华为/昆仑。

## 2026-09-18 E2 平台终态：燧原 8192 档过门，TB 1.0564

- 结构（`12de5169`，单变量）：enflame vendor BLOCK 4096→8192（阶梯越
  过 T63 峰值档的一档——add3 是纯 streaming，与 T63 的 gather 不同族）。
- submission **17393** completed/valid，8/8，均值 **1.0564 新 TB**
  （vs 1.0429，+1.3%）。**燧原 0.7553→0.9139（+21%，预注册门 ≥0.9
  刚过）**；其余芯窗口持平。
- 族边界再证：BLOCK 阶梯的峰值档按访存形态分化——gather（T63）峰在
  4096、纯 streaming（add3）8192 仍上行。
