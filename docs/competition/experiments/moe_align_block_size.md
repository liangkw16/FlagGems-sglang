# Task 84 `moe_align_block_size` 实验记录

```current
task: 84
operator: moe_align_block_size
batch: 6
validity: candidate(7/8,华为reference崩)
platform: completed(17721,e2r,7/8;七芯健康93.7-95.3)
candidate_stage: e2r
team_best_stage: -
team_best_speedup: -
sealed: no
next: 华为两发同指纹reference侧torch_npu RuntimeError(score 0)=崩溃族,封存等健康窗;昆仑scalar-serial vendor 2.7-2.8兑现在库;燧原cumsum vendor 4.6-4.8;有效即~95x
updated: 2026-09-19
```

## 过程摘要（2026-09-19 凌晨，题面 09-18 晚随批 6 扩容上线）

- 开发：s0 generic + 全 unittest 矩阵（commit `540b574a` 家族），代理
  release 多轮修复后 8/8 验证；按回执逐芯定位失败根因，vendor 修复弹
  按 codex-review 门后发射（review 抓出 moe_align_single_token 填充轮
  越界读 P1 并修复）。
- 回执与产物：见 `artifacts/competition/b6ext-*-20260919/`（verification
  + log 逐弹归档），ZIP 在各 `artifacts/competition/moe_align_block_size/` 下按 stage。
- 今日新增跨芯硬事实：XPU 拒 f32→i16 位转换（经 i32 高半字绕过）；
  XPU 拒 tt.scan/tl.cumsum 与 atomic_rmw；XPU 上张量索引 gather 与
  runtime 宽度寻址产生垃圾（hash_topk 三形态同指纹，标量串行为唯一
  可用形态）；GCU make_gcuir 拒张量索引 gather（torch 预 gather 先例
  再证）；XPU bf16 downcast 刀刃值与 eager 差 1 ulp（rtne 显式钉仍差）。

## 2026-09-21 E5 候选就绪并发射（串行 scan 尾部并行化）

- 结构（`ac2cb311`）：_scan 只留 num_routed 偏移循环（每专家 5 标量
  操作）；新增 _fill 并行内核（每专家一程序写 expert_ids 段 + 全网格
  并行哨兵毯）。旧单程序串行填充整个 buf 是全芯 8-17×（燧原 100×）
  落后的主嫌疑。预注册门：天数 ≥400 / 海光 ≥400；均值 >95.28×2 即
  结构兑现。
- 五元组：commit `ac2cb3115e85e74baa9163257018e163d5031063`；ZIP SHA `ebe9bd4c4cf876b2660d4908f5332d71d24c42c907bd6b1b009803d53173d800`；test
  `f19789f3b167bea8cc936c5f86a285e2f8db5bce939e9a971c6ee182da37744c`；回执 `top1day-20260921k/` SHA `8ccb31fde66b089643e0f5bcc4e91a1f36867d4e8d41ad3b946f18883805f099`。

## 2026-09-21 08:06 E5 平台终态（7/8 时）：串行填充并行化兑现 +34%~+125%

- 逐芯：天数 185→**298.1** / 沐曦 63→**141.7** / 燧原 4.7（vendor 未动，
  原子禁令串行版）/ 海光 162→**217.1** / 昆仑 2.8 / A 161→**305.3** /
  B 170→**182.5**；华为评估中（历史有 reference 侧 torch_npu 崩溃族）。
- **E6**：wrapper 单次零填充合并 counts/cursor/nblk（减少设备端内核
  数）。燧原/昆仑无原子并行重设计（+60 均值潜力）入账为明日首轴。
- 五元组：commit `1dbebc9252817f167a0800c747e21ad89732515d`；ZIP SHA `b4b34419f4085b2c653542dea6241a754792cda52e2b14bf99337408fcf9b98f`；test
  `f19789f3b167bea8cc936c5f86a285e2f8db5bce939e9a971c6ee182da37744c`；回执 `top1day-20260921l/` SHA `23d5c23e7257abc496c321656cad60fcb6b0bbb0b18bfc756277eed703696d7d`。

## 2026-09-21 08:23 E6 平台终态：150.52（+6.5），华为 0.0 崩溃族第二次

- 逐芯：天数 295.7 / 沐曦 **165.9**（瘦身 +17%）/ 燧原 4.7 / 海光 223.2 /
  昆仑 2.8 / A 313.6 / B 198.2 / **华为 0.0**（e5/e6 连续两次 reference
  侧崩溃族，与 e2r 同族——七芯健康）。若华为健康 ~250：均值 ~182。
  **E7 = e6 字节载体弹重掷华为**（崩溃族协议 ≤2 次）。明日首轴：
  燧原/昆仑无原子并行重设计（+60 均值潜力）。
