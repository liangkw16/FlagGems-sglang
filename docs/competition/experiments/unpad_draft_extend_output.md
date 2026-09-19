# Task 92 `unpad_draft_extend_output` 实验记录

```current
task: 92
operator: unpad_draft_extend_output
batch: 6
validity: valid
platform: completed(18340,e4,8/8,115.661x新TB;BLOCK回1024+metax w8)
candidate_stage: e4
team_best_stage: e4
team_best_speedup: 115.661
sealed: no
next: 首发即108x;轴=燧原3.5/昆仑8.9 vendor(gather/streaming配方);榜首295.62窗口待判
updated: 2026-09-20
```

## 2026-09-20 S0 首发记录 + 根因复盘

- 结构（`b7ae92fc`）：torch searchsorted 预计算行映射（T49 先例）+
  纯 gather kernel（2D grid 行×块）。8/8 首个有效 108.207x。
- **两小时"miscompile 追查"实为自身索引 bug**：src_row 已是展平行号
  （b*tpb+t），源行 stride 应为 H*D 而非 stride(0)=tpb*H*D（双重展平）。
  期间构造了 constexpr/runtime/if/mask/i32/i64 六维排查矩阵、多份"语义
  等价"探针——均因探针无意改用了正确 stride 而通过，形成"后端随机
  miscompile"假象。教训入账：**怀疑编译器前先字符化每个索引的物理
  含义**（与 T48 维度角色教训同构）。

## 2026-09-20 E1 平台终态：燧原 streaming vendor +143%，TB 109.45

- submission **18310** completed/valid，8/8，均值 **109.45 新 TB**（+1.1%）。
  **燧原 3.5→8.5（+143%，24-SIP+stages3 配方在 staged-gather 形态兑现）**；
  海光 170→184；华为 68.9→58.1（窗口回落）。

## 2026-09-20 E2 平台终态：华为 persistent +83%，TB 115.112

- submission **18321** completed/valid，8/8，均值 **115.112 新 TB**（vs
  109.45，+5.2%）。**华为 58.1→106.1（+83%，persistent 配方在 staged-gather
  形态兑现——与 T79 反例互补：gather 族的 persistent 效果按题分化）**；
  天数 191.7/A 198.7 窗口新高；燧原 8.9 持平 e1。

## 2026-09-20 E3 平台终态：BLOCK 4096 中性偏负，TB 保 e2 115.112

- submission **18329** completed/valid，8/8，均值 113.795 < TB 115.112（保 e2）。
  沐曦 99.9→113.0（+13%）但天数 191.7→184.3 / 海光 172.8→166.4 / B 133.7→
  129.1 同步回落——BLOCK 阶梯按芯分化再添一例。榜首 295.6 的逐芯形态
  （沐曦 245/海光 510）未破译，差距为结构级。

## 2026-09-20 E4 平台终态：BLOCK 回 1024 + metax w8，TB 115.661

- submission **18340** completed/valid，8/8，均值 **115.661 新 TB**（+0.5%）。
  海光 172.8→201.2（+17%）、B 133.7→142.5（+7%）；沐曦 99.9→82.6（warps8
  在此题为负——又一次 op×chip 分化）。榜首 295.6 结构差距仍在。
