# Task 51 `fla_layernorm_gated` 实验记录

```current
task: 51
operator: fla_layernorm_gated
batch: 4
validity: valid
platform: 8/8(e1,5.390025x)
team_best_stage: e1
team_best_speedup: 5.390025
sealed: no
next: e3华为constexpr-D仅+3.7%证伪关闭;昆仑0.97已修;剩沐曦rsqrt;燧原病态在评
updated: 2026-09-06
```

## S0（2026-09-05，submission 9867）
- 7/8，燧原 1830s 超时（非 correctness），其余七芯全过

## E1 燧原 vendor → **8/8 VALID**（submission 9944）
- 燧原 vendor：一 program 一行（去 grid-stride 循环）+ tl.rsqrt/tl.sigmoid
- **燧原超时→2.3398x**；逐芯：天数 9.21 / 沐曦 4.57 / 燧原 2.34 / 海光 7.74 /
  昆仑 0.944 / 华为 2.52 / A 8.73 / B 7.07

## 2026-09-05 Codex 会诊作战方案（预注册）

**华为（首发，最高置信）**：`dim` → `tl.constexpr D`，`offs < dim`
运行时 int 比较（昇腾 Vector CMP 标量退化，官方指南确认）变编译期
静态 mask，`/D` 常量折叠；行结构与单次 x load 不变（勿先搬 T40 的
1024 子块——那会迫使 RMSNorm 重载两次 x）。预期 +15~40%；门 ≥+15%。

**沐曦**：仅 `1/sqrt` → `tl.rsqrt`（消除逐 lane 向量除法；IR 不变则
取消候选）；门 ≥+12%。
**昆仑**：仅 `D<1024` 时 BLOCK 提 1024（T21 唯一成功轴；T19 multi-row
无收益、BLOCK 2048 负收益，勿重复）；门 ≥+15%；raw_result 若暴露逐
case D 分布则先查命中再发。
后置：g load 后移（缩短归约期 live 向量）；BLOCK_D≥2048 沐曦 warps8。
LayerNorm 勿改 `E[x²]-E[x]²` 单遍（大均值小方差消减误差，fp32 1e-4 风险）。

> 注：并行会话 09-05 22:56 记录已发 T51 E2（8/8、5.3939x、未超 e1，轴内容未落账本）；
> 执行本方案任一候选前先与该会话核对 E2 消费过的轴，避免重复预注册。

## E3 华为 constexpr-D vendor（2026-09-06，submission 10385）

- `_ascend` vendor：generic 数学逐字节不变，唯一变量 = `dim` →
  `tl.constexpr D`（`offs < D` 编译期折叠 + `/D` 常量折叠）；
  source `080148b`，ZIP `e3-080148b` SHA-256 `2bddb9d4…271d`
  （4 成员），screening OK
- 七芯已过：**华为 2.52→2.6182（+3.7%，远低于 ≥+15% 门——轴单发
  证伪关闭**：int 比较不是本题华为瓶颈）；天数 9.2822/沐曦 4.3918/
  海光 8.133/昆仑 0.9682（kunlunxin vendor 生效）/A 8.1964/B 6.873
- 燧原（非目标芯）卡病态盒子 waiting_callback；无论燧原落点，
  估算均值 ≤5.38 < e1 TB 5.39——非 team best
- 判定：constexpr-D 轴关闭；T51 剩余轴：沐曦 rsqrt（预注册 #2）
