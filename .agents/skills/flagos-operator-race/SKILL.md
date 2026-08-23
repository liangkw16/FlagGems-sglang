---
name: flagos-operator-race
description: "FlagOS 第二季算子竞赛的项目内闭环工作流：缓存赛题资料，锁定算子契约，调研芯片约束，开发 Triton/TLE generic 或 vendor 实现，在远端 GPU 做代理验证，生成可追溯 ZIP，并在逐次确认后网页提交和记录多芯结果。适用于本仓库中的 FlagOS/算子赛题、批次、跨芯优化、提交包、评测或榜单任务；普通 Triton 开发不自动触发。"
metadata:
  short-description: FlagOS 算子赛调研、开发、验证与提交闭环
---

# FlagOS 算子竞赛工作流

把一次提交当作可复现实验，而不是临时网页操作。最短闭环是：锁定契约 →
generic 基线 → 代理验证 → 不可变 ZIP → 人工确认 → 逐芯结果 → 最小 vendor
修复 → 账本与 Git 证据。

## 先读本地资料

从仓库根目录工作，并按需读取，不重复抓取已经落盘的资料：

- `COMPETITION.md`：入口和常用检索命令；
- `docs/competition/README.md`：评分、额度、命名和提交规范；
- `docs/competition/task-index.md`：批次和动态榜单快照；
- `docs/competition/tasks/<batch>/<task>.md`：完整题面；
- `docs/competition/reference-repositories.md`：固定 Git 引用和上游来源；
- `docs/competition/strategy-batch2.md`：候选优先级和已知语义陷阱；
- `docs/competition/experiments/<operator>.md`：该算子的实验账本。

动态状态可能过期。需要最新公开题面或榜单时运行：

```bash
python tools/sync_flagos_season2_docs.py
```

比赛截止时间、额度、登录状态和提交结果以平台当前页面为准。来源冲突时采用
更严格的截止时间，并把冲突写入账本。

## 产物布局

每个算子只维护一份源码真相：

```text
src/flaggems_sglang/ops/<operator>.py
src/flaggems_sglang/runtime/backend/_<vendor>/ops/<operator>.py
tests/test_<operator>.py
docs/competition/experiments/<operator>.md
artifacts/competition/<operator>/<stage>-<code-commit>/<operator>.zip
```

`artifacts/` 被 Git 忽略；账本必须记录源码 commit、各文件 SHA-256、ZIP
SHA-256、成员列表和平台结果，才能重新定位实际上传字节。

## 阶段 A：从题面到 S0

### 1. 锁定契约

完整读取题面后写下：

- Task 编号、operator basename 和 batch；
- 精确函数签名、输入 shape/stride/dtype；
- 输出 shape、dtype、in-place/out-of-place 语义；
- reference 公式、容差、隐藏边界；
- 支持芯片、截止时间、最低加速比和反作弊约束；
- ZIP 必需文件名及允许的 vendor 后缀。

没有锁定这些字段前不写 kernel。不要把未公开 shape、芯片型号或匿名 A/B
映射当作事实。

题面未公开 shape/dtype/stride 范围时，把它们明确标为未知，并分开记录“题面
事实”和“代理验证假设”。若公开 signature 与 reference 已能定义可执行契约，采用
保守 generic 覆盖继续；只有未知项会改变接口、输出语义或合法实现时才停止并询问。

### 2. 选择最短可行算子

优先满足：有固定上游 reference、计算结构简单、状态少、跨芯私有 API 少、
能用一个保守 Triton kernel 覆盖。把预计首次正确时间、跨芯风险和榜单收益写入
决策记录；不因单芯理论峰值选择高风险题。

### 3. 固定一手来源

先检索当前仓库和已有 Git refs，再查官方源码或文档。引用 immutable commit，
区分：

- 固定源码能证明的事实；
- 需要平台验证的硬件/编译器假设；
- 只适用于本地 NVIDIA 代理的观察。

不要复制 NVIDIA-only autotune、私有 cache hint、PDL、libdevice 或超大
warp 配置到 generic 首版。

### 4. 实现 generic 基线

S0 只追求全部支持芯片正确且每芯达到题面最低门槛：

- 核心路径实际运行 Triton/Triton-TLE；
- 不用 `try/except`、设备判断或 PyTorch fallback；
- 先用一个保守 tile、默认合法 launch 参数和完整 tail mask；
- 计算 dtype、输出 dtype、stride、空输入和特殊值严格服从题面；
- 不提前维护八份 vendor 文件。

### 5. 测试优先并远端验证

以公开函数作为测试 seam。先留下一个会失败的最小回归，再实现代码。至少覆盖：

- 题面 dtype 与容差；
- 空输入、尾块边界、非连续输入和输入不变性；
- 公式分支、极值、NaN/Inf（题面相关时）；
- 平台报错对应的精确回归 case。

按项目约定把远端 GPU 单测放后台，前台继续静态检查和资料整理。验证顺序：

1. `py_compile`；
2. Black/isort/flake8 或仓库 pre-commit；
3. 最小 unittest；
4. 主要 shape 的正确性；
5. wrapper-inclusive benchmark 与编译产物检查。

远端 NVIDIA 只能筛选语法、数值和候选，不能证明其他芯片正确或性能。

### 6. 生成不可变 ZIP

从已提交的源码构建，不维护 `submissions/` 副本。目录名使用代码 commit 短哈希。
打包后逐项检查：

- `.zip` 小于 10 MB；
- 只有 UTF-8 `.py` 文件；
- generic basename 精确为 `<operator>.py`；
- vendor basename 精确为 `<operator>_<suffix>.py`；
- 无测试、缓存、目录前缀或 macOS 垃圾文件；
- `unzip -t`、`unzip -l`、每个成员 SHA-256 和 ZIP SHA-256 均已记录；
- ZIP 内源码与对应 commit 源文件逐字节一致。

先把构建身份、验证环境、结果和已知风险写入实验账本，再 commit、push。

## 阶段 B：平台提交

网页写操作必须获得 action-time 确认。确认内容至少包括：

- Task 编号和 operator；
- ZIP 的绝对路径与 SHA-256；
- 当前剩余额度，以及本次会消耗 1 次。

旧的“继续”“可以上传”不能授权后来生成的另一份 ZIP。每个新候选重新确认。

确认后使用 `chrome:control-chrome` skill：

1. 核对登录团队、Task、batch、截止时间、两分钟间隔和额度；
2. 打开“提交代码”，阅读当次页面规则；
3. 选择已确认的绝对路径；
4. 等待平台识别正确的 `.py` 数量；
5. 只有四项基础校验通过且提交按钮启用时才点击；
6. 捕获“提交成功”、额度扣减、流水时间和状态。

页面刚选文件时可能短暂显示 `0 个 .py`；等待校验完成，不要在异常状态提交。
若 Chrome 文件权限失败，遵循 Chrome skill 的 file-upload troubleshooting。
验证码或新的风险提示交给用户处理。

## 阶段 C：逐芯结果与最小迭代

评测记录可能需要主动切换“题目说明 → 提交代码”刷新。逐芯记录：正确性、
speedup、平均值、状态、失败详情、排名、首次有效提交时间和剩余额度。页面不展示
独立 submission ID 时，用 `Task + 文件名 + 时间` 联合定位，不编造 ID。

按根因分类：

| 现象 | 最小动作 |
| --- | --- |
| 多芯同类数值失败 | 修 generic 根因，不加多份 vendor |
| 单芯编译/正确性失败 | generic 保持不变，只加该 vendor |
| `grid.x` 超硬件上限 | 从固定 vendor policy 取 grid 上限，改 grid-stride |
| 输出像归一化中间值 | 检查最终缩放、scalar broadcast 和 lowering 顺序 |
| 正确但低于题面门槛 | 先恢复门槛，再做性能排名 |
| 通过但明显落后榜首 | 每次只改 BLOCK、grid、warps、数学 lowering 或布局之一 |

vendor 文件必须自包含、保持同一函数签名并导出同一 `__all__`。已经通过的芯片
继续使用原 generic，避免无关回归。一次提交只改变一个可解释变量；若为恢复
正确性必须同时消除已知 grid 风险，在账本明确说明。

每轮重复：最小回归 → 远端代理 → 新 commit → 新 ZIP/hash → 新确认 → 平台。
至少预留两次额度给截止日前最终回归。

## 完成标准

只有同时满足以下条件才称为完成：

- 平台显示所有支持芯片通过且每芯达到门槛；
- 平均加速比、排名、逐芯结果和失败/回退历史已写入账本；
- 代码、测试和账本已 commit、push，工作树无本任务遗留修改；
- 浏览器结果页保留为可交付页面。

最终回复优先给出：通过芯片数/支持芯片数、平均加速比、排名、剩余额度、ZIP/账本路径、
commit，以及下一条单变量优化假设。尚在排队就明确写“评测中”，不把入队当通过。
