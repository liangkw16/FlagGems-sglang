---
name: flagos-operator-race
description: "FlagOS 第二季算子竞赛的项目内闭环工作流：缓存赛题资料，锁定算子契约，调研芯片约束，开发 Triton/TLE generic 或 vendor 实现，在远端 GPU 做代理验证，生成可追溯 ZIP，并在逐次确认后通过安全脚本提交和记录多芯结果。仅适用于明确关联第二季竞赛 Task、提交包、评测或榜单的请求；普通 FlagOS 仓库维护和普通 Triton 开发不触发。"
metadata:
  short-description: FlagOS 算子赛调研、开发、验证与提交闭环
---

# FlagOS 算子竞赛工作流

把一次提交当作可复现实验，而不是临时网页操作。最短闭环是：锁定契约 →
generic 基线 → 代理验证 → 不可变 ZIP → 人工确认 → 逐芯结果 → 最小 vendor
修复 → 账本与 Git 证据。

先服从请求边界：调研、审计、解释、状态报告、静态验证或现有产物验签只做只读
本地检查，不刷新快照、不连接远端、不改文件、不打包、不 commit/push、不操作
浏览器；若用户明确要求实时平台状态，优先使用平台脚本只读 GET。开发或修改请求可做
范围内的本地工作；只有明确要求 GPU/runtime 代理验证，或验证本次开发改动确有
需要时才连接远端。只有实际产生的代码或账本改动才按项目约定 commit/push，平台
提交始终另需当次确认。

## 先读本地资料

从仓库根目录工作，并按需读取，不重复抓取已经落盘的资料：

- `COMPETITION.md`：入口和常用检索命令；
- `docs/competition/README.md`：评分、额度、命名和提交规范；
- `docs/competition/task-index.md`：批次和动态榜单快照；
- `docs/competition/experiments/README.md`：当前候选、产物哈希和提交队列；
- `docs/competition/tasks/<batch>/<task>.md`：完整题面；
- `docs/competition/reference-repositories.md`：固定 Git 引用和上游来源；
- `docs/competition/strategy-batch2.md`：候选优先级和已知语义陷阱；
- `docs/competition/learning-path.md`：仅在题型学习、芯片调研或跨芯优化时读取
  对应章节和固定 backend 证据；
- `docs/competition/experiments/<operator>.md`：该算子的实验账本。

动态状态可能过期。需要最新公开题面或榜单时运行：

```bash
python tools/sync_flagos_season2_docs.py
```

同步脚本会改写本地快照；只读调研、审计或报告请求不要运行，除非用户明确要求
刷新或更新资料。

比赛截止时间、额度、登录状态和提交结果以平台当前 API/页面为准。来源冲突时采用
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

写操作前先检查 `git status --short` 和目标路径的 diff；现有改动归用户，
只 stage 本次明确的文件。若目标文件已有归属不明的改动，先停止并请用户
确认，不把它们夹带进算子、验证或账本 commit。若用户明确说有本地改动而当前
检查未发现，先核对仓库根目录、worktree、分支和目标路径；信息仍对不上就停止并
请用户定位，不把“未发现”当作继续写入的授权。提交前再检查
`git diff --cached --name-only`；若 index 中有无关的已暂存改动，不要替用户
unstage。先完整复核 `git diff HEAD -- <本次明确路径>`，再只 stage 这些路径，
并确认 `git diff --quiet -- <本次明确路径>` 成功，避免同路径的未暂存字节
被夹带。然后使用 `git commit --only -- <本次明确路径>` 隔离提交；该命令取
working-tree 字节，所以上述检查不能省略。提交后复核完整 commit diff，
并确认无关已暂存改动仍留在 index。push 前核对 upstream、push 目标和
`@{upstream}..HEAD` 的全部待推 commit；若会带上既有无关 commit，停止并请用户
决定，不改 upstream、不 force-push。

`artifacts/` 被 Git 忽略；账本必须记录源码 commit、各文件 SHA-256、ZIP
SHA-256、成员列表和平台结果，才能重新定位实际上传字节。

分别记录三个身份：`source commit` 是 ZIP 逐字节取源的提交；`verification
commit` 是最新测试和验证证据的提交；`ledger commit` 是写回产物与结果的提交。
三者可以不同，但账本必须写清关系，产物目录使用 `source commit` 短哈希。

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

以公开函数作为测试 seam。正确性修复先留下一个旧实现会失败的最小回归；纯性能
候选先声明 affected shape 和晋级阈值，存在明确未受影响路径时再加 control，并复用
完整正确性矩阵。至少覆盖：

- 题面 dtype 与容差；
- 题面允许且相关时的空输入、尾块边界、非连续输入和输入不变性；
- 公式分支、极值、NaN/Inf（题面相关时）；
- 平台报错对应的精确回归 case。

按项目约定把远端 GPU 单测放后台，前台继续静态检查和资料整理。验证顺序：

1. `py_compile`；
2. Black/isort/flake8 或仓库 pre-commit；
3. 最小 unittest；
4. 主要 shape 的正确性；
5. wrapper-inclusive benchmark 与编译产物检查。

未提交候选可用于快速筛选，但不能作为 ZIP 的最终验证证据。候选通过初筛后，
先将本次 source 和 test 按明确路径 commit，再用该 commit 的逐字节内容重跑
发布门禁。失败后修正则产生新 commit，不用已被验证记录引用的旧 commit
冒充新候选。

晋级时把 screening 的 source/test SHA-256 与提交后的 Git blob 逐项比较；任一
变化都视为新候选，旧 screening 不再为它背书。release 临时目录只从明确 commit
的 Git 对象生成，不能从当前工作树复制源码、测试或其仓库内导入依赖。

连接、传输、后台日志和证据保留按
[远端 GPU 代理验证](references/remote-validation.md) 执行。

远端 NVIDIA 只能筛选语法、数值和候选，不能证明其他芯片正确或性能。

### 6. 生成不可变 ZIP

从已提交的源码构建，不维护 `submissions/` 副本。目录名使用代码 commit 短哈希。
release 前先用 `--dry-run` 取得 source manifest；最终 ZIP 的 commit、成员集合、
成员 SHA-256 和 canonical ZIP SHA-256 必须与该 manifest 完全相同。
优先使用 Skill 自带的确定性打包器；它从指定 commit 读取 generic 和已有 vendor
源码、生成固定字节 ZIP、拒绝覆盖同路径的不同产物，并输出可直接写入账本的哈希：

```bash
python .agents/skills/flagos-operator-race/scripts/build_submission.py \
  <operator> --stage <stage> --commit <code-commit>
```

首个基线使用 `s0`，后续单变量候选使用 `e1`、`e2` 递增。

历史 ZIP 若不是该工具的规范字节，只能用 `--verify-existing` 按安全路径、唯一
basename 和提交源码内容做只读验签；结果会标记 `verified-existing-legacy` 并同时
输出实际成员路径、实际与规范 ZIP 哈希，不能据此重写旧产物。

打包后逐项检查：

- `.zip` 小于 10 MB；
- 只有 UTF-8 `.py` 文件；
- generic basename 精确为 `<operator>.py`；
- vendor basename 精确为 `<operator>_<suffix>.py`；
- 新建规范包无测试、缓存、目录前缀或 macOS 垃圾文件；历史包的安全子目录仅能
  通过只读 legacy 验签保留；
- `unzip -t`、`unzip -l`、每个成员 SHA-256 和 ZIP SHA-256 均已记录；
- ZIP 内源码与对应 commit 源文件逐字节一致。

先把构建身份、验证环境、结果和已知风险写入实验账本，再 commit、push。
若用户只要求开发或明确说暂不提交平台，到此停止：状态记为“候选就绪，
未提交”，不运行平台预检、不索要提交确认、不消耗额度。

## 平台提交与逐芯迭代

只有用户明确要求实时平台预检、提交、查看评测或基于逐芯结果迭代时，
才完整读取[平台提交与逐芯结果](references/platform-workflow.md)。“先不要提交平台”
的开发请求在 ZIP、账本、commit 和 push 完成后停止，不读取该引用、不运行平台脚本。

## 平台闭环完成标准

局部调研、开发、验证或打包请求以用户指定交付物为完成条件；以下标准只适用于
用户要求完成平台提交和八芯评测闭环时。

只有同时满足以下条件才称为完成：

- 平台显示所有支持芯片通过且每芯达到门槛；
- 平均加速比、排名、逐芯结果和失败/回退历史已写入账本；
- 代码、测试和账本已 commit、push，工作树无本任务遗留修改；
- 浏览器结果页保留为可交付页面。

最终回复优先给出：通过芯片数/支持芯片数、平均加速比、排名、剩余额度、ZIP/账本路径、
commit，以及下一条单变量优化假设。尚在排队就明确写“评测中”，不把入队当通过。
