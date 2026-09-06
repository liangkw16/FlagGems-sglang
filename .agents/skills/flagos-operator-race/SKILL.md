---
name: flagos-operator-race
description: "FlagOS 第二季算子竞赛的项目内闭环工作流：缓存赛题资料，锁定算子契约，调研芯片约束，开发 Triton/TLE generic 或 vendor 实现，在远端 GPU 做代理验证，生成可追溯 ZIP，并在实时门禁通过后通过安全脚本自动提交和记录多芯结果。仅适用于明确关联第二季竞赛 Task、提交包、评测或榜单的请求；普通 FlagOS 仓库维护和普通 Triton 开发不触发。"
metadata:
  short-description: FlagOS 算子赛调研、开发、验证与提交闭环
---

# FlagOS 算子竞赛工作流

把一次提交当作可复现实验，而不是临时网页操作。最短闭环是：锁定契约 →
generic 基线 → 代理验证与执行回执 → 不可变 ZIP → 实时平台门禁 → 单次自动提交 →
逐芯结果 → 最小 vendor 修复 → 账本与 Git 证据。

用户当前会话指令（包括用户提供的 AGENTS.md）优先于本 skill 的默认发布策略；
若已明确要求自动 commit/push，按该授权执行并保留下述待推提交隔离检查。
未被当前用户指令覆盖时，默认发布策略是“只打榜”：源码、测试和账本保留本地 Git commit 作为不可变证据，
但不 push 到任何 Git remote（包括 GitHub）。平台打榜与 GitHub 发布互相独立；任务包含
平台提交或竞赛闭环且门禁通过时照常自动提交，提交前后都不因此 push。只有用户在当前
请求中明确要求同步或发布 GitHub 时才允许 push。

先服从请求边界：调研、审计、解释、状态报告、静态验证或现有产物验签只做只读
本地检查，不刷新快照、不连接远端、不改文件、不打包、不 commit/push、不操作
浏览器；若用户明确要求实时平台状态，优先使用平台脚本只读 GET。开发或修改请求可做
范围内的本地工作；只有明确要求 GPU/runtime 代理验证，或验证本次开发改动确有
需要时才连接远端。只有实际产生的代码或账本改动才 commit，并默认仅保留在本地。平台
提交仅限当前任务明确包含提交、完整闭环或继续既有竞赛闭环；门禁通过后自动执行，
不再逐次询问。用户明确暂不提交平台或请求只读时不得运行提交 preflight。

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

## 正式评分与开发验证

按 [评分与排名](../../../docs/competition/README.md#3-评分与排名) 执行：平台正确性
合格且全部芯片 `S_i = T_bi / T_oi >= 0.1` 才是有效排名提交；排名使用全部芯片的
算术平均 `sum(S_i)/n`，完全同分时较早提交优先。先修正确性和门槛缺口，再按预计
平均加速比增量优化，不要求每芯都超过 1。评测未完成不提前判为有效或无效。
“攻克”依据首次过门槛（完全同时则高分优先），“攻占”依据有效排名第一；荣誉资格
须经专家评审，且不等于最终专项奖项或奖金。反作弊逐题核对。

KernelGen 与 GPU 都是开发验证通道，正式正确性、有效性和排名由平台判定。
辅助 MCP 的测试计数缺失只限制该份证据的覆盖结论，不是赛制无效条件；已有合格
release 回执、不可变 ZIP 及其他门禁时，不因该辅助字段缺失单独阻断已授权的提交。
本地执行回执的非零用例要求仍保留，不能用服务端 passed=true 代替实际回归。

门槛分三层，不能混用：

| 层级 | 判定与动作 |
| --- | --- |
| 提交前硬门槛 | 适用路径的必要正确性回归通过、实际 kernel 执行、源码/测试/ZIP 身份一致、实时 preflight 与一次性提交状态正常；已知相关正确性失败不能靠排除路径或 skip 隐藏 |
| 优化晋级 | 先补每芯 0.1 缺口，再比较预计全部芯片的算术平均增量；单芯回退只触发排查与门槛风险检查，不自动关闭方向；未知芯片保留未知，不填造分数 |
| 平台正式资格 | 全部芯片正确且每芯达到门槛后才排名；专家评审决定实现有效性与荣誉资格 |

目标芯资源或 MCP 覆盖证据缺失时，明确标注 target-runtime-unverified，保留已取得
证据并在已授权闭环中交平台补齐；不能将其记为通过，也不要求提交前先证明平台八芯全部通过。

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
决定，不改 upstream、不 force-push。该检查仅在用户明确要求同步或发布 GitHub 时执行。

`artifacts/` 被 Git 忽略；账本必须记录源码 commit、各文件 SHA-256、ZIP
SHA-256、成员列表和平台结果，才能重新定位实际上传字节。

分别记录三个身份：`source commit` 是 ZIP 逐字节取源的提交；`verification
commit` 是最新测试和验证证据的提交；`ledger commit` 是写回产物与结果的提交。
三者可以不同，但账本必须写清关系，产物目录使用 `source commit` 短哈希。

本仓常有多会话并行推进不同算子。写账本或实验 README 前，先用
`git log -1 -- <路径>` 核对该文件最新 commit 并重读最新字节，不基于本次
会话早先读到的旧内容直接编辑；算子账本更新时同步刷新实验 README 的
候选行，避免两个真相源分叉。

账本顶部维护机器可读的 ` ```current ` 块（task/operator/validity/
platform/team_best/sealed/next/updated），是该任务唯一的人工状态真相；
每次账本终态更新都改 CURRENT 块，并运行
`python tools/gen_experiment_index.py` 刷新生成的
`docs/competition/experiments/INDEX.md`。手写汇总表不再承载当前状态。

## SkillHub 辅助 skill

`.agents/skills/` 下装有 FlagOS SkillHub 的配套 skill，按以下边界融入闭环；
它们服从本 skill 的请求边界、只读规则和提交纪律，不改变任何门禁：

- `kernelgen-flagos`：可用于算子生成、优化、特化和多 GPU/多芯片验证，与 `gpu` 主机都是
  可选验证通道。`generate_kernel` / `autotune_kernel` 还可在指定芯片上做 benchmark、
  测量加速比；逐芯记录参考耗时、算子耗时或返回的 speedup，用于调优筛选。
  `optimize_kernel` 只改写代码，不验证、不测速。需要生成新算法或缺少可复用实现时，先用 MCP 获取候选；
  需要第二实现或瓶颈建议时，再带基线、逐 shape 数据和失败上下文调用 optimize。
  成熟上游复用、明确根因修复、测试和调度参数调整可直接实现，不要求为满足调用
  次数而重写代码。选择 MCP 生成路线时遵守其配置与调用协议；服务不可用要记录，
  不能把未调用说成已调用，也不阻塞已有独立方案的验证。
  specialize/optimize 的代码改写不等于上设备；autotune 的通过、失败和速度都必须
  按集成文档核对实际执行源与测试口径。未绑定每次执行源码的错误仅是服务端线索。
  所有产出仍须过契约、代理验证、执行回执和不可变 ZIP 门禁。本仓库布局是
  `src/flaggems_sglang/` 而非上游 `src/flag_gems/`，其 FlagGems 专用注册与
  测试布局不适用；验证接口、证据分级、失败定位与 Ascend 特化参数见集成文档。
- `gpu-container-setup-flagos`：远端 GPU 容器自动选型（NVIDIA/昇腾/Metax/
  天数/海光/AMD，按 vendor hub → BAAI Harbor → 搜索的优先级选镜像）。仓库内脚本
  路径是 `.agents/skills/gpu-container-setup-flagos/scripts/`（skill 文档中的
  `.claude/skills/...` 路径不适用）。各 vendor 检测命令、镜像仓库、挂载规格
  与已知坑见集成文档。不覆盖昆仑芯，昆仑芯沿用现有远端验证流程。
- `tle-developer-flagos`：走 Triton-TLE 路线时的开发工作流（源码真相在
  `references/tle-sources.md`，含 marker block 等护栏）。仅当赛题实现选 TLE
  或需要改 TLE 层时使用；其调参优先级、TTGIR/PTX 证据与单变量循环纪律
  对普通 Triton 调优同样适用，见集成文档。
- `flaggems-pr-review-flagos` / `flaggems-pr-submit-flagos`：面向上游 FlagGems
  仓库的 PR 审查与提交（含算子注册表与门禁脚本）。比赛平台提交不走它们；
  仅当用户要求向 FlagGems 上游提 PR 时使用，且仍遵守本 skill 的 push 授权规则。
- `perf-test-flagos` / `model-migrate-flagos` / `model-verify-flagos` /
  `flagrelease-entrance-flagos` / `install-stack-flagos` /
  `vllm-plugin-fl-setup-flagos`：模型部署与推理服务流水线，与算子赛无关，
  默认不使用（model-verify 的多芯报错速查已摘入集成文档）。

这些是第三方 skill，脚本运行拥有完整 agent 权限；首次调用某个 skill 前先
快速审阅其脚本再执行。具体用法、协议和跨芯技术事实见
[SkillHub 工具集成](references/skillhub-tools.md)。

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

每个 tile 轴至少检查 B-1/B/B+1（题面允许时），段内多 tile、非整除维度和多维
stride 分别覆盖。移植上游 kernel 时同时带走其调用方前提（例如上游先切短 segment），
不能只搬 BLOCK 参数。修过的失败 case 必须进入持久化回归矩阵。
测试入口统一使用 unittest 模块加载；`unittest.main()` 必须位于所有测试类之后。
正常通过、skip、expectedFailure 分开记账。按设备先选择适用源码；适用矩阵内的
skip/expectedFailure 仍不能让 release 晋级，其他设备的未执行路径单独记录。
新增或修改测试时，用测试模块的 `RELEASE_REQUIRED_TESTS` 列出必须保留的
`TestClass.test_method`（含数值、dtype、边界和已知回归，L2Norm 为样板）；runner
检查这些方法确实进入完整 suite 且通过。清单不能代替对 reference、容差和分支的审查。

按项目约定把远端 GPU 单测放后台，前台继续静态检查和资料整理。验证顺序：

1. `py_compile`；
2. Black/isort/flake8 或仓库 pre-commit；
3. 最小 unittest；
4. 主要 shape 的正确性；
5. 性能候选做 wrapper-inclusive benchmark、阶段计时与编译产物检查；
6. 用 `scripts/verify_release.py` 生成绑定实际执行的回执。

NVIDIA 代理看不见目标芯自身的 lowering/编译器风险。vendor 候选或改动
涉及某芯专属路径时，主动使用 KernelGen 多芯验证或已授权目标机，按
[SkillHub 工具集成](references/skillhub-tools.md)
的资源登记与证据分级为受影响芯选最强实际可用通道：已有授权厂商主机时执行
同源完整矩阵；也可用 KernelGen 在其可用设备上执行验证。调用前检查当前 schema，
携带候选源码和契约测试，核验实际执行源码、测试数、reference、环境和结果。
能绑定的数值结果记为该设备已覆盖范围的验证；缺失绑定时只记服务端线索；
服务报告通过但 tests=0 时记“服务端报告通过，覆盖未核实”。
每个目标芯单独记录 job、源码身份、用例覆盖和结果；一个设备通过不能折算成多芯通过。
镜像已知不代表主机可用；KernelGen 和主机均无有效执行结果时标注 target-runtime-unverified。
NVIDIA 数学代理不能替其他芯背书。新增 vendor 文件必须
接进该算子 unittest 矩阵（`tests/_op_variants.py`，T35 为样板）。

screening 只跑足以淘汰候选的最小回归与性能探针；无筛选需要时直接 commit 后
跑一次完整 release，不固定要求两遍全量矩阵。未提交候选不能作为 ZIP 的最终验证证据。候选通过初筛后，
先将本次 source 和 test 按明确路径 commit，再用该 commit 的逐字节内容重跑
发布门禁。失败后修正则产生新 commit，不用已被验证记录引用的旧 commit
冒充新候选。

晋级时把 screening 的 source/test SHA-256 与提交后的 Git blob 逐项比较；任一
变化都视为新候选，旧 screening 不再为它背书。release 临时目录只从明确 commit
的 Git 对象生成，不能从当前工作树复制源码、测试或其仓库内导入依赖。

v2 执行回执必须包含全部候选 source/test/依赖/runner 哈希、实际设备、适用源码、
未执行与目标芯未验证源码、完整 suite/通过用例、非空张量 shape/dtype、入口调用和
实际 kernel launch 数、退出码和日志哈希。默认验证 generic 与本设备 vendor；显式
加入的跨厂商代理路径也必须完整通过，但仍不算目标芯通过。零测试、适用路径
skip/xfail、只有空输入、无 kernel launch、漏跑必要用例或字节变化都不得晋级；
preflight 和 submit 都复核回执。哈希回执用于追溯执行，不是远程硬件签名证明。

连接、传输、后台日志和证据保留按
[远端 GPU 代理验证](references/remote-validation.md) 执行。

修改本 skill 或其门禁脚本后，先运行
`python3 -m unittest discover -s .agents/skills/flagos-operator-race/tests -v`。
按本次影响补最小真实验证：修改 GPU runner/覆盖时跑 GPU commit 字节回归与验签；
修改 MCP 调用协议时才重跑 KernelGen 生成/验证与受影响设备路由，不为未改动通道重复生成。
涉及发布门禁时，用真实 ZIP/回执配合 FakeClient
测试 preflight→submit、篡改拒绝和一次性 nonce；这不访问真实平台，不消耗提交额度。
逐阶段记录成功或失败，服务端零测试不能用“接口调用成功”掩盖。

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

先把构建身份、验证环境、结果和已知风险写入实验账本，再 commit；默认不 push。
若用户只要求开发或明确说暂不提交平台，到此停止：状态记为“候选就绪，
未提交”，不运行平台预检、不自动提交、不消耗额度。

## 平台提交与逐芯迭代

只有用户明确要求实时平台预检、提交、查看评测或基于逐芯结果迭代时，
才完整读取[平台提交与逐芯结果](references/platform-workflow.md)。失败详情
优先走文档内「失败情报通道」（submissions API 的 raw_result，勿先上
浏览器/工单）；芯片硬事实坑表与 GQA 共享 dot 等结构资产见同文档
「平台评测踩坑硬事实」一节。“先不要提交平台”
的开发请求在 ZIP、账本和本地 commit 完成后停止，不读取该引用、不运行平台脚本。

## 平台闭环完成标准

局部调研、开发、验证或打包请求以用户指定交付物为完成条件；以下标准只适用于
用户要求完成平台提交和八芯评测闭环时。

只有同时满足以下条件才称为完成：

- 平台显示所有支持芯片通过且每芯达到门槛；
- 平均加速比、排名、逐芯结果和失败/回退历史已写入账本；
- 代码、测试和账本已本地 commit，工作树无本任务遗留修改；GitHub push 不是完成条件，
  除非用户在当前请求中明确要求；
- 提交与逐芯结果以 `status`/`watch` 的 JSON 输出为准入账；浏览器结果页仅在
  用户明确要求时另存为可交付页面。

最终回复优先给出：通过芯片数/支持芯片数、平均加速比、排名、剩余额度、ZIP/账本路径、
commit、GitHub 未推送状态，以及下一条单变量优化假设。尚在排队就明确写“评测中”，
不把入队当通过。
