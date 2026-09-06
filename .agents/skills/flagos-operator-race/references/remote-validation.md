# 远端 GPU 代理验证

只在用户明确要求 GPU/runtime 代理验证，或验证本次开发改动确有需要时连接远端；
静态验证、现有 ZIP 验签、只读调研、审计或报告请求不产生远端目录、传输或任务。
当前项目默认 SSH alias 是 `gpu`，Python 是
`/home/kevin/notebook/.venv/bin/python`；先用 `ssh -G gpu` 和远端版本命令核对，
不要把解析出的 IP 写进仓库。环境变化时以 SSH config 和实际解释器为准。

## 两种证据模式

- `screening`：允许传未提交候选做快速淘汰；记录 base commit、明确文件列表以及
  本地/远端 SHA-256。结果只能标为探索证据，不能为 ZIP、发布门禁或平台提交背书。
- `release`：候选 source 和 test 已提交；source 使用打包器 manifest 绑定，test
  使用 verification commit 绑定。临时目录中的每个仓库文件都从对应 commit 的
  Git 对象生成，不从当前工作树复制。只有该模式的结果可把候选标为“就绪”。

每份日志先写明模式，不把 screening 结果升级成 release 结果。
screening 采用最小淘汰集；没有筛选需要时直接提交候选后跑完整 release。不要固定
重复两遍完整矩阵。源码、测试、runner 或依赖变化后重新取证。

## 统一执行回执

本仓 CUDA/HIP unittest 统一从隔离目录按模块加载，不直接执行测试文件。先生成
明确文件集合；默认包含 generic、全部规范 vendor、主测试、tests 包入口、variant
helper 和 runner。其他仓库内导入依赖用 `--dependency <相对路径>` 显式添加，
运行时发现未绑定的已导入依赖就拒绝晋级。

```bash
python .agents/skills/flagos-operator-race/scripts/verify_release.py prepare l2norm \
  --source-commit '<源码commit>' --verification-commit '<测试和runner commit>' \
  --directory '/tmp/flagos-l2norm-release'
```

默认 `--device-vendor nvidia`，AMD 使用 `--device-vendor amd`；运行时核对真实
CUDA/HIP 后端，不接受设备声明错配。自动选择 generic 和本设备同名 vendor，
其余源码仍全部携带、验签和静态审查，但不强制在错误设备上导入执行。
明确可兼容代理的 vendor 可重复添加 `--proxy-vendor ascend` 等参数；选中的路径
与本设备路径一样必须通过完整矩阵，不能在失败后删除选择来隐藏已知正确性问题。
本次改动的 vendor 若可在代理执行，必须加入该选择；只有确需目标 runtime 且没有
可用执行通道时才记录未验证，不能用默认范围省略本可执行的受影响路径。
`tests/_op_variants.py` 在导入前应用 runner 的路径选择，直接 unittest 不受该选择影响。
新增专属路径使用该 helper 或明确的设备测试入口，不用 blanket skip 掩盖失败。

prepare 只从 Git 对象取 release 字节；探索候选加 `--mode screening`，会取工作树
字节并标记探索证据。传输该目录后，在远端后台、串行、带 timeout 执行：

```bash
timeout 600 /home/kevin/notebook/.venv/bin/python \
  /tmp/flagos-l2norm-release/.agents/skills/flagos-operator-race/scripts/verify_release.py \
  run --directory /tmp/flagos-l2norm-release
```

runner 独占创建 v2 `verification.json` 和 `verification.log`，拒绝覆盖旧回执；记录
完整 suite 与通过方法、subTest、张量 shape/dtype、适用源码的入口及非 warmup
kernel launch 次数和环境。源码与依赖在运行前后核对哈希，结束时同步设备。
每个适用源码必须存在非空张量调用及真实 JIT launch；零测试、失败、适用矩阵的
skip/xfail、漏跑必要方法、仅空输入或未启动 kernel 均返回非零。测试模块可用
`RELEASE_REQUIRED_TESTS = ["TestClass.test_method", ...]` 绑定必须进入 suite 的
契约方法（见 `tests/test_l2norm.py`）。人工仍须审查数值 reference/容差、dtype、
tile 边界、stride、受影响分支和已知失败矩阵，launch 计数不能证明这些语义。

回执分别列出 `execution_sources`、`unexecuted_sources` 和
`target_unverified_sources`：代理执行过的其他 vendor 仍在目标芯未验证列表。
本 runner 当前只支持 CUDA/HIP 的 Triton JIT 入口；其他 runtime/TLE 启动协议需
适配后再取同等证据，不能伪填 launch。目标芯可独立使用 KernelGen 或授权主机验证，
按源码、测试和设备分别保留结果；当前 CLI 的单份回执证明适用 GPU 路径，目标证据
记入实验账本，不伪装成该 GPU 回执。目标资源缺失不单独阻断已授权平台验证；
静态检查始终不记为通过，已知相关正确性失败仍须先修复。

取回 JSON 和完整相邻日志，二者原样保留并计算 SHA。平台 preflight 和 submit
都强制校验 release 回执及其 Git 字节、覆盖和日志；旧的可选日志 SHA 不再是门禁。
旧 v1 回执可保留作历史证据；新 preflight 要求重新生成 v2，不改写旧结果补字段。
这是受信任 runner 的执行记录，不是远程硬件证明，不能排除恶意伪造的测试代码。

## 每次运行

1. 用远端 `mktemp -d /tmp/flagos-<operator>.XXXXXX` 建独立目录并设为 `0700`。
2. 只传本次 generic、vendor、`tests/test_<operator>.py` 及测试实际导入的文件，
   保留仓库相对路径。screening 在传输前记录 base commit 和当前工作区各文件哈希；
   release 先运行打包器 dry-run 或 existing 验签，取得 source commit 的成员哈希，
   测试及仓库内导入依赖绑定 verification commit，并用 `git archive`、`git show` 或
   等价的 Git 对象读取方式建立临时目录。screening 候选哈希还必须与晋级后对应
   Git blob 的 SHA-256 相同；不同时把它视为新候选，重新筛选或仅以独立 release
   结果判断。远端哈希必须逐项相同；任一不一致就停止，不能拿 HEAD、工作树或
   screening 的结果替旧 commit 的 ZIP 背书。
3. 用 `setsid`（或等价方式）建立属于该临时目录的独立进程组，再配合 `nohup` 在
   远端后台执行需要 runtime 的阶段；静态检查可先在本地仓库完成，绑定同一文件哈希，
   不必在远端重复安装格式工具。完整检查项为：
   - 目标源码和测试的 `py_compile`；
   - 优先运行仓库定义的 `pre-commit run --files <本次 Python 文件>`；远端没有
     `pre-commit` 时使用与 `.pre-commit-config.yaml` 等价的命令：
     `black --check`、`isort --check-only --profile black --line-length 80`、
     `flake8 --ignore=F405,E731,W503,E203,E704 --max-line-length=120`；
     Black 读取仓库 `pyproject.toml`（当前 79），不能误用 isort 的 80。
     若在隔离目录运行格式检查，须同时携带并绑定仓库配置文件；
   - 复验源码和测试 SHA-256，确认静态门禁没有改写已验签字节；
   - 上述 runner 的 `run`（内部使用 unittest 模块加载并强制核对执行数）。
4. 启动前为每阶段和整次运行设定并记录 wall-clock 上限，命令使用远端 `timeout`
   或等价机制，禁止无界 `nohup`。超时或任一步失败时，先核验 PID、PGID 和命令均
   属于该临时目录，再只终止该进程组并停止后续阶段；不终止其他用户或任务进程。
   只有已记录原因时才扩大上限重试。
5. 记录远端目录、PID/进程组、日志路径和启动时间；前台继续本地源码审查、上游
   检索和账本准备。有限次轮询日志，不用长时间阻塞 shell。
6. 完整 unittest 已覆盖的正确性矩阵不重复跑；性能候选再做 wrapper-inclusive benchmark 和编译
   资源检查。性能实验先声明 affected、晋级阈值和资源上限；存在明确未受影响路径
   时再加 control。确认设备没有竞争 workload 后同步执行至少五轮交替 AB/BA，按
   paired speedup 的稳定统计判断。不终止他人 workload；存在竞争、结果落在噪声内
   或资源明显退化时，不晋级候选并记录负结果。
7. 记录 GPU、driver、Python、PyTorch、Triton 和 CUDA 版本，以及最大实际验证
   shape。账本保留可直接重放的完整命令（或脚本完整 SHA-256）、完整输入文件哈希、
   完整日志 SHA-256/保留位置和 AB/BA 原始样本；不截断哈希。远端 NVIDIA 结果只
   标记为代理证据。

取回回执的传输必须完成且成功退出后，才运行本地验签和汇总。不能把“后台任务已结束”
当成“日志已取全”；先核对预期算子集合和所有回执，再生成汇总。

shell 包装必须 `set -euo pipefail` 或逐阶段显式检查返回码；禁止用 `command | tail`
的退出码或最后一行 DONE 代替测试状态。计时与正确性回执分开：性能变更保留阶段
计时及五轮配对原始样本，纯正确性修复无需为了流程重复整套性能调参。

release 完成后生成 ZIP 时，再核对最终打包器输出与 release 前 manifest 的 source
commit、成员集合、成员 SHA-256 和 canonical ZIP SHA-256；任一变化都需要重新做
release，不能只重打包后沿用旧证据。

使用远端临时目录中的源码和测试做验证，不在远端仓库分支提交或 push。NVIDIA
代理只验证实际可执行的 generic/NVIDIA 路径；其他 vendor 路径若不能在该环境执行，
只做静态检查并明确标记 runtime 未验证。日志和临时目录至少保留到对应实验账本
commit 完成；若要删除，先确认账本已包含复现实验所需的命令、版本、哈希和结果。

需要新增非 NVIDIA 远端验证环境时，先按
[SkillHub 工具集成](skillhub-tools.md)的 gpu-container-setup 小节做 vendor
检测、镜像选型和挂载（昇腾/海光/Metax/天数的设备与坑见该文档），容器内
跑 `validate_pytorch.py` 通过后再套用本文件的目录、哈希和超时纪律；昆仑芯
不适用该 skill，沿用现有流程。

## EasyTier 链路 SSH 稳定性预案(2026-08-29 实战沉淀)

本项目的 `gpu` 代理(kkgpu/192.168.5.204)经 EasyTier 隧道访问。
实测病灶与对策:

- **MTU 黑洞**:`ping -D -s <size>` 实测路径仅过 ≤1330 字节 DF 包,
  而 TUN MTU 配 1360 时 SSH kex/大段被静默丢弃,表现为小包命令通、
  长会话挂死、阵发性不可用(2026-08-29 故障 3.5 小时的根因)。
  诊断:`route get <ip>` 找 TUN 接口;`ping -D -s` 二分探测阈值。
  修复:两端 TUN MTU 降到 1280(gpu 侧已在 easytier compose 固化
  `--mtu`,Mac 侧需 GUI 或 sudo 设置)。
- **直连路径**:kkgpu 已作为 EasyTier 节点直连入网(docker compose
  `~/easytier/`,虚拟 IP 固定 `10.126.126.6`),SSH 走 `gpu-et`
  别名(10.126.126.6),不经子网代理节点。注意:节点本身在
  192.168.5.0/24 物理网内时,peer 列表不要写该网段物理 IP(会被
  mini 通告的代理路由吸进 tun0 导致 HostUnreachable)。
- **弱链路作业纪律**:连接只做触发和取件——重活 `setsid nohup
  timeout … &` 远端后台执行、结果落盘;回传大文件用 `gzip | base64
  -w0` 小包;任务严格串行(并行曾挤爆 16GB 显存);退避重试间隔
  ≥2 分钟;同网段其他主机可做跳板(`-J`)备份路由。
