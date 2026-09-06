# SkillHub 工具集成

`.agents/skills/` 下的 SkillHub skill 只作为闭环加速器，不改变契约锁定、
代理验证、不可变 ZIP 和平台门禁。本文件记录它们在竞赛闭环中的具体用法和
可复用的跨芯技术事实；边界规则以 SKILL.md 为准。这些是第三方 skill，脚本
拥有完整 agent 权限，首次调用前先审阅其脚本。

## kernelgen-flagos：MCP 生成/优化/特化/验证

### 启用前提

选择 MCP 生成路线时使用 `kernelgen-mcp` 服务，配置写入 `.mcp.json`（已配置，
Token 属个人凭据，`.mcp.json` 在 `.git/info/exclude` 本地排除，不入库）：

```json
{
  "mcpServers": {
    "kernelgen-server": {
      "type": "http",
      "url": "https://kernelgen.flagos.io/sse",
      "headers": {"Authorization": "Bearer <TOKEN>"}
    }
  }
}
```

端点以实测为准（2026-08-28）：`https://kernelgen.flagos.io/sse`（无尾斜杠，
Streamable HTTP 传输，仅收 POST；skill 原文档的 `type: "sse"` 和尾斜杠写法
会 307 跳转到网页导致连不上）。注册入口 https://kernelgen.flagos.io/mcp，
手机号验证码登录，未注册号码自动注册并需填一次试用申请（姓名/单位/机构
邮箱/用途）。Token 登录后存在页面 localStorage `userLoginInfo.mcp_token`。
选择 MCP 路线但 Token 缺失时如实报告；独立已有方案可继续验证。MCP 代码视为未验证草稿。
注意本仓库布局是 `src/flaggems_sglang/`，不是上游 `src/flag_gems/`，
kernelgen 自动检测会按通用流程路由；其 FlagGems 专用子文档中的注册、
测试布局不适用，落盘位置以本仓产物布局为准。2026-09-06 tools/list 的
device 描述列出 nvidia/huawei/haiguang/tianshu/muxi/moore/sunrise/kunlun/amd；
这是接口接受的目标列表，具体设备排队、在线与执行能力以每次响应为准。

### 使用范围与调用通道

KernelGen 可做多 GPU/多芯片验证；`gpu` 主机也是验证通道，KernelGen 还可生成算子。按任务选择通道，
不要把 MCP 限定为代码建议，也不要把改写响应自动当作设备验证结果。
先复用本仓和固定上游源码。新算法或缺少实现时用 generate；需要第二实现、跨芯
建议或瓶颈分析时用 optimize/specialize。明确根因修复、成熟实现移植和参数调整
不强制依赖 MCP。调用前带上精确契约、当前源码、逐 shape 基线和已有失败证据。

原生 `mcp__kernelgen-server__*` 未挂载时，使用已审阅的脚本客户端：

```bash
python3 .agents/skills/kernelgen-flagos/scripts/kernelgen_mcp.py list
python3 .agents/skills/kernelgen-flagos/scripts/kernelgen_mcp.py call <tool> --file request.json
```

配置存在、工具能发现、服务能连通、任务实际执行是不同状态。schema 以本次 tools/list
为准；不把历史字段或支持列表当作当前已验证能力。请求与响应落盘并记录完整 SHA-256；
凭据不进入请求文件、日志或 Git。纯审计不启动生成或设备任务。

2026-09-06 实时 tools/list 的调用分工（服务更新后重新核对）：

| 接口 | 作用与验证能力 | 调用要点 |
| --- | --- | --- |
| `generate_kernel` | 生成 PyTorch/Triton/test/benchmark，并返回 `verify_result` | 传契约、参考资料、目标 device；核验生成测试是否覆盖题面 |
| `autotune_kernel` | 生成、迭代并在目标设备验证 | 传 `pytorch_code`、`test_func_code`、`input_specs`；设置 `max_rounds` 和 `verify_timeout`；按 `continue_call` 原字段轮询 |
| `optimize_kernel` | 单次优化；服务描述明确“不验证、不迭代” | 返回源码另走验证通道，不能把响应成功记为数值通过 |
| `specialize_kernel` | 平台特化，当前知识库仅 huawei | 输出是候选；不推断已完成目标设备测试 |

`autotune_kernel` 当前没有固定 Triton 源码输入字段，不能靠提示注入宣称原样执行了
本地候选。它能验证服务生成的算子；需检查每次返回证据指向的实际版本。验证当前
仓库字节时优先用 `gpu` 的隔离测试，KernelGen 返回的新代码接入本仓测试再复验。
若服务后续提供固定源码执行接口，再按新 schema 使用，不虚构字段或接口。

### MCP 证据分级（2026-09-06 修订）

历史探测中 optimize/specialize 是改写路径；autotune 会多次改写并自选 harness。
第四批存档 24 份输出中，22 份的 `total_tests=0`，2 份是 502；这不是题面正确性验证。
历史注入实验见实验 README 的 2026-09-03 记录，下表取代旧“失败神谕/保真即初筛”规则。

| 信号 | 可采用的结论 | 不可采用的结论 |
| --- | --- | --- |
| 返回代码 | 新候选、结构建议；保存源码和差异 | 已编译、已正确、已加速 |
| 某 attempt 有编译/运行错误 | 服务端该次运行失败；保留原始错误，分开识别 harness/502 | 没有该 attempt 源码哈希时，不能归罪于当前候选或关闭方向 |
| 最终代码与输入一致 | 最终返回字节一致 | 不能反推每次 attempt 执行相同字节；忽略注解/改名后的 diff 也不是字节相等 |
| passed=True、tests=0、代码注释中的计时 | 无可采信的正确性或性能结论 | 不得晋级或代替 GPU 回归 |
| 执行源码+harness+reference 哈希、环境、逐 case 结果可核验 | 仅在实际覆盖范围内记录编译/数值结果 | 未覆盖 shape、其他芯、不同版本不获背书 |

当前未取得完整绑定结果时统一记 `mcp-unbound-observation`；不用
`mcp-device-screened` 或 `mcp-compile-screened(fidelity)` 表示候选已验证。
数量大于零的 tests 也不足以单独升级，仍需核验执行源、reference、case 与环境。
MCP speedup 的口径未对齐前仅保留原始数据，不用于晋级、关轴或平台名次推断。
不要要求 LLM 用注释回传测量值。代码注入提示不能替代实际执行接口或机器结果信封。

2026-09-06 流程实测：对同一 L2 契约分别调用 NVIDIA/Huawei 的 generate，以及
显式传入 21 项 pytest 矩阵的单轮 autotune，四份终态均返回 passed=true、tests=0。
这证明生成、设备路由和任务轮询可用，但缺少可核验的实际用例计数和执行环境；
不能据此宣称没有运行过测试，也不能记为多芯正确性通过。请求/响应保留在
`artifacts/competition/workflow-fix-20260906/kernelgen/`，实测报告见
`docs/competition/workflow-validation-20260906.md`。
遇到该结果时停止空转重试，保留服务结果并将返回源码接入 `gpu` 的独立回归；
目标芯结果仍待服务提供有效执行证据，NVIDIA 复验不替代该目标芯验证。

### 多芯验证通道与实际资源

按任务的薄弱芯片和本次改动选择 KernelGen 目标设备，分别携带同一契约、reference、
用例矩阵与失败上下文启动验证；不要因为只有一台 SSH GPU 就放弃其他芯片验证。
各 device 独立记录请求/响应、job_id、实际执行版本与结果。服务若生成/改写了不同
版本，分别标为该版本的验证，不能汇总成同一源码多芯通过。服务并发最多两个任务，
长任务按 continue_call 轮询，前台继续独立工作；设备不可用与算子失败分开记账。

| 芯片 | 历史 MCP 服务覆盖 | 容器方案 | 当前目标芯执行状态 |
| --- | --- | --- | --- |
| 华为/天数/海光/沐曦 | 有历史请求；每次另查可用性 | 有镜像和挂载方案 | 主机授权、可达性、环境及候选结果分别登记 |
| NVIDIA | 有历史请求 | 有 | `gpu`/`gpu-et` 数学代理；不映射匿名 A/B |
| AMD | 当前 schema 列出，未核验设备执行 | 有 | 未登记独立目标机，不从 NVIDIA 结果推断 |
| 燧原/昆仑 | 当前 schema 列出 sunrise/kunlun，未核验设备执行 | 本 skill 未建立 | 可尝试服务验证；未实测就标未知 |

实际资源登记见 [validation-resources.md](validation-resources.md)。镜像支持不等于
拥有硬件；gpu-container-setup 不租用主机。仅在已授权主机上部署匹配镜像，再记录
设备/驱动/runtime、容器 digest、最近健康检查和最大已测 shape。实际可用目标机
执行同源完整矩阵；缺少独立目标机时，可使用 KernelGen 的目标设备验证，按上表核验
执行绑定与覆盖。若仍无有效目标芯执行结果，保留 NVIDIA 数学代理与平台证据，
不虚填目标芯通过。

### 回归、定位和性能实验

- 编译/导入错误先最小复现；数值错误先定位首个分歧阶段，再决定修复或换结构。
  不规定“只能改一次”，也不因修复超过一行就强制重新生成。
- 测试以契约为准，逐轴覆盖 tile 边界、长段、多维 stride 和所有 vendor。
  持久化每个已发现失败 case。编译未触发、零测试、skip/xfail 均不等于通过。
- 代理 fuzz 零失配只覆盖已测输入与代理环境。不同源码的失败指纹不同不能证明
  非确定性；固定源码/输入/环境重复运行后，才分类随机性、数值边界或确定错误。
- 性能候选先记 affected shape、晋级门与基线；拆分 wrapper/搬运/GEMM/epilogue，
  用适用的编译产物检查确认瓶颈，再做至少五轮 AB/BA。NVIDIA 结论不外推其他芯。
- 执行回执与提交前复验按 remote-validation.md；没有可重放证据不宣布轴已证伪。

### Ascend 特化技术要点（来自 specialize 子文档，作为 _ascend 候选起始假设）

- **int32/int64 Vector CMP 标量退化是 elementwise 头号杀手（T40 平台
  实证 +130%）**：`cols < N` 类整数比较 lowering 成标量指令；规避 =
  行结构化每行一 program + 行内子分块、仅末子块带 mask（热路径零比较）。
  反证：care_padding 单用无效（需 multi-buffer 结构配合）、fp32 比较
  仅在宽度 <2^24 合法、大 BLOCK 跨行扁平化会行界污染（数值错）。
  care_padding 是 triton-ascend 专有 kwarg（NVIDIA 侧直接编译错），
  vendor 需按设备类型分流双内核。
- 动态 BLOCK_SIZE：`max(32768, triton.next_power_of_2(triton.cdiv(N, 65535)))`，
  保证 coreDim ≤ 65535。
- 核内子分块：`for sub in range(0, BLOCK_SIZE, BLOCK_SIZE_SUB)`，
  BLOCK_SIZE_SUB 从 1024 起，候选 512/2048/4096。
- `tl.load(..., care_padding=False)` 减少依赖；wrapper 用 `@libentry()` 并
  在 `torch_device_fn.device(x.device)` 上下文内 launch（libentry 的
  dynamic_func 只吃位置参数；repo 包导入需 try/except 降级）。
- 环境变量 `TRITON_ALL_BLOCKS_PARALLEL=1` 可降低调度开销。
- 小数据（<1000 元素）在 NPU 上可能不划算，性能按题面实际 shape 评估。
- 燧原镜像约束：任何运行期分支/特殊 load kwarg 都不可用（T31 微核
  同指纹、T39 块跳过 system_failed、T40 死循环三题互证）。
以上仍需单变量验证并在账本记录 AB/BA 证据。

## gpu-container-setup-flagos：非 NVIDIA 远端容器

脚本路径以仓库内为准：
`.agents/skills/gpu-container-setup-flagos/scripts/{detect_gpu.py,find_data_disk.py,validate_pytorch.py}`
（该 skill 文档中的 `.claude/skills/...` 路径不适用）。流程：检测 vendor →
找数据盘 → 按“vendor hub → BAAI Harbor → 搜索 → 本地镜像”选镜像 → 起容器
→ 容器内跑 `validate_pytorch.py`。不覆盖昆仑芯，昆仑芯沿用现有远端流程。
新容器只用于新增非 NVIDIA 验证环境；现有 SSH alias `gpu` 流程不变。

### 检测与设备可见性

| Vendor | 检测 | 可见性变量 |
|---|---|---|
| NVIDIA | `nvidia-smi` | `CUDA_VISIBLE_DEVICES` / `NVIDIA_VISIBLE_DEVICES` |
| 昇腾 | `npu-smi info -l`，`/dev/davinci*` | `ASCEND_DEVICE_ID` |
| Metax | `mx-smi -L`，`/dev/mx*` | `MUSA_VISIBLE_DEVICES` |
| 天数 | `ixsmi -L`，`/dev/bi*` | `COREX_VISIBLE_DEVICES` |
| 海光/AMD | `rocm-smi`（海光在 `/opt/dtk-*/bin/`，输出 HCU） | `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES` |

### 镜像来源（按优先级，BAAI Harbor 兜底）

- NVIDIA：`nvcr.io/nvidia/pytorch:<YY.MM-py3>`
- 昇腾：`ascendhub.huawei.com/public-ascendhub/pytorch-modelzoo:<tag>`
- Metax：`cr.metax-tech.com/public-library/maca-pytorch:3.3.0.4-torch2.8-py312-ubuntu24.04-amd64`（容器内用 `/opt/conda/bin/python3`）
- 天数：`hub.iluvatar.com/pytorch/iluvatar-pytorch:<tag>`
- 摩尔：`registry.mthreads.com/pytorch/mthreads-pytorch:<tag>`
- 海光：`harbor.baai.ac.cn/flagrelease-public/hygon-pytorch:2.5.1-dtk25.04-driver6.3.28`
- 兜底仓库：`harbor.baai.ac.cn` 项目 `flagrelease-public`

### 挂载要点与已知坑

- 昇腾：davinci 设备 + `/dev/davinci_manager`、`/dev/devmm_svm`、
  `/dev/hisi_hdc`；driver、ascend-toolkit、`npu-smi` 分别 `:ro` 挂载
  （不要整体挂 `/usr/local/Ascend`，会与容器内 CANN 冲突）；`LD_LIBRARY_PATH`
  需含 `driver/lib64`、`driver/lib64/driver`、`ascend-toolkit/latest/lib64`
  （及 aarch64/fwkacllib 变体）；报 `set_env.sh: No such file` 时加
  `--entrypoint ""`。
- 海光：`/opt/hyhal:ro` + `-e HIP_VISIBLE_DEVICES=...` +
  `--security-opt seccomp=unconfined`；不要挂宿主 `/opt/dtk-*`。
  "No HIP GPUs available"=缺 hyhal 挂载；"ncclCommRegister undefined
  symbol"=误挂宿主 DTK；"libhsa-runtime64.so"=HYHAL 不在 LD_LIBRARY_PATH。
- Metax：`/dev/mxcd` + `/dev/dri`、`--group-add video`、`--shm-size=16g`、
  `--ipc=host`；不要挂宿主 `/opt/maca`（LLVM 不匹配）。
- 天数：`/dev/bi[0-N]` + `/opt/iluvatar:ro`。

## tle-developer-flagos：TLE 路线

仅当赛题实现选 Triton-TLE 时使用；源码真相与 API 细节见其
`references/tle-sources.md`。改原生 Triton 时用 `// begin flagtree tle` /
`// end flagtree tle` marker 包裹，marker 不进 `third_party/tle`；构建入口
先探测（`./build.sh`、`pip install -e .`），不假设固定脚本名。可直接整体
复用的方法论：

- 调参优先级：tile 尺寸 → `num_warps` → `num_stages` → copy vs 手写
  load/store → layout/swizzle。
- 单变量循环：固定 shape/seed/grid，一次只改一个参数，每步做正确性检查和
  计时；连续 3 轮无可测改进即停止——与竞赛 e1/e2 单变量纪律一致。
- 编译产物证据：`kernel.warmup(..., grid=grid)` 后读 `compiled.asm['ttgir']`
  / `['ptx']`，写入账本作为调优依据。
- 计时骨架：循环前后 `torch.cuda.synchronize()`，`time.perf_counter()`，
  rep≈50 取统计。
- 分层排障：verifier 报指针/索引=TLE API 层；能编译但输出错=kernel 逻辑或
  lowering；local store/load 后偶发错=barrier/顺序；staging 无收益=layout
  转换或流水问题（对比前后 TTGIR/PTX 模式计数定位）。

## 多芯报错速查（来自 model-verify）

- `flag_gems.* not found` → 算子实现缺失；`Triton compilation error` →
  后端编译层；`NaN/inf/numerical mismatch` → 算子精度；集合通信 hang →
  死锁。排障开关：`USE_FLAGGEMS=0/1`、`VLLM_PLUGINS=fl`。

## 上游 PR（flaggems-pr-review / pr-submit）

仅当用户要求向 FlagGems 上游提 PR 时使用，push 授权仍守本 skill 规则。
可复用的基准纪律：dtype 用 `consts.FLOAT_DTYPES` 参数化；backward 算子
对比必须用 `torch.ops.aten.<op>_backward` 做参照，否则 speedup 虚高。

### kernel-skills 快照(tensormux/kernel-skills,MIT,2026-08-29 引入)

`references/kernel-skills-snapshot/` 内置 13 个与本赛高度相关的
SKILL.md(rope/silu-mul/int8 量化/块参数调优/tile 选择/数值稳定/
跨芯无关计划等),来源 github.com/tensormux/kernel-skills(浅克隆,
只读参考,不经 MCP)。按题用法:T35/T30→rope;T29→silu-mul;
T33/T34→int8 量化与精度调试;通用→block-parameters/tile-size/
numerically-stable/backend-agnostic。第三批按题上游打法速查见
docs/competition/batch3-upstream-playbook.md。
