# SkillHub 工具集成

`.agents/skills/` 下的 SkillHub skill 只作为闭环加速器，不改变契约锁定、
代理验证、不可变 ZIP 和平台门禁。本文件记录它们在竞赛闭环中的具体用法和
可复用的跨芯技术事实；边界规则以 SKILL.md 为准。这些是第三方 skill，脚本
拥有完整 agent 权限，首次调用前先审阅其脚本。

## kernelgen-flagos：MCP 生成/优化/特化

### 启用前提

所有生成必须走 `kernelgen-mcp` MCP 服务，配置写入 `.mcp.json`（已配置，
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
Token 未提供前不手写代码替代、不静默跳过。MCP 生成代码视为未验证草稿。
注意本仓库布局是 `src/flaggems_sglang/`，不是上游 `src/flag_gems/`，
kernelgen 自动检测会按通用流程路由；其 FlagGems 专用子文档中的注册、
测试布局不适用，落盘位置以本仓产物布局为准。官网标称支持芯片为 CUDA +
华为/天数/海光/沐曦/摩尔/曦望，不含昆仑芯和寒武纪，昆仑芯 vendor 优化
用不上 MCP 通道。

### 闭环内用法

- 新算子起手（generate）：签名、dtype、stride、空输入语义以题面契约为准，
  不信任 MCP 对题面的猜测；返回的 torch 参考实现可用于交叉核对题面公式。
- 第二轮迭代（optimize）：S0 基线提交并记录 initial speedup 后，把当前
  kernel 加优化上下文（基线、逐 shape 数据、瓶颈分析、历史迭代）交给
  `optimize_kernel`；优化前必须有已记录基线，改进幅度以基线百分比计。
- 单芯特化（specialize）：`specialize_kernel` 的 `target_platform` 当前
  只支持 `huawei`；产出按本仓 vendor 文件规范放置并走远端验证。

### 发射前 MCP 实机初筛（vendor/泛芯候选）

远端 NVIDIA 代理只能证明数学与 JIT，看不见目标芯自身的 lowering 与
编译器风险。已付学费的平台实证：T35 e1 华为 2D 广播瓦片代理全绿、平台
99.6% NaN；T40 华为"无循环直通 kernel"结构被昇腾编译器两连拒收（E4
BLOCK 4096、E5 BLOCK 512 各烧一发额度才确认与 BLOCK 无关）。对受影响
芯在 MCP 覆盖集内的候选，发射前先实机初筛，用零额度换掉这类盲发：

- 触发条件（任一命中即做）：
  1. vendor 候选的目标芯在账本中有风险指纹（NaN、编译失败、数值前科）；
  2. 候选结构（新 tile 形态、新数学 lowering、去循环化）在该芯从未平台
     验证过；
  3. 昆仑重载/守榜重掷以外的昂贵弹药首投。
  同字节方差重掷不需要初筛（字节未变，初筛无新信息）。
- 工具与覆盖：`autotune_kernel(device=<chip>)` 实机跑；华为另可用
  `specialize_kernel(target_platform="huawei")`。覆盖集 = 华为/天数/
  海光/沐曦 + CUDA（国际芯片 NVIDIA 侧）；燧原、昆仑、AMD 不在内。
  需在已加载 `kernelgen-mcp` 工具的会话中调用。
- 字节绑定（先于一切验收）：MCP 任务实际执行的源码必须就是候选字节。
  调用时显式传入候选源码，把任务输入源码 SHA-256 与候选 Git blob 的
  SHA-256 一并写入账本，逐项相等后才可采信结果。autotune 返回的
  "胜出代码"是工具生成的新候选（T30 实证为 TLE-DSA 形态），只能按
  generate 产物重走 screening，不得据此给当前候选记
  `mcp-device-screened`；工具报告 `total_tests=0` 或无法核验输入字节
  时，本次降级为仅编译信号。
- 验收门（全部满足才继续 build_submission/打包）：
  1. 字节绑定核验通过（见上条）；
  2. 目标芯实机编译通过（无 CompilationError、无 PassManager 失败）；
  3. 数值信号按 MCP 实际能力取用：任务可注入并执行题面 dtype × 代表
     shape 时验证容差与 NaN/Inf 零容忍；首次在某芯使用该协议时先确认
     任务能否执行自定义 correctness case，不能时只采编译与其自带校验
     信号，不把数值验证虚记为已完成；
  4. job 输出按惯例存 `log/kernelgen-round/out_<task>_<chip>.json`
     （该目录已加入 `.gitignore`，账本只记 SHA-256），文件 SHA-256 与
     调用参数写入该算子账本。
- 明确不采纳：MCP 自测加速比作为晋级或关轴依据。T29/T30 华为实机自测
  0.037–1.79x 对平台同源 7.26x，口径不可比（其 torch 基准疑为 torch_npu
  融合实现）；性能结论一律以平台为准，账本中不得引用 MCP 自测 speedup。
- 定位与记录：初筛通过只降低发射风险，不替代远端 NVIDIA screening、
  release 门禁和平台八芯评测。账本按验收门第 3 条的实际覆盖标注
  `mcp-device-screened`（数值+编译）或 `mcp-compile-screened`（仅编译）；
  未初筛且未平台验证的芯仍标 `static-unverified`。初筛失败不消耗平台
  额度，按错误二分协议带完整错误上下文重新生成或换结构，再走初筛。

### 多芯验证通道矩阵

发射前按受影响芯选择最强可用通道。"数学代理"指在 NVIDIA 上跑通该
模块只证明数学与 JIT，不证明目标芯自身的 lowering/编译器行为：

| 芯片 | MCP 实机初筛 | vendor 容器实机 | NVIDIA 数学代理 | 平台 |
| --- | --- | --- | --- | --- |
| 华为 / 天数 / 海光 / 沐曦 | ✓ | ✓（gpu-container-setup） | ✓ | ✓ |
| 国际 A/B（NVIDIA 侧） | ✓（CUDA） | ✓ | ✓（同机） | ✓ |
| 国际 A/B（AMD 侧） | ✗ | ✓（gpu-container-setup AMD 镜像） | ✓ | ✓ |
| 燧原 | ✗ | ✗（无镜像记录） | ✓ | ✓ |
| 昆仑 | ✗ | ✗（沿用现有远端流程） | ✓ | ✓ |

- MCP 初筛最快（免容器、零额度），但只给编译/数值信号；需要题面完整
  矩阵、wrapper-inclusive 计时或编译产物证据时，起 vendor 容器实机并
  套用 remote-validation.md 的目录/哈希/超时纪律。
- 新增 vendor 文件必须同步接进该算子的 unittest 矩阵：
  `tests/_op_variants.py::load_operator_modules` 枚举 generic 与全部
  backend 变体（T35 为样板），否则该 vendor 在代理上只被静态检查。
- NVIDIA 代理是现行单点（`gpu`/`gpu-et`）；链路中断按 remote-validation.md
  的 EasyTier 预案与弱链路作业纪律执行，必要时按 gpu-container-setup
  增设备用 NVIDIA 容器。

### 可复用协议（不用 MCP 也照做）

- 错误二分：编译/导入类（ImportError、SyntaxError、TritonCompilationError、
  NameError、签名 TypeError、AttributeError）最多自查修复一次；数值/算法类
  （assert 失败、shape 不符、NaN/Inf）不盲目自改，带着完整错误上下文重新
  生成或换实现思路，避免不收敛的修补循环。
- 上下文组装：先读本仓库同类算子（`src/flaggems_sglang/ops/` 与对应
  `runtime/backend/_<vendor>/ops/`），提炼成不超过 10 条要点作为生成/优化
  上下文传入；重试时替换旧错误条目而不是追加。
- 生成代码落盘前跑首编译 smoke：多 dtype × 多 shape（含 `(0,)` 空张量），
  JIT 编译错误只在首次调用暴露。
- 首次计时样本 > 剩余样本中位数 5 倍时按 JIT 编译开销剔除并注明。

### Ascend 特化技术要点（来自 specialize 子文档，作为 _ascend 候选起始假设）

- 动态 BLOCK_SIZE：`max(32768, triton.next_power_of_2(triton.cdiv(N, 65535)))`，
  保证 coreDim ≤ 65535。
- 核内子分块：`for sub in range(0, BLOCK_SIZE, BLOCK_SIZE_SUB)`，
  BLOCK_SIZE_SUB 从 1024 起，候选 512/2048/4096。
- `tl.load(..., care_padding=False)` 减少依赖；wrapper 用 `@libentry()` 并
  在 `torch_device_fn.device(x.device)` 上下文内 launch。
- 环境变量 `TRITON_ALL_BLOCKS_PARALLEL=1` 可降低调度开销。
- 小数据（<1000 元素）在 NPU 上可能不划算，性能按题面实际 shape 评估。
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
