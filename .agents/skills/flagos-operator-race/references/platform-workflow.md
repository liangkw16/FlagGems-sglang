# 平台提交与逐芯结果

只在用户明确要求实时平台预检、提交、查看评测或基于逐芯结果迭代时读取。

## 提交前本地只读验签

仅当本次要选择文件或提交时执行本节；纯状态或评测查询直接使用下文只读脚本，
不要求存在候选 ZIP。提交前先对实际候选做只读验签：计算 ZIP SHA-256，检查小于
10 MB、ZIP 完整性、安全普通文件、UTF-8/可编译、operator basename 和允许的
vendor 后缀。规范产物使用打包器 `--verify-existing`；安全子目录中的历史包按
basename 和提交源码内容验签，不重写字节。实际成员清单必须与候选账本明确列出的
generic/vendor 集合一致，不能因打包器自动收集 commit 中已有 vendor 而夹带文件。

同时建立 provenance：source commit、成员来源与 SHA、实际 ZIP SHA 必须能互相
对应。验签或来源不完整就停止，不进入平台预检和自动提交。

## 脚本查分

日常查分不操作浏览器。首次使用时，优先通过平台官方邮箱或手机号验证码接口登录；
命令会交互读取账号和验证码、用 IAM 接口验真，并将 token 原子写入 Git 内部的
`.git/flagos-token`（`0600`），不会打印 token，也不读取浏览器 cookie/localStorage。
必须使用已有 FlagOS 账号绑定的邮箱或手机号：

```bash
python .agents/skills/flagos-operator-race/scripts/platform_cli.py auth \
  --method email --accept-terms
# 手机号登录改为：--method phone --accept-terms
```

`--accept-terms` 是显式门禁：当前官方端点兼具登录/注册能力，邮箱或手机号输错可能
创建新账号。认证接口来自当前生产前端，若平台协议变化则命令失败关闭并回退网页登录，
不尝试抓取浏览器凭证。

后续 `status`、`preflight`、`submit` 会自动读取该文件：

```bash
python .agents/skills/flagos-operator-race/scripts/platform_cli.py status \
  --race 782kzq4m --batch 2 --task 12 --operator chunk_state
```

持续等待终态时追加 `--watch --interval 15 --timeout 900`。提交后优先直接运行 `submit`
返回的 `watch_command`；它同时绑定 `file_url` 的 SHA-256 和提交前最新记录时间，尚未
出现本次记录时不会误把同一 Task 的旧终态当成结果。`status` 只发 GET，不创建竞赛
账本或本地 intent，同时会验证 token、账号、团队、race、Task、额度和提交记录。输出
中的 `file_url` 会移除
可能存在的签名 query，并另给 `file_url_sha256`；内部去重仍使用完整 URL。
若验证码接口不可用，也可把用户合法持有的 token 一次写入 Git 内部目录；文件必须
为普通文件、绝对路径且权限不宽于 `0600`：

```bash
flagos_token_file="$(git rev-parse --absolute-git-dir)/flagos-token"
(
  umask 077
  IFS= read -r -s flagos_token
  printf '%s\n' "$flagos_token" > "$flagos_token_file"
)
unset FLAGOS_TOKEN
export FLAGOS_TOKEN_FILE="$flagos_token_file"
```

不要把 token 作为命令参数、写入仓库或从浏览器 cookie/localStorage 提取。

远端 ZIP 自动验签默认 fail closed。仅把平台官方资料或既有 `status` 输出中已核实的
对象存储 hostname 精确设为 `FLAGOS_REMOTE_ZIP_HOST`；不要从本次尚未信任的上传响应
自动派生。未配置或本次 `file_url` hostname 不匹配时不发下载 GET，只报告
`remote_verification.status=unavailable`。

```bash
export FLAGOS_REMOTE_ZIP_HOST='<已核实的对象存储hostname>'
```

## 脚本提交

本地验签通过后运行 `preflight`；它调用规范打包器的 `--verify-existing`，并只读核对
race、登录账号、登录团队、精确 Task/tid、batch、截止时间、平台实时最小提交间隔、
提交记录快照和当前剩余额度。完整 commit、ZIP SHA-256、ZIP 绝对路径和成员必须由
调用者显式提供：

```bash
python .agents/skills/flagos-operator-race/scripts/platform_cli.py preflight \
  --season 2 --race 782kzq4m --account '<账号>' --team '<团队>' \
  --batch 2 --task 12 --operator chunk_state --stage e2 \
  --commit '<40位commit>' --zip '<ZIP绝对路径>' --sha256 '<64位SHA-256>' \
  --member chunk_state.py \
  --verification-commit '<40位测试证据commit>' \
  --test-sha256 '<该commit下tests/test_<operator>.py的SHA-256>' \
  --release-log-sha256 '<release/screening日志SHA-256，可省略>'
```

`--verification-commit` 与 `--test-sha256` 是硬门：preflight 会用 git
核验该 commit 存在、`tests/test_<operator>.py` 在该 commit 的 blob
SHA-256 与给出值逐字节一致（T37 尾块缺陷上平台的教训——source commit
本身不说明哪些测试被跑过）。`--release-log-sha256` 作为人证 receipt
一并写入 intent 供审计。

每个 vendor 文件再增加一个 `--member chunk_state_<vendor>.py`。预检不发 POST；它在
Git 内部目录 `.git/flagos-platform/` 创建权限为 `0600`、十分钟有效的一次性 intent，
并打印完整 tuple、随机 nonce 和 `confirm_command`。命令中的 `--confirm` 只消费已经
验签的机器 intent，不表示人工确认。当前任务包含平台提交、完整闭环或继续既有竞赛
闭环时，逐项核对 tuple：

- race ID/赛季、登录账号、登录团队、batch、Task 编号、tid 和 operator 精确匹配；
- source commit、stage、成员集合、ZIP 绝对路径与完整 SHA-256 匹配账本和本地证据；
- verification commit 与其 `tests/test_<operator>.py` blob SHA-256 匹配
  （preflight 强制校验），release log SHA（如有）与账本一致；
- Task 为 `competing`；或为等待首个有效解的 `pending_challenge`，且平台同时明确返回
  `status=submitting`、`can_submit=true` 和 `challenge_operator`。此外必须处于提交
  时间窗、最小间隔已满足且当前剩余额度至少为 1。

全部匹配就立即运行预检输出的 `confirm_command`，无需再向用户询问。任一字段缺失、
变化或不一致都视为自动提交门禁失败，停止并报告，不能靠交互确认绕过。intent 在任何
POST 前过期或变成 `stale` 时，可重新运行只读 preflight 并重新核对；只要门禁仍全部
满足即可自动提交。每个新候选必须生成新的 ZIP SHA 和 intent。

只运行预检输出的命令：

```bash
python .agents/skills/flagos-operator-race/scripts/platform_cli.py submit \
  --confirm '<本次nonce>'
```

`submit` 持本地锁，重新验签 ZIP 并复核整个 live binding；任一值变化都把 intent 标为
`stale`，且不发 POST。状态不变时只读入一次 ZIP 字节，用同一份字节执行“上传文件”
和“正式提交”各一次。发送前 intent 先落盘为 `sending`；成功后为 `submitted`，发送
阶段任何异常为 `uncertain`。非 `prepared` nonce 永不复用，也没有 `--force`、自动
重试或可替换 API host。同一 race/account/team/Task/operator/ZIP SHA tuple 已存在有效
intent、未决发送或成功记录时不生成第二个 nonce；新候选必须有新的 ZIP SHA。

正式提交 POST 成功后，脚本先把 intent 原子写为 `submitted`，再用独立的无认证 HTTPS
请求下载上传接口返回的同一 `file_url`，核对实际字节数和 SHA-256。远端验签结果为
`verified`、`mismatch` 或 `unavailable`；后两者只表示远端字节未确认，不能把已经成功
发送的提交改成 `uncertain`，也不能据此重试提交 POST。下载请求不携带 Bearer token/Cookie、
不跟随重定向，且只读取本地 ZIP 长度再加一个字节。

出现 `sending`/`uncertain` 时先用 `status` 核对提交记录和额度，不得直接重试。若
提交记录出现同一 `file_url`，下次预检只会把旧 intent 标成已提交并拒绝重复；若仍无
可定位结果，脚本保持阻塞，不删除或手改 intent。实际平台响应和逐芯结果仍写入对应
实验账本。`watch` 超时或长期查不到本次记录时保留 intent、POST 响应、nonce、
`file_url_sha256`、提交时间和额度变化，稍后只读重查或交平台支持；纯查分请求只输出，
不修改账本。

脚本因验证码、新风险提示、认证协议或 API schema 漂移无法执行时，才回退到
`chrome:control-chrome`，且浏览器**只做只读操作**：核对登录账号、团队、Task、
剩余额度、既有提交和逐芯结果。上传与提交一律不在浏览器里自动点击——CLI 的
intent/锁/幂等键状态机不覆盖网页路径，脚本状态不确定时重复点击等于绕过
单发纪律。需要网页提交时明确告知用户并交人工执行，在账本记录跳过 CLI 的
原因；CLI 恢复后回到脚本路径。

用户人工在网页完成提交后，若平台返回 HTTPS `file_url`，仍按同一标准下载远端
ZIP，核对实际字节数和 SHA-256 与本次候选 ZIP 完全一致；页面展示的文件大小
不作为验签证据。下载失败或哈希不一致时保留“提交已发送”事实，记录远端验签
未确认/失败，并停止重试提交 POST。

## 失败情报通道与踩坑沉淀（2026-09-04 实战）

### 失败详情情报通道（优先于浏览器/工单）

`status` 输出把 `raw_result` 过滤掉了，但平台列表接口本来就带完整
失败详情。用 CLI 同源 token 直接 GET（只读）：

```python
client = HttpClient(_token())  # platform_cli.py 内部类
raw = client.get(f"{API}/races/{race}/operator-submissions",
                 {"page": 1, "page_size": 100})
for r in raw:                       # 返回即列表
    if r.get("submission_id") != 目标: continue
    for g in r.get("gpu_results") or []:
        rr = g.get("raw_result") or {}
        # rr["errors"]: 超时/崩溃/编译错全文
        # rr["failed_cases"]: [{error, params:{case_idx}, test_name,
        #   traceback, ...}] 逐 case 的断言差异与 shape 线索
```

- 先看这个再谈浏览器/工单；本轮 5 个"不可解"失败根因全部来自这里。
- 浏览器只读通道另有 IAB 标签页跨调用被重置回 about:blank 的问题，
  且失败详情在「我的参赛」后要登录——API 通道两者皆免。

### 平台评测踩坑硬事实（2026-09-04 批次实证）

| 坑 | 表象 | 处置 |
| --- | --- | --- |
| 平台 permutation 等 index 张量是 int32 | 昆仑 `index_copy_(): Expected a long tensor for index, but got Int`（NVIDIA 接受 int32，代理全绿是盲区） | wrapper 内对平台传入索引统一 `.long()` 再进 index_select/index_copy_ |
| 昇腾 BiShengHIR UB 预算 1572864 bits | 64 tile 形态报 `MLIRCompilationError: ub overflow, requires 2146304~3694592 bits` | 用 32³（或更小）tile；E2/E3 两发学费确认与数值无关 |
| 燧原 grid.y 硬限 255 | `OutOfResources: grid.y, Required: 256, Hardware limit: 255`（batch=256 放 y 轴即炸） | batch 维放 grid.x 或折叠；逐轴限制记忆（grid.x 另有小上限） |
| Triton 3.7.1 多趟 K 循环 codegen 错 | vendor GEMM rank=32 精确、rank≥64 第二趟起错 ~1e1（独立复刻加一条 store 即不复现） | `BLOCK_K=next_pow2(K)` 恒单趟；回归矩阵必须含 rank>BLOCK_K 的 case |
| 评测机负载超时 | 同字节内核上轮 0.25x 通过、本轮 1830s 超时（R 状态机器忙） | 慢内核要留性能余量；超时≠代码回归，先比对同字节历史再动 |
| 昆仑评测器崩溃族新表现 | `执行超时(1830s/1800s) + Subprocess crash: Fatal Python error: Aborted`（compile_worker 栈） | 按崩溃族协议：不计代码止损、封存等健康窗口 |
| make_block_ptr block_shape 必须 2 幂 | `Expected a list of constant integers` / `Shape element must be a power of 2`（K=100/96） | block 维用 next_pow2 填充 + boundary_check；shape/strides 可 runtime |
| 字符串补丁在 black 折行字节上静默未命中 | replace 无 assert 时"看似修复"实未命中（beta/g 漏加 pid_t*BT 白跑一轮） | 对已格式化文件做 replace 必须先 assert 旧串存在 |

### 结构资产（可直接迁移）

- **GQA 共享 dot**（源自 vllm-project/vllm-ascend#7576）：每个
  k-group 只算一次 `dot(k, trans(k))` + 掩码，组内 HPG 个 head 仅做
  逐 head 缩放与存储——generic 每 head 重算是 ratio 倍浪费；正确性
  与性能双收益（华为 0.031→correctness 绿；燧原 0.031→1.62x）。
- GitHub 情报源：vllm-ascend / flash-linear-attention-npu / sglang
  的 PR 直接搜算子名；竞赛上游 flagos-ai/FlagGems-sglang 的 PR 是
  其他队结构的公开泄露口。

## 逐芯结果与最小迭代

评测记录可能需要主动切换“题目说明 → 提交代码”刷新。逐芯记录：正确性、
speedup、平均值、状态、失败详情、排名、首次有效提交时间和剩余额度。
账本中的额度数字必须取自本次 `status` JSON，并同时记录 `observed_at`
与该次提交的 `submission_id`/`daily_seq`；带时间戳的快照章节被后续
信息修正时，另起带日期的小节，不回改旧标题下已记录的数字。页面不展示
独立 submission ID 时，用 `Task + 文件名 + 时间` 联合定位，不编造 ID。

按根因分类：

| 现象 | 最小动作 |
| --- | --- |
| 多芯同类数值失败 | 修 generic 根因，不加多份 vendor |
| 单芯编译/正确性失败 | generic 保持不变，只加该 vendor |
| `grid.x` 超硬件上限 | 从固定 vendor policy 取 grid 上限，改 grid-stride |
| 输出像归一化中间值 | 先将最终缩放、scalar broadcast 或 lowering 顺序作为假设，用精确回归或编译产物证实后再修复 |
| 正确但低于题面门槛 | 先恢复门槛，再做性能排名 |
| 通过但明显落后榜首 | 每次只改 BLOCK、grid、warps、数学 lowering 或布局之一 |
| 单芯仅评测器崩溃（compile-worker/inductor/Segfault/服务线程卡死，空 `failed_cases` 或纯基础设施栈） | 平台侧故障，不计入该题止损；按下方崩溃族协议留证、封存候选、等健康窗口重投 |

### 平台评测器崩溃族协议（昆仑实证）

同一题连续出现同指纹平台侧崩溃（约 1830s compile-worker 崩溃、
inductor `subproc_pool`、Segfault、服务线程卡死自动恢复）时：

1. 记录崩溃族证据链：同指纹次数、跨题相关性（topk/argsort/matmul/
   einsum reference 族）、同载体双投、同窗口他题对照，并更新平台工单；
2. 该类失败不计入代码止损 stop gate；候选 ZIP 封存并在账本标注
   "平台修复即转正"。止损纪律中"两次同指纹失败关轴"仅指代码侧指纹，
   与本条冲突时以本条优先；
3. 崩溃族重载**不属于常规自动提交授权**：每一发（含注释载体）都必须
   先获得用户当次明示授权。首选路径是平台工单的健康 worker rerun
   （不耗我方额度、不产生新提交记录）；注释载体改注释会产生新 ZIP
   SHA、让相同执行代码绕过候选去重，只作为用户明示授权下的最后
   手段，且每发都在账本累计探测计数。同字节重投不可执行——CLI 对
   相同 race/account/team/Task/operator/ZIP SHA tuple 直接拒绝；
4. 重载触发条件（授权之外还需同时满足，缺一不发）：对应题公开达标数
   较基线快照上升，且平台工单有回应或明确修复公告；同题探针连续
   2 发仍崩即回到封存，只保留工单路径；
5. 判定性对照（T31 E7 实证）：他队同窗口在该题八芯通过、而我方重载
   即崩——崩溃由我方 kernel 惯用法触发的假设坐实，立即停止探针，只剩
   结构改写（换 topk/索引/超越函数形态）或工单两条路。

vendor 文件必须自包含、保持同一函数签名并导出同一 `__all__`。已经通过的芯片
继续使用原 generic，避免无关回归。一次提交只改变一个可解释变量；若为恢复
正确性必须同时消除已知 grid 风险，在账本明确说明。

每轮重复：最小回归 → 适用路径的远端代理、MCP 覆盖芯的发射前实机初筛
（触发条件与验收门见 SkillHub 工具集成）或明确的静态未验证 → 新 commit →
新 ZIP/hash → 新 preflight → 单次自动提交 → 平台。默认建议至少预留两次额度给
截止日前最终回归；实时剩余额度必须写入账本和最终回复，不能靠重复提交碰运气。

止损与预算纪律（候选预注册另有约定时以预注册为准）：每个候选提交前
预注册晋级门（受影响芯阈值 + 平均阈值）与 stop gate（两次同指纹失败
或单芯显著回退即关轴）；同字节方差重掷最多两次低滚后关闭该重载路径；
每题提交预算默认按当批作战方案执行，用户明示解除时以最新指示为准。

## 第 4 批全战役经验补充（2026-09-05 收盘）

### 常胜结构（可直接迁移到后续批次）

| 结构 | 来源 | 已验证芯 | 迁移要点 |
| --- | --- | --- | --- |
| **route/materialize + long-index** | T47/T48 | 燧原+昆仑 | wrapper `index_select` 物化 + 每段规则 GEMM + `rows.long()`；昆仑必加 long |
| **FLA persistent** | T45 (vllm-ascend#7563) | 华为 | 物理 AI core grid + `tl.range` task stride + 每 task 一个 (chunk, batch×head) + UB-aware BK |
| **宽度轴归约** | T43 (Codex P1b) | 华为+燧原 | `[W_PAD≤8, D_BLOCK]` 2D tile + `tl.sum(axis=0)`；wrapper cat state+x + weight 预转 [W,D] |
| **一 program 一行** | T51 (T20 姐妹) | 燧原 | 去 grid-stride 循环；`tl.rsqrt` / `tl.sigmoid`；grid=(rows,) |
| **precomputed-pos** | T49 | 燧原 | wrapper PyTorch 预计算位置/路由张量；kernel 变纯 gather 零分支 |
| **vectorized flat** | T53 | 昆仑 | 标量循环→1024-lane 向量化；wrapper contiguous+view(-1) |

### Online softmax 关键 bug（T50 教训）

多 block KV 循环中 **`m_val = m_new` 必须在每次迭代内更新**。缺失时第二个 block 的 `alpha = exp(m_val_old - m_new)` 会把第一个 block 的累积清零。一行修复，4/4 测试全过。渐进定位法：1ext+63pre PASS → 1ext+64pre FAIL → 锁定 BLOCK_N 边界。

### 评测机繁忙模式

燧原评测机会周期性过载（连续 5×1830s 超时后恢复）。同字节平台已过 → 暂态。不要连续重试烧额度；等窗口恢复后最小变更重掷。

### 第 4 批 Day-2 新教训（2026-09-06，T54/T58）

| 教训 | 来源 | 细则 |
| --- | --- | --- |
| **tile 每根轴都要进 mask** | T54 s0 0/8 | `[HEADS_TILE=4, BLOCK_D]` 只 mask d 轴，H=2（平台 case0）时多出 head 行读进 V 区、越界写坏下一 token 输出槽 → 49% 错。本地测试 H∈{4,8,16} 全被 4 整除未暴露。**测试矩阵必须覆盖非整除 tile 的每根轴值**（H∈{2,3,5,7}） |
| **昆仑向量整除崩溃** | T58 s0 7/8 | 逐 lane `offs // runtime_scalar` 触发 PassManager::run failed。当 BLOCK 整除组宽（pow2 嵌套恒成立）时组索引跨 tile 恒定：改标量 `(pid*BLOCK)//group` 一发修复且 142x。注意 T53 昆仑 vendor 里向量整除又能跑——**按内核形态区别对待，崩溃即标量化** |
| **2D tile + axis=1 归约降级** | T54 e1 昆仑1-3%/华为大case | 行数据用 `[H_TILE, D]` 2D tile 归约在昆仑/华为出小比例大偏差；"一 program 一行"1D 归约 vendor（T51 形态）是兜底正解 |
| **题面注意事项=硬约束** | T58 | "int8 须先 cast fp32、不可 int8 直接 GEMM"虽只写在注意事项，仍按规执行（fp32 ieee dot 照样 118x），不给判罚留把柄 |
| **本地测试应对齐平台契约** | T58 bf16 scale 子测 | 自造的 bf16 scale 子测比平台（fp32 scale + atol0.5）严 50 倍，reference 自身的 bf16 乘积误差会假阳性卡候选 |

### T50 extend_attention 开发经验

从零到 5/8 (208 发 1 队过线题)：
- per-query 标量 online softmax 是正确性最稳的起步结构
- `[:, None]` 广播在所有 position×dim 交叉处必须显式标注
- 华为 correctness 问题在 softmax 算法之外（online/two-pass 等价数学均败）
- GQA 用 `q_head // GROUP_SIZE` 映射最简
