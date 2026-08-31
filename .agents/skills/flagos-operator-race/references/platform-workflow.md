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
  --member chunk_state.py
```

每个 vendor 文件再增加一个 `--member chunk_state_<vendor>.py`。预检不发 POST；它在
Git 内部目录 `.git/flagos-platform/` 创建权限为 `0600`、十分钟有效的一次性 intent，
并打印完整 tuple、随机 nonce 和 `confirm_command`。命令中的 `--confirm` 只消费已经
验签的机器 intent，不表示人工确认。当前任务包含平台提交、完整闭环或继续既有竞赛
闭环时，逐项核对 tuple：

- race ID/赛季、登录账号、登录团队、batch、Task 编号、tid 和 operator 精确匹配；
- source commit、stage、成员集合、ZIP 绝对路径与完整 SHA-256 匹配账本和本地证据；
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
`chrome:control-chrome`。先只读核对登录账号、团队、Task、剩余额度和既有提交；同一
完整 tuple 的门禁全部通过后，点击页面可见上传按钮触发 `filechooser`，无需再向用户
询问。点击前为 `waitForEvent("filechooser")` 立即挂成功与 rejection handler，再用
chooser 设置 ZIP 绝对路径；不要依赖点击隐藏 `input[type=file]`。提交按钮只点击一次；
chooser 超时、页面重载或结果不确定都转 `status` 只读核对，绝不再次点击。

浏览器 fallback 提交后，平台一旦返回 HTTPS `file_url`，下载远端 ZIP，核对实际字节数
和 SHA-256 与本次 intent 值完全一致；页面展示的文件大小不作为验签证据。下载失败或哈希
不一致时保留“提交已发送”事实，记录远端验签未确认/失败，并停止重试提交 POST。

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
3. 重载只允许新 ZIP SHA 的注释载体：新 commit → 新 tuple → 新的一次性
   preflight/submit。同字节重投不可执行——CLI 对相同
   race/account/team/Task/operator/ZIP SHA tuple 直接拒绝；也不得借
   改注释无限追投绕过单发纪律；
4. 重载触发条件（缺一不发）：对应题公开达标数较基线快照上升，且平台
   工单有回应或明确修复公告；每题探针总量硬上限 2 发，除非用户当次
   明示追加；
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
