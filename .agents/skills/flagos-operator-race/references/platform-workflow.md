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
对应。验签或来源不完整就停止，不进入平台预检和额度确认。

## 脚本查分

日常查分不操作浏览器。先从用户合法持有的登录凭证设置环境变量；不要把 token 写进
命令历史，也不要由脚本读取浏览器 cookie/localStorage：

```bash
read -r -s FLAGOS_TOKEN
export FLAGOS_TOKEN
python .agents/skills/flagos-operator-race/scripts/platform_cli.py status \
  --race 782kzq4m --batch 2 --task 12 --operator chunk_state
```

持续等待终态时追加 `--watch --interval 15 --timeout 900`。`status` 只发 GET，不创建
竞赛账本或本地 intent。也可使用绝对路径的 `FLAGOS_TOKEN_FILE`，但文件必须为普通
文件且权限不宽于 `0600`。

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
并打印完整 tuple 和随机 nonce。然后取得用户当次的一次性 action-time 确认，确认
内容至少包括：

- race ID/赛季、登录账号、登录团队、batch、Task 编号和 operator；
- ZIP 的绝对路径与完整 SHA-256；
- 当前剩余额度，以及本次会消耗 1 次。

旧的“继续”“可以上传”不能授权后来生成的另一份 ZIP。每个新候选重新确认。
一次确认只授权上述 tuple 的一次提交点击；点击后无论成功、失败或结果不确定，确认
都立即失效。

确认后只运行预检输出的命令：

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

出现 `sending`/`uncertain` 时先用 `status` 核对提交记录和额度，不得直接重试。若
提交记录出现同一 `file_url`，下次预检只会把旧 intent 标成已提交并拒绝重复；若仍无
可定位结果，脚本保持阻塞，不删除或手改 intent。实际平台响应和逐芯结果仍写入对应
实验账本。验证码、新风险提示、认证协议、API schema 漂移或未决 intent 需人工处置
时，改用 `chrome:control-chrome` 做只读复核，并在新的明确授权下补充恢复流程。

## 逐芯结果与最小迭代

评测记录可能需要主动切换“题目说明 → 提交代码”刷新。逐芯记录：正确性、
speedup、平均值、状态、失败详情、排名、首次有效提交时间和剩余额度。页面不展示
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

vendor 文件必须自包含、保持同一函数签名并导出同一 `__all__`。已经通过的芯片
继续使用原 generic，避免无关回归。一次提交只改变一个可解释变量；若为恢复
正确性必须同时消除已知 grid 风险，在账本明确说明。

每轮重复：最小回归 → 适用路径的远端代理或明确的静态未验证 → 新 commit → 新
ZIP/hash → 新确认 → 平台。默认建议至少预留两次额度给截止日前最终回归；用户可在
看到实时剩余额度后，通过当次确认明确使用保留额度。
