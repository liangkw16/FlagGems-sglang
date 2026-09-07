# Task 57 `log_scaling_tau` 实验记录

```current
task: 57
operator: log_scaling_tau
batch: 4
validity: valid
platform: 8/8(e2,10747,2.36478125x首次有效)
team_best_stage: e2
team_best_speedup: 2.36478125
sealed: no
next: 保留e2有效2.36478125排名5;warp/BLOCK无收益;华为MCP零测试且错误rank1语义,不晋级
updated: 2026-09-07
team_best_commit: 493b4956a0a39077e0db047bd64e0f1dd12a4c5b
```

## S0 2D grid 行缩放（2026-09-06，远端 GPU 2/2 OK）

- 形态：grid `(rows, cdiv(n_cols, BLOCK))`，BLOCK=min(next_pow2(n_cols),1024)；
  fp32 乘 tau 后 cast 回 x.dtype；任意尾部维度按 contiguous 展平成行。
- 远端 5070 Ti：2/2 OK（平台 5 shape + fp32 附加）。

## S0 平台进行中（2026-09-06，seq25）

- 已过 7：天数 3.97 / 沐曦 2.71 / 海光 4.70 / 昆仑 0.53 /
  华为 0.47 / A 3.32 / B 2.61；燧原评测中。

## E1 连续输出、tau stride 与一维调度（2026-09-07）

- source/verification commit: `ee551b50b7405e6abe315b61fc54a72501607242`；ledger commit 为本节所属提交。
- 旧源码最小回归：转置 x `[2,3]` + tau[::2] 在 fp16/bf16/fp32 三例全部失败；输出分配在 contiguous 之前、tau 忽略 stride 是直接根因。先转连续再分配输出，tau 使用实际 stride；2D grid 合并为 1D，去掉长尾 grid.y 限制。核心仍 fp32 Triton 乘法。
- `py_compile`、Black、isort、flake8 通过。NVIDIA release 4 方法、18 次真实 kernel launch，零 skip/xfail/errors；覆盖公开形状、三 dtype、转置、strided tau、空 batch、rank1/3、63/64/65 与 1023/1024/1025，最大 `[3,262144]`。
- 远端 `gpu:/tmp/flagos-b4-invalid.eYQz0q/t57-release`，RTX5070Ti / torch2.13.0+cu130 / triton3.7.1。执行：`timeout 420 /home/kevin/notebook/.venv/bin/python .agents/skills/flagos-operator-race/scripts/verify_release.py run --directory /tmp/flagos-b4-invalid.eYQz0q/t57-release`。
- 目标燧原 MCP job `80e51c7b-6a38-43c7-9286-a144b6651d8a`：服务报告 passed、tests=0，独立生成 v1 报告 speedup=0.7266719132865975；没有本候选源绑定，分类 mcp-unbound-observation，target-runtime-unverified，不作为本地源码正确性或平台分数。
- 晋级：八芯正确且各≥0.1，均值取平台全芯算术平均。止损：同源同指纹两次失败停止，禁止同字节重投；本轮最多一次正式提交。
- ZIP `/Users/bytedance/ccc/flagos/artifacts/competition/log_scaling_tau/e1-ee551b5/log_scaling_tau.zip`，SHA256 `ab86581d0c0b9291267efd76aea4d379716d9cff3d2562f8cdb173b781814fc3`，2160 bytes；成员 `log_scaling_tau.py` SHA256 `c32491e85b093964cc686128640ed23a675e743e027e34109735b3ffbe3345ed`。dry-run 与最终 manifest 逐字段一致。
- 证据 `/Users/bytedance/ccc/flagos/artifacts/competition/log_scaling_tau/e1-ee551b5/validation/verification.json` SHA256 `841755a208d39404370a159d83ba9796fd4d3c57a3e3b3f53c1a9fc7168a7c88`。
- 证据 `/Users/bytedance/ccc/flagos/artifacts/competition/log_scaling_tau/e1-ee551b5/validation/verification.log` SHA256 `23ca7e800ea84668ead3fa1913fdac6bd1901f7ed7906907dca5d44b1d544009`。
- 证据 `/Users/bytedance/ccc/flagos/artifacts/competition/log_scaling_tau/e1-ee551b5/validation/verification-input.json` SHA256 `3692548d719de5153e717c70ed002f2c25ca98cf791bf41032d66b0a0411a36d`。
- 证据 `/Users/bytedance/ccc/flagos/artifacts/competition/log_scaling_tau/e1-ee551b5/validation/57-kg-request.json` SHA256 `7ec8fd6c6520d31aba2229b911370f2f5d0bff5bb261e4475cfeaa79571d2bff`。
- 证据 `/Users/bytedance/ccc/flagos/artifacts/competition/log_scaling_tau/e1-ee551b5/validation/57-kg-response2.json` SHA256 `21c4c295eabd465a7a4807a33ba76df209b743e035cb3305ff3ad6ffcdd76d0c`。
- 提交前实时快照：2026-09-07T12:08:53+08:00，s0 submission10410，daily quota24/30；实际提交结果另追加。

## E1 平台结果（2026-09-07T12:29:32.275961+08:00）

- submission `10704`，daily_seq `7`，创建 `2026-09-07T12:20:45`；正式上传/提交各一次，远端ZIP SHA验签 `verified`。
- 状态 `pending`，平台average_speedup `None`；quota `21/30`（本条观测时）。

| 芯片 | 正确性/状态 | 加速比 | 实际成员 |
|---|---|---|---|
| tianshu | True / completed | 4.0595 | log_scaling_tau.py |
| muxi | True / completed | 2.683 | log_scaling_tau.py |
| enflame | True / completed | 0.54675 | log_scaling_tau.py |
| haiguang | True / completed | 4.7875 | log_scaling_tau.py |
| kunlunxin | None / waiting_callback | None | log_scaling_tau.py |
| huawei | True / completed | 0.473 | log_scaling_tau.py |
| card_a | True / completed | 3.19225 | log_scaling_tau.py |
| card_b | True / completed | 2.63825 | log_scaling_tau.py |

- 燧原已完成正确性与性能0.54675x；昆仑缺回调，不按旧S0昆仑通过记录补造本次结果，不重投。
- 证据 `/Users/bytedance/ccc/flagos/artifacts/competition/log_scaling_tau/e1-ee551b5/validation/57-submit.json` SHA256 `fe64330b996e229eeccd56c2afe0d72ae69acbadf3ac05bd7e2f2b2aed0c9a1b`。
- 证据 `/Users/bytedance/ccc/flagos/artifacts/competition/log_scaling_tau/e1-ee551b5/validation/57-status-now.json` SHA256 `48f414bfd509b201f15e526d372601c7d6c48f59b819a032fb06b1f7f7322522`。

## E2 Top1 冲刺候选（2026-09-07，提交前）

E1/submission10704 已于2026-09-07 12:40:55终态 invalid_correctness、7/8；昆仑 case0 首次比较报 runtime error299 / wait for noc idle timeout，后续参考侧报错可能受设备状态污染，不能认定为纯平台参考 bug。

新增 canonical `_kunlunxin` vendor：恢复 S0 在该芯通过的二维 `(rows, col_blocks)` 调度，去掉 E1 flat pid 的商/余数；保留 E1 的连续输出分配、tau 实际 stride、64位行偏移修复。generic 和其他芯仍走 E1，一次只隔离昆仑调度。属于新字节候选，非同 ZIP 重投。

正式测试改为遍历 selected variants，包含转置、strided tau、空输入、rank1/3、63/64/65与1023/1024/1025、最大262144列。昆仑 KernelGen 本次返回502，目标运行时未验证。

预注册：首投一次争取八芯有效。按 E1 七芯合计18.38025，加回 S0 昆仑0.53125仅为2.36394，仍低于榜首2.81646875约19.1%；过线不等于 Top1，后续需高分芯整体提速，不能继续只微调弱芯。

- source/verification commit：`493b4956a0a39077e0db047bd64e0f1dd12a4c5b`；ledger commit 为本节所属提交。
- 本地 py_compile、Black、isort、flake8 通过。NVIDIA release：4 方法、36 次 kernel launch，fail/error/skip/xfail 均为0。执行源：`src/flaggems_sglang/ops/log_scaling_tau.py`, `src/flaggems_sglang/runtime/backend/_kunlunxin/ops/log_scaling_tau.py`。
- 测试源码 SHA256：`17f953d64835eca778b3cc71144a716dfe009a249c7527c6f94e2d921719f8b7`；各输入文件 SHA 见 verification-input.json。
- 远端 `gpu:/tmp/flagos-b4-top1.8nsvBC/t57-release`，RTX5070Ti / driver610.57.04 / Python3.12.13 / torch2.13.0+cu130 / triton3.7.1。串行后台执行：`timeout 600 /home/kevin/notebook/.venv/bin/python .agents/skills/flagos-operator-race/scripts/verify_release.py run --directory /tmp/flagos-b4-top1.8nsvBC/t57-release`。
- ZIP `/Users/bytedance/ccc/flagos/artifacts/competition/log_scaling_tau/e2-493b495/log_scaling_tau.zip`，4228 bytes，SHA256 `1b0f3648dbaf3e7ae79dee4ad35bc3dd0d1aad5667ef5b18d0131ec0afe25da4`；dry-run 与最终 manifest 五项恒等字段全部匹配。
- ZIP member `log_scaling_tau.py` SHA256 `c32491e85b093964cc686128640ed23a675e743e027e34109735b3ffbe3345ed`。
- ZIP member `log_scaling_tau_kunlunxin.py` SHA256 `35d87eaa4d95fd12ba46c97632e8fedbee31e47f81ee5d9f8afa233960a23688`。
- 证据 `/Users/bytedance/ccc/flagos/artifacts/competition/log_scaling_tau/e2-493b495/validation/verification.json` SHA256 `b875d015e670608cd218f019070bcca8f165471d56a861d9c28f3e5c0b351e4f`。
- 证据 `/Users/bytedance/ccc/flagos/artifacts/competition/log_scaling_tau/e2-493b495/validation/verification.log` SHA256 `7c67e7e2abcf674c522a751fd6ea73a2b275516b6fcb2125d99efdcf95520307`。
- 证据 `/Users/bytedance/ccc/flagos/artifacts/competition/log_scaling_tau/e2-493b495/validation/verification-input.json` SHA256 `d6404c5ca4a1fa5259c622780fc70b5c622ce5653447eb1093ab9634aab93a62`。
- 附加基准、诊断及 MCP 证据清单 `/Users/bytedance/ccc/flagos/artifacts/competition/log_scaling_tau/e2-493b495/validation/evidence-sha256.json` SHA256 `a4abbded27664b0fea47346d7e48c40a54b98062e8bb7d608eff7d3cdad08d73`。
- 首轮每题最多1次正式上传/提交；本次预算上限沿用批准的 T43/T51/T52 各4、T57 3、储备6，须有新证据才继续消耗。sending/uncertain/stale_after_upload 不自动重试。

## 本轮平台结果与止损（2026-09-07T14:55:36+08:00）

- submission `10747` / daily_seq `13` / created `2026-09-07T14:51:08`；preflight与上传/提交均只执行一次，远端ZIP验签 `verified`。
- 平台原始状态 `completed` / `valid`，通过8/8、终态8/8；average_speedup `2.36478125`，is_team_best `True`；观测时额度 `17/30`。
- 昆仑二维调度恢复正确性0.5345x，E2首次八芯有效。相对当时榜首2.81646875仍需+19.10%，未获Top1。额外9组warp/block启动配置×6shape×五轮交错A/B无稳定收益，单个大shape4warp/BLOCK256降至0.706x；不生成第三个提交候选。

| 芯片 | 状态/正确性 | 加速比 | 实际文件 |
| --- | --- | ---: | --- |
| tianshu | completed/True | 4.03225 | `log_scaling_tau.py` |
| muxi | completed/True | 2.72075 | `log_scaling_tau.py` |
| enflame | completed/True | 0.56775 | `log_scaling_tau.py` |
| haiguang | completed/True | 4.77725 | `log_scaling_tau.py` |
| kunlunxin | completed/True | 0.5345 | `log_scaling_tau_kunlunxin.py` |
| huawei | completed/True | 0.39675 | `log_scaling_tau.py` |
| card_a | completed/True | 3.22175 | `log_scaling_tau.py` |
| card_b | completed/True | 2.66725 | `log_scaling_tau.py` |

- 证据 `/Users/bytedance/ccc/flagos/artifacts/competition/log_scaling_tau/e2-493b495/validation/57-submit.json` SHA256 `403d1f1e7ed59a847896696067e50cb0b4077fd8c8d63cc031b367b23fe4af09`。

- 证据 `/Users/bytedance/ccc/flagos/artifacts/competition/log_scaling_tau/e2-493b495/validation/57-status-first-round.json` SHA256 `a8cacdcdb6f0776ba2186c721abe000c8e07e22ec8f70b6aeb97daec7f7f4b28`。
- 未晋级后续实验 `/Users/bytedance/ccc/flagos/artifacts/competition/log_scaling_tau/e2-493b495/validation/followup-evidence-sha256.json` SHA256 `c051f10ea5dd950303bbf54c7bcce8e99e36e5e1f0f135dc352637b460c4ed5f`。

## 华为探索终态（2026-09-07 15:00）

- KernelGen job `50ad0f87-ea2a-46d6-bc64-3f89b158e917` 两轮完成；服务自报passed，但两轮test count均0，speedup分别0.0065838352250970665、0.006335368592540647。分类`mcp-unbound-observation`，不证明本候选执行或目标正确性。
- 返回源码在rank1分支使用`x.unsqueeze(0)`，把原来T行单列误作1行T列并仅应用tau[0]；与题目逐行缩放契约冲突，静态审计即拒绝。原参考`[T]`需逐元素使用tau。该生成源码只留证，未写入算子、未发布ZIP。
- row内串行column blocks也没有优于当前载体的证据。第二轮之后不再调用，当前无运行中的MCP任务；本轮不追加平台提交。
- 证据清单 `/Users/bytedance/ccc/flagos/artifacts/competition/log_scaling_tau/e2-493b495/validation/57-huawei-evidence-sha256.json` SHA256 `80cd14404e341a1c6398a69be87ccdde1434b6767a6aed07743713cc06b2735b`。
