# Task 57 `log_scaling_tau` 实验记录

```current
task: 57
operator: log_scaling_tau
batch: 4
validity: pending
platform: e1提交10704;7/8通过,燧原0.54675x已修复;昆仑waiting_callback
team_best_stage: -
team_best_speedup: -
sealed: no
next: 仅查10704昆仑回调,不得重复上传;最终有效性待八芯终态
updated: 2026-09-07
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
