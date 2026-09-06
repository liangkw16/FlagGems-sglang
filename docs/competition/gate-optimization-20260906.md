# 验证门槛优化与实测（2026-09-06）

门槛已分为适用路径的提交前检查、按平均收益筛选候选、平台正式资格三层。
不再要求一份 NVIDIA 回执执行全部厂商源码；全部候选字节仍必须与 Git/ZIP 一致。
单芯回退不再自动关轴，先检查 0.1 门槛风险与全部芯片的算术平均收益。

## 已落实的检查

- `verify_release.py prepare` 默认选择 generic 与 NVIDIA vendor；AMD 使用
  `--device-vendor amd`。可重复传 `--proxy-vendor` 加入兼容代理路径。
- 所选路径必须全测通过；其他源码在导入前排除并单独记账，不用 skip 冒充通过。
  代理执行成功的其他厂商代码仍保留在 `target_unverified_sources`。
- v2 回执检查完整 suite/通过方法、非空张量、非 warmup 且正 grid 的 JIT launch，
  结束时同步 GPU；不能只跑空输入或仅调用 wrapper。必要契约方法通过测试模块的
  `RELEASE_REQUIRED_TESTS` 保留，L2Norm 已接入。reference、容差与分支仍需人工审查。
- screening 使用最小淘汰集；稳定候选 commit 后只做一次完整 release。
- KernelGen 仍可生成、做多芯验证和测速。缺计数保留“服务端报告通过，覆盖未核实”，
  不单独否决具备其他合格证据的候选。本次没有改动 MCP 协议，未重复生成或设备请求。
- 相同源码、harness、环境两次同指纹失败只停止重复当前候选，定位后可用新假设继续。

## 验证结果

| 检查 | 结果 |
| --- | --- |
| 本地门禁回归 | 36/36 通过；包括不适用 vendor 不导入、适用 vendor 不能省略、空输入/无 launch/漏跑必要方法拒绝，以及哈希/一次性提交保护 |
| L2Norm GPU release | 7 个测试方法、24 条 test/subTest 记录、20 次 JIT launch，通过 |
| shrink 默认范围 | 5 个测试方法、19 条记录、generic 15 次 launch；两个目标 vendor 未执行且明确列出，通过 |
| shrink + Enflame 代理 | 5 个测试方法、23 条记录、generic 15 次和 Enflame 11 次 launch；仍标目标芯未验证，通过 |
| 真 ZIP + 真回执 + FakeClient | preflight、模拟上传/提交通过；回执篡改在上传前被拒绝；nonce 复用被拒绝；真实平台 POST 为 0 |
| 静态检查 | Black、isort、flake8、diff whitespace 检查通过 |

GPU 合计 17 个测试方法执行、66 条 test/subTest 记录、61 次 JIT launch；包含同一
shrink 矩阵的两种范围执行，不能说成 17 个不同测试。父 test 与 subTest 记录也不能
相加当作独立数值用例。GPU 为 RTX 5070 Ti，driver 610.57.04，torch 2.13.0+cu130，
Triton 3.7.1；远端目录 `gpu:/tmp/flagos-gates-z3SFJe`，后台任务已完成。

## 复现与证据

source/verification commit 均为 `0fafad041ca027d6f85ba4b6591b042662c868dc`。
本报告的 commit 是证据说明提交，未改变已验证的源码、测试或 runner。

- 证据目录：`artifacts/competition/gate-optimization-20260906/`。
- `release/run.sh` 保存三个场景的完整串行 GPU 命令；每份目录有输入 manifest、原始
  回执和完整日志。重新运行必须生成新目录，不能覆盖原回执。
- `evidence-manifest.json` 逐项绑定源码、runner、日志、回执、ZIP 清单和模拟流程脚本；
  SHA-256：`5771c92000d567683be55a879effe982a17f6689efcd31dd1e47dfccca48f1e2`。
- ZIP SHA-256：`c9032f096d7811dd375547989c2157d4420f7649d6f64f5589a58581ecbcda9a`。

```bash
python3 -B -m unittest discover -s .agents/skills/flagos-operator-race/tests -v
python3 -B artifacts/competition/gate-optimization-20260906/check_flow.py
```

本 runner 目前验证 CUDA/HIP 的 Triton JIT 路径；其他 runtime 启动协议需适配，
目标芯独立证据通过 KernelGen/授权主机保留在账本，不伪装为 NVIDIA 回执。
旧 v1 回执保留为历史材料；新 preflight 要求 v2，须重新执行，不能给旧 JSON 补字段。
本次无正式平台提交，无新平台分数，也未确认 Enflame 目标硬件正确性。
