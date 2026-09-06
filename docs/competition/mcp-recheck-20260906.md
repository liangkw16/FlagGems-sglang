# MCP 复测（2026-09-06）

实际调用了 KernelGen：工具发现、NVIDIA/Huawei 两路任务创建与轮询均跑通，
但本轮未满足“按原契约验证、结果可核验”的验收要求。问题已超出测试计数缺失：
返回代码改变契约，故意失败的对照也出现 passed=true。

## 正常双设备请求

使用 `autotune_kernel`，两路均传入相同 fp32 参考实现、21 项 pytest 矩阵，
`max_rounds=1`、`target_speedup=0`、`verify_timeout=120`。参考计算是
`x / sqrt(sum(x.float() ** 2, dim=-1, keepdim=True) + eps)`，默认 eps=1e-6，
再转换回输入 dtype。测试覆盖三种 dtype、1D/3D/4D、非整除宽度、前导维转置和零输入。
两路请求的 reference SHA 相同：
`1909e48f81b96af16c51175e2783ac47556abbc7e96edf694855bbf9c4952146`。

| 设备 | 任务状态 | 服务 passed | 测试计数 | 服务加速比 |
| --- | --- | --- | --- | --- |
| NVIDIA | completed | true | 0 | 1.7854823914177391 |
| Huawei | completed | true | 0 | 0.07177525349981671 |

以上仅为服务报告的 benchmark，基线、测试和执行环境没有完整绑定，不能当平台分数，
也不能将 Huawei 的数值直接判为比赛无效提交。

具体不符合项：

1. 两路返回的 PyTorch reference 都被改写为 `torch.nn.functional.normalize`，并非
   原样保留传入的 fp32 公式。最终响应没有回传实际使用的测试源码、逐 case 结果或环境。
2. NVIDIA 返回入口为 `workflow_l2norm(x)`，删除了 eps；传入的三个参数化测试
   函数均调用 `workflow_l2norm_triton(x, eps=1e-6)`。这些测试不可能对该返回版本
   原样通过。不能从当前结果判断是测试被改写、未执行，还是返回代码与执行版本不同。
3. Huawei 返回入口默认 eps=1e-12，原请求为 1e-6；即使显式传 eps 的部分输入可算，
   返回版本也未保持原默认契约。

## 故意失败的负对照

另启动一个 NVIDIA 单轮任务，明确提供必定抛出
`AssertionError("KERNELGEN_HARNESS_CANARY_20260906_EXPECTED_FAILURE")` 的 pytest，
要求保留失败并报告。最终返回：

- `status=completed`、`success=true`、`verify_result.passed=true`，测试数仍为 0；
- 返回的 Triton wrapper 第一条语句就是无条件抛上述 AssertionError；
- iteration_history 中有九组 benchmark 的对应 AssertionError，speedup=null。

这证明顶层成功/通过字段不足以判断可执行正确性。该对照实际得到 benchmark 异常，
没有得到预期的明确 correctness failure；不能宣称自定义测试执行链路已被完整核实。

## 原始证据与复现

请求、原始 MCP 响应、解码结果、返回源码与本地静态核验保存在
`artifacts/competition/mcp-recheck-20260906/`。`tools.json` 是本轮实时 schema，
当前仍无固定 Triton 源码执行参数。核验时仓库 HEAD 为
`cc48bfe25c0252d080c37b67adc9f279bc9a249b`；这不是服务生成代码的提交身份。
`evidence-manifest.json` 绑定 46 份请求、响应、源码及日志文件，SHA-256 为
`6a7200c9a2d7f76729b4b55497372eae8f84bfbb1bd8f96a28a6b699ee439f99`。
更新 skill 说明后，本地门禁回归 36/36 通过；该结果不替代 MCP 设备验证。

| 场景 | job_id |
| --- | --- |
| NVIDIA 正常矩阵 | `7d303980-4766-4395-9181-60161cc43254` |
| Huawei 正常矩阵 | `431a13de-c8a6-4f82-93cc-e9ce6a140adb` |
| NVIDIA 失败对照 | `c50bd144-5a5a-44e6-bc32-7ed7f19d7848` |

```bash
python3 -B artifacts/competition/mcp-recheck-20260906/audit_results.py
```

该命令只核验已保存响应、签名、默认值和无条件异常，不上 GPU，不访问网络。
本轮没有重复 GPU 代理回归，没有平台上传或正式提交。三个 MCP 任务均已终态，
不再同条件重试；当前候选验收仍依据独立 GPU/目标芯证据与平台正式评测。
