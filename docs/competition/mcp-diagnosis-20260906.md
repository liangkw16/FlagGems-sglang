# MCP 验证链路诊断（2026-09-06）

本轮跑通了 **KernelGen 生成 → 原样保留源码 → NVIDIA 独立验证**，27 个数值用例
通过。MCP 自身的正确性收集/结果可信度问题尚未修复，也未得到可采用的多芯加速比。
本次新增两个单轮负对照、一次 generate 调用和一次 GPU screening；没有平台提交。

## 对照结果

| 对照 | 结果 | 结论 |
| --- | --- | --- |
| pytest marker + 前缀/后缀测试入口 | completed / passed=true / tests=0；9 组 benchmark AssertionError；speedup=null | 改入口命名没有修复 |
| 原生 bench 的 label / parametrize | 同样零测试、passed=true、9 组 benchmark AssertionError、speedup=null | 改装饰器格式没有修复 |
| generate 明确公式、默认值与小幅输入反例 | 返回源码保留契约；服务端三次 attempt 后报 NameError，passed=false、tests=0 | 生成可用，此次服务验证失败 |
| 原样生成源码 + 独立 GPU harness | 27/27 数值 case，27 次 JIT launch，零失败/跳过 | 已验证该版本在 NVIDIA 的覆盖范围 |

两个负对照都要求生成正常 L2Norm，而测试必定抛出指定 AssertionError。服务再次把
断言移进返回 wrapper，其后内核代码不可达；iteration_history 的错误来自 benchmark。
这不能证明传入测试被原样执行，只能证明该顶层通过标志不足以验收正确性。

对照依据来自固定上游提交 `1e3262c2e53ec5cdeec14df8c74691f6adde6b46`：
[官方测试示例](https://github.com/flagos-ai/KernelGen/blob/1e3262c2e53ec5cdeec14df8c74691f6adde6b46/tools/tests/test_relu_accuracy.py)
使用 bench 原生装饰器；[转换脚本](https://github.com/flagos-ai/KernelGen/blob/1e3262c2e53ec5cdeec14df8c74691f6adde6b46/tools/kernelgen_to_flaggems.py)
明确将它们转换成 pytest。该线索支持做格式对照，但实测没有解决问题，不能据此规定
“必须使用 bench 才会验证”。已审查的公开代码中未找到正在运行的服务端验证器实现，
autotune 也未返回实际执行 harness；具体收集/汇总缺陷仍待服务端证据定位。

## 生成源码与独立复验

请求明确 `workflow_l2norm(x, eps=1e-6)`，以 fp32 计算
`x / sqrt(sum(x*x, dim=-1, keepdim=True) + eps)` 后转回输入 dtype；说明不得替换为
F.normalize，并给出 `x=[[1e-4,0]]` 在默认 eps 和 eps=1e-4 时的区分值。
本次返回 reference、Triton 均保住了公式与签名。生成测试混入额外的 `bench.testing`
导入，benchmark 出现五个同名定义；这些是需审查的问题，不能直接断言就是服务端
`NameError: name 'torch' is not defined` 的原因。

保留四份原始生成代码，独立 harness 复用已有 21 case 矩阵，并增加三种 dtype 各自的
默认/显式 eps 小幅输入，共 27 case。参考公式由测试独立计算，不调用生成 reference。
覆盖 fp32/fp16/bf16、1D/3D/4D、非整除宽度、前导维转置、零输入、dtype/shape 和 eps。

- Triton 源码 SHA-256：`7fa149b8f88dc20a75cb7e3677e59f5120fd75aac0ed84385c58f3d7f3ff1db4`。
- 独立测试 SHA-256：`cc6cd7e7e53850998c4e075dacc29a29034c1a74000e73841f6367d035e1922e`。
- 回执 SHA-256：`c3435e95f1c7f5db739f994214bb99eb8324070eb73b99e1249ded58a133107c`。
- 2 个 unittest 方法、27 个 subTest、29 条父/子记录；不能把 29 条说成 29 个数值用例。
- RTX 5070 Ti；torch 2.13.0+cu130、Triton 3.7.1、driver 610.57.04。
- 远端 `gpu:/tmp/flagos-mcp-diagnosis-yaVY26` 后台执行完成，日志已取回并验签。

这是未提交生成草稿的 v2 screening，source_commit=null，不能用于 release preflight；
runner 来自 `af0f12633e6d976abf65d69808acbb52a5a95cb6`。该结果不替代其他芯验证，
本次也没有在 GPU 运行性能 benchmark。

## 已写回 skill

在统一 KernelGen 入口修正全部子流程的错误分类：鉴权/传输与远端 harness 错误分开，
成功调用后的 NameError 不要求重新配置 Token。契约、源码、生成测试分别审查；原始
响应保留，独立 GPU 结果另记。竞赛 skill 记录两个格式对照未修复、严格契约生成的
成功与服务验证的失败，后续有服务变更或新定位证据再复测，不做同条件空转重试。

## 证据与复现

| 对照 | job_id |
| --- | --- |
| pytest 命名 | `a700a1e3-a684-4a6f-8a1e-ff9092393c4e` |
| 原生 bench | `d5d6c61e-42ae-4847-80e1-8793547c2e33` |

generate 为同步响应，无 job_id。请求、原始响应、四份代码、固定上游样例、GPU
输入与回执位于 `artifacts/competition/mcp-diagnosis-20260906/`。
`evidence-manifest.json` 绑定 44 个文件，SHA-256：
`b7420a3319d6f35d74bcb1065a458f0190983cba42e1fb2ae63198a0d91bb597`。

```bash
python3 -B artifacts/competition/mcp-diagnosis-20260906/audit_results.py
```

该命令只读核验已保存的矛盾结果、原样源码、GPU 文件/日志哈希与执行计数，已通过；
skill/文档 diff whitespace 检查通过。验证 runner 未改，本轮没有重复无关的门禁测试。
