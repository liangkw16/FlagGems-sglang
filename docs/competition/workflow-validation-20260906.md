# Skill 流程实测（2026-09-06）

GPU commit 回归、执行回执、不可变 ZIP 与发布门禁已跑通。KernelGen 的生成、多设备请求和异步任务轮询可用；本轮四份服务结果均为 passed=true、total_tests=0，尚不能确认目标芯正确性。返回源码已另在 NVIDIA 代理做实际数值回归。

## 修改与验证

- `flagos-operator-race` 与 `kernelgen-flagos` 已写明 KernelGen 生成/多 GPU/多芯验证及 GPU 主机回归的分工；optimize 不验证，generate/autotune 的结果单独核验。
- 修复 T56 展平行 stride；补 T48 长段边界、T50 全部 vendor、11 个测试文件的 unittest 入口顺序。
- preflight 和 submit 都检查执行回执：source/test/依赖/runner Git 字节、用例、源码入口调用、退出码和日志。
- 实测发现并修复 PyTorch 虚拟 `_ops.py` 被误认为真实依赖的问题；新增回归同时确认真实未绑定文件仍被拒绝。

| 阶段 | 结果 | 证据范围 |
| --- | --- | --- |
| 本地流程测试 | 34/34 通过 | 含坏哈希、零测试、skip/xfail、漏测 vendor、一次性提交与虚拟模块回归 |
| GPU release | 13 算子、96 tests、615 条 test/subTest 记录，通过 | RTX 5070 Ti 的实际执行，不能替代其他芯 |
| KernelGen generate | NVIDIA/Huawei 两次完成并返回源码 | 两份 verify_result 的测试数均为 0；不作正确性通过 |
| KernelGen autotune | 同一 reference 与 21 项 pytest 矩阵，两路单轮任务完成 | 两份终态测试数仍为 0；不空转重试 |
| 返回源码 GPU 复验 | 两份源码各 21 项，合计 42 项通过 | 原样返回字节；screening；NVIDIA 数学代理 |
| ZIP 与回执门禁 | 正常通过、篡改拒绝、重复提交拒绝均通过 | ZIP/Git/回执检查是真实的；网络传输为 FakeClient，真实平台 POST=0 |

`test/subTest` 记录包含父 test，并非 615 个互相独立用例。所有 release 回执 exit_code=0，无 skip/xfail，且逐源码入口调用数均大于 0。

## 逐算子执行

| 算子 | tests | test/subTest 记录 | 已调用源码份数 |
| --- | ---: | ---: | ---: |
| act_and_mul | 13 | 181 | 7 |
| causal_conv1d_update | 9 | 48 | 4 |
| chunk_scaled_dot_kkt | 9 | 43 | 4 |
| chunked_embedding_lora_a | 11 | 44 | 7 |
| chunked_sgmv_expand | 10 | 35 | 3 |
| chunked_sgmv_shrink | 5 | 27 | 3 |
| ernie45_rope_fused | 5 | 21 | 3 |
| extend_attention | 4 | 74 | 4 |
| fla_layernorm_gated | 6 | 41 | 4 |
| fused_gdn_gating | 5 | 20 | 3 |
| fused_norm_rope_stacked | 7 | 34 | 4 |
| l2norm | 7 | 24 | 1 |
| w8a8_block_int8_matmul | 5 | 23 | 2 |

T48 新增长段 `[63,64,65,256]`；T50 覆盖 generic/Ascend/Enflame/Kunlun；T56 覆盖三种 dtype、1D/3D/4D 和前导维转置。vendor 文件在 NVIDIA 上执行仅证明该运行路径，不能算目标芯 lowering 验证。

## 复现与身份

- source commit：`f83c73aeceaa6382c6ec6d93a082fc24e37090fb`
- verification commit：`73f6bb801607d2ba77ca8469136e568a43ccf262`
- 账本 commit：以本报告的 `git log -1 -- <path>` 为准。
- GPU：RTX 5070 Ti，driver 610.57.04，Python 3.12.13，torch 2.13.0+cu130，Triton 3.7.1。
- 最终远端目录：`gpu:/tmp/flagos-workflow-final-2mfudtkv`；后台 PID/PGID 302561，已完成。
- 本地保留目录：`artifacts/competition/workflow-fix-20260906/`，所有日志/源码/原始响应保留。
- 证据清单 SHA-256：`a3244361f2d0f7a87a2c84f9444ef308028757250fa0c613ba24579480681aac`；清单逐项绑定回执、完整日志、请求/响应和执行脚本。
- ZIP SHA-256：`c9032f096d7811dd375547989c2157d4420f7649d6f64f5589a58581ecbcda9a`。

```bash
python3 -m unittest discover -s .agents/skills/flagos-operator-race/tests -v
python3 artifacts/competition/workflow-fix-20260906/check_flow.py
```

第二条复验已有真实 GPU 回执与 Git 字节，并完整走本地模拟发布流程。`check_flow.py` 不实例化 HTTP 客户端，只使用测试中的 FakeClient。测试 intent 放在临时目录，不进入真实平台提交状态目录。

重新执行 GPU 任务时，用 `verify_release.py prepare` 指定上述 source/verification commit，生成新的独立目录；按 `release-final/run.sh` 的串行命令运行，不能覆盖原回执。源码、测试和 runner 字节变化时重新取证。

KernelGen 的完整请求和原始响应位于 `kernelgen/`；异步 job：NVIDIA `f58cd68d-f753-4139-9085-ead9789cf7e1`，Huawei `0c23db7f-5a3e-4ace-a20f-b987307cfe08`。生成参考采用 [Triton 官方逐行归约示例](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html) 的展平行与行步长方式，算子契约仍是 L2 sum-of-squares。

本轮没有正式平台提交或新性能成绩；原任务历史分数不变。KernelGen 目标芯结果需服务提供实际执行用例/环境证据后再升级，GPU 代理通过不能填补该证据缺口。
