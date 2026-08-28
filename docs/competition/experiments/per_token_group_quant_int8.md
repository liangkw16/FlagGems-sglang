# Task 33 `per_token_group_quant_int8` 实验记录

状态:S0 首次 screening **失败(17 个 subtest)**;失败明细已随诊断脚本
在远端执行,结果文件待 SSH 恢复后取回。远端 GPU 代理(kkgpu,
192.168.5.204)自 2026-08-29 05:0x CST 起 VPN 链路故障(握手挂起/
会话中断,双路由直连与 mini 跳板均不可用),阻塞修复迭代。

## 契约锁定

- 签名:`reference(x, group_size, dtype=torch.int8)`
- 输入:`x [..., K]` 任意浮点 dtype、连续;`K % group_size == 0`
- 计算:最后一维按 group_size 分组;每组
  `scale = max(|x[g]|, 1e-10) / 127`(fp32);
  `x_q[g] = clamp(x[g]/scale, -128, 127)` 后 `.to(int8)`
  (**向零截断**,非四舍五入);`x_s[g] = scale`
- 输出:`(x_q [..., K] int8, x_s x.shape[:-1]+(K//group_size,) float32)`
- 容差:fp32 1e-4 / bf16 1.5e-2 / fp16 1e-2;`x_q` 语义上需精确匹配
- 语义陷阱:reference 的 `clamp(min=1e-10)` 发生在**输入 dtype**上
  (fp16/bf16 下 ε 下溢为 0,即 reference 对全零组的 scale=0、商为
  NaN;kernel 的 ε 在 fp32 上生效)——randn 数据不触发,但属已知
  边界差异;题面容差内应不影响。
- 支持八芯;反作弊:纯 Triton,禁 fallback。

## S0(2026-08-29,未通过 screening)

- kernelgen `generate_kernel`(device=nvidia)生成;原版用 Python
  `while` 循环在其自家 42 组基准上全部编译失败(CompilationError
  27:18),按"编译类最多自查一次"协议改为 `for range` + 补非 2 次幂
  group_size 尾 mask;`tl.math.trunc` 冗余且属 libdevice(昆仑风险),
  已删除——float→int8 cast 本身向零截断。
- 结构:每 program 一组(grid = min(total_groups, 65535),组内
  grid-stride),BLOCK = next_pow2(group_size),组内 amax 归约 →
  scale → clamp → int8;逐元素两写(x_q + x_s)。
- 本地:py_compile/black/isort/flake8 全过。
- 远端 screening(gpu:/tmp/flagos-t33.BIOvd1,mode 0700):
  unittest **7 tests / 17 failures**;bench 7/7 shape 正确性 FAIL。
  已知唯一通过点:fp32 4×128 gs=128 逐位精确 → 失败面在 fp16 路径
  和/或 grid-stride(total_groups > 65535)路径,待 diag_out.txt
  (远端已执行)确认根因。
- 文件(未 commit,待修复):`src/flaggems_sglang/ops/per_token_group_quant_int8.py`
  (SHA `cd13b509…`)、`tests/test_per_token_group_quant_int8.py`
  (SHA `48cf7cf3…`)。
- 下一步:取回 `/tmp/flagos-t33.BIOvd1/{fail_list,diag_out}.txt` →
  修复 → 重跑 screening → 走 commit/ZIP/preflight/提交链。
