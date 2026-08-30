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

## S0 定稿与提交(2026-08-29 13:5x CST)

### 根因与修复(远端网络故障窗口期间完成)

- 首轮 screening 17 失败的根因(诊断数据实证):**Triton 的普通 `/`
  除法 lowering 为近似除,而 torch 是 IEEE 舍入除**;在每组 amax 边界
  元素(x/scale 数学上恰为 ±127)上差 1 ulp → 截断后 ±1。
  `diag2` 三变体实验:`tl.math.div_rn` 6/6 配置逐位匹配,
  普通 `/` 与显式倒数均失配。
- 次生问题:测试初版用 CPU 参照,CPU torch 的除法舍入与设备端不同,
  修正为**设备端参照**(平台语义)。
- autotune_kernel 实机参照(其 verify 计数存疑,total_tests=0)只作
  结构参考;wrapper `.to(torch.float32)` 方案弃用,保留 in-kernel
  `.to(tl.float32)`(T29 平台八芯先例)。

### 构建身份

| 项目 | 值 |
| --- | --- |
| source/verification commit | `a89a9ad` |
| `per_token_group_quant_int8.py` SHA-256 | `017f993a5bb0c9e79f7365152266067ba150a968632b14b1e4819b9e7ecd02dc` |
| 测试 SHA-256 | `23941f2c3358335440883ecbb3484a1ee50cde8ee01402a48ec7de6f889a16d4` |
| ZIP | `artifacts/competition/per_token_group_quant_int8/s0-a89a9ad/per_token_group_quant_int8.zip` |
| ZIP SHA-256 | `f4705c6adf9f6ab56b1c74254a6c62434bed088282779f727efbb234a77eaf81` |
| 成员 | 单文件 `per_token_group_quant_int8.py` |
| screening 目录 | `gpu:/tmp/flagos-t33.BIOvd1` |
| release 目录 | `gpu:/tmp/flagos-t33-rel.abjliD`(Git 对象,哈希逐项一致) |

### 唯一候选配置

每 program 一组(grid = min(total_groups, 65535) + 组内 grid-stride
for 循环);BLOCK = next_pow2(group_size) + 尾 mask;组内 amax 归约
(fp32)→ scale = max(amax, 1e-10)/127 → **`tl.math.div_rn` IEEE 除**
→ clamp ±128/127 → float→int8 截断向零;in-kernel fp32 转换。

### 验证

- release(Git 对象):py_compile ✓、unittest **7/7 OK**、bench 7/7
  正确性(设备端参照精确相等);
- 代理加速比(wrapper-inclusive p50):fp16 3.0–3.9x、bf16 1.8–2.6x、
  fp32 4.96x;大 shape 65536×256 3.31x。

### 提交计划

preflight tuple:season 2、race `782kzq4m`、account `15600308080`、
team `SoulCoder`、batch 3、task 33、tid 待 preflight 确认(按序号推断
`s2t1op033`)、operator `per_token_group_quant_int8`、stage `s0`、
commit `a89a9ad`、ZIP SHA 见上。门禁全过即自动单次提交。

### 跨芯风险

- `tl.math.div_rn` 属 libdevice 调用(昆仑曾对 libdevice erf 崩溃;
  但 div.rn 是基础运算,预期可用;若昆仑失败,备选:fp64 提升除法或
  __fdivrn 语义替换);
- tl.max 轴归约:燧原在 T25 对 argmax 归约失败,但 T26/T27 的
  axis-1 max 归约通过,纯 max 无索引归约风险较低。

### S0 提交记录(2026-08-29 14:0x CST)

preflight 全过(tid `s2t1op033`、额度 17/30,本次消耗 1);单次
confirm 提交成功,评测入队;逐芯结果待回填。

### S0 平台终态(sub 6333,2026-08-29 14:1x CST)

**8/8 全过,valid,平均 3.53853333x —— 一次通过。**

| 芯片 | speedup | 备注 |
| --- | ---: | --- |
| tianshu | 6.796 | |
| card_b | 6.825 | |
| haiguang | 5.759 | |
| card_a | 4.646 | |
| muxi | 2.285 | |
| huawei | 1.316 | T29 同款中等 |
| enflame | 0.448 | 弱芯(过 0.1x 门槛) |
| kunlunxin | 0.233 | `tl.math.div_rn` 通过(未触发昆仑 libdevice 崩溃) |

- 距榜首(MakeYUNAGreatAgain 4.0170x)**-11.9%**,首投即第 3 档
  (2 队达标 → 我方第 3 家 valid)。
- 后续单变量轴:燧原/昆仑 vendor(T29 已证两芯偏好差异)、BLOCK
  调优、多组并行(每 program 多组摊销归约)。

## E1:燧原/华为多组摊销 vendor(2026-08-29 16:0x CST)

- 假设:S0 燧原 0.448x/华为 1.316x 的短板是海量小 program
  (total_groups 可达 26 万,每 program 仅 ≤256 元素)的调度开销;
  两芯对结构摊销响应好(T29 E5 先例)。
- vendor = 每 program 8 组 static_range 摊销、无分支 mask 尾
  (commit `447a3f8`,两 vendor 字节一致,SHA `e58cfbc0…`);
  generic/kunlunxin 等六芯维持 S0。
- 代理(NVIDIA)上 E1b:bf16 中 shape +7%、小 shape -46% → 不换
  generic,仅 vendor;release 验证(gpu:/tmp/flagos-t33-e1.vIC17W,
  Git 对象):py_compile ✓、unittest 7/7 OK、vendor 代理基准正常。
- ZIP `e1-447a3f8`,SHA `00158207d7e269286fe21bc5344499ce206c257822348d3bb57fb556a0293a5a`,3 成员
  (generic + enflame + ascend vendor)。

### E1 提交记录(2026-08-29 16:1x CST)

preflight 全过(额度 16/30,消耗 1 → 15/30);单次 confirm 提交,
评测入队,逐芯结果待回填。

### E1 平台终态(sub 6340,2026-08-29 16:2x CST)

8/8 过,avg 3.5138(低于 S0 3.5385,**团队最佳保持 S0**):
燧原 0.448→**0.384(-14%,假设证伪)**;华为 1.316→1.307(vendor
生效但持平);其余六芯 generic 未动,波动 ≤2% 为平台噪声。

- 结论:燧原短板不是 program 调度开销,疑在 `tl.max` 归约或
  `div_rn`(libdevice)的燧原 lowering 成本;多组摊销轴对燧原/华为
  均已证伪,封闭。除换语义(破坏逐位一致)外无已知单变量可解,
  燧原 0.45x 附近视为本题固有水位。
- 额度 15/30。T33 收敛:S0 = 团队最佳 3.5385x(榜首 4.0170x,
  -11.9%)。

### E2:燧原 2D 组瓦片 vendor(2026-08-31 17:2x CST)

- 假设:E1 失败的是**顺序** static_range 摊销;e2 改真 2D 瓦片
  [4, GROUP_SIZE] 并行 axis-1 amax——T34 证明的燧原"大工作量宽
  瓦片"偏好结构类;
- 修复循环上界 bug(总瓦片数,尾瓦片由 65536 行测试抓获);
- unittest 7/7(gpu:/tmp/t33e2);commit `0233e47`,ZIP `dd6d43418a75452e2b8d1f912f80bad55f041d3546c165937b1a3955258b3044`。

### E2 修正与提交(2026-08-31 17:4x CST)

- 修正记录:误把 T29 的 metax vendor 记忆套到 T33——T33 团队最佳
  即 S0 单文件;中途两次 ZIP 组合错误(多 ascend/缺比对)已废弃
  未提交,最终 e2 = S0 generic(逐字节一致)+ 仅新燧原 2D 瓦片
  vendor(真单变量);
- commit `71c1441`,ZIP SHA `432ba4a7…`,2 成员;preflight 全过
  (额度 2/30 消耗 1 → 1/30),单次提交,昆仑/燧原终态待回填。

### E2 平台终态(sub 6951,2026-08-31 18:0x CST)

**8/8 valid,平均 4.2580x —— 新团队最佳(S0 3.5385 → +20.3%)。**

| 芯片 | S0 | E2 | 变化 |
| --- | ---: | ---: | ---: |
| **enflame** | 0.448 | **6.284** | **14 倍(2D 瓦片假设命中)** |
| tianshu | 6.796 | 7.006 | +3% |
| muxi | 2.285 | 2.308 | 持平 |
| haiguang | 5.759 | 5.567 | -3% |
| kunlunxin | 0.233 | 0.232 | 持平 |
| huawei | 1.316 | 1.296 | 持平 |
| card_a | 4.646 | 4.673 | 持平 |
| card_b | 6.825 | 6.697 | -2% |

- 跨题知识第三证:燧原"宽 2D 瓦片并行归约">>>顺序 static_range
  摊销(T33 E1 -14% vs E2 +14 倍);与 T34 整行大组互证。
- 距榜首(xuanzhengdu 6.2386)收窄至 -31.8%。
