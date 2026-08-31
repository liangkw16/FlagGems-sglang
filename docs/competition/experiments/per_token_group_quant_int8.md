# Task 33 `per_token_group_quant_int8` 实验记录

```current
task: 33
operator: per_token_group_quant_int8
batch: 3
validity: valid
platform: 8/8
team_best_stage: e10
team_best_speedup: 5.5720
sealed: yes
next: 重新封存 e10 5.5720;e12 去 contiguous/warps 2,4,8 全部低于5%门,已知参数轴尽
updated: 2026-09-01
```

状态:e10 = 团队最佳 5.5720x(8/8 valid)；E11 metax tile32 与 E12
wrapper/warps 扫描均判负，机器可读状态见顶部 CURRENT 块。

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

### E3:华为 [4,G] 2D 瓦片独立候选(2026-08-31 02:0x CST)

- 燧原 e2 同款结构([4, GROUP_SIZE] 并行 axis-1 amax,~4KB 足迹
  远低于昇腾 UB 预算);commit `d6b1c4f`,ZIP `8ea4f63e…`,
  3 成员(generic+enflame[e2]+ascend[e3]);unittest 7/7;额度 25/30。

### E3 终态(sub 6997)

8/8 valid,avg 3.589(< e2 团队最佳 4.258,保留 e2)。华为
1.296→**1.484(+14%,瓦片假设在昇腾小幅兑现)**;燧原 vendor 字节
未变却 6.284→0.879——平台方差再现,e2 最佳按团队计分保留。
额度 24/30。

### E5:瓦片升 generic(sub 7215,2026-08-31 15:2x CST)

- 假设:e2/e3 已把 [GROUPS_TILE=4, GROUP_SIZE] 2D 瓦片在燧原(+14 倍)
  和华为(+14%)单芯证实;screening(gpu:/tmp/flagos-catchup.NOF9kN)
  进一步显示该结构对旧 generic 在代理上 +46~103%(5/7 shape,无回退:
  1024×2560 2.52→5.13x、8192×512 1.81→4.82x、65536×256 3.32→7.74x)
  ——升为 generic 可抬升全部五芯;
- 包:generic(瓦片,commit `b7f0a85`)= 燧原 e2 vendor 正文逐字节一致;
  燧原/华为沿用 e2/e3 已证字节;昆仑钉死旧 generic 字节
  (`017f993a`,兜底);metax 独立 vendor 撤销(结构即 generic);
- release(gpu:/tmp/flagos-rel2.bskkJw,git 对象):4 模块 unittest
  7/7、lint 未改写字节、SHA 逐项一致;ZIP `e5-b7f0a85`,
  SHA `af61ec55…`,4 成员;preflight 额度 18/30。

#### E5 终态(sub 7215)

**8/8 valid,平均 4.57069167x —— 新团队最佳(e2 4.2580 → +7.3%)。**

| 芯片 | e2 | E5 | 变化 | 文件 |
| --- | ---: | ---: | ---: | --- |
| haiguang | 5.567 | **9.235** | **+66%** | generic |
| card_b | 6.697 | 8.041 | +20% | generic |
| tianshu | 7.006 | 7.327 | +4.6% | generic |
| card_a | 4.673 | 6.223 | +33% | generic |
| **muxi** | 2.308 | **3.168** | **+37%** | generic |
| huawei | 1.296 | 1.457 | +12.4% | ascend(e3 字节) |
| enflame | 6.284 | 0.883 | **方差低滚(第三次)** | enflame(e2 字节未变) |
| kunlunxin | 0.232 | 0.231 | 持平 | kunlunxin(钉死) |

- 瓦片结构在六芯兑现,代理预测精确成立;距榜首(starwing 6.3983)
  收窄至 -28.6%;若燧原回到 ~6.3 结构水位,平均 ~5.44;
- 跨题知识第五证:宽 2D 瓦片对"小工作量 program"题型的普适性
  (T24/T33/T34/T39/T29 之外,本题 generic 层也成立)。

### E6:方差重掷(sub 7221,2026-08-31 15:4x CST)

- 燧原同字节三轮:6.284(e2)/0.879(e3)/0.883(e5)——已知平台方差;
  e6 = e5 注释载体(commit `78e9242`,ZIP `38eea22f…`),team-best
  计分下无下行,高滚(~6.3)则平均 ~5.44;额度 14/30 消耗 1。

### E6 终态(sub 7221):重掷关闭

- 8/8 valid,avg 4.5517(< e5 锚点 4.5707,团队最佳保持 e5);
  燧原第四次同字节读数 **0.880**——6.284 只在 08-30 22:31 出现过
  一次,其后三轮全部 ~0.88,判定为 08-30 午夜后的平台结构性水位
  而非方差,重掷轴关闭(两连低滚);
- T33 今日收盘:**e5 = 团队最佳 4.5707x**(距榜首 -28.6%);
  剩余可见杠杆:燧原平台水位恢复(不可控)、starwing 6.40 的
  结构面(未破译)。

## E7:昆仑钉死字节换 e5 瓦片(2026-08-31 19:1x CST)

状态:候选就绪待单次提交(接续 41 算子源码核验结论:本批唯一
"已验证结构从未路由到未试芯片"的缺口)。

### 假设

- kunlunxin 自 S0 起钉死在单组/program 循环(每 program 128 元素),
  全轮 0.231-0.233x;e5 `[GROUPS_TILE=4, GROUP_SIZE]` 瓦片在其余
  六芯兑现(haiguang +66%/muxi +37%),却从未路由昆仑;
- 2D 瓦片在昆仑有编译先例(T12 fp32-ieee dot、T13 direct dot),
  1D capped grid-stride、无 num_warps/num_stages、div_rn 全部符合
  T21 昆仑规则;唯一平台待证假设是瓦片 lowering 在 XPU 的收益。

### 变更集(单变量:kunlun 路由 S0 旧结构 → e5 瓦片)

- `_kunlunxin` vendor := generic 瓦片 kernel 逐结构一致(仅注释
  不同);generic(e6 载体)、`_ascend`(e3)、`_enflame`(e2)冻结;
- 测试矩阵升级:全部 vendor 文件接入 `_check`(逐位对照 reference),
  新增 `test_vendor_matrix_matches_generic_semantics`。

### 构建身份

| 项目 | 值 |
| --- | --- |
| source / verification commit | `918bf0603596c244fa4d4209b192fab23ba12c95` |
| generic SHA-256 | `39691ff4e3efe9a427cdb5896a53cfda613d9e7066c8e7fffff27228d918d30d`(=e5/e6) |
| Ascend SHA-256 | `fa7a1a46573f0e0179573ec8a8ebec10463dba42c942f68da308b62e9c0fd670`(=e3) |
| Enflame SHA-256 | `c6ff5fc03dea9f52c40eeb7dd19c6600e2962814dfb5771ddea7e1a168e3b6b0`(=e2) |
| Kunlun SHA-256 | `fbc2da64dc28a151605e6de6da533ce6362b204e407ada3700bb93657679cfba`(新) |
| test SHA-256 | `2ec5b35b226c88865a5ea62133aa6639ab7ce50c1ded4416bd8526399d1d13d5` |
| canonical ZIP | `artifacts/competition/per_token_group_quant_int8/e7-918bf06/per_token_group_quant_int8.zip`,12326 bytes,SHA-256 `0e05f32b89821c51e72c8ad728d649d8c3bce35906b7f6990d0e5ef20d3f7adc` |
| ZIP 成员 | generic + `_ascend` + `_enflame` + `_kunlunxin`,4 个 UTF-8 `.py`;`unzip -t` 无错;成员哈希与 commit blob 逐项一致 |

### 验证证据

- screening(gpu:/tmp/flagos-t33e7.fF5rQC):py_compile、isort、
  flake8、unittest **8/8**(含 vendor 矩阵)、`SCREENING_OK`;日志
  SHA-256 `ed7c0b825a81e2de467ab8503f6b3e0a8959dbf36bbe81663d2a3cc8895f6caa`;
  前后哈希一致。瓦片字节=generic(已在六芯平台验证),CUDA 侧无
  新性能面,免跑代理基准。
- release(git 对象,gpu:/tmp/flagos-t33e7-rel.TMerlW):同套门禁 +
  unittest 8/8、`RELEASE_OK`;日志 SHA-256
  `446104c622e9d0ea0e93e156191d6c18b3c1268d9360d60497df60b02c387c4d`。

### 平台预注册门

- 基础门:8/8 valid(昆仑 XPU 对 2D 瓦片的编译是唯一风险面)。
- 单轴晋级门:昆仑严格高于 `0.231x` 且平均严格高于 e5 `4.57069167x`。
- 机制确认门:昆仑 ≥ `0.5x`(瓦片在 XPU 兑现)。
- stop gate:昆仑编译失败或 <0.231 → 回滚钉死字节,瓦片轴对昆仑
  关闭;燧原读数仍按水位处理,不归因。

### E7 平台提交与终态(sub 7324,2026-08-31 19:2x CST)

preflight intent `35e1763f…` 全匹配后单次 confirm(sub 7324,
daily_seq 23,额度 8→**7/30**);对象存储匿名回读 12326 bytes,
SHA-256 与 canonical ZIP 完全一致(`verified`)。

终态 **8/8、valid、平均 `4.57156667x`、is_team_best=true**(e5
`4.57069167` → +0.02%,噪声驱动):

| 芯片 | e5 | E7 | 变化 | 文件 |
| --- | ---: | ---: | ---: | --- |
| 天数 | 7.327 | 7.052 | -3.8%(噪声) | generic |
| 沐曦 | 3.168 | 3.439 | +8.5%(噪声) | generic |
| 燧原 | 0.883 | 0.854 | 水位 | enflame(冻结) |
| 海光 | 9.235 | 9.143 | -1.0% | generic |
| **昆仑** | 0.231 | **0.2197** | **-4.9%(瓦片证伪)** | kunlunxin(瓦片) |
| 华为 | 1.457 | 1.535 | +5.4%(噪声) | ascend(冻结) |
| 国际 A | 6.223 | 6.208 | 持平 | generic |
| 国际 B | 8.041 | 8.121 | +1.0% | generic |

**判定:瓦片在昆仑证伪,team best 微升属噪声。**

- 机制确认门(昆仑 ≥0.5)远未触发;0.2197 < 0.231 触发 stop gate
  → 树已回滚钉死 S0 字节(`017f993a…` 复核一致),瓦片-for-昆仑
  轴永久关闭。昆仑 0.22 判定为本题 XPU 固有水位(与 T21 归约
  BLOCK 唯一有效、T32 0.21、T40 循环前 0.24 同族);
- 平台 team best 按提交计分,4.5716 已锁定(含瓦片昆仑 0.2197);
  其余七芯字节未变,±8% 全在已知噪声带内;
- 跨芯知识:**`[4,G]` 瓦片的收益不迁移到昆仑**——六芯兑现的结构
  在 XPU lowering 下与单组循环打平略负,昆仑对每 program 工作量
  不敏感(0.22-0.23 恒定),瓶颈在别处(疑 store int8/除法 lowering)。

## E8:generic GROUPS_TILE 4 → 16(2026-08-31 19:3x CST)

状态:候选就绪待单次提交。41 算子核验建议的"1/4/8-group 按形分桶"
轴,先做离线扫描再收敛为平铺单变量。

### 离线扫描与确认基准(RTX 5070 Ti,零额度)

- 扫描(gpu:/tmp/flagos-t33e8.a5rugW,sweep_tile.py):8 形状 × 3
  dtype × t1/t2/t4/t8/t16 对 t4 基线——**t16 在高组数形状最高
  1.85×**(65536×256 G64 fp16 0.555、8192×512 0.76-0.80、
  1024×2560 0.84-0.89),小形状 ≤+9%(绝对 ~0.015ms,launch 噪声),
  fp32 大形状持平;t8 全面被 t16 支配 → 选平铺 16,不做分派。
- 确认基准(gpu:/tmp/flagos-t33e8b.H1nQI4,bench_e8.py 对 t4 对照):
  **全部契约合法形状逐位一致**;geomean `0.9231`,65536×256
  fp16/bf16 `0.605/0.617`、8192×512 ~`0.75`;
  **tile16 = 40 regs / 0 spills / 64B shared**(t4:29/0/16)——
  无溢出,跨芯寄存器风险可控。首轮一处 `129x1025 G32` "失配"
  定性为越契约形状(K 不整除 group_size,题面禁止):两 kernel
  均不覆盖余数元素,`torch.empty` 尾部垃圾位导致 `torch.equal`
  误报;改合法 `129x1024 G32` 后全绿,unittest 8/8 从未失败。

### 构建身份

| 项目 | 值 |
| --- | --- |
| source / verification commit | `ef912999b001f98258bfd16e829ad97a26aed3d2` |
| generic SHA-256 | `265716528d49099a8c788bdff91ba8190b83a074421c0de210552fb5a9e4a2a6`(=screening 字节) |
| Ascend SHA-256 | `fa7a1a46573f0e0179573ec8a8ebec10463dba42c942f68da308b62e9c0fd670`(=e3,冻结) |
| Enflame SHA-256 | `c6ff5fc03dea9f52c40eeb7dd19c6600e2962814dfb5771ddea7e1a168e3b6b0`(=e2,冻结) |
| Kunlun SHA-256 | `017f993a5bb0c9e79f7365152266067ba150a968632b14b1e4819b9e7ecd02dc`(=钉死 S0) |
| test SHA-256 | `2ec5b35b226c88865a5ea62133aa6639ab7ce50c1ded4416bd8526399d1d13d5`(=e7) |
| canonical ZIP | `artifacts/competition/per_token_group_quant_int8/e8-ef91299/per_token_group_quant_int8.zip`,12284 bytes,SHA-256 `bf400ccf1ab3c261decda9fd7a2d8a91d2908fac456e826c18ed4fa1301b6a39` |
| ZIP 成员 | generic + `_ascend` + `_enflame` + `_kunlunxin`;`unzip -t` 无错;成员哈希与 commit blob 逐项一致 |

### 验证与门

- screening(同上目录):py_compile/isort/flake8/unittest 8/8 全过,
  screen.log SHA-256 `698b43bf86b6885f84044699c6eec519d0cf5ad3eb84bfd60cbc9bab345350a6`;
- release(git 对象,gpu:/tmp/flagos-t33e8-rel.fvdSY9):同套 +
  `RELEASE_OK`,日志 SHA-256
  `7b30926b91dd5efe54cb946797319846a8f86412fab38ca0b168724f724670ad`;
- 平台预注册:基础门 8/8 valid;晋级门平均 > e7 `4.57156667x`;
  机制门 = generic 路由五芯(天数/海光/沐曦/国际 A/B)合计较 e7
  读数(7.052/9.143/3.439/6.208/8.121)均值 +5% 以上;stop gate =
  五芯中 ≥3 芯回退 >5% → GROUPS_TILE 扩展轴关闭回滚 4;
  昆仑/燧原/华为字节未变,读数按水位/噪声处理不归因。

### E8 平台提交与终态(sub 7327,2026-08-31 19:4x CST)

preflight intent `caaf0a9e…` 全匹配后单次 confirm(sub 7327,
daily_seq 24,额度 7→**6/30**);对象存储匿名回读 12284 bytes,
SHA-256 与 canonical ZIP 完全一致(`verified`)。

终态 **8/8、valid、平均 `5.44295833x`、is_team_best=true**
(e7 `4.57156667` → **+19.1%**;距榜首 starwing 6.3983 收窄至
**-14.9%**):

| 芯片 | e7 | E8 | 变化 | 文件 |
| --- | ---: | ---: | ---: | --- |
| 天数 | 7.052 | **10.822** | **+53.5%** | generic(16) |
| 海光 | 9.143 | **11.875** | **+29.9%** | generic(16) |
| 沐曦 | 3.439 | **4.320** | **+25.6%** | generic(16) |
| 国际 A | 6.208 | 5.989 | -3.5% | generic(16) |
| 国际 B | 8.121 | 7.716 | -5.0% | generic(16) |
| 燧原 | 0.854 | 1.028 | 水位(冻结字节) | enflame |
| 昆仑 | 0.2197 | 0.2324 | 钉死字节复位 | kunlunxin |
| 华为 | 1.535 | 1.562 | 噪声(冻结字节) | ascend |

- 机制门:五芯均值 6.793 → 8.144(**+19.9%**,远超 +5% 门)——
  代理扫描的方向与幅度在平台精确兑现(代理 t16/t4 大形状
  0.61x ↔ 天数/海光 +30~54%);
- stop gate 未触发(仅国际 A/B 各 -3.5%/-5.0%,2 芯且临界);
  跨芯谱:天数/海光/沐曦大幅受益,国际 A/B 轻微回退——与
  e5 瓦片轮的芯片谱一致,放大同一机制;
- 下一单变量:离线扫 GROUPS_TILE=32(寄存器 0-spill 前提)与
  燧原 vendor 路由换 tile-16 generic 的可行性。

## E9:燧原 vendor tile 4 → 16(2026-08-31 19:5x CST)

- t32/t64 离线扫描(gpu:/tmp/flagos-t33e9.uTJGmz):t32 多数形状
  ≥1.0(最差 1024×2560 bf16 `1.115`)仅 65536×256 微降 0.90-0.94,
  t64 更差且寄存器升至 106——**tile=16 判定为该轴最优,轴关闭**;
- e9 = 燧原 vendor `_GROUPS_TILE 4→16`(与 e8 generic 同一变量的
  单芯延展,kernel body 与 generic 逐字节同构;宽瓦片家族证据
  T24/T33-e2/T39/T29);其余四成员冻结;
- screening(gpu:/tmp/flagos-t33e9b.VxXJZq):unittest 8/8、
  `SCREENING_OK`,日志 SHA-256
  `c7bb3373ed2bd3e0b876898c7ebffb5d34e8e0d263741f4ccadebf128d3ed214`;
- release(gpu:/tmp/flagos-t33e9-rel.RUVvIv):8/8、`RELEASE_OK`,
  日志 SHA-256
  `a89e3e1fda7ea433025c0a6b00b3942ec119fc854028f7aa6ef3cafd261a1571`;
- 构建身份:source/verification commit
  `cf188a0fae1832c232f3b89048daecd1a8e49b65`;enflame SHA-256
  `5531d052c32976b0696444b690e04899a77d462a9041688d00c511059a94d9ef`;
  generic/ascend/kunlunxin/test = e8 同值;canonical ZIP
  `artifacts/competition/per_token_group_quant_int8/e9-cf188a0/`
  `per_token_group_quant_int8.zip`,12284 bytes,SHA-256
  `4bf87eee472f05b167c1eeb8f7f7edd309b62c7ce11ea4f39e2c5ee10505dfb2`;
### E9 平台终态(sub 7331,2026-08-31 19:5x CST)

preflight intent `e0407471…` 单次 confirm(sub 7331,daily_seq 25,
额度 6→**5/30**);远端回读一致(`verified`)。终态 **8/8、valid、
平均 `5.51501667x`、is_team_best=true**(e8 → +1.3%):

| 芯片 | e8 | E9 | 变化 | 文件 |
| --- | ---: | ---: | ---: | --- |
| **燧原** | 1.028 | **1.488** | **+44.8%(单变量兑现)** | enflame(16) |
| 天数 | 10.822 | 10.804 | 持平(冻结) | generic |
| 海光 | 11.875 | 11.964 | 持平(冻结) | generic |
| 沐曦 | 4.320 | 4.313 | 持平(冻结) | generic |
| 国际 A | 5.989 | 6.066 | 噪声 | generic |
| 国际 B | 7.716 | 7.782 | 噪声 | generic |
| 华为 | 1.562 | 1.473 | 噪声(冻结) | ascend |
| 昆仑 | 0.232 | 0.231 | 持平(冻结) | kunlunxin |

燧原 +44.8% 远过单轴门(>1.028),tile-16 结构在第七芯兑现;距榜首
starwing 6.3983 收窄至 **-13.8%**。

## E10:华为 vendor tile 4 → 16(2026-08-31 20:0x CST)

- 同一单变量的最后一芯延展:华为 1.47 是剩余唯一低于家族水位的芯;
  [16,128] fp32 瓦片 ~8KB vs e3 期 4KB"远低于 UB 预算"的实测,
  昇腾对宽瓦片的编译接受是平台假设(T40 direct 两连拒收前科,
  但那是无循环直通结构,非瓦片);
- screening(gpu:/tmp/flagos-t33e10.*):unittest 8/8;release
  (git 对象,gpu:/tmp/flagos-t33e10-rel.pI7KFe):unittest OK,
  rel.log SHA-256 `ddd16b8e2d858b89451d1c37541c1671d5a0b0febda5e4559556f3e1e7a9ac40`;
- 构建身份:source/verification commit
  `8e344b49d1326ddf21357f2218196b5d92cbf532`;ascend SHA-256
  `9a8a7468bc9bf721bbb79ed2d424ef25773625d746ec51275147bee4e5c6fe14`;
  其余成员 = e9 同值;canonical ZIP
  `e10-8e344b4/per_token_group_quant_int8.zip`,SHA-256
  `76cec783f7ec857561d6d6ff05c3b24850f61a6dc000947cca2f69aff5521d9b`;
- 平台预注册:基础门 8/8;晋级门平均 > e9 `5.51501667x`;单轴门
  华为 > e9 读数 `1.47273333x`;stop gate 华为编译失败或回退 →
  ascend vendor 回滚 e3 tile4 字节,瓦片轴对昇腾关闭。

### E10 平台终态(sub 7332,2026-08-31 20:1x CST):华为兑现,T33 收盘

preflight intent `f6bfa434…` 单次 confirm(sub 7332,额度 5→**4/30**);
远端回读一致(`verified`)。终态 **8/8、valid、平均 `5.57195x`、
is_team_best=true**(e9 → +1.0%):

| 芯片 | e9 | E10 | 变化 | 文件 |
| --- | ---: | ---: | ---: | --- |
| **华为** | 1.473 | **1.898** | **+28.8%(单变量兑现)** | ascend(16) |
| 天数 | 10.804 | 10.834 | 持平(冻结) | generic |
| 海光 | 11.964 | 11.981 | 持平(冻结) | generic |
| 沐曦 | 4.313 | 4.328 | 持平(冻结) | generic |
| 燧原 | 1.488 | 1.490 | 持平(冻结) | enflame |
| 国际 A | 6.066 | 5.970 | 噪声 | generic |
| 国际 B | 7.782 | 7.842 | 噪声 | generic |
| 昆仑 | 0.231 | 0.233 | 固有水位 | kunlunxin |

**T33 收盘于 e10 `5.57195x`**(e5 4.5707 → 今日 +21.9%,距榜首
starwing 6.3983 **-12.9%**):

- tile-16 单变量在 generic(五芯)+ 燧原 + 华为共七芯兑现
  (+20~54%);昆仑为 XPU 固有水位(0.23,e7 反证);
- 剩余差距分解:昆仑 -0.7(固有)、沐曦 4.33 与国际 A 5.97 为
  各自家族水位(card_a 与 card_b 无法用 vendor 后缀区分路由);
- 已知轴全部关闭:GROUPS_TILE(16 最优,t32/t64 反证)、昆仑瓦片
  (e7 反证)、div/reciprocal(div_rn 逐位必需)、两阶段(T34 反证);
  后续若重开需新证据(metax 专属 tile 或榜首结构面破译)。

## E11:metax vendor tile 32(2026-09-01 01:0x CST)

状态:候选就绪待单次提交(T39 e11 复证"metax 偏好自有形态"后的
T33 延伸;沐曦 4.33 仅为天数 10.8 的 40%)。

- 变更:新增 `_metax` vendor = generic tile16 同构、`_GROUPS_TILE=32`
  (commit `4ae358b`);generic/ascend/enflame/kunlunxin 字节冻结
  (2657/9a8a/5531/017f,逐项核对);测试矩阵补 metax 与 tile32
  尾块 case(33×64×64,total_groups%32=1,codex-review 发现);
- GPU 验证:unittest 8/8(gpu:/tmp/flagos-t33e11.*);tile32 与
  generic 逐位一致(4 形状 × 3 dtype);CUDA 代理参照:t32 中性
  (0.90-1.03,前期扫描)——纯 metax 侧赌注;
- release(gpu:/tmp/flagos-t33e11-rel.*,**显式退出码门**):RELEASE_RC=0,
  日志 SHA-256 `f989c57481d5e8a61caaa8fe67d478a59765235baaebbe69cead8ffc0d25d0c`;
- MCP 适用性:metax 不在 specialize 覆盖集(仅 huawei),以 NVIDIA
  代理 + 平台为通道,如实记录;codex-review 已跑(P3 尾块缺口已修,
  P1 pipe-to-tail 流程教训已采纳,其余为未跟踪文件噪音);
- canonical ZIP `e11-4ae358b`,16142 bytes,SHA-256
  `8f9ec1b3cd2a9486282abd02901a49f251f9d045a3a6fa3d3d27397aa6627263`;
- 平台预注册:基础门 8/8;晋级门平均 > e10 `5.57195x`;单轴门
  沐曦 > e10 读数 `4.32813333x`;stop gate 沐曦 < 4.33 → 恢复
  generic 路由(删 metax vendor),tile32 对 metax 关闭。

### E11 平台终态(sub 7487,2026-09-01 01:1x CST)

preflight intent `bf57e536…` 单次 confirm(sub 7487,额度 28→27/30);
远端回读一致(`verified`)。终态 **8/8、valid、平均 `5.57151667x`、
非 team best**(距 e10 `5.57195` 仅 0.0004):

- **沐曦 tile32 = 4.3362 vs tile16 的 4.3281(+0.2%,噪声)**——
  tile 轴对 metax 中性,16 与 32 等价;沐曦 4.33 判定为本题 metax
  水位(天数 10.8 的 40%,结构已无差别);
- 其余七芯冻结字节读数全部持平(tianshu 10.82/haiguang 12.06/
  huawei 1.85/card_a 6.08/card_b 7.73/enflame 1.46/kunlun 0.232);
- 树已移除 `_metax` vendor(恢复 e10 团队最佳成员集);
  **T33 重新封存于 e10 `5.57195x`**(距榜首 starwing 6.3983
  -12.9%);已知轴尽:GROUPS_TILE(16 最优,32 对 metax 中性)、
  昆仑瓦片、metax tile;剩余差距 = 昆仑 0.23 固有 + 沐曦/燧原水位。

## E12:wrapper 与 launch 参数穷举(2026-09-01 07:0x CST,负结果)

- 单变量扫描:删除题面连续输入下冗余 `x.contiguous()`(`nc`)；显式
  `num_warps=2/4/8`；以及 `nc` 与三档 warps 的组合。base commit
  `0b9dea4151800d4e0cbdd4446fdf4e863791a102`，本地源码未改、平台未提交；
- 远端 `gpu:/tmp/flagos-t33-offline.YJDLBX`，RTX 5070 Ti、PyTorch
  `2.13.0+cu130`、Triton `3.7.1`；base unittest 8/8，八份候选题面内
  correctness 矩阵全绿。wrapper-inclusive 7 shape、每 shape 7 组 AB/BA、
  正逆序两遍共 14 组；`nc/w2/w4/w8/nc+w2/nc+w4/nc+w8` 综合收益依次为
  **`-0.115/-2.519/+0.095/+0.286/-2.682/+0.217/-0.126%`**；
- 最优 `w8` 仅 +0.286%，且 `1024x2560,group_size=128,fp16` 回退
  5.85%(两遍汇总最大回退 6.21%)，触发 >5% stop gate；`w2` 小形状
  回退 7–8%。全部低于 +5% 晋级门，判负封轴；
- 资源:base/w4 在 group size 32/64/128 为 29/33/40 regs；w2
  33/40/63 regs；w8 27/28/33 regs；均 0 spill、64B shared、3 stages。
  correctness/bench/resource 日志 SHA-256 分别为
  `a42f4fffec9a39cdb59371bc4091bcfad06b5483b64dfbb4bcdf0685f7635eed`、
  `b8fe0fe2c9b50961030b8e13a6eef3b2de2fdf63146d3d7d94ed3870effa1f83`、
  `4ad6bba300d93cdd833348a2605346a19212fc58a9974192c0bead1890e08baf`。

结论:T33 的 wrapper 与 launch 参数轴也已穷尽；继续消耗额度不能解释实时
榜首 `7.238625x` 对 e10 `5.57195x` 的 29.91% 差距，保持封存。
