# Task 38 `sigmoid_gate_topk_renorm` 实验记录

```current
task: 38
operator: sigmoid_gate_topk_renorm
batch: 3
validity: invalid
platform: 7/8
team_best_stage: S0
team_best_commit: 311570f
blockers: 昆仑 topk 族 Segfault(崩溃族第15例)
sealed: no
next: e2 device global_scale 去同步候选已过 release;一次结构改写平台门
updated: 2026-09-01
```

状态:S0 候选就绪

## 契约锁定

- 签名:`reference(logits, k, n_shared_experts, route_scale, global_scale, bias)`
- `logits [T, N+S]`(末 S 列共享);`bias [N]`;`global_scale [1]` fp32
- sel=sigmoid(routed)+bias → topk(argsort 降序);routed_vals 取
  **原始 logit**(非 sigmoid);probs=sigmoid(cat(routed_vals,shared));
  归一化 ×route_scale×global_scale;输出
  `(routed_weights [T,k] 输入dtype, indices [T,k] int32, shared_weights [T,S])`
- indices 精确匹配;容差 fp32 1e-4 / bf16 1.5e-2 / fp16 1e-2;八芯

## S0(2026-08-31,commit `311570f`)

- T31 机器(迭代 argmax+min-index 平局)**双确定性选轮**:第一轮
  indices + sigmoid 概率总和,第二轮复现选点写归一化权重——规避
  同 program store→load 可见性风险;共享专家恒活跃参与归一化;
  1D capped grid。
- screening(gpu:/tmp/t38.z8kOOd,字节与 blob 一致):unittest 3/3
  (3 dtype × 7 形状含 S=0/S>T 边界);bench 4/5 shape 1.42–2.61x,
  65536×64 小 N 档 0.536x(两轮选点在大 T 小 N 下偏慢,过门槛,
  平台 mix 待证)。
- 风险:topk 族昆仑前科(T25/26/27/31 皆崩);fp32 无 dot、无
  libdevice 超越函数(sigmoid=1/(1+exp))是本次差异点。

### S0 提交记录(2026-08-31 12:2x CST)

preflight 全过(tid `s2t1op038`,额度 4/30 消耗 1 → 3/30);单次
confirm 提交,评测入队,逐芯结果待回填。

### S0 平台终态(sub 6930,2026-08-31 14:0x CST)

**7/8**——七芯全过(卡_B 1.854/沐曦 1.580/华为 1.255/海光 1.215/
card_a 1.146/天数 0.819/燧原 0.421),**昆仑 1830s 验证段
Segmentation fault**(崩溃族第 15 例,与 Aborted 同族不同信号量)。

- topk 族昆仑结论第三次复现(T25/26/27/31/38);本 kernel 无 dot、
  无 libdevice 超越函数仍崩——"最干净形态"也未能幸免,题内止损
  (1 发即停,候选 `311570f` 封存可复用)。
- 第三批 17 题全部处置完毕:8 valid + 3 题 7/8 昆仑墙封存 +
  2 题语义黑盒止损 + 4 题早前止损。额度 3/30。

## E2:global_scale 保持设备侧(2026-09-01 06:4x CST)

状态:结构改写候选 release/不可变 ZIP 就绪,待实时 preflight。该候选
不是 S0/E1 同字节或注释载体重载:kernel ABI 与可执行路径均改变,属于
崩溃族协议允许的 topk 结构改写;若昆仑仍复现同族崩溃,立即重新封存,
不得自动重载。

### 假设与单变量

- S0 wrapper 对题面 GPU `global_scale[1]` 做 host scalar extraction,
  每次调用产生 GPU→CPU 同步;T39 已测同类同步税约 25us,足以淹没
  本题短 top-k kernel,也会给 torch.compile/Inductor 增加 host graph break;
- E2 只把 tensor `global_scale` 作为指针传入 Triton,每个 program 一次
  `tl.load`;双确定性选轮、sigmoid/归一化数学、TOPK/BLOCK/grid、输出和
  Python scalar 兼容路径全部冻结。源码不含 `.item()`/`.tolist()`/`.cpu()`;
- KernelGen `optimize_kernel` 在完整负结果约束下生成同一指针方案;Huawei
  `specialize_kernel` 预检保持该 scalar pointer load 不变,其余未经实机
  验证的 NPU 改写全部拒绝,不进入候选。

### screening 与性能门

- base `2761fd8`,候选/测试 SHA-256 分别为
  `e4840878c98e2d2a061a60a5438b069f70af2868c0d10322963e38b823a82e03` /
  `33b565aed5255236706764c69b25f39e32ae5f5a6a42a28e9b8b0f9db10d9c9f`;
  `gpu:/tmp/flagos-t38-nosync.7FRzZI`,RTX 5070 Ti,PyTorch
  `2.13.0+cu130`,Triton `3.7.1`;py_compile/Black/isort/flake8、
  unittest **3/3** 全过(三 dtype×七 shape、indices 精确、tensor/scalar
  scale 输出逐位一致);screen_final2.log SHA-256
  `64a31528691b14e109fece4432db7a350c773041b1f9b16fd54b22316de1d115`;
- 五轮交替 wrapper-inclusive AB/BA,八形状 base/candidate 时间比分别
  `4.780/3.288/3.281/1.191/1.619/3.274/1.012/1.593`,几何均值
  **`2.188x`**,最差大 T/小 N 仍不回退;资源最大同为 122 registers、
  0 spill/0 scratch,candidate 某变体 107→105 registers;
- 代理晋级门(geomean ≥1.5x、任一 shape 不回退 >5%、0 新 spill)全部
  通过。benchmark script SHA-256
  `f7026a40c3a2fb5305360ab80bb8f6f18ee2f74561fec7597e1c48f42f354c98`。

### 构建身份与平台预注册门

| 项目 | 值 |
| --- | --- |
| source / verification commit | `b610b7a1ef6d494130121495b455d3b2ab9c09e8` |
| release | `gpu:/tmp/flagos-t38e2-rel.o1sM3e`,Git 对象导出,哈希前后一致,unittest 3/3,`RELEASE_OK`;日志 SHA-256 `8cfe4e5e8525000f1a17707b5279afd3d0f986589c4b2a9af2d24dfe1ced695b` |
| canonical ZIP | `artifacts/competition/sigmoid_gate_topk_renorm/e2-b610b7a/sigmoid_gate_topk_renorm.zip`,5255 bytes,SHA-256 `5dc0316b2f10bc07342405506a70a86f0e5a772ac497f16430eac83e86685c80` |
| ZIP 成员 | `sigmoid_gate_topk_renorm.py`,SHA-256 `e4840878c98e2d2a061a60a5438b069f70af2868c0d10322963e38b823a82e03`;`--verify-existing`/`unzip -t` 通过 |

- 基础门:8/8 valid、每芯 ≥0.1x;Kunlun pointer scalar load 编译/执行是
  唯一新增跨芯风险面。
- 机制门:此前通过的七芯中至少 5 芯严格高于 S0,且无芯回退 >10%;代理
  表明收益随短 shape 占比变化,不强行外推 2.188x 到八芯。
- 登顶门:平均严格高于实时榜首 Fields `7.60535x`。
- stop gate:昆仑再现 1830s/Segfault/compile-worker 同族指纹,或任一既过芯
  因 E2 失败,立即恢复封存,只留结构算法重写/工单路径,不做同字节或载体重投。
