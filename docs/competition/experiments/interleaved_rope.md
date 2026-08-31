# Task 30 `interleaved_rope` 实验记录

```current
task: 30
operator: interleaved_rope
batch: 3
validity: valid
platform: 8/8
team_best_stage: S0
team_best_speedup: 25.8353
sealed: yes
next: 实时榜首37.7641;一读一写下界已达,MCP/官方实现复核无可信46.17%路径
updated: 2026-09-01
```

状态:S0 候选就绪,待额度重置后首投(与 T29 同因:2026-08-27 团队当日
30/30 额度已耗尽;2026-08-28 00:00 重置后按 29 → 30 顺序提交)

## 契约锁定

- 签名:`reference(x, mrope_section)`
- 输入:`x [3, S, D]` 三路 RoPE 流(t/h/w);`mrope_section [s0,s1,s2]`
  python int 列表,s0+s1+s2 = D//3
- 输出:`[S, D]`,dtype 同 x;纯选择无浮点运算:
  d%3==1 且 d < s1*3 取高度流;d%3==2 且 d < s2*3 取宽度流;
  其余(d%3==0 或超段界)取时间流
- 容差:同 dtype 逐元素,选择型天然精确

## 方案(S0)

- 每列来源仅由 d 决定;s1、s2 作为标量参数传入 kernel,列向量上
  `tl.where` 选择源指针,单 pass 读源写目的
- 纯字节搬运,内存主导;无 dot、无分支结构风险

## 提交预算与止损(2026-08-27 定稿)

- 每题 5 次提交预算;S0 首投探路 → 最多 3 次 vendor 单变量迭代 → 剩 1 次留作
  截止前回归储备。
- 同指纹失败连续 2 次提前停,不烧满 5 次;额度只花在有单变量假设的候选上。
- generic dot 策略(如涉及 `tl.dot`):fp32-ieee 操作数 generic + `_tianshu`
  split-fp16 vendor;昆仑保持 fp32-ieee(T12 镜像证据,昆仑 fp16-dot 数值失败
  有平台实证)。

## S0:generic baseline

状态:候选就绪,未提交(额度阻塞,非门禁失败)
时间:2026-08-27 21:33–21:50 CST
source/verification commit(同一提交):`99c154ee4e4d7621fabae5f5552290df178254ff`

### 构建身份

| 项目 | 值 |
| --- | --- |
| 源文件 | `src/flaggems_sglang/ops/interleaved_rope.py` |
| 源文件 SHA-256 | `7614c99e91ef5efb6b77893b295acadaf1362cfc344fd8ecf7f904a7248eb2f6` |
| 测试 SHA-256 | `3cddd345a3fb73ab3e40c8737e09df183782ad35dd134d8ceaf2915eb844ce84` |
| ZIP | `artifacts/competition/interleaved_rope/s0-99c154e/interleaved_rope.zip` |
| ZIP SHA-256 | `546b18abb6250b32209660c2513d8ec2f427cdfc6ffddcbcb07e5fced4222186`(与 canonical 一致) |
| ZIP 内容 | 单个顶层文件 `interleaved_rope.py`,2149 bytes;ZIP 2285 bytes |
| screening 目录 | `gpu:/tmp/flagos-interleaved-rope.GOrHgw`,mode 0700 |
| release 目录 | `gpu:/tmp/flagos-interleaved-release.1A0tlS`,mode 0700,文件取自 Git 对象 |

ZIP 成员、commit blob、远端两目录三方 SHA-256 逐项一致。

### 唯一候选配置

- flat 输出索引,`s = offs // D`、`d = offs - s*D`;
  `from_h = (d%3==1)&(d<3*s1)`、`from_w = (d%3==2)&(d<3*s2)`,
  `stream = where(from_h,1,where(from_w,2,0))`,单次
  `load(x + stream*S*D + s*D + d)` 后 store——每元素恰好一读一写,
  dtype 保持(无浮点转换)。
- BLOCK 1024;1D grid `min(cdiv(n,1024), 65535)` + grid-stride 折叠;
  不显式设置 num_warps/num_stages。
- wrapper:`contiguous()`;`n==0` 提前返回;s1/s2 由 python int 列表取值;
  无 try/except、设备判断或 PyTorch fallback。

### 正确性

远端环境同 T29(RTX 5070 Ti、Python 3.12.13、PyTorch 2.13.0+cu130、
Triton 3.7.1)。lint:isort/flake8 远端通过,black 25.12.0 本地通过
(远端 26.5.1 漂移仅记录)。screening 与 release(取自 Git 对象)两次均
7/7 通过,覆盖:

- fp32/fp16/bf16/int64 四 dtype 精确相等(atol=rtol=0);
- 18 组 (S, D, section) 边界:D=8/9/16/63/64/65/511/512/513/1023/1024/
  1025/4096/192(Qwen2-VL 型 [16,24,24]),s1=0、s2=0、全零等退化 section;
- 逐列精确核对 96 列的流来源;
- 非连续列切片输入;输入不变性;S=0 空序列;
- 65536×1024 强制 grid 折叠路径。

### 远端 NVIDIA 代理性能(wrapper-inclusive,五组 AB/BA p50 中位数)

| dtype | S×D | op p50 (ms) | torch p50 (ms) | speedup |
| --- | ---: | ---: | ---: | ---: |
| float16 | 256×512 | 0.006048 | 0.085312 | 14.1058x |
| float16 | 4096×2048 | 0.060000 | 0.148464 | 2.4744x |
| float16 | 8192×4096 | 0.221088 | 0.395136 | 1.7872x |
| float16 | 65536×4096 | 1.737824 | 2.842240 | 1.6355x |
| bfloat16 | 256×512 | 0.006048 | 0.085216 | 14.0899x |
| bfloat16 | 4096×2048 | 0.060352 | 0.147936 | 2.4512x |
| bfloat16 | 8192×4096 | 0.221088 | 0.395264 | 1.7878x |
| bfloat16 | 65536×4096 | 1.739840 | 2.842272 | 1.6336x |
| float32 | 256×512 | 0.006208 | 0.088256 | 14.2165x |
| float32 | 4096×2048 | 0.112544 | 0.217984 | 1.9369x |
| float32 | 8192×4096 | 0.436320 | 0.772512 | 1.7705x |
| float32 | 65536×4096 | 3.465216 | 5.665504 | 1.6350x |

最差 1.6336x。

### 已知边界与风险

- 整数 `%3`、比较与 where 均为标准整数运算,无 lowering 风险面;
  NVIDIA 代理不能外推八芯,但该 kernel 无超越函数、无 dot。
- int32 索引上限 2^31 输出元素,远超合理 shape。

### 提交计划

- 额度重置后 preflight tuple:season 2、race `782kzq4m`、account
  `15600308080`、team `SoulCoder`、batch 3、task 30、tid 待 preflight
  实时确认(T29 为 `s2t1op029`,按序号推断为 `s2t1op030`)、operator
  `interleaved_rope`、stage `s0`、commit
  `99c154ee4e4d7621fabae5f5552290df178254ff`、ZIP
  `artifacts/competition/interleaved_rope/s0-99c154e/interleaved_rope.zip`、
  SHA-256 `546b18abb6250b32209660c2513d8ec2f427cdfc6ffddcbcb07e5fced4222186`、
  member `interleaved_rope.py`。

## 平台提交记录

- 2026-08-28 00:07 CST 额度重置(30/30)后,按 29→30→25→28→27→26 顺序自动
  preflight + 一次性提交;全部 tuple 与账本一致后执行 confirm。
- 提交时间约 2026-08-28 00:10 CST;submission_id `5735`;ZIP SHA-256
  `546b18abb6250b32209660c2513d8ec2f427cdfc6ffddcbcb07e5fced4222186`;state `submitted`、validity `pending`、评测入队。
- 提交后团队当日额度剩余 24/30(6 投全记录)。


### 八芯结果(S0 首投,sub 5735,终态)

**8/8 全过,valid,平均加速比 25.83534375x。** 纯选择 kernel 无跨芯
风险面,一次通过;Task 30 闭环完成,保留 S0。

## E1 候选(未提交,负结果)

- `dim` 转 `tl.constexpr`(消除逐元素整除):代理 7/7 正确,但加速比仅
  +0.5~1%(2.474→2.498x 等),低于预注册的 ≥1.08 提交信号;除法开销被
  内存成本掩盖,纯拷贝 kernel 已近带宽上限。不消耗提交预算,变更保留在
  工作区供后续批次参考。


## kernelgen MCP 结构轮负结果(2026-08-28 晚)

用户指示用 kernelgen 优化第三批;T30(8/8 valid、25.835x,榜首 29.077x)
为一目标。三轮 `optimize_kernel`(device=nvidia,带完整跨芯约束上下文)
全部否决,均为 NVIDIA 代理(RTX 5070 Ti)wrapper-inclusive 五组 AB/BA:

1. **2D 网格 + int64 索引**(小 shape 走 (row-tile × col-tile) 2D、列流
   摊销;大 shape flat):geomean 14.77x → **5.14x**。int64 逐元素乘加
   拖垮大 shape;小 shape 无收益(20.6x vs 20.8x,纯 launch-bound)。
2. **行条带 1D**(block 级除法 + 流向量按程序摊销,含 int64 清理版):
   → **4.73x**。static_range 行循环与 next_pow2(dim) 缩小块一并拖垮;
   清理版(纯 int32、无显式 launch 参数)同样回退。
3. **三元组形式**(利用 bound=3k、d%3==0 恒流 0,消全部 %3///3):
   数学错误——flat%3==d%3 仅当 dim%3==0;题面 D=512/2048/4096 均不
   整除,直接否决未跑基准。

结论:HEAD 平铺 kernel(BLOCK 1024、constexpr dim、1D capped
grid-stride)是代理可见轴上的局部最优;与榜首 12.5% 差距不可由代理
可见结构/参数轴解释(疑在逐芯差异或 wrapper 之外)。E1 constexpr
(+1%,低于 1.08 信号)维持不提交。本轮无候选、不消耗提交预算;
筛选目录 `gpu:/tmp/flagos-kgen.QODZAG`(screening 模式,base
`b03ea98`)。

## autotune_kernel(华为实机)负结果(2026-08-29 01:2x)

`autotune_kernel` device=huawei 5 轮任务完成:胜出代码为 TLE-DSA 形态
(`tle.dsa.alloc` UB 缓冲 + `tle.dsa.copy` 逐行搬运 + `torch_npu` 依赖 +
8 组 autotune 配置),其自测加速比仅 **1.79x**(我方 generic 在平台华为端
7.26x,口径若可比则更慢)。集成风险(平台编译时长、torch_npu/tle.dsa
在评测环境的可用性未证)高而收益证据不足,按止损纪律**不采纳**。
筛选记录:log/kernelgen-round/out_t30_at_huawei.json(job 877a19c9)。

## 2026-09-01 登顶复核(只读止损)

- 实时榜首升至 `37.7641x`，S0 `25.8353x` 需 **+46.17%**；逐芯为
  天数 71.864、海光 42.540、card_a 27.100、card_b 25.400、沐曦
  17.960、燧原 8.734、华为 7.255、昆仑 5.831x；
- 当前 fused kernel 每个输出元素恰好一次全局读、一次全局写且仅一次
  launch，已达到题面数据流下界。[vLLM 当前官方实现](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/rotary_embedding/mrope.py)
  仍是 `x[0].clone()` 加两次 stride-3 slice 赋值，不能提供低于一读一写的
  可迁移结构；SGLang 也未发现该独立重排算子的更低数据流实现；
- 已跑 MCP 的 2D、行条带、三元组三条结构分别显著回退或数学不成立；
  `dim constexpr` 仅 +1%，华为 autotune 也低于现有实现。现阶段继续扫
  BLOCK/warps 只能调内存 kernel 的个位数噪声，无法覆盖 46.17% 榜差；
- 因无新单变量且旧证据已覆盖结构、参数、单芯 autotune 三层，本轮不改
  源码、不消耗额度。重开条件仅限新 vendor 原语或榜首逐芯谱显示单芯可
  贡献至少 95.43 总分增量。
