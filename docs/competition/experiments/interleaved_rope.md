# Task 30 `interleaved_rope` 实验记录

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

