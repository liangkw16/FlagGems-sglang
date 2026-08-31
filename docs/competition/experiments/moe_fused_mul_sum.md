# Task 32 `moe_fused_mul_sum` 实验记录

```current
task: 32
operator: moe_fused_mul_sum
batch: 3
validity: valid
platform: 8/8
team_best_stage: S0
team_best_speedup: 4.4829
sealed: no
next: E3 宽瓦片证伪;昆仑 0.21x 与榜首 23.9 的领先面未解释
updated: 2026-08-31
```

## S0：generic baseline

状态：screening 进行中（远端 NVIDIA 代理）

生成时间：2026-08-29

### 契约

- 接口：`moe_fused_mul_sum(inputs, topk_weights, topk_ids=None,
  expert_map=None, routed_scaling_factor=None, is_ep=False)`，与题面
  reference 完全一致。
- `inputs` `[T, top_k, D]`（fp16/bf16/fp32）；`topk_weights` `[T, top_k]`；
  `topk_ids` `[T, top_k]` int32 或 None；`expert_map` `[num_experts]` int32
  或 None；`routed_scaling_factor` float 或 None；`is_ep` bool。
- 语义：`w = topk_weights.float() * (scale or 1.0)`；有 `expert_map` 时
  `w *= (expert_map[topk_ids] >= 0)`，否则 `is_ep` 时 `w *= (topk_ids >= 0)`；
  `out[t,d] = Σ_k inputs[t,k,d] * w[t,k]`，FP32 累加，输出 `[T, D]` 与
  inputs 同 dtype。
- 容差：FP32 `1e-4/1e-4`，BF16 `1.5e-2/1.5e-2`，FP16 `1e-2/1e-2`。
- 支持芯片：天数、沐曦、燧原、海光、昆仑芯、华为、国际通用 A/B（八芯）。
- 核心计算只走 Triton；无 try/except、无 PyTorch fallback、无设备分支。

### 生成路径

- 按仓库规则走 `kernelgen-mcp generate_kernel`（服务端 KernelGen 2.0.0，
  streamable HTTP JSON-RPC）；生成后由服务端验证：
  `passed=true`，`speedup=2.35x`（服务端 shape (64,8,7168) 口径）。
- 服务端生成版两处已知问题，整理进仓库时修正：
  1. 输出先建 FP32 tensor 再 `.to(dtype)`，多一轮全量读写 —— 改为直接
     以 inputs dtype 分配输出，`tl.store` 自动 cast；
  2. expert_map 索引用 `.to(tl.int64)` —— 改为 int32（num_experts 小，
     国产后端 int64 惩罚更重，与 T21 经验一致）。

### 唯一候选配置

- 2D grid `(num_tokens, ceil(D/256))`，BLOCK 256，`TOP_K` constexpr 展开。
- 每个 program 处理一个 token 的一个 hidden block：先逐 k load 标量权重
  （乘 scale 并按 expert_map/is_ep `tl.where` 置零），再 load 输入块 FP32
  累加；单次读 inputs、单次写 output。
- inputs 三个 stride 与 weights/ids stride 显式传入；input stride 在 kernel
  内 cast int64（大 shape 地址溢出保护），token/hidden 偏移 int32 起算。
- `HAS_EXPERT_MAP`/`IS_EP` 为 constexpr 分支；topk_ids 为 None 时传空
  int32 tensor 占位，两个分支均为编译期消除。
- 空 T/D 直接返回合法空输出；top_k=0 时循环体为空、输出为零，语义与
  reference 的空维求和一致。
- 显式 4 warps、1 stage；无 autotune、无 vendor 文件。

### 上游参考

- vLLM `fused_moe.py` 的 `moe_sum` 与 Fused MoE Modular Kernel 的
  TopKWeightAndReduce（mul+reduce 融合 epilogue 思想）。
- SGLang fused_moe_triton_kernels（一 token 一 hidden tile、FP32 累加）。
- 本仓库 T21 `moe_sum_reduce` 的地址公式与 launch 纪律。
- 引入的外部最佳实践参考：
  `docs/competition/reference/kernel-skills/patterns-fuse-elementwise-ops`。

### Screening（进行中）

- 模式：screening（未提交候选快速筛选）。
- base commit：`9714d53`（工作树另含用户未提交改动，未触碰）。
- 本地/远端 SHA-256 一致：
  - `src/flaggems_sglang/ops/moe_fused_mul_sum.py`
    `097009c6771e74bb2c67418f16e7d78bcea22d28b69f84be7b677c6c6c2c537e`
  - `tests/test_moe_fused_mul_sum.py`
    `5f3cabe245d20322b8965d994d2a8f540063d890585d202eb1f52576a3cb68e0`
- 远端证据目录：`gpu:/tmp/flagos-moe_fused_mul_sum.9hfSsb`（mode 0700，
  PID 191334，日志 `run.log`）。
- 远端环境：NVIDIA CUDA，torch 2.13.0+cu130，triton 3.7.1。
- 测试矩阵：三 dtype 主 shape、scale None、expert_map 掩码（含 -1 槽位）、
  is_ep 负 id 掩码、非连续输入、块边界（255/256/257/511/512/513）、
  空维（T=0/D=0/top_k=0）、平台规模 (4096,8,7168)。

### Screening 结果（第一轮，未通过 → 已定位）

- 远端 job 于 09:35 完成：`Ran 8 tests, FAILED (errors=26)`。
- 错误数完全吻合的根因：**测试 harness 传参 bug，非 kernel 缺陷**。
  5 个用例把 scale 写成第 3 个位置参数（该位是 `topk_ids`），
  wrapper 里 `topk_ids.stride(0)` 抛 AttributeError。错误分布
  3(dtype)+1(非连续)+18(块边界)+1(空维仅 (2,0,17) 到达 stride,
  另两 shape 被空维早退拦住)+3(平台规模)=26，与日志完全一致。
  修复：全部改为 `routed_scaling_factor=` 关键字传参。

### Screening 结果（第二轮，通过）

- 测试修复后同目录重跑（远端字节 = 本地字节）：
  - 源码 `f586b45b68172075ef2bbf12a215df1a8c789bb9d829b366c5cebc95a54ca238`
    （与 ff09392 commit 相同，未改动）；
  - 测试 `7895539ed8de370098fa1128046af178e5e5554a9e6eed74afee91093546c798`；
  - `python -m unittest -v`：**8/8 全部通过（UT=0）**；
  - isort/flake8 通过。
- black 口径澄清：远端临时目录无 `pyproject.toml`，black 回退默认
  line-length 88 才报 BLACK=1；仓库 `[tool.black] line-length = 79`
  下两文件均合规（本地 black 26.5.1 + py3.12 验证 unchanged）。
  远端证据命令应显式 `-l 79` 或携带 pyproject，已在 run3/run4 采用。
- 远端环境：NVIDIA CUDA（gpu:/tmp/flagos-moe_fused_mul_sum.9hfSsb，
  mode 0700），torch 2.13.0+cu130，triton 3.7.1；GPU host 间歇性
  outage 多次（与 T33 记录同型），日志轮询带退避。

（release 重验、benchmark 与 ZIP 打包待补；未提交平台）

## S0 定稿(2026-08-29 18:2x CST,接续会话记录)

- benchmark 假象澄清:本会话首轮 bench 的 AB/BA 标签反转,导致
  "0.11–0.55x" 假读数;修正后 kernel 实为 **2.56–8.75x**。
- 采用重写版(commit `63e2550`):flat 1D capped grid + 块级除法 + int32,
  消除原 2D grid 超华为 65535 上限、int64 逐元素、显式 launch 参数
  三处跨芯风险;K 循环权重/掩码全寄存器。
- unittest 8/8(gpu:/tmp/t32.qa34FX);bench 修正后:
  4096×8×7168 **8.75x**、65536×4×1024 8.50x、256×8×4096 5.96x、
  1024×16×512 5.61x、16×4×2048 2.56x、128×8×7168 fp32 1.85x。
- ZIP `s0-63e2550/moe_fused_mul_sum.zip`,SHA `0706e14647c36c26c785812bd79281a6e68b969efd56e7387d66f8e3124a915e`。

### 快照关系说明（2026-08-29 16:0x CST 补）

- `s0-ff09392/moe_fused_mul_sum.zip`（SHA
  `8e1e5660dbb16e962a99b8cc9589e26f19e790fae5c14b22e51721da43f9ba11`，
  成员 SHA `f586b45b…`）是首轮 screening 候选（2D grid 版）的不可变
  快照，已被 `63e2550` 跨芯重写取代，仅作证据链保留；active 候选为
  `s0-63e2550`。
- 该快照与 screening 第二轮 8/8 通过字节逐一致（commit blob 校验）。

### S0 平台终态(sub 6351,2026-08-29 18:5x CST)

**8/8 全过,valid,平均 4.4829x —— 超过快照榜首 HAiWORLD 4.2575x
(+5.3%),预计登顶。**

| 芯片 | speedup |
| --- | ---: |
| haiguang | 10.298 |
| tianshu | 7.599 |
| card_a | 5.329 |
| card_b | 5.238 |
| muxi | 4.079 |
| enflame | 2.011 |
| huawei | 1.100 |
| kunlunxin | 0.210 |

- 额度 12/30。后续轴:昆仑 vendor(0.21x 是唯一大短板)、huawei
  结构轴;榜首位置由后续榜单快照确认。

## E1：Kunlun 无动态 loop direct fast path

状态：平台 8/8、`valid`、非 team best；保留 S0。

### 假设与单变量

2026-08-30 02:54 CST 官方实时榜首为 HelloWorldTJU `4.542975x`
（28 次提交、9 支队伍）；SoulCoder S0 为 `4.482900x`，只差
`0.060075x`。S0 唯一大短板是 Kunlun `0.210000x`；冻结其余七芯时，
Kunlun 只需严格超过 `0.690600x` 即可登顶。

T40 同芯、同 flat capped grid-stride 结构刚由单变量平台对照证明：移除动态
outer loop 后，Kunlun 从 `0.24566667x` 提至 `0.94450000x`，达到原来的
`3.8446x`。按该倍率机械外推，T32 Kunlun 约为 `0.8074x`，平均约
`4.5576x`；该数字只用于阈值规划，不替代平台证据。反证是 T32 每 tile 还有
静态 TOP_K 累加，loop control 占比可能比 T40 更低，且 T21 同族 reduction
即使没有动态 outer loop，Kunlun 仍只有约 `0.175x`。

E1 因此只新增 `_kunlunxin` vendor：保持 S0 的 BLOCK512、flat
`block_id -> token/col_block`、静态 TOP_K、FP32 累加、mask、地址和 launch
默认值不变；`total_blocks <= 65535` 时走一 program 一 tile 的独立 direct
kernel，超限仍走 S0 原 grid-stride fallback。已知主 shape
`(4096, 8, 7168)` 为 `57,344` programs，命中 direct；六个代理 shape 中
五个命中，`(65536, 4, 1024)` 的 `131,072` blocks 保持 fallback。本轮不混入
BLOCK、2D grid 或 host chunk。

### 构建身份与验证

| 项目 | 值 |
| --- | --- |
| source / verification commit | `fd970acfd630ffe97f944dfc0ce3786b8968481f` |
| generic SHA-256 | `ffb4440c7f62097b39a60c78be4a9416f452c40328d752b2ec8eddabbef22cd8`（=S0） |
| Kunlun SHA-256 | `48f7b5aac0349b7097ab462c97adf71e5b2e47e4b9b54c11651efbdfffd0e7db` |
| test SHA-256 | `7895539ed8de370098fa1128046af178e5e5554a9e6eed74afee91093546c798`（=S0） |
| screening | `gpu:/tmp/flagos-t32-direct-screen.IRCPwu`；vendor 作为 generic 装载，8/8、静态门禁全过；日志 SHA-256 `767e1a8f3a69dd7f7ebdf2c9e9db902d2f26db844c30d074535a07fe331e4484` |
| release | `gpu:/tmp/flagos-t32-direct-release.jSEZn8`；从 commit Git 对象构建，generic/vendor 各 8/8，前后哈希一致、`RELEASE_OK`；日志 SHA-256 `af4cd3bc8268a10726bb924884309e74cef5c3496c787fb742b5e61df2528dd9` |
| canonical ZIP | `artifacts/competition/moe_fused_mul_sum/e1-fd970ac/moe_fused_mul_sum.zip`，`9059` bytes，SHA-256 `33ce51a2044692e0ddea8d1049234689752ee697901384da71d47d831ec36415` |

RTX 5070 Ti 六形态五轮 AB/BA 中，direct/control 为 `0.999–1.019x`，含
`131,072` blocks fallback；只证明数值、launch 与性能未明显回退，不外推
Kunlun 收益。benchmark 日志 SHA-256
`19f74b67e207cd007278758228d9cae898fbe07b6dbddb51d16ec5647565e94b`。
同一 scale-none 编译变体中，S0 TTIR 含 `1` 个 `scf.for`，E1 direct TTIR
为 `0`；两份证据日志 SHA-256 分别为
`d50a7ae69f592c0dad32803607060d693660406511db547b6b6184d7c00a8b6b` 与
`a439be90162c4cda4927f6cc2ffb96e482bb5ebd2ddf8a086094fc2c81a0edb6`。

### 提交预注册

02:54 CST 实时只读状态精确匹配账号 `15600308080`、团队 `SoulCoder`、
race `782kzq4m`、batch 3、Task 32、tid `s2t1op032`、operator
`moe_fused_mul_sum`、`competing/submitting`、`can_submit=true`，额度
`16/30`，最小间隔已满足。

E1 ZIP 只允许提交一次。基础门为 8/8 `valid` 且 Kunlun 选择
`moe_fused_mul_sum_kunlunxin.py`；单轴晋级门为 Kunlun 严格高于
`0.210000x` 且平均严格高于 `4.482900x`；direct 机制确认门为 Kunlun
`>=0.400000x`；冲榜门为 Kunlun 严格高于 `0.690600x` 且平均严格高于实时
榜首 `4.542975x`。若 Kunlun `<0.400000x`，停止 direct 轴；若机制确认但未
登顶，只基于新的逐芯证据决定是否重开 2D/grid 轴；若登顶，冻结 E1 字节并转
下一题。其余七芯使用 S0 generic，分数变化只视为平台波动。

### E1 平台终态（sub 6615，2026-08-30 02:56 CST）

实时 preflight tuple 全匹配，额度 `16/30`；E1 只提交一次，提交后额度
`15/30`。对象存储匿名回读 `9059` bytes，SHA-256 与 canonical ZIP 完全
一致，remote verification=`verified`；
`file_url_sha256=d0bfe771a3f8af8a12bd64eb5e1fe123ff00d8e5dd1b4b38721206fd57ccd091`。

终态 **8/8、valid、平均 `4.298675x`、非 team best**；团队最佳仍为 S0
`4.482900x`，公开榜首仍为 `4.542975x`：

| 芯片 | E1 | 相对 S0 | 文件 |
| --- | ---: | ---: | --- |
| 天数 | `7.731800x` | `+0.132400x` | generic |
| 沐曦 | `4.067200x` | `-0.011800x` | generic |
| 燧原 | `0.356800x` | `-1.653800x` | generic |
| 海光 | `10.290000x` | `-0.008000x` | generic |
| 昆仑 | `0.211600x` | `+0.001600x` | `_kunlunxin` |
| 华为 | `1.171000x` | `+0.071400x` | generic |
| 国际 A | `5.312200x` | `-0.016400x` | generic |
| 国际 B | `5.248800x` | `+0.010800x` | generic |

唯一新变量所在的 Kunlun 只提升 `0.76%`，远低于预注册 `0.400000x` 机制门，
证明 T32 的 TOP_K 载入与 FP32 累加主导耗时，T40 的动态-loop 病理不能迁移到
本题。未改字节的 Enflame 从 `2.010600x` 波动到 `0.356800x`，是本次平均下降
的主要来源，不能归因于 Kunlun vendor。按预注册永久停止 T32 direct 轴，不做
事后 BLOCK/2D-grid 扫描或重传；冻结并保留 S0 team best，转下一题。

### E1:昆仑 BLOCK 1024 vendor(2026-08-31 16:0x CST)

- 平台证据:S0 昆仑 0.210x 是唯一短板(valid 已锁定,纯排名上行);
  T21 平台实证昆仑 BLOCK 唯一有效(1024)。
- vendor = generic + `_BLOCK 512→1024`(commit `aa2c0af`,其余逐字节
  一致);代理中性(±2%);unittest 8/8(gpu:/tmp/t32k.ol4dab)。
- ZIP `e1-aa2c0af`,SHA `7d40dab2ec43548a28b5a60099117119f61586ca4c234267364ee51d3270eb24`,2 成员。

### E1 提交记录(2026-08-31 16:1x CST)

preflight 全过(额度 3/30 消耗 1 → 2/30);单次 confirm 提交,
昆仑终态待回填。

### E1 平台终态(sub 6946,2026-08-31 16:4x CST)

- 8/8 valid,avg 4.2537(团队最佳保持 S0 4.4829);
- **昆仑 BLOCK 1024 证伪**:0.210→0.203(持平),T21 的 BLOCK 轴
  结论不迁移到本题 K 循环结构;昆仑 0.21 判为本题固有水位,轴关闭;
- 伴生观察:燧原 generic 字节未变却 2.011→0.353(平台方差极大),
  佐证跨题比较需谨慎。
- 额度 2/30。

### E2:燧原 [TOP_K,BLOCK] 2D 加权归约 vendor(2026-08-31 00:4x CST)

- codex 派生项:T33 E1→E2 结构类(顺序 static_range → 并行矩阵)
  应用到 K 循环;K 维 pad 2 次幂 + 双 mask(非 2 次幂 top_k 曾使
  tl.arange 崩溃,已修);
- commit `b428879`,ZIP `bddf52e3…`,3 成员(generic+enflame+
  kunlunxin[e1 字节,已证中性]);unittest 8/8;额度 28/30。

### E3:kernelgen 二轮宽瓦片(2026-08-31,未提交,双重否决)

- kernelgen `optimize_kernel` iter1 产出 [ROWS×COLS] 宽瓦片
  (weights 2D 瓦片 + K 静态展开);入库前三处自修:TOP_K 非 2 次幂
  `tl.arange` 崩溃雷(T33 前科)、显式 `num_stages` 昆仑 invalid 参数、
  权重改 per-k 一维载入;
- 候选 commit 未产生:screening(gpu:/tmp/flagos-catchup.NOF9kN,
  `t32_candidate` SHA `0192e158…`)双重失败:
  1. **代理门失败**:六 shape AB/BA 对 S0 geomean ≈ **-4%**
     (16×4×2048 2.60→2.07x、256×8×4096 5.83→5.45x,
      65536×4×1024 8.51→8.50x,1024×16×512 +1.9%,fp32 +3.1%)
     ——远低于预注册 +30% 门;
  2. **unittest 1 失败**:65/7175 元素失配(最大 abs 7.84),
     根因 = 列掩码误用 tile 内偏移(`col_offs < hidden_dim` 应为
     `(col_tile*COLS + col_offs) < hidden_dim`,尾瓦片越界);
- 结论:宽瓦片结构在 NVIDIA 代理上不优于 S0 的窄条 K 循环,
  YY-L 23.9x 的领先面仍未被该轴解释;树已回滚 S0 字节,
  候选字节保留于 screening 载荷;本题继续持有 S0 4.4829x。
