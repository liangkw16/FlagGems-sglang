# Task 82 `hash_topk` 实验记录

```current
task: 82
operator: hash_topk
batch: 6
validity: valid
platform: e8r载体已上膛未发射(commit fb97b3da,注释only,与e8 ZIP成员逐字节同源):昆仑水位假说(e8/e9读0.13vs e5窗~1.2;T87当日昆仑1.27→7.04);预注册门kunlun≥0.4/avg>6.42825换TB——当日30/30额度于10:11耗尽,未发射
candidate_stage: e8r-armed-unfired
team_best_stage: e8
team_best_speedup: 6.42825
sealed: yes
next: 窗口19:59关闭且当日额度耗尽,载体封存;若第七批重开或明日额度赶得上其他窗口,e8r ZIP可复用
updated: 2026-09-24
```

## 过程摘要（2026-09-19 凌晨，题面 09-18 晚随批 6 扩容上线）

- 开发：s0 generic + 全 unittest 矩阵（commit `540b574a` 家族），代理
  release 多轮修复后 8/8 验证；按回执逐芯定位失败根因，vendor 修复弹
  按 codex-review 门后发射（review 抓出 moe_align_single_token 填充轮
  越界读 P1 并修复）。
- 回执与产物：见 `artifacts/competition/b6ext-*-20260919/`（verification
  + log 逐弹归档），ZIP 在各 `artifacts/competition/hash_topk/` 下按 stage。
- 今日新增跨芯硬事实：XPU 拒 f32→i16 位转换（经 i32 高半字绕过）；
  XPU 拒 tt.scan/tl.cumsum 与 atomic_rmw；XPU 上张量索引 gather 与
  runtime 宽度寻址产生垃圾（hash_topk 三形态同指纹，标量串行为唯一
  可用形态）；GCU make_gcuir 拒张量索引 gather（torch 预 gather 先例
  再证）；XPU bf16 downcast 刀刃值与 eager 差 1 ulp（rtne 显式钉仍差）。

## 2026-09-20 E5/E6 平台终态：燧原 streaming vendor +12%，TB 6.316

- **E5（18357，7/8）教训**：56fb359 的 WIDTH-constexpr vector vendor 泄漏进
  ZIP（当时判负的实验字节未回退），昆仑复现 garbage-ids——**实验字节回退
  纪律**：判负的 vendor 改动必须立即回退或删除，否则后续组合弹携带已知
  坏字节。
- E6（18358）：scalar kunlun 恢复 + enflame 24-SIP+stages3 vendor。
  **燧原 0.8→0.9（+12%）**；均值 **6.316 新 TB**（+3%）。


## 2026-09-21 E7 候选就绪（燧原官方 gcu300 规则集，待 09-22 发射）

- 官方源码调研（FlagGems _enflame gcu300 codegen / FlagTree enflame
  backend）：max_grid_size=(12,1,1)（grid-stride 吸收超额）、
  enflame_heuristics_for_num_warps 钉 2、stride 需编译期互整除才走 DMA。
  本题 vendor 应用：grid 24→12+ num_warps=2。
- 五元组：commit `1909c3d7`；ZIP
  `artifacts/competition/hash_topk/e7-14437ad/hash_topk.zip`
  SHA `5ead1539026ec587b09edc2ab8421d37d7fc7259732c0dfac37d16a945c466d0`；成员 generic/enflame/kunlunxin（generic 与非燧原 vendor 字节
  不变，账本明确列全）；回执 `day5prep-20260921/hash_topk/verification.json`
  SHA `de519ccf7e97049c…`。
- 预注册门：燧原 0.85→≥1.7(×2)；其余七芯不动（vendor-only 单变量）。codex-review
  零发现（grid-stride 边界模拟无漏算）。

## 2026-09-22 E8 候选就绪（燧原 kernel 内 one-hot match-reduce，待预筛 + 09-23 发射）

- **假设**：燧原 0.85 的瓶颈是 wrapper 里 torch 预 gather 暂存链
  （`tid2eid[input_ids.long()].long()` + `torch.gather` + `.contiguous()` +
  `.to(int32)` 共 4-6 个 i64/非连续病理算子，GCU i64 NOT_SUPPORT 走 CPU），
  而非 kernel 几何（grid=min(rows,12)+num_warps=2 已是官方 gcu300 规则集）。
- **实现**（enflame vendor 单文件，其余 7 成员字节冻结）：gather 移入 kernel 做
  one-hot match-reduce——整行线性读（NRTILE≤1024 分块，T21 先例）+
  `eids[k]==nrange` 向量比较 + `tl.sum(axis=1)`，唯一命中 lane 使规约值与直接
  gather 逐位相等；全 constexpr 实值 stride/宽度（RS0/TS0/NROUTED/WIDTH 实值，
  TOPK/NSHARED 仍 pow2 pad）；全 int32 寻址（wrapper 四条域断言，T90 e6 边界
  证明纪律：tile 松弛 ≤1024、pad lane <2× 实宽）；i64 input_ids 经 i32 指针
  位转换读低半字（小端、token id 非负 <2^31，T87 bitcast 先例），kernel 内零
  i64 类型算子；wrapper 零 torch 计算（仅两次 `torch.empty` + 发射）；消除
  tensor-index gather（make_gcuir 拒收根因，17bf3fbc 实锤）与全部 i64 暂存。
  行间隔布局（stride(1)==1、stride(0)>width）经 constexpr 实值 stride 原生支持，
  内stride≠1 才做布局拷贝（T90 e6 round-2 契约类）。
- **新回归**（tests/test_hash_topk.py，已列 RELEASE_REQUIRED_TESTS）：
  重复/乱序 eids（match-reduce≡gather 语义）、宽路由 1536/2048 分块（部分尾
  tile + 恰好整 tile，eids 钉 1023/1024/末 lane 边界）、行间隔 logits+表、
  int32 ids + shared=0 边。本地无 GPU/CUDA，测试未在本机执行——数值验证走
  既有远端代理回执。
- **预注册门（原门保留）**：8/8 valid 且燧原 ≥2.0 保留 / ≥3.0 轴确认，其余
  7 成员字节冻结；同指纹失败连续 2 次关轴；NROUTED tile 上限按 T21 BLOCK≤1024
  分块。
- **分诊附加**：expectedAvgGain 0.08 = EvokeAgent 档（燧原 0.850→3.0，
  +2.15 单芯 ÷8=+0.27 均值）× 约三成成功率（make_gcuir 对向量比较+规约的
  接受度未知是主要折扣，c2flow 18.667 大概率窗口水位不采信）；发射前先用
  kernelgen MCP 失败神谕通道零额度预筛编译，预筛拒收则本轮跳过、发射额度让给
  T86 候选；同题落选的 C8（昆仑标量 eids+行向规约）是全清单唯一贴近 0.1
  有效性门槛芯的防线候选（榜上 kunlun 0.271、e7 同字节已滑至 0.1316，
  昆仑评测环境劣化在册 chip-rulesets.md:33；再滑破 0.1 则 T82 全题按
  README.md:33 判无效），本候选回执后无条件跟进或与重掷窗口合并考虑；
  09-23 发射（今日 0/30）。


## 2026-09-22 晚 E8 上膛（工作流试运行产出，人工补上膛）

- E8（`ff252260`，_enflame 重写）：kernel 内 one-hot match-reduce——整行线性读
  [NRTILE] + eids[k]==nrange 向量比较 + tl.sum(axis=1)，消灭 wrapper 的
  torch 预 gather 链（tid2eid[...].long()+gather+contiguous+to(int32) 共
  4-6 个 i64/非连续病理算子 = GCU CPU offload 根因，即 make_gcuir 拒收
  tensor-index gather 的 17bf3fbc 教训）；全 constexpr 实值 stride、全
  int32 寻址、grid=min(rows,12)+warps2、wrapper 零 torch 计算。
  测试进 RELEASE_REQUIRED（duplicate/unsorted eids、1536/2048 tiling
  边界、row-gapped、int32 ids）。
- 五元组：commit `ff252260`；ZIP `artifacts/competition/hash_topk/e8-ff25226/hash_topk.zip`
  SHA `509ae2d9b01050f24d45aa66042c8ad56142e67789f8cd03a5e04e9b9f401dcb`（3 成员）；回执
  `day5prep-20260921/hash_topk-e8/verification.json`（6 测试 0 失败，
  generic/enflame/kunlunxin 各 11 launch）。
- 预注册门：8/8 有效且燧原 ≥2.0 保留 / ≥3.0 轴确认；其余成员字节冻结。
  注：e8 在冲榜循环试运行中两轮评审未过系评审对象错绑（编排缺陷，
  v3.1/v3.2 已修），kernel 代码本身两轮 codex-review 无 kernel 级发现。

## 2026-09-23 E9 候选：燧原标量间接读取

- 瓶颈与单变量：e8 已删掉 Torch 预 gather，但对每个 top-k 槽仍扫描
  `num_routed` 个 logits 并做 one-hot 归约。e9 在同一 kernel 内逐槽标量
  `tl.load(router_logits + row * RS0 + eid)`，把读量从
  `O(topk × num_routed)` 降到 `O(topk)`；输出归一化、wrapper、GCU
  12-CTA/2-warp 几何保持。上游 SGLang `triton_hash_topk.py` 固定提交
  `d4dcce12` 用向量间接 load；我方 GCU 历史拒该形态，故仅试标量指针。
- 源码/验证 commit：`53627c98cf479a71b0f63debee68339d4185131b`；
  generic SHA-256 `d1f9520506b4590ee8585f0d652704b75d5d05d4c47a3e28976663c005515bf4`，
  enflame `c1a9e40cc2d8997ddeac8a046bcfb9232cbab122f88501f3edcea4d92f675b4b`，
  kunlunxin `567f94257f13b4b504beba5f053ce3df9fb5fb12f6997cc2e88d2844e911a7cd`，
  test `8e685f1219e4dd21e00e0e9eef9929137a34cacd5d5385985beafb21c5223624`。
  generic/kunlunxin 与 e8 ZIP 对应成员逐字节相同，只有 enflame 变化。
- ZIP：`artifacts/competition/hash_topk/e9-53627c9/hash_topk.zip`，11703 B，
  SHA-256 `55ede0abf8d9258199e9136b437cd74c7cef206a008d62164d522358be8a744a`；
  三成员 `hash_topk.py` / `hash_topk_enflame.py` /
  `hash_topk_kunlunxin.py`，打包器 `--verify-existing` 与 `unzip -t` 均通过。
- release：`artifacts/competition/hash_topk/e9-53627c9/verification.json`，
  SHA-256 `4f17bbcd1870d8e197665e5da6f2fd17a6a7cb0eeb683d53a0d1f14ad01c03b0`；
  日志 SHA-256 `247437ee0da717b84c2254e033519870c08caf97101bf49f3475c4ebd22d0a25`。
  RTX 5070 Ti / torch 2.13.0+cu130 / Triton 3.7.1，6 测试、0 失败、0 skip，
  generic/enflame 各 11 次真实 kernel launch。Black/isort/flake8 均通过。
  燧原目标为 `target-runtime-unverified`；KernelGen 当前 schema 不提供
  固定本仓源码字节的执行接口，不把改写代码的服务结果当作同源验证。
- `codex-review --commit 53627c98 --spec .../82-hash_topk.md`：
  gpt-6-sol/max，只读审查完成，未发现可确认的问题。预注册门：平台 8/8
  正确且各芯 ≥0.1；燧原 ≥3.0 且均值 >6.42825 才保留 e9，其他七芯
  若较 e8 已证读数回退 >5% 需核对平台水位；燧原编译/数值败或低于
  3.0 则回滚到 e8 已证 ZIP 字节，不自动重试本候选。

## 2026-09-23 E9 平台终态与回滚

- 实时 preflight 绑定账户 `15600308080`、SoulCoder、T82/e9、源码
  `53627c98`、ZIP SHA `55ede0ab…a744a`、release 回执 SHA
  `4f17bbcd…03b0`，一次性提交编号 **20431**；平台确认选择
  `hash_topk_enflame.py`。8/8 全部通过且每芯超过 0.1，均分
  **6.261475 < e8 TB 6.42825**，未换团队最佳。远端 ZIP 验签在提交
  响应中因未设置可信主机变量而显示 unavailable；本地 ZIP 已独立验签，
  平台所选成员与候选一致，不把 unavailable 写成远端通过。
- 逐芯：天数 13.3102、沐曦 4.3584、燧原 **1.3048**、海光 11.8238、
  昆仑 0.2314、华为 2.8020、A 6.9316、B 9.3296。相对 e8 燧原
  1.8538 下降 29.6%，远低于预注册 ≥3.0；标量串行/随机寻址在
  GCU 上输给了 one-hot 线性扫描，不能用读量阶数替代目标芯计时判断。
- 源码 `_enflame` 成员已从 e8 ZIP 原样恢复，SHA-256
  `e30e8d7616c45e87a972d5954e63b5da97b871f3d02af19b21821386c5c0f7bc`；
  与 e8 源码 commit `ff252260` diff 为空，回滚 commit `bf4fea57`。
  e9 ZIP 与回执继续留档，不重传候选。
