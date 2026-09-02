# Task 35 `rotary_embedding` 实验记录

```current
task: 35
operator: rotary_embedding
batch: 3
validity: valid
platform: 8/8(E8,5.961875x,rank9)
team_best_stage: e8
team_best_commit: e066a9e
team_best_speedup: 5.961875
sealed: no
next: E10 tile32 提交中(最后一发)
updated: 2026-09-02
```

状态:S0 候选就绪(首 screening 全绿)

## 契约锁定

- 签名:`reference(x, cos, sin, interleaved)`;`x [T,H,D]`
  bf16(兼容 fp16/fp32);`cos/sin [T, D//2]`
- 计算(题面 reference 忽略 interleaved,恒偶奇对):x1=x[...,0::2]、
  x2=x[...,1::2];o1=x1*c-x2*s;o2=x1*s+x2*c;交错重组,fp32 计算,
  转回 x.dtype
- 容差:fp32 1e-4 / bf16 1.5e-2 / fp16 1e-2;八芯;纯 Triton

## S0(2026-08-29,commit `5406218`)

- kernelgen `generate_kernel` 生成 + 三处契约修正:while→for range
  (已知编译雷)、恒偶奇对拆分(生成版 False 分支半宽拆分与题面
  reference 不符)、store 显式 cast。
- 结构:每 program 一 (t,h) 行,constexpr HEADS 除数,偶奇 strided
  读写,1D capped grid-stride,int32。
- screening(gpu:/tmp/t35.Pw0Tob):unittest 5/5 OK;bench 7/7;
  代理加速比:4096×32×128 **11.79x**、128×4×192 4.56x、
  256×8×128 4.20x、65536×1×64 4.13x、16×32×128 4.03x、
  1024×16×128 3.72x。
- ZIP `s0-5406218/rotary_embedding.zip`,SHA
  `7f768cc3ef4896a6ad4c3326a2c9cd37406d034d209ecdc11fd604cca8482b81`,
  单成员。

### 跨芯风险

- 无超越函数、无归约、无 dot;strided 读写是常规模式;风险面小。
- 小 shape launch 主导(代理 ~4x 与 T30 平铺同水位)。

### S0 提交记录(2026-08-29 17:1x CST)

screening 字节与 commit blob 逐项一致(晋级验签 ✓);preflight 全过
(tid `s2t1op035`,额度 15/30 消耗 1 → 14/30);单次 confirm 提交,
评测入队。

### S0 平台终态(sub 6341,2026-08-29 17:2x CST)

**8/8 全过,valid,平均 4.66725x**(榜首 MakeYUNAGreatAgain 5.8830x,
-20.6%,首投即达标)。

| 芯片 | speedup |
| --- | ---: |
| tianshu | 9.986 |
| card_b | 8.987 |
| card_a | 8.354 |
| haiguang | 4.936 |
| muxi | 3.793 |
| huawei | 0.715 |
| kunlunxin | 0.298 |
| enflame | 0.269 |

短板三芯与 T29/T33 同画像;E1 假设:stride-2 偶奇双读在弱芯代价高,
改整行连续读 + 寄存器 `tl.reshape/tl.split/tl.join` 解交错。

### E1 负结果(未提交,2026-08-29)

连续整行读 + `tl.reshape/tl.split/tl.join` 寄存器解交错候选:代理
全面回退(4096×32×128 11.79→9.31x,小 shape 4.20→3.34x,余 -3%)。
NVIDIA 上 stride-2 访存硬件合并良好;且 T30 的逐元素 gather 在
燧原 8.7x/昆仑 5.8x,反证"非连续访问是弱芯瓶颈"。假设不支持,
不提交。T35 弱芯(燧原 0.27/昆仑 0.30/华为 0.72)无已证杠杆,
本题收敛:S0 = 团队最佳 4.6673x(-20.6%)。

### E1:四头瓦片 + cos/sin 复用(sub 6xxx,2026-08-31 00:1x CST)

- codex 收窄后的可迁移模型:真实全局数据复用(cos/sin 只依赖
  (token,col),每瓦片载一次广播到 4 头)+ 保持 stride-2 访问;
  昆仑 vendor 钉死 S0 字节兜底,新 generic 只达其余七芯;
- commit `581fc8c`,ZIP `8ea6dc5a…`,2 成员;
- 代理:+3%~+62%(4096×32×128 11.78→15.59x、1024×16×128
  3.76→6.09x)全 shape 无回退,unittest 5/5;preflight 全过。

### E1 平台终态(sub 6982)与 E2(2026-08-31 01:1x CST)

- **E1 七芯全过且大涨**:沐曦 3.79→**6.43(+70%)**、海光 4.94→
  **8.97(+81%)**、天数 9.99→10.33、card_a 8.35→9.59、card_b
  9.10;华为 9 case NaN(99.6%,[4,64] 2D 广播瓦片触发昇腾
  lowering 问题)——cos/sin 复用模型七芯验证成立;
- E2 = 华为钉 S0 旧字节 vendor(commit `7ed663c`,ZIP
  `90e429be…`,3 成员);预计 8/8 且平均 ~6.7(现 4.667)。
- 额度 27/30。

### E2 平台终态(sub 6990,2026-08-31 01:4x CST)

**8/8 valid,平均 5.82525x —— 新团队最佳(S0 4.6673 → +24.8%)。**

| 芯片 | S0 | E2 | 变化 |
| --- | ---: | ---: | ---: |
| tianshu | 9.986 | 10.385 | +4% |
| **muxi** | 3.793 | **6.448** | **+70%** |
| **haiguang** | 4.936 | **9.332** | **+89%** |
| card_a | 8.354 | 9.917 | +19% |
| card_b | 8.987 | 9.148 | +2% |
| huawei(S0 vendor) | 0.715 | 0.679 | 持平 |
| enflame | 0.269 | 0.397 | +48%(新 generic) |
| kunlunxin(S0 vendor) | 0.298 | 0.295 | 持平 |

- cos/sin 复用 + 四头瓦片模型六芯兑现;剩余短板燧原 0.40/昆仑 0.30
  /华为 0.68(后两者为钉死的 S0 兜底字节)。
- 距榜首 EvokeAgent 8.488 收窄至 **-31.4%**;额度 26/30。

### E3:华为 1D cos/sin 复用 vendor(sub 7220,2026-08-31)

- 假设:E1 四头瓦片的 2D 广播触发昇腾 NaN;保留 cos/sin 复用机制
  (每 4 头载一次,六芯 +48~89% 的来源)但全部 1D 化——per-head
  static_range(T33 e1 证据:昇腾对 static_range 摊销中性),
  无任何 2D 广播;commit `5fd1a58`;
- screening(gpu:/tmp/flagos-catchup.NOF9kN):unittest 5/5;
  代理上 vs S0 钉字节互有胜负(±20%,NVIDIA L2 掩盖复用收益,
  华为专属赌注不阻断);release(gpu:/tmp/flagos-rel2.bskkJw)5/5;
- ZIP `e3-5fd1a58`,SHA `c7436dd9…`,3 成员;preflight 15/30。

### E3 终态(sub 7220)

**8/8 valid,平均 5.84575x —— team best(E2 5.82525 → +0.35%)。**
华为 0.679→**0.7746(+14%,假设兑现:NaN 规避且复用生效)**;
燧原 0.399(新 generic 字节未变,稳定低值);昆仑钉死 0.289 持平。
**知识:昇腾 2D 广播 NaN 的修法 = 机制保留、访存全 1D 化。**
距榜首(EvokeAgent 8.4884)-31.2%,结构性差距不变。

### E4:燧原宽头瓦片 vendor(sub 7251,2026-08-31 16:4x CST)

- 假设:E2 generic 的 [4, HALF_DIM] 瓦片仅 ~2KB,低于燧原偏好的
  每 program 工作量(T33 +14 倍/T39 BLOCK 4096 家族);vendor =
  头瓦片自适应放宽到 16(min(next_pow2(H), 16)),kernel 体不变,
  commit `50dc5a0`;
- screening(gpu:/tmp/flagos-t35e4.C2YCvf):unittest 5/5,lint 三项
  过(修正一处 black 折行);代理 gen4 持平(±15% 记录用)。

### E4 终态(sub 7251):燧原宽瓦片证伪

- 8/8 valid,avg 5.8231(< e3 锚点 5.8458,team best 保持 e3);
- **燧原 0.399→0.401 持平——宽瓦片模型族首个反例**:本题瓶颈
  不在瓦片尺寸,疑在 stride-2 偶奇访存(与 T30 燧原 8.7x 的
  gather 反例互证);燧原轴关闭,T35 收盘于 e3 5.8458x;
- 其余七芯与 e3 读数一致(±2% 噪声,华为 0.72/昆仑 0.29 钉字节)。

## E5:generic head 复用宽度扫描(2026-09-01 07:1x CST,负结果)

- 官方结构先验：[Liger RoPE](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/rope.py)
  与 [vLLM MRoPE](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/rotary_embedding/mrope.py)
  均按 token 在一个 program 内覆盖全部 heads、只载一次 cos/sin；E4 只
  证伪燧原 vendor 宽瓦片，未覆盖 generic。因此以 e3 `HEADS_TILE=4`
  为 base，补扫固定 1/2/8/16，只有固定扫描出现清晰 crossover 才允许
  增加自适应分派；
- 远端 `gpu:/tmp/flagos-t35-tile.tp5CHs`，RTX 5070 Ti、PyTorch
  `2.13.0+cu130`、Triton `3.7.1`；wrapper-inclusive 六 shape，每候选
  每 shape 7 组真 AB/BA，整轮正反序各一次。candidate/base 时间比
  geomean：t1 `1.2143`(-17.65%)、t2 `1.0320`(-3.10%)、
  **t8 `1.0097`(-0.96%)**、t16 `1.0748`(-6.96%)；
- t8 最接近 base，但 H32 仅 +0.39~0.77%，`1024x16x128` 回退
  4.38%；t16 在 `16x32x128` 回退 16.40%。没有候选达到综合 +5%，
  也没有可构造自适应的清晰 crossover，故不扩扫 t32/full-head；
- base 与 t1/t2/t4/t8/t16 官方 unittest 全部 5/5 OK。D128/H32
  t1/t2/t4/t8/t16 为 30/33/39/40/40 regs，D192/H4 的 t16 升至
  72 regs；全候选均 0 spill、shared=0、4 warps、3 stages；
- base/t8/t16 SHA-256 分别为
  `a99d4b85adea252a738d3a38f1e7c8493604a9538afd08dde936cf7e92754d78`、
  `b286cf0afebdc70aab38497a5760a6fdbedc462ba8d50cc1a539fa89adf3b370`、
  `aaab519274f50d6f03b110c2cce8329539d3f670838e7ba0aec9c04a064ed0ee`；
  correctness/bench/resources 日志 SHA-256 分别为
  `b88a9484ed9e1def663503e8e10eedeb2288138913572c8f0a9624992f389369`、
  `0c2417ca27a61dd78a9f2d7385b0dfaa18a5f1161567874ab6639775d34d61e7`、
  `d0928b9bf8e32acfcc610676dc6a4c93d4932e0e985a82e894d3c8b5e34597d6`。

结论:官方全-head 复用结构在本题 shape/布局上不能迁移；`HEADS_TILE=4`
仍是代理局部最优。实时榜首 `8.8534x` 对 e3 `5.8458x` 的 51.45%
差距无法由该参数轴解释，不改源码、不消耗额度，保持封存。

## E6:访问模式轴——连续读 + tl.split/tl.join 解交错(2026-09-02)

**靶点修正**。e3 平台逐芯:沐曦 6.43 / 海光 8.97 / 天数 10.33 /
card_a 9.92 / card_b 9.15 / 华为 0.77 / 燧原 0.40 / 昆仑 0.29——与榜首
的差距几乎全部在三个钉死芯;E4 账本已假设燧原瓶颈在 stride-2 偶奇访存。
本轴只改 vendor,generic 冻结 e3 字节。

### E6a 探针:generic 解交错(未提交,代理门否决)

- 新 generic:整行一次连续 load → `tl.reshape [H,HALF,2]` → `tl.split`
  拆偶奇 → 计算后 `tl.join`+reshape 连续 store(仅访问模式变化);
- screening(gpu:`/tmp/flagos-rotary-e6.aHPpxC`,RTX 5070 Ti、Triton
  3.7.1):unittest 5/5(3 dtype × 8 shape,含 vendor 变体);
  paired AB/BA 五轮 15 组:仅 `4096x32x128` 半精度 +5.5~6.4%,其余
  ±1%,**geomean `1.0046` < 预注册 +10% 门,否决提交**;两版 kernel
  均 0 spill(new 28~40 regs vs old 28~40);
- 负结果知识:NVIDIA 编译器已把旧版相邻的 `x_base-1`/`x_base` 双载融
  合成宽访问,generic 五芯无该轴头寸;generic 树回滚 e3 字节。

### E6b 正式包:三 vendor 解交错(single variable per chip)

- `_ascend`(e3 0.77):保持 1D 无广播形态与 cos/sin 复用,每 head 整行
  连续 load + split 解交错 + join 连续 store;
- `_enflame`(新 vendor,当前走 generic 0.40):e3 generic tile 形态,
  访问模式换连续 + split/join;
- `_kunlunxin`(e3 0.29):保持单行一 program,整行连续 load + split/join;
- 远端 screening 同目录:unittest **5/5 OK**(generic e3 + 三新 vendor
  全部变体);三 vendor 字节 SHA-256:
  ascend `b188103d…25978`、kunlunxin `1e9e46da…3114`、
  enflame `1957dd66…74e9`,generic 保持 `a99d4b85…4d78`;
- MCP 华为 specialize+注入差分预检(候选 `b188103d…25978`,返回
  `4b659470…573`,输出存 `/tmp/kg-t35e6-ascend.json`):**MINIMAL_DIFF
  (代码行仅 +`import torch_npu`,其余为注释删减)= 零适配需求信号**
  (与 T33 tile16 平台 +28.8% 前同款);燧原/昆仑无通道,标
  `static-unverified`。

### 预注册门与 stop gate(E6b,提交前登记)

- 平台门:8/8 valid、每芯 ≥0.1x;
- vendor 芯目标:华为 ≥0.85(e3 0.77 +10%)、燧原 ≥0.50(+25%)、
  昆仑 ≥0.35(+20%);任一 vendor 芯低于其 e3 基线 −10% 即回滚该芯;
- generic 五芯(天数/沐曦/海光/国际 A/B)字节未变,读数应与 e3 一致
  (±2% 视为平台噪声);
- stop gate:任一 vendor 芯数值/编译失败 → 该芯回滚 e3 字节,同指纹
  不重投;昆仑若再现 1830s compile-worker 崩溃 → 计入崩溃族证据,
  不重投、仅工单。

### E6b 平台提交(sub 8226,2026-09-02 14:0x CST)

- 实时 preflight 核对账号/团队/batch 3/Task 35/tid `s2t1op035`、
  source=verification commit `44e9c5f`、stage e6、4 成员、ZIP
  `/Users/bytedance/ccc/flagos/artifacts/competition/rotary_embedding/e6-44e9c5f/rotary_embedding.zip`
  SHA-256 `58d349ffebdff5d292e68f826a9157c420cb59a002f44ee4451460ab50605280`
  全部匹配;intent nonce `45157a78a042ffbb583590c2fac824aa` 一次性消费,
  正式提交成功,submission `8226`,八芯入队;该 intent/ZIP 已 `submitted`,
  禁止重试;
- release 证据:git 对象临时目录 `gpu:/tmp/flagos-rel-t35e6.vcms6f`,
  远端六文件 SHA-256 与本地/ZIP 成员逐项一致,unittest 5/5
  (`release.log`);screening 证据 `gpu:/tmp/flagos-rotary-e6.aHPpxC`。

### E6 终态(sub 8226):昆仑 compile-worker 崩溃,7/8 invalid

- 七芯全过:天数 10.523 / 沐曦 6.364 / 燧原 **0.8744(+119%)** /
  海光 9.4984 / 华为 **1.663(+116%)** / card_a 9.9988 / card_b 9.1364;
- **昆仑 execution 1833537ms 后 FAIL(空 error)**——与 1830s
  compile-worker 崩溃族指纹一致(第 16+ 例),触发载体为昆仑 vendor 的
  `tl.reshape/tl.split/tl.join` 结构:XPU 编译器无法 lowering 该形态,
  计入崩溃族证据;
- 预注册 stop gate 触发:昆仑 vendor 回滚 e3 已验证字节(SHA
  `ceca925b…fd8`,平台 0.29 通过前科),华为/燧原新字节与 generic
  冻结字节保持——组 E7 应急包落袋七芯增益。

## E7:应急包——昆仑回滚 e3 + 华为/燧原解交错保持(2026-09-02)

- 单一变化:E6 包中的 `_kunlunxin` 换回 e3 字节(平台已验证
  通过),其余三成员与 E6 逐字节相同;
- 预期:七芯读数复现 E6(±2% 噪声),昆仑回到 0.29 附近,平均
  ≈ 6.04(+3.4% vs e3 5.8458);
- 预注册门:8/8 valid、平均 > 5.8458 即新团队最佳;stop gate:昆仑
  若以 e3 字节再现崩溃族指纹(排除结构因素)→ 纯平台故障,计工单
  不重投;七芯任一非噪声回退 → 该芯字节回滚并复盘。

### E7 平台提交(sub 8253,2026-09-02 14:5x CST)

- preflight 核对 source=verification commit `c54de18`、stage e7、4 成员
  (generic `a99d4b85`、ascend `b188103d`、enflame `1957dd66`、kunlun
  `ceca925b`)、ZIP SHA-256
  `7fe3f45207afecc3f7914bb0c6b0ec07e14db830685cf1b399319ec9f3525756`;
  nonce `3a4cd0b0a8adce324ae37568c55706f1` 一次性消费,submission
  `8253`(daily seq 9),额度 22→21/30;
- release 证据:git 对象目录 `gpu:/tmp/flagos-rel-t35e7.XXXXXX`,
  unittest 5/5,远端成员 SHA 与 ZIP 逐项一致。

## E8:tile8 探针——解交错结构上的瓦片宽度复扫(2026-09-02)

- 背景:E4 燧原宽头瓦片(→16)证伪的前提是 stride-2 访存;E6 换连续
  访存后瓶颈结构已变,瓦片宽度轴重新开放;
- 单变量(每芯独立):`_enflame` HEADS_TILE 4→8、`_ascend` 摊销头数
  4→8;generic 与 `_kunlunxin`(e3 字节)冻结;
- screening:`gpu:/tmp/flagos-rotary-e6.aHPpxC/{enflame,ascend}_t8.py`,
  3 dtype × 10 shape(含 H=5/7 非整瓦片)各 30/30 数值全过;
  ascend tile8 MCP specialize+注入差分 **MINIMAL_DIFF**(/tmp/
  kg-t35e7-ascend-t8.json);字节:enflame_t8、ascend_t8(见下);
- 预注册门:8/8 valid 且平均 > E7 落袋值;华为 ≥1.8(1.663 +8%)、
  燧原 ≥0.95(0.8744 +9%)才保留,否则该芯回滚 E7 字节;stop gate:
  任一芯数值/编译失败即回滚该芯,同指纹不重投;昆仑字节不动,若其
  E7 侧异常与本包无关。

### E8 平台提交(sub 8255,2026-09-02 15:1x CST)

- preflight 核对 source=verification commit `e066a9e`、stage e8、4 成员
  (generic `a99d4b85`、ascend tile8 `0fc3048e`、enflame tile8
  `a65d162b`、kunlun e3 `ceca925b`)、ZIP SHA-256
  `91444278b4aaf3d664d924c1e0a1f260bd16b8e4eaea7a2f1d6886ca6fa450dc`;
  nonce `3da4e777673bec1392e71c9447bd24f0` 一次性消费,submission
  `8255`(daily seq 10),额度 21→20/30;
- release 证据:git 对象目录 `gpu:/tmp/flagos-rel-t35e8.XXXXXX`,
  unittest 5/5,release log SHA-256 `82303ae0…cb10`;
- E7(sub 8253)与 E8(sub 8255)同时在评;两包除 ascend/enflame
  vendor 外字节相同,逐芯结果可互相对照归因。

### E7/E8 终态(2026-09-02 15:0x)

- **E7(sub 8253):8/8 valid,平均 5.90115**——昆仑 e3 字节如预期通过
  (0.2942),七芯增益落袋;但同日即被 E8 超越(is_team_best=false);
- **E8(sub 8255):8/8 valid,平均 5.961875,新团队最佳(is_team_best
  =true)**,榜首差距 5.8458→5.9619(-53% 对 c2flow 12.7588);
- 同窗成对归因(消除跨窗口方差):E7→E8 逐芯——华为 1.5472→1.7554
  (**tile8 +13.4%**)、燧原 0.881→1.0088(**tile8 +14.5%**),generic
  五芯 ±2% 噪声(天数 10.51→10.43、card_b 8.71→8.73 等),昆仑冻结
  0.294→0.296;两芯均过预注册保留门,tree 保持 E8 字节;
- 跨窗口方差注记:E6→E7 同字节(华为/燧原/generic)读数漂移 -2~
  -7%(华为 1.663→1.5472),跨提交比较必须用同窗成对;
- 累计:华为 0.77→1.7554(+128%)、燧原 0.40→1.0088(+152%),
  题目平均 5.8458→5.9619(+2.0%),rank 10→9。

## E9:tile16 探针(2026-09-02)

- tile 曲线仍在上升段(4→8 同窗 +13.4%/+14.5%),单变量推进
  `_enflame`/`_ascend` HEADS_TILE 8→16;generic 与 `_kunlunxin`(e3)
  冻结;
- screening:3 dtype × 12 shape(含 H=9/33 非整瓦片)各 36/36 数值
  全过;ascend t16 MCP specialize+注入差分 MINIMAL_DIFF
  (/tmp/kg-t35e9-ascend-t16.json);
- 预注册门:8/8 valid 且平均 > 5.961875;同窗归因华为/燧原任一芯
  相对 E8 回退 >3% 即该芯回滚 E8 字节;stop gate:数值/编译失败即
  回滚,同指纹不重投。目标:冲击 #8 ChipVoyager 6.173(需 +3.5%)。

### E9 平台提交(sub 8258,2026-09-02 15:14:38 CST)

- preflight 核对 source=verification commit `dc66469`、stage e9、4 成员
  (generic `a99d4b85`、ascend t16 `26a3f916`、enflame t16 `34ac29b0`、
  kunlun e3 `ceca925b`)、ZIP SHA-256
  `219e8b393883430f653a17df13488ca66d5ea53a324b44d7b73323cb15de7358`;
  nonce `8a17e833ab22c66c0913829d63ef4139` 一次性消费,submission
  `8258`(daily seq 11),额度 20→19/30(重复 confirm 经核验为幂等读,
  未产生第二个提交);
- release 证据:git 对象目录 `gpu:/tmp/flagos-rel-t35e9.XXXXXX`,
  unittest 5/5,release log SHA-256 `29ae8f9d…ba27`。

### E9 终态(sub 8258):tile16 目标芯续涨,平均被 generic 方差吃掉

- 8/8 valid,平均 5.8877,**未超 E8 5.961875**(is_team_best=false);
  E8 保持团队最佳,额度 19/30;
- 同窗归因(E8→E9):华为 1.7554→1.8802(**tile16 +7.1%**)、燧原
  1.0088→1.147(**+13.7%**)、昆仑冻结字节 0.296→0.327(+10%,纯方差);
  generic 冻结字节读数漂移:天数 10.426→9.901(-5.0%)、海光 9.535→
  9.248(-3.0%)、card_a -1.6%、muxi/card_b +0.5~0.7%——**同字节跨提交
  方差 ±0.3~0.5 平均量级,超过 vendor 芯单轮增益**;
- 判定:tile 曲线在两目标芯仍上升(4→8:+13~15%;8→16:+7~14%),
  E9 平均失利属 generic 方差主导,机制未证伪;按方差追击预算再发
  一发 tile32(e10),若仍未超 5.9619 则 T35 封存(连续两次方差失利
  即关轴)。

## E10:tile32 最后一发(2026-09-02)

- tile 曲线同窗证据:4→8 +13~15%、8→16 +7~14%,两目标芯未饱和;
  单变量推进 `_enflame`/`_ascend` HEADS_TILE 16→32,generic/kunlun 冻结;
- screening:3 dtype × 12 shape 各 36/36 数值全过;ascend t32 MCP
  specialize+注入差分 MINIMAL_DIFF(/tmp/kg-t35e10-ascend-t32.json);
- 预注册门:8/8 valid 且平均 > 5.961875(E8)才保留字节;**未超即
  T35 封存**(连续两次 generic 方差失利关轴),树回滚 E8 团队最佳字节;
  stop gate:任一 vendor 芯数值/编译失败回滚该芯,同指纹不重投。
