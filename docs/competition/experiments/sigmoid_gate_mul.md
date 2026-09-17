# Task 75 `sigmoid_gate_mul` 实验记录


## 2026-09-13 E2 平台终态：8/8 VALID 2.6065x 新 team best（燧原 +113%）

- **enflame 0.827 → 1.7606（+113%！）**；tianshu 经 1024 vendor 恢复
  （4.191→4.721，s0 4.769）；其余稳定。均值 2.6065 新 TB，距榜首
  Albedo 3.054 缩到 0.45（原 0.52）。
- **配方二连证**：燧原 elementwise = grid 封顶 24 + BLOCK 4096
  （T73 e3 +92% / T75 e2 +113%）。含天数分芯 vendor（1024）的
  "双赢组合"打法成立。

```current
task: 75
operator: sigmoid_gate_mul
batch: 5
validity: valid
platform: completed(16574,e10,8/8,2.89673333x微幅新TB;华为+15.6%)
candidate_stage: e10
team_best_stage: e10
team_best_speedup: 2.89673333
sealed: no
next: Ascend direct候选代理1.01924x未达1.05且fp16稳定回退，不提交；TB保持e9
updated: 2026-09-17
```

## 契约与实现（S0）

- 完整题面：[Task 75](../tasks/batch-5/75-sigmoid_gate_mul.md)（2026-09-12 晚新增五题之一）。
- flat 1024-lane x*sigmoid(gate); SGLang triton_sigmoid_gate_mul form。
- 核心计算 Triton，无 fallback；八芯 0.1x；截止 2026-09-17 19:59:59。

## 不可变身份

- source / verification commit：`e6b450fd4beb001255079ade929d3b2de51796d6`（五题同批提交）。
- source SHA-256：`df083ec35d96b1f3d08a7db4fa4f051d5fee6e747ab30bfe068276634c7f06af`。
- test SHA-256：`1f8d9b035c8706c9c98f6c186924a3d158f4ae40d5c5dbdb203853635cdb481a`。
- ZIP：`artifacts/competition/sigmoid_gate_mul/s0-e6b450f/sigmoid_gate_mul.zip`，SHA-256 `f608dbaf766c96df37e60f5d0c6a6a7ef89ffcd789f617b23165c479eee265ff`（单成员 `sigmoid_gate_mul.py`）。
- release 回执：`artifacts/competition/batch5-new5-validate-20260913/sigmoid_gate_mul/verification.json`，
  SHA-256 `93ea7bbb4a56b52982b89d21447b4afe9529bad5756c9d38af48d6c23aeae1d5`；日志 SHA-256 `a8f10c821397f48f969b52b395b846fe627e6466eb2a83ae7d80eda01914a24b`。

## 验证状态

- screening 多轮门禁拦下并修复的缺陷已记录于提交说明；
  最终 release：0 失败/错误/skip，非空 kernel launch，NVIDIA 代理范围。
- 所有八芯目标 `target-runtime-unverified`；裁决权在平台。

## 2026-09-13 E1：BLOCK 1024→4096（13878）8/8 valid 2.511（API 真值订正）

**真值**：s0(13767) enflame 0.827 / kunlun 0.295 / huawei 1.028 / tianshu 4.769
（avg 2.5382）；e1(13878) enflame **1.417（+71%）** / kunlun 0.262 /
huawei 1.369 / tianshu 4.191（avg 2.5107）。

**订正结论**：4096 对燧原 **+71%**（与此前"回退"记载相反——同样是
转录错误）；但天数 -12%（4.769→4.191）吃掉净收益，均略低于 TB。
e2 假设：**燧原再推 8192 + 昆仑 no-loop 形态**（T61 no-loop 读 1.29
vs 本题 0.26）。

## 2026-09-14 E3 候选就绪：昆仑 no-loop 探针（Codex top1 队列 3 号槽，待发射）

- 逐芯榜单修正认知：T75 昆仑其余各队 0.78-0.86，我方 0.263；T61
  "no-loop 已证"前提被 Codex 证伪（T61 generic 含 grid-stride）——
  本发是降级后的未验证假设探针，单变量=去 runtime 循环（一 program
  一 tile，BLOCK=4096、数学与 fp32 契约字节不变）。
- source commit：`185170f1…`；ZIP `e3-185170f`，4 members，SHA-256
  `09293f3a1dde27b21efac65715803d6d74fa00da337bad5596d52394fa880321`。
- release 回执 `batch5-submit-20260914/sigmoid_gate_mul/verification.json`
  SHA-256 `765dc87d3557c7a36559b0d7c856e18274072853042c27447bf842a1dca18b7a`
  （--proxy-vendor kunlunxin：vendor 17 次真实 launch 全矩阵 0F0E0S）。
- 预注册晋级门：**昆仑 ≥ 0.50**；<0.35 则暂停 no-loop 复制链
  （T71/T73 不再跟发）。

## 2026-09-14 E3 平台终态：8/8 VALID 2.696x 新 TB（submission 14570）——no-loop 假设验证

- **昆仑 vendor 被选中、passed 0.8994**（generic 循环形态 0.263 →
  3.4x）——no-loop 探针兑现，"runtime 循环形态是昆仑 streaming 低
  读数主因"升级为单题实证；队列 6/7 号槽（T71/T73 昆仑）解锁。
- 逐芯：天数 4.7862 / 沐曦 2.6003 / 燧原 1.7565 / 海光 4.0185 /
  **昆仑 0.8994** / 华为 1.3183 / A 3.2262 / B 2.9627 → 均值 2.696
  （前 TB 2.607）。榜首 GuanghuLab 3.688，差 0.99。
- 下一轴：燧原 8192（队列 9 号，leader 7.84 证明可达）与昆仑带宽
  剩余差距待逐芯榜单刷新后评估。

## 2026-09-14 E4 候选就绪：燧原 BLOCK 4096→8192（Codex 队列 9 号，待发射）

- 单变量：配方宽度轴推到 8192（该轴在本芯单调为正：1024→4096 曾
  +71%）；leader GuanghuLab 燧原 7.84 vs 我方 1.76 证明余量存在。
  cap 保持 24。
- source / verification commit：`e49a7a5…`；ZIP `e4-e49a7a5`，
  SHA-256 `0afa3136ee787eb5a92336e6e94337fe5b4ced1a8ac3187b2837376b6f65a467`。
- release 回执 `batch5-submit-20260914/sigmoid_gate_mul_e4/verification.json`
  SHA-256 `6f90a0d96c6a4ac92d05a51d3656ca6133e145e1dc44e668129b46a9a87ca663`
  （--proxy-vendor enflame：vendor 17 launch 0F0E0S）。
- 预注册晋级门：**燧原 ≥ 2.2**；无增益则宽度轴封顶（不平推 T71/T73）。

## 2026-09-14 E4 平台终态：8/8 VALID 2.739x 新TB（submission 14598）——8192 边际为正但未过门

- **燧原 2.1562**（4096 的 1.7565 → +23%）：宽度轴单调性延续但边际
  递减，差预注册门（≥2.2）一线——按纪律**不平推 T71/T73 的 8192**，
  宽度轴就此封顶。均值 2.739 > e3 的 2.696，TB 更新为 e4。
- 逐芯：天数 4.7794 / 沐曦 2.6066 / **燧原 2.1562** / 海光 3.9015 /
  昆仑 0.9003 / 华为 1.3999 / A 3.258 / B 2.9127。榜首 GuanghuLab
  3.688，差 0.95。
- 剩余燧原缺口（2.16 vs 7.84）已非宽度轴可及，需他队形态情报或
  燧原侧 profiling；今日收口守 TB。

## 2026-09-14 E5 候选就绪：燧原 i32 地址链探针（Codex 队列 1 号，待发射）

- 单变量：计分路径 offset 链去 i64（宿主按 numel<2^31 选 i32 kernel，
  ≥2^31 保留 i64 kernel——契约不收窄）；BLOCK 8192/cap24/数学不变。
- source / verification commit：`1bb5ef4…`；ZIP `e5-1bb5ef4`，
  SHA-256 `68c6755dc345b8ccebc2a4d6ab83ff28e10aac3a52be0ba4efc7980ce18aad5a`。
- release 回执 `batch5-submit-20260914/sigmoid_gate_mul_e5/verification.json`
  SHA-256 `4e218b82541b9c4dab96cc27fe7426941b3b418a785750e6b3ae74939115a125`。
- 预注册晋级门：**燧原 ≥ 2.6**；未过则关闭"显式 cast 大税"假设，
  禁止批量迁移其它 vendor。

## 2026-09-14 E5 平台状态：已提交，判决查询被 token 过期阻塞

- submission 已成功（state=submitted，intent 记录 14597 区间、
  file_url sha 198b2f69…）；随后 status 查询返回 HTTP 401（token
  过期）。重新认证需邮箱/手机验证码，agent 无法自动完成。
- 续接动作：用户重新 `platform_cli.py auth` 后以
  `status --race 782kzq4m --batch 5 --task 75 --operator sigmoid_gate_mul`
  读取 e5（i32 地址链探针）终态；预注册门 燧原 ≥2.6。

## 2026-09-14 E5 平台终态：8/8 VALID 2.7366x ≈ TB（submission 14614）——i64 税假设证伪

- **燧原 vendor passed 2.1487**（e4 i64 形态 2.1562，-0.4% 噪声级）
  ——预注册门（≥2.6）未过：GCU 上 i64 地址算术无显性 emulation 税，
  "显式 cast 大税"假设关闭，禁止向其它 vendor 批量迁移。
- 均值 2.7366 与 e4 的 2.7393 在窗口噪声内；TB 保持 e4（2.7393）。
  收盘守榜。

## 2026-09-14 E6 候选就绪：燧原 BLOCK 16384 豪赌（Codex 菜单 E，≤1 发，待发射）

- 8192 的 +23% 是宽度轴最后余量；16384 一次封顶（MAX 32768 之内但
  寄存器/UB 风险真实）。i32/i64 双 kernel 结构保持。
- source / verification commit：`425ef069…`；ZIP `e6-425ef06`，
  SHA-256 `65275bbcf4c35ea4932a802ff86cf88d0284173fc5caf37a869a7379fde2fbb3`。
- release 回执 `batch5-submit-20260914-finale/sigmoid_gate_mul-t75e6/verification.json`
  SHA-256 `2865607548ea871eac5125e2ae4fc4fa6a88aa188bc41ec17e08ddd6ec531f0f`
  （enflame vendor 17 launch 0F0E0S）。
- 预注册晋级门：**燧原 ≥ 3.6 且均值 > 2.9075**；编译/正确性失败或
  不足以升位即封顶，不试 32768、不平推。

## 2026-09-14 E6 平台终态：8/8 VALID 2.8277x 新TB（submission 14773）——16384 +30%

- **燧原 vendor passed 2.7984**（8192 的 2.1562 → **+30%**，宽度曲线
  仍在爬：1024→4096 +71% → 8192 +23% → 16384 +30%）——但差预注册
  门（≥3.6 且均值 >2.9075）未过：按纪律**封顶，不试 32768、不平推
  T71/T68**。均值 2.8277 > e5 2.7366，新 TB = e6。
- 逐芯：天数 4.7847 / 沐曦 2.6221 / **燧原 2.7984** / 海光 4.0068 /
  昆仑 0.9001 / 华为 1.3463 / A 3.2022 / B 2.9605。榜首 GuanghuLab
  3.6884，差 0.861；其燧原 7.84 与我方 2.80 的剩余差距仍是未知形态。
- 注：宽度轴收益递增而非递减是重要情报——若明日 consultation 解禁，
  32768（官方 MAX_BLOCK_SIZE）是最后一个未试档位。

## 2026-09-14 E7 候选就绪：燧原 BLOCK 32768（官方 MAX 最后一档，待发射）

- 宽度曲线 1024→4096 +71%→8192 +23%→16384 +30%（不降反升）；
  32768 = 官方 MAX_BLOCK_SIZE 内最后一档。代理 32768 编译+全矩阵
  已过（34 launch 0F0E0S）——寄存器/UB 风险未成真。
- source / verification commit：`925b3548…`；ZIP `e7-925b354`，
  SHA-256 `f633e77483e88c561d1cd0562955a57ede1e066f067bf41cf70a63613bf47d00`。
- 预注册晋级门：**燧原 ≥ 3.6 且均值 > 2.9075**；任何失败即最终封顶。

## 2026-09-15 16:25 E7R 发射：uncertain intent 解锁 + 注释载体（评测中）

- 09-14 e7 的 uncertain intent 阻塞同 tuple 新 nonce（预检实测确认）；
  按"新 ZIP SHA"纪律制作注释载体 `57573def`（执行字节=e7 32768 档，
  仅 enflame vendor 头注释 +4 行），ZIP `e7r-57573de`，
  SHA-256 `0621440557408edeb4fc2778cb4fad7d9a1ed8c6f04710c0185e8539152739a9`。
- release 回执重出（GPU 通道恢复当日）：
  `artifacts/competition/batch5-verify-20260915/sigmoid_gate_mul/`
  （exit 0，generic+燧原 vendor 各 17 launch，0 skip），回执 SHA-256
  `4529e9df1c8aff1d48625866c694d940a8b41885d5a1a5244b37b5575b5c0912`。
- submission **15215** 已提交（state=submitted，额度余 28）。
  预注册门不变：**燧原 ≥3.6 且均值 >2.9075**；任何失败即最终封顶。
- 旧 e7 uncertain intent 由平台侧自然超时/对账，不再操作。

## 2026-09-15 21:00 E7R 平台终态：8/8 VALID 2.8865x 新 TB —— 宽度轴最终封顶

- 判决（21:00 榜单确认）：燧原 **3.3**（e6 16384 读 2.80，+18%）、
  天数 4.8 / 沐曦 2.6 / 海光 4.0 / 昆仑 0.9 / 华为 1.3 / A 3.2 / B 2.9；
  均值 **2.8865**（e6 2.8277，+2%）新 TB。
- 预注册门（燧原 ≥3.6 且均值 >2.9075）两项均差之毫厘未过——按预注册
  **宽度轴最终封顶**（1024→4096→8192→16384→32768 全档扫完，曲线单调
  但递减）。32768=官方 MAX_BLOCK_SIZE，无下一档。
- 残余：榜首 GuanghuLab 燧原 7.8（我方 3.3）为非宽度形态；下一轴待
  燧原结构性证据（K-tile/其他），明后日评估。

## 2026-09-16 00:08 E9 发射（跨午夜）：华为 warps8 vendor

- _ascend vendor 新建（389e637，generic 字节 + num_warps=8 单变量；
  华为 1.3 vs Albedo 2.5）。已提交评测中；门 华为 ≥1.7。

## 2026-09-16 00:45 E9 平台终态：8/8 VALID 2.8967x —— 华为 warps8 无效

- 华为 1.278（vendor 被选，vs 1.3 持平）；均值 2.8967（+0.01 噪声级，
  记为新 best 但非轴增益）。**华为 warps 轴关闭**；燧原 3.32 维持
  e7r 水位。TB 2.897。

## 2026-09-16 Ascend bounded direct：筛选失败，零平台额度

- 固定前批PR78 `077fdc3a0d8d7af02b8b84f131e20399c962c1d3`、PR56 `9e36df5e4bfe146982bf8fa4b1f98bb496fe952c` 的成熟direct/fallback形式：从TB E9 `389e637e42560c791e68ff81d2b32f6b9fe49770`分叉，仅Ascend在grid≤65535移除loop，超大输入保留原persistent；BLOCK4096/warps8/数学与其他四源不变。候选SHA-256 `5922f0b3dfb0aab202a4d9316d28664a1ca13dd13d44bf57509637d113166ee1`。
- IR硬门通过：6/6 inspected基线保留1个循环、direct为0；5/5必需方法通过。27桶×5轮AB/BA端到端几何均值 **1.01924<1.05**，每轮最低1.01195<1.02，最差 **0.903817<0.95**。fp16 n4194305五轮均0.9019–0.9063，n270369中位0.923641，属稳定回退；kernel-only整体1.01118。fp16 direct出现2 spills（baseline0），只记NVIDIA lowering线索，不外推昇腾。
- 全部原始IR、timings、环境、正确性和资源证据已取回验签于 `artifacts/competition/t75-direct-preparation-20260916/`；benchmark SHA-256 `0ba26557af4cdcac7eebaa68114ab66b75e9b839e2d8441f4c1748aaddec5b35`，日志 `36bd20c7eeac978925ddf942cf596b3dbd5c8858d38bdb6f1d4a61198ba4beb5`。远端`/tmp/flagos-t75-direct.IMTSF2`、PID387860、timeout600、EXIT0，GPU已释放。
- 未晋升源码/测试、未建ZIP/intent、未上传或提交。保持TB E9=2.89671667。关闭本次全dtype direct候选；不事后排除fp16、改门或继续扫宽度来掩盖失败。

## 2026-09-17 E10：Ascend persistent vendor，已提交

- 结构（`842a8169`）：仅新增/重写 `_ascend` vendor = generic kernel 字节不变 + persistent launch（`num_vectorcore`，fallback 40；T64 E7 已证载体形态，华为 +44%）。其余成员字节冻结。host 侧 `_worker_count` 仅对 TensorMetadata mock 缺 device 做默认值护栏（T65 grid 测试需要），非计算 fallback。
- screening（RTX 5070 Ti）：unittest 3 项全绿；black/isort/flake8 过。
- release v2（source=verification commit `842a81694a9fed9530ea5ada88343eea3415ccfc`）：3 项全过 0F/E/S/X，generic/ascend 各 17 真实 launch，exit 0。回执 `artifacts/competition/persist-batch-20260917/sigmoid_gate_mul-verification.json` SHA-256 `55791111169a8b305b5938781ce3f33fb28c12ddf52674332a5c9cc5f6505ef0`。ascend target-runtime-unverified。
- ZIP：`artifacts/competition/sigmoid_gate_mul/e10-842a816/sigmoid_gate_mul.zip（8841 bytes）`，SHA-256 `d1d8c4ab9090f05454dc4f062c0d06bbdde396958f46e75686c5059adcd2c141`，5（generic/ascend/enflame/iluvatar/kunlunxin；后四与 e9 冻结集合一致） 成员。
- 预注册门：8/8 有效且均值 > 2.89671667；华为 ≥ 2.6 为 persistent 轴正信号。一次候选一次判决。

## 2026-09-17 E10 单次平台提交

12:26，submission **16574**（daily_seq 3），evaluating；发后额度 27/30。八芯终态另节记录。
## 2026-09-17 E10 平台终态：8/8 有效，微幅新 TB 2.89673333x

submission 16574 completed/valid，8/8，均值 **2.89673333x**（+0.0006%，名义换 TB）。逐芯（vs e9 TB）：huawei 1.278→**1.47693333（+15.6%）**，其余七芯冻结字节窗口漂移（-3%~+1.4%）相互抵消。华为预注册门 2.6 未过：persistent 在本题只给弱正信号，他队 16-20x 缺口需其他结构。轴按门关闭，不重掷。
