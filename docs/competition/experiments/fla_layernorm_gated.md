# Task 51 `fla_layernorm_gated` 实验记录

```current
task: 51
operator: fla_layernorm_gated
batch: 4
validity: valid
platform: 8/8(e4,5.195325x非最佳);e2最佳5.3939x
team_best_stage: e2
team_best_speedup: 5.3939
sealed: no
next: 保留e2最佳5.3939;多行tile平台回退,关闭本轴;需新目标芯结构证据
updated: 2026-09-07
team_best_commit: 5985b1ce09fe6e7cec374c271aa1c25940264783
```

## S0（2026-09-05，submission 9867）
- 7/8，燧原 1830s 超时（非 correctness），其余七芯全过

## E1 燧原 vendor → **8/8 VALID**（submission 9944）
- 燧原 vendor：一 program 一行（去 grid-stride 循环）+ tl.rsqrt/tl.sigmoid
- **燧原超时→2.3398x**；逐芯：天数 9.21 / 沐曦 4.57 / 燧原 2.34 / 海光 7.74 /
  昆仑 0.944 / 华为 2.52 / A 8.73 / B 7.07

## 2026-09-05 Codex 会诊作战方案（预注册）

**华为（首发，最高置信）**：`dim` → `tl.constexpr D`，`offs < dim`
运行时 int 比较（昇腾 Vector CMP 标量退化，官方指南确认）变编译期
静态 mask，`/D` 常量折叠；行结构与单次 x load 不变（勿先搬 T40 的
1024 子块——那会迫使 RMSNorm 重载两次 x）。预期 +15~40%；门 ≥+15%。

**沐曦**：仅 `1/sqrt` → `tl.rsqrt`（消除逐 lane 向量除法；IR 不变则
取消候选）；门 ≥+12%。
**昆仑**：仅 `D<1024` 时 BLOCK 提 1024（T21 唯一成功轴；T19 multi-row
无收益、BLOCK 2048 负收益，勿重复）；门 ≥+15%；raw_result 若暴露逐
case D 分布则先查命中再发。
后置：g load 后移（缩短归约期 live 向量）；BLOCK_D≥2048 沐曦 warps8。
LayerNorm 勿改 `E[x²]-E[x]²` 单遍（大均值小方差消减误差，fp32 1e-4 风险）。

> 注：并行会话 09-05 22:56 记录已发 T51 E2（8/8、5.3939x、未超 e1，轴内容未落账本）；
> 执行本方案任一候选前先与该会话核对 E2 消费过的轴，避免重复预注册。

## E3 华为 constexpr-D vendor（2026-09-06，submission 10385）

- `_ascend` vendor：generic 数学逐字节不变，唯一变量 = `dim` →
  `tl.constexpr D`（`offs < D` 编译期折叠 + `/D` 常量折叠）；
  source `080148b`，ZIP `e3-080148b` SHA-256 `2bddb9d4…271d`
  （4 成员），screening OK
- 七芯已过：**华为 2.52→2.6182（+3.7%，远低于 ≥+15% 门——轴单发
  证伪关闭**：int 比较不是本题华为瓶颈）；天数 9.2822/沐曦 4.3918/
  海光 8.133/昆仑 0.9682（kunlunxin vendor 生效）/A 8.1964/B 6.873
- 燧原（非目标芯）卡病态盒子 waiting_callback；无论燧原落点，
  估算均值 ≤5.38 < e1 TB 5.39——非 team best
- 判定：constexpr-D 轴关闭；T51 剩余轴：沐曦 rsqrt（预注册 #2）

## E4 Top1 冲刺候选（2026-09-07，提交前）

先校正旧记录：2026-09-07 只读查询确认 E2/submission10276 才是 team best 5.3939（source 5985b1ce09fe6e7cec374c271aa1c25940264783），高于 E1 5.390025；旧节“未超 E1”是过期表述。E3/submission10385 已终态 7/8 invalid_correctness，燧原 1830s 超时，不能继续写在评。

参考 GitHub PR50 的多行 tile 调度，针对本题非分组 RMS/LayerNorm 契约适配：仅 `rows>=128 and dim<=1024` 使用 BLOCK_R=4，其余1；列归约改 axis=1，权重/偏置保留列 mask，原数学运算顺序不变。测试新增行数 127/128/129 等 tile 边界、列127/128/129与1023/1024/1025、三dtype、两种norm/gate和stride。

五轮交错代理 A/B：`[4096,128]` 1.50x；`[4096,256/512/1024]` 1.008/1.003/1.001x；`[4096,8192]` 1.00x；阈值边界约0.995–1.008x。早期 rows>=8 导致 `[32,512]` 0.697x，已缩小条件并复测；单独 rsqrt 仅≈1.016x，未采用。

预注册：一次平台探索检验隐藏 case 是否受益；目标超过 5.3939，冲榜需当时榜首6.668225×1.03≈6.86827。只有一个代理 case 明显获益，不承诺总体收益；无均值提升则关闭多行阈值轴。

- source/verification commit：`5544a77d406e1f22e735f6cba2bd4ead8f80d4f8`；ledger commit 为本节所属提交。
- 本地 py_compile、Black、isort、flake8 通过。NVIDIA release：7 方法、83 次 kernel launch，fail/error/skip/xfail 均为0。执行源：`src/flaggems_sglang/ops/fla_layernorm_gated.py`。
- 测试源码 SHA256：`dc8e1281482839cecd2d225f89265caf1493803df135ab608516fb7f6959e0eb`；各输入文件 SHA 见 verification-input.json。
- 远端 `gpu:/tmp/flagos-b4-top1.8nsvBC/t51-release`，RTX5070Ti / driver610.57.04 / Python3.12.13 / torch2.13.0+cu130 / triton3.7.1。串行后台执行：`timeout 600 /home/kevin/notebook/.venv/bin/python .agents/skills/flagos-operator-race/scripts/verify_release.py run --directory /tmp/flagos-b4-top1.8nsvBC/t51-release`。
- ZIP `/Users/bytedance/ccc/flagos/artifacts/competition/fla_layernorm_gated/e4-5544a77/fla_layernorm_gated.zip`，14982 bytes，SHA256 `05e9e495d68e068cfd3f0aa764d8af84fbed86d47a68a7108d26ef7def8350e7`；dry-run 与最终 manifest 五项恒等字段全部匹配。
- ZIP member `fla_layernorm_gated.py` SHA256 `e1aa34a0f3d851473998a2848175b0539ee9be681b2c4d753f06a922efdf309e`。
- ZIP member `fla_layernorm_gated_ascend.py` SHA256 `92baf6f3a08d4675cee828d451250be42a8b977d3f889eb59ca8c582654fb0c4`。
- ZIP member `fla_layernorm_gated_enflame.py` SHA256 `7be3674559520a8f77d34208580761b4d98cd4eb8ff414ea6a6f74ca7a39bae8`。
- ZIP member `fla_layernorm_gated_kunlunxin.py` SHA256 `66e08be2b4d870028f63ddceb89dd6e6b66ed79ad333e29757ad901332b1ebdd`。
- 证据 `/Users/bytedance/ccc/flagos/artifacts/competition/fla_layernorm_gated/e4-5544a77/validation/verification.json` SHA256 `08476719b7cc4736e179bc9c16b593ef882d8dd8478911796fd3f1acfcb64795`。
- 证据 `/Users/bytedance/ccc/flagos/artifacts/competition/fla_layernorm_gated/e4-5544a77/validation/verification.log` SHA256 `7ea153e1e09fb8aee45357ec6ce7d39ac4685117954d2e3ec5c744f6107fcbf1`。
- 证据 `/Users/bytedance/ccc/flagos/artifacts/competition/fla_layernorm_gated/e4-5544a77/validation/verification-input.json` SHA256 `35c1b19af362fe9b0a0d91adf37f10e3f48bee314fa975b1a0000c3e2632aa7f`。
- 附加基准、诊断及 MCP 证据清单 `/Users/bytedance/ccc/flagos/artifacts/competition/fla_layernorm_gated/e4-5544a77/validation/evidence-sha256.json` SHA256 `45a1aa10a777047186b2409b759228b4cfa61b990190f56dfcffcb84d7ee251c`。
- 首轮每题最多1次正式上传/提交；本次预算上限沿用批准的 T43/T51/T52 各4、T57 3、储备6，须有新证据才继续消耗。sending/uncertain/stale_after_upload 不自动重试。

## 本轮平台结果与止损（2026-09-07T14:55:36+08:00）

- submission `10746` / daily_seq `12` / created `2026-09-07T14:48:28`；preflight与上传/提交均只执行一次，远端ZIP验签 `verified`。
- 平台原始状态 `completed` / `valid`，通过8/8、终态8/8；average_speedup `5.195325`，is_team_best `False`；观测时额度 `18/30`。
- E4 比最佳 E2 回退约3.68%；沐曦3.7252对旧4.5654回退18.40%，其他芯波动也影响均值。不能把全部回退归因多行分支：BLOCK_R=1仍改变IR维度，且隐藏性能case未透出。此轴无净收益，停止追加平台试验。

| 芯片 | 状态/正确性 | 加速比 | 实际文件 |
| --- | --- | ---: | --- |
| tianshu | completed/True | 9.235 | `fla_layernorm_gated.py` |
| muxi | completed/True | 3.7252 | `fla_layernorm_gated.py` |
| enflame | completed/True | 2.3446 | `fla_layernorm_gated_enflame.py` |
| haiguang | completed/True | 7.9656 | `fla_layernorm_gated.py` |
| kunlunxin | completed/True | 0.9616 | `fla_layernorm_gated_kunlunxin.py` |
| huawei | completed/True | 2.3488 | `fla_layernorm_gated_ascend.py` |
| card_a | completed/True | 8.0856 | `fla_layernorm_gated.py` |
| card_b | completed/True | 6.8962 | `fla_layernorm_gated.py` |

- 证据 `/Users/bytedance/ccc/flagos/artifacts/competition/fla_layernorm_gated/e4-5544a77/validation/51-submit.json` SHA256 `78c9eb2c5ea6416ce70162d585f9acc914c7d17f6538457a2867473408f84182`。

- 证据 `/Users/bytedance/ccc/flagos/artifacts/competition/fla_layernorm_gated/e4-5544a77/validation/51-status-first-round.json` SHA256 `f1a1c304a944abf543a866887fdce262d865c483b97053f2e61a212e1b81a7b4`。
