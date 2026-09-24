# Task 94 `concat_mla_absorb_q` 实验记录

```current
task: 94
operator: concat_mla_absorb_q
batch: 7
validity: valid(候选e2在途)
platform: e4(21159)在途;kunlun轴实测:e1(1024地板,双store)0.0814>e2(RPP8)0.0656>e3(精确512/64)0.0278=>=1024宽向量是昆仑硬规律(T98 e2平行证:0.350->0.124);e4=单1024向量单store流+w4;华为grid修复兑现0.16(榜0.98);天数1.82/海光2.31已过榜首水位
candidate_stage: s0
team_best_stage: e1(21144,无效但天数1.774#1/海光2.256平榜首)
team_best_speedup: -
sealed: yes
next: watch e4 昆仑>=0.1;若过线即有效(天数/海光#1,昆仑次优0.62仍需结构);华为0.16->0.98留E5(persistent)
updated: 2026-09-25
```

## 契约

- `(a, b)`：沿最后一维 cat，`[d0,d1,a_last]+[d0,d1,b_last] → [d0,d1,a_last+b_last]` bf16；
  题面明示两源行 stride 不同（不得假设连续）；exact 纯搬移。

## 榜单靶标

EvokeAgent 5.50 分（仅 2 队）：tianshu 1.700 / muxi 1.216 / haiguang 2.258 / kunlun 1.409 /
**huawei 0.420** / A 1.101 / B 1.420。华为第一仅 0.42 = 低垂果实；海光 2.26 显示上行空间。

## S0 结构（commit b6641097）

- 单 kernel：BLOCK_R=4 行瓦片；行 id 分解 (i0,i1)=divmod(row,d1)，a/b 各自
  stride 算基址；先 a_last 后 b_last 两段列循环（BLOCK_C=512 mask），一次写出行；
  i64 寻址；num_warps=8。
- 代理 sweep：小 shape 全配置 launch-bound（~6.1µs vs torch cat 2.7µs）；平台大 shape
  下单 kernel 结构 vs torch.cat 多 kernel 占优；config 保持 r4c512w8。

## 证据

- NVIDIA 代理 screening（配置未变）4 测试 0 失败 9 launch（21:15 回执有效）。

## 情报

- 快照 SHA-256 `061475d31505bb81f16a5a212fa7cfd6e4b1142188cb02da2c202962efedadb2`。

## 不可变身份（s0 v2，2026-09-24，契约 spec 高阶复审后）

- source commit = verification commit = `daf77923f6913447b65a8c226f34297ce13b4440`（review v1 4项已修（含最内层 stride）；本轮 spec 复审 T94 无新发现）。
- ZIP：`artifacts/competition/concat_mla_absorb_q/s0-daf7792/concat_mla_absorb_q.zip`，SHA-256 `924da41ce714a9b230a904025e44e633c6c314a39000c50a9e0142d54326ae3f`（单成员 `concat_mla_absorb_q.py`，generic-only）。
- release 回执：`artifacts/competition/b7-s0-release2-20260924/concat_mla_absorb_q/verification.json`（mode=release，exit 0，绑定该 commit 字节），SHA-256 `68f754f808229fa146405cc2380304a320fda1c37fee356c670d14fe2fb47b1c`；
  日志 SHA-256 `cf0467390f1f11c19b993408cf62a851a431d43329ebb9dc7e25c6132335c09b`。NVIDIA RTX 5070 Ti / torch 2.13.0+cu130 / triton 3.7.1。
- 提交脚本：`artifacts/competition/b7-s0-release2-20260924/submit-batch7-s0.sh`（v2，全部参数预烘焙）。
- review 历史：v1（commit 级，gpt-6-astra medium）4 项全修；v2（--base 对照契约 spec，high）5 项 P2 全修；
  三轮修复均经 screening + release 双门禁复跑全绿。
