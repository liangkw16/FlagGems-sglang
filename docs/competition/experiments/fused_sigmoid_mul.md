# Task 97 `fused_sigmoid_mul` 实验记录

```current
task: 97
operator: fused_sigmoid_mul
batch: 7
validity: candidate-sealed
platform: 未提交（quota 0/30 未重置；release 门禁全过，候选封存待投）
candidate_stage: s0
team_best_stage: -
team_best_speedup: -
sealed: yes
next: s0 已封存待投；E1 轴=昆仑 tanh 原生 lowering 假设/华为 EA 专配——Padé sigmoid 已被 codex-ask+本地实证否决入 generic（gate=-6 误差 8%、gate=-8 翻号，详见 plan §6）
updated: 2026-09-24
```

## 契约

- `(attn_output, gate)`：`out = attn * sigmoid(gate)`，fp32 计算、按输入 dtype 存储；
  flat 同形路径 + strided 路径（gate 3D 非连续，需按 stride 读）；per-dtype 容差。

## 榜单靶标

EvokeAgent 5.00 分：tianshu 5.301(c2flow) / muxi 3.033 / haiguang 4.102 /
**kunlun 2.007** / **huawei 1.486**（EA 独占两芯第一，他队 0.60-0.96）/ A 3.776 / B 3.150。
T90/T81 华为经验（子块结构 1.23→1.5 轴）可迁移。

## S0 结构（commit b6641097）

- 单 kernel flat map BLOCK=2048/w8；ATTN_CONT/GATE_CONT 双 constexpr 分支：
  连续时纯 flat 偏移；非连续时 t=offs//hidden 分解 + gate 3D stride 直读（无 contiguous 拷贝）；
  `tl.sigmoid` fp32（T90 已验证跨芯形式）。
- 代理 sweep：4096×4096 torch 825.75µs vs 我方 115.3µs（**7.1x**，torch 5 kernel）；
  strided3d 66.1µs。

## 证据

- NVIDIA 代理 screening（BLOCK=4096 旧配置）6 测试 0 失败 18 launch；b2048 复筛中。

## 情报

- 快照 SHA-256 `061475d31505bb81f16a5a212fa7cfd6e4b1142188cb02da2c202962efedadb2`。

## 不可变身份（s0 v2，2026-09-24，契约 spec 高阶复审后）

- source commit = verification commit = `daf77923f6913447b65a8c226f34297ce13b4440`（review v1 grid/块失配已修；review v2：strided 分支与 flat offs 全面 i64 化）。
- ZIP：`artifacts/competition/fused_sigmoid_mul/s0-daf7792/fused_sigmoid_mul.zip`，SHA-256 `40003a157420b8a5598fede4bc02aea5051d149da29d451bac0043116a720b51`（单成员 `fused_sigmoid_mul.py`，generic-only）。
- release 回执：`artifacts/competition/b7-s0-release2-20260924/fused_sigmoid_mul/verification.json`（mode=release，exit 0，绑定该 commit 字节），SHA-256 `96836a78ee559aac6c11889a5b03ddcc97d0482aea0f3149f57fbeb9eb7ae122`；
  日志 SHA-256 `4cb32043bbc3d80ad5babd5a2bc8626c048a93a13a9fbb69869428ae698e8922`。NVIDIA RTX 5070 Ti / torch 2.13.0+cu130 / triton 3.7.1。
- 提交脚本：`artifacts/competition/b7-s0-release2-20260924/submit-batch7-s0.sh`（v2，全部参数预烘焙）。
- review 历史：v1（commit 级，gpt-6-astra medium）4 项全修；v2（--base 对照契约 spec，high）5 项 P2 全修；
  三轮修复均经 screening + release 双门禁复跑全绿。
