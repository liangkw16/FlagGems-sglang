# Task 93 `add_constant` 实验记录

```current
task: 93
operator: add_constant
batch: 7
validity: candidate-sealed
platform: 未提交（quota 0/30 未重置；release 门禁全过，候选封存待投）
candidate_stage: s0
team_best_stage: -
team_best_speedup: -
sealed: yes
next: s0 已封存待投：quota 0/30（观测至 21:56 未重置，watcher 运行中）；quota>0 即按 artifacts/competition/b7-s0-release-20260924/submit-batch7-s0.sh 提交（六题已烘焙）；E1 轴=华为专配（kernelgen 3 轮失败详情不可得）/BLOCK 扫描
updated: 2026-09-24
```

## 契约

- `def reference(src, constant)`：1D 连续 int32，`out = src + constant`，精确。
- baseline 是 C++ 模板（常量编译期折叠 + kMaxVecBytes 向量化）；>2^20 元素走向量路径。

## 榜单靶标（逐芯 #1 读数，score=Σ1/rank 口径，见 plan 文档）

wangteam 4.15 分领跑但全场均值 <1.13：tianshu 1.129 / muxi 1.012 / haiguang 0.971 /
kunlun 0.910 / huawei 0.988 / A 1.005 / B 1.008。全芯 ~1.0+ 即可拿 5.5+ 分。
关键分化：华为其他队 0.14-0.34（wangteam 0.988 独家配方），昆仑次优 0.48-0.72。

## S0 结构（commit b6641097）

- `CONSTANT: tl.constexpr` 编译期折叠（对齐 baseline 模板 trick）；
- flat map，BLOCK=1024 / num_warps=4（4M 元素代理 11.69µs ≈ torch 11.70µs）；
- i64 仅寻址；mask 全覆盖；`if numel` 防 0 元素。

## 证据

- NVIDIA 代理 screening（BLOCK=4096 旧配置）4 测试 0 失败；1024/4 新配置复筛中。
- 代理 launch 开销实测：Triton ~4.8µs vs torch.add ~2.6µs（小 shape 全场 <1.0 的成因假设）。
- KernelGen 华为 autotune job `6e09dda3`：3 轮后 failed，服务端已清理记录（poll 404），失败详情不可恢复；华为专配轴待平台逐芯反馈后重开（generate_kernel 新 job 或依 s0 华为读数定位）。
- KernelGen 昆仑后端仍 502（T98 请求已归档 `t98-kunlun-tune-req.json`）。

## 情报

- 快照 `docs/competition/data/batch7-intel-20260924.json`
  SHA-256 `061475d31505bb81f16a5a212fa7cfd6e4b1142188cb02da2c202962efedadb2`。

## 不可变身份（s0 v2，2026-09-24，契约 spec 高阶复审后）

- source commit = verification commit = `daf77923f6913447b65a8c226f34297ce13b4440`（review v1 4项（1P1+3P2）已修；本轮 spec 复审 T93 无新发现）。
- ZIP：`artifacts/competition/add_constant/s0-daf7792/add_constant.zip`，SHA-256 `cad85701d335dbe17a747e10280eec4920859563391c783cfd5c523b1ede6cf9`（单成员 `add_constant.py`，generic-only）。
- release 回执：`artifacts/competition/b7-s0-release2-20260924/add_constant/verification.json`（mode=release，exit 0，绑定该 commit 字节），SHA-256 `89d7b8ba239b5d04f21d8540e3450825367bd0cb0cf6db96a47b0ce4ec10efb3`；
  日志 SHA-256 `e654eed4ed5f5167f19d6dc96677a068d5adb3f9b59653bfe2ce4d7ec132370d`。NVIDIA RTX 5070 Ti / torch 2.13.0+cu130 / triton 3.7.1。
- 提交脚本：`artifacts/competition/b7-s0-release2-20260924/submit-batch7-s0.sh`（v2，全部参数预烘焙）。
- review 历史：v1（commit 级，gpt-6-astra medium）4 项全修；v2（--base 对照契约 spec，high）5 项 P2 全修；
  三轮修复均经 screening + release 双门禁复跑全绿。
