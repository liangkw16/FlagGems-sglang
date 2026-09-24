# Task 95 `create_chunked_prefix_cache_kv_indices` 实验记录

```current
task: 95
operator: create_chunked_prefix_cache_kv_indices
batch: 7
validity: candidate-sealed
platform: 未提交（quota 0/30 未重置；release 门禁全过，候选封存待投）
candidate_stage: s0
team_best_stage: -
team_best_speedup: -
sealed: yes
next: s0 已封存待投；E1 轴=昆仑 34.6 结构项与 huawei #2（>139.9）
updated: 2026-09-24
```

## 契约

- `(req_to_token, req_pool_indices, chunk_start_idx, chunk_seq_lens, chunk_cu_seq_lens, chunk_kv_indices)`：
  请求 i 拷贝 `req_to_token[pool_i, start_i:start_i+n_i] → out[cu_i:cu_i+n_i]`；
  reference/baseline 都 `chunk_kv_indices.clone()` 后返回（clone 成本打平，尾部哨兵保留）；exact。

## 榜单靶标

GuanghuLab 4.117 分（rank 列按 ranking_score 而非均值排序——c2flow 均值 117 仅排 #3）：
tianshu 133.2 / muxi 25.0 / haiguang 48.4 / **kunlun 34.6(thunguo)** / huawei 519.6(c2flow
疑慢窗，靶标取金狐狸 139.9 争 #2) / A 37.0 / B 42.4。T79 同族经验直移
（T79 我方 173.4x：天数 340/海光 285/A 310）。

## S0 结构（commit b6641097）

- `out = clone()`；单 kernel 每 program 一个请求行：4 个标量 load（pool/start/n/cu）
  + 动态列循环（BLOCK_C=1024 mask，n=0 自动跳过）；`pool.to(i64)*stride0` 寻址；num_warps=4。
- 代理 sweep：req64×chunk2048 pyloop 1320µs vs 我方 5.75µs（**230x**）；配置间无差异（launch-bound）。

## 证据

- NVIDIA 代理 screening（配置未变）6 测试 0 失败 6 launch（21:15 回执有效）。

## 情报

- T95 榜单 rank/均值错位是 ranking_score 公式的直接体现（见 plan 文档第 0 节）。
- 快照 SHA-256 `061475d31505bb81f16a5a212fa7cfd6e4b1142188cb02da2c202962efedadb2`。

## 不可变身份（s0，2026-09-24）

- source commit = verification commit = `90d732fabdfe198b82bbfd5be662cc4ee77904b4`。
- ZIP：`artifacts/competition/create_chunked_prefix_cache_kv_indices/s0-90d732f/create_chunked_prefix_cache_kv_indices.zip`，SHA-256 `b5ca8359221be0eb71512a668fadcc862dc10e227df46d1dd59120c363acda11`（单成员 `create_chunked_prefix_cache_kv_indices.py`，generic-only）。
- release 回执：`artifacts/competition/b7-s0-release-20260924/create_chunked_prefix_cache_kv_indices/verification.json`（mode=release，exit 0，绑定该 commit 字节），SHA-256 `cb5679f7713d9330ad8a109d8bf3f7184009048b59acc57fb28ef19b1224c369`；
  日志 SHA-256 `42f38379b1134175ff361d127b747a2a35062d12acee84d89447902e8ef59f4d`。NVIDIA RTX 5070 Ti / torch 2.13.0+cu130 / triton 3.7.1。
- codex-review（commit 级，gpt-6-astra）：4 项发现（1×P1 grid/块失配、3×P2）全部修复后复筛/release 全绿。
