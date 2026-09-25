# Task 95 `create_chunked_prefix_cache_kv_indices` 实验记录

```current
task: 95
operator: create_chunked_prefix_cache_kv_indices
batch: 7
validity: valid(7/7,s0)
platform: e2(21223)invalid_correctness:fill kernel逐块标量窗口扫描全芯减速2x+昆仑编译墙(100%失配);已回滚s0字节,TB保s0(21123,valid 82x)
candidate_stage: s0
team_best_stage: s0
team_best_speedup: -
sealed: yes
next: e2教训入册:结构假设代理自证后还需全链路代理计时;融合填充轴关闭(需排序无关且O(1)每元素的成员判定才能复活);榜首水位天数133/海光50仍1.6x,结构未破译
updated: 2026-09-25
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

## 不可变身份（s0 v2，2026-09-24，契约 spec 高阶复审后）

- source commit = verification commit = `daf77923f6913447b65a8c226f34297ce13b4440`（review v2：补 req_to_token.stride(1) 列偏移 + 四元数据向量 stride）。
- ZIP：`artifacts/competition/create_chunked_prefix_cache_kv_indices/s0-daf7792/create_chunked_prefix_cache_kv_indices.zip`，SHA-256 `017cd27f3128b31a52ca42ea67b643b44daf031233ff337bcdd87609db763559`（单成员 `create_chunked_prefix_cache_kv_indices.py`，generic-only）。
- release 回执：`artifacts/competition/b7-s0-release2-20260924/create_chunked_prefix_cache_kv_indices/verification.json`（mode=release，exit 0，绑定该 commit 字节），SHA-256 `56854353527d6e930ade59b4fbe407926a4fb5758eea4ae4190f03ae9e075200`；
  日志 SHA-256 `1a10d59e66404b57858898a429b35b5c42422d6de5c6eaba2e89bc9bec657439`。NVIDIA RTX 5070 Ti / torch 2.13.0+cu130 / triton 3.7.1。
- 提交脚本：`artifacts/competition/b7-s0-release2-20260924/submit-batch7-s0.sh`（v2，全部参数预烘焙）。
- review 历史：v1（commit 级，gpt-6-astra medium）4 项全修；v2（--base 对照契约 spec，high）5 项 P2 全修；
  三轮修复均经 screening + release 双门禁复跑全绿。
