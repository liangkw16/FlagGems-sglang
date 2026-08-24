# 厂商 Triton backend 源码本地缓存

本目录是**只读上游镜像**，用于离线做跨芯约束审查（warp 语义、`num_warps`
上限、dtype/dot 精度、片上存储上限、可调参数集合），避免每次都联网或撞
GitHub API 限流。

不要手改缓存字节。刷新与验签：

```bash
python tools/fetch_vendor_backends.py            # 抓取/刷新
python tools/fetch_vendor_backends.py --verify   # 仅按 manifest 校验 SHA-256
```

固定 commit（与 [learning-path.md](../../learning-path.md) 和
[reference-repositories.md](../../reference-repositories.md) 保持一致）：

| 上游 | commit |
| --- | --- |
| `flagos-ai/FlagTree` | `c1ea8285a06e97afad9dd2644bc71f2efca072f4` |
| `triton-lang/triton` | `dff2f7d03532e9ca0598c728c60c204ae7555fc9` |
| `Ascend/triton-ascend` | `865691e2e9b656bc58008170207b4108d92e8dd1` |

每个文件的上游 URL、字节数和 SHA-256 见 [`manifest.json`](manifest.json)。

## 目录与芯片对应

| 目录 | 芯片 / vendor 后缀 | 上游位置 |
| --- | --- | --- |
| `kunlunxin/` | 昆仑芯 `_kunlunxin` | FlagTree `third_party/xpu/backend/` |
| `enflame/` | 燧原 `_enflame` | FlagTree `third_party/enflame/backend/` |
| `iluvatar/` | 天数智芯 `_iluvatar` | FlagTree `third_party/iluvatar/backend/` |
| `metax/` | 沐曦 `_metax` | FlagTree `third_party/metax/backend/` |
| `hygon/` | 海光 `_hygon` | FlagTree `third_party/hcu/backend/` |
| `ascend/` | 华为昇腾 `_ascend` | triton-ascend `docs/en/programming_guide/` |
| `amd/` | 国际通用 AMD 路径 `_amd` | triton `third_party/amd/backend/` |
| `nvidia/` | 国际通用 NVIDIA 路径 `_nvidia` | triton `third_party/nvidia/backend/` |

华为不在 FlagTree（走独立 triton-ascend 仓库），缓存的是官方 Vector Operator
编程指南而非 compiler 源码。

## 这些源码能证明什么、不能证明什么

**能证明**（编译期/ABI 层，静态可读）：

- `warp_size` 默认值与 launch 时实际 block 线程数；
- `num_warps` / `num_ctas` / `num_stages` 是否有效、上限、是否 assert；
- `default_dot_input_precision` 与允许集合；
- `max_shared` / `max_local` / `max_dsm` 等 `OutOfResources` 判据；
- 固定物理并行度（如昆仑 `nclusters`/`ncores`）；
- 可调环境变量集合（如 `TRITONXPU_*`）。

**不能证明**：

- **`grid` 展平上限 65535**。该常数在全部 8 份缓存源码中都不存在
  （`grep -rn 65535` 只命中海光 driver 的无关 `gridX*gridY*gridZ==0` 判断）。
  它是 runtime/驱动层约束，只能由平台报错反推——昆仑
  `uni_sram PassManager::run`、华为 `coreDim=114688`、燧原 `grid.x=256512`
  都是平台侧证据，本地不可预测。
- 比赛 worker 的具体型号、驱动版本、Triton 版本、隐藏 shape；
- 任何性能数字。

缓存的 commit 也不等于比赛 worker 实际使用的版本；结论冲突时以平台逐芯结果
为准。
