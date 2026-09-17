# data/ 快照命名与新鲜度说明

本目录存放带时间戳的**不可变证据快照**与两个由同步脚本维护的**动态文件**。
引用任何快照时以文件内字段（`observed_at` / `as_of` / `fetched_at`）为准，
不要按文件名里的修饰词推测时间顺序。

## 动态文件（同步脚本覆盖写，勿当历史证据）

- `race-overview.json` / `task-catalog.json`：`tools/sync_flagos_season2_docs.py`
  每次运行刷新；当前值与 `task-index.md` 顶部同步时间一致。

## 同日多份快照族（按 observed_at 排序，勿按文件名）

| 族 | 时间顺序（旧 → 新） |
| --- | --- |
| `leaderboard-batch6-perchip-20260917*.json` | 无后缀（20:51，开题快照，T76–T81 逐芯） |
| `leaderboard-batch5-perchip-20260917*.json` | 无后缀（10:57）→ `-evening`（13:32，注意"evening"名不副实）→ `-final`（17:00，终局口径） |
| `top1-intel-20260916*` | `-live` → `-refresh2` → `-final` |
| `batch5-intel-20260914*` | `-afternoon` → `-evening` → `-late` → `-night` → `-night2` |
| `batch4-intel-20260910*` | `-noon` → `-1240` → `batch4-final-intel-20260909` 之外的 `batch4-final-20260910-1440` 为终局 |
| `our-submissions-all-20260910*` | `-1250` → `-1400`（两份各 ~10MB 近重复，仅时间不同） |

## 引用规则

1. 文档引用快照时同时写明 observed_at 与 SHA-256（终局复盘的样板见
   `batch5-final-review-20260917.md`）。
2. "当前榜单"类论断只能引用 `race-overview.json`/`task-index.md` 或最新
   observed_at 的快照；更早的同族文件只作历史对照。
3. 新快照落盘前先确认 observed_at 晚于同族既有文件，避免时间倒挂的命名。
