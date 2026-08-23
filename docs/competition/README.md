# FlagOS 第二届算子挑战赛：要求与提交规范

> 调研时间：2026-08-23（Asia/Shanghai）。动态状态以[比赛页](https://flagos.io/race-detail-season2?id=782kzq4m)为准。

## 1. 比赛概览

- 赛道一：SGLang 框架算子在多款芯片上的性能优化。
- SGLang 联合主办；核心计算必须用 Triton 或 Triton-TLE。
- 奖池 `¥139,000+`，约 200 道算子题，计划分 13 批发布，赛期为 2026 年 8–11 月。
- 本地仓库是官方 [`flagos-ai/FlagGems-sglang`](https://github.com/flagos-ai/FlagGems-sglang) 的 `master` 工作树；第一批公开 harness 在远端分支 `origin/flagos-sglang-batch1`。

## 2. 当前批次与时间

第一批共 7 题，已停止提交并进入评审。第二批共 17 题，API 给出的窗口为：

- 开发/提交：2026-08-20 20:00 至 **2026-08-27 19:59:59**。
- 专家评审：2026-08-28 至 2026-09-03。
- 入选方案 PR：2026-09-04 至 2026-09-10。

上传组件把截止日渲染成 `23:59`，与赛题 API 和赛制倒计时的 `19:59:59` 不一致。按更严格的 **19:59:59** 执行，建议最晚 19:00 前完成最终提交。

完整题目、实时提交数和榜单快照见 [task-index.md](task-index.md)。

## 3. 评分与排名

1. 先过正确性；每题容差写在对应题面中。
2. 每一款支持芯片的加速比都必须 `>= 0.1`，否则不进入有效性能排名。
3. 以全部支持芯片的平均加速比排名。
4. 平均加速比相同，较早提交者优先。
5. 赛题统一反作弊：实际执行路径必须运行 Triton/Triton-TLE 自定义 kernel；不得通过 `try/except`、设备判断或其他分支失败后回退到 PyTorch 原生算子。

当前 24 题都支持 8 类芯片：天数智芯、沐曦、燧原、海光、昆仑芯、华为、国际通用芯片 A、国际通用芯片 B。

## 4. ZIP 提交规范

平台前端的实际校验规则如下：

- 只接受 `.zip`，最大 10 MB。
- ZIP 中除 macOS 垃圾文件外，所有文件都必须以 `.py` 结尾。
- 必须包含通用实现，文件 basename 精确为 `<operator>.py`；可以位于 ZIP 子目录中。
- 其他 Python 文件只能是 `<operator>_<gpu>.py`。
- 源文件使用 UTF-8；核心计算不得只调用 PyTorch。

项目内规范产物把所有成员放在 ZIP 根目录；平台也允许安全子目录中的同名
basename，已有历史包只能做只读内容验签，不要为改目录结构重写原产物。

芯片后缀：

| 芯片 | 后缀 |
| --- | --- |
| 天数智芯 | `_iluvatar` |
| 沐曦 | `_metax` |
| 燧原 | `_enflame` |
| 海光 | `_hygon` |
| 昆仑芯 | `_kunlunxin` |
| 华为 | `_ascend` |
| 国际通用芯片 A/B | `_amd`、`_nvidia` |

平台公开的 `gpu_catalog` 没有披露 A/B 与 AMD/NVIDIA 的一一对应关系；上传校验器同时接受 `_amd` 和 `_nvidia`，不要凭 A/B 名称猜映射。

最小提交包示例：

```text
softcap_out.zip
└── softcap_out.py
```

需要单独优化某芯片时：

```text
softcap_out.zip
├── softcap_out.py
├── softcap_out_ascend.py
└── softcap_out_nvidia.py
```

先按项目 skill 提交已选中的 source/test、完成 release 门禁，并从 manifest 复制完整
source commit；不要用可能仍指向旧字节的 `HEAD` 代替：

```bash
source_commit="SOURCE_COMMIT_FULL_SHA"
python .agents/skills/flagos-operator-race/scripts/build_submission.py \
  softcap_out --stage s0 --commit "$source_commit" --dry-run
python .agents/skills/flagos-operator-race/scripts/build_submission.py \
  softcap_out --stage s0 --commit "$source_commit"
```

产物写入 `artifacts/competition/softcap_out/s0-<短提交>/softcap_out.zip`；命令输出
成员来源、成员 SHA-256、实际 ZIP SHA-256 和规范 ZIP SHA-256。

## 5. 提交额度

- 默认团队每天总计 15 次，覆盖当前批次全部题目，不是每题 15 次。
- 团队至少 80% 成员完成开发者资源绑定后，可提升到每天 30 次。
- 两次提交至少间隔 2 分钟。
- 当前页面调研时账号尚未提交第二批题目；未执行任何上传或提交操作。

## 6. 平台提交与最终 PR 是两件事

比赛阶段先在网页上传 ZIP。只有评审后被确认“攻占”算子的团队，才在 PR 窗口向官方仓库提交代码。

官方仓库布局：

```text
src/flaggems_sglang/
├── ops/<operator>.py
└── runtime/backend/_<vendor>/ops/<operator>.py
```

- 通用实现放 `src/flaggems_sglang/ops/`。
- 芯片实现放对应 `runtime/backend/_<vendor>/ops/`。
- 算子模块应定义 `__all__ = ["<operator>"]`；注册器会扫描直接位于 `ops/` 下的模块，不必修改 `ops/__init__.py`。
- PR 标题格式：`[FlagOS Competition-Track1] Add [Kernel Name] Triton Kernel for sglang`。
- 必须签署 [CLA](https://cla-assistant.io/flagos-ai/FlagGems-sglang)。
- 官方给出的实现样例是 [commit 9642557](https://github.com/flagos-ai/FlagGems-sglang/commit/9642557dabcd277dabdb8abd09d1bb42e0af3b6b)，但它不是当前 `master` 的祖先，只应参考文件结构和写法，不要直接从该提交建第二批分支。

## 7. 本地仓库与 CI 注意事项

- 当前公开远端分支只有 `master` 和 `flagos-sglang-batch1`，尚无第二批官方分支或标签。
- GitHub Actions 只在 `master` push 或以 `master` 为 base 的 PR 上运行；单独 push topic branch 不触发。
- 现有 selector 对新增第二批 generic/vendor 文件通常找不到测试，PR CI 很可能只验证风格，不能代替比赛隐藏 harness。
- benchmark 步骤允许失败，不会证明性能达标；最终正确性和加速比仍以比赛平台的 8 芯片结果为准。
- 本项目是 Python/Triton，不适用工作区约定的 Go remote unit test。

第一批公开 harness 可这样检索，而不切换工作树：

```bash
git grep -n "CORRECTNESS_CASES\|BENCH_CASES" origin/flagos-sglang-batch1 -- tests benchmark
git show origin/flagos-sglang-batch1:tests/test_chunk_local_cumsum_scalar.py
```

## 8. 本地资料结构与来源

```text
docs/competition/
├── README.md                 # 本文：要求和提交规范
├── task-index.md             # 两批赛题与动态榜单快照
├── strategy-batch2.md        # 第二批开发优先级与复用线索
├── learning-path.md          # 题型学习和八芯固定资料入口
├── reference-repositories.md # 已抓取 Git refs 与固定上游链接
├── data/race-overview.json   # 公开赛程、芯片目录和全局统计
├── data/task-catalog.json    # 清洗后的公开结构化数据
└── tasks/
    ├── batch-1/*.md          # 第一批 7 道完整题面/参考实现
    └── batch-2/*.md          # 第二批 17 道完整题面/参考实现
```

资料来源：

- [FlagOS 比赛页](https://flagos.io/race-detail-season2?id=782kzq4m)
- [官方仓库](https://github.com/flagos-ai/FlagGems-sglang)
- [官方 PR 列表](https://github.com/flagos-ai/FlagGems-sglang/pulls)
- 公开只读 API：`https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/{operator}`

更新题面与榜单快照：

```bash
python tools/sync_flagos_season2_docs.py
```
