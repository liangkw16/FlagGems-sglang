#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Generate docs/competition/experiments/INDEX.md from ledger CURRENT blocks.

Each operator ledger's top ```current fenced block is the single
human-maintained truth for that task (see the flagos-operator-race SKILL).
Hand-written summary tables elsewhere are historical. This script renders
the index and fails loudly on malformed blocks so a stale index cannot be
regenerated silently.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "docs" / "competition" / "experiments"
REQUIRED = {"task", "operator", "validity", "platform", "updated"}


def parse_current(block_text):
    fields = {}
    for line in block_text.strip().splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    return fields


def main():
    rows, missing = [], []
    for path in sorted(EXP.glob("*.md")):
        if path.name in {"README.md", "INDEX.md"}:
            continue
        text = path.read_text(encoding="utf-8")
        marker = "```current"
        start = text.find(marker)
        if start < 0:
            missing.append(path.name)
            continue
        end = text.find("```", start + len(marker))
        fields = parse_current(text[start + len(marker):end])
        absent = REQUIRED - set(fields)
        if absent:
            print(
                f"{path.name}: CURRENT block missing {sorted(absent)}",
                file=sys.stderr,
            )
            return 1
        if not fields["task"].isdigit():
            print(f"{path.name}: CURRENT 'task' must be numeric", file=sys.stderr)
            return 1
        fields["ledger"] = path.stem
        rows.append(fields)
    rows.sort(key=lambda f: int(f["task"]))

    lines = [
        "# 实验状态索引（GENERATED）",
        "",
        "> 由 `tools/gen_experiment_index.py` 从各账本顶部 ` ```current ` 块生成，",
        "> 不要手改本文件；状态更新只改账本 CURRENT 块，然后重跑脚本。",
        "",
        "| Task | 算子 | 有效性 | 平台 | 团队最佳 | 封存 | 下一步 | 更新 | 账本 |",
        "| ---: | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for f in rows:
        best = f.get("team_best_stage", "-")
        if f.get("team_best_speedup"):
            best = f"{best} {f['team_best_speedup']}x"
        lines.append(
            f"| {f['task']} | {f['operator']} | {f['validity']} | "
            f"{f['platform']} | {best} | {f.get('sealed', '-')} | "
            f"{f.get('next', '-')} | {f['updated']} | "
            f"[{f['ledger']}]({f['ledger']}.md) |"
        )
    if missing:
        lines += ["", "缺 CURRENT 块（未计入索引）：" + "、".join(missing)]
    (EXP / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote INDEX.md: {len(rows)} entries, {len(missing)} missing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
