#!/usr/bin/env python3
"""Sync public FlagOS season-2 operator tasks into searchable local files."""

from __future__ import annotations

import gzip
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


RACE_ID = "782kzq4m"
API_BASE = f"https://flagos.io/flagos/api/v1/races/{RACE_ID}"
RACE_URL = f"https://flagos.io/race-detail-season2?id={RACE_ID}"
OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "competition"

PUBLIC_FIELDS = (
    "tid",
    "task_no",
    "batch_task_no",
    "operator",
    "name",
    "description",
    "batch_no",
    "supported_gpus",
    "supported_gpu_count",
    "submit_start_at",
    "submit_end_at",
    "status",
    "competition_status",
    "speedup_threshold",
    "submission_count",
    "participating_team_count",
    "current_leader_team_name",
    "current_best_speedup",
    "current_leader_passed_gpu_count",
)


def get_json(path: str) -> dict:
    request = Request(
        f"{API_BASE}/{path}",
        headers={
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "Lang": "CN",
            "User-Agent": "FlagOS-local-doc-sync/1.0",
        },
    )
    with urlopen(request, timeout=30) as response:
        payload = response.read()
        if response.headers.get("Content-Encoding") == "gzip":
            payload = gzip.decompress(payload)
    result = json.loads(payload)
    if result.get("code") != 200:
        raise RuntimeError(f"FlagOS API failed for {path}: {result}")
    return result["data"]


def public_task(task: dict) -> dict:
    result = {key: task.get(key) for key in PUBLIC_FIELDS}
    result["threshold_team_count"] = (task.get("status_detail") or {}).get(
        "threshold_team_count"
    )
    return result


def speedup(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}x"


def build_index(tasks_by_batch: dict[int, list[dict]], fetched_at: str) -> str:
    lines = [
        "# 第二届 FlagOS 算子赛题索引",
        "",
        f"> 来源：[比赛页]({RACE_URL})；同步时间：`{fetched_at}`。",
        "> 状态和榜单会变化，运行 `python tools/sync_flagos_season2_docs.py` 更新。",
        "",
    ]
    for batch_no, tasks in tasks_by_batch.items():
        lines.extend(
            [
                f"## 第 {batch_no} 批",
                "",
                "| 题号 | 算子 | 状态 | 提交/队伍 | 达标队伍 | 当前第一 | 平均加速比 |",
                "| ---: | --- | --- | ---: | ---: | --- | ---: |",
            ]
        )
        for task in tasks:
            rel = (
                f"tasks/batch-{batch_no}/"
                f"{task['task_no']:02d}-{task['operator']}.md"
            )
            lines.append(
                f"| {task['task_no']} | [{task['operator']}]({rel}) | "
                f"{task['competition_status']} | {task['submission_count']}/"
                f"{task['participating_team_count']} | "
                f"{task['threshold_team_count'] or 0} | "
                f"{task['current_leader_team_name'] or '-'} | "
                f"{speedup(task['current_best_speedup'])} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    fetched_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    overview = get_json("operator-overview")
    tasks_by_batch: dict[int, list[dict]] = {}
    details: dict[str, dict] = {}

    BATCHES = (1, 2, 3, 4)
    for batch_no in BATCHES:
        tasks = [
            public_task(item)
            for item in get_json(f"operator-tasks?batch_no={batch_no}")
        ]
        tasks.sort(key=lambda item: item["task_no"])
        tasks_by_batch[batch_no] = tasks
        for task in tasks:
            details[task["operator"]] = get_json(
                f"operator-tasks/{task['operator']}"
            )

    all_tasks = [task for tasks in tasks_by_batch.values() for task in tasks]
    # platform keeps reshaping (batch counts grew and the overview lost
    # current_batch on 2026-08-29); warn instead of failing the sync
    for batch_no, want in {1: 7, 2: 17, 3: 17, 4: 6}.items():
        got = len(tasks_by_batch.get(batch_no, []))
        if got != want:
            print(f"WARN: batch {batch_no} has {got} tasks (expected {want})")
    if len({task["operator"] for task in all_tasks}) != len(all_tasks):
        print("WARN: duplicate operator names across batches")
    if not all(
        re.fullmatch(r"[A-Za-z0-9_]+", task["operator"]) for task in all_tasks
    ):
        print("WARN: unexpected operator name characters")
    cb = (overview.get("current_batch") or {}).get("batch_no")
    if cb is not None and cb != BATCHES[-1]:
        print(f"WARN: current_batch is now {cb}")

    for batch_no, tasks in tasks_by_batch.items():
        task_dir = OUT_DIR / "tasks" / f"batch-{batch_no}"
        task_dir.mkdir(parents=True, exist_ok=True)
        for task in tasks:
            detail = details[task["operator"]]
            header = (
                f"<!-- source: {API_BASE}/operator-tasks/{task['operator']} -->\n"
                f"<!-- synced_at: {fetched_at} -->\n\n"
            )
            (task_dir / f"{task['task_no']:02d}-{task['operator']}.md").write_text(
                header + detail["content"].rstrip() + "\n", encoding="utf-8"
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "task-index.md").write_text(
        build_index(tasks_by_batch, fetched_at), encoding="utf-8"
    )
    data_dir = OUT_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "race-overview.json").write_text(
        json.dumps(
            {
                key: overview.get(key)
                for key in ("as_of", "current_batch", "gpu_catalog", "stats")
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (data_dir / "task-catalog.json").write_text(
        json.dumps(
            {"race_id": RACE_ID, "fetched_at": fetched_at, "tasks": all_tasks},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Synced {len(all_tasks)} tasks into {OUT_DIR}")


if __name__ == "__main__":
    main()
