#!/usr/bin/env python3
"""Read-only FlagOS race intel puller.

Pulls the live quota, every batch task row and (optionally) the per-team
per-chip leaderboard for each task, then writes one JSON snapshot under
``docs/competition/data/``. GET only: it never uploads, submits or touches
any intent, so it costs no submission quota.

Usage::

    python tools/pull_race_intel.py --batch 4 --out docs/competition/data/snap.json
    python tools/pull_race_intel.py --batch 4 --no-leaderboards
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

SKILL_SCRIPTS = (
    Path(__file__).resolve().parents[1]
    / ".agents"
    / "skills"
    / "flagos-operator-race"
    / "scripts"
)
sys.path.insert(0, str(SKILL_SCRIPTS))

from platform_cli import API, HttpClient, _token  # noqa: E402


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat()


def pull(race: str, batch: int, *, leaderboards: bool) -> dict:
    client = HttpClient(_token())
    quota = client.get(f"{API}/races/{race}/operator-submissions/quota")
    tasks = client.get(
        f"{API}/races/{race}/operator-tasks", {"page": 1, "page_size": 100}
    )
    rows = tasks if isinstance(tasks, list) else tasks.get("list") or []
    selected = [t for t in rows if not batch or t.get("batch_no") == batch]
    boards: dict[str, list] = {}
    if leaderboards:
        for task in selected:
            tid = task.get("tid")
            if not tid:
                continue
            board = client.get(
                f"{API}/races/{race}/operator-tasks/{tid}/leaderboard",
                {"page": 1, "page_size": 50},
            )
            boards[tid] = board if isinstance(board, list) else (
                board.get("list") or []
            )
    return {
        "observed_at": _now(),
        "race_id": race,
        "batch_no": batch,
        "quota": quota,
        "tasks": selected,
        "leaderboards": boards,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--race", default="782kzq4m")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--out", default="")
    parser.add_argument("--no-leaderboards", action="store_true")
    args = parser.parse_args()

    snapshot = pull(
        args.race, args.batch, leaderboards=not args.no_leaderboards
    )
    payload = json.dumps(snapshot, ensure_ascii=False, indent=2)
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {target} ({len(payload)} bytes)")
    else:
        print(payload)
    quota = snapshot["quota"]
    if isinstance(quota, dict):
        print(
            "quota remaining="
            f"{quota.get('remaining')}/{quota.get('total')} "
            f"used={quota.get('used')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
