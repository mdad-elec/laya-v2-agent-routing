"""The measurement campaign: every selected item of every suite, on every cell, as Atlas outcome rows.

A cell is `provider/model@effort`. A backend answers `ask(cell, messages) -> (text, usage)`; the
runner never knows which kind (the dss gateway, a plain OpenAI-compatible endpoint, Anthropic).

Carried over from v2's cell_bench, because each was measured the hard way:
- resumable: a (item, cell, rep) with an `ok` or `unscored` row is never asked again; an `error`
  row is retried on the next run and is never a measured 0;
- one lane per seat (provider), so a subscription seat gets one call at a time;
- a seat limit stops THAT seat's lane loudly, records nothing for the refused call, and the run
  reports it; the other lanes carry on.
A suite scores through `suite.run(item, ask)`, so an agentic suite (BFCL) runs its own episode loop.
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from suites.base import Suite, split_of


class SeatLimit(Exception):
    """The seat refused for its usage window (429, "usage limit", ...): stop this seat, not the run."""


def _unscored_type():
    from suites.chat.suite import Unscored

    return Unscored


def _done(path: Path) -> set[tuple[str, str, int]]:
    """Keys whose LATEST row is ok or unscored: measured, not to be asked again."""
    latest: dict[tuple, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                r = json.loads(line)
                latest[(r["item_id"], r["cell"], r["rep"])] = r["status"]
    return {k for k, status in latest.items() if status in ("ok", "unscored")}


def run_campaign(suites: list[Suite], cells: list[str], backend, out: Path, harness_sha: str, reps: int = 1,
                 items_for=lambda suite: suite.load()) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _done(out)
    lock = threading.Lock()
    report = {"answered": 0, "errors": 0, "unscored": 0, "limited": {}}
    work: dict[str, list] = {}
    for suite in suites:
        for item in items_for(suite):
            for cell in cells:
                for rep in range(reps):
                    if (item.item_id, cell, rep) not in done:
                        work.setdefault(cell.split("/", 1)[0], []).append((suite, item, cell, rep))
    Unscored = _unscored_type()

    def lane(seat: str, jobs: list) -> None:
        for suite, item, cell, rep in jobs:
            usage: dict = {}

            def ask(messages, _cell=cell, usage=usage):
                text, u = backend.ask(_cell, messages)
                for k, v in u.items():  # an episode asks many times: usage adds up
                    usage[k] = usage.get(k, 0) + (v or 0)
                return text

            row = {"item_id": item.item_id, "suite": suite.name, "suite_revision": suite.revision, "domain": suite.domain,
                   "split": split_of(item.item_id), "cell": cell, "rep": rep, "harness_sha": harness_sha}
            started = time.monotonic()
            try:
                score, _transcript = suite.run(item, ask)
                row.update(status="ok", score=float(score), error=None)
            except SeatLimit as limit:
                with lock:
                    report["limited"][seat] = str(limit)
                return  # this seat is spent for its window: nothing recorded for the refused call
            except Unscored as why:
                row.update(status="unscored", score=None, error=str(why))
            except Exception as e:  # noqa: BLE001 -- recorded with its text, retried next run, never a 0
                row.update(status="error", score=None, error=f"{type(e).__name__}: {e}"[:2000])
            row.update(latency_ms=usage.get("latency_ms") or int((time.monotonic() - started) * 1000),
                       input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"),
                       cost_microusd_list=usage.get("cost_microusd"),
                       measured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
            with lock:
                with out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, sort_keys=True) + "\n")
                key = {"ok": "answered", "error": "errors", "unscored": "unscored"}[row["status"]]
                report[key] += 1

    with ThreadPoolExecutor(max_workers=max(1, len(work))) as pool:
        for future in [pool.submit(lane, seat, jobs) for seat, jobs in work.items()]:
            future.result()
    return report
