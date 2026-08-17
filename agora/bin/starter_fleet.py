#!/usr/bin/env python3
"""starter_fleet.py — generate and validate all sixteen starter charters.

    python bin/starter_fleet.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from _path import bootstrap

bootstrap()

from agora.fleet import render_fleet, starter_charters, validate_all  # noqa: E402


def main() -> int:
    charters = starter_charters()
    results = validate_all(charters)
    print(render_fleet())
    print()

    out = Path(__file__).resolve().parent.parent / "charters"
    out.mkdir(exist_ok=True)
    for charter in charters:
        name = charter.id.rsplit("/", 1)[-1]
        (out / f"{name}.json").write_text(
            json.dumps(charter.to_dict(), indent=2, sort_keys=True) + "\n"
        )

    ok = sum(1 for r in results if r.ok)
    print(f"{ok}/{len(results)} charters validate · written to {out}")
    for result in results:
        if not result.ok:
            print(f"  {result.charter_id}: {'; '.join(result.problems)}")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
