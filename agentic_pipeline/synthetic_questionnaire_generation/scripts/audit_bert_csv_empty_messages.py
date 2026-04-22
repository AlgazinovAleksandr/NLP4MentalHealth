#!/usr/bin/env python3
"""Print how many BERT rows have empty / very short user_message after [SEP]."""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

SEP = " [SEP] "


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    csv_path = root / "data" / "dataset" / "synthetic_triagem_n1000_bert.csv"
    if not csv_path.is_file():
        print(f"Not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    empty = Counter()
    short = Counter()
    for r in rows:
        t = r.get("text") or ""
        if SEP not in t:
            continue
        _, msg = t.split(SEP, 1)
        msg = msg.strip()
        if msg == "":
            empty[r["label"]] += 1
        if len(msg) <= 5:
            short[r["label"]] += 1

    print(f"File: {csv_path}")
    print(f"Rows: {len(rows)}")
    print(f"EMPTY user_message by label: {dict(empty)} (total {sum(empty.values())})")
    print(f"len<=5 by label: {dict(short)} (total {sum(short.values())})")


if __name__ == "__main__":
    main()
