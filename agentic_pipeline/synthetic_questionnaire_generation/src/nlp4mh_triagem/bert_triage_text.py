#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

DEFAULT_SEP = " [SEP] "


def serialize_answers_for_bert(answers: dict) -> str:
    return json.dumps(answers, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def build_bert_triage_text(
    answers: dict,
    user_message: str | None,
    *,
    sep: str | None = None,
) -> str:
    s = sep if sep is not None else os.environ.get("BERT_TEXT_SEP", DEFAULT_SEP)
    body = serialize_answers_for_bert(answers)
    msg = (user_message or "").strip()
    return body + s + msg


def load_jsonl_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def export_csv(
    rows: list[dict],
    csv_path: Path,
    *,
    sep: str | None = None,
    include_sample_id: bool = True,
) -> int:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fields = ["text", "label", "label_from_rules"]
        if include_sample_id:
            fields.insert(0, "sample_id")
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            ans = row.get("answers")
            if not isinstance(ans, dict):
                continue
            msg = row.get("user_message")
            if msg is None:
                msg = row.get("persona_description")
            text = build_bert_triage_text(ans, str(msg) if msg is not None else "", sep=sep)
            lab = row.get("label")
            if not lab:
                continue
            lfr = row.get("label_from_rules")
            lfr_s = "" if lfr is None else str(lfr).strip()
            out = {"text": text, "label": lab, "label_from_rules": lfr_s}
            if include_sample_id:
                out["sample_id"] = row.get("sample_id", "")
            w.writerow(out)
            n += 1
    return n


def export_jsonl(rows: list[dict], jsonl_path: Path, *, sep: str | None = None) -> int:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            ans = row.get("answers")
            if not isinstance(ans, dict):
                continue
            msg = row.get("user_message")
            if msg is None:
                msg = row.get("persona_description")
            lab = row.get("label")
            if not lab:
                continue
            text = build_bert_triage_text(ans, str(msg) if msg is not None else "", sep=sep)
            lfr = row.get("label_from_rules")
            obj: dict = {"sample_id": row.get("sample_id"), "text": text, "label": lab}
            if lfr is not None and str(lfr).strip():
                obj["label_from_rules"] = lfr
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser(description="Export BERT text = answers JSON + SEP + user_message")
    p.add_argument("jsonl", type=Path, help="Input dataset .jsonl")
    p.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: same as input)",
    )
    p.add_argument(
        "--stem",
        type=str,
        default=None,
        help="Output basename without extension (default: input stem + _bert)",
    )
    p.add_argument(
        "--sep",
        type=str,
        default=None,
        help="Override separator between answers and user_message",
    )
    p.add_argument("--jsonl-out", action="store_true", help="Also write .jsonl with text+label")
    args = p.parse_args()

    inp = args.jsonl.resolve()
    if not inp.is_file():
        print(f"Not a file: {inp}", file=sys.stderr)
        sys.exit(1)

    out_dir = (args.out_dir or inp.parent).resolve()
    stem = args.stem or (inp.stem + "_bert")
    csv_path = out_dir / f"{stem}.csv"
    rows = load_jsonl_rows(inp)
    n_csv = export_csv(rows, csv_path, sep=args.sep, include_sample_id=True)
    print(f"Wrote {n_csv} rows → {csv_path}", flush=True)
    if args.jsonl_out:
        jl = out_dir / f"{stem}.jsonl"
        n_jl = export_jsonl(rows, jl, sep=args.sep)
        print(f"Wrote {n_jl} rows → {jl}", flush=True)


if __name__ == "__main__":
    main()
