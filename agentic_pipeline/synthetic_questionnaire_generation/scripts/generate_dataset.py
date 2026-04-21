#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import re
import signal
import sys
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path

_PKG = Path(__file__).resolve().parent.parent
DATASET_DIR = _PKG / "data" / "dataset"

sys.path.insert(0, str(_PKG / "src"))

from nlp4mh_triagem.generation_common import (  # noqa: E402
    build_batch_prompt,
    call_openrouter,
    extract_json_array,
    load_dotenv_layered,
)

_LABEL_ORDER = ("relaxed", "concerned", "urgent")
_VALID_JOINT = frozenset(_LABEL_ORDER)


def label_mode_from_env() -> str:
    v = (os.environ.get("DATASET_LABEL_MODE") or "joint").strip().lower()
    if v in ("rules", "answers", "formula"):
        return "rules"
    return "joint"


def parse_joint_label_from_item(item: dict) -> str | None:
    for k in ("joint_label", "joint_routing_label", "expected_joint_label"):
        v = item.get(k)
        if isinstance(v, str):
            s = v.strip()
            if s in _VALID_JOINT:
                return s
    return None


def split_target_n_triadic(n: int, weights: tuple[float, float, float] = (0.6, 0.3, 0.1)) -> dict[str, int]:
    w0, w1, w2 = weights
    exact = [n * w0, n * w1, n * w2]
    out = [int(x) for x in exact]
    rem = n - sum(out)
    order = sorted(range(3), key=lambda i: exact[i] - out[i], reverse=True)
    for k in range(rem):
        out[order[k]] += 1
    return dict(zip(_LABEL_ORDER, out))


def label_caps_from_env(target_n: int) -> tuple[dict[str, int], int]:
    cr = os.environ.get("DATASET_CAP_RELAXED", "").strip()
    cc = os.environ.get("DATASET_CAP_CONCERNED", "").strip()
    cu = os.environ.get("DATASET_CAP_URGENT", "").strip()
    if cr and cc and cu:
        caps = {"relaxed": int(cr), "concerned": int(cc), "urgent": int(cu)}
        s = sum(caps.values())
        if s != target_n:
            print(
                f"WARNING: DATASET_CAP_* sum is {s}, DATASET_TARGET_N is {target_n}. "
                f"Using effective dataset size {s}.",
                flush=True,
            )
        return caps, s
    caps = split_target_n_triadic(target_n)
    return caps, target_n

load_dotenv_layered()
from nlp4mh_triagem.questionnaire_validate import validate_answers  # noqa: E402

DATASET_MODELS = [
    "google/gemini-2.0-flash-001",
    "qwen/qwen-2.5-72b-instruct",
    "openai/gpt-4o-mini",
]


def answers_fingerprint(answers: dict) -> str:
    return json.dumps(answers, sort_keys=True, ensure_ascii=False)


def dedupe_mode_from_env() -> str:
    v = (os.environ.get("DATASET_DEDUPE") or "answers_and_message").strip().lower()
    if v in ("answers", "answers_only", "a"):
        return "answers_only"
    return "answers_and_message"


def row_fingerprint(answers: dict, user_message: str, mode: str) -> str:
    if mode == "answers_only":
        return answers_fingerprint(answers)
    key = {
        "answers": answers,
        "user_message": user_message.strip().casefold()[:12000],
    }
    return json.dumps(key, sort_keys=True, ensure_ascii=False)


def resolve_dataset_prompt_path() -> Path:
    override = os.environ.get("DATASET_PROMPT_FILE", "").strip()
    if override:
        p = Path(override)
        if not p.is_absolute():
            p = _PKG / p
        if p.is_file():
            return p
        print(f"WARNING: DATASET_PROMPT_FILE not found ({p}); using default.", flush=True)
    default = _PKG / "english_psych_user_message_dataset_prompt.txt"
    if default.is_file():
        return default
    return _PKG / "test_questionnaire_prompt.txt"


def parse_sample_index(sample_id: str) -> int | None:
    m = re.match(r"^ds_(\d+)$", str(sample_id))
    return int(m.group(1)) if m else None


def output_stem(target_n: int) -> str:
    base = (os.environ.get("DATASET_FILE_STEM") or "synthetic_triagem").strip() or "synthetic_triagem"
    return f"{base}_n{target_n}"


def dataset_paths(target_n: int) -> tuple[Path, Path, Path]:
    stem = output_stem(target_n)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    return (
        DATASET_DIR / f"{stem}.jsonl",
        DATASET_DIR / f"{stem}.csv",
        DATASET_DIR / f"{stem}_manifest.json",
    )


def load_existing_dataset(
    jsonl_path: Path,
    models: list[str],
    dedupe_mode: str,
) -> tuple[list[dict], set[str], dict[str, int], int]:
    if not jsonl_path.is_file():
        return [], set(), {m: 0 for m in models}, 1
    rows: list[dict] = []
    seen: set[str] = set()
    counts = {m: 0 for m in models}
    max_idx = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(row)
            ans = row.get("answers")
            if isinstance(ans, dict):
                msg = str(row.get("user_message") or row.get("persona_description") or "")
                seen.add(row_fingerprint(ans, msg, dedupe_mode))
            m = row.get("source_model")
            if m in counts:
                counts[m] += 1
            idx = parse_sample_index(str(row.get("sample_id", "")))
            if idx is not None:
                max_idx = max(max_idx, idx)
    return rows, seen, counts, max_idx + 1 if max_idx else 1


def build_canonical_row(
    *,
    sample_id: str,
    source_model: str,
    answers: dict,
    user_message: str,
    item: dict | None = None,
    label_mode: str = "joint",
) -> dict | None:
    vr = validate_answers(answers)
    if not vr.ok or vr.computed_score is None or vr.computed_flag is None:
        return None
    if label_mode == "rules":
        lab = vr.computed_flag
    else:
        lab = parse_joint_label_from_item(item or {})
        if lab is None:
            return None
    return {
        "sample_id": sample_id,
        "source_model": source_model,
        "label": lab,
        "label_from_rules": vr.computed_flag,
        "total_score": vr.computed_score,
        "user_message": user_message,
        "answers": answers,
    }


def write_jsonl_csv(rows: list[dict], jsonl_path: Path, csv_path: Path) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(
            [
                "sample_id",
                "source_model",
                "label",
                "label_from_rules",
                "total_score",
                "user_message",
                "answers_json",
            ]
        )
        for row in rows:
            um = row.get("user_message")
            if um is None and row.get("persona_description") is not None:
                um = row.get("persona_description")
            w.writerow(
                [
                    row.get("sample_id", ""),
                    row.get("source_model", ""),
                    row.get("label", ""),
                    row.get("label_from_rules", ""),
                    row.get("total_score", ""),
                    um if um is not None else "",
                    json.dumps(row.get("answers", {}), ensure_ascii=False),
                ]
            )


def run_generation() -> None:
    target_n = int(os.environ.get("DATASET_TARGET_N", "1000"))
    batch_max = int(os.environ.get("DATASET_BATCH_MAX", "15"))
    temperature = float(os.environ.get("DATASET_TEMPERATURE", "0.35"))
    max_batches = int(os.environ.get("DATASET_MAX_BATCHES_PER_MODEL", "400"))
    max_tokens = int(os.environ.get("DATASET_MAX_TOKENS", "24576"))
    fresh = os.environ.get("DATASET_FRESH_START", "").strip().lower() in ("1", "true", "yes")
    resume_off = os.environ.get("DATASET_RESUME", "").strip().lower() in ("0", "false", "no")

    models = list(DATASET_MODELS)
    label_caps, effective_n = label_caps_from_env(target_n)
    dedupe_mode = dedupe_mode_from_env()
    label_mode = label_mode_from_env()
    jsonl_path, csv_path, manifest_path = dataset_paths(effective_n)
    prompt_path = resolve_dataset_prompt_path()

    rows: list[dict] = []
    seen: set[str] = set()
    per_model: dict[str, int] = {m: 0 for m in models}
    next_idx = 1
    auto_resumed = False

    if fresh:
        print("DATASET_FRESH_START: starting empty (existing jsonl will be overwritten).", flush=True)
    elif resume_off:
        print("DATASET_RESUME=0: starting empty (existing file will be overwritten).", flush=True)
    elif jsonl_path.is_file():
        rows, seen, per_model, next_idx = load_existing_dataset(
            jsonl_path, models, dedupe_mode
        )
        label_counts = Counter(r["label"] for r in rows)
        over = [lab for lab in _LABEL_ORDER if label_counts.get(lab, 0) > label_caps[lab]]
        if over:
            print(
                "ERROR: existing jsonl already exceeds label caps "
                f"(counts={dict(label_counts)}, caps={label_caps}; over={over}). "
                "Use DATASET_FRESH_START=1 or adjust DATASET_CAP_* / DATASET_TARGET_N.",
                flush=True,
            )
            sys.exit(1)
        incomplete = any(label_counts.get(lab, 0) < label_caps[lab] for lab in _LABEL_ORDER) or len(
            rows
        ) < effective_n
        if not incomplete:
            print(
                f"Dataset already complete ({len(rows)} rows, label caps satisfied). "
                "Set DATASET_FRESH_START=1 to rebuild from scratch.",
                flush=True,
            )
            write_jsonl_csv(rows, jsonl_path, csv_path)
            labels = Counter(r["label"] for r in rows)
            manifest_path.write_text(
                json.dumps(
                    {
                        "dataset_target_n_requested": target_n,
                        "dataset_effective_n": effective_n,
                        "label_caps": label_caps,
                        "status": "complete_no_api_calls",
                        "collected_total": len(rows),
                        "per_model_counts": per_model,
                        "label_counts": dict(labels),
                        "prompt_file": str(prompt_path.name),
                        "dedupe_mode": dedupe_mode,
                        "label_mode": label_mode,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            return
        auto_resumed = True
        print(
            f"Resuming: loaded {len(rows)} rows, next sample_id index {next_idx}. "
            f"Per-model: {per_model}, label_counts: {dict(label_counts)}, caps: {label_caps}",
            flush=True,
        )

    skipped_invalid = 0
    skipped_duplicate = 0
    skipped_label_full = 0
    batches_used: dict[str, int] = {m: 0 for m in models}
    label_counts = Counter(r["label"] for r in rows)

    base_prompt = prompt_path.read_text(encoding="utf-8")
    print(
        f"Using prompt: {prompt_path.name} (dedupe={dedupe_mode}, label_mode={label_mode})",
        flush=True,
    )
    prompt_id = next_idx  # disposable ids for LLM persona_id range
    model_idx = 0
    no_accept_batches = 0
    checkpoint_batches = os.environ.get("DATASET_CHECKPOINT_OFF", "").strip().lower() not in (
        "1",
        "true",
        "yes",
    )

    def persist_disk(status: str, note: str | None = None) -> None:
        write_jsonl_csv(rows, jsonl_path, csv_path)
        labels = Counter(r["label"] for r in rows)
        manifest = {
            "dataset_target_n_requested": target_n,
            "dataset_effective_n": effective_n,
            "label_caps": label_caps,
            "models": models,
            "status": status,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "collected_total": len(rows),
            "per_model_counts": per_model,
            "label_counts": dict(labels),
            "skipped_invalid": skipped_invalid,
            "skipped_duplicate": skipped_duplicate,
            "skipped_label_full": skipped_label_full,
            "prompt_file": str(prompt_path.relative_to(_PKG)),
            "dedupe_mode": dedupe_mode,
            "label_mode": label_mode,
            "batches_per_model": batches_used,
            "batch_max": batch_max,
            "temperature": temperature,
            "max_tokens_per_request": max_tokens,
            "auto_resumed_from_file": auto_resumed,
            "fresh_start": fresh,
            "jsonl": str(jsonl_path.relative_to(_PKG)),
            "csv": str(csv_path.relative_to(_PKG)),
        }
        if note:
            manifest["last_save_note"] = note
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        suffix = f" ({note})" if note else ""
        print(f"Saved {len(rows)} rows → {jsonl_path.name}{suffix}", flush=True)
        if note == "finished":
            print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)

    def _signal_handler(signum: int, _frame) -> None:
        persist_disk("interrupted", f"signal={signum}")
        raise SystemExit(130 if signum == signal.SIGINT else 128 + signum)

    signal.signal(signal.SIGINT, _signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _signal_handler)

    def labels_still_needed() -> bool:
        return any(label_counts[lab] < label_caps[lab] for lab in _LABEL_ORDER)

    while labels_still_needed():
        model = None
        for _ in range(len(models)):
            cand = models[model_idx % len(models)]
            model_idx += 1
            if batches_used[cand] < max_batches:
                model = cand
                break
        if model is None:
            print("WARNING: every model hit DATASET_MAX_BATCHES_PER_MODEL; stopping early.", flush=True)
            break

        ask = batch_max
        start_id = prompt_id
        end_id = prompt_id + ask - 1
        prompt = build_batch_prompt(
            base_prompt, start_id, end_id, ask, distribution_profile="dataset"
        )
        batch_accepted = 0
        try:
            text = call_openrouter(
                model, prompt, temperature=temperature, max_tokens=max_tokens
            )
            data = extract_json_array(text)
        except Exception as e:
            print(f"[{model}] batch error: {e}", flush=True)
            no_accept_batches += 1
            if no_accept_batches >= len(models) * 5:
                break
            continue
        batches_used[model] += 1
        prompt_id = end_id + 1

        if not isinstance(data, list):
            no_accept_batches += 1
            if no_accept_batches >= 80:
                break
            continue

        for item in data:
            if not labels_still_needed():
                break
            if not isinstance(item, dict):
                skipped_invalid += 1
                continue
            ans = item.get("answers")
            if not isinstance(ans, dict):
                skipped_invalid += 1
                continue
            msg = str(
                item.get("user_message")
                or item.get("persona_description")
                or ""
            )
            fp = row_fingerprint(ans, msg, dedupe_mode)
            if fp in seen:
                skipped_duplicate += 1
                continue
            sid = f"ds_{next_idx:06d}"
            canon = build_canonical_row(
                sample_id=sid,
                source_model=model,
                answers=ans,
                user_message=msg,
                item=item,
                label_mode=label_mode,
            )
            if canon is None:
                skipped_invalid += 1
                continue
            lab = canon["label"]
            if label_counts[lab] >= label_caps[lab]:
                skipped_label_full += 1
                continue
            seen.add(fp)
            rows.append(canon)
            per_model[model] += 1
            label_counts[lab] += 1
            next_idx += 1
            batch_accepted += 1

        if batch_accepted == 0:
            no_accept_batches += 1
            if no_accept_batches >= 80:
                print("WARNING: too many batches with no accepted rows; stopping.", flush=True)
                break
        else:
            no_accept_batches = 0
            if checkpoint_batches and batch_accepted > 0:
                persist_disk("in_progress", "after_batch")

    incomplete_caps = any(label_counts[lab] < label_caps[lab] for lab in _LABEL_ORDER)
    if incomplete_caps:
        print(
            f"WARNING: incomplete label mix: {dict(label_counts)} vs caps {label_caps}.",
            flush=True,
        )

    persist_disk(
        "complete" if not incomplete_caps else "stopped_early",
        "finished",
    )
    print(f"Wrote {len(rows)} rows to {jsonl_path}", flush=True)


if __name__ == "__main__":
    run_generation()
