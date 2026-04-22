#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

from nlp4mh_triagem.bert_triage_text import DEFAULT_SEP, build_bert_triage_text  # noqa: E402

DEFAULT_ANSWERS: dict = {
    "q_gender": "male",
    "q_age": 21,
    "q_occupation": "student",
    "q_has_mental_concern": "fine",
    "q_phq2_1": 3,
    "q_phq2_2": 3,
    "q_gad2_1": 3,
    "q_gad2_2": 3,
    "q_daily_impact": "not_at_all",
    "q_prior_help": "never",
    "q_app_goal": ["just_curious"],
}

DEFAULT_USER_MESSAGE = "i m kill myself"


def _split_model_input(text: str) -> tuple[str, str]:
    sep = os.environ.get("BERT_TEXT_SEP", DEFAULT_SEP)
    if sep in text:
        a, b = text.split(sep, 1)
        return a.strip(), b.strip()
    return text, ""


def _norm_id2label(cfg) -> dict[int, str]:
    raw = getattr(cfg, "id2label", None) or {}
    out: dict[int, str] = {}
    for k, v in raw.items():
        out[int(k)] = str(v)
    return out


def read_answers_interactive() -> dict | None:
    prompt = "\nAnswers: .json path | one-line JSON | code | help | quit: "
    line = input(prompt).strip()
    if not line or line.lower() in ("quit", "exit", "q"):
        return None
    if line.lower() in ("code", "defaults", "default"):
        return dict(DEFAULT_ANSWERS)
    if line.lower() == "help":
        print(
            "\nExample:\n",
            json.dumps(DEFAULT_ANSWERS, separators=(",", ":"), ensure_ascii=False),
            "\n`code` — DEFAULT_ANSWERS.\n",
            flush=True,
        )
        return read_answers_interactive()

    p = Path(line)
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}", flush=True)
        return read_answers_interactive()
    if not isinstance(obj, dict):
        print("JSON must be an object {...}", flush=True)
        return read_answers_interactive()
    return obj


def read_user_message_interactive() -> str:
    print(
        "User message (multi-line). End with a line `.`\nEmpty first line + `.` → DEFAULT_USER_MESSAGE.",
        flush=True,
    )
    lines: list[str] = []
    while True:
        try:
            ln = input()
        except EOFError:
            break
        if ln.strip() == ".":
            break
        lines.append(ln)
    body = "\n".join(lines).strip()
    if not body:
        return DEFAULT_USER_MESSAGE
    return body


@torch.inference_mode()
def predict(
    model,
    tokenizer,
    id2label: dict[int, str],
    text: str,
    *,
    max_length: int,
    device: torch.device,
) -> tuple[str, list[tuple[str, float]]]:
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    logits = model(**enc).logits[0]
    probs = torch.softmax(logits, dim=-1).cpu().tolist()
    pred_i = int(torch.argmax(logits).item())
    ranked = sorted(
        ((id2label[i], probs[i]) for i in range(len(probs))),
        key=lambda x: -x[1],
    )
    return id2label[pred_i], ranked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True, help="Checkpoint dir, e.g. runs/.../best_model")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--from-code", action="store_true", help="Run once with DEFAULT_* then exit")
    ap.add_argument("--verbose", action="store_true", help="Print model input preview")
    args = ap.parse_args()

    mp = args.model.resolve()
    if not mp.is_dir():
        print(f"Not a directory: {mp}", file=sys.stderr)
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(str(mp))
    model = AutoModelForSequenceClassification.from_pretrained(str(mp))
    model.eval()
    id2label = _norm_id2label(model.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Loaded model from {mp}", flush=True)
    print(f"Device: {device}. Labels: {id2label}", flush=True)

    def run_one(ans: dict, msg: str, *, show_input: bool = False) -> None:
        text = build_bert_triage_text(ans, msg)
        label, ranked = predict(
            model,
            tokenizer,
            id2label,
            text,
            max_length=args.max_length,
            device=device,
        )
        print("\n--- BERT prediction ---", flush=True)
        print(f"label: {label}", flush=True)
        for name, p in ranked:
            print(f"  {name}: {p:.4f}", flush=True)
        print(f"\n(char length of model input: {len(text)})", flush=True)

        try:
            from nlp4mh_triagem.questionnaire_validate import validate_answers

            vr = validate_answers(ans)
            if vr.ok and vr.computed_flag is not None:
                print("\n--- Rules (answers only) ---", flush=True)
                print(
                    f"  {vr.computed_flag}  (total_score={vr.computed_score})",
                    flush=True,
                )
        except Exception as e:
            print(f"(Could not run validate_answers: {e})", flush=True)

        if show_input:
            _answers_part, tail = _split_model_input(text)
            print("\n--- Model input ---", flush=True)
            print(f"  answers JSON length: {len(_answers_part)} chars", flush=True)
            if not (msg or "").strip():
                print("  user_message: <empty>", flush=True)
            else:
                preview = tail[:500] + ("…" if len(tail) > 500 else "")
                print(f"  user_message ({len(tail)} chars): {preview!r}", flush=True)
            head = text[:350] + ("…" if len(text) > 350 else "")
            print(f"  full input start: {head!r}", flush=True)

    if args.from_code:
        if not str(DEFAULT_USER_MESSAGE).strip():
            print("WARNING: DEFAULT_USER_MESSAGE is empty.", flush=True)
        run_one(
            dict(DEFAULT_ANSWERS),
            DEFAULT_USER_MESSAGE,
            show_input=True,
        )
        return

    while True:
        ans = read_answers_interactive()
        if ans is None:
            print("Bye.", flush=True)
            break
        msg = read_user_message_interactive()
        run_one(ans, msg, show_input=args.verbose)


if __name__ == "__main__":
    main()
