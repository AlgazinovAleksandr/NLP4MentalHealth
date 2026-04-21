#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


LABELS = ("relaxed", "concerned", "urgent")
LABEL2ID = {lab: i for i, lab in enumerate(LABELS)}
ID2LABEL = {i: lab for i, lab in enumerate(LABELS)}


def _eval_metric_key(metric: str) -> str:
    m = metric.strip().lower()
    if m == "f1_macro":
        return "eval_f1_macro"
    if m == "f1_urgent":
        return "eval_f1_urgent"
    if m == "recall_urgent":
        return "eval_recall_urgent"
    if m in ("combined_urgent", "urgent_priority"):
        return "eval_urgent_priority"
    raise ValueError(f"Unknown metric_for_best_model: {metric}")


class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor | None = None, **kwargs):
        self._class_weights = class_weights
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        inputs_fwd = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**inputs_fwd)
        logits = outputs.logits
        if self._class_weights is not None:
            w = self._class_weights.to(logits.device, dtype=logits.dtype)
            loss_fct = torch.nn.CrossEntropyLoss(weight=w)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def load_xy(csv_path: Path) -> tuple[list[str], list[int]]:
    df = pd.read_csv(csv_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise SystemExit("CSV must have columns: text, label")
    texts: list[str] = []
    labels: list[int] = []
    for _, row in df.iterrows():
        lab = str(row["label"]).strip()
        if lab not in LABEL2ID:
            continue
        t = row["text"]
        if not isinstance(t, str) or not t.strip():
            continue
        texts.append(t.strip())
        labels.append(LABEL2ID[lab])
    if len(texts) < 30:
        raise SystemExit(f"Too few rows after filtering: {len(texts)}")
    return texts, labels


def stratified_three_way(
    texts: list[str],
    labels: list[int],
    *,
    test_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[tuple, tuple, tuple]:
    if test_ratio <= 0 or val_ratio <= 0 or test_ratio + val_ratio >= 1.0:
        raise ValueError("Need 0 < test_ratio + val_ratio < 1")

    x_tv, x_te, y_tv, y_te = train_test_split(
        texts,
        labels,
        test_size=test_ratio,
        stratify=labels,
        random_state=seed,
    )
    val_of_tv = val_ratio / (1.0 - test_ratio)
    x_tr, x_va, y_tr, y_va = train_test_split(
        x_tv,
        y_tv,
        test_size=val_of_tv,
        stratify=y_tv,
        random_state=seed,
    )
    return (x_tr, y_tr), (x_va, y_va), (x_te, y_te)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True, help="BERT CSV from scripts/bert_triage_text.py")
    p.add_argument("--output_dir", type=Path, default=Path("runs/triage_bert"))
    p.add_argument("--model_name", type=str, default="bert-base-uncased")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_ratio", type=float, default=0.15)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument(
        "--early_stopping_patience",
        type=int,
        default=2,
        help="0 = off. F1-based early stop only if transformers supports "
        "EarlyStoppingCallback(metric_name=...).",
    )
    p.add_argument(
        "--metric_for_best_model",
        type=str,
        default="f1_macro",
        choices=("f1_macro", "f1_urgent", "recall_urgent", "combined_urgent"),
        help="Checkpoint selection on val. ``combined_urgent`` = 0.65*recall_urgent + 0.35*f1_macro "
        "(favors catching true urgent).",
    )
    p.add_argument(
        "--class_weight_mode",
        type=str,
        default="none",
        choices=("none", "balanced"),
        help="``balanced`` = inverse-frequency CE weights on train labels (often raises urgent recall).",
    )
    p.add_argument(
        "--urgent_weight_mult",
        type=float,
        default=1.0,
        help="Extra multiplier on the urgent class weight (only if class_weight_mode=balanced).",
    )
    args = p.parse_args()

    csv_path = args.csv.resolve()
    if not csv_path.is_file():
        print(f"File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    texts, labels = load_xy(csv_path)
    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = stratified_three_way(
        texts,
        labels,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(
        f"Split sizes — train: {len(y_tr)}, val: {len(y_va)}, test: {len(y_te)} "
        f"(test_ratio={args.test_ratio}, val_ratio={args.val_ratio})",
        flush=True,
    )

    class_weights: torch.Tensor | None = None
    if args.class_weight_mode == "balanced":
        y_np = np.asarray(y_tr, dtype=np.int64)
        cw = compute_class_weight(
            class_weight="balanced",
            classes=np.array([0, 1, 2], dtype=np.int64),
            y=y_np,
        )
        cw = torch.tensor(cw, dtype=torch.float32)
        cw[2] *= float(args.urgent_weight_mult)
        class_weights = cw
        print(
            f"class_weight_mode=balanced tensor (relaxed, concerned, urgent): {cw.tolist()}",
            flush=True,
        )
    elif args.urgent_weight_mult != 1.0:
        print("NOTE: urgent_weight_mult ignored unless class_weight_mode=balanced.", flush=True)

    eval_metric = _eval_metric_key(args.metric_for_best_model)
    print(
        f"metric_for_best_model={args.metric_for_best_model} → {eval_metric}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    print(
        "Note: classifier.weight/bias 'MISSING' in checkpoint logs is normal for "
        "AutoModelForSequenceClassification from a pretrained encoder.",
        flush=True,
    )

    def tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )

    rm = ["text"]
    ds_tr = Dataset.from_dict({"text": x_tr, "labels": y_tr}).map(
        tok, batched=True, remove_columns=rm
    )
    ds_va = Dataset.from_dict({"text": x_va, "labels": y_va}).map(
        tok, batched=True, remove_columns=rm
    )
    ds_te = Dataset.from_dict({"text": x_te, "labels": y_te}).map(
        tok, batched=True, remove_columns=rm
    )
    cols = ["input_ids", "attention_mask", "labels"]
    ds_tr.set_format(type="torch", columns=cols)
    ds_va.set_format(type="torch", columns=cols)
    ds_te.set_format(type="torch", columns=cols)

    def compute_metrics(eval_pred):
        logits, lab = eval_pred
        pred = np.argmax(logits, axis=-1)
        out = {
            "accuracy": float(accuracy_score(lab, pred)),
            "f1_macro": float(f1_score(lab, pred, average="macro", zero_division=0)),
        }
        f1_each = f1_score(
            lab, pred, average=None, labels=[0, 1, 2], zero_division=0
        )
        rec_each = recall_score(
            lab, pred, average=None, labels=[0, 1, 2], zero_division=0
        )
        for i, name in enumerate(LABELS):
            out[f"f1_{name}"] = float(f1_each[i])
            out[f"recall_{name}"] = float(rec_each[i])
        out["urgent_priority"] = float(0.65 * rec_each[2] + 0.35 * out["f1_macro"])
        return out

    steps_per_epoch = max(1, math.ceil(len(ds_tr) / args.batch_size))
    total_opt_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(0.06 * total_opt_steps))

    targs = TrainingArguments(
        output_dir=str(out),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        seed=args.seed,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=eval_metric,
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=max(10, len(ds_tr) // (args.batch_size * 5)),
        report_to="none",
    )

    callbacks = []
    if args.early_stopping_patience > 0:
        es_sig = inspect.signature(EarlyStoppingCallback.__init__)
        if "metric_name" in es_sig.parameters:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=args.early_stopping_patience,
                    metric_name=eval_metric,
                )
            )
        else:
            print(
                "NOTE: EarlyStoppingCallback has no metric_name in this transformers "
                "version — skipping early stopping (still using load_best_model_at_end on "
                f"{eval_metric}). Upgrade transformers for patience-based early stop.",
                flush=True,
            )

    trainer_kw = dict(
        model=model,
        args=targs,
        train_dataset=ds_tr,
        eval_dataset=ds_va,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    tr_sig = inspect.signature(Trainer.__init__)
    if "tokenizer" in tr_sig.parameters:
        trainer_kw["tokenizer"] = tokenizer
    elif "processing_class" in tr_sig.parameters:
        trainer_kw["processing_class"] = tokenizer
    if class_weights is not None:
        trainer = WeightedTrainer(class_weights=class_weights, **trainer_kw)
    else:
        trainer = Trainer(**trainer_kw)

    trainer.train()

    print("\n=== Validation (best checkpoint, last eval) ===", flush=True)
    val_metrics = trainer.evaluate(ds_va)
    for k in sorted(val_metrics.keys()):
        if not k.startswith("_"):
            print(f"  {k}: {val_metrics[k]}", flush=True)

    print("\n=== Test set ===", flush=True)
    pred = trainer.predict(ds_te)
    y_true = np.array(ds_te["labels"])
    y_pred = np.argmax(pred.predictions, axis=-1)
    rec_te = recall_score(
        y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0
    )
    test_metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_per_class": {
            name: float(rec_te[i]) for i, name in enumerate(LABELS)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=list(LABELS),
            zero_division=0,
        ),
    }
    print(test_metrics["classification_report"], flush=True)
    print("confusion_matrix [rows=true, cols=pred] order relaxed/concerned/urgent:")
    print(np.array(test_metrics["confusion_matrix"]), flush=True)

    metrics_path = out / "metrics.json"
    payload = {
        "csv": str(csv_path),
        "model_name": args.model_name,
        "metric_for_best_model": args.metric_for_best_model,
        "class_weight_mode": args.class_weight_mode,
        "urgent_weight_mult": args.urgent_weight_mult,
        "splits": {"train": len(y_tr), "val": len(y_va), "test": len(y_te)},
        "val_metrics": {k: v for k, v in val_metrics.items() if not k.startswith("_")},
        "test_metrics": {
            "accuracy": test_metrics["accuracy"],
            "f1_macro": test_metrics["f1_macro"],
            "recall_per_class": test_metrics["recall_per_class"],
            "confusion_matrix": test_metrics["confusion_matrix"],
        },
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {metrics_path}", flush=True)

    trainer.save_model(str(out / "best_model"))
    tokenizer.save_pretrained(str(out / "best_model"))
    print(f"Saved model + tokenizer → {out / 'best_model'}", flush=True)


if __name__ == "__main__":
    main()
