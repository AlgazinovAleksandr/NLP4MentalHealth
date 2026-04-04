#!/usr/bin/env python3
"""
Generate synthetic questionnaires via OpenRouter (multiple models), validate against Bogdan's schema,
write per-model JSONL/CSV under data/generated/ and update MODEL_COMPARISON_REPORT.md.

The prompt file test_questionnaire_prompt.txt is never modified; batch size and persona_id ranges
are appended at runtime when generating more than 15 rows.

Env:
  API_KEY          — OpenRouter key (required)
  BASE_URL         — default https://openrouter.ai/api/v1
  SYNTH_MODELS     — comma-separated override of model list
  SYNTH_TARGET_N   — total questionnaires per model (default 50)
  SYNTH_BATCH_MAX  — max rows per API call (default 15; use 10 if JSON truncates on 32k-context providers)
  SYNTH_MAX_TOKENS — completion budget (default 28000; must leave room for prompt within model context)
  SYNTH_FORCE_REGEN — если 1/true: перегенерировать даже при существующем jsonl (по умолчанию пропуск)
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

_DIR = Path(__file__).resolve().parent
_REPO_PIPELINE = _DIR.parent
DATA_DIR = _DIR / "data" / "generated"
load_dotenv(_REPO_PIPELINE / ".env")
load_dotenv(_DIR / ".env")

sys.path.insert(0, str(_DIR))

from questionnaire_validate import metric_matches_for_row, validate_row  # noqa: E402

try:
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI
except ImportError:
    print("Install dependencies: pip install -r ../requirements.txt", file=sys.stderr)
    raise

# Models that succeeded in the last project run + DeepSeek (OpenRouter slugs).
DEFAULT_MODELS = [
    "openai/gpt-4o-mini",
    "google/gemini-2.0-flash-001",
    "meta-llama/llama-3.3-70b-instruct",
    "qwen/qwen-2.5-72b-instruct",
    "mistralai/mistral-small-3.1-24b-instruct",
    "deepseek/deepseek-chat",
    "z-ai/glm-4.5-air",
]

# Порядок строк и субъективные баллы — как в MODEL_COMPARISON_REPORT.md (не редактировать отчёт вручную под скрипт).
REPORT_TABLE_MODEL_ORDER = [
    "google/gemini-2.0-flash-001",
    "openai/gpt-4o-mini",
    "qwen/qwen-2.5-72b-instruct",
    "deepseek/deepseek-chat",
    "meta-llama/llama-3.3-70b-instruct",
    "mistralai/mistral-small-3.1-24b-instruct",
    "z-ai/glm-4.5-air",
]
REPORT_SUBJECTIVE_SCORES = {
    "google/gemini-2.0-flash-001": "**7.5**",
    "openai/gpt-4o-mini": "**6.5**",
    "qwen/qwen-2.5-72b-instruct": "**7.5**",
    "deepseek/deepseek-chat": "**6.5**",
    "meta-llama/llama-3.3-70b-instruct": "**6.5**",
    "mistralai/mistral-small-3.1-24b-instruct": "**5.5**",
    "z-ai/glm-4.5-air": "**7.0**",
}

REPORT_RASCHIFROVKA = """## Расшифровка

| Колонка | Смысл |
|---------|--------|
| **Schema** | Доля строк, где `answers` проходят проверку кодов, шкал, skip-logic (`questionnaire_validate.py`). |
| **Consistency** | Доля строк, где **и** `expected_total_score`, **и** `expected_flag` совпали с пересчётом по правилам из `registration_questionnaire_v0.0.1.json`. Низкая consistency при высокой schema обычно значит: ответы ок, модель ошиблась в «подписи» к метрике. |
| **Flag OK** | Среди строк с валидной schema: доля, где пересчитанный **флаг** (`relaxed` / `concerned` / `urgent`) совпал с `expected_flag`. |
| **Субъективно /10** | По качеству данных в строках (персоны, согласованность с answers); число строк не штрафуется. Не замена метрикам. |"""

REPORT_VIVOD = """## Вывод

По объёму прогона (50 строк) сильнее **Gemini** и **Qwen**; по субъективной оценке содержимого (без учёта N) **GLM** близок к лидерам за счёт **Flag OK** и чистой **schema** на выборке. **gpt-4o-mini** сильнее по **consistency** и **Flag OK** на валидных строках, но даёт **6 битых** записей по схеме. **`expected_total_score` из LLM ненадёжен** у всех — считать балл и флаг кодом. **Mistral** и **GLM** — неполный файл; для фиксированного N это ограничение. Эталон — **`questionnaire_sample_1.csv`**."""

PROMPT_FILE = _DIR / "test_questionnaire_prompt.txt"
REPORT_FILE = _DIR / "MODEL_COMPARISON_REPORT.md"

BATCH_WRAPPER = """
---
AUTOMATION OVERRIDE (this request only):
The specification above is unchanged. Ignore the literal counts "15" and persona_id range "test_001"…"test_015"
in the TASK and OUTPUT FORMAT sections for THIS response only.

Return a JSON array of exactly {batch_n} objects.
Use persona_id strings "test_{start:03d}" through "test_{end:03d}" (inclusive, three-digit zero-padded).
Keep the same option codes, skip logic, scoring rules, and output fields as in the specification.

For this batch of {batch_n} rows, target this flag distribution (counts are integers summing to {batch_n}):
  — relaxed: {n_relaxed}
  — concerned: {n_concerned}
  — urgent: {n_urgent} (include "self_harm" in q_concern_areas where the persona is urgent)
  — edge cases (borderline total_score 2 or 3): {n_edge}

Maintain demographic and mental-health profile diversity as in the TASK section.
If the specification asks for a markdown validation table after the JSON array, you may append it after the closing "]" of the array.
"""


def load_prompt() -> str:
    return PROMPT_FILE.read_text(encoding="utf-8")


def distribution_targets(batch_n: int) -> tuple[int, int, int, int]:
    """Hamilton: relaxed, concerned, urgent, edge from ratio 4:7:2:2 over 15."""
    nums = [4, 7, 2, 2]
    d = 15
    exact = [nums[i] * batch_n / d for i in range(4)]
    out = [int(x) for x in exact]
    rem = batch_n - sum(out)
    order = sorted(range(4), key=lambda i: exact[i] - out[i], reverse=True)
    for k in range(rem):
        out[order[k]] += 1
    return (out[0], out[1], out[2], out[3])


def build_batch_prompt(base: str, start_id: int, end_id: int, batch_n: int) -> str:
    r, c, u, e = distribution_targets(batch_n)
    assert r + c + u + e == batch_n
    suffix = BATCH_WRAPPER.format(
        batch_n=batch_n,
        start=start_id,
        end=end_id,
        n_relaxed=r,
        n_concerned=c,
        n_urgent=u,
        n_edge=e,
    )
    return base.rstrip() + "\n" + suffix


def extract_json_array(text: str) -> list:
    """Parse first JSON array from model output (handles ```json fences)."""
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", t)
    if fence:
        t = fence.group(1).strip()
    start = t.find("[")
    if start < 0:
        raise ValueError("No JSON array start '[' found")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(t)):
        c = t[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
            continue
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return json.loads(t[start : i + 1])
    raise ValueError("Unclosed JSON array")


def persona_sort_key(row: dict) -> int:
    pid = str(row.get("persona_id", ""))
    m = re.match(r"test_(\d+)$", pid)
    return int(m.group(1)) if m else 10**9


@dataclass
class ModelRunStats:
    model: str
    parse_error: str | None = None
    batch_errors: list[str] = field(default_factory=list)
    n_rows: int = 0
    target_n: int = 0
    rows_valid_schema: int = 0
    rows_full_consistency: int = 0
    rows_score_match: int = 0
    rows_flag_match: int = 0
    rows_metric_eligible: int = 0  # schema-valid rows used for score/flag rates
    errors_sample: list[str] = field(default_factory=list)

    @property
    def schema_rate(self) -> float:
        return self.rows_valid_schema / self.n_rows if self.n_rows else 0.0

    @property
    def consistency_rate(self) -> float:
        return self.rows_full_consistency / self.n_rows if self.n_rows else 0.0

    @property
    def score_rate(self) -> float:
        return self.rows_score_match / self.rows_metric_eligible if self.rows_metric_eligible else 0.0

    @property
    def flag_rate(self) -> float:
        return self.rows_flag_match / self.rows_metric_eligible if self.rows_metric_eligible else 0.0


def call_model(model: str, prompt: str, temperature: float = 0.25) -> str:
    api_key = os.environ.get("API_KEY")
    base_url = os.environ.get("BASE_URL", "https://openrouter.ai/api/v1")
    if not api_key:
        raise RuntimeError("API_KEY is not set (OpenRouter key)")
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        timeout=600,
        max_retries=2,
        max_tokens=int(os.environ.get("SYNTH_MAX_TOKENS", "28000")),
    )
    msg = llm.invoke([HumanMessage(content=prompt)])
    return msg.content if hasattr(msg, "content") else str(msg)


def evaluate_rows(data: list[dict]) -> tuple[int, int, int, int, int, list[str]]:
    """Returns schema_ok, full_consistency, score_match, flag_match, metric_eligible, error_samples."""
    schema_ok = full_consistent = score_m = flag_m = eligible = 0
    collected: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            collected.append("non-object row")
            continue
        ans = item.get("answers")
        if not isinstance(ans, dict):
            collected.append(f"{item.get('persona_id')}: missing answers object")
            continue

        vr_schema = validate_row({**item, "answers": ans}, check_expected=False)
        if vr_schema.ok:
            schema_ok += 1
            sm, fm = metric_matches_for_row({**item, "answers": ans})
            if sm is not None:
                eligible += 1
                if sm:
                    score_m += 1
                if fm:
                    flag_m += 1

        vr_full = validate_row({**item, "answers": ans}, check_expected=True)
        if vr_full.ok:
            full_consistent += 1
        elif vr_schema.ok:
            collected.extend(vr_full.errors[:2])

    return schema_ok, full_consistent, score_m, flag_m, eligible, collected[:12]


def run_single_api_call(model: str, prompt: str) -> tuple[list[dict] | None, str | None]:
    try:
        text = call_model(model, prompt)
    except Exception as e:
        return None, f"API error: {e}"
    try:
        data = extract_json_array(text)
    except Exception as e:
        return None, f"JSON parse: {e}"
    if not isinstance(data, list):
        return None, "Top-level JSON is not an array"
    return data, None


def run_model_batched(model: str, base_prompt: str, target_n: int, batch_max: int) -> tuple[ModelRunStats, list[dict] | None]:
    st = ModelRunStats(model=model, target_n=target_n)
    merged: list[dict] = []
    start_id = 1
    remaining = target_n

    while remaining > 0:
        batch_n = min(batch_max, remaining)
        end_id = start_id + batch_n - 1
        prompt = build_batch_prompt(base_prompt, start_id, end_id, batch_n)
        data, err = run_single_api_call(model, prompt)
        if err:
            st.batch_errors.append(f"batch test_{start_id:03d}-test_{end_id:03d}: {err}")
            st.parse_error = err
            break
        merged.extend(data)
        start_id = end_id + 1
        remaining -= batch_n

    st.n_rows = len(merged)
    if not merged:
        return st, None

    sch, fc, sm, fm, el, samples = evaluate_rows(merged)
    st.rows_valid_schema = sch
    st.rows_full_consistency = fc
    st.rows_score_match = sm
    st.rows_flag_match = fm
    st.rows_metric_eligible = el
    st.errors_sample = samples

    merged.sort(key=persona_sort_key)
    return st, merged


def model_file_stem(model: str, target_n: int) -> str:
    safe = model.replace("/", "_").replace(":", "_")
    return f"generated_{safe}_n{target_n}"


def output_paths(model: str, target_n: int, out_dir: Path | None = None) -> tuple[Path, Path]:
    # Нельзя использовать Path.with_suffix: у slug модели бывают точки (gemini-2.0-flash-001),
    # и тогда suffix режется по последней точке → неверный путь и срыв кэша.
    root = out_dir or DATA_DIR
    stem = model_file_stem(model, target_n)
    return root / f"{stem}.jsonl", root / f"{stem}.csv"


def load_cached_jsonl(jsonl_path: Path) -> list[dict] | None:
    if not jsonl_path.is_file() or jsonl_path.stat().st_size == 0:
        return None
    rows: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows if rows else None


def stats_from_evaluated_data(model: str, target_n: int, data: list[dict]) -> ModelRunStats:
    st = ModelRunStats(model=model, target_n=target_n)
    st.n_rows = len(data)
    sch, fc, sm, fm, el, samples = evaluate_rows(data)
    st.rows_valid_schema = sch
    st.rows_full_consistency = fc
    st.rows_score_match = sm
    st.rows_flag_match = fm
    st.rows_metric_eligible = el
    st.errors_sample = samples
    return st


def write_outputs(model: str, data: list[dict], out_dir: Path, target_n: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path, csv_path = output_paths(model, target_n, out_dir)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["persona_id", "persona_description", "expected_flag", "expected_total_score", "answers_json"])
        for row in data:
            w.writerow(
                [
                    row.get("persona_id", ""),
                    row.get("persona_description", ""),
                    row.get("expected_flag", ""),
                    row.get("expected_total_score", ""),
                    json.dumps(row.get("answers", {}), ensure_ascii=False),
                ]
            )


def migrate_old_generated_files() -> None:
    """Move legacy generated_*.csv/jsonl from package root into data/generated/."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for path in _DIR.glob("generated_*"):
        if path.is_file():
            dest = DATA_DIR / path.name
            if not dest.exists():
                shutil.move(str(path), str(dest))


def _sort_stats_for_report(stats: list[ModelRunStats]) -> list[ModelRunStats]:
    order = {m: i for i, m in enumerate(REPORT_TABLE_MODEL_ORDER)}
    return sorted(stats, key=lambda s: (order.get(s.model, 10_000), s.model))


def build_report(stats: list[ModelRunStats], target_n: int) -> str:
    """Текст как в MODEL_COMPARISON_REPORT.md (субъективные баллы и вывод — константы в модуле)."""
    lines = [
        f"# Сравнение моделей: синтетические анкеты (n{target_n})",
        "",
        f"данные: `data/generated/generated_*_n{target_n}.jsonl`.",
        "",
        "## Таблица",
        "",
        "| Модель | Строк | Schema | Consistency | Flag OK | Субъективно /10 |",
        "|--------|------:|-------:|------------:|--------:|----------------:|",
    ]
    for s in _sort_stats_for_report(stats):
        subj = REPORT_SUBJECTIVE_SCORES.get(s.model, "—")
        flag_star = ""
        if s.n_rows > 0 and s.rows_metric_eligible < s.n_rows:
            flag_star = "*"
        lines.append(
            f"| `{s.model}` | {s.n_rows} | {s.schema_rate:.0%} | {s.consistency_rate:.0%} | "
            f"{s.flag_rate:.0%}{flag_star} | {subj} |"
        )

    footnotes: list[str] = []
    for s in _sort_stats_for_report(stats):
        if s.n_rows > 0 and s.rows_metric_eligible < s.n_rows:
            bad = s.n_rows - s.rows_metric_eligible
            footnotes.append(
                f"*У `{s.model}` для **Flag OK** знаменатель **{s.rows_metric_eligible}** (не **{s.n_rows}**): "
                f"{bad} строк с невалидной schema в расчёт не входят."
            )
    if footnotes:
        lines.append("")
        lines.extend(footnotes)

    if stats and all(s.n_rows == 0 for s in stats):
        lines.extend(
            [
                "",
                "> Ни одна модель не вернула строк. Проверьте `API_KEY`, `BASE_URL` и доступность моделей на OpenRouter.",
            ]
        )

    lines.extend(["", REPORT_RASCHIFROVKA, "", REPORT_VIVOD, ""])
    return "\n".join(lines)


def main() -> None:
    migrate_old_generated_files()

    raw = os.environ.get("SYNTH_MODELS", "")
    models = [m.strip() for m in raw.split(",") if m.strip()] or DEFAULT_MODELS
    target_n = int(os.environ.get("SYNTH_TARGET_N", "50"))
    batch_max = int(os.environ.get("SYNTH_BATCH_MAX", "15"))

    base_prompt = load_prompt()
    all_stats: list[ModelRunStats] = []
    force_regen = os.environ.get("SYNTH_FORCE_REGEN", "").strip().lower() in ("1", "true", "yes")

    for model in models:
        print(f"--- {model} (target_n={target_n}) ---", flush=True)
        jsonl_path, _csv_path = output_paths(model, target_n)
        if not force_regen:
            cached = load_cached_jsonl(jsonl_path)
            if cached is not None:
                data_sorted = sorted(cached, key=persona_sort_key)
                st = stats_from_evaluated_data(model, target_n, data_sorted)
                all_stats.append(st)
                write_outputs(model, data_sorted, DATA_DIR, target_n)
                print(
                    f"  SKIP (есть {jsonl_path.name}): rows={st.n_rows} schema={st.rows_valid_schema} "
                    f"consistent={st.rows_full_consistency} flag_ok={st.rows_flag_match}/{st.rows_metric_eligible}",
                    flush=True,
                )
                continue

        st, data = run_model_batched(model, base_prompt, target_n, batch_max)
        all_stats.append(st)
        if data is not None:
            write_outputs(model, data, DATA_DIR, target_n)
            print(
                f"  rows={st.n_rows} schema={st.rows_valid_schema} consistent={st.rows_full_consistency} "
                f"flag_ok={st.rows_flag_match}/{st.rows_metric_eligible}",
                flush=True,
            )
        else:
            print(f"  FAILED: {st.parse_error}", flush=True)

    report = build_report(all_stats, target_n)
    REPORT_FILE.write_text(report, encoding="utf-8")
    print(f"Report written to {REPORT_FILE}")
    print(f"Data directory: {DATA_DIR}")


if __name__ == "__main__":
    main()
