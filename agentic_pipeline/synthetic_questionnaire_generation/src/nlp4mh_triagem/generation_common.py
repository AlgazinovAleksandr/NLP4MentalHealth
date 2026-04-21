from __future__ import annotations

import json
import os
import re
from pathlib import Path

from openai import OpenAI

_ENV_LAYERED_DONE = False

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def load_dotenv_layered() -> None:
    global _ENV_LAYERED_DONE
    if os.environ.get("DOTENV_RELOAD", "").strip().lower() in ("1", "true", "yes"):
        _ENV_LAYERED_DONE = False
    if _ENV_LAYERED_DONE:
        return
    try:
        from dotenv import dotenv_values
    except ImportError:
        _ENV_LAYERED_DONE = True
        return

    for path in (_PACKAGE_ROOT / ".env", _PACKAGE_ROOT.parent / ".env"):
        if not path.is_file():
            continue
        data = dotenv_values(path, encoding="utf-8-sig")
        for k, v in data.items():
            if k is None or v is None:
                continue
            key = str(k).strip().lstrip("\ufeff")
            val = str(v).strip()
            if val == "":
                continue
            os.environ[key] = val
    _ENV_LAYERED_DONE = True


def _normalize_base_url(url: str) -> str:
    u = url.strip().strip("'\"").rstrip("/")
    if not u:
        return "https://openrouter.ai/api/v1"
    if "aitunnel.ru" in u.lower() and not u.endswith("/v1"):
        u = f"{u}/v1"
    return u


def _resolve_base_url(api_key: str) -> str:
    load_dotenv_layered()
    raw = (os.environ.get("BASE_URL") or "").strip().strip("'\"")
    if api_key.startswith("sk-aitunnel-"):
        if "aitunnel.ru" not in raw.lower():
            raw = "https://api.aitunnel.ru/v1"
    if not raw:
        raw = "https://openrouter.ai/api/v1"
    return _normalize_base_url(raw)


def _resolve_api_key() -> str:
    load_dotenv_layered()
    raw = (
        os.environ.get("API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("AITUNNEL_API_KEY")
    )
    key = (raw or "").strip()
    if key.lower().startswith("bearer "):
        key = key[7:].strip()
    return key


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

BATCH_WRAPPER_DATASET = """
---
AUTOMATION OVERRIDE (this request only):
The specification above is unchanged. Ignore the literal counts "15" and persona_id range "test_001"…"test_015"
in the TASK and OUTPUT FORMAT sections for THIS response only.

Return a JSON array of exactly {batch_n} objects.
Use persona_id strings "test_{start:03d}" through "test_{end:03d}" (inclusive, three-digit zero-padded).
Keep the same option codes, skip logic, scoring rules, and output fields as in the specification.

For this batch of {batch_n} rows, target this **joint_label** distribution (counts sum to {batch_n}):
  — relaxed: {n_relaxed}
  — concerned: {n_concerned}
  — urgent: {n_urgent}

``joint_label`` is defined in the spec (form + ``user_message``), not answers-only shortcuts.
Among relaxed + concerned ``joint_label`` rows, include several **near_threshold** rows where the
spec applies; they count toward relaxed/concerned above.

Follow free-text rules in the spec (`user_message` and/or `persona_description`): high-noise,
empty, non-Latin, emoji-heavy variants for a substantial fraction of rows when the spec asks for it.
If the specification asks for a markdown validation table after the JSON array, you may append it after the closing "]" of the array.
"""


def distribution_targets(batch_n: int) -> tuple[int, int, int, int]:
    nums = [4, 7, 2, 2]
    d = 15
    exact = [nums[i] * batch_n / d for i in range(4)]
    out = [int(x) for x in exact]
    rem = batch_n - sum(out)
    order = sorted(range(4), key=lambda i: exact[i] - out[i], reverse=True)
    for k in range(rem):
        out[order[k]] += 1
    return (out[0], out[1], out[2], out[3])


def distribution_targets_dataset(batch_n: int) -> tuple[int, int, int]:
    nums = [6, 3, 1]
    d = 10
    exact = [nums[i] * batch_n / d for i in range(3)]
    out = [int(x) for x in exact]
    rem = batch_n - sum(out)
    order = sorted(range(3), key=lambda i: exact[i] - out[i], reverse=True)
    for k in range(rem):
        out[order[k]] += 1
    assert sum(out) == batch_n
    return (out[0], out[1], out[2])


def build_batch_prompt(
    base: str,
    start_id: int,
    end_id: int,
    batch_n: int,
    *,
    distribution_profile: str = "legacy",
) -> str:
    if distribution_profile == "dataset":
        r, c, u = distribution_targets_dataset(batch_n)
        suffix = BATCH_WRAPPER_DATASET.format(
            batch_n=batch_n,
            start=start_id,
            end=end_id,
            n_relaxed=r,
            n_concerned=c,
            n_urgent=u,
        )
    else:
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


def _consume_json_object(s: str, start: int) -> tuple[dict | None, int]:
    if start >= len(s) or s[start] != "{":
        return None, start
    depth = 0
    in_str = False
    esc = False
    for j in range(start, len(s)):
        c = s[j]
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
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                chunk = s[start : j + 1]
                try:
                    return json.loads(chunk), j + 1
                except json.JSONDecodeError:
                    return None, start
    return None, start


def _salvage_json_array_of_objects(s: str, arr_start: int) -> list:
    out: list = []
    i = arr_start + 1
    n = len(s)
    while i < n:
        while i < n and s[i] in " \t\n\r,":
            i += 1
        if i >= n:
            break
        if s[i] == "]":
            break
        obj, nxt = _consume_json_object(s, i)
        if obj is None:
            break
        out.append(obj)
        i = nxt
    return out


def extract_json_array(text: str) -> list:
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
    salvaged = _salvage_json_array_of_objects(t, start)
    if salvaged:
        return salvaged
    raise ValueError(
        "Unclosed JSON array and no complete objects could be salvaged "
        "(try DATASET_MAX_TOKENS / SYNTH_MAX_TOKENS, smaller DATASET_BATCH_MAX, "
        "or shorten free-text instructions in the prompt)."
    )


def call_openrouter(
    model: str,
    prompt: str,
    temperature: float = 0.25,
    *,
    max_tokens: int | None = None,
) -> str:
    api_key = _resolve_api_key()
    base_url = _resolve_base_url(api_key)
    if not api_key:
        raise RuntimeError(
            "No API key after loading .env under synthetic_questionnaire_generation/ and agentic_pipeline/.env. "
            "Set API_KEY, OPENAI_API_KEY, or AITUNNEL_API_KEY. For AITunnel use BASE_URL=https://api.aitunnel.ru/v1 "
            "and a key from https://aitunnel.ru/panel/keys ."
        )
    if max_tokens is None:
        max_tokens = int(os.environ.get("SYNTH_MAX_TOKENS", "28000"))

    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["API_KEY"] = api_key

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=600.0,
        max_retries=2,
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    choice = resp.choices[0].message
    content = choice.content
    return content if content is not None else ""
