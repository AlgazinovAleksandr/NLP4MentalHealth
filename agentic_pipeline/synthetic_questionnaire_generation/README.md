# Synthetic questionnaire + BERT triage

Пайплайн: **LLM** по промпту генерирует «анкета + первое сообщение»; в режиме **`joint`** целевой **`label`** задаётся вместе с текстом, **`label_from_rules`** считается только по анкете. Затем **`scripts/bert_triage_text.py`** собирает вход (JSON + `[SEP]` + сообщение), **`scripts/train_triage_bert.py`** учит три класса: `relaxed` / `concerned` / `urgent`.

Рабочая директория:

```bash
cd NLP4MentalHealth/agentic_pipeline/synthetic_questionnaire_generation
```

---

## Структура

| Путь | Назначение |
|------|------------|
| **`scripts/`** | Точки входа: генерация датасета, экспорт текста для BERT, обучение, инференс. |
| **`src/nlp4mh_triagem/`** | Библиотека: API, скоринг анкеты, валидация, комбинированный триаж, сборка BERT-текста. |
| **`data/dataset/`** | Канонические данные: `synthetic_triagem_n1000.jsonl`, `.csv`, `_bert.csv`, `_manifest.json`. |
| **`runs/`** | Чекпоинты (в `.gitignore`; **веса не удаляются** скриптами из этого README). |
| **`questionnaire_sample_1.csv`** | 15 эталонных персон; используется из **`../main.py`** — путь не менять. |
| **`docs/reference/`** | Справочные заметки, не участвуют в пайплайне. |
| **`requirements_train.txt`** | Зависимости для обучения BERT. |

Промпты и схема анкеты лежат в корне пакета: **`english_psych_user_message_dataset_prompt.txt`**, **`test_questionnaire_prompt.txt`**, **`registration_questionnaire_v0.0.1.json`**.

Текущий объём основного датасета см. в **`data/dataset/synthetic_triagem_n1000_manifest.json`** (поле **`canonical_dataset_rows`**) и фактически в `wc -l` по jsonl.

---

## Команды

### 1. Сгенерировать датасет (LLM)

```bash
DATASET_FRESH_START=1 DATASET_TARGET_N=1000 python3 scripts/generate_dataset.py
```

### 2. Экспорт для BERT

```bash
python3 scripts/bert_triage_text.py data/dataset/synthetic_triagem_n1000.jsonl
```

### 3. Обучение

```bash
python3 scripts/train_triage_bert.py \
  --csv data/dataset/synthetic_triagem_n1000_bert.csv \
  --output_dir runs/triage_bert_run1
```

Для упора на recall по **`urgent`**: `--metric_for_best_model combined_urgent`, `--class_weight_mode balanced`, `--urgent_weight_mult` — см. `scripts/train_triage_bert.py --help`.

### 4. Инференс

```bash
python3 scripts/predict_triage_bert.py --model runs/triage_bert_run1/best_model
```

### 5. Smoke-тест комбинированного триажа (без сети)

Из корня пакета, с `PYTHONPATH=src`:

```bash
PYTHONPATH=src python3 -m nlp4mh_triagem.triage_combined
```

---

## Зависимости

- Генерация: клиент `openai` + ключи в `.env` (см. `src/nlp4mh_triagem/generation_common.py`).
- Обучение: **`requirements_train.txt`** в venv проекта.
