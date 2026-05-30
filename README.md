# NLP4MentalHealth

### Тема 64: NLP в Mental Health

**Состав команды:**
- Алгазинов Александр Иванович
- Габовский Богдан Александрович
- Рыжов Алексей Евгеньевич
- Кожевников Илья Андреевич

**Куратор:** Костров Вячеслав

---

## Описание проекта

Система психологической поддержки на основе NLP и LLM. Пользователь заполняет анкету, которую анализирует дообученный BERT-классификатор. Затем агент-интервьюер задаёт уточняющие вопросы, агент-диагностик составляет вероятностную оценку состояния, и на её основе запускается поддерживающий чат с агентом-терапевтом.

---

## Ключевые компоненты

```
NLP4MentalHealth/
├── ML/
│   └── disease_classification/          # BERT-классификатор тяжести состояния
│       ├── mental-health-bert-classifier.ipynb
│       └── bert_mental_health_model/    # Веса модели (не в git)
│
├── agentic_pipeline/                    # Агентный пайплайн (LangGraph)
│   ├── interviewer/                     # LLM #1 — агент-интервьюер
│   ├── therapist/                       # LLM #3 — агент-терапевт
│   ├── synthetic_questionnaire_generation/
│   │   └── runs/triage_bert_neutral/best_model/  # Веса BERT-триажа (не в git)
│   ├── run_pipeline.py                  # Запуск полного CLI-пайплайна
│   └── .env                            # API-ключи (не в git, см. .env.example)
│
└── triage_interview_app/               # Streamlit-приложение
    ├── backend/                         # FastAPI + BERT-инференс
    ├── streamlit_app.py                 # UI
    ├── interview_runner.py              # Адаптер LangGraph для Streamlit
    └── pipeline_chat.py                # Чат с агентом-терапевтом
```

### Как устроен пайплайн

```
Анкета пользователя
       │
       ▼
BERT-классификатор → класс: relaxed / concerned / urgent
       │
       ▼
LLM #1 — Агент-интервьюер (LangGraph)
  • Задаёт до 4 уточняющих вопросов на основе ответов анкеты (2 для relaxed, 4 для concerned, 3 для urgent)
  • Останавливается, когда собрано достаточно сигнала
       │
       ▼
LLM #2 — Агент-диагностик
  • Синтезирует анкету + интервью
  • Выдаёт вероятностную оценку по 11 состояниям (GAD, MDD, PTSD, ...)
       │
       ▼
LLM #3 — Агент-терапевт
  • Получает динамический системный промпт (класс + результат диагностики)
  • Ведёт открытый поддерживающий диалог
  • В интерфейсе история интервью остаётся видна вверху; сообщения терапевта появляются ниже разделителя — всё выглядит как одна непрерывная сессия
```

---

## Запуск Streamlit-приложения (полный пайплайн)

### 1. Настройка окружения

```bash
cd triage_interview_app
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Веса BERT-триажа

Положите файлы модели (`config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `training_args.bin`) в:

```
agentic_pipeline/synthetic_questionnaire_generation/runs/triage_bert_neutral/best_model/
```

Или укажите путь вручную:

```bash
export TRIAGE_MODEL_DIR="/полный/путь/к/best_model"
```

### 3. Настройка API-ключей (LLM)

```bash
cp agentic_pipeline/.env.example agentic_pipeline/.env
# Откройте .env и заполните API_KEY, MODEL_NAME, BASE_URL
```

Файл `.env.example` содержит шаблон с OpenRouter. Зарегистрируйтесь на [openrouter.ai](https://openrouter.ai), создайте ключ в разделе «Keys» (бесплатный тариф доступен) и вставьте его в `API_KEY`. Поддерживается любой провайдер, совместимый с OpenAI API.

### 4–5. Запуск: два терминала одновременно

Приложение состоит из двух отдельных процессов, которые должны работать параллельно:

- **FastAPI-бэкенд** — загружает BERT-модель и выполняет инференс (классификация анкеты). Это локальный сервер, без него Streamlit не может получить результат триажа.
- **Streamlit-интерфейс** — веб-интерфейс для пользователя, обращается к бэкенду по HTTP.

**Терминал 1 — FastAPI-бэкенд:**

```bash
cd triage_interview_app
source .venv/bin/activate
uvicorn backend.main:app --host 127.0.0.1 --port 8765
```

Проверка: [http://127.0.0.1:8765/docs](http://127.0.0.1:8765/docs) — если открывается, бэкенд работает.

**Терминал 2 — Streamlit-интерфейс** (не закрывая терминал 1):

```bash
cd triage_interview_app
source .venv/bin/activate
streamlit run streamlit_app.py
```

Приложение откроется по адресу [http://localhost:8501](http://localhost:8501).

---

### Альтернатива: Docker Compose (один терминал)

Требует установленного [Docker Desktop](https://www.docker.com/products/docker-desktop/) (или Docker Engine + Compose plugin).

После шагов 2–3 выше:

```bash
docker compose up --build
```

- Приложение: [http://localhost:8501](http://localhost:8501)
- API docs: [http://localhost:8765/docs](http://localhost:8765/docs)

При первом запуске образ собирается и устанавливаются зависимости (~несколько минут). Последующие запуски без `--build` стартуют быстрее. Данные сессий сохраняются в именованном Docker-томе между перезапусками.

Остановка:

```bash
docker compose down
```

---

## Запуск CLI-пайплайна (без UI)

```bash
cd agentic_pipeline
source .venv/bin/activate        # или используйте окружение из triage_interview_app

# Полный пайплайн (интервьюер → диагностик → терапевт)
python run_pipeline.py --questionnaire path/to/answers.json

# Без BERT (правила вместо модели, для разработки)
python run_pipeline.py --questionnaire path/to/answers.json --skip-bert

# Только интервьюер (без терапевта, встроенная тестовая анкета)
python main.py
python main.py --persona 5        # конкретная персона из questionnaire_sample_1.csv
python main.py --max-questions 3
```

---

## ML-модели

| Модель | Расположение | Назначение |
|--------|-------------|-----------|
| BERT-триаж | `agentic_pipeline/.../best_model/` | Классификация анкеты: relaxed / concerned / urgent |
| BERT-классификатор болезней | `ML/disease_classification/bert_mental_health_model/` | Классификация по 11 психическим состояниям |

Обе модели не хранятся в git (слишком большие). Структура папок сохранена через `.gitkeep`.

---

## Структура репозитория

| Папка | Содержимое |
|-------|-----------|
| `EDA/` | Разведочный анализ данных по датасетам |
| `ML/` | Бейзлайны, BERT-классификатор, ноутбуки |
| `agentic_pipeline/` | LangGraph-пайплайн: интервьюер, диагностик, терапевт |
| `triage_interview_app/` | Streamlit UI + FastAPI бэкенд |
