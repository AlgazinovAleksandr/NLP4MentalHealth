# Запуск приложения (модель → API → Streamlit)
## 1. Окружение
Из корня репозитория `NLP4MentalHealth/`:
```bash
cd triage_interview_app
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Нужен интернет при первой установке (torch, transformers и т.д.).
## 2. Веса BERT (куда положить и как указать)
Бэкенд читает каталог **`best_model`** (как после `train_triage_bert.py`: внутри лежат `config.json`, `model.safetensors`, токенизатор и т.д.).
**Вариант A — без переменных:** положить (или оставить) чекпоинт в одном из путей, которые код ищет сам:
- `agentic_pipeline/synthetic_questionnaire_generation/runs/triage_bert_neutral/best_model`
- или `agentic_pipeline/synthetic_questionnaire_generation/runs/triage_bert_run1/best_model`
Используется **первый существующий** из списка (`triage_bert_neutral`, затем `triage_bert_run1`). См. `backend/config.py` → `default_model_dir()`.
**Вариант B — своя папка:** экспортируй **абсолютный** путь к каталогу с весами (это и есть `best_model`, не родительский `runs/.../run_name` без `best_model`, если только там нет файлов модели):
```bash
export TRIAGE_MODEL_DIR="/полный/путь/к/best_model"
```
Если указал путь «как из другого места в репо», бэкенд может переназначить его под `synthetic_questionnaire_generation/runs/...` — см. `_remap_env_model_path` в `backend/config.py`.
Опционально для UI (схема полей анкеты в Streamlit):
```bash
export QUESTIONNAIRE_JSON="/полный/путь/к/registration_questionnaire_v0.0.1.json"
```
По умолчанию берётся файл из `agentic_pipeline/synthetic_questionnaire_generation/`.
## 3. Backend (FastAPI + BERT)
Терминал 1, из **`triage_interview_app/`** (чтобы импортировался пакет `backend`):
```bash
cd /путь/к/NLP4MentalHealth/triage_interview_app
source .venv/bin/activate
export TRIAGE_MODEL_DIR="/полный/путь/к/best_model"   # опционально, см. выше
uvicorn backend.main:app --host 127.0.0.1 --port 8765
```
Проверка: в браузере открыть `http://127.0.0.1:8765/docs`.
## 4. Streamlit
Терминал 2, тоже из **`triage_interview_app/`**:
```bash
cd /путь/к/NLP4MentalHealth/triage_interview_app
source .venv/bin/activate
# если API не на 8765:
# export TRIAGE_API_URL="http://127.0.0.1:8765"
streamlit run streamlit_app.py
```
Streamlit по умолчанию ходит в **`TRIAGE_API_URL`** (если не задан — `http://127.0.0.1:8765`). Порт должен совпадать с `uvicorn`.
---
# Где подключать реальный чат (CHAT_PIPELINE)

1. **`pipeline_chat.py`** — замените тело функции **`run_pipeline(messages, latest_user_message)`** на вызов вашего пайплайна (LLM, RAG и т.д.). Сигнатуру лучше сохранить.
2. **`streamlit_app.py`** — поиск по **`run_pipeline`** или комментарий **`CHAT_PIPELINE`** в блоке чата: там формируется ответ ассистента после сообщения пользователя.

Другие файлы менять не обязательно, если контракт ответа остаётся строкой.
