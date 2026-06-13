---
marp: true
theme: default
paginate: true
style: |
  @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Raleway:wght@400;500;600;700&display=swap');

  :root {
    --accent: #8b0000;
    --accent-light: #b22222;
    --accent-dim: rgba(139,0,0,0.15);
    --bg: #ffffff;
    --fg: #1a1a1a;
    --card: #f7f7f7;
    --card-dark: #f0f0f0;
    --border: #e0e0e0;
    --muted: #666666;
    --success: #1a7a3a;
  }

  section {
    background: var(--bg);
    color: var(--fg);
    font-family: 'Raleway', sans-serif;
    font-weight: 400;
    padding: 52px 72px;
    line-height: 1.6;
  }

  h1 {
    font-family: 'Outfit', sans-serif;
    font-weight: 800;
    font-size: 2.6em;
    color: var(--accent);
    letter-spacing: -0.02em;
    line-height: 1.15;
    margin-bottom: 0.2em;
  }

  h2 {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 1.7em;
    color: var(--fg);
    margin-bottom: 0.35em;
    border-bottom: 3px solid var(--accent);
    padding-bottom: 0.2em;
  }

  h3 {
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    font-size: 0.75em;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.18em;
    margin-bottom: 0.1em;
  }

  strong { color: var(--accent); font-weight: 700; }

  code {
    background: #f2e8e8;
    color: var(--accent);
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 0.88em;
  }

  pre {
    background: #1a1a1a;
    border-radius: 10px;
    padding: 20px 24px;
    font-size: 0.78em;
    line-height: 1.7;
    border-left: 4px solid var(--accent);
  }

  pre code {
    background: transparent;
    color: #e0e0e0;
    padding: 0;
    font-size: 1em;
  }

  table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0 5px;
    font-size: 0.82em;
  }

  table th {
    background: var(--accent);
    color: #ffffff;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-size: 0.78em;
    padding: 10px 18px;
    border: none;
  }

  table th:first-child { border-radius: 8px 0 0 8px; }
  table th:last-child  { border-radius: 0 8px 8px 0; }

  table td {
    background: var(--card);
    padding: 11px 18px;
    border: none;
    color: var(--fg);
  }

  table td:first-child { border-radius: 8px 0 0 8px; }
  table td:last-child  { border-radius: 0 8px 8px 0; }

  section.lead {
    background: var(--accent);
    color: #ffffff;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
  }

  section.lead h1 {
    color: #ffffff;
    font-size: 3em;
    line-height: 1.1;
  }

  section.lead h2 {
    color: rgba(255,255,255,0.85);
    font-size: 1.25em;
    border-bottom: none;
    font-weight: 500;
    margin-bottom: 0.2em;
  }

  section.lead p  { color: rgba(255,255,255,0.75); font-size: 0.95em; }
  section.lead h3 { color: rgba(255,255,255,0.6); }
  section.lead strong { color: #ffffff; }

  .divider {
    width: 60px;
    height: 3px;
    background: rgba(255,255,255,0.5);
    border-radius: 2px;
    margin: 14px auto 18px;
  }

  .divider-dark {
    width: 60px;
    height: 3px;
    background: var(--accent);
    border-radius: 2px;
    margin: 8px 0 18px;
  }

  .card-row {
    display: flex;
    gap: 20px;
    margin-top: 18px;
  }

  .card {
    flex: 1;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 24px 20px;
  }

  .card h4 {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 1em;
    color: var(--accent);
    margin: 0 0 8px;
  }

  .card p {
    color: var(--muted);
    font-size: 0.8em;
    line-height: 1.55;
    margin: 0;
  }

  .card ul {
    color: var(--muted);
    font-size: 0.8em;
    line-height: 1.7;
    margin: 0;
    padding-left: 18px;
  }

  .pill-row {
    display: flex;
    gap: 10px;
    justify-content: center;
    flex-wrap: wrap;
    margin-top: 20px;
  }

  .pill {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.35);
    border-radius: 20px;
    padding: 5px 18px;
    font-size: 0.68em;
    color: rgba(255,255,255,0.85);
    letter-spacing: 0.07em;
    text-transform: uppercase;
  }

  .tag {
    display: inline-block;
    background: var(--accent-dim);
    border: 1px solid rgba(139,0,0,0.25);
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.72em;
    color: var(--accent);
    font-weight: 600;
    letter-spacing: 0.05em;
    margin: 2px;
  }

  .pipeline-box {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px 24px;
    font-family: 'Outfit', monospace;
    font-size: 0.82em;
    line-height: 2;
    color: var(--fg);
    margin-top: 16px;
  }

  .pipeline-box .arrow { color: var(--accent); font-weight: 700; }
  .pipeline-box .label { color: var(--muted); font-size: 0.85em; }

  section::after {
    font-family: 'Raleway', sans-serif;
    font-size: 0.65em;
    color: #aaaaaa;
  }

  footer {
    font-family: 'Raleway', sans-serif;
    font-size: 0.6em;
    color: #bbbbbb;
  }
footer: 'NLP4MentalHealth · 2026'
---

<!-- _class: lead -->

# LLM Applications<br>for Mental Health

<div class="divider"></div>

## NLP-система психологической поддержки

<br>

<div style="color: rgba(255,255,255,0.8); font-size: 0.88em; line-height: 2;">
Алгазинов А.И. &nbsp;·&nbsp; Габовский Б.А. &nbsp;·&nbsp; Рыжов А.Е. &nbsp;·&nbsp; Кожевников И.А.<br>
<span style="font-size: 0.85em; color: rgba(255,255,255,0.55);">Куратор: Костров Вячеслав</span>
</div>

<div class="pill-row">
  <span class="pill">NLP</span>
  <span class="pill">LLM</span>
  <span class="pill">Mental Health</span>
  <span class="pill">2026</span>
</div>

---

### Мотивация

## Проблема и роль LLM

<div class="card-row">
  <div class="card" style="border-left: 4px solid #c0392b;">
    <h4>Проблема</h4>
    <ul>
      <li>Психологическая помощь труднодоступна: очереди, стоимость, стигматизация</li>
      <li>Большинство людей с ментальными расстройствами не получают помощи вовремя</li>
      <li>Первичный скрининг занимает время специалиста</li>
      <li>Нет масштабируемого инструмента первичного контакта</li>
    </ul>
  </div>
  <div class="card" style="border-left: 4px solid var(--accent);">
    <h4>Что меняют LLM</h4>
    <ul>
      <li>Ведут естественный диалог — без жёстких скриптов</li>
      <li>NLP автоматизирует первичный скрининг и триаж</li>
      <li>BERT-классификатор оценивает состояние по анкете мгновенно</li>
      <li>Агентный пайплайн масштабируется без участия врача</li>
    </ul>
  </div>
</div>

> **Цель:** не заменить психолога, а обеспечить первый, доступный контакт.

---

### Постановка задачи

## Техническая цель

<div class="card-row">
  <div class="card" style="flex: 1.2; border-left: 4px solid var(--accent);">
    <h4>Задача</h4>
    <ul>
      <li>Разработать NLP-систему первичной психологической поддержки пользователей</li>
      <li>Вход: структурированная анкета &rarr; триаж &rarr; диагностическое интервью &rarr; поддерживающий чат</li>
      <li>Обучение и тестирование на клинических данных или верифицированной синтетике</li>
    </ul>
  </div>
  <div style="flex: 1; display: flex; flex-direction: column; gap: 12px;">
    <div class="card" style="border-left: 4px solid var(--accent);">
      <h4>Критерии качества</h4>
      <ul style="font-size: 0.78em; line-height: 1.6;">
        <li>Accuracy / F1 классификатора уровня риска</li>
        <li>Качество и безопасность агентских ответов</li>
      </ul>
    </div>
    <div class="card" style="border-left: 4px solid #aaaaaa;">
      <h4 style="color: var(--muted);">Планы по оценке</h4>
      <ul style="font-size: 0.78em; color: var(--muted); line-height: 1.6;">
        <li>Human evaluation (оценка психологами)</li>
        <li>LLM-as-a-judge + A/B тестирование агентов</li>
      </ul>
    </div>
  </div>
</div>

---

### BERT-триаж: данные

## Синтетическая генерация обучающих данных

<div class="card-row">
  <div class="card" style="flex: 1.1;">
    <h4>Задача и проблема данных</h4>
    <p>Классифицировать анкету регистрации &rarr; <strong>relaxed / concerned / urgent</strong>.<br><br>
    Размеченных реальных данных нет — слишком чувствительная область.<br><br>
    Решение: <strong>генерация синтетических анкет через LLM</strong>.</p>
  </div>
  <div class="card" style="flex: 1.1;">
    <h4>Что сделано</h4>
    <ul>
      <li><strong>1 000</strong> синтетических анкет (<code>synthetic_triagem_n1000.csv</code>)</li>
      <li>Правило-based верификация меток:<br>
        <code>concern_level × 3 + PHQ-2 + GAD-2 + daily_impact + prior_help</code></li>
      <li><code>self_harm</code> в списке проблем &rarr; всегда <strong>urgent</strong></li>
      <li>Score ≤ 2 &rarr; <strong>relaxed</strong>; иначе &rarr; <strong>concerned</strong></li>
      <li>Сравнение <strong>7 LLM-моделей</strong> для генерации</li>
    </ul>
  </div>
</div>

<div style="margin-top: 16px; font-size: 0.8em; color: var(--muted);">
  <span class="tag">LLM-in-the-loop</span>
  <span class="tag">Rule-based verification</span>
  <span class="tag">PHQ-2 / GAD-2</span>
  <span class="tag">Model comparison</span>
</div>

---

### BERT-триаж: данные

## Структура данных: поля анкеты

<style scoped>
table { font-size: 0.62em; border-spacing: 0 3px; margin: 0 auto; width: max-content; }
table th { padding: 6px 12px; font-size: 0.72em; }
table td { padding: 6px 12px; }
</style>

| Секция | Поля | Тип | Роль в скоринге |
|--------|------|-----|-----------------|
| §1 Demographics | `q_gender` · `q_age`* · `q_occupation` | cat / num | (контекст) |
| §2 Concern | `q_has_mental_concern` | ordinal 0–3 | concern_score &times; **3** |
| | `q_concern_areas` (multi) | bool flags | **self_harm** ⚠ &rarr; urgent |
| §3 PHQ-2 / GAD-2 | `q_phq2_1` · `q_phq2_2` · `q_gad2_1` · `q_gad2_2` | scale 0–3 | sum ≥ 3 &rarr; +**3** (каждый) |
| §4 Impact | `q_daily_impact` · `q_duration`** | ordinal 0–4 | impact_score &times; **1** |
| §5 Help History | `q_prior_help` · `q_diagnosis_known`** | ordinal 0–1 | concern_score &times; **1** |
| §6 App Goal | `q_app_goal` (multi) | bool flags | (персонализация UX) |

<div style="font-size: 0.48em; color: var(--muted); margin-top: 2px;">
  <center>* возраст&nbsp;·&nbsp; ** skip-logic поля &nbsp;·&nbsp; 
  <strong>14</strong> вопросов (макс.) / <strong>9</strong> (мин.)</center>
</div>

---

### BERT-триаж: данные

## Структура данных: скоринг и метки

<div class="card-row" style="gap: 16px; margin-top: 6px; align-items: flex-start;">
  <div style="flex: 1;">
    <div class="card" style="border-left: 4px solid var(--accent); padding: 12px 14px;">
      <h4 style="margin: 0 0 6px; font-size: 0.78em;">Формула скоринга</h4>
      <div style="font-size: 0.62em; font-family: monospace; line-height: 1.85; color: var(--fg);">
        <span style="color: var(--muted);">score =</span> concern_score &times; <strong>3</strong><br>
        <span style="color: var(--muted);">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+</span> (PHQ-2 sum ≥ 3) &times; <strong>3</strong><br>
        <span style="color: var(--muted);">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+</span> (GAD-2 sum ≥ 3) &times; <strong>3</strong><br>
        <span style="color: var(--muted);">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+</span> impact_score &times; <strong>1</strong><br>
        <span style="color: var(--muted);">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+</span> prior_help_score &times; <strong>1</strong>
      </div>
    </div>
    <div style="margin-top: 8px; font-size: 0.55em; color: var(--muted); line-height: 1.5; padding: 0 4px;">
      PHQ-2 / GAD-2 — клинические скрининги, порог: sum ≥ 3
    </div>
  </div>
  <div style="flex: 1; display: flex; flex-direction: column; gap: 8px;">
    <div style="background: #f0f9f4; border: 1.5px solid #1a7a3a; border-radius: 8px; padding: 10px 14px;">
      <strong><div style="font-size: 0.78em; font-weight: 800; color: #1a7a3a;">relaxed</div></strong>
      <div style="font-size: 0.58em; color: #2d6a4f; line-height: 1.4;">score ≤ 2 · Информ. чат-бот без классификатора</div>
    </div>
    <strong><div style="background: #fdf8ec; border: 1.5px solid #b8860b; border-radius: 8px; padding: 10px 14px;">
      <div style="font-size: 0.78em; font-weight: 800; color: #b8860b;">concerned</div></strong>
      <div style="font-size: 0.58em; color: #7d6608; line-height: 1.4;">score ≥ 3 · Бинарный классификатор + интервью</div>
    </div>
    <strong><div style="background: #fdf0f0; border: 1.5px solid var(--accent); border-radius: 8px; padding: 10px 14px;">
      <div style="font-size: 0.78em; font-weight: 800; color: var(--accent);">urgent ⚠</div></strong>
      <div style="font-size: 0.58em; color: #7a0000; line-height: 1.4;">self_harm в ответах · Кризисный протокол</div>
    </div>
  </div>
</div>

---

### BERT-классификатор

## Датасет и обучение

<div class="card-row">
  <div class="card" style="flex: 1.2;">
    <h4>Датасет</h4>
    <p><strong>Combined Data (Kaggle)</strong><br>
    53 043 примера &nbsp;·&nbsp; 7 классов психических состояний<br><br>
    Модель: <code>bert-base-uncased</code> + <code>BertForSequenceClassification</code><br><br>
    Гиперпараметрический поиск с <strong>MLflow</strong>:<br>
    lr ∈ {2e-5, 3e-5} &nbsp;·&nbsp; dropout ∈ {0.1, 0.3}<br>
    4 запуска × 3 эпохи</p>
  </div>
  <div class="card" style="flex: 1.8; background: #1a1a1a; color: #e0e0e0;">
    <h4 style="color: #b22222;">Фрагмент MLflow-трекинга</h4>
    <pre style="margin: 0; padding: 0; background: transparent; border: none; font-size: 0.78em; line-height: 1.75; color: #e0e0e0;"><code>mlflow.log_param("lr", lr)
mlflow.log_param("dropout", dropout)

# per epoch
mlflow.log_metric("val_f1", val_f1, step=epoch)
mlflow.log_metric("train_loss", loss, step=epoch)

mlflow.log_artifact("confusion_matrix.png")
mlflow.log_artifact("classification_report.txt")</code></pre>
  </div>
</div>

<div style="margin-top: 14px; font-size: 0.8em; color: var(--muted);">
  <span class="tag">bert-base-uncased</span>
  <span class="tag">MLflow</span>
  <span class="tag">Hyperparameter search</span>
  <span class="tag">53 043 примера</span>
  <span class="tag">7 классов</span>
</div>

---

### BERT: результаты

## Метрики и артефакты

<style scoped>
  img[src="loss_validation.jpg"] { max-height: 150px; object-fit: contain; display: block; margin: 0 auto; }
  img[src="conf_matrix.jpg"] { max-height: 340px; object-fit: contain; display: block; margin: 0 auto; }
</style>

<div class="card-row" style="align-items: flex-start;">
  <div style="flex: 1;">
    <table>
      <tr><th>Метрика</th><th>Значение</th></tr>
      <tr><td>Val Accuracy</td><td><strong>0.82</strong></td></tr>
      <tr><td>Val F1 (weighted)</td><td><strong>0.83</strong></td></tr>
    </table>
    <div style="margin-top: 16px; background: var(--card); border-radius: 10px; padding: 14px 18px; font-size: 0.78em; color: var(--muted);">
      <strong style="color: var(--fg);">Артефакты MLflow</strong><br>
      <code>classification_report.txt</code><br>
      <code>confusion_matrix.png</code><br>
      <code>best_model/</code> &nbsp;(веса BERT)
    </div>
  </div>
  <div style="flex: 1.1; display: flex; flex-direction: column; gap: 8px;">
    <div style="background: var(--card); border-radius: 10px; padding: 8px 12px; border-left: 4px solid var(--accent);">
      <img src="loss_validation.jpg">
    </div>
    <div style="background: var(--card); border-radius: 10px; padding: 8px 12px; border-left: 4px solid #b8860b;">
      <img src="conf_matrix.jpg">
    </div>
  </div>
</div>

---

<!-- _class: lead -->

# Агентный пайплайн

<div class="divider"></div>

## LangGraph · LangChain · OpenRouter

<div class="pill-row">
  <span class="pill">LangGraph</span>
  <span class="pill">Human-in-the-loop</span>
  <span class="pill">Structured Output</span>
  <span class="pill">3 LLM-агента</span>
</div>

---

### Агентный пайплайн

## Архитектура: поток данных

<div style="margin-bottom: 14px; font-size: 0.9em; color: var(--muted);">
  Пользователь заполняет анкету &rarr; система оценивает состояние &rarr; проводит интервью &rarr; ставит диагноз &rarr; запускает поддерживающий чат.
</div>

<div class="pipeline-box">
  <span style="color: var(--accent); font-weight: 700;">Анкета (JSON)</span>
  <span class="arrow"> ──▶ </span>
  <span style="font-weight: 600;">BERT-триаж</span>
  <span class="arrow"> ──▶ </span>
  <span style="font-weight: 600;">Интервьюер</span>
  <span class="label">(LLM #1)</span>
  <span class="arrow"> ──▶ </span>
  <span style="font-weight: 600;">Диагностик</span>
  <span class="label">(LLM #2)</span>
  <span class="arrow"> ──▶ </span>
  <span style="font-weight: 600;">Терапевт</span>
  <span class="label">(LLM #3)</span>
</div>

<div class="card-row" style="margin-top: 20px;">
  <div class="card" style="text-align: center; padding: 16px 12px;">
    <div style="font-size: 1.8em; font-weight: 800; color: #1a7a3a;">relaxed</div>
    <div style="font-size: 0.75em; color: var(--muted); margin-top: 4px;">Низкий уровень стресса · 2 вопроса</div>
  </div>
  <div class="card" style="text-align: center; padding: 16px 12px;">
    <div style="font-size: 1.8em; font-weight: 800; color: #b8860b;">concerned</div>
    <div style="font-size: 0.75em; color: var(--muted); margin-top: 4px;">Умеренное беспокойство · 4 вопроса</div>
  </div>
  <div class="card" style="text-align: center; padding: 16px 12px;">
    <div style="font-size: 1.8em; font-weight: 800; color: var(--accent);">urgent</div>
    <div style="font-size: 0.75em; color: var(--muted); margin-top: 4px;">Требует приоритета · 3 вопроса</div>
  </div>
</div>

---

### Стек

## Технологический стэк

<div class="card-row" style="flex-wrap: wrap; gap: 14px; margin-top: 10px;">
  <div class="card" style="flex: 0 0 calc(25% - 11px); padding: 10px 14px;">
    <h4 style="margin: 0 0 4px;">FastAPI</h4>
    <p>REST API: <code>/validate</code> · <code>/predict</code> · <code>/health</code></p>
  </div>
  <div class="card" style="flex: 0 0 calc(25% - 11px); padding: 10px 14px;">
    <h4 style="margin: 0 0 4px;">Streamlit</h4>
    <p>UI-автомат: анкета → интервью → чат</p>
  </div>
  <div class="card" style="flex: 0 0 calc(25% - 11px); padding: 10px 14px;">
    <h4 style="margin: 0 0 4px;">Docker Compose</h4>
    <p>Один запуск, два сервиса, персистентный том</p>
  </div>
  <div class="card" style="flex: 0 0 calc(25% - 11px); padding: 10px 14px;">
    <h4 style="margin: 0 0 4px;">LangGraph</h4>
    <p>Граф с прерываниями, human-in-the-loop</p>
  </div>
  <div class="card" style="flex: 0 0 calc(25% - 11px); padding: 10px 14px;">
    <h4 style="margin: 0 0 4px;">OpenRouter</h4>
    <p>Любой OpenAI-compatible LLM-провайдер</p>
  </div>
  <div class="card" style="flex: 0 0 calc(25% - 11px); padding: 10px 14px;">
    <h4 style="margin: 0 0 4px;">Fine-tuned BERT</h4>
    <p>Классификатор триажа на 1 000 синтетических анкет</p>
  </div>
  <div class="card" style="flex: 0 0 calc(25% - 11px); padding: 10px 14px;">
    <h4 style="margin: 0 0 4px;">Pydantic</h4>
    <p>Структурированный вывод диагностика</p>
  </div>
  <div class="card" style="flex: 0 0 calc(25% - 11px); padding: 10px 14px;">
    <h4 style="margin: 0 0 4px;">SQLite · SQLAlchemy</h4>
    <p>Логирование сессий, авто-создание БД</p>
  </div>
</div>

---

<!-- _style: "padding: 0; background: #1a1a1a;" -->
<!-- _footer: "" -->
<!-- _paginate: skip -->

<video src="demo_final.mp4" autoplay muted loop controls style="width: 100%; height: 680px; object-fit: contain; display: block;"></video>

---

### Результаты

## Что получилось

<div class="card-row">
  <div class="card" style="border-top: 4px solid var(--accent);">
    <h4>Метрики BERT</h4>
    <div style="font-size: 0.78em; color: var(--muted); line-height: 1.9;">
      Accuracy &nbsp;&nbsp;&nbsp;&rarr;&nbsp; <strong style="color: var(--fg);">0.82</strong><br>
      F1 weighted &rarr; <strong style="color: var(--fg);">0.83</strong><br>
      <br>
      <span style="font-size: 0.9em;">7 классов психических состояний</span>
    </div>
  </div>
  <div class="card" style="border-top: 4px solid #1a7a3a;">
    <h4 style="color: #1a7a3a;">Технические достижения</h4>
    <ul style="font-size: 0.78em;">
      <li>End-to-end пайплайн от анкеты до чата</li>
      <li>Docker Compose: один запуск</li>
      <li>Human-in-the-loop через <code>interrupt()</code></li>
      <li>Dynamic prompt: тон терапевта зависит от класса триажа</li>
      <li>SQLite-логирование сессий</li>
    </ul>
  </div>
  <div class="card" style="border-top: 4px solid #b8860b;">
    <h4 style="color: #b8860b;">Качество агентов</h4>
    <ul style="font-size: 0.78em;">
      <li>Структурированный вывод (Pydantic)</li>
      <li><strong>11</strong> диагностических состояний</li>
      <li>Адаптивный тон терапевта</li>
      <li>Диагноз не раскрывается пользователю</li>
      <li>OpenRouter: совместим с любым OpenAI API</li>
    </ul>
  </div>
</div>

---

### Выводы

## Что сделано и что дальше

<div class="card-row">
  <div class="card" style="border-left: 4px solid var(--accent); flex: 1.1;">
    <h4>Что сделано</h4>
    <ul>
      <li>Синтетическая генерация данных + верификация правилами</li>
      <li>BERT-триаж: fine-tuning + MLflow + сравнение 7 моделей</li>
      <li>LangGraph-пайплайн с тремя LLM-агентами</li>
      <li>Human-in-the-loop интервью с прерываниями</li>
      <li>Рабочее веб-приложение: Streamlit + FastAPI + Docker</li>
    </ul>
  </div>
  <div class="card" style="border-left: 4px solid #aaaaaa; flex: 1.1;">
    <h4 style="color: var(--muted);">Что дальше</h4>
    <ul style="color: var(--muted);">
      <li>Дообучение на реальных клинических данных</li>
      <li>Пользовательское тестирование (UX-исследование)</li>
      <li>RAG для агента-терапевта (клинические протоколы)</li>
      <li>Расширение на мобильную платформу</li>
    </ul>
  </div>
</div>

<div style="margin-top: 22px; text-align: center; font-size: 0.82em; color: var(--muted);">
  Репозиторий: <strong style="color: var(--fg);">NLP4MentalHealth</strong> &nbsp;·&nbsp; Стек: Python · BERT · LangGraph · FastAPI · Streamlit · Docker
</div>
