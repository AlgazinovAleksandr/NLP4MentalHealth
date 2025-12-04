## Бейзлайны: классификация (Reddit) и генеративная оценка

### Классификационный бейзлайн (Reddit)
- **Данные**: `reddit_mental_health_posts_preprocessed.csv` (выбрана подвыборка после dropna).
- **Текст**: `Title + Body` → поле `text`.
- **Эмбеддинги**: tf-idf+SVD, Sentence-BERT (`all-MiniLM-L6-v2`), Word2Vec (усреднение).
- **Модели**: Logistic Regression, KNN, CatBoost, MLP.
- **Метрика**: balanced accuracy (stratified split).

Результаты (balanced accuracy):

| Embedding     | CatBoost  | KNN      | Logistic Regression | MLP      |
|---------------|-----------|----------|---------------------|----------|
| Sentence-BERT | 0.749341  | 0.728169 | 0.747167            | 0.762219 |
| Word2Vec      | 0.690497  | 0.597548 | 0.684865            | 0.713112 |
| tf-idf+SVD    | 0.743215  | 0.583596 | 0.745709            | 0.756078 |

- **Лучшее сочетание**: MLP + Sentence-BERT (0.7622).
- **Файл с результатами**: `NLP4MentalHealth/ML/reddit_baseline_results_balanced_accuracy.csv`.
- **Ноутбук**: `NLP4MentalHealth/ML/reddit_baseline.ipynb`.

### Генеративная оценка ответов (LLM-as-Judge)
- **Данные**: смесь `Amod/mental_health_counseling_conversations` и `ShenLab/MentalChat16K (Interview_Data_6K.csv)`; `EVAL_N` наблюдений (по умолчанию 100).
- **Генераторы** (локально): по умолчанию
  - `meta-llama/Llama-3.2-1B-Instruct`
  - `microsoft/Phi-3-mini-4k-instruct`
  - `Qwen/Qwen2.5-1.5B-Instruct`
- **Судьи** (локально): тот же набор моделей по умолчанию.
- **Рубрика**:
  - eshro: empathy, safety, helpful, ontopic, overall (1–5)
  - cape: disclaimer [done/partial/missing], no_diagnosis, plan [done/partial/missing], risk_escalation, no_pii
  - trust: pass (true/false)
- **Агрегации**: средние по eshro/cape/trust на уровне пары генератор×судья и на уровне генератора.
- **Выводы**: CSV с построчными оценками и агрегациями.

Итоговые метрики (агрегировано по генератору):

| Генератор                          | eshro_overall | cape_score | trust_pass |
|------------------------------------|---------------|------------|------------|
| Qwen/Qwen2.5-1.5B-Instruct         | 0.440         | 0.259      | 0.243      |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 1.337         | 0.420      | 0.643      |
| microsoft/Phi-3-mini-4k-instruct   | 0.720         | 0.289      | 0.473      |

- Лучшие значения: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` по `eshro_overall` и `trust_pass`.
