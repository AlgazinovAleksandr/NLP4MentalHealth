## Сводный EDA

### Коротко
- MentalChat16K: смешанный корпус (синтетика + интервью), удобен для стартового SFT по стилю и охвату тем; одноходовый, англ., без меток рисков.
- Amod MHCC: реальные пары вопрос–ответ от профессионалов; хорош для стиля/эмпатии; одноходовый, англ., RAIL‑D.
- Reddit Mental Health Posts: крупный многообъектный корпус с 5 сабреддит‑классами; пригоден для классификации; короткие посты.
- C‑SSRS Reddit 500: небольшой размеченный набор по суицидальности; годится для прототипов/валидации, риск переобучения для крупных моделей.

### Таблица сравнения (метрики из ноутбуков)

| Датасет | Источник | Происхождение | Мультиходовость | Всего (сыр.) | После баз. фильтров | После порога длины | Средн. input/context | Медиана input/context | Средн. output/response | Медиана output/response | Язык | Лицензия |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| MentalChat16K | ShenLab/MentalChat16K | synthetic + interview (9 774/6 310) | нет | 16 084 | 16 005 | 15 963 | 96.05 | 67 | 316.65 | 333 | EN | проверять на HF |
| Amod MHCC | Amod/mental_health_counseling_conversations | real (проф. ответы) | нет | 3 512 | 2 467 | 2 459 | 58.11 | 48 | 187.67 | 149 | EN | RAIL‑D |
| Reddit Mental Health Posts | solomonk/reddit_mental_health_posts | real (Reddit) | нет | 151 288 | см. ноут | см. ноут | 51 | 21 | — | — | EN | проверять на HF |
| C‑SSRS Reddit 500 | Kaggle (C‑SSRS Reddit users posts) | real (Reddit, размеч.) | нет | 500 | — | — | ≈246.27 | — | — | — | EN | Kaggle  |

Примечания:
- MentalChat16K: `input`/`output`; разделение по `origin` из официальных CSV.
- Amod MHCC: `context`/`response`.
 - Reddit Mental Health Posts: 5 классов (`ADHD`, `aspergers`, `depression`, `OCD`, `ptsd`); среднее ≈51 слова, медиана 21, 99‑й перцентиль 381.
 - C‑SSRS Reddit 500: уникальных очищённых текстов 388


### Ссылки на ноутбуки
- `NLP4MentalHealth/EDA/MentalChat16K_EDA.ipynb`
- `NLP4MentalHealth/EDA/Amod_MHCC_EDA.ipynb`
- `mental-health/Helios_MHCB_EDA.ipynb`
- `NLP4MentalHealth/EDA/Reddit_Mental_Health_Posts_EDA.ipynb`
- `NLP4MentalHealth/EDA/Checkpoint_2_EDA_Kozhevnikov.ipynb`
