# Сервис для работы с текстовыми запросами

FastAPI сервис с авторизацией через Keycloak и хранением истории запросов в Postgres.

## О приложении

- FastAPI-приложение (`fastapi-app`) c эндпоинтами:
  - `POST /forward` — обработка текста и сохранение результата в базу;
  - `GET /history`, `DELETE /history` — история запросов (только для администратора);
  - `GET /stats` — простая статистика по запросам;
  - `GET /docs` — Swagger UI, доступен только администратору.
- Keycloak (`fastapi_app` realm, клиент `fastapi-keycloak`), выдаёт JWT‑токены.
- PostgreSQL/alchemysql/alembic, хранение данных Keycloak и таблица `prompt_records` приложения.

## Как запустить

1. $ cd devops
2. $ docker compose up --build -d

После этого:

- приложение доступно по адресу `http://localhost:8080`;
- Keycloak — по `http://localhost:9000/auth`.

Останов - `docker compose down -v`.

## Тестовые пользователи

Keycloak создаёт двух пользователей в realm `fastapi_app`:

- `test_non_admin` / `123` — роль `user` в клиенте `fastapi-keycloak`;
- `test_admin` / `123` — роли `user` и `admin` в клиенте `fastapi-keycloak`.

Обычный пользователь может ходить на `/forward` и `/stats`, администратор дополнительно на `/history` и `/docs`.

## Примеры запросов

- Запросы в [request-examples.http](./request-examples.http)

## TODO

- Расширить БД, вынести diagnosis в отдельную таблицу
- Добавить простой фронт
- Сделать асинхронный предикт диагноза
- Рефакторинг в формат чата/чат-бота