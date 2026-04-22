from __future__ import annotations

"""
CHAT_PIPELINE — точка интеграции чата.

Реальную логику подключайте в `run_pipeline` ниже.
Список файлов см. `CHAT_PIPELINE.md`.
"""


def run_pipeline(
    messages: list[dict[str, str]],
    latest_user_message: str,
) -> str:
    """
    Placeholder until the real run-pipeline is wired (another developer).

    Args:
        messages: History (system/user/assistant) as role/content dicts.
        latest_user_message: Newest user turn (last message in history).
    """
    _ = (messages, latest_user_message)
    return "Ок"
