import os
import requests
from fastapi import FastAPI, Request

app = FastAPI()

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")


def get_error_text(response: requests.Response) -> str:
    """Аккуратно пытается достать текст ошибки из ответа DeepSeek."""
    try:
        data = response.json()
    except ValueError:
        return f"DeepSeek вернул не-JSON ответ (HTTP {response.status_code})"

    if isinstance(data, dict) and "error" in data:
        error = data["error"]
        if isinstance(error, dict):
            return error.get("message", str(error))
        return str(error)

    return f"DeepSeek вернул HTTP {response.status_code}, но без поля 'choices'"


@app.post("/")
async def main(request: Request):
    body = await request.json()
    user_text = body["request"]["original_utterance"]

    # Если API-ключ не задан — сразу говорим об этом
    if not DEEPSEEK_API_KEY:
        return {
            "version": body["version"],
            "session": body["session"],
            "response": {
                "end_session": False,
                "text": "Ошибка: не задан API-ключ DeepSeek на сервере (DEEPSEEK_API_KEY).",
            },
        }

    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": user_text}],
            },
            timeout=20,
        )
    except requests.RequestException as exc:
        return {
            "version": body["version"],
            "session": body["session"],
            "response": {
                "end_session": False,
                "text": f"Ошибка сети при запросе к DeepSeek: {exc}",
            },
        }

    # Если DeepSeek ответил ошибкой (401, 404, 429 и т.д.) — показываем её
    if response.status_code != 200:
        return {
            "version": body["version"],
            "session": body["session"],
            "response": {
                "end_session": False,
                "text": f"Ошибка DeepSeek (HTTP {response.status_code}): {get_error_text(response)}",
            },
        }

    data = response.json()
    choices = data.get("choices") if isinstance(data, dict) else None

    # Если поля "choices" нет или список пустой — не падаем с KeyError
    if not choices:
        return {
            "version": body["version"],
            "session": body["session"],
            "response": {
                "end_session": False,
                "text": f"Некорректный ответ DeepSeek: {get_error_text(response)}",
            },
        }

    try:
        answer = choices[0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return {
            "version": body["version"],
            "session": body["session"],
            "response": {
                "end_session": False,
                "text": "DeepSeek вернул неожиданную структуру ответа. Попробуйте ещё раз.",
            },
        }

    return {
        "version": body["version"],
        "session": body["session"],
        "response": {
            "end_session": False,
            "text": answer,
        },
    }
