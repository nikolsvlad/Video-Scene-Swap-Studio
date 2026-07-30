import os
import re
import json
import base64
import requests
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-3.5-flash:generateContent"
)


def _video_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _empty_result(issue: str) -> dict:
    return {
        "character_consistent": False,
        "scene_matches": False,
        "quality_score": 0,
        "issues": issue,
        "recommendation": "",
    }


def check_video(video_path: str, scene_description: str) -> dict:
    """
    Отправляет видео в Gemini, просит подробно оценить результат.

    Возвращает словарь:
      {
        "character_consistent": bool,
        "scene_matches": bool,
        "quality_score": int (1-10),
        "issues": str,
        "recommendation": str,
      }
    """
    if not GEMINI_API_KEY:
        return _empty_result("GEMINI_API_KEY не задан в .env")

    if not os.path.exists(video_path):
        return _empty_result(f"Файл не найден: {video_path}")

    try:
        video_b64 = _video_to_base64(video_path)
    except Exception as e:
        return _empty_result(f"Не удалось прочитать видео: {e}")

    prompt = (
        f"На этом видео должен быть человек, а фон/сцена должны соответствовать "
        f"описанию: '{scene_description}'. "
        f"Оцени результат честно и придирчиво, как эксперт по видеомонтажу. "
        f"Обрати внимание на: сохранение контура персонажа, артефакты по краям, "
        f"соответствие освещения персонажа и фона, общую убедительность композиции.\n\n"
        f"Ответь строго в формате JSON, без пояснений вне JSON:\n"
        f'{{"character_consistent": true/false, '
        f'"scene_matches": true/false, '
        f'"quality_score": <1-10>, '
        f'"issues": "конкретное описание проблем на русском (или пустая строка)", '
        f'"recommendation": "что изменить в следующей попытке, чтобы улучшить результат"}}'
    )

    payload = {
        "contents": [{
            "parts": [
                {"inline_data": {"mime_type": "video/mp4", "data": video_b64}},
                {"text": prompt},
            ]
        }]
    }
    headers = {"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"}

    try:
        response = requests.post(GEMINI_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return _empty_result(f"Ошибка запроса к Gemini: {e}")

    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        return _empty_result(f"Не удалось распарсить ответ модели: {raw_text[:200]}")

    try:
        result = json.loads(match.group())
    except json.JSONDecodeError as e:
        return _empty_result(f"Невалидный JSON от модели: {e}")

    result.setdefault("character_consistent", False)
    result.setdefault("scene_matches", False)
    result.setdefault("quality_score", 0)
    result.setdefault("issues", "")
    result.setdefault("recommendation", "")

    return result
