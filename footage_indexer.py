import os
import re
import json
import base64
import time
import requests
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Модель для описания видео. Актуальная GA-модель на момент написания —
# gemini-3.6-flash (сильнее и дешевле gemini-3.5-flash). Если она недоступна
# в вашем регионе/аккаунте — верните "gemini-3.5-flash".
GENERATE_MODEL = "gemini-3.6-flash"
GENERATE_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GENERATE_MODEL}:generateContent"
)

# Сколько раз повторять запрос при временных ошибках сервиса (503/429/500)
# и с какой начальной паузой (сек), увеличивающейся экспоненциально.
MAX_RETRIES = 4
RETRY_BASE_DELAY = 5

# Модель для эмбеддингов (мультимодальная, но здесь эмбеддится только текст описания)
EMBED_MODEL = "gemini-embedding-2-preview"
EMBED_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{EMBED_MODEL}:embedContent"
)

EMBED_DIMENSIONS = 768  # 768 / 1536 / 3072 — компромисс между точностью и размером индекса

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".webm", ".mkv")

# Название подпапки с метаданными, создаётся автоматически внутри папки футажей,
# если явно не указана другая папка для метаданных.
DEFAULT_METADATA_SUBDIR = "_metadata"

HEADERS = {"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"}


class IndexerError(Exception):
    pass


def _post_with_retry(url: str, payload: dict, timeout: int, log_callback=print):
    """
    POST-запрос с повторными попытками при временных ошибках сервиса
    (503 Service Unavailable, 429 Too Many Requests, 500 Internal Server Error).
    Прочие ошибки (например 400 — неверный запрос, 404 — модель не найдена)
    пробрасываются сразу, без повторов, так как повтор их не исправит.
    """
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(url, headers=HEADERS, json=payload, timeout=timeout)
            if response.status_code in (429, 500, 503):
                delay = RETRY_BASE_DELAY * attempt
                log_callback(
                    f"Сервис временно недоступен ({response.status_code}), "
                    f"повтор через {delay} сек (попытка {attempt}/{MAX_RETRIES})..."
                )
                last_error = requests.exceptions.HTTPError(
                    f"{response.status_code} Server Error for url: {url}"
                )
                time.sleep(delay)
                continue
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            last_error = e
            delay = RETRY_BASE_DELAY * attempt
            log_callback(f"Ошибка сети/запроса, повтор через {delay} сек (попытка {attempt}/{MAX_RETRIES}): {e}")
            time.sleep(delay)

    raise IndexerError(f"Не удалось получить ответ после {MAX_RETRIES} попыток: {last_error}")


VIDEO_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
}


def _video_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _mime_type_for(video_path: str) -> str:
    ext = os.path.splitext(video_path)[1].lower()
    return VIDEO_MIME_TYPES.get(ext, "video/mp4")


def _describe_video(video_path: str, log_callback=print) -> str:
    """Просит Gemini коротко описать видео: сцена, объекты, настроение, теги."""
    if not GEMINI_API_KEY:
        raise IndexerError("GEMINI_API_KEY не задан в .env")

    video_b64 = _video_to_base64(video_path)

    prompt = (
        "Опиши это видео одним абзацем для поиска по банку футажей: что происходит, "
        "какая обстановка/сцена, какие объекты и люди видны, какое настроение/тон "
        "(например: юмористический, тревожный, спокойный). Если это мем — опиши, "
        "в чём его суть, и в каких ситуациях его уместно использовать. "
        "ВАЖНО: описывай только то, что видно и слышно уверенно. Если не можешь "
        "точно определить язык речи/песни, точные слова или другие неочевидные детали — "
        "не угадывай и не указывай их вообще, лучше опусти эту деталь. "
        "Не используй markdown, только обычный текст, 2-4 предложения."
    )

    payload = {
        "contents": [{
            "parts": [
                {"inline_data": {"mime_type": _mime_type_for(video_path), "data": video_b64}},
                {"text": prompt},
            ]
        }],
        "generationConfig": {"temperature": 0},
    }

    response = _post_with_retry(GENERATE_URL, payload, timeout=120, log_callback=log_callback)
    data = response.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def _embed_text(text: str, log_callback=print) -> list:
    """Превращает текст в вектор через Gemini Embedding 2."""
    if not GEMINI_API_KEY:
        raise IndexerError("GEMINI_API_KEY не задан в .env")

    payload = {
        "content": {"parts": [{"text": text}]},
        "output_dimensionality": EMBED_DIMENSIONS,
    }

    response = _post_with_retry(EMBED_URL, payload, timeout=60, log_callback=log_callback)
    data = response.json()
    return data["embedding"]["values"]


def _cosine_similarity(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def resolve_metadata_dir(folder_path: str, metadata_dir: str = None) -> str:
    """Возвращает папку метаданных: указанную явно, либо '_metadata' внутри папки футажей."""
    path = metadata_dir or os.path.join(folder_path, DEFAULT_METADATA_SUBDIR)
    os.makedirs(path, exist_ok=True)
    return path


def _meta_filename_for(video_path: str) -> str:
    """Имя файла метаданных = безопасное имя видео + .json (без исходного расширения)."""
    stem = os.path.splitext(os.path.basename(video_path))[0]
    safe_stem = re.sub(r"[^a-zA-Z0-9А-Яа-яЁё_-]", "_", stem)
    return safe_stem + ".json"


def _load_metadata_file(meta_path: str) -> dict:
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_metadata_file(meta_path: str, record: dict):
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def _load_all_metadata(metadata_dir: str) -> list:
    """Читает все .json-файлы метаданных из папки в один список записей."""
    records = []
    if not os.path.isdir(metadata_dir):
        return records
    for name in sorted(os.listdir(metadata_dir)):
        if not name.lower().endswith(".json"):
            continue
        try:
            records.append(_load_metadata_file(os.path.join(metadata_dir, name)))
        except (json.JSONDecodeError, OSError):
            continue
    return records


def index_folder(folder_path: str, metadata_dir: str = None, log_callback=print,
                  skip_existing: bool = True) -> int:
    """
    Проходит по всем видео в папке, генерирует описание + эмбеддинг,
    сохраняет КАЖДУЮ запись в СВОЙ файл (metadata_dir/<имя_видео>.json).
    Если skip_existing=True — пропускает файлы, у которых уже есть метаданные
    с тем же mtime (т.е. видео не менялось с прошлой индексации).

    Возвращает количество вновь проиндексированных/обновлённых файлов.
    """
    if not os.path.isdir(folder_path):
        raise IndexerError(f"Папка не найдена: {folder_path}")

    resolved_metadata_dir = resolve_metadata_dir(folder_path, metadata_dir)

    files = [
        os.path.join(folder_path, name)
        for name in sorted(os.listdir(folder_path))
        if name.lower().endswith(VIDEO_EXTENSIONS)
    ]

    if not files:
        log_callback("В папке не найдено видеофайлов")
        return 0

    added = 0
    for i, path in enumerate(files, start=1):
        mtime = os.path.getmtime(path)
        meta_path = os.path.join(resolved_metadata_dir, _meta_filename_for(path))

        if skip_existing and os.path.exists(meta_path):
            try:
                existing = _load_metadata_file(meta_path)
            except (json.JSONDecodeError, OSError):
                existing = None
            if existing and existing.get("mtime") == mtime:
                log_callback(f"[{i}/{len(files)}] Пропускаю (уже проиндексирован): {os.path.basename(path)}")
                continue

        log_callback(f"[{i}/{len(files)}] Описываю: {os.path.basename(path)}...")
        try:
            description = _describe_video(path, log_callback=log_callback)
        except Exception as e:
            log_callback(f"[{i}/{len(files)}] ОШИБКА описания {os.path.basename(path)}: {e}")
            continue

        log_callback(f"[{i}/{len(files)}] Эмбеддинг: {os.path.basename(path)}...")
        try:
            vector = _embed_text(description, log_callback=log_callback)
        except Exception as e:
            log_callback(f"[{i}/{len(files)}] ОШИБКА эмбеддинга {os.path.basename(path)}: {e}")
            continue

        record = {
            "path": path,
            "description": description,
            "embedding": vector,
            "mtime": mtime,
        }
        # Каждая запись — в свой собственный файл. Это и есть "отдельная папка
        # метаданных": одна папка, но каждый файл содержит метаданные ровно
        # одного видео, независимо от остальных.
        _save_metadata_file(meta_path, record)
        added += 1

        # Небольшая пауза, чтобы не упереться в лимит запросов в секунду.
        time.sleep(0.5)

    total = len(_load_all_metadata(resolved_metadata_dir))
    log_callback(
        f"Готово. Новых/обновлённых файлов метаданных: {added}. "
        f"Всего в папке метаданных: {total} ({resolved_metadata_dir})"
    )
    return added


def search(query: str, folder_path: str = None, metadata_dir: str = None,
           top_k: int = 5, log_callback=print) -> list:
    """
    Ищет футажи, наиболее близкие по смыслу к текстовому запросу.
    Можно передать либо folder_path (папку с футажами — метаданные найдутся
    в её подпапке '_metadata'), либо сразу metadata_dir напрямую.
    Возвращает список словарей: {"path", "description", "score"}, отсортированный
    по убыванию score (от 0 до 1, чем больше — тем более похоже).
    """
    resolved_metadata_dir = metadata_dir or (
        os.path.join(folder_path, DEFAULT_METADATA_SUBDIR) if folder_path else None
    )
    if not resolved_metadata_dir:
        raise IndexerError("Не указана ни папка футажей, ни папка метаданных")

    records = _load_all_metadata(resolved_metadata_dir)
    if not records:
        raise IndexerError("Метаданные не найдены. Сначала проиндексируйте папку с футажами.")

    query_vector = _embed_text(query, log_callback=log_callback)

    results = []
    for r in records:
        score = _cosine_similarity(query_vector, r["embedding"])
        results.append({
            "path": r["path"],
            "description": r["description"],
            "score": score,
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]
