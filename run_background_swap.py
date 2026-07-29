"""
Пример CLI-запуска замены фона видео без GUI.

Вся логика обращения к Hugging Face Space вынесена в executor.process_video,
чтобы не дублировать код с гуи-приложением.
"""

import os
from executor import process_video

# --- Пути: всё что нужно менять - тут, в одном месте ---
INPUT_DIR = "input"

SOURCE_VIDEO = os.path.join(INPUT_DIR, "source_clip.mp4")
BACKGROUND_VIDEO = os.path.join(INPUT_DIR, "bg_hell.mp4")

if __name__ == "__main__":
    final_path = process_video(
        SOURCE_VIDEO,
        BACKGROUND_VIDEO,
        bg_kind="video",
        fast_mode=True,
        no_timeout=False,
        log_callback=print,
    )
    print("Результат сохранён:", final_path)