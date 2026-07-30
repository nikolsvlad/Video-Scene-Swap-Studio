import os
from executor import process_video

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
