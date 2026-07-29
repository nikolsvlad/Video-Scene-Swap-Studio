"""
GUI-программа: замена фона видео + независимая панель QA-агента.

Запуск для разработки:  python gui_app.py
Превращение в .exe:      pyinstaller --onefile --windowed gui_app.py
"""

import os
import re
import io
import json
import shutil
import contextlib
import subprocess
import datetime
import threading
import time
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

from dotenv import load_dotenv
from gradio_client import Client, handle_file

import qa_agent
import footage_indexer

load_dotenv()
TOKEN = os.environ.get("HF_TOKEN", "")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHECK_INTERVAL_SECONDS = 30
TIMEOUT_SECONDS = 600

# --- Настройки ffmpeg-конвертера ---
FFMPEG_PATH = shutil.which("ffmpeg")

# формат контейнера -> список подходящих видеокодеков
# *_nvenc / *_qsv / *_amf — аппаратные энкодеры (Nvidia / Intel Quick Sync / AMD),
# работают только если есть соответствующее железо и драйверы с поддержкой ffmpeg
FORMAT_CODECS = {
    "mp4": ["libx264", "h264_nvenc", "h264_qsv", "h264_amf", "libx265", "mpeg4", "copy"],
    "mkv": ["libx264", "h264_nvenc", "h264_qsv", "h264_amf", "libx265", "libvpx-vp9", "copy"],
    "webm": ["libvpx-vp9", "libvpx"],
    "mov": ["libx264", "h264_nvenc", "h264_qsv", "h264_amf", "libx265", "mpeg4", "copy"],
    "avi": ["mpeg4", "libxvid"],
}

# кодеки, для которых имеет смысл выставлять -crf (программные энкодеры)
CRF_CAPABLE_CODECS = {"libx264", "libx265", "libvpx-vp9", "libvpx"}

# формат контейнера -> список ДОПУСТИМЫХ аудиокодеков.
# Важно: webm жёстко требует Vorbis/Opus — aac/mp3 в него не пишутся
# вообще (ffmpeg падает на этапе записи заголовка, а не кодирования).
FORMAT_AUDIO_CODECS = {
    "mp4": ["aac", "libmp3lame", "copy", "none"],
    "mkv": ["aac", "libmp3lame", "libopus", "copy", "none"],
    "webm": ["libopus", "libvorbis", "none"],
    "mov": ["aac", "libmp3lame", "copy", "none"],
    "avi": ["libmp3lame", "aac", "copy", "none"],
}

RESOLUTIONS = {
    "Оригинал": None,
    "3840x2160 (4K)": "3840:2160",
    "1920x1080 (Full HD)": "1920:1080",
    "1280x720 (HD)": "1280:720",
    "854x480 (SD)": "854:480",
    "640x360": "640:360",
    "Свой размер...": "custom",
}

FONT_HEADER = ("Segoe UI", 11, "bold")
FONT_NORMAL = ("Segoe UI", 9)
FONT_MONO = ("Consolas", 9)

COLOR_BG = "#f4f5f7"
COLOR_PANEL = "#ffffff"
COLOR_ACCENT = "#2f5d8a"
COLOR_ACCENT_TEXT = "#ffffff"
COLOR_BORDER = "#d0d3d8"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Video Scene Swap Studio")
        self.geometry("1150x820")
        self.minsize(1000, 650)
        self.configure(bg=COLOR_BG)

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TLabelframe", background=COLOR_PANEL, bordercolor=COLOR_BORDER)
        style.configure("TLabelframe.Label", background=COLOR_PANEL, font=FONT_HEADER)
        style.configure("TFrame", background=COLOR_PANEL)
        style.configure("TLabel", background=COLOR_PANEL, font=FONT_NORMAL)
        style.configure("TButton", font=FONT_NORMAL, padding=6)
        style.configure("TRadiobutton", background=COLOR_PANEL, font=FONT_NORMAL)
        style.configure("TCheckbutton", background=COLOR_PANEL, font=FONT_NORMAL)

        # --- Переменные: обработка видео ---
        self.source_path = tk.StringVar()
        self.bg_path = tk.StringVar()
        self.mode = tk.StringVar(value="file")
        self.bg_kind = tk.StringVar(value="video")
        self.quality_mode = tk.StringVar(value="fast")
        self.no_timeout = tk.BooleanVar(value=False)
        self.auto_convert = tk.BooleanVar(value=False)
        self.last_result_path = None

        # --- Переменные: панель QA-агента (независимая) ---
        self.qa_video_path = tk.StringVar()
        self.qa_scene_description = tk.StringVar(value="ад, огонь, лава")

        # --- Переменные: панель ffmpeg-конвертера (независимая) ---
        self.conv_source_path = tk.StringVar()
        self.conv_format = tk.StringVar(value="mp4")
        self.conv_codec = tk.StringVar(value="libx264")
        self.conv_resolution = tk.StringVar(value="Оригинал")
        self.conv_custom_width = tk.StringVar(value="1280")
        self.conv_custom_height = tk.StringVar(value="720")
        self.conv_audio_codec = tk.StringVar(value="aac")
        self.conv_crf = tk.IntVar(value=23)

        # --- Переменные: панель ручных эндпоинтов (независимая) ---
        self.manual_space_id = tk.StringVar(value="")
        self.manual_token = tk.StringVar(value="")
        self.manual_api_name = tk.StringVar(value="/predict")

        # --- Переменные: панель индексации футажей (независимая) ---
        self.index_folder_path = tk.StringVar()
        self.index_search_query = tk.StringVar()
        self.index_skip_existing = tk.BooleanVar(value=True)

        self.log_queue = queue.Queue()

        self._build_layout()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self._poll_log_queue()

    def on_close(self):
        self.destroy()
        os._exit(0)

    # ------------------------------------------------------------------
    # Общий каркас: заголовок + две колонки + лог снизу
    # ------------------------------------------------------------------

    def _build_layout(self):
        header = tk.Frame(self, bg=COLOR_ACCENT, height=48)
        header.pack(fill="x", side="top")
        tk.Label(
            header, text="Video Scene Swap Studio",
            bg=COLOR_ACCENT, fg=COLOR_ACCENT_TEXT, font=("Segoe UI", 14, "bold")
        ).pack(side="left", padx=16, pady=8)

        # PanedWindow вместо простого pack: гарантирует, что лог-панель снизу
        # никогда не будет вытолкнута за пределы окна, даже если одна из вкладок
        # notebook'а (например "Ручные эндпоинты") требует много места —
        # ttk.Notebook считает нужный размер по САМОЙ БОЛЬШОЙ вкладке, а не по
        # текущей, и без PanedWindow это может "съесть" место у лога.
        main_pane = tk.PanedWindow(
            self, orient=tk.VERTICAL, bg=COLOR_BG, sashwidth=6, sashrelief="raised"
        )
        main_pane.pack(fill="both", expand=True, padx=12, pady=12)

        body = tk.Frame(main_pane, bg=COLOR_BG)
        notebook = ttk.Notebook(body)
        notebook.pack(fill="both", expand=True)

        tab_process = ttk.Frame(notebook, padding=12)
        tab_qa = ttk.Frame(notebook, padding=12)
        tab_convert = ttk.Frame(notebook, padding=12)
        tab_index = ttk.Frame(notebook, padding=12)
        tab_manual = ttk.Frame(notebook, padding=12)

        notebook.add(tab_process, text="Обработка видео")
        notebook.add(tab_qa, text="QA-агент")
        notebook.add(tab_convert, text="Конвертер (ffmpeg)")
        notebook.add(tab_index, text="Банк футажей")
        notebook.add(tab_manual, text="Ручные эндпоинты")

        self._build_processing_panel(tab_process)
        self._build_agent_panel(tab_qa)
        self._build_convert_panel(tab_convert)
        self._build_footage_index_panel(tab_index)
        self._build_manual_endpoint_panel(tab_manual)

        # --- Лог: отдельная панель с гарантированным минимумом высоты ---
        log_frame = tk.Frame(main_pane, bg=COLOR_BG)

        log_header = tk.Frame(log_frame, bg=COLOR_BG)
        log_header.pack(fill="x")
        tk.Label(log_header, text="Журнал событий", bg=COLOR_BG, font=FONT_HEADER).pack(side="left")
        ttk.Button(log_header, text="Копировать лог", command=self.copy_log).pack(side="right")

        self.console = scrolledtext.ScrolledText(
            log_frame, height=10, bg="#1e1e1e", fg="#d4d4d4",
            insertbackground="white", font=FONT_MONO
        )
        self.console.pack(fill="both", expand=True, pady=(6, 0))
        self._make_selectable_readonly(self.console)

        main_pane.add(body, minsize=320, stretch="always")
        main_pane.add(log_frame, minsize=140, stretch="never")

    # ------------------------------------------------------------------
    # Левая панель: обработка видео
    # ------------------------------------------------------------------

    def _build_processing_panel(self, parent):
        ttk.Label(parent, text="1. Исходное видео").pack(anchor="w")
        row1 = ttk.Frame(parent)
        row1.pack(fill="x", pady=(2, 10))
        ttk.Entry(row1, textvariable=self.source_path, state="readonly").pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(row1, text="Обзор...", command=self.pick_source).pack(side="left", padx=(6, 0))

        ttk.Label(parent, text="2. Способ задать сцену").pack(anchor="w")
        mode_row = ttk.Frame(parent)
        mode_row.pack(fill="x", pady=(2, 4))
        ttk.Radiobutton(
            mode_row, text="Файл фона", variable=self.mode, value="file",
            command=self._refresh_mode
        ).pack(side="left")
        ttk.Radiobutton(
            mode_row, text="Текстовый промпт (пока недоступно)", variable=self.mode,
            value="prompt", command=self._refresh_mode
        ).pack(side="left", padx=(12, 0))

        self.file_frame = ttk.Frame(parent)
        self.file_frame.pack(fill="x", pady=(0, 10))

        kind_row = ttk.Frame(self.file_frame)
        kind_row.pack(fill="x", pady=(0, 4))
        ttk.Radiobutton(kind_row, text="Видео-фон", variable=self.bg_kind, value="video").pack(side="left")
        ttk.Radiobutton(kind_row, text="Картинка-фон", variable=self.bg_kind, value="image").pack(
            side="left", padx=(12, 0)
        )

        row2 = ttk.Frame(self.file_frame)
        row2.pack(fill="x")
        ttk.Entry(row2, textvariable=self.bg_path, state="readonly").pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(row2, text="Обзор...", command=self.pick_background).pack(side="left", padx=(6, 0))

        self.prompt_frame = ttk.Frame(parent)
        self.prompt_text = tk.Text(self.prompt_frame, height=3, font=FONT_NORMAL)
        self.prompt_text.pack(fill="x")

        self._refresh_mode()

        ttk.Label(parent, text="3. Режим обработки").pack(anchor="w", pady=(4, 0))
        quality_row = ttk.Frame(parent)
        quality_row.pack(fill="x", pady=(2, 4))
        ttk.Radiobutton(
            quality_row, text="Быстрый (экономит квоту)", variable=self.quality_mode, value="fast"
        ).pack(anchor="w")
        ttk.Radiobutton(
            quality_row, text="Медленный (лучше качество краёв)", variable=self.quality_mode, value="slow"
        ).pack(anchor="w")

        ttk.Checkbutton(
            parent, text="Без лимита ожидания", variable=self.no_timeout
        ).pack(anchor="w", pady=(2, 10))

        ttk.Checkbutton(
            parent, text="Автоматически конвертировать результат (ffmpeg)",
            variable=self.auto_convert
        ).pack(anchor="w", pady=(0, 2))
        ttk.Label(
            parent,
            text="Использует настройки формата/кодека/разрешения\nсо вкладки «Конвертер (ffmpeg)»",
            font=("Segoe UI", 8), foreground="gray", justify="left"
        ).pack(anchor="w", pady=(0, 10))

        ttk.Button(
            parent, text="Обработать видео", command=self.run_processing
        ).pack(fill="x", pady=(4, 4))

        self.status_label = tk.Label(
            parent, text="Готов к работе", fg="gray", bg=COLOR_PANEL, font=FONT_NORMAL,
            anchor="w", justify="left", wraplength=420
        )
        self.status_label.pack(fill="x", pady=(4, 0))

    def _refresh_mode(self):
        if self.mode.get() == "file":
            self.prompt_frame.pack_forget()
            self.file_frame.pack(fill="x", pady=(0, 10))
        else:
            self.file_frame.pack_forget()
            self.prompt_frame.pack(fill="x", pady=(0, 10))

    def pick_source(self):
        path = filedialog.askopenfilename(
            title="Выбери исходное видео",
            filetypes=[("Видео", "*.mp4 *.mov *.avi *.webm")]
        )
        if path:
            self.source_path.set(path)

    def pick_background(self):
        if self.bg_kind.get() == "video":
            filetypes = [("Видео", "*.mp4 *.mov *.avi *.webm")]
            title = "Выбери видео фона"
        else:
            filetypes = [("Картинки", "*.jpg *.jpeg *.png *.webp")]
            title = "Выбери картинку фона"
        path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        if path:
            self.bg_path.set(path)

    def run_processing(self):
        if not self.source_path.get():
            messagebox.showwarning("Внимание", "Сначала выбери исходное видео")
            return
        if self.mode.get() == "file" and not self.bg_path.get():
            messagebox.showwarning("Внимание", "Выбери видео фона")
            return
        if self.mode.get() == "prompt":
            messagebox.showinfo(
                "Пока недоступно",
                "Режим текстового промпта требует генеративной модели, "
                "которая сейчас не подключена. Используй режим 'Файл фона'."
            )
            return
        if self.auto_convert.get() and not FFMPEG_PATH:
            messagebox.showerror(
                "ffmpeg не найден",
                "Автоконвертация включена, но ffmpeg не найден в PATH.\n"
                "Установи ffmpeg или сними галочку."
            )
            return

        self.status_label.config(text="Обрабатываю...", fg="orange")
        thread = threading.Thread(target=self._process_in_background, daemon=True)
        thread.start()

    def _process_in_background(self):
        try:
            self.log("[Обработка] Отправляю запрос на Hugging Face Space...")
            result_path = process_video(
                self.source_path.get(), self.bg_path.get(), self.bg_kind.get(),
                fast_mode=(self.quality_mode.get() == "fast"),
                no_timeout=self.no_timeout.get(),
                log_callback=lambda m: self.log(f"[Обработка] {m}"),
            )
            self.log(f"[Обработка] Готово: {result_path}")

            if self.auto_convert.get():
                self.status_label.config(text="Конвертирую (ffmpeg)...", fg="orange")
                self.log("[Обработка] Автоконвертация (ffmpeg)...")
                try:
                    scale = self._resolve_conv_scale()
                    result_path = convert_video(
                        result_path,
                        output_format=self.conv_format.get(),
                        video_codec=self.conv_codec.get(),
                        scale=scale,
                        crf=self.conv_crf.get(),
                        audio_codec=self.conv_audio_codec.get(),
                        log_callback=lambda m: self.log(f"[Обработка] [ffmpeg] {m}"),
                    )
                    self.log(f"[Обработка] Конвертация завершена: {result_path}")
                except Exception as e:
                    # Не роняем весь пайплайн, если конвертация не удалась —
                    # результат замены фона уже готов, просто оставляем его как есть.
                    self.log(f"[Обработка] Конвертация не удалась, оставляю исходный файл: {e}")

            self.last_result_path = result_path
            self.status_label.config(text=f"Готово:\n{result_path}", fg="green")

            # Автоматически подставляем свежий (итоговый) результат в панель агента,
            # но пользователь может выбрать любой другой файл вручную
            self.qa_video_path.set(result_path)
            self._refresh_agent_hint()

            messagebox.showinfo("Готово", f"Результат сохранён:\n{result_path}")
        except Exception as e:
            self.log(f"[Обработка] ОШИБКА: {e}")
            self.status_label.config(text="Ошибка", fg="red")
            messagebox.showerror("Ошибка", str(e))

    # ------------------------------------------------------------------
    # Правая панель: независимый QA-агент
    # ------------------------------------------------------------------

    def _build_agent_panel(self, parent):
        ttk.Label(
            parent,
            text="Агент проверяет ЛЮБОЕ видео на диске — не обязательно\n"
                 "только что обработанное. Удобно, если файл уже готов,\n"
                 "а проверка в прошлый раз не была запущена.",
            justify="left"
        ).pack(anchor="w", pady=(0, 10))

        ttk.Label(parent, text="Видео для проверки").pack(anchor="w")
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(2, 4))
        ttk.Entry(row, textvariable=self.qa_video_path, state="readonly").pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(row, text="Обзор...", command=self.pick_qa_video).pack(side="left", padx=(6, 0))

        self.agent_hint_label = tk.Label(
            parent, text="", bg=COLOR_PANEL, fg="gray", font=("Segoe UI", 8), anchor="w"
        )
        self.agent_hint_label.pack(fill="x", pady=(0, 8))

        ttk.Label(parent, text="Ожидаемое описание сцены").pack(anchor="w")
        ttk.Entry(parent, textvariable=self.qa_scene_description).pack(fill="x", pady=(2, 10))

        self.qa_button = ttk.Button(
            parent, text="Запустить проверку", command=self.run_qa_check
        )
        self.qa_button.pack(fill="x", pady=(0, 10))

        # --- Подробная карточка результата ---
        result_box = tk.Frame(parent, bg="#f8f9fa", relief="groove", bd=1)
        result_box.pack(fill="both", expand=True)

        self.qa_verdict_label = tk.Label(
            result_box, text="Результат проверки появится здесь",
            bg="#f8f9fa", fg="gray", font=("Segoe UI", 11, "bold"),
            anchor="w", justify="left"
        )
        self.qa_verdict_label.pack(fill="x", padx=10, pady=(10, 4))

        self.qa_score_label = tk.Label(
            result_box, text="", bg="#f8f9fa", fg="black",
            font=("Segoe UI", 10, "bold"), anchor="w"
        )
        self.qa_score_label.pack(fill="x", padx=10)

        self.qa_flags_label = tk.Label(
            result_box, text="", bg="#f8f9fa", fg="black",
            font=FONT_NORMAL, anchor="w", justify="left"
        )
        self.qa_flags_label.pack(fill="x", padx=10, pady=(4, 4))

        ttk.Separator(result_box, orient="horizontal").pack(fill="x", padx=10, pady=4)

        tk.Label(
            result_box, text="Замечания агента:", bg="#f8f9fa",
            font=("Segoe UI", 9, "bold"), anchor="w"
        ).pack(fill="x", padx=10)
        self.qa_issues_label = tk.Label(
            result_box, text="—", bg="#f8f9fa", fg="#444",
            font=FONT_NORMAL, anchor="w", justify="left", wraplength=420
        )
        self.qa_issues_label.pack(fill="x", padx=10, pady=(0, 6))

        tk.Label(
            result_box, text="Рекомендация для следующей попытки:", bg="#f8f9fa",
            font=("Segoe UI", 9, "bold"), anchor="w"
        ).pack(fill="x", padx=10)
        self.qa_recommendation_label = tk.Label(
            result_box, text="—", bg="#f8f9fa", fg="#444",
            font=FONT_NORMAL, anchor="w", justify="left", wraplength=420
        )
        self.qa_recommendation_label.pack(fill="x", padx=10, pady=(0, 10))

        self.qa_last_result_text = ""
        ttk.Button(
            parent, text="Копировать результат", command=self.copy_qa_result
        ).pack(fill="x", pady=(8, 0))

    def pick_qa_video(self):
        path = filedialog.askopenfilename(
            title="Выбери видео для проверки агентом",
            initialdir=OUTPUT_DIR,
            filetypes=[("Видео", "*.mp4 *.mov *.avi *.webm")]
        )
        if path:
            self.qa_video_path.set(path)
            self._refresh_agent_hint()

    def _refresh_agent_hint(self):
        if self.qa_video_path.get() == self.last_result_path and self.last_result_path:
            self.agent_hint_label.config(text="Это свежий результат обработки")
        else:
            self.agent_hint_label.config(text="Выбран файл вручную")

    def run_qa_check(self):
        if not self.qa_video_path.get():
            messagebox.showwarning("Внимание", "Сначала выбери видео для проверки")
            return

        self.qa_button.config(state="disabled")
        self.qa_verdict_label.config(text="Проверяю...", fg="orange")
        self.log("[Агент] Запускаю QA-проверку (Gemini)...")
        thread = threading.Thread(target=self._run_qa_in_background, daemon=True)
        thread.start()

    def _run_qa_in_background(self):
        try:
            result = qa_agent.check_video(
                self.qa_video_path.get(), self.qa_scene_description.get()
            )
            self.log(
                f"[Агент] персонаж={result['character_consistent']}, "
                f"сцена={result['scene_matches']}, "
                f"оценка={result['quality_score']}/10"
            )

            QUALITY_THRESHOLD = 6
            passed = (
                result["character_consistent"]
                and result["scene_matches"]
                and result["quality_score"] >= QUALITY_THRESHOLD
            )
            verdict_text = "✅ ПРОШЛО ПРОВЕРКУ" if passed else "❌ НЕ ПРОШЛО ПРОВЕРКУ"
            verdict_color = "#2e7d32" if passed else "#c62828"

            self.qa_verdict_label.config(text=verdict_text, fg=verdict_color)
            self.qa_score_label.config(text=f"Оценка качества: {result['quality_score']}/10")
            self.qa_flags_label.config(
                text=(
                    f"Персонаж узнаваем: {'да' if result['character_consistent'] else 'нет'}   |   "
                    f"Сцена соответствует: {'да' if result['scene_matches'] else 'нет'}"
                )
            )
            self.qa_issues_label.config(text=result["issues"] or "—")
            self.qa_recommendation_label.config(text=result["recommendation"] or "—")

            self.qa_last_result_text = (
                f"{verdict_text}\n"
                f"Видео: {self.qa_video_path.get()}\n"
                f"Ожидаемая сцена: {self.qa_scene_description.get()}\n"
                f"Оценка качества: {result['quality_score']}/10\n"
                f"Персонаж узнаваем: {'да' if result['character_consistent'] else 'нет'}\n"
                f"Сцена соответствует: {'да' if result['scene_matches'] else 'нет'}\n"
                f"Замечания агента: {result['issues'] or '—'}\n"
                f"Рекомендация для следующей попытки: {result['recommendation'] or '—'}"
            )

        except Exception as e:
            self.log(f"[Агент] ОШИБКА: {e}")
            self.qa_verdict_label.config(text="⚠️ ОШИБКА ПРОВЕРКИ", fg="orange")
            self.qa_issues_label.config(text=str(e))
            self.qa_last_result_text = f"ОШИБКА ПРОВЕРКИ\nВидео: {self.qa_video_path.get()}\n{e}"
        finally:
            self.qa_button.config(state="normal")

    def copy_qa_result(self):
        if not self.qa_last_result_text:
            messagebox.showinfo("Нечего копировать", "Сначала запусти проверку")
            return
        self.clipboard_clear()
        self.clipboard_append(self.qa_last_result_text)
        self.log("(результат QA-проверки скопирован в буфер обмена)")

    # ------------------------------------------------------------------
    # Панель ffmpeg-конвертера (независимая)
    # ------------------------------------------------------------------

    def _build_convert_panel(self, parent):
        if not FFMPEG_PATH:
            warn = tk.Label(
                parent,
                text="⚠️ ffmpeg не найден в PATH. Установи ffmpeg и добавь его в PATH,\n"
                     "чтобы конвертер заработал (ffmpeg.org/download.html).",
                bg=COLOR_PANEL, fg="#c62828", font=FONT_NORMAL, justify="left", anchor="w"
            )
            warn.pack(anchor="w", pady=(0, 10))

        ttk.Label(parent, text="1. Видео для конвертации").pack(anchor="w")
        row1 = ttk.Frame(parent)
        row1.pack(fill="x", pady=(2, 10))
        ttk.Entry(row1, textvariable=self.conv_source_path, state="readonly").pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(row1, text="Обзор...", command=self.pick_convert_source).pack(side="left", padx=(6, 0))

        opts_row = ttk.Frame(parent)
        opts_row.pack(fill="x", pady=(0, 10))

        fmt_col = ttk.Frame(opts_row)
        fmt_col.pack(side="left", fill="x", expand=True, padx=(0, 6))
        ttk.Label(fmt_col, text="2. Формат (контейнер)").pack(anchor="w")
        fmt_combo = ttk.Combobox(
            fmt_col, textvariable=self.conv_format, state="readonly",
            values=list(FORMAT_CODECS.keys())
        )
        fmt_combo.pack(fill="x", pady=(2, 0))
        fmt_combo.bind("<<ComboboxSelected>>", self._refresh_codec_options)

        codec_col = ttk.Frame(opts_row)
        codec_col.pack(side="left", fill="x", expand=True, padx=(6, 0))
        ttk.Label(codec_col, text="3. Видеокодек").pack(anchor="w")
        self.codec_combo = ttk.Combobox(
            codec_col, textvariable=self.conv_codec, state="readonly",
            values=FORMAT_CODECS[self.conv_format.get()]
        )
        self.codec_combo.pack(fill="x", pady=(2, 0))

        ttk.Label(parent, text="4. Разрешение").pack(anchor="w", pady=(4, 0))
        res_row = ttk.Frame(parent)
        res_row.pack(fill="x", pady=(2, 4))
        res_combo = ttk.Combobox(
            res_row, textvariable=self.conv_resolution, state="readonly",
            values=list(RESOLUTIONS.keys()), width=20
        )
        res_combo.pack(side="left")
        res_combo.bind("<<ComboboxSelected>>", self._refresh_custom_res)

        self.custom_res_frame = ttk.Frame(res_row)
        ttk.Entry(self.custom_res_frame, textvariable=self.conv_custom_width, width=6).pack(side="left")
        ttk.Label(self.custom_res_frame, text=" x ").pack(side="left")
        ttk.Entry(self.custom_res_frame, textvariable=self.conv_custom_height, width=6).pack(side="left")
        self._refresh_custom_res()

        quality_row = ttk.Frame(parent)
        quality_row.pack(fill="x", pady=(6, 4))
        ttk.Label(quality_row, text="5. Качество (CRF, меньше = лучше и тяжелее)").pack(anchor="w")
        crf_row = ttk.Frame(parent)
        crf_row.pack(fill="x", pady=(2, 10))
        ttk.Scale(
            crf_row, from_=0, to=51, orient="horizontal",
            variable=self.conv_crf, command=self._on_crf_change
        ).pack(side="left", fill="x", expand=True)
        self.crf_value_label = ttk.Label(crf_row, text=str(self.conv_crf.get()), width=3)
        self.crf_value_label.pack(side="left", padx=(6, 0))

        audio_row = ttk.Frame(parent)
        audio_row.pack(fill="x", pady=(0, 10))
        ttk.Label(audio_row, text="6. Аудиокодек").pack(anchor="w")
        self.audio_combo = ttk.Combobox(
            audio_row, textvariable=self.conv_audio_codec, state="readonly",
            values=FORMAT_AUDIO_CODECS[self.conv_format.get()], width=20
        )
        self.audio_combo.pack(anchor="w", pady=(2, 0))

        ttk.Button(
            parent, text="Конвертировать", command=self.run_conversion
        ).pack(fill="x", pady=(4, 4))

        self.conv_status_label = tk.Label(
            parent, text="Готов к работе", fg="gray", bg=COLOR_PANEL, font=FONT_NORMAL,
            anchor="w", justify="left", wraplength=420
        )
        self.conv_status_label.pack(fill="x", pady=(4, 0))

    def _refresh_codec_options(self, event=None):
        fmt = self.conv_format.get()
        codecs = FORMAT_CODECS.get(fmt, [])
        self.codec_combo.config(values=codecs)
        if self.conv_codec.get() not in codecs and codecs:
            self.conv_codec.set(codecs[0])

        audio_codecs = FORMAT_AUDIO_CODECS.get(fmt, [])
        self.audio_combo.config(values=audio_codecs)
        if self.conv_audio_codec.get() not in audio_codecs and audio_codecs:
            self.conv_audio_codec.set(audio_codecs[0])

    def _refresh_custom_res(self, event=None):
        if self.conv_resolution.get() == "Свой размер...":
            self.custom_res_frame.pack(side="left", padx=(12, 0))
        else:
            self.custom_res_frame.pack_forget()

    def _on_crf_change(self, value):
        self.crf_value_label.config(text=str(int(float(value))))

    def pick_convert_source(self):
        path = filedialog.askopenfilename(
            title="Выбери видео для конвертации",
            filetypes=[("Видео", "*.mp4 *.mov *.avi *.webm *.mkv *.flv")]
        )
        if path:
            self.conv_source_path.set(path)

    def _resolve_conv_scale(self):
        """Возвращает строку scale для ffmpeg (или None), либо бросает ValueError."""
        resolution_key = self.conv_resolution.get()
        scale = RESOLUTIONS.get(resolution_key)
        if scale == "custom":
            try:
                w = int(self.conv_custom_width.get())
                h = int(self.conv_custom_height.get())
                scale = f"{w}:{h}"
            except ValueError:
                raise ValueError("Некорректная ширина/высота для конвертации")
        return scale

    def run_conversion(self):
        if not FFMPEG_PATH:
            messagebox.showerror("ffmpeg не найден", "Установи ffmpeg и добавь его в PATH.")
            return
        if not self.conv_source_path.get():
            messagebox.showwarning("Внимание", "Сначала выбери видео для конвертации")
            return

        try:
            scale = self._resolve_conv_scale()
        except ValueError as e:
            messagebox.showwarning("Внимание", str(e))
            return

        self.conv_status_label.config(text="Конвертирую...", fg="orange")
        thread = threading.Thread(
            target=self._run_conversion_in_background,
            args=(scale,),
            daemon=True,
        )
        thread.start()

    def _run_conversion_in_background(self, scale):
        try:
            result_path = convert_video(
                self.conv_source_path.get(),
                output_format=self.conv_format.get(),
                video_codec=self.conv_codec.get(),
                scale=scale,
                crf=self.conv_crf.get(),
                audio_codec=self.conv_audio_codec.get(),
                log_callback=lambda m: self.log(f"[Конвертер] {m}"),
            )
            self.log(f"[Конвертер] Готово: {result_path}")
            self.conv_status_label.config(text=f"Готово:\n{result_path}", fg="green")
            messagebox.showinfo("Готово", f"Результат сохранён:\n{result_path}")
        except Exception as e:
            self.log(f"[Конвертер] ОШИБКА: {e}")
            self.conv_status_label.config(text="Ошибка", fg="red")
            messagebox.showerror("Ошибка", str(e))

    # ------------------------------------------------------------------
    # Панель индексации банка футажей (независимая, эмбеддинги Gemini)
    # ------------------------------------------------------------------

    def _build_footage_index_panel(self, parent):
        ttk.Label(
            parent,
            text="Индексирует папку с видео (футажи/мемы): для каждого файла\n"
                 "Gemini генерирует описание и эмбеддинг, дальше можно искать\n"
                 "по смыслу текстовым запросом — не нужно помнить имена файлов.",
            justify="left"
        ).pack(anchor="w", pady=(0, 10))

        ttk.Label(parent, text="1. Папка с футажами").pack(anchor="w")
        row1 = ttk.Frame(parent)
        row1.pack(fill="x", pady=(2, 4))
        ttk.Entry(row1, textvariable=self.index_folder_path, state="readonly").pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(row1, text="Обзор...", command=self.pick_index_folder).pack(side="left", padx=(6, 0))

        ttk.Checkbutton(
            parent, text="Пропускать уже проиндексированные файлы (не переиндексировать заново)",
            variable=self.index_skip_existing
        ).pack(anchor="w", pady=(2, 8))

        self.index_button = ttk.Button(
            parent, text="Проиндексировать папку", command=self.run_index_folder
        )
        self.index_button.pack(fill="x", pady=(0, 4))

        self.index_status_label = tk.Label(
            parent, text="Метаданные будут сохранены в подпапку «_metadata» внутри выбранной папки",
            fg="gray", bg=COLOR_PANEL,
            font=FONT_NORMAL, anchor="w", justify="left", wraplength=800
        )
        self.index_status_label.pack(fill="x", pady=(0, 10))

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=(4, 10))

        ttk.Label(parent, text="2. Поиск по смыслу").pack(anchor="w")
        row2 = ttk.Frame(parent)
        row2.pack(fill="x", pady=(2, 4))
        ttk.Entry(row2, textvariable=self.index_search_query).pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(row2, text="Найти", command=self.run_index_search).pack(side="left", padx=(6, 0))

        ttk.Label(parent, text="Результаты (от наиболее похожего)").pack(anchor="w", pady=(8, 0))
        self.index_results_text = scrolledtext.ScrolledText(
            parent, height=12, font=FONT_MONO, wrap="word"
        )
        self.index_results_text.pack(fill="both", expand=True, pady=(2, 0))
        self._make_selectable_readonly(self.index_results_text)

        ttk.Button(
            parent, text="Копировать результаты", command=self.copy_index_results
        ).pack(fill="x", pady=(6, 0))

    def pick_index_folder(self):
        path = filedialog.askdirectory(title="Выбери папку с футажами")
        if path:
            self.index_folder_path.set(path)

    def run_index_folder(self):
        if not self.index_folder_path.get():
            messagebox.showwarning("Внимание", "Сначала выбери папку с футажами")
            return

        self.index_button.config(state="disabled")
        self.index_status_label.config(text="Индексирую...", fg="orange")
        self.log("[Индексация] Начинаю индексацию папки футажей...")
        thread = threading.Thread(target=self._run_index_folder_in_background, daemon=True)
        thread.start()

    def _run_index_folder_in_background(self):
        try:
            folder = self.index_folder_path.get()
            metadata_dir = footage_indexer.resolve_metadata_dir(folder)
            added = footage_indexer.index_folder(
                folder,
                metadata_dir=metadata_dir,
                log_callback=lambda m: self.log(f"[Индексация] {m}"),
                skip_existing=self.index_skip_existing.get(),
            )
            self.index_status_label.config(
                text=f"Готово. Новых/обновлённых файлов: {added}. Метаданные: {metadata_dir}",
                fg="green",
            )
            self.log(f"[Индексация] Готово, новых/обновлённых записей: {added}")
        except Exception as e:
            self.log(f"[Индексация] ОШИБКА: {e}")
            self.index_status_label.config(text="Ошибка индексации", fg="red")
            messagebox.showerror("Ошибка", str(e))
        finally:
            self.index_button.config(state="normal")

    def run_index_search(self):
        query = self.index_search_query.get().strip()
        if not query:
            messagebox.showwarning("Внимание", "Введи текстовый запрос для поиска")
            return
        if not self.index_folder_path.get():
            messagebox.showwarning("Внимание", "Сначала выбери папку с футажами")
            return

        self._set_readonly_text(self.index_results_text, "Ищу...")
        self.log(f"[Индексация] Поиск по запросу: {query}")
        thread = threading.Thread(target=self._run_index_search_in_background, args=(query,), daemon=True)
        thread.start()

    def _run_index_search_in_background(self, query):
        try:
            results = footage_indexer.search(
                query,
                folder_path=self.index_folder_path.get(),
                top_k=8,
                log_callback=lambda m: self.log(f"[Индексация] {m}"),
            )
            if not results:
                self._set_readonly_text(self.index_results_text, "Ничего не найдено.")
                return

            lines = []
            for i, r in enumerate(results, start=1):
                lines.append(
                    f"{i}. [{r['score']:.3f}] {os.path.basename(r['path'])}\n"
                    f"   Путь: {r['path']}\n"
                    f"   Описание: {r['description']}\n"
                )
            self._set_readonly_text(self.index_results_text, "\n".join(lines))
            self.log(f"[Индексация] Найдено результатов: {len(results)}")
        except Exception as e:
            self.log(f"[Индексация] ОШИБКА поиска: {e}")
            self._set_readonly_text(self.index_results_text, f"Ошибка: {e}")

    def copy_index_results(self):
        text = self.index_results_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showinfo("Нечего копировать", "Сначала выполни поиск")
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        self.log("(результаты поиска футажей скопированы в буфер обмена)")

    # ------------------------------------------------------------------
    # Панель ручных эндпоинтов Hugging Face (произвольная модель)
    # ------------------------------------------------------------------

    def _build_manual_endpoint_panel(self, parent):
        ttk.Label(
            parent,
            text="Позволяет дёрнуть ЛЮБОЙ Space на Hugging Face вручную —\n"
                 "не только тот, что зашит в 'Обработка видео'. Полезно для\n"
                 "тестирования новых моделей до того, как встраивать их в пайплайн.",
            justify="left"
        ).pack(anchor="w", pady=(0, 10))

        ttk.Label(parent, text="1. Space ID (например: author/space-name)").pack(anchor="w")
        ttk.Entry(parent, textvariable=self.manual_space_id).pack(fill="x", pady=(2, 8))

        row_token = ttk.Frame(parent)
        row_token.pack(fill="x", pady=(0, 8))
        token_col = ttk.Frame(row_token)
        token_col.pack(side="left", fill="x", expand=True, padx=(0, 6))
        ttk.Label(token_col, text="2. Токен (пусто = взять HF_TOKEN из .env)").pack(anchor="w")
        ttk.Entry(token_col, textvariable=self.manual_token, show="*").pack(fill="x", pady=(2, 0))

        api_col = ttk.Frame(row_token)
        api_col.pack(side="left", fill="x", expand=True, padx=(6, 0))
        ttk.Label(api_col, text="3. api_name эндпоинта").pack(anchor="w")
        ttk.Entry(api_col, textvariable=self.manual_api_name).pack(fill="x", pady=(2, 0))

        ttk.Button(
            parent, text="Показать API этого Space", command=self.run_show_api
        ).pack(fill="x", pady=(0, 8))

        ttk.Label(
            parent,
            text="Список эндпоинтов и их параметров появится в окне ниже.\n"
                 "Используй имена параметров как ключи в JSON справа.",
            font=("Segoe UI", 8), foreground="gray", justify="left"
        ).pack(anchor="w", pady=(0, 4))

        split = ttk.Frame(parent)
        split.pack(fill="both", expand=True, pady=(4, 8))

        api_frame = ttk.Frame(split)
        api_frame.pack(side="left", fill="both", expand=True, padx=(0, 6))
        ttk.Label(api_frame, text="Описание API (только чтение)").pack(anchor="w")
        self.manual_api_info = scrolledtext.ScrolledText(
            api_frame, height=8, font=FONT_MONO, wrap="word"
        )
        self.manual_api_info.pack(fill="both", expand=True, pady=(2, 0))
        self._make_selectable_readonly(self.manual_api_info)
        ttk.Button(
            api_frame, text="Копировать описание API", command=self.copy_manual_api_info
        ).pack(fill="x", pady=(4, 0))

        params_frame = ttk.Frame(split)
        params_frame.pack(side="left", fill="both", expand=True, padx=(6, 0))
        ttk.Label(params_frame, text="4. Параметры вызова (JSON)").pack(anchor="w")
        self.manual_params_text = tk.Text(params_frame, height=8, font=FONT_MONO, wrap="word")
        self.manual_params_text.pack(fill="both", expand=True, pady=(2, 0))
        self.manual_params_text.insert(
            "1.0",
            '{\n'
            '  "some_param": "значение",\n'
            '  "video_input": "C:/путь/к/файлу.mp4"\n'
            '}\n'
        )
        ttk.Label(
            params_frame,
            text="Строки с существующими путями к файлам автоматически\n"
                 "оборачиваются в handle_file() — не нужно делать это вручную.",
            font=("Segoe UI", 8), foreground="gray", justify="left"
        ).pack(anchor="w", pady=(4, 0))

        ttk.Button(
            parent, text="Выполнить", command=self.run_manual_call
        ).pack(fill="x", pady=(4, 4))

        self.manual_status_label = tk.Label(
            parent, text="Готов к работе", fg="gray", bg=COLOR_PANEL, font=FONT_NORMAL,
            anchor="w", justify="left", wraplength=800
        )
        self.manual_status_label.pack(fill="x", pady=(4, 4))

        ttk.Label(parent, text="Результат вызова").pack(anchor="w")
        self.manual_result_text = scrolledtext.ScrolledText(
            parent, height=5, font=FONT_MONO, wrap="word"
        )
        self.manual_result_text.pack(fill="both", expand=False, pady=(2, 0))
        self._make_selectable_readonly(self.manual_result_text)
        ttk.Button(
            parent, text="Копировать результат вызова", command=self.copy_manual_result
        ).pack(fill="x", pady=(6, 0))

    def copy_manual_api_info(self):
        text = self.manual_api_info.get("1.0", tk.END).strip()
        if not text:
            messagebox.showinfo("Нечего копировать", "Сначала загрузи описание API")
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        self.log("(описание API скопировано в буфер обмена)")

    def copy_manual_result(self):
        text = self.manual_result_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showinfo("Нечего копировать", "Сначала выполни вызов эндпоинта")
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        self.log("(результат вызова скопирован в буфер обмена)")

    def _get_manual_token(self):
        return self.manual_token.get().strip() or TOKEN

    def run_show_api(self):
        space_id = self.manual_space_id.get().strip()
        if not space_id:
            messagebox.showwarning("Внимание", "Укажи Space ID")
            return

        self.manual_status_label.config(text="Загружаю описание API...", fg="orange")
        thread = threading.Thread(target=self._show_api_in_background, args=(space_id,), daemon=True)
        thread.start()

    def _show_api_in_background(self, space_id):
        try:
            client = Client(space_id, token=self._get_manual_token())

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                client.view_api(print_info=True)
            info_text = buf.getvalue() or "(пусто — Space не вернул описание API)"

            self._set_readonly_text(self.manual_api_info, info_text)
            self.manual_status_label.config(text="Описание API загружено", fg="green")
            self.log(f"[Ручной эндпоинт] Показал API для {space_id}")
        except Exception as e:
            self.log(f"[Ручной эндпоинт] ОШИБКА при загрузке API: {e}")
            self.manual_status_label.config(text="Ошибка загрузки API", fg="red")
            self._set_readonly_text(self.manual_api_info, f"Ошибка: {e}")

    def run_manual_call(self):
        space_id = self.manual_space_id.get().strip()
        if not space_id:
            messagebox.showwarning("Внимание", "Укажи Space ID")
            return

        raw_params = self.manual_params_text.get("1.0", tk.END).strip()
        try:
            params = json.loads(raw_params) if raw_params else {}
        except json.JSONDecodeError as e:
            messagebox.showerror("Ошибка JSON", f"Не удалось разобрать параметры: {e}")
            return

        api_name = self.manual_api_name.get().strip() or None

        self.manual_status_label.config(text="Выполняю запрос...", fg="orange")
        self._set_readonly_text(self.manual_result_text, "")
        thread = threading.Thread(
            target=self._run_manual_call_in_background,
            args=(space_id, api_name, params),
            daemon=True,
        )
        thread.start()

    def _wrap_local_files(self, value):
        """Рекурсивно оборачивает строки-пути к существующим файлам в handle_file()."""
        if isinstance(value, str):
            if os.path.isfile(value):
                return handle_file(value)
            return value
        if isinstance(value, dict):
            return {k: self._wrap_local_files(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._wrap_local_files(v) for v in value]
        return value

    def _run_manual_call_in_background(self, space_id, api_name, params):
        try:
            client = Client(space_id, token=self._get_manual_token())
            wrapped_params = self._wrap_local_files(params)

            self.log(f"[Ручной эндпоинт] Вызываю {space_id} {api_name or ''} с параметрами: {params}")
            kwargs = dict(wrapped_params)
            if api_name:
                kwargs["api_name"] = api_name
            result = client.predict(**kwargs)

            result_text = json.dumps(result, ensure_ascii=False, indent=2, default=str)
            self._set_readonly_text(self.manual_result_text, result_text)
            self.manual_status_label.config(text="Готово", fg="green")
            self.log("[Ручной эндпоинт] Вызов успешно завершён")

            saved_path = self._save_first_output_file(result, space_id)
            if saved_path:
                self.log(f"[Ручной эндпоинт] Файл из результата сохранён: {saved_path}")
                self.manual_status_label.config(
                    text=f"Готово. Файл сохранён: {saved_path}", fg="green"
                )
        except Exception as e:
            self.log(f"[Ручной эндпоинт] ОШИБКА: {e}")
            self.manual_status_label.config(text="Ошибка вызова", fg="red")
            self._set_readonly_text(self.manual_result_text, f"Ошибка: {e}")

    def _save_first_output_file(self, result, space_id):
        """Ищет в результате первый путь к существующему файлу и копирует его в output/."""
        found = []

        def walk(value):
            if isinstance(value, str) and os.path.isfile(value):
                found.append(value)
            elif isinstance(value, dict):
                for v in value.values():
                    walk(v)
            elif isinstance(value, (list, tuple)):
                for v in value:
                    walk(v)

        walk(result)
        if not found:
            return None

        src = found[0]
        ext = os.path.splitext(src)[1] or ".bin"
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", space_id)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dest_name = f"manual_{safe_name}_{timestamp}{ext}"
        dest_path = os.path.join(OUTPUT_DIR, dest_name)
        shutil.copy(src, dest_path)
        return dest_path

    def _make_selectable_readonly(self, widget):
        """
        Держит текстовое поле в состоянии 'normal' (чтобы работало выделение
        мышью), но обрабатывает Ctrl+C/Ctrl+A напрямую по КОДУ ФИЗИЧЕСКОЙ
        клавиши (event.keycode), а не по символу (event.keysym) — символ
        зависит от раскладки клавиатуры (например, в русской раскладке Ctrl+C
        даёт keysym 'Cyrillic_es', а не 'c', и привязка по символу не сработает).
        Код клавиши (keycode) от раскладки не зависит: клавиша с буквой C
        физически всегда имеет один и тот же код.
        Всё остальное (печать символов, Delete, Backspace, Ctrl+V) блокируется —
        правки делает только программа через _set_readonly_text.
        """
        COPY_KEYCODE = 67   # физическая клавиша 'C'
        SELECT_ALL_KEYCODE = 65  # физическая клавиша 'A'
        ALLOWED_NAV_KEYSYMS = (
            "Left", "Right", "Up", "Down", "Home", "End",
            "Prior", "Next", "Tab", "Shift_L", "Shift_R",
            "Control_L", "Control_R",
        )

        def on_key(event):
            ctrl_held = bool(event.state & 0x4)

            if ctrl_held and event.keycode == COPY_KEYCODE:
                try:
                    selected = widget.get("sel.first", "sel.last")
                    widget.clipboard_clear()
                    widget.clipboard_append(selected)
                except tk.TclError:
                    pass  # нечего копировать — выделения нет
                return "break"

            if ctrl_held and event.keycode == SELECT_ALL_KEYCODE:
                widget.tag_add("sel", "1.0", "end-1c")
                return "break"

            if ctrl_held and event.keysym == "Insert":
                try:
                    selected = widget.get("sel.first", "sel.last")
                    widget.clipboard_clear()
                    widget.clipboard_append(selected)
                except tk.TclError:
                    pass
                return "break"

            if event.keysym in ALLOWED_NAV_KEYSYMS:
                return None

            return "break"

        widget.config(state="normal")
        widget.bind("<Key>", on_key)

    def _set_readonly_text(self, widget, text):
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)

    # ------------------------------------------------------------------
    # Общий лог
    # ------------------------------------------------------------------

    def copy_log(self):
        text = self.console.get("1.0", tk.END)
        self.clipboard_clear()
        self.clipboard_append(text)
        self.log("(лог скопирован в буфер обмена)")

    def log(self, message: str):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_queue.put(f"[{timestamp}] {message}")

    def _poll_log_queue(self):
        try:
            while True:
                message = self.log_queue.get_nowait()
                self.console.insert(tk.END, message + "\n")
                self.console.see(tk.END)
        except queue.Empty:
            pass
        self.after(200, self._poll_log_queue)


def process_video(source_path: str, background_path: str, bg_kind: str = "video",
                   fast_mode: bool = True, no_timeout: bool = False,
                   log_callback=print) -> str:
    client = Client(
        "innova-ai/video-background-removal",
        token=TOKEN,
        httpx_kwargs={"timeout": 600},
    )

    if bg_kind == "video":
        bg_type = "Video"
        bg_image_arg = None
        bg_video_arg = {"video": handle_file(background_path), "subtitles": None}
    else:
        bg_type = "Image"
        bg_image_arg = handle_file(background_path)
        bg_video_arg = None

    log_callback(f"Отправляю задачу в очередь (режим: {'быстрый' if fast_mode else 'медленный'})...")
    job = client.submit(
        vid={"video": handle_file(source_path), "subtitles": None},
        bg_type=bg_type,
        bg_image=bg_image_arg,
        bg_video=bg_video_arg,
        color="#00FF00",
        fps=0,
        video_handling="loop",
        fast_mode=fast_mode,
        max_workers=10,
        api_name="/fn",
    )

    waited = 0
    while not job.done():
        time.sleep(CHECK_INTERVAL_SECONDS)
        waited += CHECK_INTERVAL_SECONDS
        log_callback(f"...в процессе, прошло {waited} сек, статус: {job.status().code}")

        if not no_timeout and waited >= TIMEOUT_SECONDS:
            raise TimeoutError(
                f"Space не ответил за {TIMEOUT_SECONDS} сек. "
                f"Вероятно перегружен или завис. Включи 'Без лимита ожидания' при повторе."
            )

    result = None
    last_error = None
    for attempt in range(1, 4):
        try:
            result = job.result()
            break
        except Exception as e:
            last_error = e
            log_callback(f"Не удалось скачать результат (попытка {attempt}/3): {e}")
            time.sleep(5)

    if result is None:
        raise RuntimeError(f"Не удалось скачать результат после 3 попыток: {last_error}")

    _, final_output_video, _ = result

    source_name = os.path.splitext(os.path.basename(source_path))[0]
    bg_name = os.path.splitext(os.path.basename(background_path))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    final_name = f"{source_name}_{bg_name}_{timestamp}.mp4"
    final_path = os.path.join(OUTPUT_DIR, final_name)
    shutil.copy(final_output_video["video"], final_path)
    return final_path


def convert_video(source_path: str, output_format: str, video_codec: str,
                   scale: str = None, crf: int = 23, audio_codec: str = "aac",
                   log_callback=print) -> str:
    """
    Конвертирует видео через ffmpeg: формат, кодек, разрешение, аудиокодек.

    scale: строка вида "1920:1080" или None (оставить оригинальное разрешение).
    crf: 0-51, меньше = выше качество и больше размер файла (не используется для 'copy').
    audio_codec: "aac" / "copy" / "libmp3lame" / "none" (без звука).
    """
    if not FFMPEG_PATH:
        raise RuntimeError(
            "ffmpeg не найден в PATH. Установи ffmpeg (ffmpeg.org/download.html) "
            "и убедись, что команда 'ffmpeg' доступна из терминала."
        )
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Файл не найден: {source_path}")

    allowed_audio = FORMAT_AUDIO_CODECS.get(output_format, [])
    if audio_codec != "copy" and allowed_audio and audio_codec not in allowed_audio:
        raise ValueError(
            f"Аудиокодек '{audio_codec}' несовместим с контейнером .{output_format}. "
            f"Подходят: {', '.join(allowed_audio)}. "
            f"(Например, .webm принимает только Vorbis/Opus, но не AAC/MP3.)"
        )

    source_name = os.path.splitext(os.path.basename(source_path))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"{source_name}_conv_{timestamp}.{output_format}"
    output_path = os.path.join(OUTPUT_DIR, output_name)

    cmd = [FFMPEG_PATH, "-y", "-i", source_path]

    if video_codec == "copy":
        # При copy перекодирования нет, так что четность размеров не важна.
        pass
    elif scale:
        cmd += ["-vf", f"scale={scale}:force_original_aspect_ratio=disable,format=yuv420p"]
    else:
        # Без явного разрешения оставляем оригинальный размер, но подгоняем
        # ширину/высоту до чётных значений — большинство кодеков (H.264/H.265/VP9)
        # кодируют в yuv420p, где нечётные размеры кадра валят энкодер с
        # ошибкой "Invalid argument" (chroma-плоскости не делятся на 2).
        cmd += ["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p"]

    cmd += ["-c:v", video_codec]
    if video_codec != "copy" and video_codec in CRF_CAPABLE_CODECS:
        if video_codec in ("libvpx-vp9", "libvpx"):
            # У VP9/VP8 -crf работает только в паре с -b:v 0 (constant quality mode),
            # иначе энкодер остаётся в режиме fixed-bitrate и падает с Invalid argument.
            # crf=0 у vp9 означает lossless и требует отдельного флага, поэтому
            # снизу подстраховываемся минимумом 4.
            safe_crf = max(4, crf)
            cmd += ["-crf", str(safe_crf), "-b:v", "0"]
        else:
            cmd += ["-crf", str(crf)]
            if video_codec in ("libx264", "libx265"):
                cmd += ["-preset", "medium"]

    if audio_codec == "none":
        cmd += ["-an"]
    else:
        cmd += ["-c:a", audio_codec]

    cmd += [output_path]

    log_callback(f"Запускаю: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, bufsize=1,
    )

    duration_seconds = None
    for line in process.stdout:
        line = line.strip()
        if not line:
            continue

        if duration_seconds is None:
            m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", line)
            if m:
                h, mnt, s = m.groups()
                duration_seconds = int(h) * 3600 + int(mnt) * 60 + float(s)

        m = re.search(r"time=(\d+):(\d+):(\d+\.\d+)", line)
        if m:
            h, mnt, s = m.groups()
            current_seconds = int(h) * 3600 + int(mnt) * 60 + float(s)
            if duration_seconds:
                percent = min(100, int(current_seconds / duration_seconds * 100))
                log_callback(f"...прогресс {percent}%")
        elif "error" in line.lower():
            log_callback(line)

    process.wait()

    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg завершился с ошибкой (код {process.returncode})")

    if not os.path.exists(output_path):
        raise RuntimeError("ffmpeg отработал без ошибок, но выходной файл не найден")

    return output_path


if __name__ == "__main__":
    app = App()
    app.mainloop()