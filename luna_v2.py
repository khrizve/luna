"""Luna — mystical AI desktop assistant.

Fantasy-themed three-pane dashboard (CustomTkinter) on top of the original
voice-assistant backend: speech I/O (Piper TTS + Google speech recognition),
Gemini-powered Q&A, webcam face recognition, weather, web browsing/video
playback, and OS control (shutdown/restart/sleep/open apps).

New in this version: a persistent Memories store that Luna actually uses as
context when answering, a Settings page for API keys / color theme / voice,
lightweight Tasks & Reminders, and quick actions for opening apps or talking
to Luna.
"""

import datetime
import json
import logging
import os
import platform
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import uuid
import wave
import webbrowser
from logging.handlers import RotatingFileHandler
from pathlib import Path

import customtkinter as ctk
import pyjokes
import requests
import speech_recognition as sr
from PIL import Image, ImageDraw
from piper import PiperVoice
from piper.download_voices import download_voice

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# ------------------------------------------------------------
# Persistent data (settings / memories / tasks / reminders / chat log)
# ------------------------------------------------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
SETTINGS_PATH = os.path.join(DATA_DIR, "settings.json")
MEMORIES_PATH = os.path.join(DATA_DIR, "memories.json")
TASKS_PATH = os.path.join(DATA_DIR, "tasks.json")
REMINDERS_PATH = os.path.join(DATA_DIR, "reminders.json")
CHAT_HISTORY_PATH = os.path.join(DATA_DIR, "chat_history.json")


def load_json(path: str, default):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path: str, data) -> None:
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logging.error(f"Failed saving {path}: {e}")


# ------------------------------------------------------------
# Config — config.py / env vars are the bootstrap defaults; anything saved
# from the Settings page (data/settings.json) takes priority after that.
# ------------------------------------------------------------
try:
    import config as _cfg_mod
except ImportError:
    _cfg_mod = None


def _bootstrap(attr: str, env_key: str, default):
    if _cfg_mod is not None and getattr(_cfg_mod, attr, None):
        return getattr(_cfg_mod, attr)
    return os.environ.get(env_key, default)


WEATHER_API_KEY = _bootstrap("WEATHER_API_KEY", "WEATHER_API_KEY", "")
GEMINI_API_KEY = _bootstrap("GEMINI_API_KEY", "GEMINI_API_KEY", "")
HOME_CITY = _bootstrap("HOME_CITY", "HOME_CITY", "your city")
FACE_MATCH_THRESHOLD = float(_bootstrap("FACE_MATCH_THRESHOLD", "FACE_MATCH_THRESHOLD", 0.60))
PIPER_VOICE_NAME = _bootstrap("PIPER_VOICE", "PIPER_VOICE", "en_US-amy-medium")
USER_NAME = _bootstrap("USER_NAME", "LUNA_USER_NAME", "there")
THEME_NAME = "amethyst"

_saved_settings = load_json(SETTINGS_PATH, {})
WEATHER_API_KEY = _saved_settings.get("weather_api_key") or WEATHER_API_KEY
GEMINI_API_KEY = _saved_settings.get("gemini_api_key") or GEMINI_API_KEY
HOME_CITY = _saved_settings.get("home_city") or HOME_CITY
PIPER_VOICE_NAME = _saved_settings.get("piper_voice") or PIPER_VOICE_NAME
USER_NAME = _saved_settings.get("user_name") or USER_NAME
THEME_NAME = _saved_settings.get("theme") or THEME_NAME

FACE_MODEL = "sface"  # bumped if the recognizer model ever changes, to invalidate old face DBs
FACES_MODELS_DIR = "models"
FACE_DETECTOR_MODEL_PATH = os.path.join(FACES_MODELS_DIR, "face_detection_yunet_2023mar.onnx")
FACE_RECOGNIZER_MODEL_PATH = os.path.join(FACES_MODELS_DIR, "face_recognition_sface_2021dec.onnx")
FACE_DETECTOR_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
FACE_RECOGNIZER_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
_log_handler = RotatingFileHandler("luna.log", maxBytes=2_000_000, backupCount=3)
_log_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[_log_handler])

if not _CV2_AVAILABLE:
    logging.warning("OpenCV not installed — face detection disabled.")
if not _NUMPY_AVAILABLE:
    logging.warning("numpy not installed — face recognition disabled.")

# ------------------------------------------------------------
# Theme palette
# ------------------------------------------------------------
BG_DARK = "#07070f"
PANEL_BG = "#0e0e26"
PANEL_BG2 = "#161634"
PANEL_BORDER = "#332a66"
TEXT_LIGHT = "#e8e6fb"
TEXT_MUTED = "#8b86b8"

mode_colors = {
    "idle": "#9370db",
    "listening": "#00e5ff",
    "speaking": "#ff69b4",
    "executing": "#ffd76a",
    "confirm": "#ff4d4d",
}

THEMES = {
    "amethyst": {"name": "Amethyst Night", "accent": "#9370db", "accent2": "#00e5ff"},
    "crimson": {"name": "Crimson Ember", "accent": "#e6455e", "accent2": "#ffd76a"},
    "emerald": {"name": "Emerald Grove", "accent": "#2fd48f", "accent2": "#8be9fd"},
    "frost": {"name": "Frost Veil", "accent": "#6fb8ff", "accent2": "#e6e6fa"},
}

PIPER_VOICE_OPTIONS = [
    "en_US-amy-medium",
    "en_US-lessac-medium",
    "en_US-kristin-medium",
    "en_GB-alan-medium",
    "en_GB-southern_english_female-low",
]

LUNA_QUOTES = [
    "Even in the darkest nights, I'll be here... just for you.",
    "The stars whisper your name tonight.",
    "Magic is just patience wearing a cloak.",
    "Every question is a doorway — let's open it together.",
    "I keep your secrets safer than the moon keeps hers.",
    "Small steps still cross galaxies.",
    "Rest now — I'll keep watch over your tasks.",
    "Curiosity is the truest spell there is.",
    "Better days are coming... keep going.",
    "Between the stars, I found a friend in you.",
]

FONT_BRAND = ("Georgia", 24, "bold")
FONT_H1 = ("Georgia", 20, "bold")
FONT_H2 = ("Segoe UI", 13, "bold")
FONT_BODY = ("Segoe UI", 12)
FONT_SMALL = ("Segoe UI", 11)
FONT_TINY = ("Segoe UI", 9)


def apply_settings(new_values: dict) -> None:
    """Persists Settings-page edits and updates the live in-memory config."""
    global GEMINI_API_KEY, WEATHER_API_KEY, HOME_CITY, USER_NAME, THEME_NAME, PIPER_VOICE_NAME
    for key, value in new_values.items():
        if key == "gemini_api_key":
            GEMINI_API_KEY = value
        elif key == "weather_api_key":
            WEATHER_API_KEY = value
        elif key == "home_city":
            HOME_CITY = value
        elif key == "user_name":
            USER_NAME = value
        elif key == "theme":
            THEME_NAME = value
        elif key == "piper_voice":
            PIPER_VOICE_NAME = value
    merged = load_json(SETTINGS_PATH, {})
    merged.update(new_values)
    save_json(SETTINGS_PATH, merged)


# ------------------------------------------------------------
# Face model loading (YuNet + SFace, downloaded once into ./models)
# ------------------------------------------------------------
_face_detector = None
_face_recognizer = None
_FACE_MODELS_READY = False


def _ensure_face_model_files() -> bool:
    os.makedirs(FACES_MODELS_DIR, exist_ok=True)
    for path, url in (
        (FACE_DETECTOR_MODEL_PATH, FACE_DETECTOR_MODEL_URL),
        (FACE_RECOGNIZER_MODEL_PATH, FACE_RECOGNIZER_MODEL_URL),
    ):
        if os.path.exists(path):
            continue
        try:
            logging.info(f"Downloading face model: {url}")
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            with open(path, "wb") as f:
                f.write(resp.content)
        except Exception as e:
            logging.error(f"Failed to download face model {url}: {e}")
            return False
    return True


def _init_face_models() -> bool:
    global _face_detector, _face_recognizer, _FACE_MODELS_READY
    if not (_CV2_AVAILABLE and _NUMPY_AVAILABLE):
        return False
    if not _ensure_face_model_files():
        return False
    try:
        _face_detector = cv2.FaceDetectorYN_create(
            FACE_DETECTOR_MODEL_PATH, "", (320, 320), 0.8, 0.3, 5000
        )
        _face_recognizer = cv2.FaceRecognizerSF_create(FACE_RECOGNIZER_MODEL_PATH, "")
        _FACE_MODELS_READY = True
    except Exception as e:
        logging.error(f"Face model init error: {e}")
        _FACE_MODELS_READY = False
    return _FACE_MODELS_READY


_init_face_models()
if not _FACE_MODELS_READY:
    logging.warning("Face detection/recognition models unavailable — check ./models download.")


# ------------------------------------------------------------
# Voice — Piper neural TTS
# ------------------------------------------------------------
PIPER_VOICES_DIR = "voices"
_piper_voice = None
_PIPER_READY = False


def _piper_files_missing(model_path: str, config_path: str) -> bool:
    return not (
        os.path.exists(model_path) and os.path.getsize(model_path) > 0
        and os.path.exists(config_path) and os.path.getsize(config_path) > 0
    )


def _ensure_piper_voice_files(voice_name: str):
    os.makedirs(PIPER_VOICES_DIR, exist_ok=True)
    model_path = os.path.join(PIPER_VOICES_DIR, f"{voice_name}.onnx")
    config_path = os.path.join(PIPER_VOICES_DIR, f"{voice_name}.onnx.json")
    if _piper_files_missing(model_path, config_path):
        try:
            logging.info(f"Downloading Piper voice '{voice_name}' (first run only)...")
            download_voice(voice_name, Path(PIPER_VOICES_DIR))
        except Exception as e:
            logging.error(f"Failed to download Piper voice '{voice_name}': {e}")
            return None
    if _piper_files_missing(model_path, config_path):
        return None
    return model_path, config_path


def _init_piper_voice() -> bool:
    global _piper_voice, _PIPER_READY
    paths = _ensure_piper_voice_files(PIPER_VOICE_NAME)
    if paths is None:
        return False
    model_path, config_path = paths
    try:
        _piper_voice = PiperVoice.load(model_path, config_path=config_path)
        _PIPER_READY = True
    except Exception as e:
        logging.error(f"Piper voice init error: {e}")
        _PIPER_READY = False
    return _PIPER_READY


_init_piper_voice()
if not _PIPER_READY:
    logging.warning(
        "Piper TTS unavailable — check internet access on first run (needed to "
        "download the voice model) and that 'piper-tts' is installed."
    )

_AUDIO_PLAYERS = [p for p in ("paplay", "aplay", "ffplay") if shutil.which(p)]
if not _AUDIO_PLAYERS:
    logging.warning(
        "No audio player found (looked for paplay/aplay/ffplay) — Piper speech "
        "won't be audible. Install alsa-utils or pulseaudio-utils."
    )

# ------------------------------------------------------------
# Shared state
# ------------------------------------------------------------
speech_done_event = threading.Event()
_stop_speech_flag = threading.Event()
_pending_system_action = None
_app_running = True

FACES_DIR = "faces"
os.makedirs(FACES_DIR, exist_ok=True)
FACES_DB_PATH = os.path.join(FACES_DIR, "faces_db.json")
FACES_NPZ_PATH = os.path.join(FACES_DIR, "face_encodings.npz")

_known_face_encodings: list = []
_known_face_names: list = []
_last_greeted_name = None
_last_greeted_time = 0.0
GREET_COOLDOWN_SECONDS = 30
_awaiting_face_name = threading.Event()

# Only the camera thread writes _latest_frame; all reads go through
# _get_shared_frame(), avoiding a second competing cv2.VideoCapture(0).
_latest_frame = None
_face_cam_lock = threading.Lock()
_FRAME_WAIT_TIMEOUT = 3.0

app = None  # the LunaApp instance, set at the bottom of the file


def _load_face_db():
    global _known_face_encodings, _known_face_names
    if not _FACE_MODELS_READY:
        return
    try:
        if os.path.exists(FACES_DB_PATH) and os.path.exists(FACES_NPZ_PATH):
            with open(FACES_DB_PATH, "r") as f:
                data = json.load(f)
            if data.get("model") != FACE_MODEL:
                logging.warning(
                    f"Stored face DB was built with '{data.get('model')}', not '{FACE_MODEL}' "
                    "— ignoring old data. Re-register faces with 'remember my face'."
                )
                return
            npz = np.load(FACES_NPZ_PATH)
            _known_face_names = data.get("names", [])
            _known_face_encodings = [npz[f"enc_{i}"] for i in range(len(_known_face_names))]
            logging.info(f"Loaded {len(_known_face_names)} known face(s).")
    except Exception as e:
        logging.error(f"Face DB load error: {e}")


def _save_face_db():
    if not _FACE_MODELS_READY:
        return
    try:
        with open(FACES_DB_PATH, "w") as f:
            json.dump({"model": FACE_MODEL, "names": _known_face_names}, f)
        np.savez(FACES_NPZ_PATH, **{f"enc_{i}": enc for i, enc in enumerate(_known_face_encodings)})
    except Exception as e:
        logging.error(f"Face DB save error: {e}")


def register_face(name: str, encoding) -> None:
    name = name.strip().title()
    if name in _known_face_names:
        _known_face_encodings[_known_face_names.index(name)] = encoding
    else:
        _known_face_names.append(name)
        _known_face_encodings.append(encoding)
    _save_face_db()


def _cosine_distance(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))


def identify_face(encoding):
    if not _known_face_encodings:
        return None
    distances = [_cosine_distance(encoding, known) for known in _known_face_encodings]
    best_idx = int(np.argmin(distances))
    if distances[best_idx] < FACE_MATCH_THRESHOLD:
        return _known_face_names[best_idx]
    return None


def _detect_raw_faces(frame):
    if not _FACE_MODELS_READY:
        return None
    h, w = frame.shape[:2]
    _face_detector.setInputSize((w, h))
    try:
        _, faces = _face_detector.detect(frame)
    except Exception as e:
        logging.error(f"Face detection error: {e}")
        return None
    return faces


def _detect_face_boxes(frame) -> list:
    faces = _detect_raw_faces(frame)
    if faces is None:
        return []
    boxes = []
    for f in faces:
        x, y, w, h = f[:4].astype(int)
        boxes.append((y, x + w, y + h, x))
    return boxes


def _get_face_embeddings(frame) -> list:
    faces = _detect_raw_faces(frame)
    if faces is None:
        return []
    embeddings = []
    for f in faces:
        try:
            aligned = _face_recognizer.alignCrop(frame, f)
            embeddings.append(_face_recognizer.feature(aligned).flatten())
        except Exception as e:
            logging.error(f"Face embedding error: {e}")
    return embeddings


def _get_shared_frame():
    deadline = time.time() + _FRAME_WAIT_TIMEOUT
    while time.time() < deadline:
        with _face_cam_lock:
            if _latest_frame is not None:
                return _latest_frame.copy()
        time.sleep(0.05)
    return None


_load_face_db()


# ------------------------------------------------------------
# GUI — fantasy three-pane dashboard
# ------------------------------------------------------------
class LunaApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Luna — Mystical Assistant")
        self.geometry("1536x1000")
        self.minsize(1180, 760)
        self.configure(fg_color=BG_DARK)

        self.accent = THEMES[THEME_NAME]["accent"]
        self.accent2 = THEMES[THEME_NAME]["accent2"]
        self._theme_appliers = []
        self._avatar_cache = {}

        self.chat_history = load_json(CHAT_HISTORY_PATH, [])
        self.memories = load_json(MEMORIES_PATH, [])
        self.tasks = load_json(TASKS_PATH, [])
        self.reminders = load_json(REMINDERS_PATH, [])

        self.current_mode = "idle"
        self.current_page_key = "home"
        self.editing_memory_id = None
        self.quote_index = 0
        self.start_time = time.time()
        self.voice_listen_active = False

        self._build_layout()
        self._show_page("home")
        self._rotate_quote()
        self._tick_system_info()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------------- layout scaffolding ----------------
    def _build_layout(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_left_sidebar()
        self._build_center()
        self._build_right_sidebar()

    def _section_header(self, parent, text):
        ctk.CTkLabel(parent, text=text, font=FONT_H2, text_color=TEXT_MUTED).pack(
            anchor="w", padx=20, pady=(10, 2)
        )

    # ---------------- left sidebar ----------------
    def _build_left_sidebar(self):
        bar = ctk.CTkFrame(self, width=260, fg_color=PANEL_BG, corner_radius=0)
        bar.grid(row=0, column=0, sticky="nsew")
        bar.grid_propagate(False)

        brand = ctk.CTkFrame(bar, fg_color="transparent")
        brand.pack(pady=(28, 18), padx=22, fill="x")
        brand_label = ctk.CTkLabel(brand, text="☾ Luna", font=FONT_BRAND, text_color=self.accent)
        brand_label.pack(anchor="w")
        self._theme_appliers.append(lambda: brand_label.configure(text_color=self.accent))
        ctk.CTkLabel(brand, text="Your AI Assistant", font=FONT_SMALL, text_color=TEXT_MUTED).pack(anchor="w")

        nav_items = [
            ("home", "🏠  Home"),
            ("chat", "💬  Chat"),
            ("voice", "🎙  Voice"),
            ("memories", "🧠  Memories"),
            ("settings", "⚙  Settings"),
        ]
        self.nav_buttons = {}
        nav_frame = ctk.CTkFrame(bar, fg_color="transparent")
        nav_frame.pack(fill="x", padx=14, pady=6)
        for key, label in nav_items:
            btn = ctk.CTkButton(
                nav_frame, text=label, anchor="w", font=FONT_BODY,
                fg_color="transparent", hover_color=PANEL_BORDER, text_color=TEXT_LIGHT,
                corner_radius=10, height=42, command=lambda k=key: self._show_page(k),
            )
            btn.pack(fill="x", pady=3)
            self.nav_buttons[key] = btn

        ctk.CTkFrame(bar, fg_color=PANEL_BORDER, height=1).pack(fill="x", padx=20, pady=14)
        ctk.CTkLabel(bar, text="").pack(expand=True, fill="both")

        quote_wrap = ctk.CTkFrame(bar, fg_color="transparent")
        quote_wrap.pack(side="bottom", fill="x", padx=18, pady=22)
        self.quote_label = ctk.CTkLabel(
            quote_wrap, text="", font=("Georgia", 11, "italic"), text_color=TEXT_MUTED,
            wraplength=210, justify="center",
        )
        self.quote_label.pack()
        quote_sig = ctk.CTkLabel(quote_wrap, text="— Luna", font=("Georgia", 10, "italic"), text_color=self.accent)
        quote_sig.pack(pady=(4, 0))
        self._theme_appliers.append(lambda: quote_sig.configure(text_color=self.accent))

    # ---------------- center column ----------------
    def _build_center(self):
        center = ctk.CTkFrame(self, fg_color=BG_DARK, corner_radius=0)
        center.grid(row=0, column=1, sticky="nsew")
        center.grid_rowconfigure(0, weight=1)
        center.grid_columnconfigure(0, weight=1)

        self.page_container = ctk.CTkFrame(center, fg_color=BG_DARK)
        self.page_container.grid(row=0, column=0, sticky="nsew", padx=18, pady=(18, 8))
        self.page_container.grid_rowconfigure(0, weight=1)
        self.page_container.grid_columnconfigure(0, weight=1)

        self.pages = {}
        for key, builder in (
            ("home", self._build_home_page),
            ("chat", self._build_chat_page),
            ("voice", self._build_voice_page),
            ("memories", self._build_memories_page),
            ("settings", self._build_settings_page),
        ):
            frame = ctk.CTkFrame(self.page_container, fg_color="transparent")
            frame.grid(row=0, column=0, sticky="nsew")
            builder(frame)
            self.pages[key] = frame

        self._build_input_bar(center)

    def _show_page(self, key):
        for k, btn in self.nav_buttons.items():
            if k == key:
                btn.configure(fg_color=self.accent, text_color="#ffffff")
            else:
                btn.configure(fg_color="transparent", text_color=TEXT_LIGHT)
        self.pages[key].tkraise()
        self.current_page_key = key
        if key == "chat":
            self._render_chat(self.chat_page_scroll)
        elif key == "memories":
            self._render_memories()

    # ---------------- Home page ----------------
    def _build_home_page(self, frame):
        hero = ctk.CTkFrame(frame, fg_color=PANEL_BG, corner_radius=22, border_width=2, border_color=PANEL_BORDER)
        hero.pack(fill="both", expand=True)

        self.banner_frame = ctk.CTkFrame(
            hero, fg_color=PANEL_BG2, corner_radius=20, height=220,
            border_width=2, border_color=self._blend_hex(self.accent, PANEL_BG2, 0.4),
        )
        self.banner_frame.pack(fill="x", padx=30, pady=(26, 10))
        self.banner_frame.pack_propagate(False)
        self._theme_appliers.append(
            lambda: self.banner_frame.configure(border_color=self._blend_hex(self.accent, PANEL_BG2, 0.4))
        )
        self.banner_label = ctk.CTkLabel(self.banner_frame, text="", image=None)
        self.banner_label.place(relx=0.5, rely=0.5, anchor="center")
        self._banner_last_size = (0, 0)
        self.banner_frame.bind("<Configure>", self._on_banner_resize)

        self.mode_dot = ctk.CTkFrame(
            self.banner_frame, width=14, height=14, corner_radius=7, fg_color=mode_colors["idle"], border_width=0,
        )
        self.mode_dot.place(relx=0.965, rely=0.08, anchor="ne")

        self.greeting_label = ctk.CTkLabel(
            hero, text=f"Hello {USER_NAME}...  ♡", font=("Georgia", 22, "bold"), text_color=TEXT_LIGHT
        )
        self.greeting_label.pack(pady=(4, 2))
        ctk.CTkLabel(
            hero, text="I'm Luna, your AI assistant. What would you like to do today?",
            font=FONT_BODY, text_color=TEXT_MUTED, wraplength=520, justify="center",
        ).pack(pady=(0, 6))

        self.mode_status_label = ctk.CTkLabel(hero, text="✦ Idle ✦", font=("Segoe UI", 11, "bold"), text_color=self.accent)
        self.mode_status_label.pack(pady=(0, 10))

        preview_wrap = ctk.CTkFrame(hero, fg_color=PANEL_BG2, corner_radius=16)
        preview_wrap.pack(fill="both", expand=True, padx=40, pady=(0, 14))
        self.home_chat_scroll = ctk.CTkScrollableFrame(preview_wrap, fg_color="transparent")
        self.home_chat_scroll.pack(fill="both", expand=True, padx=10, pady=10)

        chip_row = ctk.CTkFrame(hero, fg_color="transparent")
        chip_row.pack(pady=(0, 18))
        for chip in ["💭 Explain something", "💻 Help with coding", "📅 Plan my day", "🎲 Tell me a story"]:
            ctk.CTkButton(
                chip_row, text=chip, font=FONT_SMALL, fg_color=PANEL_BG2, hover_color=PANEL_BORDER,
                text_color=TEXT_LIGHT, corner_radius=16, height=30,
                command=lambda t=chip: self._send_quick_prompt(t),
            ).pack(side="left", padx=5)

        self._render_chat(self.home_chat_scroll, limit=6)

    def _send_quick_prompt(self, chip_label):
        mapping = {
            "💭 Explain something": "Can you explain something interesting to me?",
            "💻 Help with coding": "I need help with coding.",
            "📅 Plan my day": "Help me plan my day.",
            "🎲 Tell me a story": "Tell me a short story.",
        }
        prompt = mapping.get(chip_label, chip_label)
        self.append_message("user", prompt)
        threading.Thread(target=process_command, args=(prompt.lower(),), daemon=True).start()

    # ---------------- Chat page ----------------
    def _build_chat_page(self, frame):
        ctk.CTkLabel(frame, text="Chat History", font=FONT_H1, text_color=TEXT_LIGHT).pack(anchor="w", pady=(4, 10))
        wrap = ctk.CTkFrame(frame, fg_color=PANEL_BG, corner_radius=18, border_width=2, border_color=PANEL_BORDER)
        wrap.pack(fill="both", expand=True)
        self.chat_page_scroll = ctk.CTkScrollableFrame(wrap, fg_color="transparent")
        self.chat_page_scroll.pack(fill="both", expand=True, padx=14, pady=14)

    # ---------------- chat rendering (shared by Home + Chat page) ----------------
    def _render_chat(self, container, limit=None):
        for w in container.winfo_children():
            w.destroy()
        history = self.chat_history[-limit:] if limit else self.chat_history
        if not history:
            ctk.CTkLabel(container, text="No messages yet — say hello!", font=FONT_SMALL, text_color=TEXT_MUTED).pack(pady=10)
            return
        tag_colors = {"user": None, "luna": None, "system": "#ffd76a", "error": "#ff6b6b"}
        for msg in history:
            role = msg.get("role", "luna")
            is_user = role == "user"
            row = ctk.CTkFrame(container, fg_color="transparent")
            row.pack(fill="x", pady=4)
            text_color = TEXT_LIGHT
            bubble_color = self.accent if is_user else PANEL_BG2
            if role in ("system", "error"):
                bubble_color = PANEL_BG2
                text_color = tag_colors.get(role, TEXT_MUTED)
            elif is_user:
                text_color = "#ffffff"
            bubble = ctk.CTkLabel(
                row, text=msg.get("text", ""), font=FONT_BODY, justify="left", text_color=text_color,
                fg_color=bubble_color, corner_radius=14, wraplength=380,
            )
            bubble.pack(side="right" if is_user else "left", padx=8, ipadx=8, ipady=6)
        try:
            container.update_idletasks()
            container._parent_canvas.yview_moveto(1.0)
        except Exception:
            pass

    def _refresh_chat_views(self):
        if hasattr(self, "home_chat_scroll"):
            self._render_chat(self.home_chat_scroll, limit=6)
        if hasattr(self, "chat_page_scroll") and self.current_page_key == "chat":
            self._render_chat(self.chat_page_scroll)
        self.update_overview()

    def append_message(self, role, text):
        def _do():
            entry = {"role": role, "text": text, "ts": datetime.datetime.now().isoformat()}
            self.chat_history.append(entry)
            save_json(CHAT_HISTORY_PATH, self.chat_history[-500:])
            self._refresh_chat_views()
        self.after(0, _do)

    # ---------------- mode / status ----------------
    def set_mode(self, mode):
        def _do():
            self.current_mode = mode
            color = mode_colors.get(mode, self.accent)
            if hasattr(self, "mode_dot"):
                self.mode_dot.configure(fg_color=color)
            if hasattr(self, "mode_status_label"):
                self.mode_status_label.configure(text=f"✦ {mode.capitalize()} ✦", text_color=color)
        self.after(0, _do)

    def set_status(self, text):
        def _do():
            if hasattr(self, "mode_status_label"):
                self.mode_status_label.configure(text=text)
        self.after(0, _do)

    def set_face_status(self, text):
        def _do():
            if hasattr(self, "info_labels"):
                self.info_labels["camera"].configure(text=text)
        self.after(0, _do)

    # ---------------- persistent input bar ----------------
    def _build_input_bar(self, parent):
        bar = ctk.CTkFrame(parent, fg_color=PANEL_BG, corner_radius=20, height=60, border_width=1, border_color=PANEL_BORDER)
        bar.grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 18))
        bar.grid_propagate(False)
        bar.grid_columnconfigure(0, weight=1)

        self.chat_entry = ctk.CTkEntry(
            bar, placeholder_text="Type your message...", font=FONT_BODY,
            fg_color="transparent", border_width=0, text_color=TEXT_LIGHT,
        )
        self.chat_entry.grid(row=0, column=0, sticky="ew", padx=(20, 8), pady=10)
        self.chat_entry.bind("<Return>", lambda e: self._handle_send())

        ctk.CTkButton(
            bar, text="🔇", width=36, height=36, corner_radius=18, fg_color=PANEL_BG2,
            hover_color=PANEL_BORDER, command=mute_luna,
        ).grid(row=0, column=1, padx=2, pady=8)

        ctk.CTkButton(
            bar, text="🎙", width=42, height=42, corner_radius=21, fg_color=PANEL_BG2,
            hover_color=PANEL_BORDER, command=self._handle_mic,
        ).grid(row=0, column=2, padx=4, pady=8)

        send_btn = ctk.CTkButton(
            bar, text="➤", width=42, height=42, corner_radius=21,
            fg_color=self.accent, hover_color=self.accent2, command=self._handle_send,
        )
        send_btn.grid(row=0, column=3, padx=(4, 10), pady=8)
        self._theme_appliers.append(lambda: send_btn.configure(fg_color=self.accent, hover_color=self.accent2))

    def _handle_send(self):
        text = self.chat_entry.get().strip()
        if not text:
            return
        self.chat_entry.delete(0, "end")
        self.append_message("user", text)
        threading.Thread(target=process_command, args=(text.lower(),), daemon=True).start()

    def _handle_mic(self):
        threading.Thread(target=self._mic_worker, daemon=True).start()

    def _mic_worker(self):
        cmd = listen()
        if cmd:
            process_command(cmd)

    # ---------------- avatar image ----------------
    def _load_avatar_image(self, size):
        if size in self._avatar_cache:
            return self._avatar_cache[size]
        candidates = [
            os.path.join("assets", n)
            for n in ("luna_portrait.png", "luna_portrait.jpg", "miss_luna.jpeg", "miss_luna.jpg", "miss_luna.png")
        ]
        path = next((p for p in candidates if os.path.exists(p)), None)
        img = None
        if path:
            try:
                img = Image.open(path).convert("RGBA").resize((size, size), Image.LANCZOS)
            except Exception:
                img = None
        if img is None:
            img = self._placeholder_avatar(size)
        mask = Image.new("L", (size, size), 0)
        ImageDraw.Draw(mask).ellipse((0, 0, size, size), fill=255)
        out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        out.paste(img, (0, 0), mask)
        photo = ctk.CTkImage(light_image=out, dark_image=out, size=(size, size))
        self._avatar_cache[size] = photo
        return photo

    @staticmethod
    def _placeholder_avatar(size):
        ph = Image.new("RGBA", (size, size), (20, 12, 46, 255))
        d = ImageDraw.Draw(ph)
        c = size // 2
        r = int(size * 0.32)
        d.ellipse((c - r, c - r, c + r, c + r), fill=(95, 55, 160, 255))
        d.ellipse((c - r + 10, c - r - 6, c + r - 22, c + r - 4), fill=(20, 12, 46, 255))
        return ph

    @staticmethod
    def _blend_hex(hex_a: str, hex_b: str, t: float) -> str:
        """Blends hex_a into hex_b by fraction t (0-1) — used to fake a
        semi-transparent, theme-colored border since Tk colors have no
        alpha channel."""
        a = tuple(int(hex_a[i:i + 2], 16) for i in (1, 3, 5))
        b = tuple(int(hex_b[i:i + 2], 16) for i in (1, 3, 5))
        blended = tuple(int(a[i] * t + b[i] * (1 - t)) for i in range(3))
        return "#{:02x}{:02x}{:02x}".format(*blended)

    def _load_banner_image(self, width, height):
        cache_key = ("banner", width, height)
        if cache_key in self._avatar_cache:
            return self._avatar_cache[cache_key]
        candidates = [
            os.path.join("assets", n)
            for n in ("luna_banner.png", "luna_banner.jpg", "luna_portrait.png", "luna_portrait.jpg",
                      "miss_luna.jpeg", "miss_luna.jpg", "miss_luna.png")
        ]
        path = next((p for p in candidates if os.path.exists(p)), None)
        img = None
        if path:
            try:
                from PIL import ImageOps
                img = ImageOps.fit(Image.open(path).convert("RGBA"), (width, height), Image.LANCZOS)
            except Exception:
                img = None
        if img is None:
            img = self._placeholder_banner(width, height)
        radius = 18
        mask = Image.new("L", (width, height), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, width, height), radius=radius, fill=255)
        out = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        out.paste(img, (0, 0), mask)
        photo = ctk.CTkImage(light_image=out, dark_image=out, size=(width, height))
        self._avatar_cache[cache_key] = photo
        return photo

    @staticmethod
    def _placeholder_banner(width, height):
        ph = Image.new("RGBA", (width, height), (18, 14, 42, 255))
        d = ImageDraw.Draw(ph)
        cx, cy = width * 0.24, height * 0.5
        r = height * 0.32
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(95, 55, 160, 255))
        d.ellipse((cx - r + 18, cy - r - 10, cx + r - 34, cy + r - 8), fill=(18, 14, 42, 255))
        import random as _r
        rng = _r.Random(42)  # fixed seed so the star field doesn't jump around on resize
        for _ in range(18):
            x = rng.uniform(width * 0.4, width * 0.98)
            y = rng.uniform(height * 0.06, height * 0.94)
            s = rng.uniform(1.2, 2.6)
            d.ellipse((x - s, y - s, x + s, y + s), fill=(210, 200, 255, 160))
        return ph

    def _on_banner_resize(self, event):
        # Inset by 8px so the frame's own border ring stays visible on every
        # side (an edge-to-edge image would otherwise paint over it).
        w, h = max(120, event.width - 8), max(60, event.height - 8)
        last_w, last_h = self._banner_last_size
        if abs(w - last_w) < 6 and abs(h - last_h) < 6:
            return
        self._banner_last_size = (w, h)
        self.banner_label.configure(image=self._load_banner_image(w, h))

    # ---------------- Voice page ----------------
    def _build_voice_page(self, frame):
        ctk.CTkLabel(frame, text="Voice", font=FONT_H1, text_color=TEXT_LIGHT).pack(anchor="w", pady=(4, 4))
        ctk.CTkLabel(
            frame, text="Choose the neural voice Luna speaks with (Piper TTS, fully offline once downloaded).",
            font=FONT_SMALL, text_color=TEXT_MUTED, wraplength=520, justify="left",
        ).pack(anchor="w", pady=(0, 14))

        box = self._settings_section(frame, "Assistant Voice")
        row = ctk.CTkFrame(box, fg_color="transparent")
        row.pack(fill="x", pady=6)
        ctk.CTkLabel(row, text="Voice", font=FONT_SMALL, text_color=TEXT_MUTED, width=100, anchor="w").pack(side="left")
        self.voice_menu = ctk.CTkOptionMenu(
            row, values=PIPER_VOICE_OPTIONS, fg_color=PANEL_BG2,
            button_color=self.accent, button_hover_color=self.accent2,
        )
        self.voice_menu.set(PIPER_VOICE_NAME if PIPER_VOICE_NAME in PIPER_VOICE_OPTIONS else PIPER_VOICE_OPTIONS[0])
        self.voice_menu.pack(side="left", padx=(6, 0))
        self._theme_appliers.append(
            lambda: self.voice_menu.configure(button_color=self.accent, button_hover_color=self.accent2)
        )

        btn_row = ctk.CTkFrame(box, fg_color="transparent")
        btn_row.pack(fill="x", pady=(10, 4))
        apply_btn = ctk.CTkButton(btn_row, text="Apply Voice", fg_color=self.accent, hover_color=self.accent2, command=self._apply_voice)
        apply_btn.pack(side="left")
        self._theme_appliers.append(lambda: apply_btn.configure(fg_color=self.accent, hover_color=self.accent2))
        ctk.CTkButton(
            btn_row, text="🔊 Test Speech", fg_color=PANEL_BG2, hover_color=PANEL_BORDER,
            command=lambda: threading.Thread(target=speak, args=("Hello! This is how I sound.",), daemon=True).start(),
        ).pack(side="left", padx=8)

        self.voice_status_label = ctk.CTkLabel(box, text=("Voice ready ✓" if _PIPER_READY else "Voice not ready — apply to download."), font=FONT_TINY, text_color=TEXT_MUTED)
        self.voice_status_label.pack(anchor="w", pady=(6, 0))

    def _apply_voice(self):
        chosen = self.voice_menu.get()
        self.voice_status_label.configure(text="Downloading & applying voice…")

        def worker():
            apply_settings({"piper_voice": chosen})
            ok = _init_piper_voice()
            text = "Voice ready ✓" if ok else "Failed to load voice — check your connection."
            self.after(0, lambda: self.voice_status_label.configure(text=text))

        threading.Thread(target=worker, daemon=True).start()

    # ---------------- Memories page ----------------
    def _build_memories_page(self, frame):
        ctk.CTkLabel(frame, text="Memories", font=FONT_H1, text_color=TEXT_LIGHT).pack(anchor="w", pady=(4, 4))
        ctk.CTkLabel(
            frame, text="Things Luna remembers about you — she uses these to personalize her answers.",
            font=FONT_SMALL, text_color=TEXT_MUTED,
        ).pack(anchor="w", pady=(0, 10))

        add_wrap = ctk.CTkFrame(frame, fg_color=PANEL_BG, corner_radius=16, border_width=1, border_color=PANEL_BORDER)
        add_wrap.pack(fill="x", pady=(0, 14))
        self.memory_input = ctk.CTkTextbox(
            add_wrap, height=70, fg_color=PANEL_BG2, text_color=TEXT_LIGHT, corner_radius=10, font=FONT_BODY,
        )
        self.memory_input.pack(fill="x", padx=14, pady=(14, 8))
        btn_row = ctk.CTkFrame(add_wrap, fg_color="transparent")
        btn_row.pack(fill="x", padx=14, pady=(0, 14))
        self.memory_save_btn = ctk.CTkButton(
            btn_row, text="Save Memory", fg_color=self.accent, hover_color=self.accent2, command=self._save_memory,
        )
        self.memory_save_btn.pack(side="left")
        self._theme_appliers.append(lambda: self.memory_save_btn.configure(fg_color=self.accent, hover_color=self.accent2))
        ctk.CTkButton(
            btn_row, text="Cancel Edit", fg_color=PANEL_BG2, hover_color=PANEL_BORDER, command=self._cancel_memory_edit,
        ).pack(side="left", padx=8)

        list_wrap = ctk.CTkFrame(frame, fg_color=PANEL_BG, corner_radius=16, border_width=1, border_color=PANEL_BORDER)
        list_wrap.pack(fill="both", expand=True)
        self.memories_scroll = ctk.CTkScrollableFrame(list_wrap, fg_color="transparent")
        self.memories_scroll.pack(fill="both", expand=True, padx=10, pady=10)
        self._render_memories()

    def _render_memories(self):
        for w in self.memories_scroll.winfo_children():
            w.destroy()
        if not self.memories:
            ctk.CTkLabel(self.memories_scroll, text="No memories yet — add one above.", text_color=TEXT_MUTED).pack(pady=10)
            return
        for mem in reversed(self.memories):
            card = ctk.CTkFrame(self.memories_scroll, fg_color=PANEL_BG2, corner_radius=12)
            card.pack(fill="x", pady=5, padx=2)
            ctk.CTkLabel(
                card, text=mem["text"], font=FONT_BODY, text_color=TEXT_LIGHT,
                wraplength=380, justify="left", anchor="w",
            ).pack(fill="x", padx=12, pady=(10, 4))
            meta_row = ctk.CTkFrame(card, fg_color="transparent")
            meta_row.pack(fill="x", padx=12, pady=(0, 10))
            ts = mem.get("created_at", "")[:16].replace("T", " ")
            ctk.CTkLabel(meta_row, text=ts, font=FONT_TINY, text_color=TEXT_MUTED).pack(side="left")
            ctk.CTkButton(
                meta_row, text="Edit", width=50, height=24, font=FONT_TINY, fg_color="transparent",
                hover_color=PANEL_BORDER, text_color=self.accent2, command=lambda m=mem: self._edit_memory(m),
            ).pack(side="right", padx=2)
            ctk.CTkButton(
                meta_row, text="Delete", width=55, height=24, font=FONT_TINY, fg_color="transparent",
                hover_color="#5a1414", text_color="#ff6b6b", command=lambda m=mem: self._delete_memory(m),
            ).pack(side="right", padx=2)

    def _save_memory(self):
        text = self.memory_input.get("1.0", "end").strip()
        if not text:
            return
        if self.editing_memory_id:
            for m in self.memories:
                if m["id"] == self.editing_memory_id:
                    m["text"] = text
                    break
            self.editing_memory_id = None
            self.memory_save_btn.configure(text="Save Memory")
        else:
            self.memories.append({"id": str(uuid.uuid4()), "text": text, "created_at": datetime.datetime.now().isoformat()})
        save_json(MEMORIES_PATH, self.memories)
        self.memory_input.delete("1.0", "end")
        self._render_memories()

    def _edit_memory(self, mem):
        self.editing_memory_id = mem["id"]
        self.memory_input.delete("1.0", "end")
        self.memory_input.insert("1.0", mem["text"])
        self.memory_save_btn.configure(text="Update Memory")

    def _cancel_memory_edit(self):
        self.editing_memory_id = None
        self.memory_input.delete("1.0", "end")
        self.memory_save_btn.configure(text="Save Memory")

    def _delete_memory(self, mem):
        self.memories = [m for m in self.memories if m["id"] != mem["id"]]
        save_json(MEMORIES_PATH, self.memories)
        self._render_memories()

    # ---------------- Settings page ----------------
    def _settings_section(self, parent, title):
        box = ctk.CTkFrame(parent, fg_color=PANEL_BG, corner_radius=16, border_width=1, border_color=PANEL_BORDER)
        box.pack(fill="x", pady=8)
        header = ctk.CTkLabel(box, text=title, font=FONT_H2, text_color=self.accent)
        header.pack(anchor="w", padx=16, pady=(12, 4))
        self._theme_appliers.append(lambda: header.configure(text_color=self.accent))
        inner = ctk.CTkFrame(box, fg_color="transparent")
        inner.pack(fill="x", padx=16, pady=(0, 12))
        return inner

    def _build_settings_page(self, frame):
        scroll = ctk.CTkScrollableFrame(frame, fg_color="transparent")
        scroll.pack(fill="both", expand=True)
        ctk.CTkLabel(scroll, text="Settings", font=FONT_H1, text_color=TEXT_LIGHT).pack(anchor="w", pady=(4, 14))

        api_box = self._settings_section(scroll, "API & Profile")
        self.settings_entries = {}
        field_values = {
            "gemini_api_key": GEMINI_API_KEY, "weather_api_key": WEATHER_API_KEY,
            "home_city": HOME_CITY, "user_name": USER_NAME,
        }
        for key, label, mask in [
            ("gemini_api_key", "Gemini API Key", "*"),
            ("weather_api_key", "Weather API Key", "*"),
            ("home_city", "Home City", None),
            ("user_name", "Your Name", None),
        ]:
            row = ctk.CTkFrame(api_box, fg_color="transparent")
            row.pack(fill="x", pady=5)
            ctk.CTkLabel(row, text=label, font=FONT_SMALL, text_color=TEXT_MUTED, width=140, anchor="w").pack(side="left")
            entry = ctk.CTkEntry(row, fg_color=PANEL_BG2, text_color=TEXT_LIGHT, border_width=0, show=mask)
            entry.pack(side="left", fill="x", expand=True, padx=(6, 0))
            entry.insert(0, field_values.get(key, ""))
            self.settings_entries[key] = entry
        save_api_btn = ctk.CTkButton(api_box, text="Save API Settings", fg_color=self.accent, hover_color=self.accent2, command=self._save_api_settings)
        save_api_btn.pack(anchor="e", pady=(10, 4))
        self._theme_appliers.append(lambda: save_api_btn.configure(fg_color=self.accent, hover_color=self.accent2))

        theme_box = self._settings_section(scroll, "Appearance")
        row = ctk.CTkFrame(theme_box, fg_color="transparent")
        row.pack(fill="x", pady=5)
        ctk.CTkLabel(row, text="Theme", font=FONT_SMALL, text_color=TEXT_MUTED, width=140, anchor="w").pack(side="left")
        self.theme_menu = ctk.CTkOptionMenu(
            row, values=[t["name"] for t in THEMES.values()], command=self._on_theme_selected,
            fg_color=PANEL_BG2, button_color=self.accent, button_hover_color=self.accent2,
        )
        self.theme_menu.set(THEMES[THEME_NAME]["name"])
        self.theme_menu.pack(side="left", padx=(6, 0))
        self._theme_appliers.append(
            lambda: self.theme_menu.configure(button_color=self.accent, button_hover_color=self.accent2)
        )

        sys_box = self._settings_section(scroll, "System")
        ctk.CTkLabel(
            sys_box, text="Use the power icons in the right sidebar to shut down, restart, or sleep this computer.",
            font=FONT_SMALL, text_color=TEXT_MUTED, wraplength=440, justify="left",
        ).pack(anchor="w", pady=6)

    def _save_api_settings(self):
        vals = {k: e.get().strip() for k, e in self.settings_entries.items()}
        apply_settings(vals)
        self.append_message("system", "⚙ Settings saved.")

    def _on_theme_selected(self, display_name):
        key = next((k for k, v in THEMES.items() if v["name"] == display_name), THEME_NAME)
        self.accent = THEMES[key]["accent"]
        self.accent2 = THEMES[key]["accent2"]
        apply_settings({"theme": key})
        for fn in self._theme_appliers:
            try:
                fn()
            except Exception:
                pass
        self._show_page(self.current_page_key)

    # ---------------- right sidebar ----------------
    def _build_right_sidebar(self):
        bar = ctk.CTkFrame(self, width=300, fg_color=PANEL_BG, corner_radius=0)
        bar.grid(row=0, column=2, sticky="nsew")
        bar.grid_propagate(False)

        prof = ctk.CTkFrame(bar, fg_color="transparent")
        prof.pack(fill="x", padx=20, pady=(26, 14))
        avatar_small = ctk.CTkLabel(prof, text="", image=self._load_avatar_image(52))
        avatar_small.grid(row=0, column=0, rowspan=2, padx=(0, 10))
        ctk.CTkLabel(prof, text="Luna", font=("Segoe UI", 16, "bold"), text_color=TEXT_LIGHT).grid(row=0, column=1, sticky="w")
        status_row = ctk.CTkFrame(prof, fg_color="transparent")
        status_row.grid(row=1, column=1, sticky="w")
        ctk.CTkLabel(status_row, text="●", text_color="#33d17a", font=FONT_TINY).pack(side="left")
        ctk.CTkLabel(status_row, text=" Online", text_color="#33d17a", font=FONT_SMALL).pack(side="left")

        self._section_header(bar, "◈  Quick Actions")
        qa_grid = ctk.CTkFrame(bar, fg_color="transparent")
        qa_grid.pack(fill="x", padx=20, pady=(6, 16))
        qa_grid.grid_columnconfigure((0, 1), weight=1)
        actions = [
            ("💬", "Ask Anything", self.qa_ask_anything),
            ("✅", "Create Task", self.qa_create_task),
            ("🗂", "Open Apps", self.qa_open_apps),
            ("🎙", "Voice Chat", self.qa_toggle_voice),
        ]
        for i, (icon, label, cmd) in enumerate(actions):
            r, c = divmod(i, 2)
            btn = ctk.CTkButton(
                qa_grid, text=f"{icon}\n{label}", font=FONT_SMALL, fg_color=PANEL_BG2,
                hover_color=PANEL_BORDER, text_color=TEXT_LIGHT, corner_radius=14, height=64, command=cmd,
            )
            btn.grid(row=r, column=c, padx=6, pady=6, sticky="nsew")
            if label == "Voice Chat":
                self.voice_chat_btn = btn

        self._section_header(bar, "🗓  Today's Overview")
        overview_wrap = ctk.CTkFrame(bar, fg_color="transparent")
        overview_wrap.pack(fill="x", padx=20, pady=(4, 16))
        self.overview_values = {
            "tasks": self._clickable_row(overview_wrap, "✅", "Tasks", self.show_tasks_popup),
            "reminders": self._clickable_row(overview_wrap, "🔔", "Reminders", self.show_reminders_popup),
            "messages": self._clickable_row(overview_wrap, "💬", "Messages", lambda: self._show_page("chat")),
            "system": self._clickable_row(overview_wrap, "⚙", "System", lambda: self._show_page("settings")),
        }

        self._section_header(bar, "🖥  System Info")
        info = ctk.CTkFrame(bar, fg_color="transparent")
        info.pack(fill="x", padx=20, pady=(6, 10))
        self.info_labels = {}
        for key, label in [("time", "Time"), ("date", "Date"), ("mood", "Mood"), ("uptime", "Uptime"), ("camera", "Camera")]:
            row = ctk.CTkFrame(info, fg_color="transparent")
            row.pack(fill="x", pady=3)
            ctk.CTkLabel(row, text=label, font=FONT_SMALL, text_color=TEXT_MUTED).pack(side="left")
            val = ctk.CTkLabel(row, text="—", font=("Segoe UI", 11, "bold"), text_color=TEXT_LIGHT)
            val.pack(side="right")
            self.info_labels[key] = val

        power_row = ctk.CTkFrame(bar, fg_color="transparent")
        power_row.pack(fill="x", padx=20, pady=(4, 12))
        ctk.CTkButton(power_row, text="⏻ Shutdown", width=80, height=28, font=FONT_TINY, fg_color="#3a0a0a",
                      hover_color="#5a1414", command=lambda: self._confirm_power("shutdown")).pack(side="left", padx=2)
        ctk.CTkButton(power_row, text="⟳ Restart", width=80, height=28, font=FONT_TINY, fg_color="#2a2a0a",
                      hover_color="#4a4a14", command=lambda: self._confirm_power("restart")).pack(side="left", padx=2)
        ctk.CTkButton(power_row, text="☾ Sleep", width=80, height=28, font=FONT_TINY, fg_color="#0a1a3a",
                      hover_color="#14285a", command=lambda: self._confirm_power("sleep")).pack(side="left", padx=2)

        ctk.CTkLabel(bar, text="").pack(expand=True, fill="both")
        self.right_quote_label = ctk.CTkLabel(
            bar, text="", font=("Georgia", 10, "italic"), text_color=TEXT_MUTED, wraplength=250, justify="center",
        )
        self.right_quote_label.pack(side="bottom", pady=18, padx=20)

    def _clickable_row(self, parent, icon, name, command):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=3)
        left = ctk.CTkLabel(row, text=f"{icon}  {name}", font=FONT_SMALL, text_color=TEXT_LIGHT, anchor="w")
        left.pack(side="left")
        val = ctk.CTkLabel(row, text="—", font=("Segoe UI", 11, "bold"), text_color=self.accent2, anchor="e")
        val.pack(side="right")
        for widget in (row, left, val):
            widget.bind("<Button-1>", lambda e, c=command: c())
        self._theme_appliers.append(lambda v=val: v.configure(text_color=self.accent2))
        return val

    # ---------------- quick actions ----------------
    def qa_ask_anything(self):
        self._show_page("chat")
        self.chat_entry.focus_set()

    def qa_create_task(self):
        dialog = ctk.CTkInputDialog(text="What's the task?", title="New Task")
        val = dialog.get_input()
        if val:
            self.tasks.append({"id": str(uuid.uuid4()), "text": val, "done": False, "created_at": datetime.datetime.now().isoformat()})
            save_json(TASKS_PATH, self.tasks)
            self.update_overview()
            self.append_message("system", f"✅ Task added: {val}")

    def qa_open_apps(self):
        self._show_open_apps_popup()

    def qa_toggle_voice(self):
        self.voice_listen_active = not self.voice_listen_active
        if self.voice_listen_active:
            self.voice_chat_btn.configure(fg_color=self.accent)
            threading.Thread(target=voice_loop, daemon=True).start()
        else:
            self.voice_chat_btn.configure(fg_color=PANEL_BG2)

    # ---------------- overview / system info ticking ----------------
    def update_overview(self):
        if not hasattr(self, "overview_values"):
            return
        pending = sum(1 for t in self.tasks if not t.get("done"))
        upcoming = sum(1 for r in self.reminders if not r.get("done"))
        today = datetime.date.today().isoformat()
        msgs_today = sum(1 for m in self.chat_history if m.get("ts", "").startswith(today))
        healthy = bool(GEMINI_API_KEY) and _PIPER_READY
        self.overview_values["tasks"].configure(text=f"{pending} pending")
        self.overview_values["reminders"].configure(text=f"{upcoming} upcoming")
        self.overview_values["messages"].configure(text=f"{msgs_today} today")
        self.overview_values["system"].configure(
            text=("All good" if healthy else "Check Settings"),
            text_color=("#33d17a" if healthy else "#ff9d4d"),
        )

    def _tick_system_info(self):
        now = datetime.datetime.now()
        self.info_labels["time"].configure(text=now.strftime("%I:%M %p"))
        self.info_labels["date"].configure(text=now.strftime("%b %d, %Y"))
        mood_map = {"idle": "Calm ♡", "listening": "Curious ✦", "speaking": "Chatty ♪", "executing": "Focused ⚡", "confirm": "Alert ⚠"}
        self.info_labels["mood"].configure(text=mood_map.get(self.current_mode, "Calm ♡"))
        elapsed = int(time.time() - self.start_time)
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        self.info_labels["uptime"].configure(text=f"{h:02d}:{m:02d}:{s:02d}")
        self.update_overview()
        self.after(1000, self._tick_system_info)

    def _rotate_quote(self):
        text = LUNA_QUOTES[self.quote_index % len(LUNA_QUOTES)]
        self.quote_index += 1
        if hasattr(self, "quote_label"):
            self.quote_label.configure(text=f"“{text}”")
        if hasattr(self, "right_quote_label"):
            self.right_quote_label.configure(text=f"“{text}”")
        self.after(9000, self._rotate_quote)

    # ---------------- popups / modals ----------------
    def show_tasks_popup(self):
        self._show_list_popup("Tasks", self.tasks, TASKS_PATH, "New task", "No tasks yet — add one!")

    def show_reminders_popup(self):
        self._show_list_popup("Reminders", self.reminders, REMINDERS_PATH, "New reminder", "No reminders yet — add one!")

    def _show_list_popup(self, title, items, path, add_prompt, empty_text):
        modal = ctk.CTkToplevel(self)
        modal.title(title)
        modal.geometry("380x440")
        modal.configure(fg_color=PANEL_BG)
        modal.transient(self)
        modal.update_idletasks()
        modal.deiconify()
        modal.wait_visibility()
        modal.grab_set()

        ctk.CTkLabel(modal, text=title, font=FONT_H1, text_color=TEXT_LIGHT).pack(pady=(16, 8))
        scroll = ctk.CTkScrollableFrame(modal, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=14, pady=(0, 8))

        def refresh():
            for w in scroll.winfo_children():
                w.destroy()
            if not items:
                ctk.CTkLabel(scroll, text=empty_text, text_color=TEXT_MUTED).pack(pady=10)
                return
            for it in list(items):
                row = ctk.CTkFrame(scroll, fg_color=PANEL_BG2, corner_radius=10)
                row.pack(fill="x", pady=4)
                done = it.get("done", False)
                cb = ctk.CTkCheckBox(
                    row, text=it["text"], font=FONT_SMALL, text_color=(TEXT_MUTED if done else TEXT_LIGHT),
                    onvalue=True, offvalue=False, command=lambda i=it: toggle(i),
                )
                if done:
                    cb.select()
                cb.pack(side="left", padx=8, pady=8, fill="x", expand=True)
                ctk.CTkButton(
                    row, text="✕", width=28, height=28, fg_color="transparent", hover_color="#5a1414",
                    text_color="#ff6b6b", command=lambda i=it: remove(i),
                ).pack(side="right", padx=6)

        def toggle(it):
            it["done"] = not it.get("done", False)
            save_json(path, items)
            refresh()
            self.update_overview()

        def remove(it):
            items.remove(it)
            save_json(path, items)
            refresh()
            self.update_overview()

        def add_new():
            dialog = ctk.CTkInputDialog(text=add_prompt, title=add_prompt)
            val = dialog.get_input()
            if val:
                items.append({"id": str(uuid.uuid4()), "text": val, "done": False, "created_at": datetime.datetime.now().isoformat()})
                save_json(path, items)
                refresh()
                self.update_overview()

        refresh()
        ctk.CTkButton(modal, text="+ Add", fg_color=self.accent, hover_color=self.accent2, command=add_new).pack(pady=(0, 16))

    def _show_open_apps_popup(self):
        modal = ctk.CTkToplevel(self)
        modal.title("Open App")
        modal.geometry("320x300")
        modal.configure(fg_color=PANEL_BG)
        modal.transient(self)
        modal.update_idletasks()
        modal.deiconify()
        modal.wait_visibility()
        modal.grab_set()
        ctk.CTkLabel(modal, text="Open an App", font=FONT_H1, text_color=TEXT_LIGHT).pack(pady=(16, 10))
        apps = [
            ("🌐", "Browser", "browser"), ("📁", "Files", "files"), ("⌨", "Terminal", "terminal"),
            ("🧮", "Calculator", "calculator"), ("📝", "Notepad", "notepad"),
        ]
        for icon, label, kind in apps:
            ctk.CTkButton(
                modal, text=f"{icon}  {label}", anchor="w", font=FONT_BODY, fg_color=PANEL_BG2,
                hover_color=PANEL_BORDER, text_color=TEXT_LIGHT, corner_radius=10, height=38,
                command=lambda k=kind, m=modal: (_launch_app(k), m.destroy()),
            ).pack(fill="x", padx=20, pady=4)

    def _confirm_power(self, action):
        modal = ctk.CTkToplevel(self)
        modal.title("Confirm")
        modal.geometry("340x160")
        modal.configure(fg_color=PANEL_BG)
        modal.transient(self)
        modal.update_idletasks()
        modal.deiconify()
        modal.wait_visibility()
        modal.grab_set()
        verbs = {"shutdown": "shut down", "restart": "restart", "sleep": "put to sleep"}
        ctk.CTkLabel(
            modal, text=f"Are you sure you want to {verbs[action]} the system?",
            font=FONT_BODY, text_color=TEXT_LIGHT, wraplength=280, justify="center",
        ).pack(pady=(24, 16), padx=16)
        row = ctk.CTkFrame(modal, fg_color="transparent")
        row.pack()

        def confirm():
            modal.destroy()
            threading.Thread(target=_execute_system_action, args=(action,), daemon=True).start()

        ctk.CTkButton(row, text="Yes, proceed", fg_color="#a83232", hover_color="#c94040", command=confirm).pack(side="left", padx=8)
        ctk.CTkButton(row, text="Cancel", fg_color=PANEL_BG2, hover_color=PANEL_BORDER, command=modal.destroy).pack(side="left", padx=8)

    def on_close(self):
        global _app_running
        _app_running = False
        _stop_speech_flag.set()
        speech_done_event.set()
        self.voice_listen_active = False
        logging.info("Luna window closed cleanly.")
        self.destroy()


# ------------------------------------------------------------
# Speak / listen
# ------------------------------------------------------------
def mute_luna():
    _stop_speech_flag.set()
    if app:
        app.set_mode("idle")
        app.set_status("✦ Idle ✦")


def speak(text: str):
    """Speak without blocking the UI. Completion is signalled via
    speech_done_event and awaited from the calling thread, not the main
    thread, so the UI stays responsive while speech plays."""
    speech_done_event.clear()
    if app:
        app.set_mode("speaking")
        app.set_status("✦ Speaking ✦")
        app.append_message("luna", text)

    def run_speech_task():
        _stop_speech_flag.clear()
        wav_path = None
        proc = None
        try:
            if not (_PIPER_READY and _AUDIO_PLAYERS):
                logging.error("TTS unavailable — Piper voice or audio player missing.")
                return
            if _stop_speech_flag.is_set():
                return
            fd, wav_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            with wave.open(wav_path, "wb") as wav_file:
                _piper_voice.synthesize_wav(text, wav_file)
            if _stop_speech_flag.is_set():
                return
            player = _AUDIO_PLAYERS[0]
            cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", wav_path] if player == "ffplay" else [player, wav_path]
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            while proc.poll() is None:
                if _stop_speech_flag.is_set():
                    proc.terminate()
                    break
                time.sleep(0.05)
        except Exception as e:
            logging.error(f"TTS error: {e}")
        finally:
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass
            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except Exception:
                    pass
            speech_done_event.set()

    threading.Thread(target=run_speech_task, daemon=True).start()
    speech_done_event.wait()
    if app:
        app.set_mode("idle")
        app.set_status("✦ Idle ✦")


def listen(timeout: int = 6, phrase_limit: int = 12) -> str:
    if app:
        app.set_mode("listening")
        app.set_status("✦ Listening ✦")

    recognizer = sr.Recognizer()
    try:
        mic = sr.Microphone()
    except (OSError, AttributeError) as e:
        logging.error(f"Microphone unavailable: {e}")
        if app:
            app.append_message("error", "Microphone not found — use the text input below.")
            app.set_mode("idle")
            app.set_status("✦ Idle ✦")
        return ""

    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_limit)
        except sr.WaitTimeoutError:
            if app:
                app.set_mode("idle")
                app.set_status("✦ Idle ✦")
            return ""

    if app:
        app.set_mode("executing")
        app.set_status("✦ Processing ✦")

    try:
        command = recognizer.recognize_google(audio)
        logging.info(f"Voice command: {command}")
        if app:
            app.append_message("user", command)
        return command.lower()
    except sr.UnknownValueError:
        speak("Sorry, I didn't catch that.")
        return ""
    except sr.RequestError as e:
        logging.error(f"Speech recognition API error: {e}")
        speak("Network error. Please try again.")
        return ""


def voice_loop():
    """Background continuous listening loop, toggled by the Voice Chat quick action."""
    if app:
        app.append_message("system", "🎙 Voice chat activated — listening…")
    while app and app.voice_listen_active and _app_running:
        command = listen()
        if not _app_running or not (app and app.voice_listen_active):
            break
        if command:
            if not process_command(command):
                if app:
                    app.voice_listen_active = False
                    app.voice_chat_btn.configure(fg_color=PANEL_BG2)
                break
    if app:
        app.set_mode("idle")
        app.set_status("✦ Idle ✦")


# ------------------------------------------------------------
# Info / utility commands
# ------------------------------------------------------------
def check_online_status() -> bool:
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=5)
        return True
    except OSError:
        return False


def tell_online_status():
    speak("Yes, you are online." if check_online_status() else "No, you are offline.")


def get_time():
    speak(f"The current time is {datetime.datetime.now().strftime('%I:%M %p')}.")


def get_date():
    speak(f"Today's date is {datetime.date.today().strftime('%A, %B %d, %Y')}.")


def tell_joke():
    speak(f"Here's a joke: {pyjokes.get_joke()}")


def get_weather(city: str):
    if app:
        app.set_mode("executing")
        app.set_status("✦ Executing ✦")
    if not WEATHER_API_KEY:
        speak("Weather API key is not configured. Please add it on the Settings page.")
        return
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    try:
        resp = requests.get(url, timeout=10).json()
        if resp.get("main"):
            temp = resp["main"]["temp"]
            desc = resp["weather"][0]["description"]
            speak(f"The current temperature in {city} is {temp}°C with {desc}.")
        elif resp.get("message"):
            speak(f"Sorry, I couldn't get the weather: {resp['message']}")
        else:
            speak("Weather information is unavailable right now.")
    except requests.exceptions.RequestException as e:
        logging.error(f"Weather API error: {e}")
        speak("A network error occurred while fetching the weather.")


def get_location():
    if app:
        app.set_mode("executing")
        app.set_status("✦ Executing ✦")
    speak(f"Based on your configuration, you are located in {HOME_CITY}.")


def play_video_auto(video_name: str) -> bool:
    if not video_name:
        speak("Please specify the video name.")
        return False
    if check_online_status():
        try:
            import yt_dlp
            speak(f"Searching for {video_name} online.")
            ydl_opts = {"quiet": True, "no_warnings": True, "extract_flat": True, "default_search": "ytsearch1", "noplaylist": True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"ytsearch1:{video_name}", download=False)
                entries = (info or {}).get("entries", [])
                if entries:
                    vid_id = entries[0].get("id")
                    if vid_id:
                        webbrowser.open(f"https://www.youtube.com/watch?v={vid_id}")
                        speak(f"Playing {video_name} on YouTube.")
                        return True
            speak("Could not find the video online. Checking your local library.")
        except Exception as e:
            logging.error(f"Online video error: {e}")
            speak("Online search failed. Checking your local library.")
    try:
        video_dir = os.path.join(os.path.expanduser("~"), "Videos")
        for root, _, files in os.walk(video_dir):
            for f in files:
                if video_name.lower() in f.lower() and f.endswith((".mp4", ".avi", ".mkv", ".mp3", ".wav", ".flac")):
                    path = os.path.join(root, f)
                    sys_p = platform.system().lower()
                    if sys_p == "windows":
                        os.startfile(path)
                    else:
                        subprocess.run(["xdg-open" if sys_p == "linux" else "open", path])
                    speak(f"Playing {f} from your local library.")
                    return True
        speak("I couldn't find that video in your local library either.")
        return False
    except Exception as e:
        logging.error(f"Offline video error: {e}")
        speak("An error occurred while searching your local library.")
        return False


def play_video(command: str):
    if app:
        app.set_mode("executing")
        app.set_status("✦ Executing ✦")
    if "stop" in command:
        speak("I can't stop playback once it's opened in your browser or player — please close it there.")
        return
    video_name = command.replace("play", "").strip()
    if video_name:
        play_video_auto(video_name)
    else:
        speak("Please specify the video name.")


def perform_web_browsing(command: str):
    """Handles 'open <site>' and 'search <query>'. General questions are
    routed to ask_gemini() instead — see process_command()."""
    if app:
        app.set_mode("executing")
        app.set_status("✦ Executing ✦")
    command = command.lower().strip()
    if command.startswith("open"):
        site = command.replace("open", "").strip()
        if "youtube" in site:
            speak("Opening YouTube.")
            webbrowser.open("https://www.youtube.com")
        elif "google" in site:
            speak("Opening Google.")
            webbrowser.open("https://www.google.com")
        elif "." in site:
            speak(f"Opening {site}.")
            webbrowser.open(f"https://{site}")
        else:
            speak(f"Opening {site}.")
            webbrowser.open(f"https://www.{site}.com")
    elif command.startswith("search"):
        query = command.replace("search", "").replace("for", "").strip()
        if not query:
            speak("Please tell me what to search for.")
            return
        speak(f"Searching for {query}.")
        webbrowser.open(f"https://google.com/search?q={query.replace(' ', '+')}")
    else:
        speak("Sorry, I didn't understand that browsing command.")


def _launch_app(kind: str):
    sys_p = platform.system().lower()
    try:
        if kind == "browser":
            webbrowser.open("https://www.google.com")
        elif kind == "files":
            home = os.path.expanduser("~")
            if sys_p == "windows":
                os.startfile(home)
            elif sys_p == "darwin":
                subprocess.Popen(["open", home])
            else:
                subprocess.Popen(["xdg-open", home])
        elif kind == "terminal":
            if sys_p == "windows":
                subprocess.Popen(["cmd.exe"])
            elif sys_p == "darwin":
                subprocess.Popen(["open", "-a", "Terminal"])
            else:
                subprocess.Popen(["x-terminal-emulator"])
        elif kind == "calculator":
            if sys_p == "windows":
                subprocess.Popen(["calc.exe"])
            elif sys_p == "darwin":
                subprocess.Popen(["open", "-a", "Calculator"])
            else:
                subprocess.Popen(["gnome-calculator"])
        elif kind == "notepad":
            if sys_p == "windows":
                subprocess.Popen(["notepad.exe"])
            elif sys_p == "darwin":
                subprocess.Popen(["open", "-a", "TextEdit"])
            else:
                subprocess.Popen(["gedit"])
        if app:
            app.append_message("system", f"🗂 Opened {kind}.")
    except Exception as e:
        logging.error(f"Failed to launch {kind}: {e}")
        if app:
            app.append_message("error", f"Couldn't open {kind} — it may not be installed.")


# ------------------------------------------------------------
# System control (shutdown / restart / sleep)
# ------------------------------------------------------------
def _execute_system_action(action: str):
    sys_p = platform.system().lower()
    if action == "shutdown":
        speak("Shutting down the system now.")
        subprocess.run(["shutdown", "/s", "/f", "/t", "5"] if sys_p == "windows" else ["shutdown", "-h", "now"])
    elif action == "restart":
        speak("Restarting the system now.")
        subprocess.run(["shutdown", "/r", "/f", "/t", "5"] if sys_p == "windows" else ["reboot"])
    elif action == "sleep":
        speak("Putting the system to sleep.")
        try:
            if sys_p == "windows":
                subprocess.run(["rundll32", "powrprof.dll,SetSuspendState", "Sleep"], check=True)
            elif sys_p == "linux":
                try:
                    subprocess.run(["systemctl", "suspend"], check=True)
                except subprocess.CalledProcessError:
                    subprocess.run(["pm-suspend"], check=True)
            elif sys_p == "darwin":
                subprocess.run(["pmset", "sleepnow"])
        except subprocess.CalledProcessError:
            speak("Could not execute sleep command. Check system privileges.")


def perform_system_action(command: str):
    """Requires spoken confirmation before shutdown/restart/sleep executes
    (used for voice commands — GUI power buttons confirm via a modal instead)."""
    global _pending_system_action
    if app:
        app.set_mode("confirm")
        app.set_status("✦ Confirm? ✦")
    if "shutdown" in command:
        _pending_system_action = "shutdown"
        speak("Are you sure you want to shut down? Say 'yes' to confirm or 'cancel' to abort.")
    elif "restart" in command:
        _pending_system_action = "restart"
        speak("Are you sure you want to restart? Say 'yes' to confirm or 'cancel' to abort.")
    elif "sleep" in command or "hibernate" in command:
        _pending_system_action = "sleep"
        speak("Are you sure you want to sleep the system? Say 'yes' to confirm or 'cancel' to abort.")
    else:
        _pending_system_action = None
        speak("I don't recognise that system command.")


def handle_confirmation(response: str) -> bool:
    global _pending_system_action
    if _pending_system_action is None:
        return False
    if "yes" in response or "confirm" in response or "do it" in response:
        action = _pending_system_action
        _pending_system_action = None
        _execute_system_action(action)
        return True
    elif "no" in response or "cancel" in response or "abort" in response:
        _pending_system_action = None
        speak("System action cancelled.")
        return True
    return False


# ------------------------------------------------------------
# Gemini (general Q&A) — uses saved Memories as extra context
# ------------------------------------------------------------
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_SYSTEM_PROMPT = (
    "You are Luna, a friendly and concise voice assistant. "
    "Reply in 1-3 short sentences suitable for text-to-speech. "
    "No markdown, no lists, no special characters."
)


def ask_gemini(question: str) -> str:
    """General-purpose fallback for anything not covered by a specific
    command, including 'what is/who is/tell about' questions."""
    if not GEMINI_API_KEY:
        return "I'm not sure how to help with that yet. You can add a Gemini API key on the Settings page."
    system_prompt = GEMINI_SYSTEM_PROMPT
    if app and app.memories:
        mem_texts = [m["text"] for m in app.memories[-20:]]
        system_prompt += "\n\nKnown context about the user (from memory):\n- " + "\n- ".join(mem_texts)
    try:
        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
            params={"key": GEMINI_API_KEY},
            headers={"content-type": "application/json"},
            json={
                "contents": [{"role": "user", "parts": [{"text": question}]}],
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "generationConfig": {"maxOutputTokens": 256},
            },
            timeout=15,
        )
        data = resp.json()
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            for part in parts:
                if "text" in part:
                    return part["text"].strip()
        if data.get("error"):
            logging.error(f"Gemini API error: {data['error']}")
            if data["error"].get("code") == 429:
                return "I've hit my free-tier request limit for now — please try again in a minute."
        return "I couldn't generate a response right now."
    except Exception as e:
        logging.error(f"Gemini API error: {e}")
        return "I had trouble reaching the AI service. Please check your connection."


# ------------------------------------------------------------
# Face detection + recognition (background camera thread)
# ------------------------------------------------------------
def _face_camera_loop():
    """Opens the webcam once and holds it for the app's lifetime. Every
    frame is stashed in _latest_frame under _face_cam_lock so command
    handlers can read it via _get_shared_frame() without opening a second,
    competing VideoCapture."""
    global _last_greeted_name, _latest_frame

    if not _FACE_MODELS_READY:
        if app:
            app.set_face_status("models unavailable")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        if app:
            app.set_face_status("no webcam found")
        logging.warning("Face camera: no webcam found.")
        return

    if app:
        app.set_face_status("active — watching…")
    logging.info("Face camera loop started.")

    RECOGNITION_EVERY_N = 10
    frame_count = 0

    while _app_running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame_count += 1
        with _face_cam_lock:
            _latest_frame = frame.copy()

        face_locations = _detect_face_boxes(frame)
        face_count = len(face_locations)

        if face_count == 0:
            if app:
                app.set_face_status("no face detected")
            _last_greeted_name = None
        else:
            if app:
                app.set_face_status(f"{face_count} face(s) detected")
            if frame_count % RECOGNITION_EVERY_N == 0:
                for encoding in _get_face_embeddings(frame):
                    _handle_face_encoding(encoding)

        time.sleep(0.03)

    cap.release()
    with _face_cam_lock:
        _latest_frame = None
    if app:
        app.set_face_status("off")
    logging.info("Face camera loop stopped.")


def _handle_face_encoding(encoding):
    global _last_greeted_name, _last_greeted_time

    now = time.time()
    name = identify_face(encoding)

    if name is not None:
        if name != _last_greeted_name or (now - _last_greeted_time) > GREET_COOLDOWN_SECONDS:
            _last_greeted_name = name
            _last_greeted_time = now
            greeting = f"Hello {name}! Welcome back. How can I help you today?"
            if app:
                app.append_message("system", f"👁 Recognised: {name}")
            threading.Thread(target=speak, args=(greeting,), daemon=True).start()
    elif not _awaiting_face_name.is_set():
        _awaiting_face_name.set()
        if app:
            app.append_message("system", "👁 Unknown face detected — asking for name…")
        threading.Thread(target=_ask_unknown_face_name, args=(encoding,), daemon=True).start()


def _strip_name_fillers(text: str) -> str:
    for filler in ["my name is", "i am", "it's", "its", "call me", "i'm"]:
        text = text.replace(filler, "")
    return text.strip().title()


def _ask_unknown_face_name(encoding):
    try:
        speak("Hello! I don't recognise you yet. What is your name?")
        name_said = listen(timeout=8, phrase_limit=5)
        if name_said and name_said not in ("", "cancel", "nothing", "nevermind"):
            name_said = _strip_name_fillers(name_said)
            if name_said:
                register_face(name_said, encoding)
                global _last_greeted_name, _last_greeted_time
                _last_greeted_name = name_said
                _last_greeted_time = time.time()
                if app:
                    app.append_message("system", f"👁 New face registered: {name_said}")
                speak(f"Nice to meet you, {name_said}! I'll remember you from now on.")
            else:
                speak("I didn't catch a name. No worries, you can tell me later.")
        else:
            speak("No problem, feel free to tell me your name any time.")
    finally:
        _awaiting_face_name.clear()


def start_face_recognition():
    if not _FACE_MODELS_READY:
        logging.warning("Face recognition skipped — libraries not available.")
        if app:
            app.set_face_status("models unavailable")
        return
    t = threading.Thread(target=_face_camera_loop, daemon=True)
    t.start()
    return t


def _handle_who_am_i():
    if not _FACE_MODELS_READY:
        speak("Face recognition models are not available (check opencv-python is installed and models downloaded successfully).")
        return
    speak("Let me take a look at you.")
    frame = _get_shared_frame()
    if frame is None:
        speak("I can't access the camera right now. Please make sure the webcam is connected.")
        return
    identified = None
    encs = _get_face_embeddings(frame)
    if encs:
        identified = identify_face(encs[0])
    if identified:
        speak(f"You are {identified}! Great to see you.")
    else:
        speak("I don't recognise you yet. Would you like to tell me your name so I can remember you?")


def _handle_learn_face_command():
    if not _FACE_MODELS_READY:
        speak("Face recognition models are not available (check opencv-python is installed and models downloaded successfully).")
        return
    speak("Sure! Please look at the camera. What is your name?")
    name_said = listen(timeout=8, phrase_limit=5)
    if not name_said:
        speak("I didn't hear a name. Please try again.")
        return
    name_said = _strip_name_fillers(name_said)
    if not name_said:
        speak("I couldn't make out a name. Please try again.")
        return
    frame = _get_shared_frame()
    if frame is None:
        speak("I can't access the camera right now. Please make sure the webcam is connected.")
        return
    encoding = None
    encs = _get_face_embeddings(frame)
    if encs:
        encoding = encs[0]
    if encoding is not None:
        register_face(name_said, encoding)
        if app:
            app.append_message("system", f"👁 Face saved: {name_said}")
        speak(f"Done! I've saved your face as {name_said}. I'll recognise you next time.")
    else:
        speak("I couldn't detect a face. Please make sure you're in front of the camera and well-lit.")


def _handle_forget_face_command():
    speak("Whose face should I forget? Please say the name.")
    name_said = listen(timeout=8, phrase_limit=5)
    if not name_said:
        speak("I didn't hear a name.")
        return
    name_said = name_said.title().strip()
    if name_said in _known_face_names:
        idx = _known_face_names.index(name_said)
        _known_face_names.pop(idx)
        _known_face_encodings.pop(idx)
        _save_face_db()
        if app:
            app.append_message("system", f"👁 Face removed: {name_said}")
        speak(f"I've forgotten {name_said}'s face.")
    else:
        speak(f"I don't have a face saved for {name_said}.")


# ------------------------------------------------------------
# Command processing
# ------------------------------------------------------------
def process_command(command: str) -> bool:
    """Handles a text or voice command. Returns False if the assistant
    should exit."""
    if not command:
        return True

    if _pending_system_action is not None:
        if handle_confirmation(command):
            return True

    if "who are you" in command or "your name" in command:
        speak(
            "I'm Luna, your mystical virtual assistant! "
            "I can tell you the time, date, weather, play videos, browse the web, "
            "recognise faces, remember things about you, and answer almost any question "
            "thanks to my AI brain. How can I help you today?"
        )
    elif "who am i" in command or "do you know me" in command or "recognise me" in command:
        _handle_who_am_i()
    elif "remember my face" in command or "learn my face" in command:
        _handle_learn_face_command()
    elif "forget me" in command or "forget my face" in command:
        _handle_forget_face_command()
    elif "time" in command:
        get_time()
    elif "date" in command:
        get_date()
    elif "joke" in command:
        tell_joke()
    elif "weather" in command:
        speak("Which city would you like the weather for?")
        city = listen()
        if city:
            get_weather(city)
    elif "location" in command:
        speak("Let me check your configured location.")
        get_location()
    elif "am i online" in command or "internet" in command:
        tell_online_status()
    elif "play" in command:
        play_video(command)
    elif any(w in command for w in ["shutdown", "restart", "sleep", "hibernate"]):
        perform_system_action(command)
    elif command.startswith(("open", "search")):
        perform_web_browsing(command)
    elif any(w in command for w in ["exit", "bye", "goodbye", "see you"]):
        speak("Goodbye! May the stars guide your path.")
        return False
    else:
        # Covers "what is/who is/tell about" plus anything else — the LLM
        # answers these better than a canned web-search summary would.
        if app:
            app.set_mode("executing")
            app.set_status("✦ Thinking ✦")
        speak(ask_gemini(command))

    return True


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    app = LunaApp()
    app.append_message(
        "luna",
        f"Hello {USER_NAME}! I'm Luna, your mystical assistant. Type below or tap the "
        "mic whenever you're ready — or turn on Voice Chat for hands-free conversation.",
    )
    start_face_recognition()
    try:
        app.mainloop()
    except Exception as e:
        logging.critical(f"Main window loop crashed: {e}")