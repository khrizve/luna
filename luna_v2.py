"""Luna — mystical AI desktop assistant.

Fantasy-themed three-pane dashboard (PyQt6) on top of the original
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
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import wave
import webbrowser
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pyjokes
import requests
import speech_recognition as sr
from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPalette, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog, QFrame, QGridLayout, QHBoxLayout,
    QInputDialog, QLabel, QLineEdit, QMainWindow, QPlainTextEdit, QPushButton, QScrollArea,
    QStackedWidget, QVBoxLayout, QWidget,
)
from piper import PiperVoice
from piper.download_voices import download_voice

try:
    import cv2
    # opencv-python ships its own Qt plugins and points Qt at them, which breaks
    # PyQt6 on Linux ("could not load the Qt platform plugin xcb"). Undo that.
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

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
# GUI — fantasy three-pane dashboard (PyQt6)
# ------------------------------------------------------------
UI_FONT_FAMILIES = ["Segoe UI", "Noto Sans", "DejaVu Sans", "Arial"]
BODY_PX = 13
BUBBLE_MAX_WIDTH = 420
CHAT_VIEW_LIMIT = 500

NAV_ITEMS = [
    ("home", "🏠  Home"),
    ("chat", "💬  Chat"),
    ("voice", "🎙  Voice"),
    ("memories", "🧠  Memories"),
    ("settings", "⚙  Settings"),
]
PORTRAIT_FILES = ("luna_portrait.png", "luna_portrait.jpg", "miss_luna.jpeg", "miss_luna.jpg", "miss_luna.png")
BANNER_FILES = ("luna_banner.png", "luna_banner.jpg") + PORTRAIT_FILES


def build_stylesheet(accent: str, accent2: str) -> str:
    """Application-wide stylesheet. Rebuilt whenever the theme changes, which
    restyles every widget that picks up an accent color."""
    return f"""
    QWidget {{ color: {TEXT_LIGHT}; font-size: {BODY_PX}px; }}
    QMainWindow, #center {{ background: {BG_DARK}; }}
    QDialog {{ background: {PANEL_BG}; }}
    QLabel {{ background: transparent; }}

    QScrollArea {{ background: transparent; border: none; }}
    #transparent, QScrollArea > QWidget > QWidget {{ background: transparent; }}
    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
    QScrollBar::handle:vertical {{ background: {PANEL_BORDER}; border-radius: 4px; min-height: 30px; }}
    QScrollBar::handle:vertical:hover {{ background: {accent}; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}

    #sidebar {{ background: {PANEL_BG}; }}
    #divider {{ background: {PANEL_BORDER}; border: none; min-height: 1px; max-height: 1px; }}
    #hero {{ background: {PANEL_BG}; border: 2px solid {PANEL_BORDER}; border-radius: 22px; }}
    #panel {{ background: {PANEL_BG}; border: 1px solid {PANEL_BORDER}; border-radius: 16px; }}
    #inner {{ background: {PANEL_BG2}; border-radius: 12px; }}
    #previewPanel {{ background: {PANEL_BG2}; border-radius: 16px; }}
    #inputBar {{ background: {PANEL_BG}; border: 1px solid {PANEL_BORDER}; border-radius: 20px; }}

    #brand {{ font-family: Georgia; font-size: 24px; font-weight: bold; color: {accent}; }}
    #brandSig {{ font-family: Georgia; font-size: 11px; font-style: italic; color: {accent}; }}
    #h1 {{ font-family: Georgia; font-size: 20px; font-weight: bold; }}
    #h2 {{ font-size: 13px; font-weight: bold; color: {TEXT_MUTED}; }}
    #greeting {{ font-family: Georgia; font-size: 22px; font-weight: bold; }}
    #profileName {{ font-size: 16px; font-weight: bold; }}
    #muted {{ color: {TEXT_MUTED}; font-size: 12px; }}
    #tiny {{ color: {TEXT_MUTED}; font-size: 10px; }}
    #quote {{ font-family: Georgia; font-size: 11px; font-style: italic; color: {TEXT_MUTED}; }}
    #sectionTitle {{ color: {accent}; font-size: 13px; font-weight: bold; }}
    #overviewValue {{ color: {accent2}; font-size: 11px; font-weight: bold; }}
    #infoValue {{ font-size: 11px; font-weight: bold; }}

    #bubbleUser, #bubbleLuna, #bubbleSystem, #bubbleError {{ border-radius: 14px; padding: 8px 12px; }}
    #bubbleUser {{ background: {accent}; color: #ffffff; }}
    #bubbleLuna {{ background: {PANEL_BG2}; }}
    #bubbleSystem {{ background: {PANEL_BG2}; color: #ffd76a; }}
    #bubbleError {{ background: {PANEL_BG2}; color: #ff6b6b; }}

    QPushButton {{ background: {PANEL_BG2}; border: none; border-radius: 10px; padding: 8px 16px; }}
    QPushButton:hover {{ background: {PANEL_BORDER}; }}
    QPushButton#primary {{ background: {accent}; color: #ffffff; }}
    QPushButton#primary:hover {{ background: {accent2}; }}
    QPushButton#danger {{ background: #a83232; color: #ffffff; }}
    QPushButton#danger:hover {{ background: #c94040; }}
    QPushButton#nav {{ background: transparent; text-align: left; padding: 11px 14px; }}
    QPushButton#nav:hover {{ background: {PANEL_BORDER}; }}
    QPushButton#nav:checked {{ background: {accent}; color: #ffffff; }}
    QPushButton#chip {{ border-radius: 16px; padding: 6px 14px; font-size: 12px; }}
    QPushButton#qa {{ border-radius: 14px; padding: 8px; font-size: 12px; }}
    QPushButton#qa:checked {{ background: {accent}; color: #ffffff; }}
    QPushButton#appBtn {{ text-align: left; padding: 9px 14px; }}
    QPushButton#roundBtn {{ border-radius: 21px; padding: 0; font-size: 15px; }}
    QPushButton#roundBtnSmall {{ border-radius: 18px; padding: 0; font-size: 14px; }}
    QPushButton#roundBtnAccent {{ background: {accent}; color: #ffffff; border-radius: 21px; padding: 0; font-size: 15px; }}
    QPushButton#roundBtnAccent:hover {{ background: {accent2}; }}
    QPushButton#link {{ background: transparent; color: {accent2}; font-size: 11px; padding: 3px 10px; border-radius: 6px; }}
    QPushButton#link:hover {{ background: {PANEL_BORDER}; }}
    QPushButton#dangerLink {{ background: transparent; color: #ff6b6b; font-size: 11px; padding: 3px 10px; border-radius: 6px; }}
    QPushButton#dangerLink:hover {{ background: #5a1414; }}
    QPushButton#powerShutdown {{ background: #3a0a0a; font-size: 11px; padding: 6px 4px; border-radius: 8px; }}
    QPushButton#powerShutdown:hover {{ background: #5a1414; }}
    QPushButton#powerRestart {{ background: #2a2a0a; font-size: 11px; padding: 6px 4px; border-radius: 8px; }}
    QPushButton#powerRestart:hover {{ background: #4a4a14; }}
    QPushButton#powerSleep {{ background: #0a1a3a; font-size: 11px; padding: 6px 4px; border-radius: 8px; }}
    QPushButton#powerSleep:hover {{ background: #14285a; }}

    QLineEdit, QPlainTextEdit {{ background: {PANEL_BG2}; border: none; border-radius: 10px; padding: 8px 10px;
        selection-background-color: {accent}; }}
    QLineEdit#chatEntry {{ background: transparent; padding: 0 8px; }}
    QComboBox {{ background: {PANEL_BG2}; border: none; border-radius: 8px; padding: 6px 12px; min-width: 220px; }}
    QComboBox QAbstractItemView {{ background: {PANEL_BG2}; border: 1px solid {PANEL_BORDER};
        selection-background-color: {accent}; outline: 0; }}
    QCheckBox {{ spacing: 10px; }}
    QCheckBox::indicator {{ width: 18px; height: 18px; border-radius: 5px; border: 2px solid {PANEL_BORDER};
        background: {PANEL_BG}; }}
    QCheckBox::indicator:checked {{ background: {accent}; border-color: {accent}; }}
    """


def _blend_hex(hex_a: str, hex_b: str, t: float) -> str:
    """Blends hex_a into hex_b by fraction t (0-1) — used to fake a
    semi-transparent, theme-colored border."""
    a = tuple(int(hex_a[i:i + 2], 16) for i in (1, 3, 5))
    b = tuple(int(hex_b[i:i + 2], 16) for i in (1, 3, 5))
    blended = tuple(int(a[i] * t + b[i] * (1 - t)) for i in range(3))
    return "#{:02x}{:02x}{:02x}".format(*blended)


def _discard(widget: QWidget) -> None:
    """Removes a widget right away (hidden + unparented) and frees it once Qt is idle."""
    widget.hide()
    widget.setParent(None)
    widget.deleteLater()


def _clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            _discard(widget)
        elif item.layout() is not None:
            _clear_layout(item.layout())


def _load_pixmap(names):
    """First existing image from ./assets matching one of `names`, or None."""
    path = next((p for p in (os.path.join("assets", n) for n in names) if os.path.exists(p)), None)
    if path is None:
        return None
    pixmap = QPixmap(path)
    return None if pixmap.isNull() else pixmap


def _draw_cover(p: QPainter, target: QRectF, pixmap: QPixmap) -> None:
    """Draws `pixmap` scaled to fill `target`, cropping the overflow evenly."""
    iw, ih = pixmap.width(), pixmap.height()
    scale = max(target.width() / iw, target.height() / ih)
    sw, sh = target.width() / scale, target.height() / scale
    p.drawPixmap(target, pixmap, QRectF((iw - sw) / 2, (ih - sh) / 2, sw, sh))


def _paint_banner_placeholder(p: QPainter, rect: QRectF) -> None:
    p.fillRect(rect, QColor(18, 14, 42))
    cx, cy = rect.x() + rect.width() * 0.24, rect.y() + rect.height() * 0.5
    r = rect.height() * 0.32
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(95, 55, 160))
    p.drawEllipse(QPointF(cx, cy), r, r)
    p.setBrush(QColor(18, 14, 42))
    p.drawEllipse(QRectF(cx - r + 18, cy - r - 10, 2 * r - 52, 2 * r + 2))
    rng = random.Random(42)  # fixed seed so the star field doesn't jump around on repaint
    p.setBrush(QColor(210, 200, 255, 160))
    for _ in range(18):
        x = rect.x() + rng.uniform(rect.width() * 0.4, rect.width() * 0.98)
        y = rect.y() + rng.uniform(rect.height() * 0.06, rect.height() * 0.94)
        s = rng.uniform(1.2, 2.6)
        p.drawEllipse(QPointF(x, y), s, s)


def make_avatar_pixmap(size: int) -> QPixmap:
    """Circular portrait (or a painted placeholder), rendered at 2x for crispness."""
    out = QPixmap(size * 2, size * 2)
    out.setDevicePixelRatio(2)
    out.fill(Qt.GlobalColor.transparent)
    p = QPainter(out)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    clip = QPainterPath()
    clip.addEllipse(0, 0, size, size)
    p.setClipPath(clip)
    target = QRectF(0, 0, size, size)
    portrait = _load_pixmap(PORTRAIT_FILES)
    if portrait is not None:
        _draw_cover(p, target, portrait)
    else:
        p.fillRect(target, QColor(20, 12, 46))
        c, r = size / 2, size * 0.32
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(95, 55, 160))
        p.drawEllipse(QPointF(c, c), r, r)
        p.setBrush(QColor(20, 12, 46))
        p.drawEllipse(QRectF(c - r + 10, c - r - 6, 2 * r - 32, 2 * r + 2))
    p.end()
    return out


class ClickableRow(QWidget):
    """Overview row: icon + name on the left, a value on the right; emits clicked."""
    clicked = pyqtSignal()

    def __init__(self, icon: str, name: str, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 3, 0, 3)
        lay.addWidget(QLabel(f"{icon}  {name}"))
        lay.addStretch(1)
        self.value = QLabel("—")
        self.value.setObjectName("overviewValue")
        lay.addWidget(self.value)

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)


class BannerWidget(QWidget):
    """Home banner: portrait (or painted placeholder) in a rounded, theme-colored
    frame, with the assistant's mode indicator dot in the corner."""

    def __init__(self, accent: str, parent=None):
        super().__init__(parent)
        self.setFixedHeight(220)
        self._accent = accent
        self._dot = QColor(mode_colors["idle"])
        self._image = _load_pixmap(BANNER_FILES)

    def set_accent(self, accent: str) -> None:
        self._accent = accent
        self.update()

    def set_dot_color(self, color: str) -> None:
        self._dot = QColor(color)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        outer = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(PANEL_BG2))
        p.drawRoundedRect(outer, 20, 20)

        # Inset so the frame's own border ring stays visible on every side.
        inner = outer.adjusted(3, 3, -3, -3)
        clip = QPainterPath()
        clip.addRoundedRect(inner, 17, 17)
        p.save()
        p.setClipPath(clip)
        if self._image is not None:
            _draw_cover(p, inner, self._image)
        else:
            _paint_banner_placeholder(p, inner)
        p.restore()

        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(_blend_hex(self._accent, PANEL_BG2, 0.4)), 2))
        p.drawRoundedRect(outer, 20, 20)

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(self._dot)
        p.drawEllipse(QPointF(self.width() * 0.965 - 7, self.height() * 0.08 + 7), 7, 7)
        p.end()


class ChatView(QScrollArea):
    """Scrollable list of chat bubbles, shared by the Home preview and the Chat page.
    Messages are appended incrementally; `limit` caps how many stay on screen."""

    def __init__(self, limit=None, parent=None):
        super().__init__(parent)
        self._limit = limit
        self._rows = []
        self._empty = None
        self._pin = True  # stay scrolled to the newest message
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        body.setObjectName("transparent")
        self._layout = QVBoxLayout(body)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(8)
        self._layout.addStretch(1)
        self.setWidget(body)
        self.verticalScrollBar().rangeChanged.connect(self._on_range_changed)
        self._show_empty()

    def _on_range_changed(self, _low, high):
        if self._pin:
            self.verticalScrollBar().setValue(high)

    def _show_empty(self):
        self._empty = QLabel("No messages yet — say hello!")
        self._empty.setObjectName("muted")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._layout.insertWidget(0, self._empty)

    def _hide_empty(self):
        if self._empty is not None:
            _discard(self._empty)
            self._empty = None

    def _drop_row(self, row):
        self._layout.removeWidget(row)
        _discard(row)

    def set_history(self, history) -> None:
        for row in self._rows:
            self._drop_row(row)
        self._rows.clear()
        if self._limit:
            history = history[-self._limit:]
        self._pin = True
        if not history:
            if self._empty is None:
                self._show_empty()
            return
        self._hide_empty()
        for entry in history:
            self._append_row(entry)

    def add_message(self, entry: dict) -> None:
        bar = self.verticalScrollBar()
        self._pin = bar.value() >= bar.maximum() - 40
        self._hide_empty()
        self._append_row(entry)
        if self._limit and len(self._rows) > self._limit:
            self._drop_row(self._rows.pop(0))

    def _append_row(self, entry: dict) -> None:
        row = self._make_row(entry)
        self._layout.insertWidget(self._layout.count() - 1, row)
        self._rows.append(row)

    @staticmethod
    def _make_row(entry: dict) -> QWidget:
        role = entry.get("role", "luna")
        text = entry.get("text", "")
        bubble = QLabel(text)
        bubble.setObjectName({"user": "bubbleUser", "system": "bubbleSystem", "error": "bubbleError"}.get(role, "bubbleLuna"))
        bubble.setTextFormat(Qt.TextFormat.PlainText)
        bubble.setWordWrap(True)
        bubble.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        # Word-wrapped QLabels pick a narrow width on their own, so size the
        # bubble from the text: as wide as it needs, capped at BUBBLE_MAX_WIDTH.
        font = bubble.font()
        font.setPixelSize(BODY_PX)
        metrics = QFontMetrics(font)
        natural = max((metrics.horizontalAdvance(line) for line in text.splitlines()), default=0) + 34
        bubble.setFixedWidth(min(BUBBLE_MAX_WIDTH, max(natural, 48)))

        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        if role == "user":
            lay.addStretch(1)
            lay.addWidget(bubble)
        else:
            lay.addWidget(bubble)
            lay.addStretch(1)
        return row


class LunaApp(QMainWindow):
    # Backend threads (speech, face camera, voice loop...) only ever talk to the
    # UI through these signals, so every widget is touched from the Qt main thread.
    sig_message = pyqtSignal(str, str)
    sig_mode = pyqtSignal(str)
    sig_status = pyqtSignal(str)
    sig_face = pyqtSignal(str)
    sig_voice_btn = pyqtSignal(bool)
    sig_voice_status = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Luna — Mystical Assistant")
        self.resize(1536, 1000)
        self.setMinimumSize(1180, 760)

        self.accent = THEMES[THEME_NAME]["accent"]
        self.accent2 = THEMES[THEME_NAME]["accent2"]

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
        self._system_color = None
        self._closed = False

        self._build_layout()
        self._apply_theme()
        self.home_chat_view.set_history(self.chat_history)
        self.chat_page_view.set_history(self.chat_history)

        self.sig_message.connect(self._on_message)
        self.sig_mode.connect(self._on_mode)
        self.sig_status.connect(self._on_status)
        self.sig_face.connect(self._on_face)
        self.sig_voice_btn.connect(self._on_voice_btn)
        self.sig_voice_status.connect(self._on_voice_status)

        self._show_page("home")
        self._rotate_quote()
        self._tick_system_info()
        self._quote_timer = QTimer(self)
        self._quote_timer.timeout.connect(self._rotate_quote)
        self._quote_timer.start(9000)
        self._tick_timer = QTimer(self)
        self._tick_timer.timeout.connect(self._tick_system_info)
        self._tick_timer.start(1000)

    # ---------------- thread-safe API used by the backend ----------------
    def _emit(self, signal_name, *args):
        """Emits a UI signal from any thread. Backend threads can outlive the
        window during shutdown; once it is closing or already destroyed the
        update is simply dropped instead of raising. The signal is looked up by
        name inside the try block because merely touching it on a deleted
        window already raises RuntimeError."""
        if self._closed:
            return
        try:
            getattr(self, signal_name).emit(*args)
        except RuntimeError:  # underlying C++ window already deleted
            self._closed = True

    def append_message(self, role, text):
        self._emit("sig_message", role, text)

    def set_mode(self, mode):
        self._emit("sig_mode", mode)

    def set_status(self, text):
        self._emit("sig_status", text)

    def set_face_status(self, text):
        self._emit("sig_face", text)

    def set_voice_chat_active(self, active):
        self._emit("sig_voice_btn", active)

    # ---------------- signal slots (main thread) ----------------
    def _on_message(self, role, text):
        entry = {"role": role, "text": text, "ts": datetime.datetime.now().isoformat()}
        self.chat_history.append(entry)
        save_json(CHAT_HISTORY_PATH, self.chat_history[-500:])
        self.home_chat_view.add_message(entry)
        self.chat_page_view.add_message(entry)
        self.update_overview()

    def _on_mode(self, mode):
        self.current_mode = mode
        color = mode_colors.get(mode, self.accent)
        self.banner.set_dot_color(color)
        self.mode_status_label.setText(f"✦ {mode.capitalize()} ✦")
        self._style_mode_label(color)

    def _on_status(self, text):
        self.mode_status_label.setText(text)

    def _on_face(self, text):
        self.info_labels["camera"].setText(text)

    def _on_voice_btn(self, active):
        self.voice_chat_btn.setChecked(active)

    def _on_voice_status(self, text):
        self.voice_status_label.setText(text)

    def _style_mode_label(self, color):
        self.mode_status_label.setStyleSheet(f"color: {color}; font-size: 12px; font-weight: bold;")

    # ---------------- layout scaffolding ----------------
    def _build_layout(self):
        central = QFrame()
        central.setObjectName("center")
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_left_sidebar())
        root.addWidget(self._build_center(), 1)
        root.addWidget(self._build_right_sidebar())

    def _apply_theme(self):
        QApplication.instance().setStyleSheet(build_stylesheet(self.accent, self.accent2))
        self.banner.set_accent(self.accent)
        if self.current_mode == "idle":
            self._style_mode_label(self.accent)

    @staticmethod
    def _label(text, name=None, wrap=False, align=None):
        label = QLabel(text)
        if name:
            label.setObjectName(name)
        if wrap:
            label.setWordWrap(True)
        if align is not None:
            label.setAlignment(align)
        return label

    def _section_header(self, layout, text):
        layout.addSpacing(10)
        layout.addWidget(self._label(text, "h2"))
        layout.addSpacing(6)

    # ---------------- left sidebar ----------------
    def _build_left_sidebar(self):
        bar = QFrame()
        bar.setObjectName("sidebar")
        bar.setFixedWidth(260)
        lay = QVBoxLayout(bar)
        lay.setContentsMargins(22, 28, 22, 22)
        lay.setSpacing(0)

        lay.addWidget(self._label("☾ Luna", "brand"))
        lay.addWidget(self._label("Your AI Assistant", "muted"))
        lay.addSpacing(20)

        self.nav_buttons = {}
        group = QButtonGroup(self)
        group.setExclusive(True)
        for key, label in NAV_ITEMS:
            btn = QPushButton(label)
            btn.setObjectName("nav")
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _=False, k=key: self._show_page(k))
            group.addButton(btn)
            lay.addWidget(btn)
            lay.addSpacing(6)
            self.nav_buttons[key] = btn

        lay.addSpacing(8)
        divider = QFrame()
        divider.setObjectName("divider")
        lay.addWidget(divider)
        lay.addStretch(1)

        center = Qt.AlignmentFlag.AlignCenter
        self.quote_label = self._label("", "quote", wrap=True, align=center)
        lay.addWidget(self.quote_label)
        lay.addSpacing(4)
        lay.addWidget(self._label("— Luna", "brandSig", align=center))
        return bar

    # ---------------- center column ----------------
    def _build_center(self):
        center = QFrame()
        center.setObjectName("center")
        lay = QVBoxLayout(center)
        lay.setContentsMargins(18, 18, 18, 18)
        lay.setSpacing(10)

        self.pages = QStackedWidget()
        self.page_index = {}
        for key, builder in (
            ("home", self._build_home_page),
            ("chat", self._build_chat_page),
            ("voice", self._build_voice_page),
            ("memories", self._build_memories_page),
            ("settings", self._build_settings_page),
        ):
            self.page_index[key] = self.pages.addWidget(builder())
        lay.addWidget(self.pages, 1)
        lay.addWidget(self._build_input_bar())
        return center

    def _show_page(self, key):
        self.pages.setCurrentIndex(self.page_index[key])
        self.nav_buttons[key].setChecked(True)
        self.current_page_key = key
        if key == "memories":
            self._render_memories()

    # ---------------- Home page ----------------
    def _build_home_page(self):
        hero = QFrame()
        hero.setObjectName("hero")
        lay = QVBoxLayout(hero)
        lay.setContentsMargins(30, 26, 30, 18)
        lay.setSpacing(0)
        center = Qt.AlignmentFlag.AlignCenter

        self.banner = BannerWidget(self.accent)
        lay.addWidget(self.banner)
        lay.addSpacing(14)

        self.greeting_label = self._label(f"Hello {USER_NAME}...  ♡", "greeting", align=center)
        lay.addWidget(self.greeting_label)
        lay.addSpacing(2)
        lay.addWidget(self._label(
            "I'm Luna, your AI assistant. What would you like to do today?", "muted", wrap=True, align=center,
        ))
        lay.addSpacing(6)
        self.mode_status_label = self._label("✦ Idle ✦", align=center)
        self._style_mode_label(self.accent)
        lay.addWidget(self.mode_status_label)
        lay.addSpacing(10)

        preview = QFrame()
        preview.setObjectName("previewPanel")
        preview_lay = QVBoxLayout(preview)
        preview_lay.setContentsMargins(10, 10, 10, 10)
        self.home_chat_view = ChatView(limit=6)
        preview_lay.addWidget(self.home_chat_view)
        lay.addWidget(preview, 1)
        lay.addSpacing(14)

        chips = QHBoxLayout()
        chips.setSpacing(10)
        chips.addStretch(1)
        for chip in ["💭 Explain something", "💻 Help with coding", "📅 Plan my day", "🎲 Tell me a story"]:
            btn = QPushButton(chip)
            btn.setObjectName("chip")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _=False, t=chip: self._send_quick_prompt(t))
            chips.addWidget(btn)
        chips.addStretch(1)
        lay.addLayout(chips)
        return hero

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
    def _build_chat_page(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 4, 0, 0)
        lay.setSpacing(10)
        lay.addWidget(self._label("Chat History", "h1"))
        wrap = QFrame()
        wrap.setObjectName("panel")
        wrap_lay = QVBoxLayout(wrap)
        wrap_lay.setContentsMargins(14, 14, 14, 14)
        self.chat_page_view = ChatView(limit=CHAT_VIEW_LIMIT)
        wrap_lay.addWidget(self.chat_page_view)
        lay.addWidget(wrap, 1)
        return page

    # ---------------- persistent input bar ----------------
    def _build_input_bar(self):
        bar = QFrame()
        bar.setObjectName("inputBar")
        bar.setFixedHeight(60)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(20, 8, 10, 8)
        lay.setSpacing(6)

        self.chat_entry = QLineEdit()
        self.chat_entry.setObjectName("chatEntry")
        self.chat_entry.setPlaceholderText("Type your message...")
        self.chat_entry.returnPressed.connect(self._handle_send)
        lay.addWidget(self.chat_entry, 1)

        for text, name, size, slot in (
            ("🔇", "roundBtnSmall", 36, mute_luna),
            ("🎙", "roundBtn", 42, self._handle_mic),
            ("➤", "roundBtnAccent", 42, self._handle_send),
        ):
            btn = QPushButton(text)
            btn.setObjectName(name)
            btn.setFixedSize(size, size)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(slot)
            lay.addWidget(btn)
        return bar

    def _handle_send(self):
        text = self.chat_entry.text().strip()
        if not text:
            return
        self.chat_entry.clear()
        self.append_message("user", text)
        threading.Thread(target=process_command, args=(text.lower(),), daemon=True).start()

    def _handle_mic(self):
        threading.Thread(target=self._mic_worker, daemon=True).start()

    def _mic_worker(self):
        cmd = listen()
        if cmd:
            process_command(cmd)

    # ---------------- Voice page ----------------
    def _build_voice_page(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 4, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self._label("Voice", "h1"))
        lay.addSpacing(4)
        lay.addWidget(self._label(
            "Choose the neural voice Luna speaks with (Piper TTS, fully offline once downloaded).", "muted", wrap=True,
        ))
        lay.addSpacing(14)

        box = self._settings_section(lay, "Assistant Voice")
        self.voice_menu = QComboBox()
        self.voice_menu.addItems(PIPER_VOICE_OPTIONS)
        self.voice_menu.setCurrentText(PIPER_VOICE_NAME if PIPER_VOICE_NAME in PIPER_VOICE_OPTIONS else PIPER_VOICE_OPTIONS[0])
        box.addLayout(self._form_row("Voice", self.voice_menu, label_width=100))

        buttons = QHBoxLayout()
        apply_btn = QPushButton("Apply Voice")
        apply_btn.setObjectName("primary")
        apply_btn.clicked.connect(self._apply_voice)
        test_btn = QPushButton("🔊 Test Speech")
        test_btn.clicked.connect(
            lambda: threading.Thread(target=speak, args=("Hello! This is how I sound.",), daemon=True).start()
        )
        buttons.addWidget(apply_btn)
        buttons.addWidget(test_btn)
        buttons.addStretch(1)
        box.addLayout(buttons)

        self.voice_status_label = self._label(
            "Voice ready ✓" if _PIPER_READY else "Voice not ready — apply to download.", "tiny",
        )
        box.addWidget(self.voice_status_label)
        lay.addStretch(1)
        return page

    def _apply_voice(self):
        chosen = self.voice_menu.currentText()
        self.voice_status_label.setText("Downloading & applying voice…")

        def worker():
            apply_settings({"piper_voice": chosen})
            ok = _init_piper_voice()
            self._emit("sig_voice_status", "Voice ready ✓" if ok else "Failed to load voice — check your connection.")

        threading.Thread(target=worker, daemon=True).start()

    # ---------------- Memories page ----------------
    def _build_memories_page(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 4, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self._label("Memories", "h1"))
        lay.addSpacing(4)
        lay.addWidget(self._label(
            "Things Luna remembers about you — she uses these to personalize her answers.", "muted",
        ))
        lay.addSpacing(10)

        add_wrap = QFrame()
        add_wrap.setObjectName("panel")
        add_lay = QVBoxLayout(add_wrap)
        add_lay.setContentsMargins(14, 14, 14, 14)
        add_lay.setSpacing(8)
        self.memory_input = QPlainTextEdit()
        self.memory_input.setFixedHeight(70)
        add_lay.addWidget(self.memory_input)
        buttons = QHBoxLayout()
        self.memory_save_btn = QPushButton("Save Memory")
        self.memory_save_btn.setObjectName("primary")
        self.memory_save_btn.clicked.connect(self._save_memory)
        cancel_btn = QPushButton("Cancel Edit")
        cancel_btn.clicked.connect(self._cancel_memory_edit)
        buttons.addWidget(self.memory_save_btn)
        buttons.addWidget(cancel_btn)
        buttons.addStretch(1)
        add_lay.addLayout(buttons)
        lay.addWidget(add_wrap)
        lay.addSpacing(14)

        list_wrap = QFrame()
        list_wrap.setObjectName("panel")
        list_lay = QVBoxLayout(list_wrap)
        list_lay.setContentsMargins(10, 10, 10, 10)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        body.setObjectName("transparent")
        self.memories_layout = QVBoxLayout(body)
        self.memories_layout.setContentsMargins(2, 2, 2, 2)
        self.memories_layout.setSpacing(10)
        scroll.setWidget(body)
        list_lay.addWidget(scroll)
        lay.addWidget(list_wrap, 1)
        self._render_memories()
        return page

    def _render_memories(self):
        _clear_layout(self.memories_layout)
        if not self.memories:
            self.memories_layout.addWidget(
                self._label("No memories yet — add one above.", "muted", align=Qt.AlignmentFlag.AlignCenter)
            )
        else:
            for mem in reversed(self.memories):
                self.memories_layout.addWidget(self._memory_card(mem))
        self.memories_layout.addStretch(1)

    def _memory_card(self, mem):
        card = QFrame()
        card.setObjectName("inner")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(12, 10, 12, 8)
        lay.setSpacing(4)
        text = self._label(mem["text"], wrap=True)
        text.setTextFormat(Qt.TextFormat.PlainText)
        text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(text)

        meta = QHBoxLayout()
        meta.addWidget(self._label(mem.get("created_at", "")[:16].replace("T", " "), "tiny"))
        meta.addStretch(1)
        edit_btn = QPushButton("Edit")
        edit_btn.setObjectName("link")
        edit_btn.clicked.connect(lambda _=False, m=mem: self._edit_memory(m))
        delete_btn = QPushButton("Delete")
        delete_btn.setObjectName("dangerLink")
        delete_btn.clicked.connect(lambda _=False, m=mem: self._delete_memory(m))
        meta.addWidget(edit_btn)
        meta.addWidget(delete_btn)
        lay.addLayout(meta)
        return card

    def _save_memory(self):
        text = self.memory_input.toPlainText().strip()
        if not text:
            return
        if self.editing_memory_id:
            for m in self.memories:
                if m["id"] == self.editing_memory_id:
                    m["text"] = text
                    break
            self.editing_memory_id = None
            self.memory_save_btn.setText("Save Memory")
        else:
            self.memories.append({"id": str(uuid.uuid4()), "text": text, "created_at": datetime.datetime.now().isoformat()})
        save_json(MEMORIES_PATH, self.memories)
        self.memory_input.clear()
        self._render_memories()

    def _edit_memory(self, mem):
        self.editing_memory_id = mem["id"]
        self.memory_input.setPlainText(mem["text"])
        self.memory_save_btn.setText("Update Memory")

    def _cancel_memory_edit(self):
        self.editing_memory_id = None
        self.memory_input.clear()
        self.memory_save_btn.setText("Save Memory")

    def _delete_memory(self, mem):
        self.memories = [m for m in self.memories if m["id"] != mem["id"]]
        save_json(MEMORIES_PATH, self.memories)
        self._render_memories()

    # ---------------- Settings page ----------------
    def _settings_section(self, parent_layout, title):
        """Adds a titled card to parent_layout and returns the layout to fill."""
        box = QFrame()
        box.setObjectName("panel")
        outer = QVBoxLayout(box)
        outer.setContentsMargins(16, 12, 16, 12)
        outer.setSpacing(6)
        outer.addWidget(self._label(title, "sectionTitle"))
        inner = QVBoxLayout()
        inner.setSpacing(10)
        outer.addLayout(inner)
        parent_layout.addWidget(box)
        parent_layout.addSpacing(8)
        return inner

    def _form_row(self, label, widget, label_width=140):
        row = QHBoxLayout()
        caption = self._label(label, "muted")
        caption.setFixedWidth(label_width)
        row.addWidget(caption)
        row.addWidget(widget, 1)
        return row

    def _build_settings_page(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        body.setObjectName("transparent")
        lay = QVBoxLayout(body)
        lay.setContentsMargins(0, 4, 8, 0)
        lay.setSpacing(0)
        lay.addWidget(self._label("Settings", "h1"))
        lay.addSpacing(14)

        api_box = self._settings_section(lay, "API & Profile")
        self.settings_entries = {}
        field_values = {
            "gemini_api_key": GEMINI_API_KEY, "weather_api_key": WEATHER_API_KEY,
            "home_city": HOME_CITY, "user_name": USER_NAME,
        }
        for key, label, secret in [
            ("gemini_api_key", "Gemini API Key", True),
            ("weather_api_key", "Weather API Key", True),
            ("home_city", "Home City", False),
            ("user_name", "Your Name", False),
        ]:
            entry = QLineEdit(field_values.get(key, ""))
            if secret:
                entry.setEchoMode(QLineEdit.EchoMode.Password)
            api_box.addLayout(self._form_row(label, entry))
            self.settings_entries[key] = entry
        save_row = QHBoxLayout()
        save_row.addStretch(1)
        save_btn = QPushButton("Save API Settings")
        save_btn.setObjectName("primary")
        save_btn.clicked.connect(self._save_api_settings)
        save_row.addWidget(save_btn)
        api_box.addLayout(save_row)

        theme_box = self._settings_section(lay, "Appearance")
        self.theme_menu = QComboBox()
        self.theme_menu.addItems([t["name"] for t in THEMES.values()])
        self.theme_menu.setCurrentText(THEMES[THEME_NAME]["name"])
        self.theme_menu.textActivated.connect(self._on_theme_selected)
        theme_box.addLayout(self._form_row("Theme", self.theme_menu))

        sys_box = self._settings_section(lay, "System")
        sys_box.addWidget(self._label(
            "Use the power icons in the right sidebar to shut down, restart, or sleep this computer.", "muted", wrap=True,
        ))
        lay.addStretch(1)
        scroll.setWidget(body)
        return scroll

    def _save_api_settings(self):
        vals = {k: e.text().strip() for k, e in self.settings_entries.items()}
        apply_settings(vals)
        self.append_message("system", "⚙ Settings saved.")

    def _on_theme_selected(self, display_name):
        key = next((k for k, v in THEMES.items() if v["name"] == display_name), THEME_NAME)
        self.accent = THEMES[key]["accent"]
        self.accent2 = THEMES[key]["accent2"]
        apply_settings({"theme": key})
        self._apply_theme()

    # ---------------- right sidebar ----------------
    def _build_right_sidebar(self):
        bar = QFrame()
        bar.setObjectName("sidebar")
        bar.setFixedWidth(300)
        lay = QVBoxLayout(bar)
        lay.setContentsMargins(20, 26, 20, 18)
        lay.setSpacing(0)

        profile = QHBoxLayout()
        profile.setSpacing(10)
        avatar = QLabel()
        avatar.setPixmap(make_avatar_pixmap(52))
        avatar.setFixedSize(52, 52)
        profile.addWidget(avatar)
        who = QVBoxLayout()
        who.setSpacing(0)
        who.addWidget(self._label("Luna", "profileName"))
        online = QLabel("●  Online")
        online.setStyleSheet("color: #33d17a; font-size: 12px;")
        who.addWidget(online)
        profile.addLayout(who)
        profile.addStretch(1)
        lay.addLayout(profile)
        lay.addSpacing(14)

        self._section_header(lay, "◈  Quick Actions")
        grid = QGridLayout()
        grid.setSpacing(12)
        actions = [
            ("💬", "Ask Anything", self.qa_ask_anything),
            ("✅", "Create Task", self.qa_create_task),
            ("🗂", "Open Apps", self.qa_open_apps),
            ("🎙", "Voice Chat", self.qa_toggle_voice),
        ]
        for i, (icon, label, cmd) in enumerate(actions):
            btn = QPushButton(f"{icon}\n{label}")
            btn.setObjectName("qa")
            btn.setMinimumHeight(64)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(cmd)
            if label == "Voice Chat":
                btn.setCheckable(True)
                self.voice_chat_btn = btn
            grid.addWidget(btn, *divmod(i, 2))
        lay.addLayout(grid)
        lay.addSpacing(14)

        self._section_header(lay, "🗓  Today's Overview")
        self.overview_values = {}
        for key, icon, name, handler in (
            ("tasks", "✅", "Tasks", self.show_tasks_popup),
            ("reminders", "🔔", "Reminders", self.show_reminders_popup),
            ("messages", "💬", "Messages", lambda: self._show_page("chat")),
            ("system", "⚙", "System", lambda: self._show_page("settings")),
        ):
            row = ClickableRow(icon, name)
            row.clicked.connect(handler)
            lay.addWidget(row)
            self.overview_values[key] = row.value
        lay.addSpacing(14)

        self._section_header(lay, "🖥  System Info")
        self.info_labels = {}
        for key, label in [("time", "Time"), ("date", "Date"), ("mood", "Mood"), ("uptime", "Uptime"), ("camera", "Camera")]:
            row = QHBoxLayout()
            row.setContentsMargins(0, 3, 0, 3)
            row.addWidget(self._label(label, "muted"))
            row.addStretch(1)
            val = self._label("—", "infoValue")
            row.addWidget(val)
            lay.addLayout(row)
            self.info_labels[key] = val
        lay.addSpacing(10)

        power = QHBoxLayout()
        power.setSpacing(4)
        for text, name, action in (
            ("⏻ Shutdown", "powerShutdown", "shutdown"),
            ("⟳ Restart", "powerRestart", "restart"),
            ("☾ Sleep", "powerSleep", "sleep"),
        ):
            btn = QPushButton(text)
            btn.setObjectName(name)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _=False, a=action: self._confirm_power(a))
            power.addWidget(btn)
        lay.addLayout(power)

        lay.addStretch(1)
        self.right_quote_label = self._label("", "quote", wrap=True, align=Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.right_quote_label)
        return bar

    # ---------------- quick actions ----------------
    def qa_ask_anything(self):
        self._show_page("chat")
        self.chat_entry.setFocus()

    @staticmethod
    def _new_item(text):
        return {"id": str(uuid.uuid4()), "text": text, "done": False, "created_at": datetime.datetime.now().isoformat()}

    def qa_create_task(self):
        val, ok = QInputDialog.getText(self, "New Task", "What's the task?")
        val = val.strip()
        if ok and val:
            self.tasks.append(self._new_item(val))
            save_json(TASKS_PATH, self.tasks)
            self.update_overview()
            self.append_message("system", f"✅ Task added: {val}")

    def qa_open_apps(self):
        self._show_open_apps_popup()

    def qa_toggle_voice(self):
        self.voice_listen_active = not self.voice_listen_active
        self.voice_chat_btn.setChecked(self.voice_listen_active)
        if self.voice_listen_active:
            threading.Thread(target=voice_loop, daemon=True).start()

    # ---------------- overview / system info ticking ----------------
    def update_overview(self):
        pending = sum(1 for t in self.tasks if not t.get("done"))
        upcoming = sum(1 for r in self.reminders if not r.get("done"))
        today = datetime.date.today().isoformat()
        msgs_today = sum(1 for m in self.chat_history if m.get("ts", "").startswith(today))
        healthy = bool(GEMINI_API_KEY) and _PIPER_READY
        self.overview_values["tasks"].setText(f"{pending} pending")
        self.overview_values["reminders"].setText(f"{upcoming} upcoming")
        self.overview_values["messages"].setText(f"{msgs_today} today")
        system_value = self.overview_values["system"]
        system_value.setText("All good" if healthy else "Check Settings")
        color = "#33d17a" if healthy else "#ff9d4d"
        if color != self._system_color:
            self._system_color = color
            system_value.setStyleSheet(f"color: {color};")

    def _tick_system_info(self):
        now = datetime.datetime.now()
        self.info_labels["time"].setText(now.strftime("%I:%M %p"))
        self.info_labels["date"].setText(now.strftime("%b %d, %Y"))
        mood_map = {"idle": "Calm ♡", "listening": "Curious ✦", "speaking": "Chatty ♪", "executing": "Focused ⚡", "confirm": "Alert ⚠"}
        self.info_labels["mood"].setText(mood_map.get(self.current_mode, "Calm ♡"))
        elapsed = int(time.time() - self.start_time)
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        self.info_labels["uptime"].setText(f"{h:02d}:{m:02d}:{s:02d}")
        self.update_overview()

    def _rotate_quote(self):
        text = f"“{LUNA_QUOTES[self.quote_index % len(LUNA_QUOTES)]}”"
        self.quote_index += 1
        self.quote_label.setText(text)
        self.right_quote_label.setText(text)

    # ---------------- popups / modals ----------------
    def _make_dialog(self, title, width, height):
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(width, height)
        dlg.setModal(True)
        return dlg

    def show_tasks_popup(self):
        self._show_list_popup("Tasks", self.tasks, TASKS_PATH, "New task", "No tasks yet — add one!")

    def show_reminders_popup(self):
        self._show_list_popup("Reminders", self.reminders, REMINDERS_PATH, "New reminder", "No reminders yet — add one!")

    def _show_list_popup(self, title, items, path, add_prompt, empty_text):
        dlg = self._make_dialog(title, 380, 440)
        root = QVBoxLayout(dlg)
        root.setContentsMargins(14, 16, 14, 16)
        root.setSpacing(8)
        root.addWidget(self._label(title, "h1", align=Qt.AlignmentFlag.AlignCenter))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        body.setObjectName("transparent")
        rows = QVBoxLayout(body)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setSpacing(8)
        scroll.setWidget(body)
        root.addWidget(scroll, 1)

        def toggle(it, cb, checked):
            it["done"] = checked
            save_json(path, items)
            cb.setStyleSheet(f"color: {TEXT_MUTED};" if checked else "")
            self.update_overview()

        def remove(it):
            items.remove(it)
            save_json(path, items)
            refresh()
            self.update_overview()

        def make_row(it):
            row = QFrame()
            row.setObjectName("inner")
            lay = QHBoxLayout(row)
            lay.setContentsMargins(10, 6, 6, 6)
            cb = QCheckBox(it["text"])
            done = it.get("done", False)
            cb.setChecked(done)
            if done:
                cb.setStyleSheet(f"color: {TEXT_MUTED};")
            cb.toggled.connect(lambda checked, i=it, c=cb: toggle(i, c, checked))
            lay.addWidget(cb, 1)
            del_btn = QPushButton("✕")
            del_btn.setObjectName("dangerLink")
            del_btn.setFixedSize(28, 28)
            del_btn.clicked.connect(lambda _=False, i=it: remove(i))
            lay.addWidget(del_btn)
            return row

        def refresh():
            _clear_layout(rows)
            if not items:
                rows.addWidget(self._label(empty_text, "muted", align=Qt.AlignmentFlag.AlignCenter))
            for it in list(items):
                rows.addWidget(make_row(it))
            rows.addStretch(1)

        def add_new():
            val, ok = QInputDialog.getText(dlg, add_prompt, add_prompt)
            val = val.strip()
            if ok and val:
                items.append(self._new_item(val))
                save_json(path, items)
                refresh()
                self.update_overview()

        refresh()
        add_btn = QPushButton("+ Add")
        add_btn.setObjectName("primary")
        add_btn.clicked.connect(add_new)
        root.addWidget(add_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        dlg.exec()

    def _show_open_apps_popup(self):
        dlg = self._make_dialog("Open App", 320, 300)
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.setSpacing(8)
        lay.addWidget(self._label("Open an App", "h1", align=Qt.AlignmentFlag.AlignCenter))
        apps = [
            ("🌐", "Browser", "browser"), ("📁", "Files", "files"), ("⌨", "Terminal", "terminal"),
            ("🧮", "Calculator", "calculator"), ("📝", "Notepad", "notepad"),
        ]
        for icon, label, kind in apps:
            btn = QPushButton(f"{icon}  {label}")
            btn.setObjectName("appBtn")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _=False, k=kind: (_launch_app(k), dlg.accept()))
            lay.addWidget(btn)
        lay.addStretch(1)
        dlg.exec()

    def _confirm_power(self, action):
        dlg = self._make_dialog("Confirm", 340, 160)
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(16, 24, 16, 16)
        verbs = {"shutdown": "shut down", "restart": "restart", "sleep": "put to sleep"}
        lay.addWidget(self._label(
            f"Are you sure you want to {verbs[action]} the system?", wrap=True, align=Qt.AlignmentFlag.AlignCenter,
        ))
        lay.addStretch(1)

        def confirm():
            dlg.accept()
            threading.Thread(target=_execute_system_action, args=(action,), daemon=True).start()

        row = QHBoxLayout()
        row.addStretch(1)
        yes_btn = QPushButton("Yes, proceed")
        yes_btn.setObjectName("danger")
        yes_btn.clicked.connect(confirm)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dlg.reject)
        row.addWidget(yes_btn)
        row.addWidget(cancel_btn)
        row.addStretch(1)
        lay.addLayout(row)
        dlg.exec()

    def closeEvent(self, event):
        global _app_running
        self._closed = True
        _app_running = False
        _stop_speech_flag.set()
        speech_done_event.set()
        self.voice_listen_active = False
        logging.info("Luna window closed cleanly.")
        event.accept()


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
                    app.set_voice_chat_active(False)
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
def _log_uncaught_exception(exc_type, exc_value, exc_tb):
    # PyQt6 aborts the process on an unhandled exception inside a slot; log instead.
    logging.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))


def _build_palette() -> QPalette:
    palette = QPalette()
    for role, color in (
        (QPalette.ColorRole.Window, BG_DARK),
        (QPalette.ColorRole.WindowText, TEXT_LIGHT),
        (QPalette.ColorRole.Base, PANEL_BG2),
        (QPalette.ColorRole.AlternateBase, PANEL_BG),
        (QPalette.ColorRole.Text, TEXT_LIGHT),
        (QPalette.ColorRole.Button, PANEL_BG2),
        (QPalette.ColorRole.ButtonText, TEXT_LIGHT),
        (QPalette.ColorRole.PlaceholderText, TEXT_MUTED),
        (QPalette.ColorRole.Highlight, THEMES[THEME_NAME]["accent"]),
        (QPalette.ColorRole.HighlightedText, "#ffffff"),
    ):
        palette.setColor(role, QColor(color))
    return palette


if __name__ == "__main__":
    sys.excepthook = _log_uncaught_exception
    qt_app = QApplication(sys.argv)
    qt_app.setStyle("Fusion")
    ui_font = QFont()
    ui_font.setFamilies(UI_FONT_FAMILIES)
    qt_app.setFont(ui_font)
    qt_app.setPalette(_build_palette())

    app = LunaApp()
    app.show()
    app.append_message(
        "luna",
        f"Hello {USER_NAME}! I'm Luna, your mystical assistant. Type below or tap the "
        "mic whenever you're ready — or turn on Voice Chat for hands-free conversation.",
    )
    face_thread = start_face_recognition()
    try:
        exit_code = qt_app.exec()
    except Exception as e:
        logging.critical(f"Main window loop crashed: {e}")
        exit_code = 1
    _app_running = False
    if face_thread is not None:
        face_thread.join(timeout=3)  # let the camera loop release the webcam before Qt tears down
    sys.exit(exit_code)