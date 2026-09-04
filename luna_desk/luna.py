"""Luna — desktop voice assistant (Tkinter UI)."""

import datetime
import json
import logging
import os
import platform
import random
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import wave
from logging.handlers import RotatingFileHandler
from pathlib import Path
from tkinter import Text, Scrollbar, Entry
import tkinter as tk
import webbrowser

import pyjokes
import requests
import speech_recognition as sr
from PIL import Image, ImageDraw, ImageTk
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

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
try:
    import config
    WEATHER_API_KEY = getattr(config, "WEATHER_API_KEY", os.environ.get("WEATHER_API_KEY", ""))
    GEMINI_API_KEY = getattr(config, "GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
    HOME_CITY = getattr(config, "HOME_CITY", os.environ.get("HOME_CITY", "your city"))
    FACE_MATCH_THRESHOLD = float(getattr(config, "FACE_MATCH_THRESHOLD", os.environ.get("FACE_MATCH_THRESHOLD", 0.60)))
    PIPER_VOICE_NAME = getattr(config, "PIPER_VOICE", os.environ.get("PIPER_VOICE", "en_US-amy-medium"))
except ImportError:
    WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY", "")
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    HOME_CITY = os.environ.get("HOME_CITY", "your city")
    FACE_MATCH_THRESHOLD = float(os.environ.get("FACE_MATCH_THRESHOLD", 0.60))
    PIPER_VOICE_NAME = os.environ.get("PIPER_VOICE", "en_US-amy-medium")

# Face detection/recognition uses OpenCV's built-in YuNet (detector) and
# SFace (recognizer) DNN models — small ONNX files, no TensorFlow/PyTorch
# dependency. They're downloaded once into ./models on first run.
FACE_MODEL = "sface"  # label used to invalidate incompatible saved face DBs if this ever changes
FACES_MODELS_DIR = "models"
FACE_DETECTOR_MODEL_PATH = os.path.join(FACES_MODELS_DIR, "face_detection_yunet_2023mar.onnx")
FACE_RECOGNIZER_MODEL_PATH = os.path.join(FACES_MODELS_DIR, "face_recognition_sface_2021dec.onnx")
FACE_DETECTOR_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
FACE_RECOGNIZER_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

# ------------------------------------------------------------
# Logging (rotating so luna.log can't grow without bound)
# ------------------------------------------------------------
_log_handler = RotatingFileHandler("luna.log", maxBytes=2_000_000, backupCount=3)
_log_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[_log_handler])

if not _CV2_AVAILABLE:
    logging.warning("OpenCV not installed — face detection disabled.")
if not _NUMPY_AVAILABLE:
    logging.warning("numpy not installed — face recognition disabled.")

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
# Voice setup — Piper neural TTS with a female voice
# ------------------------------------------------------------
# Piper is a fast, fully offline neural TTS engine with natural-sounding
# voices — a real fix for Linux, where pyttsx3 just wraps espeak and every
# voice (male or female) sounds robotic. The .onnx model + .onnx.json
# config for PIPER_VOICE_NAME are downloaded once into ./voices on first
# run and reused after that. Change PIPER_VOICE_NAME (or set env var
# PIPER_VOICE) to try a different voice — see https://github.com/rhasspy/piper/blob/master/VOICES.md
PIPER_VOICES_DIR = "voices"
_piper_voice = None
_PIPER_READY = False


def _piper_files_missing(model_path: str, config_path: str) -> bool:
    return not (
        os.path.exists(model_path) and os.path.getsize(model_path) > 0
        and os.path.exists(config_path) and os.path.getsize(config_path) > 0
    )


def _ensure_piper_voice_files(voice_name: str):
    """Downloads the .onnx model + .onnx.json config for `voice_name` into
    ./voices on first run. Returns (model_path, config_path), or None on
    failure (e.g. no internet on first run)."""
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
        "Piper TTS unavailable — check internet access on first run (needed "
        "to download the voice model) and that 'piper-tts' is installed."
    )

# Piper only synthesizes audio; something still has to play the WAV. Prefer
# paplay/aplay (installed on virtually every Linux desktop) and fall back
# to ffplay if present.
_AUDIO_PLAYERS = [p for p in ("paplay", "aplay", "ffplay") if shutil.which(p)]
if not _AUDIO_PLAYERS:
    logging.warning(
        "No audio player found (looked for paplay/aplay/ffplay) — Piper "
        "speech won't be audible. Install alsa-utils or pulseaudio-utils."
    )

# ------------------------------------------------------------
# Shared state
# ------------------------------------------------------------
current_mode = "idle"
mode_colors = {
    "idle": "#9370db",
    "listening": "#00ffff",
    "speaking": "#ff69b4",
    "executing": "#ffd700",
    "confirm": "#ff4444",
}

speech_done_event = threading.Event()
_stop_speech_flag = threading.Event()
_pending_system_action: str | None = None
_app_running = True

# ------------------------------------------------------------
# Face recognition state
# ------------------------------------------------------------
FACES_DIR = "faces"
os.makedirs(FACES_DIR, exist_ok=True)
FACES_DB_PATH = os.path.join(FACES_DIR, "faces_db.json")
FACES_NPZ_PATH = os.path.join(FACES_DIR, "face_encodings.npz")

_known_face_encodings: list = []
_known_face_names: list = []

_last_greeted_name: str | None = None
_last_greeted_time: float = 0.0
GREET_COOLDOWN_SECONDS = 30

_awaiting_face_name = threading.Event()

# Only the camera thread writes _latest_frame; all reads go through
# _get_shared_frame(). This avoids opening a second cv2.VideoCapture(0),
# which fails silently on most webcams.
_latest_frame = None
_face_cam_lock = threading.Lock()
_FRAME_WAIT_TIMEOUT = 3.0


def _load_face_db():
    global _known_face_encodings, _known_face_names
    if not (_FACE_MODELS_READY):
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
    if not (_FACE_MODELS_READY):
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


def identify_face(encoding) -> str | None:
    if not _known_face_encodings:
        return None
    distances = [_cosine_distance(encoding, known) for known in _known_face_encodings]
    best_idx = int(np.argmin(distances))
    if distances[best_idx] < FACE_MATCH_THRESHOLD:
        return _known_face_names[best_idx]
    return None


def _detect_raw_faces(frame):
    """Runs the YuNet detector and returns its raw Nx15 output (bbox +
    5 landmarks + score per face), or None if nothing was found."""
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
    """Cheap per-frame detection for drawing boxes / counting faces.
    Returns a list of (top, right, bottom, left) tuples."""
    faces = _detect_raw_faces(frame)
    if faces is None:
        return []
    boxes = []
    for f in faces:
        x, y, w, h = f[:4].astype(int)
        boxes.append((y, x + w, y + h, x))
    return boxes


def _get_face_embeddings(frame) -> list:
    """Detect + embed every face in a frame in one pass (SFace, 128-d)."""
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
# Window & canvas
# ------------------------------------------------------------
WINDOW_W, WINDOW_H = 480, 860
CANVAS_W, CANVAS_H = 430, 270

window = tk.Tk()
window.title("Luna — Mystical Assistant")
window.geometry(f"{WINDOW_W}x{WINDOW_H}")
window.configure(bg="#0a0a1a")
window.attributes("-alpha", 0.98)
window.geometry("+{}+{}".format(
    (window.winfo_screenwidth() // 2) - (WINDOW_W // 2),
    (window.winfo_screenheight() // 2) - (WINDOW_H // 2) - 30,
))

canvas = tk.Canvas(window, width=CANVAS_W, height=CANVAS_H, bg="#0a0a1a", highlightthickness=0)
canvas.pack(pady=6)


class Particle:
    def __init__(self):
        self.reset()
        self.y = random.randint(0, CANVAS_H)

    def reset(self):
        self.x = random.randint(10, CANVAS_W - 10)
        self.y = 0
        self.size = random.randint(1, 3)
        self.speed = random.uniform(0.4, 1.6)
        self.color = random.choice(["#9370db", "#00ffff", "#ff69b4", "#ffd700", "#ffffff"])
        self.id = None

    def draw(self):
        self.id = canvas.create_oval(self.x, self.y, self.x + self.size, self.y + self.size,
                                      fill=self.color, outline="", tags="particle")

    def move(self):
        self.y += self.speed
        if self.y > CANVAS_H:
            self.reset()
        canvas.coords(self.id, self.x, self.y, self.x + self.size, self.y + self.size)


particles = []
for _ in range(20):
    p = Particle()
    p.draw()
    particles.append(p)


def animate_particles():
    for p in particles:
        p.move()
    window.after(80, animate_particles)


animate_particles()


def create_magical_image(image_path: str, size: int) -> ImageTk.PhotoImage:
    try:
        img = Image.open(image_path).resize((size, size), Image.Resampling.LANCZOS)
        mask = Image.new("L", (size, size), 0)
        ImageDraw.Draw(mask).ellipse((0, 0, size, size), fill=255)
        result = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        result.paste(img, (0, 0), mask)
        return ImageTk.PhotoImage(result)
    except Exception:
        ph = Image.new("RGBA", (size, size), (30, 10, 60, 255))
        d = ImageDraw.Draw(ph)
        c = size // 2
        d.ellipse((c - 80, c - 80, c + 80, c + 80), fill=(70, 30, 120, 255))
        d.ellipse((c - 40, c - 40, c + 40, c + 40), outline=(255, 255, 255, 255), width=3)
        d.ellipse((c - 20, c - 20, c + 20, c + 20), outline=(255, 255, 255, 255), width=2)
        d.ellipse((c - 15, c - 5, c + 5, c + 15), fill=(255, 255, 255, 255))
        d.ellipse((c - 10, c - 5, c, c + 15), fill=(70, 30, 120, 255))
        d.text((c - 25, c + 25), "LUNA", fill=(255, 255, 255, 255))
        return ImageTk.PhotoImage(ph)


AVATAR_SIZE = 180
luna_photo = create_magical_image(os.path.join("assets", "miss_luna.jpeg"), AVATAR_SIZE)
image_obj = canvas.create_image(CANVAS_W // 2, CANVAS_H // 2, image=luna_photo)


def _do_update_border_color(mode: str):
    global current_mode
    current_mode = mode
    color = mode_colors.get(mode, "#9370db")
    canvas.delete("border")
    cx, cy, r = CANVAS_W // 2, CANVAS_H // 2, AVATAR_SIZE // 2
    glow = 10
    for i in range(glow, 0, -1):
        if i > glow - 4:
            rr = int(int(color[1:3], 16) * (i / glow) * 0.5 + int(color[1:3], 16) * 0.5)
            gg = int(int(color[3:5], 16) * (i / glow) * 0.5 + int(color[3:5], 16) * 0.5)
            bb = int(int(color[5:7], 16) * (i / glow) * 0.5 + int(color[5:7], 16) * 0.5)
            gc = f"#{rr:02x}{gg:02x}{bb:02x}"
        else:
            gc = color
        canvas.create_oval(cx - r - i, cy - r - i, cx + r + i, cy + r + i, outline=gc, width=2, tags="border")


_pulse_radius = 0
_pulse_growing = True


def _animate_mic_pulse():
    global _pulse_radius, _pulse_growing
    if current_mode != "listening":
        canvas.delete("mic_pulse")
        window.after(200, _animate_mic_pulse)
        return
    canvas.delete("mic_pulse")
    cx, cy, base_r = CANVAS_W // 2, CANVAS_H // 2, AVATAR_SIZE // 2
    pr = base_r + 15 + _pulse_radius
    canvas.create_oval(cx - pr, cy - pr, cx + pr, cy + pr, outline="#00ffff",
                        width=max(1, 3 - _pulse_radius // 8), tags="mic_pulse", dash=(6, 4))
    if _pulse_growing:
        _pulse_radius += 2
        if _pulse_radius >= 24:
            _pulse_growing = False
    else:
        _pulse_radius -= 2
        if _pulse_radius <= 0:
            _pulse_growing = True
    window.after(60, _animate_mic_pulse)


_animate_mic_pulse()


def safe_ui(func, *args):
    """Schedule a UI update on the main Tkinter thread."""
    window.after(0, func, *args)


def update_border_color(mode: str):
    safe_ui(_do_update_border_color, mode)


_do_update_border_color("idle")

# ------------------------------------------------------------
# Response box
# ------------------------------------------------------------
response_frame = tk.Frame(window, bg="#0a0a1a")
response_frame.pack(pady=6, fill="x", padx=20)

fantasy_box = tk.Canvas(response_frame, width=CANVAS_W, height=200, bg="#0a0a1a", highlightthickness=0)
fantasy_box.pack()
fantasy_box.create_rectangle(10, 10, CANVAS_W - 10, 190, outline="#9370db", width=2, fill="#0f0f28", stipple="gray12")
fantasy_box.create_text(22, 22, text="♆", font=("Arial", 13), fill="#00ffff")
fantasy_box.create_text(CANVAS_W - 22, 22, text="☽", font=("Arial", 13), fill="#ff69b4")
fantasy_box.create_text(22, 180, text="☄", font=("Arial", 13), fill="#ffd700")
fantasy_box.create_text(CANVAS_W - 22, 180, text="✦", font=("Arial", 13), fill="#9370db")

text_frame = tk.Frame(fantasy_box, bg="#0f0f28")
fantasy_box.create_window(CANVAS_W // 2, 100, window=text_frame, width=CANVAS_W - 56, height=162, anchor="center")

response_scrollbar = Scrollbar(text_frame, orient="vertical")
response_text = Text(
    text_frame,
    font=("Segoe UI", 11) if platform.system() == "Windows" else ("Helvetica Neue", 11),
    fg="#e6e6fa", bg="#0f0f28", wrap="word",
    yscrollcommand=response_scrollbar.set, relief="flat", highlightthickness=0, padx=6, pady=4,
)
response_scrollbar.config(command=response_text.yview)
response_scrollbar.pack(side="right", fill="y")
response_text.pack(side="left", fill="both", expand=True)

response_text.tag_configure("luna", foreground="#e6e6fa")
response_text.tag_configure("user", foreground="#9370db")
response_text.tag_configure("system", foreground="#ffd700")
response_text.tag_configure("error", foreground="#ff4444")
response_text.tag_configure("face", foreground="#00ffff")

response_text.insert("1.0", "✦ Luna is waking up…\n", "system")
response_text.config(state="disabled")


def _do_append_response(text: str, tag: str = "luna"):
    response_text.config(state="normal")
    response_text.insert("end", text + "\n", tag)
    response_text.see("end")
    response_text.config(state="disabled")


# ------------------------------------------------------------
# Status label & orb
# ------------------------------------------------------------
status_label = tk.Label(window, text="✦ Idle ✦", font=("Papyrus", 10, "bold"), fg=mode_colors["idle"], bg="#0a0a1a")
status_label.pack(pady=4)

orb_indicator = canvas.create_oval(CANVAS_W - 22, 12, CANVAS_W - 8, 26, fill=mode_colors["idle"], outline="", tags="orb")


def _update_orb():
    canvas.itemconfig(orb_indicator, fill=mode_colors.get(current_mode, "#9370db"))
    window.after(500, _update_orb)


_update_orb()


def _do_update_status(text: str):
    status_label.config(text=text, fg=mode_colors.get(current_mode, "#9370db"))
    window.update_idletasks()


# ------------------------------------------------------------
# Face status bar
# ------------------------------------------------------------
face_status_frame = tk.Frame(window, bg="#0a0a1a")
face_status_frame.pack(pady=2, padx=20, fill="x")

face_cam_label = tk.Label(
    face_status_frame,
    text="👁 Camera: initializing…" if (_FACE_MODELS_READY) else "👁 Camera: models unavailable — check requirements + internet on first run",
    font=("Segoe UI", 9) if platform.system() == "Windows" else ("Helvetica Neue", 9),
    fg="#00ffff", bg="#0a0a1a", anchor="w",
)
face_cam_label.pack(side="left", fill="x", expand=True)

_face_thumb_size = 80
face_thumb_canvas = tk.Canvas(face_status_frame, width=_face_thumb_size, height=_face_thumb_size,
                               bg="#0a0a1a", highlightthickness=1, highlightbackground="#9370db")
if _FACE_MODELS_READY:
    face_thumb_canvas.pack(side="right", padx=4)

_face_thumb_photo = None


def _do_update_face_status(text: str):
    face_cam_label.config(text=f"👁 {text}")


def _do_update_face_thumb(pil_img: Image.Image):
    global _face_thumb_photo
    pil_img = pil_img.resize((_face_thumb_size, _face_thumb_size), Image.Resampling.LANCZOS)
    _face_thumb_photo = ImageTk.PhotoImage(pil_img)
    face_thumb_canvas.delete("all")
    face_thumb_canvas.create_image(0, 0, anchor="nw", image=_face_thumb_photo)


def mute_luna():
    _stop_speech_flag.set()
    safe_ui(_do_update_border_color, "idle")
    safe_ui(_do_update_status, "✦ Idle ✦")


mute_button = tk.Button(
    window, text="⏹  Mute / Stop", font=("Papyrus", 9, "bold"), fg="#ffffff", bg="#3a0a4a",
    activebackground="#6a1a8a", activeforeground="#ffffff", relief="flat", cursor="hand2", command=mute_luna,
)
mute_button.pack(pady=4)

# ------------------------------------------------------------
# Text input fallback
# ------------------------------------------------------------
input_frame = tk.Frame(window, bg="#0a0a1a")
input_frame.pack(pady=6, padx=20, fill="x")

text_input = Entry(
    input_frame,
    font=("Segoe UI", 11) if platform.system() == "Windows" else ("Helvetica Neue", 11),
    fg="#e6e6fa", bg="#1a1a3a", insertbackground="#e6e6fa", relief="flat",
    highlightthickness=1, highlightcolor="#9370db", highlightbackground="#3a0a4a",
)
text_input.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 8))
text_input.insert(0, "Type a command or question…")
text_input.config(fg="#666688")


def _on_entry_click(event):
    if text_input.get() == "Type a command or question…":
        text_input.delete(0, "end")
        text_input.config(fg="#e6e6fa")


def _on_focus_out(event):
    if not text_input.get():
        text_input.insert(0, "Type a command or question…")
        text_input.config(fg="#666688")


text_input.bind("<FocusIn>", _on_entry_click)
text_input.bind("<FocusOut>", _on_focus_out)

send_button = tk.Button(
    input_frame, text="Send ✦", font=("Papyrus", 9, "bold"), fg="#ffffff", bg="#3a0a4a",
    activebackground="#6a1a8a", activeforeground="#ffffff", relief="flat", cursor="hand2",
)
send_button.pack(side="right")

tk.Label(window, text="✦ Luna — Mystical Assistant ✦", font=("Papyrus", 9), fg="#ba55d3", bg="#0a0a1a").pack(side="bottom", pady=4)

runes_frame = tk.Frame(window, bg="#0a0a1a")
runes_frame.pack(side="bottom", pady=2)
for sym in ["✧", "⋆", "✦", "✶", "✸", "✹", "✺", "✻"]:
    tk.Label(runes_frame, text=sym, font=("Arial", 11), fg="#9370db", bg="#0a0a1a").pack(side="left", padx=3)


# ------------------------------------------------------------
# Speak / listen
# ------------------------------------------------------------
def speak(text: str):
    """Speak without blocking the Tkinter event loop.

    Completion is signalled via speech_done_event and awaited from the
    calling (assistant) thread, not the main thread — so the UI stays
    responsive while speech plays.
    """
    speech_done_event.clear()
    safe_ui(_do_update_border_color, "speaking")
    safe_ui(_do_update_status, "✦ Speaking ✦")
    safe_ui(_do_append_response, f"Luna: {text}", "luna")

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
            if player == "ffplay":
                cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", wav_path]
            else:
                cmd = [player, wav_path]
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

    safe_ui(_do_update_border_color, "idle")
    safe_ui(_do_update_status, "✦ Idle ✦")


def listen(timeout: int = 6, phrase_limit: int = 12) -> str:
    safe_ui(_do_update_border_color, "listening")
    safe_ui(_do_update_status, "✦ Listening ✦")

    recognizer = sr.Recognizer()
    try:
        mic = sr.Microphone()
    except (OSError, AttributeError) as e:
        logging.error(f"Microphone unavailable: {e}")
        safe_ui(_do_append_response, "System: Microphone not found — use the text input below.", "error")
        safe_ui(_do_update_border_color, "idle")
        safe_ui(_do_update_status, "✦ Idle ✦")
        return ""

    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_limit)
        except sr.WaitTimeoutError:
            safe_ui(_do_update_border_color, "idle")
            safe_ui(_do_update_status, "✦ Idle ✦")
            return ""

    safe_ui(_do_update_border_color, "executing")
    safe_ui(_do_update_status, "✦ Processing ✦")

    try:
        command = recognizer.recognize_google(audio)
        logging.info(f"Voice command: {command}")
        safe_ui(_do_append_response, f"You: {command}", "user")
        return command.lower()
    except sr.UnknownValueError:
        speak("Sorry, I didn't catch that.")
        return ""
    except sr.RequestError as e:
        logging.error(f"Speech recognition API error: {e}")
        speak("Network error. Please try again.")
        return ""


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
    safe_ui(_do_update_border_color, "executing")
    safe_ui(_do_update_status, "✦ Executing ✦")

    if not WEATHER_API_KEY:
        speak("Weather API key is not configured. Please add it to config.py or set the WEATHER_API_KEY environment variable.")
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
    safe_ui(_do_update_border_color, "executing")
    safe_ui(_do_update_status, "✦ Executing ✦")
    speak(f"Based on your configuration, you are located in {HOME_CITY}.")


def play_video_auto(video_name: str) -> bool:
    if not video_name:
        speak("Please specify the video name.")
        return False

    if check_online_status():
        try:
            import yt_dlp
            speak(f"Searching for {video_name} online.")
            ydl_opts = {"quiet": True, "no_warnings": True, "extract_flat": True,
                        "default_search": "ytsearch1", "noplaylist": True}
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
    safe_ui(_do_update_border_color, "executing")
    safe_ui(_do_update_status, "✦ Executing ✦")
    if "stop" in command:
        # Videos open in an external browser/player, so Luna can't stop
        # them programmatically — be honest about that instead of
        # pretending a no-op pygame mixer call did something.
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
    safe_ui(_do_update_border_color, "executing")
    safe_ui(_do_update_status, "✦ Executing ✦")
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
    """Requires spoken confirmation before shutdown/restart/sleep executes."""
    global _pending_system_action
    safe_ui(_do_update_border_color, "confirm")
    safe_ui(_do_update_status, "✦ Confirm? ✦")

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
        return "I'm not sure how to help with that yet. You can add a Gemini API key to config.py to enable AI-powered answers."
    try:
        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
            params={"key": GEMINI_API_KEY},
            headers={"content-type": "application/json"},
            json={
                "contents": [{"role": "user", "parts": [{"text": question}]}],
                "systemInstruction": {"parts": [{"text": GEMINI_SYSTEM_PROMPT}]},
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
    handlers can read it via _get_shared_frame() without opening a
    second, competing VideoCapture."""
    global _last_greeted_name, _last_greeted_time, _latest_frame

    if not (_FACE_MODELS_READY):
        safe_ui(_do_update_face_status, "Camera: face models unavailable — check opencv-python is installed and internet is available for first-run model download")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        safe_ui(_do_update_face_status, "Camera: no webcam found")
        logging.warning("Face camera: no webcam found.")
        return

    safe_ui(_do_update_face_status, "Camera: active — watching for faces…")
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

        display_frame = frame.copy()
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(display_frame, (left, top), (right, bottom), (147, 112, 219), 2)

        if frame_count % 3 == 0:
            pil_img = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
            safe_ui(_do_update_face_thumb, pil_img)

        face_count = len(face_locations)
        if face_count == 0:
            safe_ui(_do_update_face_status, "Camera: no face detected")
            _last_greeted_name = None
        else:
            safe_ui(_do_update_face_status, f"Camera: {face_count} face(s) detected")
            if frame_count % RECOGNITION_EVERY_N == 0:
                for encoding in _get_face_embeddings(frame):
                    _handle_face_encoding(encoding)

        time.sleep(0.03)

    cap.release()
    with _face_cam_lock:
        _latest_frame = None
    safe_ui(_do_update_face_status, "Camera: off")
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
            safe_ui(_do_append_response, f"👁 Recognised: {name}", "face")
            threading.Thread(target=speak, args=(greeting,), daemon=True).start()
    elif not _awaiting_face_name.is_set():
        _awaiting_face_name.set()
        safe_ui(_do_append_response, "👁 Unknown face detected — asking for name…", "face")
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
                safe_ui(_do_append_response, f"👁 New face registered: {name_said}", "face")
                speak(f"Nice to meet you, {name_said}! I'll remember you from now on.")
            else:
                speak("I didn't catch a name. No worries, you can tell me later.")
        else:
            speak("No problem, feel free to tell me your name any time.")
    finally:
        _awaiting_face_name.clear()


def start_face_recognition():
    if not (_FACE_MODELS_READY):
        logging.warning("Face recognition skipped — libraries not available.")
        safe_ui(_do_update_face_status, "Camera: face models unavailable")
        return
    t = threading.Thread(target=_face_camera_loop, daemon=True)
    t.start()
    return t


def _handle_who_am_i():
    if not (_FACE_MODELS_READY):
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
    if not (_FACE_MODELS_READY):
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
        safe_ui(_do_append_response, f"👁 Face saved: {name_said}", "face")
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
        safe_ui(_do_append_response, f"👁 Face removed: {name_said}", "face")
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
            "recognise faces, and answer almost any question thanks to my AI brain. "
            "How can I help you today?"
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
        safe_ui(_do_update_border_color, "executing")
        safe_ui(_do_update_status, "✦ Thinking ✦")
        speak(ask_gemini(command))

    return True


def assistant_loop():
    speak("Hello! I am Luna, your mystical virtual assistant. How can I help you today?")
    while _app_running:
        command = listen()
        if not _app_running:
            break
        if not process_command(command):
            break
        if current_mode != "idle":
            safe_ui(_do_update_border_color, "idle")
            safe_ui(_do_update_status, "✦ Idle ✦")


def start_assistant():
    t = threading.Thread(target=assistant_loop, daemon=True)
    t.start()
    return t


def handle_text_input(event=None):
    raw = text_input.get().strip()
    if not raw or raw == "Type a command or question…":
        return
    text_input.delete(0, "end")
    safe_ui(_do_append_response, f"You: {raw}", "user")
    threading.Thread(target=process_command, args=(raw.lower(),), daemon=True).start()


send_button.config(command=handle_text_input)
text_input.bind("<Return>", handle_text_input)


def on_close():
    global _app_running
    _app_running = False
    _stop_speech_flag.set()
    speech_done_event.set()
    logging.info("Luna window closed cleanly.")
    window.destroy()


window.protocol("WM_DELETE_WINDOW", on_close)

if __name__ == "__main__":
    start_face_recognition()
    start_assistant()
    try:
        window.mainloop()
    except Exception as e:
        logging.critical(f"Main window loop crashed: {e}")