import asyncio
import threading
import json
import os
import queue
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path

import sounddevice as sd
from google import genai
from google.genai import types
from agents.router import get_agent_router
from automation.scheduler import start_automation_services
from core.config import get_config
from core.logging_utils import setup_logging
from ui import JarvisUI
from memory.context_manager import get_context_manager
from memory.memory_manager import (
    load_memory, update_memory, format_memory_for_prompt,
    should_extract_memory, extract_memory
)

from actions.flight_finder     import flight_finder
from actions.open_app          import open_app
from actions.weather_report    import weather_action
from actions.send_message      import send_message
from actions.reminder          import reminder
from actions.computer_settings import computer_settings
from actions.screen_processor  import screen_process
from actions.youtube_video     import youtube_video
from actions.cmd_control       import cmd_control
from actions.desktop           import desktop_control
from actions.browser_control   import browser_control
from actions.instagram_dm      import instagram_dm
from actions.instagram_ai      import instagram_auto_mode, instagram_read_all, instagram_reply_all
from actions.file_controller   import file_controller
from actions.code_helper       import code_helper
from actions.dev_agent         import dev_agent
from actions.telegram_tool     import telegram_read, telegram_send
from actions.web_search        import web_search as web_search_action
from actions.computer_control  import computer_control
from actions.game_updater      import game_updater
from actions.learn             import learn


def get_base_dir():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent


BASE_DIR        = get_base_dir()
API_CONFIG_PATH = BASE_DIR / "config" / "api_keys.json"
PROMPT_PATH     = BASE_DIR / "core" / "prompt.txt"
AUTOSTART_CMD_PATH = BASE_DIR / "config" / "jarvis_autostart.cmd"
WINDOWS_STARTUP_DIR = (
    Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming")))
    / "Microsoft"
    / "Windows"
    / "Start Menu"
    / "Programs"
    / "Startup"
)
WINDOWS_STARTUP_VBS_PATH = WINDOWS_STARTUP_DIR / "JARVIS Mark XXXV.vbs"
APP_CONFIG          = get_config(BASE_DIR)
LIVE_MODEL          = APP_CONFIG.models.gemini_live_model
CHANNELS            = 1
SEND_SAMPLE_RATE    = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE          = 4800   # 0.2s at 24000Hz — smooth playback


class _BufferedAudioPlayer:

    def __init__(self, samplerate: int, channels: int, blocksize: int):
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=8192)
        self._pending = bytearray()
        self._closed = False
        self._stream = sd.RawOutputStream(
            samplerate=samplerate,
            channels=channels,
            dtype="int16",
            blocksize=blocksize,
            latency="high",
            callback=self._callback,
        )

    def _callback(self, outdata, frames, time_info, status):
        view = memoryview(outdata).cast("B")
        filled = 0

        while filled < len(view):
            if self._pending:
                take = min(len(self._pending), len(view) - filled)
                view[filled:filled + take] = self._pending[:take]
                del self._pending[:take]
                filled += take
                continue

            try:
                chunk = self._queue.get_nowait()
            except queue.Empty:
                break

            self._pending.extend(chunk)

        if filled < len(view):
            view[filled:] = b"\x00" * (len(view) - filled)

    def start(self):
        self._stream.start()

    def write(self, chunk: bytes):
        if self._closed:
            return
        try:
            self._queue.put(chunk, timeout=0.1)
        except queue.Full:
            pass

    def clear(self):
        self._pending.clear()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def has_pending_audio(self) -> bool:
        return bool(self._pending) or not self._queue.empty()

    def stop(self):
        self._closed = True
        self._stream.stop()
        self._stream.close()


def _get_api_key() -> str:
    if APP_CONFIG.gemini_api_key:
        return APP_CONFIG.gemini_api_key
    with open(API_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["gemini_api_key"]


def _load_system_prompt() -> str:
    try:
        return PROMPT_PATH.read_text(encoding="utf-8")
    except Exception:
        return (
            "You are JARVIS, Tony Stark's AI assistant. "
            "Be concise, direct, and always use the provided tools to complete tasks. "
            "Never simulate or guess results — always call the appropriate tool."
        )


# ── Hafıza ────────────────────────────────────────────────────────────────────
_last_memory_input = ""

# ── Chat History (last session) ───────────────────────────────────────────────
_CHAT_HISTORY_PATH = BASE_DIR / "memory" / "last_chat.json"
_MAX_CHAT_TURNS    = 20   # how many turns to remember


def _save_chat_turn(user: str, jarvis: str) -> None:
    """Append one turn to last_chat.json, keep only last _MAX_CHAT_TURNS."""
    try:
        _CHAT_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        history: list = []
        if _CHAT_HISTORY_PATH.exists():
            try:
                history = json.loads(_CHAT_HISTORY_PATH.read_text(encoding="utf-8"))
            except Exception:
                history = []
        history.append({
            "user":   user,
            "jarvis": jarvis,
            "time":   datetime.now().strftime("%Y-%m-%d %H:%M")
        })
        history = history[-_MAX_CHAT_TURNS:]
        _CHAT_HISTORY_PATH.write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as e:
        print(f"[ChatHistory] Save failed: {e}")


def _load_chat_history_for_prompt() -> str:
    """Return last session chat as a formatted string for the system prompt."""
    try:
        if not _CHAT_HISTORY_PATH.exists():
            return ""
        history = json.loads(_CHAT_HISTORY_PATH.read_text(encoding="utf-8"))
        if not history:
            return ""
        lines = ["[LAST SESSION CONVERSATION]"]
        for turn in history:
            lines.append(f"User: {turn.get('user', '')}")
            lines.append(f"Jarvis: {turn.get('jarvis', '')}")
        lines.append("[END OF LAST SESSION]")
        return "\n".join(lines)
    except Exception:
        return ""


def _get_launch_command() -> str:
    if getattr(sys, "frozen", False):
        executable = Path(sys.executable)
        return (
            "@echo off\r\n"
            "timeout /t 15 /nobreak >nul\r\n"
            f'cd /d "{executable.parent}"\r\n'
            f'start "" "{executable}"\r\n'
        )

    python_exe = Path(sys.executable)
    if python_exe.name.lower() == "python.exe":
        pythonw = python_exe.with_name("pythonw.exe")
        if pythonw.exists():
            python_exe = pythonw

    return (
        "@echo off\r\n"
        "timeout /t 15 /nobreak >nul\r\n"
        f'cd /d "{BASE_DIR}"\r\n'
        f'"{python_exe}" "{BASE_DIR / "main.py"}"\r\n'
    )


def ensure_windows_startup() -> tuple[bool, str]:
    try:
        AUTOSTART_CMD_PATH.parent.mkdir(parents=True, exist_ok=True)
        WINDOWS_STARTUP_DIR.mkdir(parents=True, exist_ok=True)

        AUTOSTART_CMD_PATH.write_text(_get_launch_command(), encoding="utf-8")

        launcher = str(AUTOSTART_CMD_PATH).replace('"', '""')
        WINDOWS_STARTUP_VBS_PATH.write_text(
            'Set shell = CreateObject("WScript.Shell")\r\n'
            f'shell.Run Chr(34) & "{launcher}" & Chr(34), 0, False\r\n',
            encoding="utf-8"
        )
        return True, str(WINDOWS_STARTUP_VBS_PATH)
    except Exception as exc:
        return False, str(exc)


def _build_startup_greeting_prompt() -> str:
    hour = datetime.now().hour
    if hour < 12:
        salutation = "صباح الخير"
    elif hour < 18:
        salutation = "نهارك سعيد"
    else:
        salutation = "مساء الخير"

    return (
        "Reply with one short spoken sentence in Moroccan Darija. "
        f"Start with '{salutation}', welcome the user back, "
        "and say JARVIS is ready."
    )


def _normalize_local_command_text(text: str) -> str:
    normalized = (text or "").lower().replace("j.a.r.v.i.s", "jarvis")
    normalized = normalized.replace("\u200f", " ").replace("\u200e", " ")
    normalized = re.sub(r"[^\w\u0600-\u06FF]+", " ", normalized, flags=re.UNICODE)
    return re.sub(r"\s+", " ", normalized).strip()


def _contains_wake_word(text: str) -> bool:
    tokens = set(_normalize_local_command_text(text).split())
    return "jarvis" in tokens or "\u062c\u0627\u0631\u0641\u064a\u0633" in tokens


def _is_wake_only_command(text: str) -> bool:
    normalized = _normalize_local_command_text(text)
    return normalized in {"jarvis", "\u062c\u0627\u0631\u0641\u064a\u0633"}


def _is_hide_command(text: str) -> bool:
    normalized = _normalize_local_command_text(text)
    return normalized in {
        "hide",
        "jarvis hide",
        "hide jarvis",
        "\u0627\u062e\u062a\u0641\u064a",
        "\u0627\u062e\u062a\u0641\u0649",
        "\u062c\u0627\u0631\u0641\u064a\u0633 \u0627\u062e\u062a\u0641\u064a",
        "\u062c\u0627\u0631\u0641\u064a\u0633 \u0627\u062e\u062a\u0641\u0649",
        "\u0627\u062e\u062a\u0641\u064a \u062c\u0627\u0631\u0641\u064a\u0633",
        "\u0627\u062e\u062a\u0641\u0649 \u062c\u0627\u0631\u0641\u064a\u0633",
    }


def _iter_input_devices() -> list[tuple[int, dict]]:
    try:
        devices = sd.query_devices()
    except Exception:
        return []

    results: list[tuple[int, dict]] = []
    for idx, device in enumerate(devices):
        try:
            if int(device.get("max_input_channels", 0) or 0) > 0:
                results.append((idx, device))
        except Exception:
            continue
    return results


def _load_audio_preferences() -> tuple[int | str | None, str | None]:
    try:
        with open(API_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return None, None

    preferred_index = None
    if "input_device_index" in cfg:
        preferred_index = cfg.get("input_device_index")
    elif "microphone_device_index" in cfg:
        preferred_index = cfg.get("microphone_device_index")

    preferred_name = None
    if "input_device_name" in cfg:
        preferred_name = cfg.get("input_device_name")
    elif "microphone_device_name" in cfg:
        preferred_name = cfg.get("microphone_device_name")
    return preferred_index, preferred_name


def _can_open_input_device(device_index: int) -> bool:
    try:
        with sd.InputStream(
            samplerate=SEND_SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=CHUNK_SIZE,
            device=device_index,
            callback=lambda indata, frames, time_info, status: None,
        ):
            return True
    except Exception:
        return False


def _select_input_device() -> tuple[int | None, str]:
    devices = _iter_input_devices()
    if not devices:
        return None, "default input device"

    preferred_index, preferred_name = _load_audio_preferences()
    input_indices = {idx for idx, _ in devices}

    if isinstance(preferred_index, str) and preferred_index.isdigit():
        preferred_index = int(preferred_index)

    if isinstance(preferred_index, int) and preferred_index in input_indices and _can_open_input_device(preferred_index):
        chosen_name = next(
            (str(device.get("name", f"input-{preferred_index}")) for idx, device in devices if idx == preferred_index),
            f"input-{preferred_index}",
        )
        return preferred_index, chosen_name

    if preferred_name:
        wanted = str(preferred_name).strip().lower()
        for idx, device in devices:
            name = str(device.get("name", ""))
            if wanted and wanted in name.lower() and _can_open_input_device(idx):
                return idx, name

    default_input = None
    try:
        default_device = sd.default.device
        if isinstance(default_device, (list, tuple)) and default_device:
            default_input = default_device[0]
        elif isinstance(default_device, int):
            default_input = default_device
    except Exception:
        default_input = None

    bad_tokens = (
        "stereo mix",
        "steam streaming",
        "speaker",
        "output",
        "virtual",
    )
    ranked: list[tuple[int, int, str]] = []

    for idx, device in devices:
        name = str(device.get("name", ""))
        lowered = name.lower()
        score = 0

        if "microphone array" in lowered:
            score += 120
        if "headset" in lowered:
            score += 40
        if "microphone" in lowered:
            score += 60
        if "mic" in lowered:
            score += 15
        if idx == default_input:
            score += 10
        if any(token in lowered for token in bad_tokens):
            score -= 200

        if _can_open_input_device(idx):
            ranked.append((score, idx, name))

    if not ranked:
        return None, "default input device"

    ranked.sort(reverse=True)
    best_score, best_idx, best_name = ranked[0]
    if best_score < 0 and default_input in input_indices:
        fallback_name = next(
            (str(device.get("name", f"input-{default_input}")) for idx, device in devices if idx == default_input),
            f"input-{default_input}",
        )
        return default_input, fallback_name

    return best_idx, best_name


def _update_memory_async(user_text: str, jarvis_text: str) -> None:
    global _last_memory_input

    user_text   = (user_text   or "").strip()
    jarvis_text = (jarvis_text or "").strip()

    if len(user_text) < 5 or user_text == _last_memory_input:
        return
    _last_memory_input = user_text

    try:
        get_context_manager(APP_CONFIG).ingest_conversation(user_text, jarvis_text, platform="main", user_id="user")
    except Exception as e:
        if "429" not in str(e):
            print(f"[Memory] warning: {e}")


# ── Tool declarations ─────────────────────────────────────────────────────────
TOOL_DECLARATIONS = [
    {
        "name": "open_app",
        "description": (
            "Opens any application on the Windows computer. "
            "Use this whenever the user asks to open, launch, or start any app, "
            "website, or program. Always call this tool — never just say you opened it."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "app_name": {
                    "type": "STRING",
                    "description": "Exact name of the application (e.g. 'WhatsApp', 'Chrome', 'Spotify')"
                }
            },
            "required": ["app_name"]
        }
    },
    {
        "name": "web_search",
        "description": "Searches the web for any information.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query":  {"type": "STRING", "description": "Search query"},
                "mode":   {"type": "STRING", "description": "search (default) or compare"},
                "items":  {"type": "ARRAY", "items": {"type": "STRING"}, "description": "Items to compare"},
                "aspect": {"type": "STRING", "description": "price | specs | reviews"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "weather_report",
        "description": "Gets real-time weather information for a city.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "city": {"type": "STRING", "description": "City name"}
            },
            "required": ["city"]
        }
    },
    {
        "name": "send_message",
        "description": "Sends a text message via WhatsApp, Telegram, or other messaging platform.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "receiver":     {"type": "STRING", "description": "Recipient contact name"},
                "message_text": {"type": "STRING", "description": "The message to send"},
                "platform":     {"type": "STRING", "description": "Platform: WhatsApp, Telegram, etc."}
            },
            "required": ["receiver", "message_text", "platform"]
        }
    },
    {
        "name": "reminder",
        "description": "Sets a timed reminder using Windows Task Scheduler.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "date":    {"type": "STRING", "description": "Date in YYYY-MM-DD format"},
                "time":    {"type": "STRING", "description": "Time in HH:MM format (24h)"},
                "message": {"type": "STRING", "description": "Reminder message text"}
            },
            "required": ["date", "time", "message"]
        }
    },
    {
        "name": "youtube_video",
        "description": (
            "Controls YouTube. Use for: playing videos, summarizing a video's content, "
            "getting video info, or showing trending videos."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "play | summarize | get_info | trending (default: play)"},
                "query":  {"type": "STRING", "description": "Search query for play action"},
                "save":   {"type": "BOOLEAN", "description": "Save summary to Notepad (summarize only)"},
                "region": {"type": "STRING", "description": "Country code for trending e.g. TR, US"},
                "url":    {"type": "STRING", "description": "Video URL for get_info action"},
            },
            "required": []
        }
    },
    {
        "name": "screen_process",
        "description": (
            "Captures and analyzes the screen, webcam, local image file, or local video file. "
            "MUST be called when user asks what is on screen, what you see, "
            "analyze my screen, look at camera, inspect a photo, or inspect a video. "
            "You have NO visual ability without this tool. "
            "If the user gives a local file path, pass it in path. "
            "After calling this tool, stay SILENT - the vision module speaks directly."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "angle": {"type": "STRING", "description": "'screen', 'camera', 'image', or 'video'. Default: 'screen'"},
                "path":  {"type": "STRING", "description": "Optional local file path for image/video analysis"},
                "frame_count": {"type": "INTEGER", "description": "Optional number of sampled frames for video analysis (default: 3)"},
                "text":  {"type": "STRING", "description": "The question or instruction about the visual input"}
            },
            "required": ["text"]
        }
    },
    {
        "name": "computer_settings",
        "description": (
            "Controls the computer: volume, brightness, window management, keyboard shortcuts, "
            "typing text on screen, closing apps, fullscreen, dark mode, WiFi, restart, shutdown, "
            "scrolling, tab management, zoom, screenshots, lock screen, refresh/reload page, "
            "Start menu, search, notifications, quick settings, clipboard history, and JARVIS show/hide. "
            "Use for ANY single computer control command on the user's PC. "
            "Prefer this tool over chat when a direct system action is requested. "
            "NEVER route to agent_task."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "The action to perform"},
                "description": {"type": "STRING", "description": "Natural language description of what to do"},
                "value":       {"type": "STRING", "description": "Optional value: volume level, text to type, etc."}
            },
            "required": []
        }
    },
    {
        "name": "browser_control",
        "description": (
            "Controls the web browser. Use for: opening websites, searching the web, "
            "clicking elements, filling forms, scrolling, any web-based task. "
            "Do NOT use for Instagram direct messages. Use instagram_dm instead."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "go_to | search | click | type | scroll | fill_form | smart_click | smart_type | get_text | press | close"},
                "url":         {"type": "STRING", "description": "URL for go_to action"},
                "query":       {"type": "STRING", "description": "Search query for search action"},
                "selector":    {"type": "STRING", "description": "CSS selector for click/type"},
                "text":        {"type": "STRING", "description": "Text to click or type"},
                "description": {"type": "STRING", "description": "Element description for smart_click/smart_type"},
                "direction":   {"type": "STRING", "description": "up or down for scroll"},
                "key":         {"type": "STRING", "description": "Key name for press action"},
                "incognito":   {"type": "BOOLEAN", "description": "Open in private/incognito mode"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "instagram_dm",
        "description": (
            "THE ONLY tool for Instagram direct messages in Opera. "
            "Use it to open the inbox, read the latest thread, send an exact reply, "
            "or generate and send an AI reply from the logged-in Instagram account."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {
                    "type": "STRING",
                    "description": "open_inbox | read_latest | reply_latest | reply_with_ai (default: reply_with_ai)"
                },
                "thread": {
                    "type": "STRING",
                    "description": "Optional thread/contact name to match. If omitted, use the latest thread."
                },
                "reply_text": {
                    "type": "STRING",
                    "description": "Exact reply text for reply_latest, or optional override text for reply_with_ai."
                },
                "instructions": {
                    "type": "STRING",
                    "description": "Optional style instructions for generated replies."
                },
            },
            "required": ["action"]
        }
    },
    {
        "name": "instagram_read_all",
        "description": (
            "Reads Instagram inbox threads from the logged-in session using Playwright. "
            "Use for reading all DMs or unread conversations."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "unread_only": {"type": "BOOLEAN", "description": "Read unread chats only"},
                "limit": {"type": "INTEGER", "description": "Maximum number of threads to inspect"}
            },
            "required": []
        }
    },
    {
        "name": "instagram_reply_all",
        "description": (
            "Reads Instagram DMs and replies automatically with contextual AI-generated replies. "
            "Supports personality selection and bulk reply mode."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "unread_only": {"type": "BOOLEAN", "description": "Reply only to unread chats"},
                "bulk_limit": {"type": "INTEGER", "description": "Maximum chats to reply to"},
                "personality": {"type": "STRING", "description": "warm | professional | jarvis | flirty"},
                "smart_filter": {"type": "BOOLEAN", "description": "Ignore spam-like accounts"}
            },
            "required": []
        }
    },
    {
        "name": "instagram_auto_mode",
        "description": (
            "Starts or stops Instagram auto-reply background mode with human-like delays."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "mode": {"type": "STRING", "description": "start or stop"},
                "personality": {"type": "STRING", "description": "Reply personality preset"},
                "unread_only": {"type": "BOOLEAN", "description": "Only reply to unread chats"}
            },
            "required": []
        }
    },
    {
        "name": "telegram_send",
        "description": "Sends a Telegram message or media using the configured Telegram account or bot.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "chat": {"type": "STRING", "description": "Chat ID, username, or contact"},
                "text": {"type": "STRING", "description": "Message text"},
                "media_path": {"type": "STRING", "description": "Optional local media file path"}
            },
            "required": ["chat", "text"]
        }
    },
    {
        "name": "telegram_read",
        "description": "Reads Telegram chats from the configured account.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "limit": {"type": "INTEGER", "description": "Maximum number of chats to read"},
                "unread_only": {"type": "BOOLEAN", "description": "Read only chats with unread messages"}
            },
            "required": []
        }
    },
    {
        "name": "file_controller",
        "description": "Manages files and folders: list, create, delete, move, copy, rename, read, write, find, disk usage.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "list | create_file | create_folder | delete | move | copy | rename | read | write | find | largest | disk_usage | organize_desktop | info"},
                "path":        {"type": "STRING", "description": "File/folder path or shortcut: desktop, downloads, documents, home"},
                "destination": {"type": "STRING", "description": "Destination path for move/copy"},
                "new_name":    {"type": "STRING", "description": "New name for rename"},
                "content":     {"type": "STRING", "description": "Content for create_file/write"},
                "name":        {"type": "STRING", "description": "File name to search for"},
                "extension":   {"type": "STRING", "description": "File extension to search (e.g. .pdf)"},
                "count":       {"type": "INTEGER", "description": "Number of results for largest"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "cmd_control",
        "description": (
            "Runs CMD/terminal commands via natural language: disk space, processes, "
            "system info, network, find files, PowerShell tasks, or anything appropriate for the command line. "
            "Use it when the user wants real terminal work done on their machine."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "task":    {"type": "STRING", "description": "Natural language description of what to do"},
                "visible": {"type": "BOOLEAN", "description": "Open visible CMD window. Default: true"},
                "command": {"type": "STRING", "description": "Optional: exact command if already known"},
            },
            "required": ["task"]
        }
    },
    {
        "name": "desktop_control",
        "description": "Controls the desktop: wallpaper, organize, clean, list, stats.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "wallpaper | wallpaper_url | organize | clean | list | stats | task"},
                "path":   {"type": "STRING", "description": "Image path for wallpaper"},
                "url":    {"type": "STRING", "description": "Image URL for wallpaper_url"},
                "mode":   {"type": "STRING", "description": "by_type or by_date for organize"},
                "task":   {"type": "STRING", "description": "Natural language desktop task"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "code_helper",
        "description": "Writes, edits, explains, runs, or builds code files.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "write | edit | explain | run | build | auto (default: auto)"},
                "description": {"type": "STRING", "description": "What the code should do or what change to make"},
                "language":    {"type": "STRING", "description": "Programming language (default: python)"},
                "output_path": {"type": "STRING", "description": "Where to save the file"},
                "file_path":   {"type": "STRING", "description": "Path to existing file for edit/explain/run/build"},
                "code":        {"type": "STRING", "description": "Raw code string for explain"},
                "args":        {"type": "STRING", "description": "CLI arguments for run/build"},
                "timeout":     {"type": "INTEGER", "description": "Execution timeout in seconds (default: 30)"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "dev_agent",
        "description": "Builds complete multi-file projects from scratch: plans, writes files, installs deps, opens VSCode, runs and fixes errors.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "description":  {"type": "STRING", "description": "What the project should do"},
                "language":     {"type": "STRING", "description": "Programming language (default: python)"},
                "project_name": {"type": "STRING", "description": "Optional project folder name"},
                "timeout":      {"type": "INTEGER", "description": "Run timeout in seconds (default: 30)"},
            },
            "required": ["description"]
        }
    },
    {
        "name": "agent_task",
        "description": (
            "Executes complex multi-step tasks requiring multiple different tools. "
            "Examples: 'research X and save to file', 'find and organize files'. "
            "DO NOT use for single commands. NEVER use for Steam/Epic — use game_updater."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "goal":     {"type": "STRING", "description": "Complete description of what to accomplish"},
                "priority": {"type": "STRING", "description": "low | normal | high (default: normal)"}
            },
            "required": ["goal"]
        }
    },
    {
        "name": "computer_control",
        "description": (
            "Direct computer control for the user's machine: type, click, drag, hotkeys, scroll, move mouse, "
            "clipboard, screenshots, focus windows, and find elements on screen. "
            "Use it whenever precise on-screen interaction is the fastest path."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "type | smart_type | click | double_click | right_click | hotkey | press | scroll | move | drag | copy | paste | screenshot | wait | wait_image | clear_field | focus_window | screen_size | mouse_position | screen_find | screen_click | random_data | user_data"},
                "text":        {"type": "STRING", "description": "Text to type or paste"},
                "x":           {"type": "INTEGER", "description": "X coordinate"},
                "y":           {"type": "INTEGER", "description": "Y coordinate"},
                "x1":          {"type": "INTEGER", "description": "Start X coordinate for drag"},
                "y1":          {"type": "INTEGER", "description": "Start Y coordinate for drag"},
                "x2":          {"type": "INTEGER", "description": "End X coordinate for drag"},
                "y2":          {"type": "INTEGER", "description": "End Y coordinate for drag"},
                "keys":        {"type": "STRING", "description": "Key combination e.g. 'ctrl+c'"},
                "key":         {"type": "STRING", "description": "Single key e.g. 'enter'"},
                "direction":   {"type": "STRING", "description": "up | down | left | right"},
                "amount":      {"type": "INTEGER", "description": "Scroll amount (default: 3)"},
                "seconds":     {"type": "NUMBER",  "description": "Seconds to wait"},
                "timeout":     {"type": "INTEGER", "description": "Timeout in seconds for wait_image"},
                "duration":    {"type": "NUMBER",  "description": "Duration for move/drag actions"},
                "button":      {"type": "STRING",  "description": "Mouse button for click/drag: left | right | middle"},
                "title":       {"type": "STRING",  "description": "Window title for focus_window"},
                "description": {"type": "STRING",  "description": "Element description for screen_find/screen_click"},
                "type":        {"type": "STRING",  "description": "Data type for random_data"},
                "field":       {"type": "STRING",  "description": "Field for user_data: name|email|city"},
                "clear_first": {"type": "BOOLEAN", "description": "Clear field before typing (default: true)"},
                "path":        {"type": "STRING",  "description": "Save path for screenshot"},
                "image":       {"type": "STRING",  "description": "Image path for click/wait_image"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "game_updater",
        "description": (
            "THE ONLY tool for ANY Steam or Epic Games request. "
            "Use for: installing, downloading, updating games, listing installed games, "
            "checking download status, scheduling updates. "
            "ALWAYS call directly for any Steam/Epic/game request. "
            "NEVER use agent_task, browser_control, or web_search for Steam/Epic."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":    {"type": "STRING",  "description": "update | install | list | download_status | schedule | cancel_schedule | schedule_status (default: update)"},
                "platform":  {"type": "STRING",  "description": "steam | epic | both (default: both)"},
                "game_name": {"type": "STRING",  "description": "Game name (partial match supported)"},
                "app_id":    {"type": "STRING",  "description": "Steam AppID for install (optional)"},
                "hour":      {"type": "INTEGER", "description": "Hour for scheduled update 0-23 (default: 3)"},
                "minute":    {"type": "INTEGER", "description": "Minute for scheduled update 0-59 (default: 0)"},
                "shutdown_when_done": {"type": "BOOLEAN", "description": "Shut down PC when download finishes"},
            },
            "required": []
        }
    },
    {
        "name": "flight_finder",
        "description": "Searches Google Flights and speaks the best options.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "origin":      {"type": "STRING",  "description": "Departure city or airport code"},
                "destination": {"type": "STRING",  "description": "Arrival city or airport code"},
                "date":        {"type": "STRING",  "description": "Departure date (any format)"},
                "return_date": {"type": "STRING",  "description": "Return date for round trips"},
                "passengers":  {"type": "INTEGER", "description": "Number of passengers (default: 1)"},
                "cabin":       {"type": "STRING",  "description": "economy | premium | business | first"},
                "save":        {"type": "BOOLEAN", "description": "Save results to Notepad"},
            },
            "required": ["origin", "destination", "date"]
        }
    },
    {
        "name": "save_memory",
        "description": (
            "Save an important personal fact about the user to long-term memory. "
            "Call this silently whenever the user reveals something worth remembering: "
            "name, age, city, job, preferences, hobbies, relationships, projects, or future plans. "
            "Do NOT call for: weather, reminders, searches, or one-time commands. "
            "Do NOT announce that you are saving — just call it silently. "
            "Values must be in English regardless of the conversation language."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "category": {
                    "type": "STRING",
                    "description": (
                        "identity — name, age, birthday, city, job, language, nationality | "
                        "preferences — favorite food/color/music/film/game/sport, hobbies | "
                        "projects — active projects, goals, things being built | "
                        "relationships — friends, family, partner, colleagues | "
                        "wishes — future plans, things to buy, travel dreams | "
                        "notes — habits, schedule, anything else worth remembering"
                    )
                },
                "key":   {"type": "STRING", "description": "Short snake_case key (e.g. name, favorite_food, sister_name)"},
                "value": {"type": "STRING", "description": "Concise value in English (e.g. Fatih, pizza, older sister)"},
            },
            "required": ["category", "key", "value"]
        }
    },
{
    "name": "learn",
    "description": "Learn a topic and save it to knowledge base",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "topic": {"type": "STRING"}
        },
        "required": ["topic"]
    }
}
]


class JarvisLive:

    def __init__(self, ui: JarvisUI):
        self.ui             = ui
        self.config         = APP_CONFIG
        self.context_manager = get_context_manager(self.config)
        self.agent_router   = get_agent_router(self.context_manager, self.config)
        self.session        = None
        self.audio_in_queue = None
        self.out_queue      = None
        self._loop          = None
        self._is_speaking   = False
        self._startup_greeted = False
        self._hide_after_startup_greeting = False
        self._suppress_response_audio = False
        self._response_turn_active = False
        self._speaking_lock = threading.Lock()
        self._camera_watch_thread: threading.Thread | None = None
        self._camera_watch_stop = threading.Event()
        self._camera_watch_lock = threading.Lock()
        self._camera_watch_active = False
        self._camera_request_lock = threading.Lock()
        self._input_device_index, self._input_device_name = _select_input_device()
        self.ui.on_text_command = self._on_text_command
        self.ui.on_camera_shortcut = self._on_camera_shortcut

    def _on_text_command(self, text: str):
        if self._handle_local_command(text):
            return
        payload = self.context_manager.build_turn_payload(text)
        if not self._loop or not self.session:
            try:
                local_reply = self.agent_router.execute(payload, parallel=False).output
                self.ui.write_log(f"Jarvis: {local_reply}")
            except Exception as exc:
                self.ui.write_log(f"ERR: Local fallback failed: {exc}")
            return
        asyncio.run_coroutine_threadsafe(
            self.session.send_client_content(
                turns={"parts": [{"text": payload}]},
                turn_complete=True
            ),
            self._loop
        )

    def _on_camera_shortcut(self):
        with self._camera_watch_lock:
            if self._camera_watch_active:
                self._camera_watch_active = False
                self._camera_watch_stop.set()
                self.ui.write_log("SYS: Camera live mode stopping...")
                return

            self._camera_watch_active = True
            self._camera_watch_stop.clear()
            self._camera_watch_thread = threading.Thread(
                target=self._camera_watch_loop,
                daemon=True,
                name="JarvisCameraWatch",
            )
            self._camera_watch_thread.start()
            self.ui.write_log("SYS: Camera live mode enabled.")

    def _trigger_camera_vision(
        self,
        *,
        source: str = "camera shortcut",
        prompt: str = "Describe what you see through the camera briefly.",
        mute_audio: bool = False,
        skip_if_busy: bool = False,
        bring_to_front: bool = True,
    ):
        self._hide_after_startup_greeting = False
        if bring_to_front:
            self.ui.show_window()
        self.ui.set_state("PROCESSING")

        def run_camera():
            acquired = (
                self._camera_request_lock.acquire(blocking=False)
                if skip_if_busy else
                self._camera_request_lock.acquire()
            )
            if not acquired:
                return
            try:
                ok = screen_process(
                    parameters={"angle": "camera", "text": prompt, "mute_audio": mute_audio},
                    response=None,
                    player=self.ui,
                    session_memory=None,
                )
                if not ok:
                    self.ui.write_log(f"ERR: Camera vision failed for {source}.")
            finally:
                self._camera_request_lock.release()

        threading.Thread(target=run_camera, daemon=True).start()

    def _camera_watch_loop(self):
        first_pass = True
        try:
            while not self._camera_watch_stop.is_set():
                prompt = (
                    "Describe what you see through the camera briefly."
                    if first_pass else
                    "Check the camera again and describe only meaningful visual changes in one short sentence."
                )
                self._trigger_camera_vision(
                    source="Ctrl+Alt+C",
                    prompt=prompt,
                    mute_audio=not first_pass,
                    skip_if_busy=True,
                    bring_to_front=first_pass,
                )
                first_pass = False
                if self._camera_watch_stop.wait(4.0):
                    break
        finally:
            with self._camera_watch_lock:
                self._camera_watch_active = False
                self._camera_watch_thread = None
                self._camera_watch_stop.clear()
            self.ui.write_log("SYS: Camera live mode stopped.")

    def _handle_local_command(
        self,
        text: str,
        *,
        from_transcript: bool = False,
        announce: bool = True,
    ) -> bool:
        if not text:
            return False

        if _is_hide_command(text):
            if from_transcript:
                self._suppress_response_audio = True
            self._hide_after_startup_greeting = False
            self.ui.hide_to_background()
            if announce:
                self.ui.write_log("SYS: Hidden in background. Say Jarvis or جارفيس to bring it back.")
            return True

        if _is_wake_only_command(text):
            was_hidden = self.ui.window_hidden
            if was_hidden:
                self.ui.show_window()
                if announce:
                    self.ui.write_log("SYS: JARVIS window restored.")
            if from_transcript:
                self._suppress_response_audio = True
            return was_hidden or from_transcript

        if self.ui.window_hidden and _contains_wake_word(text):
            self.ui.show_window()

        return False

    def set_speaking(self, value: bool):
        with self._speaking_lock:
            self._is_speaking = value
        if value:
            self.ui.set_state("SPEAKING")
        elif not self.ui.muted:
            self.ui.set_state("LISTENING")

    def speak(self, text: str):
        if not self._loop or not self.session:
            return
        asyncio.run_coroutine_threadsafe(
            self.session.send_client_content(
                turns={"parts": [{"text": text}]},
                turn_complete=True
            ),
            self._loop
        )

    def speak_error(self, tool_name: str, error: str):
        short = str(error)[:120]
        self.ui.write_log(f"ERR: {tool_name} — {short}")
        self.speak(f"Sir, {tool_name} encountered an error. {short}")

    def _build_config(self) -> types.LiveConnectConfig:
        mem_str    = self.context_manager.build_prompt_context()
        sys_prompt = _load_system_prompt()

        now      = datetime.now()
        time_str = now.strftime("%A, %B %d, %Y — %I:%M %p")
        time_ctx = (
            f"[CURRENT DATE & TIME]\n"
            f"Right now it is: {time_str}\n"
            f"Use this to calculate exact times for reminders.\n\n"
        )

        parts = [time_ctx]
        if mem_str:
            parts.append(mem_str)

        chat_history = _load_chat_history_for_prompt()
        if chat_history:
            parts.append(chat_history)

        parts.append(sys_prompt)

        return types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            output_audio_transcription={},
            input_audio_transcription={},
            system_instruction="\n".join(parts),
            tools=[{"function_declarations": TOOL_DECLARATIONS}],
            session_resumption=types.SessionResumptionConfig(),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Charon"
                    )
                )
            ),
        )

    async def _execute_tool(self, fc) -> types.FunctionResponse:
        name = fc.name
        args = dict(fc.args or {})

        print(f"[JARVIS] Tool call: {name} {args}")
        self.ui.set_state("THINKING")

        # ── save_memory: sessiz, hızlı, Gemini'ye bildirim yok ───────────────
        if name == "save_memory":
            category = args.get("category", "notes")
            key      = args.get("key", "")
            value    = args.get("value", "")
            if key and value:
                self.context_manager.remember_fact(category, key, value, source="tool_call")
                print(f"[Memory] save_memory: {category}/{key} = {value}")
            if not self.ui.muted:
                self.ui.set_state("LISTENING")
            return types.FunctionResponse(
                id=fc.id, name=name,
                response={"result": "ok", "silent": True}
            )

        loop   = asyncio.get_event_loop()
        result = "Done."

        try:
            if name == "open_app":
                r = await loop.run_in_executor(None, lambda: open_app(parameters=args, response=None, player=self.ui))
                result = r or f"Opened {args.get('app_name')}."

            elif name == "weather_report":
                r = await loop.run_in_executor(None, lambda: weather_action(parameters=args, player=self.ui))
                result = r or "Weather delivered."

            elif name == "browser_control":
                r = await loop.run_in_executor(None, lambda: browser_control(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "instagram_dm":
                r = await loop.run_in_executor(None, lambda: instagram_dm(parameters=args, player=self.ui, speak=self.speak))
                result = r or "Done."

            elif name == "instagram_read_all":
                r = await loop.run_in_executor(None, lambda: instagram_read_all(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "instagram_reply_all":
                r = await loop.run_in_executor(None, lambda: instagram_reply_all(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "instagram_auto_mode":
                r = await loop.run_in_executor(None, lambda: instagram_auto_mode(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "file_controller":
                r = await loop.run_in_executor(None, lambda: file_controller(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "send_message":
                r = await loop.run_in_executor(None, lambda: send_message(parameters=args, response=None, player=self.ui, session_memory=None))
                result = r or f"Message sent to {args.get('receiver')}."

            elif name == "telegram_send":
                r = await loop.run_in_executor(None, lambda: telegram_send(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "telegram_read":
                r = await loop.run_in_executor(None, lambda: telegram_read(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "reminder":
                r = await loop.run_in_executor(None, lambda: reminder(parameters=args, response=None, player=self.ui))
                result = r or "Reminder set."

            elif name == "youtube_video":
                r = await loop.run_in_executor(None, lambda: youtube_video(parameters=args, response=None, player=self.ui))
                result = r or "Done."

            elif name == "screen_process":
                threading.Thread(
                    target=screen_process,
                    kwargs={"parameters": args, "response": None,
                            "player": self.ui, "session_memory": None},
                    daemon=True
                ).start()
                result = "Vision module activated. Stay completely silent — vision module will speak directly."

            elif name == "computer_settings":
                r = await loop.run_in_executor(None, lambda: computer_settings(parameters=args, response=None, player=self.ui))
                result = r or "Done."

            elif name == "cmd_control":
                r = await loop.run_in_executor(None, lambda: cmd_control(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "desktop_control":
                r = await loop.run_in_executor(None, lambda: desktop_control(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "code_helper":
                r = await loop.run_in_executor(None, lambda: code_helper(parameters=args, player=self.ui, speak=self.speak))
                result = r or "Done."

            elif name == "dev_agent":
                r = await loop.run_in_executor(None, lambda: dev_agent(parameters=args, player=self.ui, speak=self.speak))
                result = r or "Done."

            elif name == "agent_task":
                routed = await loop.run_in_executor(
                    None,
                    lambda: self.agent_router.execute(args.get("goal", ""), parallel=True),
                )
                result = routed.output

            elif name == "web_search":
                r = await loop.run_in_executor(None, lambda: web_search_action(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "computer_control":
                r = await loop.run_in_executor(None, lambda: computer_control(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "game_updater":
                r = await loop.run_in_executor(None, lambda: game_updater(parameters=args, player=self.ui, speak=self.speak))
                result = r or "Done."

            elif name == "flight_finder":
                r = await loop.run_in_executor(None, lambda: flight_finder(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "learn":
                r = await loop.run_in_executor(None, lambda: learn(parameters=args, player=self.ui))
                result = r or "Learning completed."

            else:
                result = f"Unknown tool: {name}"

        except Exception as e:
            result = f"Tool '{name}' failed: {e}"
            traceback.print_exc()
            self.speak_error(name, e)

        if not self.ui.muted:
            self.ui.set_state("LISTENING")

        print(f"[JARVIS] Tool result: {name} -> {str(result)[:80]}")

        # ── Result: tek cümle söyle, dur ──────────────────────────────────────
        return types.FunctionResponse(
            id=fc.id, name=name,
            response={"result": result}
        )

    async def _send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            await self.session.send_realtime_input(media=msg)

    def _queue_audio_chunk(self, data: bytes):
        if not self.out_queue:
            return
        try:
            self.out_queue.put_nowait({"data": data, "mime_type": "audio/pcm"})
        except asyncio.QueueFull:
            # Drop overflow instead of breaking the audio loop.
            pass

    async def _listen_audio(self):
        print(f"[JARVIS] Mic starting on {self._input_device_name}")
        loop = asyncio.get_event_loop()
        self.ui.write_log(f"SYS: Listening via {self._input_device_name}.")

        def callback(indata, frames, time_info, status):
            with self._speaking_lock:
                jarvis_speaking = self._is_speaking
            if not jarvis_speaking and not self.ui.muted:
                data = indata.tobytes()
                loop.call_soon_threadsafe(self._queue_audio_chunk, data)

        def _stream_kwargs(device_index: int | None) -> dict:
            kwargs = {
                "samplerate": SEND_SAMPLE_RATE,
                "channels": CHANNELS,
                "dtype": "int16",
                "blocksize": CHUNK_SIZE,
                "callback": callback,
            }
            if device_index is not None:
                kwargs["device"] = device_index
            return kwargs

        try:
            try:
                with sd.InputStream(**_stream_kwargs(self._input_device_index)):
                    print(f"[JARVIS] Mic stream open on {self._input_device_name}")
                    while True:
                        await asyncio.sleep(0.1)
            except Exception as first_error:
                if self._input_device_index is None:
                    raise
                print(f"[JARVIS] Mic device failed ({self._input_device_name}): {first_error}")
                self.ui.write_log(
                    f"ERR: Mic device '{self._input_device_name}' failed. Falling back to default input."
                )
                self._input_device_index = None
                self._input_device_name = "default input device"
                with sd.InputStream(**_stream_kwargs(None)):
                    print("[JARVIS] Mic stream open on default input device")
                    while True:
                        await asyncio.sleep(0.1)
        except Exception as e:
            print(f"[JARVIS] Mic error: {e}")
            self.ui.write_log(f"ERR: Microphone stream failed: {e}")
            raise

    async def _receive_audio(self):
        print("[JARVIS] Receive loop started")
        out_buf, in_buf = [], []

        try:
            while True:
                async for response in self.session.receive():

                    if response.data:
                        self._response_turn_active = True
                        self.audio_in_queue.put_nowait(response.data)

                    if response.server_content:
                        sc = response.server_content

                        if sc.output_transcription and sc.output_transcription.text:
                            self._response_turn_active = True
                            self.set_speaking(True)
                            txt = sc.output_transcription.text.strip()
                            if txt:
                                out_buf.append(txt)

                        if sc.input_transcription and sc.input_transcription.text:
                            txt = sc.input_transcription.text.strip()
                            if txt:
                                in_buf.append(txt)
                                self._handle_local_command(
                                    txt,
                                    from_transcript=True,
                                    announce=False,
                                )

                        if sc.turn_complete:
                            handled_local_turn = self._suppress_response_audio
                            self._response_turn_active = False

                            full_in = " ".join(in_buf).strip()
                            if full_in:
                                self.ui.write_log(f"You: {full_in}")
                                handled_local_turn = (
                                    self._handle_local_command(full_in, from_transcript=True)
                                    or handled_local_turn
                                )
                            in_buf = []

                            full_out = " ".join(out_buf).strip()
                            if full_out and not handled_local_turn:
                                self.ui.write_log(f"Jarvis: {full_out}")
                                if self._hide_after_startup_greeting:
                                    self._hide_after_startup_greeting = False
                                    self.ui.hide_to_background()
                            out_buf = []
                            self._suppress_response_audio = False

                            if (
                                full_in
                                and len(full_in) > 5
                                and not _is_hide_command(full_in)
                                and not _is_wake_only_command(full_in)
                            ):
                                threading.Thread(
                                    target=_update_memory_async,
                                    args=(full_in, full_out),
                                    daemon=True
                                ).start()

                                if full_out:
                                    threading.Thread(
                                        target=_save_chat_turn,
                                        args=(full_in, full_out),
                                        daemon=True
                                    ).start()

                    if response.tool_call:
                        fn_responses = []
                        for fc in response.tool_call.function_calls:
                            print(f"[JARVIS] Executing function: {fc.name}")
                            fr = await self._execute_tool(fc)
                            fn_responses.append(fr)
                        await self.session.send_tool_response(
                            function_responses=fn_responses
                        )
                        # ── Boş turn YOK — bu "Anladım." sorununu yaratıyordu ──

        except Exception as e:
            self._response_turn_active = False
            print(f"[JARVIS] Receive error: {e}")
            traceback.print_exc()
            raise

    async def _play_audio(self):
        print("[JARVIS] Playback loop started")
        player = _BufferedAudioPlayer(
            samplerate=RECEIVE_SAMPLE_RATE,
            channels=CHANNELS,
            blocksize=CHUNK_SIZE,
        )
        player.start()
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(self.audio_in_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    if not self._response_turn_active and not player.has_pending_audio():
                        self.set_speaking(False)
                    continue

                if self._suppress_response_audio:
                    player.clear()
                    continue

                self.set_speaking(True)
                player.write(chunk)
        except Exception as e:
            print(f"[JARVIS] Playback error: {e}")
            raise
        finally:
            self.set_speaking(False)
            player.stop()

    async def _deliver_startup_greeting(self):
        if self._startup_greeted or self.ui.muted or not self.session:
            return

        await asyncio.sleep(2)
        if self._startup_greeted or self.ui.muted or not self.session:
            return

        try:
            await self.session.send_client_content(
                turns={"parts": [{"text": _build_startup_greeting_prompt()}]},
                turn_complete=True
            )
            self._startup_greeted = True
            self._hide_after_startup_greeting = True
        except Exception as e:
            print(f"[JARVIS] Startup greeting failed: {e}")

    async def _keepalive(self):
        """Send a silent ping every 20s to prevent WebSocket timeout."""
        while True:
            await asyncio.sleep(20)
            try:
                if self.session:
                    # Send empty audio chunk — keeps the connection alive
                    await self.session.send_realtime_input(
                        media=types.Blob(data=b"\x00" * 320, mime_type="audio/pcm;rate=16000")
                    )
            except Exception:
                break   # session is dead — let gather() handle reconnect

    async def run(self):
        client = genai.Client(
            api_key=_get_api_key(),
            http_options={"api_version": "v1beta"}
        )

        reconnect_delay = 3

        while True:
            try:
                print("[JARVIS] Connecting...")
                self.ui.set_state("THINKING")
                config = self._build_config()

                async with client.aio.live.connect(model=LIVE_MODEL, config=config) as session:
                    self.session        = session
                    self._loop          = asyncio.get_event_loop()
                    self.audio_in_queue = asyncio.Queue()
                    self.out_queue      = asyncio.Queue(maxsize=10)

                    print("[JARVIS] Connected.")
                    self.ui.set_state("LISTENING")
                    self.ui.write_log("SYS: JARVIS online.")
                    reconnect_delay = 3   # reset backoff on successful connect

                    await asyncio.gather(
                        self._send_realtime(),
                        self._listen_audio(),
                        self._receive_audio(),
                        self._play_audio(),
                        self._deliver_startup_greeting(),
                        self._keepalive(),
                    )

            except Exception as e:
                err = str(e)
                if "1006" in err or "1011" in err or "keepalive" in err.lower() or "abnormal" in err.lower():
                    print(f"[JARVIS] Connection dropped (ping timeout) — reconnecting in {reconnect_delay}s...")
                else:
                    print(f"[JARVIS] Warning: {e}")
                    traceback.print_exc()

            self.set_speaking(False)
            self.ui.set_state("THINKING")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 30)  # backoff: 3→6→12→24→30s max


def main():
    setup_logging(APP_CONFIG, force=True)
    start_automation_services(APP_CONFIG)
    ui = JarvisUI("face.png")
    startup_ready, startup_info = ensure_windows_startup()
    if startup_ready:
        ui.write_log("SYS: Auto-start enabled for Windows startup.")
    else:
        ui.write_log(f"ERR: Auto-start setup failed: {startup_info}")

    def runner():
        ui.wait_for_api_key()
        jarvis = JarvisLive(ui)
        try:
            asyncio.run(jarvis.run())
        except KeyboardInterrupt:
            print("\nShutting down...")

    threading.Thread(target=runner, daemon=True).start()
    ui.root.mainloop()


if __name__ == "__main__":
    main()
