import os
import json
import hashlib
import ipaddress
import secrets
from html import escape
import sys
import time
import threading
import wave
from collections import Counter
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request as UrlRequest, urlopen
from urllib.parse import urlparse

import gradio as gr
import torch
from transformers import AutoModel


def find_project_root() -> Path:
    """Find a directory containing the local Whisper-Pinyin source tree."""
    app_dir = Path(__file__).resolve().parent
    for candidate in (app_dir, *app_dir.parents):
        if (candidate / "whisper").is_dir():
            return candidate
    raise RuntimeError(
        "Could not find the Whisper-Pinyin source tree. Upload the `whisper/` "
        "directory with this Space app."
    )


PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))


MODEL_REPO_ID = os.getenv("WHISPER_PINYIN_MODEL_ID", "walston/whisper-pinyin")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_AUDIO_PATH = (
    Path(__file__).resolve().parent
    / "examples"
    / "example.wav"
)
ANALYTICS_PATH = Path(
    os.getenv("WHISPER_PINYIN_ANALYTICS_PATH", "/tmp/whisper_pinyin_demo_analytics.json")
)
PRIVATE_DAILY_USERS_PATH = Path(
    os.getenv(
        "WHISPER_PINYIN_PRIVATE_DAILY_USERS_PATH",
        "/tmp/whisper_pinyin_private_daily_users.jsonl",
    )
)
ADMIN_EXPORT_TOKEN = os.getenv("ADMIN_EXPORT_TOKEN", "")
UTC_PLUS_8 = timezone(timedelta(hours=8))
ANALYTICS_LOCK = threading.Lock()
APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;500;600;700;800&display=swap');

.gradio-container,
.gradio-container * {
  font-family: Quicksand, Avenir, "Segoe UI", system-ui, -apple-system, sans-serif;
  letter-spacing: 0;
}
"""


def empty_analytics() -> dict:
    return {
        "days": {},
        "visitors": {},
        "sessions": {},
        "total_generations": 0,
        "total_audio_duration": 0.0,
        "total_dwell_seconds": 0.0,
        "total_visits": 0,
        "failed_generations": 0,
        "countries": {},
        "devices": {},
        "sources": {},
    }


def today_utc8() -> str:
    return datetime.now(UTC_PLUS_8).strftime("%Y-%m-%d")


def now_utc8_label() -> str:
    return datetime.now(UTC_PLUS_8).strftime("%Y-%m-%d %H:%M:%S UTC+8")


def read_analytics() -> dict:
    if not ANALYTICS_PATH.exists():
        return empty_analytics()
    try:
        with ANALYTICS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return empty_analytics()

    defaults = empty_analytics()
    defaults.update(data)
    return defaults


def write_analytics(data: dict) -> None:
    ANALYTICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = ANALYTICS_PATH.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2, sort_keys=True)
    tmp_path.replace(ANALYTICS_PATH)


def day_bucket(data: dict, day: str) -> dict:
    days = data.setdefault("days", {})
    bucket = days.setdefault(
        day,
        {
            "users": {},
            "sessions": {},
            "visits": 0,
            "generations": 0,
            "failed_generations": 0,
            "audio_duration": 0.0,
            "dwell_seconds": 0.0,
            "countries": {},
            "devices": {},
            "sources": {},
        },
    )
    bucket.setdefault("users", {})
    bucket.setdefault("sessions", {})
    bucket.setdefault("visits", 0)
    bucket.setdefault("generations", 0)
    bucket.setdefault("failed_generations", 0)
    bucket.setdefault("audio_duration", 0.0)
    bucket.setdefault("dwell_seconds", 0.0)
    bucket.setdefault("countries", {})
    bucket.setdefault("devices", {})
    bucket.setdefault("sources", {})
    return bucket


def request_headers(request: gr.Request | None) -> dict:
    headers = getattr(request, "headers", None)
    if not headers:
        return {}
    return {str(k).lower(): str(v) for k, v in dict(headers).items()}


def client_host(request: gr.Request | None, headers: dict) -> str:
    forwarded_for = headers.get("x-forwarded-for", "")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    client = getattr(request, "client", None)
    return getattr(client, "host", "") or "unknown"


def username_from_request(request: gr.Request | None, headers: dict) -> str:
    username = getattr(request, "username", None)
    if username:
        return str(username)
    for name in (
        "x-forwarded-user",
        "x-auth-request-user",
        "x-auth-request-preferred-username",
        "x-hf-username",
    ):
        value = headers.get(name, "").strip()
        if value:
            return value
    return "anonymous"


def visitor_id_from_request(request: gr.Request | None) -> str:
    headers = request_headers(request)
    raw_id = "|".join(
        [
            client_host(request, headers),
            headers.get("user-agent", "unknown-agent"),
            headers.get("accept-language", "unknown-language"),
        ]
    )
    return hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:16]


def session_id_from_request(request: gr.Request | None, visitor_id: str) -> str:
    session_hash = getattr(request, "session_hash", None)
    if session_hash:
        return str(session_hash)
    return visitor_id


def country_from_headers(headers: dict, ip_address: str) -> str:
    for name in (
        "cf-ipcountry",
        "x-country-code",
        "x-vercel-ip-country",
        "x-appengine-country",
        "fly-client-ip-country",
    ):
        value = headers.get(name, "").strip()
        if value:
            return value.upper()
    return country_from_ip(ip_address)


@lru_cache(maxsize=4096)
def country_from_ip(ip_address: str) -> str:
    try:
        ip_obj = ipaddress.ip_address(ip_address)
    except ValueError:
        return "Unknown"
    if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_reserved or ip_obj.is_multicast:
        return "Unknown"

    request = UrlRequest(
        f"https://ipapi.co/{ip_obj}/country_name/",
        headers={"User-Agent": "whisper-pinyin-demo/1.0"},
    )
    try:
        with urlopen(request, timeout=1.5) as response:
            country = response.read(80).decode("utf-8", errors="ignore").strip()
    except (HTTPError, URLError, TimeoutError, OSError):
        return "Unknown"
    if not country or country.lower().startswith(("undefined", "error", "reserved")):
        return "Unknown"
    return country


def device_from_user_agent(user_agent: str) -> str:
    ua = user_agent.lower()
    if "bot" in ua or "crawler" in ua or "spider" in ua:
        return "Bot"
    if "ipad" in ua or "tablet" in ua:
        return "Tablet"
    if "mobile" in ua or "iphone" in ua or "android" in ua:
        return "Mobile"
    if user_agent:
        return "Desktop"
    return "Unknown"


def source_from_headers(headers: dict) -> str:
    referer = headers.get("referer", "").strip()
    if not referer:
        return "Direct"
    parsed = urlparse(referer)
    return parsed.netloc or "Direct"


def bump_counter(target: dict, key: str, amount: int = 1) -> None:
    target[key] = int(target.get(key, 0)) + amount


def export_private_daily_user(
    *,
    day: str,
    visitor_id: str,
    session_id: str,
    username: str,
    ip_address: str,
    country: str,
    device: str,
    source: str,
    headers: dict,
) -> None:
    PRIVATE_DAILY_USERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "date_utc8": day,
        "seen_at_utc8": now_utc8_label(),
        "username": username,
        "ip_address": ip_address,
        "country_region": country,
        "device": device,
        "source": source,
        "visitor_id": visitor_id,
        "session_id": session_id,
        "user_agent": headers.get("user-agent", ""),
        "accept_language": headers.get("accept-language", ""),
    }
    with PRIVATE_DAILY_USERS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    os.chmod(PRIVATE_DAILY_USERS_PATH, 0o600)


def record_visit(request: gr.Request | None) -> str:
    headers = request_headers(request)
    visitor_id = visitor_id_from_request(request)
    session_id = session_id_from_request(request, visitor_id)
    ip_address = client_host(request, headers)
    username = username_from_request(request, headers)
    now = time.time()
    day = today_utc8()
    country = country_from_headers(headers, ip_address)
    device = device_from_user_agent(headers.get("user-agent", ""))
    source = source_from_headers(headers)

    with ANALYTICS_LOCK:
        data = read_analytics()
        bucket = day_bucket(data, day)
        visitors = data.setdefault("visitors", {})
        is_new_visitor = visitor_id not in visitors
        is_new_daily_visitor = visitor_id not in bucket["users"]
        is_new_daily_session = session_id not in bucket["sessions"]
        visitors.setdefault(visitor_id, {"first_seen": now})
        visitors[visitor_id]["last_seen"] = now
        bucket["users"][visitor_id] = now
        bucket["sessions"][session_id] = now

        session = data.setdefault("sessions", {}).setdefault(
            session_id,
            {"visitor_id": visitor_id, "started_at": now, "last_seen": now},
        )
        if is_new_daily_session:
            data["total_visits"] = int(data.get("total_visits", 0)) + 1
            bucket["visits"] = int(bucket.get("visits", 0)) + 1
        elapsed = min(max(now - float(session.get("last_seen", now)), 0.0), 60.0)
        if elapsed >= 1.0:
            data["total_dwell_seconds"] = float(data.get("total_dwell_seconds", 0.0)) + elapsed
            bucket["dwell_seconds"] = float(bucket.get("dwell_seconds", 0.0)) + elapsed
        session["visitor_id"] = visitor_id
        session["last_seen"] = now

        if is_new_visitor:
            bump_counter(data.setdefault("countries", {}), country)
            bump_counter(data.setdefault("devices", {}), device)
            bump_counter(data.setdefault("sources", {}), source)
        if is_new_daily_visitor:
            bump_counter(bucket["countries"], country)
            bump_counter(bucket["devices"], device)
            bump_counter(bucket["sources"], source)
            export_private_daily_user(
                day=day,
                visitor_id=visitor_id,
                session_id=session_id,
                username=username,
                ip_address=ip_address,
                country=country,
                device=device,
                source=source,
                headers=headers,
            )

        write_analytics(data)
    return session_id


def record_generation(audio_duration: float, request: gr.Request | None) -> None:
    record_visit(request)
    day = today_utc8()
    with ANALYTICS_LOCK:
        data = read_analytics()
        bucket = day_bucket(data, day)
        duration = max(float(audio_duration or 0.0), 0.0)
        data["total_generations"] = int(data.get("total_generations", 0)) + 1
        data["total_audio_duration"] = float(data.get("total_audio_duration", 0.0)) + duration
        bucket["generations"] = int(bucket.get("generations", 0)) + 1
        bucket["audio_duration"] = float(bucket.get("audio_duration", 0.0)) + duration
        write_analytics(data)


def record_generation_failure(request: gr.Request | None) -> None:
    record_visit(request)
    day = today_utc8()
    with ANALYTICS_LOCK:
        data = read_analytics()
        bucket = day_bucket(data, day)
        data["failed_generations"] = int(data.get("failed_generations", 0)) + 1
        bucket["failed_generations"] = int(bucket.get("failed_generations", 0)) + 1
        write_analytics(data)


def default_audio_value() -> str | None:
    if not DEFAULT_AUDIO_PATH.exists():
        return None
    try:
        with wave.open(str(DEFAULT_AUDIO_PATH), "rb") as audio_file:
            sample_width = audio_file.getsampwidth()
            frame_count = audio_file.getnframes()
    except (OSError, wave.Error):
        return None

    if sample_width != 2 or frame_count <= 0:
        return None
    return str(DEFAULT_AUDIO_PATH)


def format_seconds(seconds: float) -> str:
    seconds = max(float(seconds or 0.0), 0.0)
    if seconds < 60:
        return f"{seconds:.1f}s" if seconds and seconds < 10 else f"{seconds:.0f}s"
    minutes = seconds / 60.0
    if minutes < 60:
        whole_minutes = int(seconds // 60)
        whole_seconds = int(seconds % 60)
        return f"{whole_minutes}m {whole_seconds}s"
    hours = int(seconds // 3600)
    whole_minutes = int((seconds % 3600) // 60)
    return f"{hours}h {whole_minutes}m"


def top_item_rows(mapping: dict, limit: int = 4) -> str:
    if not mapping:
        return '<div class="wp-list-row"><span>Unknown</span><strong>0</strong></div>'
    items = Counter({str(k): int(v) for k, v in mapping.items()}).most_common(limit)
    return "\n".join(
        f'<div class="wp-list-row"><span>{escape(key)}</span><strong>{value}</strong></div>'
        for key, value in items
    )


def analytics_html() -> str:
    with ANALYTICS_LOCK:
        data = read_analytics()
    day = today_utc8()
    bucket = day_bucket(data, day)
    now = time.time()
    daily_generations = int(bucket.get("generations", 0))
    all_time_users = len(data.get("visitors", {}))
    all_time_generations = int(data.get("total_generations", 0))
    failed_generations = int(bucket.get("failed_generations", 0))
    average_audio = (
        float(bucket.get("audio_duration", 0.0)) / daily_generations
        if daily_generations
        else 0.0
    )
    sessions = data.get("sessions", {})
    daily_sessions = bucket.get("sessions", {})
    active_now = sum(
        1
        for session_id in daily_sessions
        if now - float(sessions.get(session_id, {}).get("last_seen", 0.0)) <= 120.0
    )
    total_dwell = float(bucket.get("dwell_seconds", 0.0))
    session_count = max(len(daily_sessions), 1)
    average_session_time = total_dwell / session_count
    updated_at = datetime.now(UTC_PLUS_8)

    stat_cards = [
        ("All-time Users", f"{all_time_users}"),
        ("All-time Generations", f"{all_time_generations}"),
        ("Average Audio Length", format_seconds(average_audio)),
        ("Active Now", f"{active_now}"),
        ("Average Session Time", format_seconds(average_session_time)),
    ]
    stat_cards_html = "\n".join(
        f"""
        <div class="wp-stat-card">
          <div class="wp-stat-label">{label}</div>
          <div class="wp-stat-value">{value}</div>
        </div>
        """
        for label, value in stat_cards
    )
    list_cards = [
        ("Countries / Regions", top_item_rows(bucket.get("countries", {}))),
        ("Devices", top_item_rows(bucket.get("devices", {}))),
        ("Referring Sites", top_item_rows(bucket.get("sources", {}))),
        (
            "Failed Decodes",
            f'<div class="wp-list-row"><span>Failed</span><strong>{failed_generations}</strong></div>',
        ),
    ]
    list_cards_html = "\n".join(
        f"""
        <div class="wp-list-card">
          <div class="wp-list-title">{title}</div>
          <div class="wp-list">{rows}</div>
        </div>
        """
        for title, rows in list_cards
    )
    return f"""
    <style>
      .wp-analytics {{
        box-sizing: border-box;
        width: 100%;
        border: 1px solid #ccd9ea;
        border-radius: 10px;
        padding: 18px;
        background: #f8fbff;
        color: #0b2341;
        font-family: Quicksand, Avenir, "Segoe UI", system-ui, -apple-system, sans-serif;
        letter-spacing: 0;
      }}
      .wp-analytics-head {{
        display: flex;
        justify-content: space-between;
        gap: 16px;
        align-items: flex-start;
        margin-bottom: 18px;
      }}
      .wp-title {{
        font-size: 22px;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 8px;
      }}
      .wp-subtitle {{
        color: #4a5875;
        font-size: 15px;
        line-height: 1.35;
      }}
      .wp-time {{
        color: #4a5875;
        font-size: 15px;
        line-height: 1.25;
        padding-top: 14px;
        white-space: nowrap;
      }}
      .wp-stat-grid {{
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 12px;
        margin-bottom: 12px;
      }}
      .wp-stat-card,
      .wp-list-card {{
        box-sizing: border-box;
        border: 1px solid #d3dfed;
        border-radius: 10px;
        background: white;
      }}
      .wp-stat-card {{
        min-height: 92px;
        padding: 14px;
      }}
      .wp-stat-label {{
        color: #4a5875;
        font-size: 14px;
        line-height: 1.25;
        white-space: nowrap;
        overflow-wrap: anywhere;
      }}
      .wp-stat-value {{
        color: #081d3b;
        font-size: 26px;
        font-weight: 800;
        line-height: 1.1;
        margin-top: 10px;
      }}
      .wp-list-grid {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
      }}
      .wp-list-card {{
        min-height: 86px;
        padding: 14px;
      }}
      .wp-list-title {{
        color: #081d3b;
        font-size: 16px;
        font-weight: 800;
        line-height: 1.2;
        margin-bottom: 10px;
      }}
      .wp-list-row {{
        display: flex;
        justify-content: space-between;
        gap: 16px;
        color: #0b2341;
        font-size: 15px;
        line-height: 1.25;
      }}
      .wp-list-row span {{
        min-width: 0;
        overflow-wrap: anywhere;
      }}
      .wp-list-row strong {{
        color: #081d3b;
        font-weight: 800;
        white-space: nowrap;
      }}
      @media (max-width: 1200px) {{
        .wp-stat-grid {{
          grid-template-columns: repeat(3, minmax(0, 1fr));
        }}
        .wp-list-grid {{
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }}
      }}
      @media (max-width: 720px) {{
        .wp-analytics {{
          padding: 14px;
        }}
        .wp-analytics-head {{
          display: block;
        }}
        .wp-title {{
          font-size: 20px;
        }}
        .wp-subtitle,
        .wp-time {{
          font-size: 14px;
        }}
        .wp-time {{
          padding-top: 10px;
        }}
        .wp-stat-grid,
        .wp-list-grid {{
          grid-template-columns: 1fr;
        }}
        .wp-stat-label,
        .wp-list-row {{
          font-size: 14px;
        }}
        .wp-stat-value {{
          font-size: 24px;
        }}
        .wp-list-title {{
          font-size: 15px;
        }}
      }}
    </style>
    <section class="wp-analytics">
      <div class="wp-analytics-head">
        <div>
          <div class="wp-title">Live Demo Analytics</div>
          <div class="wp-subtitle">
            Updated from this Space runtime for {updated_at.strftime("%Y-%m-%d")}
            in Beijing / Singapore Time (UTC+8).
          </div>
        </div>
        <div class="wp-time">{updated_at.strftime("%H:%M:%S")} UTC+8</div>
      </div>
      <div class="wp-stat-grid">{stat_cards_html}</div>
      <div class="wp-list-grid">{list_cards_html}</div>
    </section>
    """


@lru_cache(maxsize=1)
def load_model() -> torch.nn.Module:
    model = AutoModel.from_pretrained(
        MODEL_REPO_ID,
        trust_remote_code=True,
    )
    model.to(DEVICE)
    model.eval()
    return model


def refresh_analytics(request: gr.Request):
    record_visit(request)
    return analytics_html()


def load_default_audio():
    return default_audio_value()


def admin_status_ui(token: str):
    if not ADMIN_EXPORT_TOKEN:
        return "Admin export is disabled.", None
    if not token or not secrets.compare_digest(token, ADMIN_EXPORT_TOKEN):
        return "Forbidden.", None

    with ANALYTICS_LOCK:
        data = read_analytics()
    day = today_utc8()
    bucket = day_bucket(data, day)
    export_exists = PRIVATE_DAILY_USERS_PATH.exists()
    export_size = PRIVATE_DAILY_USERS_PATH.stat().st_size if export_exists else 0
    export_mtime = (
        datetime.fromtimestamp(PRIVATE_DAILY_USERS_PATH.stat().st_mtime, UTC_PLUS_8).strftime(
            "%Y-%m-%d %H:%M:%S UTC+8"
        )
        if export_exists
        else "Not created yet"
    )
    lines = [
        "Private Export Enabled: Yes",
        f"Export Path: {PRIVATE_DAILY_USERS_PATH}",
        f"Export Exists: {'Yes' if export_exists else 'No'}",
        f"Export Size: {export_size} bytes",
        f"Export Updated: {export_mtime}",
        f"All-time Users: {len(data.get('visitors', {}))}",
        f"All-time Generations: {int(data.get('total_generations', 0))}",
        f"Failed Decodes: {int(bucket.get('failed_generations', 0))}",
    ]
    download_path = str(PRIVATE_DAILY_USERS_PATH) if export_exists else None
    return "\n".join(lines), download_path


def decode_audio(audio_path: str, request: gr.Request):
    if not audio_path:
        record_generation_failure(request)
        return "", "", "Decode failed: please upload an audio file first.", analytics_html()

    try:
        model = load_model()
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        result = model.decode_file(audio_path, device=DEVICE)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        decode_time = time.perf_counter() - start_time
        rtf = decode_time / max(result["duration"], 1e-6)
        record_generation(result["duration"], request)
    except Exception as exc:
        record_generation_failure(request)
        return "", "", f"Decode failed: {exc}", analytics_html()

    info = (
        f"Device: {DEVICE}\n"
        f"Model repo: {MODEL_REPO_ID}\n"
        f"Input duration: {result['duration']:.2f}s\n"
        f"Decode time: {decode_time:.2f}s\n"
        f"RTF: {rtf:.2f}"
    )
    return result["pinyin"], result["tokens"], info, analytics_html()


with gr.Blocks(title="Whisper-Pinyin Demo", css=APP_CSS) as demo:
    gr.Markdown(
        "# Whisper-Pinyin\n"
        "Upload Mandarin speech audio to decode Pinyin with the cross-augmentation "
        "continuous checkpoint."
    )
    with gr.Row():
        audio_input = gr.Audio(
            sources=["upload", "microphone"],
            type="filepath",
            label="Audio",
            value=default_audio_value(),
        )
        with gr.Column():
            syllable_output = gr.Textbox(label="Decoded Pinyin", lines=4)
            token_output = gr.Textbox(label="Raw token sequence", lines=4)
            info_output = gr.Textbox(label="Runtime info", lines=5)

    decode_button = gr.Button("Decode", variant="primary")
    analytics_output = gr.HTML(value=analytics_html())
    decode_button.click(
        fn=decode_audio,
        inputs=audio_input,
        outputs=[syllable_output, token_output, info_output, analytics_output],
    )
    demo.load(fn=load_default_audio, inputs=None, outputs=audio_input)
    demo.load(fn=refresh_analytics, inputs=None, outputs=analytics_output)
    refresh_timer = gr.Timer(value=15)
    refresh_timer.tick(fn=refresh_analytics, inputs=None, outputs=analytics_output)
    with gr.Accordion("Admin", open=False):
        admin_token = gr.Textbox(label="Token", type="password")
        admin_button = gr.Button("Refresh admin status")
        admin_status = gr.Textbox(label="Admin status", lines=8)
        admin_file = gr.File(label="Daily user JSONL")
        admin_button.click(
            fn=admin_status_ui,
            inputs=admin_token,
            outputs=[admin_status, admin_file],
        )


if __name__ == "__main__":
    allowed_paths = [str(DEFAULT_AUDIO_PATH)] if DEFAULT_AUDIO_PATH.exists() else None
    demo.queue().launch(allowed_paths=allowed_paths)
