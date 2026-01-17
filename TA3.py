import time
import re
import os
from datetime import datetime
from collections import defaultdict
import threading
from thefuzz import fuzz
import json
import random
import sounddevice as sd
import soundfile as sf
import requests
import edge_tts
import asyncio
import numpy as np
from google import genai
from google.genai import types
from pydub import AudioSegment
import io
import sqlite3
from rapidfuzz import process
import uvicorn
from state import ui_state, ui_lock
import pyaudio
from pynput import mouse


# === DATABASE SETUP ===
DB_PATH = os.path.join("Data", "boatlabs.db")
conn = sqlite3.connect(DB_PATH, check_same_thread=False)

# =========================
# Environment / Gemini
# =========================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY in your environment.")

live_client = genai.Client(api_key=GEMINI_API_KEY)
blocking_client  = genai.Client(api_key=GEMINI_API_KEY)
ENGINEER_SYSTEM_PROMPT = (
    "You are a calm, concise race engineer for Minecraft Ice Boat Racing named Timothy Antonelli. "
    "Prefer brief sentences;" #include only the most relevant data. "
    "You can respond conversationally or with racing terminology when appropriate. "
    #"You like doing walltaps and blockstops to save time"
    #"Blue Ice is the fastest ice, packed ice and normal ice are slower"
)

MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
CONFIG = {
    "response_modalities": ["AUDIO"],
    "system_instruction": ENGINEER_SYSTEM_PROMPT + "If the user asks about pace, fuel, tires, traffic, delta, or strategy, do not answer. Respond with ‘Let me check that.’",
    "realtime_input_config": {
        "automatic_activity_detection": {"disabled": True}
    },
    "input_audio_transcription": {},   # transcribe user speech
    "output_audio_transcription": {}   # transcribe model audio output
}
activity_started = False


# =========================
# Transcription queues
# =========================
input_transcripts = asyncio.Queue()
output_transcripts = asyncio.Queue()

with open("commands.json", "r") as f:
    command_phrases = json.load(f)

# =========================
# Push-to-talk state
# =========================

ptt_active = asyncio.Event()   # True while Mouse5 held

# === Config ===
user_name = "Sandorus"
API_URL = "https://api.boatlabs.net/v1/timingsystems/getActiveHeats"
log_file_path = os.path.expanduser(
    r'C:\Users\Sandorus\AppData\Roaming\ModrinthApp\profiles\Ice Boat Racing (1)\logs\latest.log')
vcInputIndex = 1 #1 for tonor mic, 9 for discord
vcOutputIndex = 14 # 14 for speakers, 23 for discord, 25 for Voicemeeter

lap_times = []
current_lap = 0
fastest_lap = None
message = ""
previous_data = {}
realtime_text = ""

pit_delta = 22.0   # default value

previous_pit_counts = defaultdict(int)
pit_laps_map = defaultdict(list)

drivers = defaultdict(lambda: {"laps": {}, "lap_times": []})
SESSION_RESTART_AFTER = 14 * 60  # 14 minutes (seconds)

session_start_time = None
reconnect_requested = asyncio.Event()
gemma_queue = asyncio.Queue()


# =========================
# Audio config
# =========================

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

pya = pyaudio.PyAudio()
audio_stream = None

audio_queue_mic = asyncio.Queue(maxsize=5)
audio_queue_output = asyncio.Queue()

# === MEMORY SYSTEM ===
MEMORY_LIMIT = 5
memory_recent = []
memory_summary = ""
memory_lock = threading.Lock()

# === PIT STRATEGY ===
pit_strategy = [("Soft", 30), ("Hard", 33), ("Soft", 40)]
pit_laps = []
lap_counter = 0
for compound, duration in pit_strategy:
    lap_counter += duration
    pit_laps.append((lap_counter, compound))

# === TOOL DEFINITIONS ===
tool_instructions = """
You have access to the following tools:

1. get_driver_pace(driver_name, track_name)
   - Returns the average pace (fastest 30% non-pit laps)
   - Input JSON:
      {"tool": "get_driver_pace", "driver_name": "...", "track_name": "..."}

2. set_pit_delta(value)
   - Sets the current pit delta (seconds)
   - Input JSON:
      {"tool": "set_pit_delta", "value": NUMBER}

3. get_pit_air()
   - Computes who you will rejoin behind and ahead if you pit now.
   - Uses pit_delta seconds.
   - Output JSON:
      {"tool": "get_pit_air"}

If no tool is needed, respond normally.
"""
# === Keywords ===
RACE_KEYWORDS = {
    "pace", "lap time", "delta", "gap",
    "air", "traffic", "dirty air",
    "fuel", "tires", "tyres",
    "pit", "box", "strategy", "stint"
}

def requires_tools(text: str) -> bool:
    t = text.lower()
    return any(re.search(rf"\b{k}\b", t) for k in RACE_KEYWORDS)

def find_device_index(device_name):
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if device_name.lower() in dev['name'].lower():
            return i
    return None

def parse_lap_time(lap_str):
    if ':' in lap_str:
        match = re.match(r"(\d+):(\d+)\.(\d+)", lap_str)
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            milliseconds = int(match.group(3))
            return minutes * 60 + seconds + milliseconds / 1000
    else:
        match = re.match(r"(\d+)\.(\d+)", lap_str)
        if match:
            seconds = int(match.group(1))
            milliseconds = int(match.group(2))
            return seconds + milliseconds / 1000
    return None

def format_lap_time(sec: float) -> str:
    if sec is None:
        return "n/a"
    minutes = int(sec // 60)
    seconds = int(sec % 60)
    milliseconds = int(round((sec - int(sec)) * 1000))
    return f"{minutes}:{seconds:02d}.{milliseconds:03d}"


def parse_timestamp(log_line):
    match = re.match(r"\[(\d{2}:\d{2}:\d{2})\]", log_line)
    if match:
        return datetime.strptime(match.group(1), "%H:%M:%S")
    return None

def get_current_positions(drivers_dict=None):
    api_data = fetch_api_data(user_name)
    if not api_data:
        return []

    # The API data is already ordered by position
    # Each entry has: "name", "timeDiff", "pits", etc.
    positions = []
    for i, entry in enumerate(api_data):
        name = entry["name"]
        time_diff = entry.get("timeDiff", None)
        laps = drivers.get(name, {}).get("lap_times", [])
        lap_count = len(laps)
        positions.append((name, time_diff, lap_count))
    return positions

# =========================
# Mouse listener
# =========================

def start_mouse_listener(loop):
    def on_click(x, y, button, pressed):
        if button == mouse.Button.x2:  # Mouse5
            if pressed:
                loop.call_soon_threadsafe(ptt_active.set)
                print("🎤 PTT ON")
            else:
                loop.call_soon_threadsafe(ptt_active.clear)
                print("🛑 PTT OFF")

    mouse.Listener(on_click=on_click).start()


# =========================
# Mic capture (PTT)
# =========================

async def listen_audio():
    """Capture microphone audio only while PTT is active."""
    global audio_stream

    mic_info = pya.get_default_input_device_info()
    audio_stream = await asyncio.to_thread(
        pya.open,
        format=FORMAT,
        channels=CHANNELS,
        rate=SEND_SAMPLE_RATE,
        input=True,
        input_device_index=mic_info["index"],
        frames_per_buffer=CHUNK_SIZE,
    )

    kwargs = {"exception_on_overflow": False} if __debug__ else {}

    while True:
        # Wait until PTT is pressed
        await ptt_active.wait()

        # Stream audio while PTT is held
        while ptt_active.is_set():
            data = await asyncio.to_thread(audio_stream.read, CHUNK_SIZE, **kwargs)
            await audio_queue_mic.put({"data": data, "mime_type": "audio/pcm"})

        # Avoid busy loop
        await asyncio.sleep(0.01)


# =========================
# Send audio to Gemini
# =========================

async def send_realtime(session):
    global activity_started

    while True:
        await ptt_active.wait()

        if not activity_started:
            await session.send_realtime_input(activity_start=types.ActivityStart())
            activity_started = True

        while ptt_active.is_set():
            msg = await audio_queue_mic.get()
            await session.send_realtime_input(
                audio=types.Blob(
                    data=msg["data"],
                    mime_type="audio/pcm;rate=16000"
                )
            )

        if activity_started:
            await session.send_realtime_input(activity_end=types.ActivityEnd())
            activity_started = False
            if reconnect_requested.is_set():
                print("🔁 Reconnecting after user finished speaking")
                raise asyncio.CancelledError


# =========================
# Receive Gemini audio
# =========================

async def receive_audio(session):
    while True:
        turn = session.receive()
        async for response in turn:
            # Handle Gemini audio playback
            if response.server_content and response.server_content.model_turn:
                for part in response.server_content.model_turn.parts:
                    if part.inline_data and isinstance(part.inline_data.data, bytes):
                        audio_queue_output.put_nowait(part.inline_data.data)

            # Handle output transcription (Gemini speech)
            if response.server_content and response.server_content.output_transcription:
                text = response.server_content.output_transcription.text
                await output_transcripts.put(text)
                print(f"[Gemini said]: {text}")

            # Handle input transcription (user speech)
            if response.server_content and response.server_content.input_transcription:
                text = response.server_content.input_transcription.text
                await input_transcripts.put(text)
                print(f"[User said]: {text}")

# =========================
# Playback
# =========================

async def play_audio():
    """Play audio responses."""
    stream = await asyncio.to_thread(
        pya.open,
        format=FORMAT,
        channels=CHANNELS,
        rate=RECEIVE_SAMPLE_RATE,
        output=True,
    )

    while True:
        audio = await audio_queue_output.get()
        await asyncio.to_thread(stream.write, audio)

# =========================        
# reconnection
# =========================
async def session_watchdog():
    """Request reconnection after 14 minutes if idle."""
    while True:
        await asyncio.sleep(5)

        if session_start_time is None:
            continue

        elapsed = time.monotonic() - session_start_time

        if elapsed >= SESSION_RESTART_AFTER:
            if not ptt_active.is_set():
                print("🔁 Session age limit reached — reconnecting")
                reconnect_requested.set()
                return
            else:
                # User is talking — wait until they finish
                await ptt_active.wait()

def replace_fragment(command: str, fragment: str, resolved: str):
    if not fragment or not resolved:
        return command
    pattern = re.compile(re.escape(fragment), re.IGNORECASE)
    return pattern.sub(resolved, command)

async def route_user_commands(live_session):
    while True:
        user_text = await input_transcripts.get()

        if requires_tools(user_text):
            print("🔵 Routing to Gemma (tools)")

            # Enqueue Gemma work (non-blocking)
            gemma_queue.put_nowait({
                "text": user_text,
                "live_session": live_session
            })

            # Gemini should still respond immediately
            # (system prompt handles "checking / stand by")

        else:
            print("🟢 Routing to Gemini (fast path)")
            # Nothing to do — Gemini already heard the transcript

        print(
            f"[ROUTER] '{user_text}' → "
            f"{'TOOLS' if requires_tools(user_text) else 'FAST'}"
        )

        # Store memory AFTER routing
        add_memory(user_text, "")


async def gemma_worker():
    while True:
        job = await gemma_queue.get()

        try:
            # Run Gemma safely
            result_text = await asyncio.to_thread(generate_engineer_text, job["text"])

            # Send result to Gemini as assistant content
            await job["live_session"].send_client_content(
                turns={
                    "role": "assistant",
                    "parts": [{"text": result_text}]
                },
                turn_complete=True
            )

        except Exception as e:
            print("[Gemma] Error processing job:", e)
            # Send a fallback message so Gemini doesn't hang
            await job["live_session"].send_client_content(
                turns={
                    "role": "assistant",
                    "parts": [{"text": "Unable to process tool request at the moment."}]
                },
                turn_complete=True
            )

        finally:
            gemma_queue.task_done()


def fetch_api_data(user_name: str):
    """
    Fetch the active heats and return all drivers from the heat containing `user_name`.
    Normalizes to a list of driver dicts + metadata.
    """
    try:
        resp = requests.get(API_URL, timeout=10)
        resp.raise_for_status()
        heats = resp.json()
        if isinstance(heats, str):
            # API returned a message instead of data
            print(f"[API] Message from server: {heats}")
            return {"event": "Race hasn't started yet", "drivers": []}

        if not isinstance(heats, list):
            print(f"[API] Unexpected API format: {type(heats)}")
            return {"event": "Race hasn't started yet", "drivers": []}
    except Exception as e:
        print(f"[API] Error fetching data: {e}")
        return None

    try:
        # Look through all heats to find the one with our driver
        for heat in heats:
            drivers = heat.get("drivers", [])
            # Sort drivers by position first
            sorted_drivers = sorted(
                (d for d in drivers if isinstance(d, dict)),
                key=lambda x: x.get("position", 999)
            )

            # Build cumulative gap_to_leader for each driver
            cumulative_gap = 0
            normalized = []
            for i, d in enumerate(sorted_drivers):
                delta_to_driver_before = d.get("deltaToDriverBefore", 0)
                gap_to_leader = 0 if i == 0 else cumulative_gap + delta_to_driver_before
                cumulative_gap = gap_to_leader

                gap_ahead = delta_to_driver_before / 1000.0 if i > 0 else None
                gap_behind = None
                if i < len(sorted_drivers) - 1:
                    gap_behind = sorted_drivers[i + 1].get("deltaToDriverBefore", 0) / 1000.0

                normalized.append({
                    "name": d.get("name"),
                    "position": d.get("position"),
                    "lap": d.get("lap"),
                    "time": d.get("time", 0),
                    "gap_to_leader": round(gap_to_leader / 1000, 2),
                    "gap_ahead": round(gap_ahead, 2) if gap_ahead is not None else None,
                    "gap_behind": round(gap_behind, 2) if gap_behind is not None else None,
                    "pits": heat.get("pits", 0),
                })

            # Find your driver
            if any(user_name.lower() in d["name"].lower() for d in normalized):
                return {
                    "event": heat.get("event_name"),
                    "round": heat.get("round_name"),
                    "heat": heat.get("heat_name"),
                    "track": heat.get("track"),
                    "laps_total": heat.get("laps"),
                    "drivers": normalized,
                }

        # If no heat found with your driver
        return {"event": "Race hasn't started yet", "drivers": []}

    except Exception as e:
        print(f"[API] Error processing heat data: {e}")
        return {"event": "Race hasn't started yet", "drivers": []}
    
def generate_engineer_text(user_request: str) -> str:
    state = build_race_state_summary(user_name)  # use driver_name not user_name
    memory_section = f"""Long-term memory:
    {memory_summary if memory_summary else "None"}

    Recent interactions:
    {json.dumps(memory_recent, indent=2) if memory_recent else "None"}
    """

    try:
        # FIRST CALL — ask Gemma if tool is needed
        prompt = f"""
{ENGINEER_SYSTEM_PROMPT}

{tool_instructions}

Race State:
{state}

Memory:
{memory_section}

User request:
{user_request}

If you want to call a tool, output ONLY the JSON tool call.
Otherwise answer normally.
"""

        resp = blocking_client.models.generate_content(
            model="gemma-3-27b-it",
            contents=[prompt],
            config=types.GenerateContentConfig(
                max_output_tokens=200,
                temperature=0.5,
                top_p=0.9,
                response_mime_type="text/plain"
            ),
        )

        text = resp.text.strip()

        # Try to interpret the model output as a tool call
        # Try parsing tool call JSON
        clean = extract_json(text)

        if clean is None:
            # No JSON → normal answer
            safe_text = (
                text.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
            )
            print("llm generated :",safe_text)
            return safe_text
        
        else:
            try:
                tool_call = json.loads(clean)
            except json.JSONDecodeError:
                # No JSON → normal text response
                safe_text = (
                    text.replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;")
                )
                
                return safe_text


            # --------------------------------------------------------
            # TOOL CALL EXECUTION
            # --------------------------------------------------------

            tool_result = None

            # get_driver_pace tool
            if tool_call.get("tool") == "get_driver_pace":
                avg_ms = get_driver_pace(
                    conn,
                    tool_call["track_name"],
                    tool_call["driver_name"]
                    
                )
                tool_result = {
                    "driver_name": tool_call["driver_name"],
                    "track_name": tool_call["track_name"],
                    "average_ms": avg_ms
                }

            # set_pit_delta tool
            elif tool_call.get("tool") == "set_pit_delta":
                new_val = set_pit_delta(tool_call["value"])
                tool_result = {"new_value": new_val}

            elif tool_call.get("tool") == "get_pit_air":
                tool_result = get_pit_air()

            # If the tool was not recognized
            if tool_result is None:
                return "Radio error. Tool call not recognized."

            # --------------------------------------------------------
            # SECOND LLM CALL (Summarize results)
            # --------------------------------------------------------

            followup_prompt = (
                f"""

                Tool result:\n{json.dumps(tool_result)}

                User request:{user_request}
                
                Now produce the final natural-language answer."""
            )
            print(followup_prompt)
            followup = blocking_client.models.generate_content(
                model="gemma-3-27b-it",
                contents=[followup_prompt],
                config=types.GenerateContentConfig(
                    max_output_tokens=150,
                    temperature=0.4,
                    response_mime_type="text/plain"
                ),
            )

            final_text = followup.text.strip()
            final_text = (
                final_text.replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;")
            )
            return final_text

    except Exception as e:
        return f"Radio error. {str(e)}"
    
def extract_json(text: str) -> str | None:
    """
    Extracts the first {...} JSON object from the text.
    Ignores any markdown code fences or unrelated output.
    """
    # Use a regex to find a JSON object
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if not match:
        return None

    candidate = match.group(0).strip()

    # Optionally strip trailing backticks or similar garbage
    candidate = candidate.rstrip('`').rstrip().strip()

    return candidate

def add_memory(user_text: str, assistant_text: str):
    global memory_recent, memory_summary

    with memory_lock:
        memory_recent.append({
            "user": user_text,
            "assistant": assistant_text
        })

        # If too long → summarize oldest interaction into long-term summary
        if len(memory_recent) > MEMORY_LIMIT:
            oldest = memory_recent.pop(0)

            summary_text = (
    f"Existing summary:\n{memory_summary if memory_summary else ''}\n\n"
    f"New interaction:\nUser: {oldest['user']}\nAssistant: {oldest['assistant']}\n\n"
    "Task:\nRewrite the memory summary including essential info and remove redundancy."
)

            try:
                result = blocking_client.models.generate_content(
                    model="gemma-3-27b-it",
                    contents=[types.Content(parts=[types.Part.from_text(text = summary_text)])],
                    config=types.GenerateContentConfig(
                        max_output_tokens=200,
                        temperature=0.3,
                    )
                )
                memory_summary = (getattr(result, "text", "") or "").strip()

            except Exception as e:
                print(f"[Memory] Error summarizing memory: {e}")



def build_race_state_summary(user_name: str, max_positions: int = 5, clean_laps_count: int = 3) -> str:
    try:
        data = fetch_api_data(user_name)
        if not data:
            return f"No race data available for {user_name}."

        drivers = data["drivers"]
        if not isinstance(drivers, list):
            return f"No race data available for {user_name}."

        def format_driver_list(drivers):
            lines = ["Drivers:"]
            for d in drivers:
                try:
                    name = d["name"]
                    pos = d["position"]
                    lap = d["lap"]
                    pits = d.get("pits", 0)
                    gap = d.get("gap_ahead")
                    gap_str = f"{gap:.3f}s" if gap is not None else "0.000s"
                    lines.append(f"{pos}. {name} | Lap {lap} | Gap to car ahead: {gap_str} | Pits: {pits}")
                except Exception:
                    lines.append(f"{d.get('name', 'Unknown')} | Data unavailable")
            return "\n".join(lines)

        driver_list_str = format_driver_list(drivers)

        # Next scheduled pit
        next_pit = next((lap for lap, _ in pit_laps if lap >= current_lap), None)
        next_compound = next((c for l, c in pit_laps if l == next_pit), None) if next_pit else None
        next_pit_str = f"lap {next_pit} ({next_compound})" if next_pit else "none"

        # Fastest lap
        fl = format_lap_time(fastest_lap) if fastest_lap else "n/a"

        lines = [
            f"Event: {data.get('event', 'n/a')} ({data.get('round', 'n/a')} / {data.get('heat', 'n/a')})",
            f"Track: {data.get('track', 'n/a')} | Total laps: {data.get('laps_total', 'n/a')}",
            f"Driver: {user_name}",
            f"Current lap: {current_lap}",
            f"Fastest lap: {fl}",
            f"Next scheduled pit: {next_pit_str}",
            f"Pit delta: {pit_delta:.1f} seconds",
            "",
            driver_list_str
        ]
        return "\n".join(lines)

    except Exception as e:
        print(f"[Race Summary] Error building race state: {e}")
        return f"No race data available for {user_name}."
    
# Tool implementation
def get_driver_pace(conn, track_name: str, driver_name: str):
    cursor = conn.cursor()

    cursor.execute("""
        SELECT laps.lap_time_ms
        FROM laps
        JOIN rounds ON laps.round_id = rounds.round_id
        WHERE laps.driver_name = ?
          AND rounds.track_name = ?
          AND laps.pit_stop = 0
          AND laps.lap_time_ms IS NOT NULL
        ORDER BY laps.lap_time_ms ASC
    """, (driver_name, track_name))

    rows = [r[0] for r in cursor.fetchall()]

    if not rows:
        return None  # no laps found

    # Take the fastest 30%
    count = max(1, int(len(rows) * 0.30))
    fastest_segment = rows[:count]

    # Average them
    avg_ms = sum(fastest_segment) / len(fastest_segment)
    return avg_ms

def set_pit_delta(new_value: float):
    global pit_delta
    pit_delta = float(new_value)
    with ui_lock:
        ui_state["pit_delta"] = pit_delta
    return pit_delta

def get_pit_air():
    """
    Computes expected traffic after a pitstop.
    Uses: pit_delta, fetch_api_data(), current user position.
    Returns dict with fields:
      - ahead_name
      - ahead_gap
      - behind_name
      - behind_gap
      - nearby_drivers: list of {"name": ..., "gap": ...}
    """

    data = fetch_api_data(user_name)
    if not data:
        return {"error": "no_api_data"}

    drivers = data["drivers"]

    # get user's current gap_to_leader
    user = next((d for d in drivers if d["name"] == user_name), None)
    if not user:
        return {"error": "user_not_found"}

    user_gap = user["gap_to_leader"]  # seconds from leader

    # Pit exit gap:
    # We ADD pit_delta → you “jump backward” in cumulative gap order
    exit_gap = user_gap + pit_delta

    # Determine which drivers match that gap
    # Create sorted list by gap_to_leader
    ordered = sorted(drivers, key=lambda x: x["gap_to_leader"])

    # Find who you will slot between
    ahead = None
    behind = None

    for i in range(len(ordered) - 1):
        g1 = ordered[i]["gap_to_leader"]
        g2 = ordered[i + 1]["gap_to_leader"]

        if g1 <= exit_gap <= g2:
            ahead = ordered[i + 1]   # driver in front after pit
            behind = ordered[i]      # driver behind after pit
            break

    # If exit_gap is beyond everyone
    if ahead is None:
        ahead = ordered[-1]
        behind = ordered[-2]

    # Compute gaps relative to your pit-exit position
    ahead_gap = ahead["gap_to_leader"] - exit_gap
    behind_gap = exit_gap - behind["gap_to_leader"]

    # Nearby traffic within ±5s
    nearby = []
    for d in ordered:
        diff = abs(d["gap_to_leader"] - exit_gap)
        if diff <= 5.0:
            nearby.append({"name": d["name"], "gap": round(diff, 3)})

    # Remove the user from nearby in case they appear
    nearby = [d for d in nearby if d["name"] != user_name]

    return {
        "ahead_name": ahead["name"],
        "ahead_gap": round(ahead_gap, 3),
        "behind_name": behind["name"],
        "behind_gap": round(behind_gap, 3),
        "nearby_drivers": nearby,
        "exit_gap": exit_gap,
        "pit_delta": pit_delta
    }

def resolve_driver_name(conn, spoken_name: str):
    cursor = conn.cursor()
    cursor.execute("SELECT driver_name FROM drivers")
    driver_names = [row[0] for row in cursor.fetchall()]

    # Remove "Timothy" from fuzzy matching
    fuzzy_candidates = [name for name in driver_names if name.lower() != "timothy"]

    # Exact match check
    for name in fuzzy_candidates:
        if name.lower() in spoken_name.lower():
            return name, name   # (resolved_name, matched_fragment)

    # Fuzzy match on entire spoken text
    best_match, score, _ = process.extractOne(spoken_name, fuzzy_candidates)
    if score < 70:
        return None, None

    return best_match, best_match


def resolve_track_name(conn, spoken: str):
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT track_name FROM events")
    tracks = [t[0] for t in cursor.fetchall()]

    # Exact match check
    for name in tracks:
        if name.lower() in spoken.lower():
            return name, name

    best_match, score, _ = process.extractOne(spoken, tracks)
    if score < 70:
        return None, None

    return best_match, best_match

# =========================
# Main
# =========================

async def run():
    global session_start_time, reconnect_requested

    loop = asyncio.get_running_loop()
    start_mouse_listener(loop)
    asyncio.create_task(gemma_worker())

    while True:
        reconnect_requested.clear()

        try:
            async with live_client.aio.live.connect(
                model=MODEL,
                config=CONFIG,
            ) as live_session:

                session_start_time = time.monotonic()
                print("Connected to Gemini (new session)")
                print("Hold Mouse5 to talk.")

                async with asyncio.TaskGroup() as tg:
                    tg.create_task(listen_audio())
                    tg.create_task(send_realtime(live_session))
                    tg.create_task(receive_audio(live_session))
                    tg.create_task(play_audio())
                    tg.create_task(session_watchdog())
                    tg.create_task(route_user_commands(live_session)) 
                    

        except asyncio.CancelledError:
            print("Session restarting…")
            await asyncio.sleep(0.5)  # small grace delay
            continue

        except Exception as e:
            print("Session error:", e)
            await asyncio.sleep(2)


def main_loop():
    
    print("Race tracking started (via API)...")
    while True:
        data = fetch_api_data(user_name)

        if not data or "drivers" not in data:
            time.sleep(2)
            continue

        for entry in data["drivers"]:
            if not isinstance(entry, dict):
                continue

            name = entry["name"]
            time_diff = entry.get("deltaToDriverBefore", 0)
            pits = entry["pits"]
            #team_color = entry["teamColor1"]

            # Initialize previous data if new
            if name not in previous_data:
                previous_data[name] = {
                    "timeDiff": time_diff,
                    "pits": pits
                }
                continue  # Skip comparison on first pass

            name = entry["name"]
            pits = entry["pits"]

            # Get lap number (based on how many lap times exist)
            lap_num = len(drivers[name]["lap_times"])

            # Detect pit increase
            if pits > previous_pit_counts[name]:
                pit_laps_map[name].append(lap_num)
            previous_pit_counts[name] = pits


            # Compare timeDiff to detect movement or lap changes
            old_diff = previous_data[name]["timeDiff"]
            if time_diff != old_diff:
                print(f"{name} new timeDiff: {time_diff}ms (was {old_diff}ms)")

            # Compare pit count
            old_pits = previous_data[name]["pits"]
            if pits > old_pits:
                print(f"{name} entered pits! ({old_pits} → {pits})")

            # Update saved data
            previous_data[name]["timeDiff"] = time_diff
            previous_data[name]["pits"] = pits

        with ui_lock:
            ui_state["drivers"] = {
                d["name"]: {
                    "position": d["position"],
                    "lap": d["lap"],
                    "gap_to_leader": d["gap_to_leader"],
                    "gap_ahead": d["gap_ahead"],
                    "gap_behind": d["gap_behind"],
                    "pits": d["pits"],
                }
                for d in data["drivers"]
            }


        time.sleep(5)  # Poll every 5 second (adjust as needed)

if __name__ == "__main__":
    # === 3. API polling thread ===
    api_thread = threading.Thread(target=main_loop, daemon=True)
    api_thread.start()

    # === 4. Log file reader thread ===
    #log_thread = threading.Thread(target=log_reader_loop, daemon=True)
    #log_thread.start()
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("Interrupted by user.")

    #Start UI


    threading.Thread(
        target=lambda: uvicorn.run(
            "ui_server:app",
            host="127.0.0.1",
            port=8765,
            log_level="warning"
        ),
        daemon=True
    ).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting program.")
    
