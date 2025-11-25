import time
import re
import os
from datetime import datetime
from collections import defaultdict
import threading
import heapq
from thefuzz import fuzz
import json
import random
from RealtimeSTT import AudioToTextRecorder
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

# === DATABASE SETUP ===
DB_PATH = os.path.join("Data", "boatlabs.db")
conn = sqlite3.connect(DB_PATH, check_same_thread=False)


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY in your environment.")

genai_client = genai.Client(api_key=GEMINI_API_KEY)

ENGINEER_SYSTEM_PROMPT = (
    "You are a calm, concise race engineer for Minecraft Ice Boat Racing named Timothy Antonelli. "
    "Prefer brief sentences; include only the most relevant data. "
    #"You like doing walltaps and blockstops to save time"
    #"Blue Ice is the fastest ice, packed ice and normal ice are slower"
)


# Simple pronoun mapping for first → second person
PRONOUN_MAP = {
    "I": "you",
    "me": "you",
    "my": "your",
    "mine": "yours",
    "we": "you all",
    "us": "you",
    "you": "I",
    "your": "my",
    "yours": "mine",
}

with open("commands.json", "r") as f:
    command_phrases = json.load(f)


tts_queue = []
tts_lock = threading.Lock()

def tts_worker():
    while True:
        with tts_lock:
            if tts_queue:
                priority, message, callback = heapq.heappop(tts_queue)
            else:
                message = None
                callback = None  # Initialize callback too

        if message:
            play_tts_message(message)

        if callback:
            callback()
        else:
            time.sleep(0.01)  # Avoid tight spinning

# === Config ===
user_name = "Sandorus"
API_URL = "https://api.boatlabs.net/v1/timingsystems/getActiveHeats"
log_file_path = os.path.expanduser(
    r'C:\Users\Sandorus\AppData\Roaming\ModrinthApp\profiles\Ice Boat Racing (1)\logs\latest.log')
vcInputIndex = 1 #1 for tonor mic, 9 for discord
vcOutputIndex = 14 # 14 for speakers, 23 for discord, 25 for Voicemeeter

TRIGGER_WORDS = ["Timothy Antonelli","Antonelli","Antonelly","Timothy","Timmy"]

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

If no tool is needed, respond normally.
"""


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

async def play_message(message: str):
    voice = "en-GB-SoniaNeural"
    rate = "+20%"
    # Generate MP3 audio in memory
    tts = edge_tts.Communicate(text=message, voice=voice, rate=rate)
    mp3_bytes = b""
    async for chunk in tts.stream():
        if chunk["type"] == "audio":
            mp3_bytes += chunk["data"]

    # Decode MP3 → PCM in memory
    audio = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / (2**15)

    # Ensure stereo
    if audio.channels == 1:
        samples = np.column_stack((samples, samples))
    elif audio.channels > 2:
        samples = samples[:, :2]
    print("playing message")
    # Play via sounddevice
    sd.default.device = vcOutputIndex
    sd.play(samples, samplerate=audio.frame_rate, device=vcOutputIndex)
    sd.wait()


def play_tts_message(message: str):
    """
    Sync wrapper: run the async play_message in its own event loop.
    """
    

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(play_message(message))
    loop.close()


def play_notification_sound(path="E:/Songs/Sound effects/F1_Radio_-_Notification_Sound.mp3"):
    """Plays a notification sound asynchronously without blocking TTS."""
    def _play():
        data, fs = sf.read(path)
        # Ensure stereo
        if data.ndim == 1:
            data = np.column_stack((data, data))
        elif data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)
        elif data.shape[1] > 2:
            data = data[:, :2]

        sd.default.device = vcOutputIndex 
        sd.play(data, fs)
        sd.wait()

    threading.Thread(target=_play, daemon=True).start()

# Updated queue_tts_message to include optional callback
def queue_tts_message(message, priority=5, callback=None):
    with tts_lock:
        heapq.heappush(tts_queue, (priority, message, callback))


def mirror_question(user_text):
    # Find trigger word in the text
    trigger_regex = r'\b(' + '|'.join(TRIGGER_WORDS) + r')\b'
    match = re.search(trigger_regex, user_text, flags=re.IGNORECASE)
    if not match:
        return None

    # Take everything after the trigger word
    after_trigger = user_text[match.end():].strip()

    # Only mirror if the sentence ends in a question mark
    if not after_trigger.endswith("?"):
        return None

    # Replace first person words with second person words
    words = after_trigger.split()
    mirrored_words = [PRONOUN_MAP.get(w, PRONOUN_MAP.get(w.lower(), w)) for w in words]

    mirrored_text = " ".join(mirrored_words)
    # Optionally, prepend a prefix to sound natural
    tts_message = f"You want me to check {mirrored_text}"
    return tts_message

def new_listener():
    print("Wait until it says 'speak now'")
    recorder = AudioToTextRecorder(input_device_index=vcInputIndex, model="tiny.en",
        enable_realtime_transcription=False,
        use_main_model_for_realtime=True,
        realtime_processing_pause=0.2,
        on_realtime_transcription_update=process_realtime_update,
        no_log_file=True,
        batch_size=16,
        )

    while True:
        recorder.text(process_text)

def process_text(command: str):
    if not any(trigger.lower() in command.lower() for trigger in TRIGGER_WORDS):
        print("[Voice] Heard:", command)
        return
    
    if not command:
        return

    command = command.replace("-", "dash")
    original_command = command

    driver_resolved, _ = resolve_driver_name(conn, command)
    track_resolved, _ = resolve_track_name(conn, command)

    # Build resolved info only if at least one match exists
    resolved_info = ""

    if driver_resolved or track_resolved:
        resolved_info = "\nResolved:"
        if driver_resolved:
            resolved_info += f"\n  driver_name: {driver_resolved}"
        if track_resolved:
            resolved_info += f"\n  track_name: {track_resolved}"

    clean_command = f"Original STT: '{original_command}'{resolved_info}"

    print("[Voice] Heard:", clean_command)

    threading.Thread(
        target=handle_llm_and_tts, 
        args=(clean_command,), 
        daemon=True
    ).start()

    # Pre-response filler
    if command.strip().endswith("?"):
        msg = mirror_question(command)
        time.sleep(0.1)
    else:
        msg = "Let me think, um"
        time.sleep(0.2)

    time.sleep(0.1)
    queue_tts_message(msg, priority=3)
    play_notification_sound()

def replace_fragment(command: str, fragment: str, resolved: str):
    if not fragment or not resolved:
        return command
    pattern = re.compile(re.escape(fragment), re.IGNORECASE)
    return pattern.sub(resolved, command)

def handle_llm_and_tts(command: str):
# Generate text response (Gemma)
    text = generate_engineer_text(command)
    print(f">> [Engineer text] {text}")

    # Add to memory
    add_memory(command, text)

    # Split text into chunks only when . or ? is followed by a space or end of string
    # This avoids splitting decimals like 1.233
    chunks = re.split(r'([.?])(?=\s|$)', text)

    # Recombine punctuation with preceding text
    messages = []
    i = 0
    while i < len(chunks) - 1:
        part = chunks[i].strip()
        punct = chunks[i + 1]
        if part:
            messages.append(part + punct)
        i += 2
    # Add any leftover text
    if i < len(chunks):
        leftover = chunks[i].strip()
        if leftover:
            messages.append(leftover)

    # Queue the messages: first chunk priority=3, then increasing priority for each subsequent message
    base_priority = 3
    increment = 1

    for idx, msg in enumerate(messages):
        priority = base_priority + idx * increment
        queue_tts_message(msg, priority=priority)
    
    

def process_realtime_update(text: str):
    global realtime_text
    realtime_text = text


def is_improving(user_name):
    clean_laps = get_last_clean_laps(user_name)
    if not clean_laps or len(clean_laps) < 5:
        # Not enough data to compare reliably
        return "Not enough clean lap data to determine improvement."

    n = len(clean_laps)
    segment_size = max(1, n // 5)  # 20% of laps (rounded down)

    recent_segment = clean_laps[-segment_size:]
    previous_segment = clean_laps[-2*segment_size:-segment_size]

    if not previous_segment:
        return "Not enough previous lap data to compare improvement."

    recent_avg = sum(recent_segment) / len(recent_segment)
    previous_avg = sum(previous_segment) / len(previous_segment)

    diff = previous_avg - recent_avg  # positive means improvement

    if diff > 0:
        msg = f"You are improving, your recent laps are {diff:.3f} seconds faster on average."
    elif diff < 0:
        msg = f"You are getting slower by {-diff:.3f} seconds on average recently."
    else:
        msg = "Your pace is consistent over recent laps."

    return msg


def follow(file):
    file.seek(0, os.SEEK_END)
    while True:
        line = file.readline()
        if not line:
            time.sleep(0.1)
            continue
        yield line

def compute_last2_avg(times):
    return sum(times[-2:]) / 2 if len(times) >= 2 else None

def compute_total_time(times):
    return sum(times)

def get_last_clean_laps(name, count=3):
                all_laps = drivers[name]["lap_times"]
                pit_laps = pit_laps_map[name]
                clean_laps = [t for i, t in enumerate(all_laps, 1) if i not in pit_laps]
                return clean_laps[-count:] if len(clean_laps) >= count else None

def fetch_api_data(user_name: str):
    """
    Fetch the active heats and return all drivers from the heat containing `user_name`.
    Normalizes to a list of driver dicts + metadata.
    """
    try:
        resp = requests.get(API_URL, timeout=10)
        resp.raise_for_status()
        heats = resp.json()
    except Exception as e:
        print(f"[API] Error fetching data: {e}")
        return None

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
            # Gap to leader = sum of deltaToDriverBefore for all drivers ahead
            delta_to_driver_before = d.get("deltaToDriverBefore", 0)
            if i == 0:
                gap_to_leader = 0
            else:
                gap_to_leader = cumulative_gap + delta_to_driver_before

            # Update cumulative_gap for next iteration
            cumulative_gap = gap_to_leader

            # Gap to driver ahead and behind
            gap_ahead = delta_to_driver_before / 1000.0 if i > 0 else None
            gap_behind = None
            if i < len(sorted_drivers) - 1:
                gap_behind = sorted_drivers[i + 1].get("deltaToDriverBefore", 0) / 1000.0

            normalized.append({
                "name": d.get("name"),
                "position": d.get("position"),
                "lap": d.get("lap"),
                "time": d.get("time", 0),
                "gap_to_leader": round(gap_to_leader / 1000, 2),  # ms → seconds, 2 decimal places
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

    print(f"[API] No active heat found for driver '{user_name}'")
    return None
    
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

        resp = genai_client.models.generate_content(
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
            return safe_text
        
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

        # If the tool was not recognized
        if tool_result is None:
            return "Radio error. Tool call not recognized."

        # --------------------------------------------------------
        # SECOND LLM CALL (Summarize results)
        # --------------------------------------------------------

        followup_prompt = (
            f"""{ENGINEER_SYSTEM_PROMPT}

            Tool result:\n{json.dumps(tool_result)}

            User request:{user_request}
            
            Now produce the final natural-language answer."""
        )
        print(followup_prompt)
        followup = genai_client.models.generate_content(
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
                result = genai_client.models.generate_content(
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
    return pit_delta


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

def main_loop():
    
    print("Race tracking started (via API)...")
    while True:
        data = fetch_api_data(user_name)

        for entry in data:
            name = entry["name"]
            time_diff = entry["timeDiff"]
            pits = entry["pits"]
            team_color = entry["teamColor1"]

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

        time.sleep(5)  # Poll every 5 second (adjust as needed)

def log_reader_loop():
    global current_lap, fastest_lap

    with open(log_file_path, "r", encoding="utf-8") as logfile:
        message = ("Race tracking started. Waiting for lap data...")
        print(message)
        queue_tts_message(message, priority=0)
        play_notification_sound()
        loglines = follow(logfile)

        for line in loglines:
            line_handled = False
            lap_time = None
            name = None

            match1 = re.search(r"You finished lap in: ([\d:.]+)", line)
            if match1:
                print(f"[DEBUG] Matched own lap: {line.strip()}")
                lap_time = parse_lap_time(match1.group(1))
                name = user_name
                lap_num = len(drivers[name]["lap_times"]) + 1  # Increment lap count for yourself
                line_handled = True


            # For own fastest lap announcement
            match2 = re.search(rf"{user_name} new fastest lap of ([\d:.]+)", line)
            if match2:
                print(f"[DEBUG] Matched own fastest: {line.strip()}")
                lap_time = parse_lap_time(match2.group(1))
                name = user_name
                line_handled = True

            # For other drivers' fastest lap announcement
            match2_5 = re.search(r"(\w+) new fastest lap of ([\d:.]+)", line)
            if match2_5:
                name = match2_5.group(1)
                lap_time = parse_lap_time(match2_5.group(2))
                line_handled = True
                message = (name +", set the fastest lap")
                queue_tts_message(message, priority=5)

            match3 = re.search(r"(\w+) finished lap (\d+) in: ([\d:.]+)", line)
            if match3:
                print(f"[DEBUG] Matched other driver lap: {line.strip()}")
                name = match3.group(1)
                lap_num = int(match3.group(2))  # Parse lap number for other drivers
                lap_time = parse_lap_time(match3.group(3))
                line_handled = True


            timestamp = parse_timestamp(line)
            if not timestamp:
                timestamp = datetime.now()

            if line_handled and lap_time is not None and name:
                if not timestamp:
                    timestamp = datetime.now()

                # Lap number = current laps completed + 1
                lap_num = len(drivers[name]["lap_times"]) + 1

                # Append lap time and store timestamp at lap_num
                drivers[name]["lap_times"].append(lap_time)
                drivers[name]["laps"][lap_num] = timestamp

                print(f"[DEBUG] Added lap for {name} | Lap {lap_num} | Time: {lap_time:.3f}")
                print(f"[DEBUG] Lap {lap_num} stored for {name} at {timestamp.strftime('%H:%M:%S')}")

                # If this is your driver, update current_lap, fastest lap etc. as usual
                if name == user_name:
                    current_lap = lap_num
                    lap_times.append(lap_time)
                    if fastest_lap is None or lap_time < fastest_lap:
                        fastest_lap = lap_time

                    for lap_number, compound in pit_laps:
                        if current_lap == lap_number:
                            message = f"pit for '{compound}' this lap"
                            print(f">> {message}")
                            queue_tts_message(message, priority=1)
                            break
                               
                    # === GAP/ORDER MESSAGES (using JSON API instead of timestamps) ===
                    api_data = fetch_api_data(user_name)
                    if not api_data:
                        continue  # skip this frame if API didn't return usable data

                    # The API returns the drivers in order already
                    position_names = [entry["name"] for entry in api_data]

                    if user_name in position_names:
                        you_index = position_names.index(user_name)
                        you_time_diff = next((e["timeDiff"] for e in api_data if e["name"] == user_name), None)
                        leader_time_diff = api_data[0]["timeDiff"]

                        if you_index > 0:
                            ahead_entry = api_data[you_index - 1]
                            gap = (you_time_diff - ahead_entry["timeDiff"]) / 1000  # ms → seconds
                            if gap > 0.7:
                                print(f">> You are {gap:.3f}s behind {ahead_entry['name']}, push push")
                                queue_tts_message(f"You are {gap:.3f}s behind {ahead_entry['name']}, push push", priority=4)
                            elif gap < 0.7:
                                print(f">> You are in Tandem with {ahead_entry['name']}, be careful")
                                queue_tts_message(f"You are in Tandem with {ahead_entry['name']}, be careful", priority=2)

                        if you_index < len(api_data) - 1:
                            behind_entry = api_data[you_index + 1]
                            gap = (behind_entry["timeDiff"] - you_time_diff) / 1000  # ms → seconds
                            if gap < 0.7:
                                print(f">> You are in Tandem with {behind_entry['name']}, be careful")
                                queue_tts_message(f"You are in Tandem with {behind_entry['name']}, be careful", priority=2)

                    print(f"Lap {current_lap}: {lap_time:.3f}s | Fastest: {fastest_lap:.3f}s")

if __name__ == "__main__":
    # === 1. TTS thread ===
    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    tts_thread.start()

    # === 2. Voice control thread ===
    voice_thread = threading.Thread(target=new_listener, daemon=True)
    voice_thread.start()

    # === 3. API polling thread ===
    api_thread = threading.Thread(target=main_loop, daemon=True)
    api_thread.start()

    # === 4. Log file reader thread ===
    log_thread = threading.Thread(target=log_reader_loop, daemon=True)
    log_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting program.")
    
