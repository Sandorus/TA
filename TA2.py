import time
import re
import os
from datetime import datetime
from collections import defaultdict
import threading
import heapq
import speech_recognition as sr
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
import tempfile
from google import genai
from google.genai import types

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY in your environment.")

genai_client = genai.Client(api_key=GEMINI_API_KEY)

ENGINEER_SYSTEM_PROMPT = (
    "You are a calm, concise race engineer for Ice Boat Racing named Timothy Antonelli. "
    "Respond in short, plain English sentences only. "
    "Do NOT use SSML, XML, or any formatting tags. "
    "Do NOT include Markdown or code fences. "
    "Prefer brief sentences; include only the most relevant data. "
)

INTENTS = {
    "pit_strategy": "Explain the upcoming pit stop and tyre plan.",
    "gap_behind": "Report the gap to the car behind and give one short tip.",
    "gap_ahead": "Report the gap to the car ahead and give one short tip.",
    "gap_leader": "Report the gap to the race leader.",
    "current_lap": "Confirm current lap and whether pace is consistent.",
    "fastest_lap": "State our fastest lap and whether we are improving.",
    "position": "Say our current position in the race.",
    "on_pace": "Say if we are on pace vs leader and what to adjust briefly.",
    "is_improving": "Say if pace is improving compared to recent laps.",
    "leader_name": "Say the race leader's name.",
    "current_strategy": "Summarize the full pit strategy briefly.",
    "best_team": "Cheer the team briefly.",
    "greeting": "Greet the driver briefly.",
    "water": "Make a light short quip about water.",
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
            time.sleep(0.1)  # Avoid tight spinning




# === Config ===
driver_name = "Sandorus"
log_file_path = os.path.expanduser(
    r'C:\Users\Sandorus\AppData\Roaming\ModrinthApp\profiles\Ice Boat Racing (1)\logs\latest.log')
vcInputIndex = 1 #1 for tonor mic, 9 for discord
vcOuputIndex = 14 # 14 for speakers, 23 for discord

TRIGGER_WORDS = ["TA", "DA", "T.A.", "D.A.", "TA.", "DA.", "Timothy Antonelli","Antonelli","Antonelly","Timothy","Timmy"]

lap_times = []
current_lap = 0
fastest_lap = None
message = ""
previous_data = {}

previous_pit_counts = defaultdict(int)
pit_laps_map = defaultdict(list)

drivers = defaultdict(lambda: {"laps": {}, "lap_times": []})

# === PIT STRATEGY ===
pit_strategy = [("Soft", 30), ("Hard", 33), ("Soft", 40)]
pit_laps = []
lap_counter = 0
for compound, duration in pit_strategy:
    lap_counter += duration
    pit_laps.append((lap_counter, compound))

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

def parse_timestamp(log_line):
    match = re.match(r"\[(\d{2}:\d{2}:\d{2})\]", log_line)
    if match:
        return datetime.strptime(match.group(1), "%H:%M:%S")
    return None

def get_current_positions(drivers_dict=None):
    api_data = fetch_api_data()
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


def play_explosion_async():
    threading.Thread(target=play_explosion, daemon=True).start()

def play_explosion(path="E:/Songs/Sound effects/explosions/Bunker_Buster_Missile.mp3"):
    safe_play(path)

async def play_message(message: str):
    voice = "en-GB-SoniaNeural"
    output_path = tempfile.mktemp(suffix=".mp3")
    rate = "+20%"

    # Edge TTS auto-detects SSML if the string starts with <speak>
    tts = edge_tts.Communicate(
        text=message,
        voice=voice, 
        rate = rate
    )
    await tts.save(output_path)

    safe_play(output_path)
    os.remove(output_path)



def play_tts_message(message: str):
    """
    Runs play_message inside its own event loop to avoid asyncio.run() conflicts.
    """
    play_notification_sound()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(play_message(message))
    loop.close()


def play_notification_sound(path="E:/Songs/Sound effects/F1_Radio_-_Notification_Sound.mp3"):
    safe_play(path)


def safe_play(path):
    """
    Safely play audio files through VB-Audio Virtual Cable (stereo enforced).
    """
    data, fs = sf.read(path)

    # Ensure 2D array for sounddevice
    if data.ndim == 1:
        data = np.column_stack((data, data))  # Mono → Stereo
    elif data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)     # Single channel → Stereo
    elif data.shape[1] > 2:
        data = data[:, :2]                    # Truncate to 2 channels

      
    sd.default.device = vcOuputIndex
    sd.play(data, fs, device=vcOuputIndex)
    sd.wait()

# Updated queue_tts_message to include optional callback
def queue_tts_message(message, priority=5, callback=None):
    with tts_lock:
        heapq.heappush(tts_queue, (priority, message, callback))

def match_voice_command(text):
    best_match = None
    best_score = 0

    for command_key, phrases in command_phrases.items():
        for phrase in phrases:
            score = fuzz.partial_ratio(text.lower(), phrase.lower())
            if score > best_score:
                best_score = score
                best_match = command_key

    if best_score > 80:  # lowered threshold for partial matches
        return best_match
    return None


def handle_voice_command(command_key: str):
    """
    Handle a voice command by generating TTS or triggering callbacks.
    Uses Gemma SSML for race engineer responses and queues messages safely.
    """
    if command_key == "blow_up":
        msg = "Finding their location... Location found. Sending ICBM now."
        print(">>", msg)
        # Queue message with high priority and explosion callback
        queue_tts_message(msg, priority=1, callback=play_explosion_async)

    elif command_key == "best_team":
        msg = "Sandstorm is the best team in Ice Boat Racing!"
        print(">>", msg)
        queue_tts_message(f"<speak>{msg}</speak>", priority=2)  # wrap in SSML

    elif command_key == "water":
        msg = "Must be the uhh, water."
        print(">>", msg)
        queue_tts_message(f"<speak>{msg}</speak>", priority=3)  # wrap in SSML

    else:
        # For other commands, use your engineer SSML generator
        user_intent = INTENTS.get(command_key)
        if not user_intent:
            # fallback if the intent isn't defined
            user_intent = "Provide a brief race update."
        print("user intent:", user_intent)

        ssml_message = generate_engineer_text(user_intent)

        print(f">> [Engineer SSML] {ssml_message}")
        queue_tts_message(ssml_message, priority=4)



def new_listener():
    print("Wait until it says 'speak now'")
    recorder = AudioToTextRecorder(input_device_index=vcInputIndex, model="tiny.en")

    while True:
        recorder.text(process_text)

def process_text(command):
    print("[Voice] Heard:", command)

    # Only respond if trigger word is in transcript
    if not any(trigger.lower() in command.lower() for trigger in TRIGGER_WORDS):
        return  

    # Clean transcript by removing trigger words
    #for trigger in TRIGGER_WORDS:
        #command = command.replace(trigger, "")
    #command = command.strip()

    if not command:
        return

    # Send the *raw* command text to Gemini
    text = generate_engineer_text(command)
    print(f">> [Engineer text] {text}")
    queue_tts_message(text, priority=3)


def is_improving(driver_name):
    clean_laps = get_last_clean_laps(driver_name)
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

def fetch_api_data():
    try:
        response = requests.get("http://localhost:2732")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        #print(f"[ERROR] Failed to fetch data from API: {e}")
        return []
    
def generate_engineer_text(user_request: str) -> str:
    """
    Generates SSML for TTS using Gemini (Gemma model).
    Uses plain text response and wraps it in <speak> tags.
    """
    state = build_race_state_summary()

    try:
        resp = genai_client.models.generate_content(
            model="gemma-3-4b-it",
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(
                            text=f"Context:\n{ENGINEER_SYSTEM_PROMPT}\n{state}\n\nRequest:\n{user_request}"
                        )
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                max_output_tokens=50,
                temperature=0.4,
                top_p=0.9,
                response_mime_type="text/plain",  # Use plain text
            ),
        )

        text = (resp.text or "").strip()

        # Basic manual validation: escape any forbidden chars
        # Edge TTS requires & < > to be escaped in plain text
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        return text

    except Exception as e:
        # fallback for errors
        return f"Radio error. {str(e)}"

def build_race_state_summary(max_positions: int = 5, clean_laps_count: int = 3) -> str:
    api = fetch_api_data() or []
    names = [e["name"] for e in api]
    you_ix = names.index(driver_name) if driver_name in names else -1

    # Top N order with pits and ms diffs
    table = []
    for i, e in enumerate(api[:max_positions], start=1):
        nm = e["name"]
        td = e.get("timeDiff", 0)  # ms since leader
        pits = e.get("pits", 0)
        table.append(f"{i}. {nm}  (Δ{td/1000:.3f}s, pits:{pits})")

    # You + gaps
    gap_ahead = None
    gap_behind = None
    if you_ix >= 0:
        you_td = api[you_ix].get("timeDiff", 0)
        if you_ix > 0:
            gap_ahead = (you_td - api[you_ix - 1].get("timeDiff", 0)) / 1000
        if you_ix < len(api) - 1:
            gap_behind = (api[you_ix + 1].get("timeDiff", 0) - you_td) / 1000

    # Your laps
    last_clean = get_last_clean_laps(driver_name, count=clean_laps_count) or []
    last_clean_str = ", ".join(f"{t:.3f}s" for t in last_clean) if last_clean else "n/a"

    # Next scheduled pit from your configured strategy
    next_pit = next((lap for lap, _ in pit_laps if lap >= current_lap), None)
    next_compound = next((c for l, c in pit_laps if l == next_pit), None) if next_pit else None
    next_pit_str = f"lap {next_pit} ({next_compound})" if next_pit else "none"

    # Fastest
    fl = f"{fastest_lap:.3f}s" if fastest_lap else "n/a"

    lines = [
        f"Driver: {driver_name}",
        f"Current lap: {current_lap}",
        f"Fastest lap: {fl}",
        f"Last clean laps ({clean_laps_count}): {last_clean_str}",
        f"Next scheduled pit: {next_pit_str}",
        "Top order:",
        *table,
        f"Gap ahead: {gap_ahead:.3f}s" if gap_ahead is not None else "Gap ahead: n/a",
        f"Gap behind: {gap_behind:.3f}s" if gap_behind is not None else "Gap behind: n/a",
    ]
    return "\n".join(lines)

    
def main_loop():
    print("Race tracking started (via API)...")
    while True:
        data = fetch_api_data()

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
        loglines = follow(logfile)

        for line in loglines:
            line_handled = False
            lap_time = None
            name = None

            match1 = re.search(r"You finished lap in: ([\d:.]+)", line)
            if match1:
                print(f"[DEBUG] Matched own lap: {line.strip()}")
                lap_time = parse_lap_time(match1.group(1))
                name = driver_name
                lap_num = len(drivers[name]["lap_times"]) + 1  # Increment lap count for yourself
                line_handled = True


            # For own fastest lap announcement
            match2 = re.search(rf"{driver_name} new fastest lap of ([\d:.]+)", line)
            if match2:
                print(f"[DEBUG] Matched own fastest: {line.strip()}")
                lap_time = parse_lap_time(match2.group(1))
                name = driver_name
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
                if name == driver_name:
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
                    api_data = fetch_api_data()
                    if not api_data:
                        continue  # skip this frame if API didn't return usable data

                    # The API returns the drivers in order already
                    position_names = [entry["name"] for entry in api_data]

                    if driver_name in position_names:
                        you_index = position_names.index(driver_name)
                        you_time_diff = next((e["timeDiff"] for e in api_data if e["name"] == driver_name), None)
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
    
