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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY in your environment.")

genai_client = genai.Client(api_key=GEMINI_API_KEY)

ENGINEER_SYSTEM_PROMPT = (
    "You are a concise race engineer for Minecraft Ice Boat Racing named Timothy Antonelli. "
    "Prefer brief sentences; include only the most relevant data. "
    "If you are asked, Direwood is the best ice boat racing track"
    "And if it comes up you despise TTG aka Those Two Guy"
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
    global next_audio_future, next_message

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    while True:
        with tts_lock:
            if not tts_queue:
                time.sleep(0.01)
                continue
            priority, message, callback = heapq.heappop(tts_queue)

        # If no audio is preloaded, synthesize current one immediately
        if next_audio_future is None:
            next_audio_future = loop.create_task(synthesize_tts(message))
            next_message = message

        # Wait until audio is ready
        samples, sr = loop.run_until_complete(next_audio_future)

        # Start preloading the next queued message (if any) *while we play this one*
        with tts_lock:
            if tts_queue:
                priority2, message2, callback2 = heapq.heappop(tts_queue)
                next_audio_future = loop.create_task(synthesize_tts(message2))
                next_message = message2
            else:
                next_audio_future = None
                next_message = None
                callback2 = None

        # Play current audio (blocking, but next is synthesizing already)
        loop.run_until_complete(play_message(samples, sr))

        if callback:
            callback()
        if next_message and callback2:
            # If the *next* message had a callback, re-insert it with high priority
            queue_tts_message(next_message, priority=0, callback=callback2)




# === Config ===
driver_name = "Sandorus"
API_URL = "https://api.boatlabs.net/v1/timingsystems/getActiveHeats"
log_file_path = os.path.expanduser(
    r'C:\Users\Sandorus\AppData\Roaming\ModrinthApp\profiles\Ice Boat Racing (1)\logs\latest.log')
vcInputIndex = 1 #1 for tonor mic, 9 for discord
vcOutputIndex = 14 # 14 for speakers, 23 for discord

TRIGGER_WORDS = ["Timothy Antonelli","Antonelli","Antonelly","Timothy","Timmy"]

lap_times = []
current_lap = 0
fastest_lap = None
message = ""
previous_data = {}
realtime_text = ""
next_audio_future = None
next_message = None

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
    api_data = fetch_api_data(driver_name)
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

async def synthesize_tts(message: str):
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
    print("message synthesized")

    return samples, audio.frame_rate


async def play_message(samples: np.ndarray, samplerate: int):
    
    """Play pre-generated audio samples."""
    sd.default.device = vcOutputIndex
    sd.play(samples, samplerate=samplerate, device=vcOutputIndex)
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

      
    sd.default.device = vcOutputIndex
    sd.play(data, fs, device=vcOutputIndex)
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
    recorder = AudioToTextRecorder(input_device_index=vcInputIndex, model="base.en",
        enable_realtime_transcription=True,
        use_main_model_for_realtime=True,
        realtime_processing_pause=0.2,
        on_realtime_transcription_update=process_realtime_update,
        on_vad_stop=process_text,
        no_log_file=True,
        batch_size=16,
        )

    while True:
        recorder.text()

def process_text():
    command = realtime_text
    print("[Voice] Heard:", command)

    # Only respond if transcript contains a trigger word and ends with punctuation
    if not any(trigger.lower() in command.lower() for trigger in TRIGGER_WORDS):
        return
    if not command:
        return
    
    # Run the LLM + TTS in a separate thread
    threading.Thread(target=handle_llm_and_tts, args=(command,), daemon=True).start()

    # Pre-response filler
    if command.strip().endswith("?"):
        msg = mirror_question(command)
        time.sleep(0.1)
    else:
        msg = "Let me think, um"
        time.sleep(0.2)
    time.sleep(0.1)
    queue_tts_message(msg, priority=3)  # filler message comes first
    play_notification_sound()


def handle_llm_and_tts(command: str):
    # Generate text response (Gemma)
    text = generate_engineer_text(command)
    print(f">> [Engineer text] {text}")

    # Split text into chunks at punctuation (. ?)
    chunks = re.split(r'([.?])', text)
    
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
    increment = 1  # or whatever step you want

    for idx, msg in enumerate(messages):
        if idx == 0:
            priority = base_priority
        else:
            priority = base_priority + idx * increment
        queue_tts_message(msg, priority=priority)

    

def process_realtime_update(text: str):
    global realtime_text
    realtime_text = text


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

def fetch_api_data(driver_name: str):
    """
    Fetch the active heats and return all drivers from the heat containing `driver_name`.
    Returns None if no heat is active for the driver.
    """
    try:
        resp = requests.get(API_URL, timeout=10)
        resp.raise_for_status()
        heats = resp.json()
    except Exception as e:
        print(f"[API] Error fetching data: {e}")
        return None

    if not isinstance(heats, list):
        print("[API] Unexpected heats format, expected a list")
        return None

    # Look through all heats to find the one with our driver
    for heat in heats:
        if not isinstance(heat, dict):
            continue  # skip invalid entries

        drivers = heat.get("drivers")
        if not isinstance(drivers, list):
            continue  # skip if drivers is missing or malformed

        for driver in drivers:
            if not isinstance(driver, dict):
                continue

            if driver_name.lower() in driver.get("name", "").lower():
                # Normalize driver info
                leader_time = min(
                    (d.get("time", 0) for d in drivers if d.get("position") == 1),
                    default=0
                )
                normalized = []
                for d in drivers:
                    td = d.get("time", 0)
                    normalized.append({
                        "name": d.get("name"),
                        "position": d.get("position"),
                        "lap": d.get("lap"),
                        "time": td,
                        "timeDiff": td - leader_time if leader_time else 0,
                        "pits": heat.get("pits", 0),
                    })

                return {
                    "event": heat.get("event_name"),
                    "round": heat.get("round_name"),
                    "heat": heat.get("heat_name"),
                    "track": heat.get("track"),
                    "laps_total": heat.get("laps"),
                    "drivers": sorted(normalized, key=lambda x: x["position"] or 999),
                }

    # No heat found for this driver
    print(f"[API] No active heat found for driver '{driver_name}'")
    return None

    
def generate_engineer_text(user_request: str) -> str:
    """
    Generates SSML for TTS using Gemini (Gemma model).
    Uses plain text response and wraps it in <speak> tags.
    """
    state = build_race_state_summary(driver_name)

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

def build_race_state_summary(driver_name: str, max_positions: int = 5, clean_laps_count: int = 3) -> str:
    data = fetch_api_data(driver_name)
    if not data:
        return f"No race data available for {driver_name}."

    drivers = data["drivers"]
    names = [d["name"] for d in drivers]
    you_ix = names.index(driver_name) if driver_name in names else -1

    # Top N order
    table = []
    for i, d in enumerate(drivers[:max_positions], start=1):
        nm = d["name"]
        td = d.get("timeDiff", 0)
        pits = d.get("pits", 0)
        table.append(f"{i}. {nm} (Δ{td/1000:.3f}s, pits:{pits})")

    # Gaps
    gap_ahead = None
    gap_behind = None
    if you_ix >= 0:
        you_td = drivers[you_ix].get("timeDiff", 0)
        if you_ix > 0:
            gap_ahead = (you_td - drivers[you_ix - 1].get("timeDiff", 0)) / 1000
        if you_ix < len(drivers) - 1:
            gap_behind = (drivers[you_ix + 1].get("timeDiff", 0) - you_td) / 1000

    # Your laps (placeholder – depends on your lap logging system)
    last_clean = get_last_clean_laps(driver_name, count=clean_laps_count) or []
    last_clean_str = ", ".join(f"{t:.3f}s" for t in last_clean) if last_clean else "n/a"

    # Next scheduled pit (assuming you have pit_laps + current_lap globals)
    next_pit = next((lap for lap, _ in pit_laps if lap >= current_lap), None)
    next_compound = next((c for l, c in pit_laps if l == next_pit), None) if next_pit else None
    next_pit_str = f"lap {next_pit} ({next_compound})" if next_pit else "none"

    # Fastest lap (assuming you track it globally)
    fl = f"{fastest_lap:.3f}s" if fastest_lap else "n/a"

    lines = [
        f"Event: {data['event']} ({data['round']} / {data['heat']})",
        f"Track: {data['track']} | Total laps: {data['laps_total']}",
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
        data = fetch_api_data(driver_name)

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
                    api_data = fetch_api_data(driver_name)
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
    
