import time
import re
import os
from datetime import datetime
from collections import defaultdict
import pyttsx3
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
vcInputIndex = 1
sd.default.device = (14, None)  # (output, input)
TRIGGER_WORDS = ["TA", "DA", "T.A.", "D.A.", "TA.", "DA.", "Timothy Antonelli","Antonelli","Timothy"]

lap_times = []
current_lap = 0
fastest_lap = None
message = ""
previous_data = {}

drivers = defaultdict(lambda: {"laps": {}, "lap_times": []})

# === PIT STRATEGY ===
pit_strategy = [("Soft", 30), ("Hard", 33), ("Soft", 40)]
pit_laps = []
lap_counter = 0
for compound, duration in pit_strategy:
    lap_counter += duration
    pit_laps.append((lap_counter, compound))

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

def get_current_positions(drivers_dict):
    finishing_times = []
    for name, data in drivers_dict.items():
        last_lap = len(data["lap_times"])
        if last_lap > 0 and last_lap in data["laps"]:
            finishing_times.append((name, data["laps"][last_lap], last_lap))
        else:
            print(f"[DEBUG] Skipping {name}: lap {last_lap} timestamp not found")
    return sorted(finishing_times, key=lambda x: x[1])

def play_explosion_async():
    threading.Thread(target=play_explosion, daemon=True).start()

def play_explosion():
    data, fs = sf.read('E:/Songs/Sound effects/explosions/Bunker_Buster_Missile.mp3')
    sd.play(data, fs)
    sd.wait()


def play_tts_message(message):
    data, fs = sf.read('E:/Songs/Sound effects/F1_Radio_-_Notification_Sound.mp3')
    sd.play(data, fs)
    sd.wait()
    engine = pyttsx3.init()
    engine.say(message)
    engine.runAndWait()

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


def handle_voice_command(command_key):
    if command_key == "pit_strategy":
        if pit_laps:
            next_pit = next((lap for lap, _ in pit_laps if lap >= current_lap), None)
            if next_pit:
                compound = next(c for l, c in pit_laps if l == next_pit)
                msg = f"You will pit on lap {next_pit} for {compound}"
            else:
                msg = "No more pit stops scheduled"
        else:
            msg = "Pit strategy not available"
        print(">>", msg)
        queue_tts_message(msg, priority=3)

    elif command_key == "blow_up":
        msg = "Finding their location........ Location found. Sending ICBM Now"
        print(">>", msg)
        queue_tts_message(msg, priority=1, callback=play_explosion_async)

    elif command_key == "gap_behind":
        positions = get_current_positions(drivers)
        names = [n for n, _, _ in positions]
        if driver_name in names:
            you_index = names.index(driver_name)
            if you_index < len(names) - 1:
                _, behind_time, _ = positions[you_index + 1]
                you_time = drivers[driver_name]["laps"].get(current_lap)
                if you_time:
                    gap = (behind_time - you_time).total_seconds()
                    msg = f"Gap to car behind is {gap:.2f} seconds"
                else:
                    msg = "No lap data for gap"
                print(">>", msg)
                queue_tts_message(msg, priority=3)

    elif command_key == "gap_leader":
        positions = get_current_positions(drivers)
        if positions:
            leader_name, leader_time, _ = positions[0]
            if leader_name != driver_name:
                you_time = drivers[driver_name]["laps"].get(current_lap)
                if you_time:
                    gap = (you_time - leader_time).total_seconds()
                    msg = f"Gap to leader is {gap:.2f} seconds"
                else:
                    msg = "No lap data for leader gap"
                print(">>", msg)
                queue_tts_message(msg, priority=3)

    elif command_key == "current_lap":
        msg = f"You are on lap {current_lap}"
        print(">>", msg)
        queue_tts_message(msg, priority=3)

    elif command_key == "fastest_lap":
        if fastest_lap:
            msg = f"Your fastest lap is {fastest_lap:.3f} seconds"
        else:
            msg = "No lap data available yet"
        print(">>", msg)
        queue_tts_message(msg, priority=3)

    elif command_key == "position":
        positions = get_current_positions(drivers)
        for i, (name, _, _) in enumerate(positions):
            if name == driver_name:
                msg = f"You are in position {i + 1}"
                print(">>", msg)
                queue_tts_message(msg, priority=3)
                break
        else:
            queue_tts_message("You are not in the standings yet", priority=5)

    elif command_key == "on_pace":
        positions = get_current_positions(drivers)
        if positions:
            leader_name = positions[0][0]
            leader_avg = compute_last2_avg(drivers[leader_name]["lap_times"])
            you_avg = compute_last2_avg(drivers[driver_name]["lap_times"])
            if leader_avg is not None and you_avg is not None:
                diff = abs(you_avg - leader_avg)
                if you_avg < leader_avg:
                    msg = f"You are {diff:.3f} seconds faster than the leader, keep it up"
                else:
                    msg = f"You are {diff:.3f} seconds off pace, push harder"
            else:
                msg = "Not enough data to calculate pace"
        else:
            msg = "No race data yet"
        print(">>", msg)
        queue_tts_message(msg, priority=3)

    elif command_key == "current_strategy":
        strategy_list = ", then ".join([f"{compound} for {laps} laps" for compound, laps in pit_strategy])
        msg = f"Your strategy is: {strategy_list}"
        print(">>", msg)
        queue_tts_message(msg, priority=3)

    elif command_key == "best_team":
        msg = "Sandstorm is the best team in Ice Boat Racing!"
        print(">>", msg)
        queue_tts_message(msg, priority=1)

    elif command_key == "greeting":
        msg = "Hi, I can hear you!"
        print(">>", msg)
        queue_tts_message(msg, priority=2)
        
    

    else:
        msg = random.choice([
            "Sorry, I didn't understand that command.",
            f"What was that {driver_name}?",
            f"{driver_name}, please repeat."
        ])
        print(">>", msg)
        queue_tts_message(msg, priority=5)


def voice_command_listener():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("Voice command listener started. Speak clearly.")

    while True:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            try:
                print("[Voice] Listening...")
                audio = recognizer.listen(source, timeout=5)
                command = recognizer.recognize_google(audio)
                print("[Voice] Heard:", command)
                matched = match_voice_command(command)
                if matched:
                    handle_voice_command(matched)
                else:
                    msg = random.choice([
                        "Sorry, I didn't understand that command.",
                        f"What was that {driver_name}?",
                        f"{driver_name}, please repeat."
                    ])
                    queue_tts_message(msg, priority=6)
            except sr.UnknownValueError:
                print("[Voice] Could not understand audio")
            except sr.RequestError as e:
                print("[Voice] Recognition error:", e)
            except sr.WaitTimeoutError:
                pass
            time.sleep(1)

def new_listener():
    print("Wait until it says 'speak now'")
    recorder = AudioToTextRecorder(input_device_index=vcInputIndex, model="base.en")

    while True:
        recorder.text(process_text)

def process_text(command):
    print("[Voice] Heard:", command)

    if not any(trigger.lower() in command.lower() for trigger in TRIGGER_WORDS):
        return  # Ignore if no trigger word is detected

    # Clean the command by removing trigger words
    for trigger in TRIGGER_WORDS:
        command = command.replace(trigger, "")
    command = command.strip()

    matched = match_voice_command(command)
    if matched:
        handle_voice_command(matched)
    else:
        msg = random.choice([
            "Sorry, I didn't understand that command.",
            f"What was that {driver_name}?",
            f"{driver_name}, please repeat."
        ])
        queue_tts_message(msg, priority=6)

        

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

def fetch_api_data():
    try:
        response = requests.get("http://localhost:2732")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"[ERROR] Failed to fetch data from API: {e}")
        return []
    
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

        time.sleep(1)  # Poll every second (adjust as needed)

if __name__ == "__main__":
    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    tts_thread.start()

    voice_thread = threading.Thread(target=new_listener, daemon=True)
    voice_thread.start()


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

                    # === GAP/ORDER MESSAGES ===
                    positions = get_current_positions(drivers)
                    position_names = [n for n, _, _ in positions]

                    if driver_name in position_names:
                        you_index = position_names.index(driver_name)
                        leader_name = positions[0][0]

                        leader_avg = compute_last2_avg(drivers[leader_name]["lap_times"])
                        you_avg = compute_last2_avg(drivers[driver_name]["lap_times"])

                        if you_avg is not None and leader_avg is not None:
                            diff = round(abs(you_avg - leader_avg), 3)
                            if you_avg > leader_avg:
                                print(f">> You are {diff:.3f}s off pace")
                                message = (f">> You are {diff:.3f}s off pace")
                                queue_tts_message(message, priority=5)
                            elif you_avg < leader_avg:
                                print(f">> You are {diff:.3f}s faster than the leader, keep it up")
                                message = (f">> You are {diff:.3f}s faster than the leader, keep it up")
                                queue_tts_message(message, priority=4)

                        you_lap = current_lap

                        if you_index > 0:
                            ahead_name, ahead_timestamp, _ = positions[you_index - 1]
                            ahead_laps = drivers[ahead_name]["laps"]
                            if you_lap in ahead_laps:
                                gap = (timestamp - ahead_laps[you_lap]).total_seconds()
                                print(f">> You are {gap:.3f}s behind {ahead_name}, push push")
                                message = (f">> You are {gap:.3f}s behind {ahead_name}, push push")
                                queue_tts_message(message, priority=4)
                        if you_index < len(positions) - 1:
                            behind_name, behind_timestamp, _ = positions[you_index + 1]
                            if you_lap in drivers[driver_name]["laps"]:
                                gap = (behind_timestamp - drivers[driver_name]["laps"][you_lap]).total_seconds()
                                if gap < 0.7:
                                    print(f">> You are in Tandem, be careful")
                                    message = (f">> You are in Tandem, be careful")
                                    queue_tts_message(message, priority=2)

                    print(f"Lap {current_lap}: {lap_time:.3f}s | Fastest: {fastest_lap:.3f}s")
