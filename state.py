import threading

ui_state = {
    "assistant_state": "idle",   # idle | listening | thinking | speaking
    "realtime_stt": "",
    "last_engineer_text": "",
    "current_lap": 0,
    "fastest_lap": None,
    "pit_delta": 22.0,
    "drivers": {},
}

ui_lock = threading.Lock()
