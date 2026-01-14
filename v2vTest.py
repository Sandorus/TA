import asyncio
import os
import pyaudio
from pynput import mouse
from google import genai
from google.genai import types

import time

SESSION_RESTART_AFTER = 14 * 60  # 14 minutes (seconds)

session_start_time = None
reconnect_requested = asyncio.Event()
shutdown_event = asyncio.Event()


# =========================
# Environment / Gemini
# =========================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY in your environment.")

client = genai.Client(api_key=GEMINI_API_KEY)

MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
CONFIG = {
    "response_modalities": ["AUDIO"],
    "system_instruction": "You are a helpful and friendly AI assistant.",
    "realtime_input_config": {
        "automatic_activity_detection": {"disabled": True}
    },
}
activity_started = False

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

# =========================
# Push-to-talk state
# =========================

ptt_active = asyncio.Event()   # True while Mouse5 held

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
    """Send audio and activity start/end messages."""
    global activity_started

    while True:
        # Wait for PTT
        await ptt_active.wait()

        # Start activity once per press
        if not activity_started:
            await session.send_realtime_input(
                activity_start=types.ActivityStart()
            )
            activity_started = True

        # Send audio while held
        while ptt_active.is_set():
            msg = await audio_queue_mic.get()
            await session.send_realtime_input(
                audio=types.Blob(
                    data=msg["data"],
                    mime_type="audio/pcm;rate=16000",
                )
            )

        # PTT released → end activity
        if activity_started:
            await session.send_realtime_input(
                activity_end=types.ActivityEnd()
            )
            activity_started = False
            if reconnect_requested.is_set():
                print("🔁 Reconnecting after user finished speaking")
                raise asyncio.CancelledError



# =========================
# Receive Gemini audio
# =========================

async def receive_audio(session):
    """Receive audio responses from Gemini."""
    while True:
        turn = session.receive()
        async for response in turn:
            if response.server_content and response.server_content.model_turn:
                for part in response.server_content.model_turn.parts:
                    if part.inline_data and isinstance(part.inline_data.data, bytes):
                        audio_queue_output.put_nowait(part.inline_data.data)

        # Clear playback if interrupted
        while not audio_queue_output.empty():
            audio_queue_output.get_nowait()

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


# =========================
# Main
# =========================

async def run():
    global session_start_time, reconnect_requested

    loop = asyncio.get_running_loop()
    start_mouse_listener(loop)

    while True:
        reconnect_requested.clear()

        try:
            async with client.aio.live.connect(
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

        except asyncio.CancelledError:
            print("Session restarting…")
            await asyncio.sleep(0.5)  # small grace delay
            continue

        except Exception as e:
            print("Session error:", e)
            await asyncio.sleep(2)


# =========================
# Entry point
# =========================

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("Interrupted by user.")
