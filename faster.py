import asyncio
import threading
import time
import io
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
from RealtimeSTT import AudioToTextRecorder
import edge_tts
import os
from google import genai
from google.genai import types

# --- Gemma 3 setup ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY in your environment.")
genai_client = genai.Client(api_key=GEMINI_API_KEY)

ENGINEER_SYSTEM_PROMPT = (
    "You are a calm, concise race engineer named Timothy Antonelli. "
    "Respond in short, plain English sentences only."
)

TRIGGER_WORDS = ["Timothy", "ta", "da"]
vcOutputIndex = 14  # playback device

# --- Latency logs ---
latency_log = {}

async def generate_llm_response(prompt: str) -> str:
    latency_log["llm_start"] = time.perf_counter()
    resp = genai_client.models.generate_content(
        model="gemma-3-1b",
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=
            f"{ENGINEER_SYSTEM_PROMPT}\nRequest: {prompt}"
        )])],
        config=types.GenerateContentConfig(
            max_output_tokens=50,
            temperature=0.4,
            top_p=0.9,
            response_mime_type="text/plain"
        ),
    )
    text = (resp.text or "").strip()
    latency_log["llm_end"] = time.perf_counter()
    return text

async def play_edge_tts(text: str):
    latency_log["tts_start"] = time.perf_counter()
    voice = "en-GB-SoniaNeural"
    rate = "+20%"
    tts = edge_tts.Communicate(text=text, voice=voice, rate=rate)
    mp3_bytes = b""
    async for chunk in tts.stream():
        if chunk["type"] == "audio":
            mp3_bytes += chunk["data"]
    audio = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / (2 ** 15)
    if audio.channels == 1:
        samples = np.column_stack((samples, samples))
    elif audio.channels > 2:
        samples = samples[:, :2]
    sd.default.device = vcOutputIndex
    sd.play(samples, samplerate=audio.frame_rate)
    sd.wait()
    latency_log["tts_end"] = time.perf_counter()
    print(f"--- Latency Log ---")
    print(f"Speech start → LLM start: {latency_log['llm_start'] - latency_log['speech_start']:.3f}s")
    print(f"LLM generation: {latency_log['llm_end'] - latency_log['llm_start']:.3f}s")
    print(f"LLM end → TTS start: {latency_log['tts_start'] - latency_log['llm_end']:.3f}s")
    print(f"TTS playback duration: {latency_log['tts_end'] - latency_log['tts_start']:.3f}s")
    print("-------------------")

async def handle_llm_and_tts(transcript: str):
    llm_text = await generate_llm_response(transcript)
    await play_edge_tts(llm_text)

def main():
    # --- Setup asyncio loop ---
    main_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(main_loop)
    threading.Thread(target=main_loop.run_forever, daemon=True).start()

    # --- Realtime STT callback ---
    def on_realtime_update(text: str):
        print(text)
        text_clean = text.strip()
        #if any(trigger.lower() in text_clean.lower() for trigger in TRIGGER_WORDS):
            #if text_clean.endswith(('.', '!', '?')):
        latency_log["speech_start"] = time.perf_counter()
        asyncio.run_coroutine_threadsafe(handle_llm_and_tts(text_clean), main_loop)

    # --- Recorder setup ---
    recorder = AudioToTextRecorder(
        input_device_index=1,
        enable_realtime_transcription=True,
        use_main_model_for_realtime=False,
        realtime_model_type="tiny.en",
        realtime_processing_pause=0.1,
        on_realtime_transcription_update=on_realtime_update,
    )

    print("🎤 Speak one sentence containing a trigger word...")
    while True:
        recorder.text(lambda txt: None)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Windows multiprocessing safe
    main()
