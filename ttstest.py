import asyncio
import time
import io
import subprocess
from pathlib import Path
import tempfile
from pydub import AudioSegment
import edge_tts
import pyttsx3
import sounddevice as sd
import soundfile as sf
import numpy as np


# ========= Utilities ========= #

def play_wav_bytes(wav_bytes: bytes, device=14, block=True):
    """Play raw WAV bytes via sounddevice on a specific device index."""
    data, samplerate = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
    start_time = time.perf_counter()
    sd.play(data, samplerate=samplerate, device=device, blocking=block)
    return start_time


# ========= EDGE-TTS ========= #

VOICE = "en-US-AriaNeural"

async def edge_tts_mp3(text: str) -> bytes:
    """Generate MP3 audio bytes directly from Edge-TTS"""
    tts = edge_tts.Communicate(text=text, voice=VOICE)
    mp3_bytes = b""
    async for chunk in tts.stream():  # default MP3
        if chunk["type"] == "audio":
            mp3_bytes += chunk["data"]
    return mp3_bytes


def play_mp3_bytes(mp3_bytes: bytes, device_index: int = None, block=True) -> float:
    """Play MP3 bytes on a specific output device using sounddevice"""
    audio = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
    samples = np.array(audio.get_array_of_samples())
    np_audio = samples.astype(np.float32) / (2**15)

    if audio.channels > 1:
        np_audio = np_audio.reshape((-1, audio.channels))

    t_start = time.perf_counter()
    sd.play(np_audio, samplerate=audio.frame_rate, device=device_index, blocking=block)
    t_end = time.perf_counter()

    return t_start if block else t_end


def run_edge_tts(text: str, device=14) -> float:
    print("\n--- Edge TTS ---")
    t0 = time.perf_counter()
    mp3_bytes = asyncio.run(edge_tts_mp3(text))
    t1 = play_mp3_bytes(mp3_bytes, device_index=device, block=True)
    latency = t1 - t0
    print(f"Edge-TTS latency to first sound: {latency:.3f}s")
    return latency


# ========= PIPER ========= #
'''
PIPER_CLI = "piper"  # assumes piper binary is in PATH
MODEL_PATH = Path("models/en_GB-northern_english_male-medium.onnx")  # adjust to your model

def run_piper(text: str, device=14) -> float:
    print("\n--- Piper ---")
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        out_wav = Path(td) / "out.wav"
        cmd = [PIPER_CLI, "--model", str(MODEL_PATH), "--output_file", str(out_wav)]
        subprocess.run(cmd, input=text.encode("utf-8"), check=True)
        wav_bytes = out_wav.read_bytes()
    t1 = play_wav_bytes(wav_bytes, device=device, block=True)
    latency = t1 - t0
    print(f"Piper latency to first sound: {latency:.3f}s")
    return latency '''


# ========= PYTTSX3 ========= #

def run_pyttsx3(text: str, device=14) -> float:
    print("\n--- pyttsx3 ---")
    t0 = time.perf_counter()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        wav_path = Path(tmp.name)
    engine = pyttsx3.init()
    engine.save_to_file(text, str(wav_path))
    engine.runAndWait()
    wav_bytes = wav_path.read_bytes()
    t1 = play_wav_bytes(wav_bytes, device=device, block=True)
    latency = t1 - t0
    print(f"pyttsx3 latency to first sound: {latency:.3f}s")
    wav_path.unlink(missing_ok=True)
    return latency


# ========= Main ========= #

def main():
    text = "Testing streaming TTS with Edge-TTS and sounddevice, measuring latency to first sound. Testing streaming TTS with Edge-TTS and sounddevice, measuring latency to first sound."
    device_index = 14  # change this to your target output device

    lat_edge = run_edge_tts(text, device=device_index)
    #lat_piper = run_piper(text, device=device_index)
    lat_pyttsx3 = run_pyttsx3(text, device=device_index)

    print("\n=== Summary (lower is better) ===")
    print(f"Edge-TTS : {lat_edge:.3f}s")
    #print(f"Piper    : {lat_piper:.3f}s")
    print(f"pyttsx3  : {lat_pyttsx3:.3f}s")

if __name__ == "__main__":
    main()
