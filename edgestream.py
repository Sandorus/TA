import asyncio
import edge_tts
import io
import time
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
from threading import Lock

VOICE = "en-US-AriaNeural"

class StreamingPlayer:
    """Thread-safe buffer playback for streaming TTS chunks"""
    def __init__(self, samplerate=24000, channels=1, device_index=None):
        self.buffer = np.zeros((0, channels), dtype=np.float32)
        self.lock = Lock()
        self.samplerate = samplerate
        self.channels = channels
        self.device_index = device_index
        self.playing = True

    def add_audio(self, pcm: np.ndarray):
        with self.lock:
            self.buffer = np.concatenate((self.buffer, pcm))

    def callback(self, outdata, frames, time_, status):
        with self.lock:
            if len(self.buffer) >= frames:
                outdata[:] = self.buffer[:frames]
                self.buffer = self.buffer[frames:]
            else:
                outdata[:len(self.buffer)] = self.buffer
                outdata[len(self.buffer):] = 0
                self.buffer = np.zeros((0, self.channels), dtype=np.float32)
                if not self.playing:
                    raise sd.CallbackStop()

    def start(self):
        self.stream = sd.OutputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            device=self.device_index,
            dtype='float32',
            callback=self.callback,
            blocksize=1024
        )
        self.stream.start()

    def stop(self):
        self.playing = False
        self.stream.stop()
        self.stream.close()


async def edge_tts_stream(text: str, player: StreamingPlayer):
    tts = edge_tts.Communicate(text=text, voice=VOICE)
    first_chunk_time = None
    async for chunk in tts.stream():
        if chunk["type"] == "audio":
            # Decode MP3 chunk to PCM
            audio = AudioSegment.from_file(io.BytesIO(chunk["data"]), format="mp3")
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / (2**15)
            if audio.channels > 1:
                samples = samples.reshape((-1, audio.channels))
            else:
                samples = samples.reshape((-1, 1))

            # Record timestamp of first audio chunk
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter()

            player.add_audio(samples)

    return first_chunk_time


def run_edge_tts_streaming(text: str, device_index=14):
    print("\n--- Edge TTS Streaming ---")
    player = StreamingPlayer(device_index=device_index)
    player.start()

    start_time = time.perf_counter()
    first_chunk_time = asyncio.run(edge_tts_stream(text, player))
    player.stop()
    end_time = time.perf_counter()

    if first_chunk_time:
        latency = first_chunk_time - start_time
        print(f"Latency to first audio chunk: {latency:.3f}s")
    print(f"Total TTS stream duration: {end_time - start_time:.3f}s")


if __name__ == "__main__":
    run_edge_tts_streaming(
        "Testing streaming TTS with Edge-TTS and sounddevice, measuring latency to first sound. Testing streaming TTS with Edge-TTS and sounddevice, measuring latency to first sound.",
        device_index=14
    )
