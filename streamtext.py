from google import genai
import asyncio
import edge_tts
import sounddevice as sd
import io
import numpy as np
from pydub import AudioSegment
from google.genai import types
VOICE = "en-US-AriaNeural"
client = genai.Client(api_key="AIzaSyDxlGRyUFnhMLivHk2IeVNzJjaupVFubWE")

# --- Async function to TTS a sentence into PCM samples ---
async def tts_to_samples(sentence: str):
    tts = edge_tts.Communicate(sentence, VOICE)
    mp3_bytes = io.BytesIO()

    async for chunk in tts.stream():
        if chunk["type"] == "audio":
            mp3_bytes.write(chunk["data"])

    # Decode MP3 -> PCM with pydub
    mp3_bytes.seek(0)
    audio = AudioSegment.from_file(mp3_bytes, format="mp3")

    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    samples /= np.iinfo(audio.array_type).max  # normalize to [-1, 1]

    return samples, audio.frame_rate


# --- Sentence streaming generator from Gemini ---
def stream_sentences(prompt: str):
    buffer = ""
    sentence_endings = {".", "?", "!"}

    for chunk in client.models.generate_content_stream(
        model="gemma-3-4b-it",
        contents=[prompt],
        config=types.GenerateContentConfig(
                max_output_tokens=100,
                temperature=1.3,
                top_p=0.9,
                response_mime_type="text/plain",  # Use plain text
            ),
    ):
        text = chunk.text
        buffer += text

        if any(buffer.endswith(p) for p in sentence_endings):
            yield buffer.strip()
            buffer = ""

    if buffer.strip():
        yield buffer.strip()


# --- Playback worker: consumes queued audio and plays sequentially ---
async def playback_worker(queue: asyncio.Queue, device=None):
    while True:
        item = await queue.get()
        if item is None:  # poison pill to stop
            break

        samples, sr = item
        sd.play(samples, samplerate=sr, device=device)
        sd.wait()  # wait until sentence finishes
        queue.task_done()


# --- Main driver ---
async def main():
    prompt = "Tell me a short story about a dragon who learns to play the guitar."
    device_index = None  # set to e.g. 14 if you want a specific device

    queue = asyncio.Queue()

    # Start playback worker in background
    playback_task = asyncio.create_task(playback_worker(queue, device=device_index))

    print("🎤 Gemini generating + queuing sentences...")

    # Stream Gemini sentences as they arrive
    async def produce_sentences():
        for sentence in stream_sentences(prompt):
            print(f"➡️ New sentence: {sentence}")
            samples, sr = await tts_to_samples(sentence)
            await queue.put((samples, sr))

        # When done, stop playback worker
        await queue.put(None)

    await produce_sentences()
    await playback_task  # wait until playback finishes


if __name__ == "__main__":
    asyncio.run(main())