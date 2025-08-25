import asyncio
import edge_tts
import sounddevice as sd
import soundfile as sf
import tempfile
import os
import numpy as np

async def play_tts(message: str):
    voice = "en-US-GuyNeural"
    output_path = tempfile.mktemp(suffix=".mp3")

    # Generate TTS audio
    tts = edge_tts.Communicate(text=message, voice=voice)
    await tts.save(output_path)

    # Read audio file
    data, fs = sf.read(output_path)

    # Ensure stereo (2 channels) for compatibility
    if data.ndim == 1:
        data = np.column_stack([data, data])
    elif data.shape[1] > 2:
        data = data[:, :2]

    # Play audio
    sd.play(data, fs,device=18)
    sd.wait()

    # Cleanup
    os.remove(output_path)

def main():
    message = "Hello, this is a test of the text to speech system."
    asyncio.run(play_tts(message))

if __name__ == "__main__":
    main()
