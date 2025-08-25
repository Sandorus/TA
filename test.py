import asyncio
import edge_tts

async def main():
    tts = edge_tts.Communicate(
        text="Hello, this is a test.",
        voice="en-GB-SoniaNeural"
    )
    await tts.save("test.mp3")

asyncio.run(main())
