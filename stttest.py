import time
from RealtimeSTT import AudioToTextRecorder

# Globals to track timing
last_realtime_timestamp = None
final_transcript_timestamp = None

realtime_text = ""

def process_realtime_update(text: str):
    global realtime_text
    global last_realtime_timestamp
    realtime_text = text
    now = time.perf_counter()
    last_realtime_timestamp = now
    print(f"[Realtime +{now-start_time:.2f}s] {text}")

def process_stabilized(text: str):
    global final_transcript_timestamp
    final_transcript_timestamp = time.perf_counter()
    print(f"[Final transcript +{final_transcript_timestamp-start_time:.2f}s] {text}")

def process_vad():
    
    print(f"[vad text: {realtime_text}")

if __name__ == '__main__':
    # Create recorder with realtime enabled, tiny.en for speed
    recorder = AudioToTextRecorder(
        enable_realtime_transcription=True,
        use_main_model_for_realtime=False,
        realtime_model_type="base.en",
        realtime_processing_pause=0.1,
        on_realtime_transcription_update=process_realtime_update,
        on_realtime_transcription_stabilized=process_stabilized,
        print_transcription_time=True,
        on_vad_stop=process_vad,
        
    )

    print("🎤 Speak one sentence...")

    start_time = time.perf_counter()
    # Record and transcribe one sentence (batch + stabilized)
    final_text = recorder.text(process_stabilized)

    # Print latency difference between last realtime and final transcript
    if last_realtime_timestamp is not None and final_transcript_timestamp is not None:
        lag = final_transcript_timestamp - last_realtime_timestamp
        print("\n=== Latency Summary ===")
        print(f"Last realtime text timestamp: +{last_realtime_timestamp - start_time:.2f}s")
        print(f"Final transcript timestamp:  +{final_transcript_timestamp - start_time:.2f}s")
        print(f"Time lag between last realtime text and final transcript: {lag:.2f}s")
