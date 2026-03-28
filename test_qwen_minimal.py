import os
import sys
import numpy as np
import librosa

CAPS_WRITER_DIR = r"C:\Users\shihu\Documents\software\CapsWriter-Offline-20260304\CapsWriter-Offline"
INTERNAL_DIR = os.path.join(CAPS_WRITER_DIR, "internal")
for path in [CAPS_WRITER_DIR, INTERNAL_DIR]:
    if path not in sys.path:
        sys.path.append(path)

from util.qwen_asr_gguf import create_asr_engine

def main():
    audio_path = r"test\test_dialogue.mp3"
    model_dir = os.path.join(CAPS_WRITER_DIR, "models", "Qwen3-ASR", "Qwen3-ASR-1.7B")
    
    print("Loading 5s audio...")
    try:
        audio, _ = librosa.load(audio_path, sr=16000, duration=5.0)
    except Exception as e:
        print(f"Failed to load audio: {e}")
        return

    print("Initializing Qwen3-ASR...")
    try:
        engine = create_asr_engine(
            model_dir=model_dir,
            encoder_frontend_fn="qwen3_asr_encoder_frontend.int4.onnx",
            encoder_backend_fn="qwen3_asr_encoder_backend.int4.onnx",
            llm_fn="qwen3_asr_llm.q4_k.gguf",
            n_threads=1,
            vulkan_enable=False,
            verbose=True
        )
    except Exception as e:
        print(f"Failed to initialize engine: {e}")
        return

    print("Transcribing...")
    try:
        stream = engine.create_stream()
        stream.accept_waveform(16000, audio)
        engine.decode_stream(stream)
        print(f"Success! Result: {stream.result.text}")
    except Exception as e:
        print(f"Failed to transcribe: {e}")

if __name__ == "__main__":
    main()
