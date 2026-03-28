import os
import sys
import json
import numpy as np
import librosa

CAPS_WRITER_DIR = r"C:\Users\shihu\Documents\software\CapsWriter-Offline-20260304\CapsWriter-Offline"
INTERNAL_DIR = os.path.join(CAPS_WRITER_DIR, "internal")
for path in [CAPS_WRITER_DIR, INTERNAL_DIR]:
    if path not in sys.path:
        sys.path.append(path)

from util.qwen_asr_gguf import create_asr_engine

def main():
    audio_path = "test_dialogue.mp3"
    model_dir = os.path.join(CAPS_WRITER_DIR, "models", "Qwen3-ASR", "Qwen3-ASR-1.7B")
    
    print("Loading 10s audio...")
    audio, _ = librosa.load(audio_path, sr=16000, duration=10.0)
    
    print("Initializing Qwen3-ASR (n_threads=1)...")
    engine = create_asr_engine(
        model_dir=model_dir,
        encoder_frontend_fn="qwen3_asr_encoder_frontend.fp16.onnx",
        encoder_backend_fn="qwen3_asr_encoder_backend.fp16.onnx",
        llm_fn="qwen3_asr_llm.q4_k.gguf",
        use_dml=False,
        vulkan_enable=False,
        verbose=False,
        enable_aligner=False,
        n_threads=1
    )
    
    print("Transcribing 10s chunk...")
    try:
        result = engine.engine.asr(audio, context=None, language="Chinese")
        print(f"Success! Result: {result.text}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    main()
