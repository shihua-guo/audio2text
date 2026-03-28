import os
import sys
import numpy as np

CAPS_WRITER_DIR = r"C:\Users\shihu\Documents\software\CapsWriter-Offline-20260304\CapsWriter-Offline"
INTERNAL_DIR = os.path.join(CAPS_WRITER_DIR, "internal")
for path in [CAPS_WRITER_DIR, INTERNAL_DIR]:
    if path not in sys.path:
        sys.path.append(path)

from util.qwen_asr_gguf.inference.encoder import QwenAudioEncoder

MODEL_DIR = os.path.join(CAPS_WRITER_DIR, "models", "Qwen3-ASR", "Qwen3-ASR-1.7B")
fe_path = os.path.join(MODEL_DIR, "qwen3_asr_encoder_frontend.fp16.onnx")
be_path = os.path.join(MODEL_DIR, "qwen3_asr_encoder_backend.fp16.onnx")

print("Initializing QwenAudioEncoder...")
try:
    encoder = QwenAudioEncoder(fe_path, be_path, use_dml=False)
    print("Encoder initialized successfully!")
    
    # Test encoding
    dummy_wav = np.zeros(16000 * 2).astype(np.float32)
    print("Testing encoding...")
    embd, t = encoder.encode(dummy_wav)
    print(f"Encoded successfully. Shape: {embd.shape}, Time: {t:.2f}s")
except Exception as e:
    print(f"Encoder failed: {e}")
    import traceback
    traceback.print_exc()
