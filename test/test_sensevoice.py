import os
import sys
import numpy as np

CAPS_WRITER_DIR = r"C:\Users\shihu\Documents\software\CapsWriter-Offline-20260304\CapsWriter-Offline"
INTERNAL_DIR = os.path.join(CAPS_WRITER_DIR, "internal")
for path in [CAPS_WRITER_DIR, INTERNAL_DIR]:
    if path not in sys.path:
        sys.path.append(path)

from util.server.server_init_recognizer import create_fun_asr_engine

print("Initializing SenseVoice-Small engine...")
try:
    # Based on config_server.py defaults for SenseVoice-Small
    model_dir = os.path.join(CAPS_WRITER_DIR, "models", "SenseVoice-Small")
    engine = create_fun_asr_engine(
        model_dir=model_dir,
        use_dml=False,
        use_cuda=False
    )
    print("SenseVoice engine initialized successfully!")
    
    # Test transcription
    dummy_wav = np.zeros(16000 * 5).astype(np.float32)
    print("Testing transcription...")
    # SenseVoice-Small asr method might have different signature
    # In util/fun_asr_gguf/asr_engine.py
    result = engine.asr(dummy_wav)
    print(f"Result: {result.text}")
except Exception as e:
    print(f"SenseVoice failed: {e}")
    import traceback
    traceback.print_exc()
