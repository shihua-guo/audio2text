import os
import sys
import json
import numpy as np

CAPS_WRITER_DIR = r"C:\Users\shihu\Documents\software\CapsWriter-Offline-20260304\CapsWriter-Offline"
INTERNAL_DIR = os.path.join(CAPS_WRITER_DIR, "internal")
os.environ["GGML_VULKAN_DISABLE"] = "1"
os.environ["GGML_OPENCL_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

for path in [CAPS_WRITER_DIR, INTERNAL_DIR]:
    if path not in sys.path:
        sys.path.append(path)

from util.qwen_asr_gguf import create_asr_engine
import librosa

def main():
    if len(sys.argv) < 3: return
    audio_path = sys.argv[1]
    model_dir = sys.argv[2]
    
    # Initialize Engine
    engine = create_asr_engine(
        model_dir=model_dir,
        encoder_frontend_fn="qwen3_asr_encoder_frontend.fp16.onnx",
        encoder_backend_fn="qwen3_asr_encoder_backend.fp16.onnx",
        llm_fn="qwen3_asr_llm.q4_k.gguf",
        use_dml=False,
        vulkan_enable=False,
        verbose=False,
        enable_aligner=False,
        chunk_size=10.0,
        n_threads=1 # Single thread for absolute stability in subprocess
    )
    
    audio, _ = librosa.load(audio_path, sr=16000)
    result = engine.engine.asr(audio, context=None, language="Chinese")
    print(json.dumps({"text": result.text}))

if __name__ == "__main__":
    main()
