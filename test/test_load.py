import os
import sys

CAPS_WRITER_DIR = r"C:\Users\shihu\Documents\software\CapsWriter-Offline-20260304\CapsWriter-Offline"
INTERNAL_DIR = os.path.join(CAPS_WRITER_DIR, "internal")
for path in [CAPS_WRITER_DIR, INTERNAL_DIR]:
    if path not in sys.path:
        sys.path.append(path)

from util.qwen_asr_gguf.inference import llama
from pathlib import Path

model_path = r"C:\Users\shihu\Documents\software\CapsWriter-Offline-20260304\CapsWriter-Offline\models\Qwen3-ASR\Qwen3-ASR-1.7B\qwen3_asr_llm.q4_k.gguf"

print(f"Testing model load from: {model_path}")
print(f"File exists: {os.path.exists(model_path)}")

try:
    model = llama.LlamaModel(model_path)
    print("Model loaded successfully!")
    print(f"Vocab size: {llama.llama_vocab_n_tokens(model.vocab)}")
except Exception as e:
    print(f"Failed to load model: {e}")
    import traceback
    traceback.print_exc()
