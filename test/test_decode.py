import os
import sys
import numpy as np

CAPS_WRITER_DIR = r"C:\Users\shihu\Documents\software\CapsWriter-Offline-20260304\CapsWriter-Offline"
INTERNAL_DIR = os.path.join(CAPS_WRITER_DIR, "internal")
for path in [CAPS_WRITER_DIR, INTERNAL_DIR]:
    if path not in sys.path:
        sys.path.append(path)

from util.qwen_asr_gguf.inference import llama
from pathlib import Path

model_path = r"C:\Users\shihu\Documents\software\CapsWriter-Offline-20260304\CapsWriter-Offline\models\Qwen3-ASR\Qwen3-ASR-1.7B\qwen3_asr_llm.q4_k.gguf"

print("Initializing LlamaModel...")
model = llama.LlamaModel(model_path)
print("Initializing LlamaContext...")
ctx = llama.LlamaContext(model, n_ctx=2048)

print("Testing llama_decode with a dummy batch...")
# Create a simple batch with one token
batch = llama.LlamaBatch(1)
batch.token[0] = model.token_to_id("<|im_start|>")
batch.pos[0] = 0
batch.n_seq_id[0] = 1
batch.seq_id[0][0] = 0
batch.logits[0] = 1
batch.n_tokens = 1

try:
    res = ctx.decode(batch)
    print(f"Decode result: {res} (0 means success)")
except Exception as e:
    print(f"Decode failed: {e}")
    import traceback
    traceback.print_exc()
