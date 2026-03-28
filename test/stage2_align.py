import os
import sys
import json
import numpy as np

CAPS_WRITER_DIR = r"C:\Users\shihu\Documents\software\CapsWriter-Offline-20260304\CapsWriter-Offline"
INTERNAL_DIR = os.path.join(CAPS_WRITER_DIR, "internal")
for path in [CAPS_WRITER_DIR, INTERNAL_DIR]:
    if path not in sys.path:
        sys.path.append(path)

from util.qwen_asr_gguf.inference.aligner import QwenForcedAligner
from util.qwen_asr_gguf.inference.schema import AlignerConfig
import librosa

# Patch for QwenAudioEncoder
from util.qwen_asr_gguf.inference.encoder import QwenAudioEncoder
original_init = QwenAudioEncoder.__init__
def patched_init(self, *args, **kwargs):
    kwargs.pop('warmup_sec', None)
    return original_init(self, *args, **kwargs)
QwenAudioEncoder.__init__ = patched_init

def main():
    if len(sys.argv) < 4: return
    audio_path = sys.argv[1]
    text = sys.argv[2]
    model_dir = sys.argv[3]
    
    align_config = AlignerConfig(
        model_dir=model_dir,
        encoder_frontend_fn="qwen3_asr_encoder_frontend.fp16.onnx",
        encoder_backend_fn="qwen3_asr_encoder_backend.fp16.onnx",
        llm_fn="qwen3_asr_llm.q4_k.gguf",
        use_dml=False,
        n_ctx=4096 # Larger context for alignment
    )
    
    audio, _ = librosa.load(audio_path, sr=16000)
    aligner = QwenForcedAligner(align_config)
    res = aligner.align(audio, text, language="Chinese")
    
    items = [{'start': i.start_time, 'end': i.end_time, 'text': i.text} for i in res.items]
    print(json.dumps(items))

if __name__ == "__main__":
    main()
