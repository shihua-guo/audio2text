import os
import sys
import sherpa_onnx
import librosa

CAPS_WRITER_DIR = r"C:\Users\shihu\Documents\software\CapsWriter-Offline-20260304\CapsWriter-Offline"
INTERNAL_DIR = os.path.join(CAPS_WRITER_DIR, "internal")
for path in [CAPS_WRITER_DIR, INTERNAL_DIR]:
    if path not in sys.path:
        sys.path.append(path)

def main():
    with open("check_result.log", "w", encoding="utf-8") as log_f:
        def log_print(*args):
            msg = " ".join(map(str, args))
            print(msg)
            log_f.write(msg + "\n")

        sense_dir = os.path.join(CAPS_WRITER_DIR, "models", "SenseVoice-Small", "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17")
        model_path = os.path.join(sense_dir, "model.onnx")
        tokens_path = os.path.join(sense_dir, "tokens.txt")
        
        log_print("Initializing SenseVoice Engine...")
        recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=model_path, tokens=tokens_path, use_itn=True, num_threads=4
        )
        
        # Load from 30s to 35s
        log_print("Loading audio (30s-35s)...")
        audio, _ = librosa.load("test_dialogue.mp3", sr=16000, offset=30.0, duration=5.0)
        
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, audio)
        recognizer.decode_stream(stream)
        result = stream.result
        
        log_print(f"Result text: {result.text}")
        log_print(f"Tokens count: {len(result.tokens)}")
        if len(result.tokens) > 0:
            log_print(f"First 5 tokens: {result.tokens[:5]}")
            log_print(f"First 5 timestamps: {result.timestamps[:5]}")

if __name__ == "__main__":
    main()
