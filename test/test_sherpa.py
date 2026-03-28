import os
import sys
import numpy as np

# Add CapsWriter-Offline and its internal libraries to sys.path 
CAPS_WRITER_DIR = r"C:\Users\shihu\Documents\software\CapsWriter-Offline-20260304\CapsWriter-Offline"
INTERNAL_DIR = os.path.join(CAPS_WRITER_DIR, "internal")
for path in [CAPS_WRITER_DIR, INTERNAL_DIR]:
    if path not in sys.path:
        sys.path.append(path)

try:
    import sherpa_onnx
    import soundfile as sf
    print("Successfully imported sherpa_onnx and soundfile.")
except ImportError as e:
    print(f"Error: Could not import dependencies. {e}")
    sys.exit(1)

model_dir = os.path.join(CAPS_WRITER_DIR, "models", "SenseVoice-Small", "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17")
model = os.path.join(model_dir, "model.onnx")
tokens = os.path.join(model_dir, "tokens.txt")

def transcribe_sensevoice(audio_path, output_srt):
    print(f"Transcribing (SenseVoice): {audio_path}")
    
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=model,
        tokens=tokens,
        use_itn=True,
        debug=False,
        num_threads=4,
    )
    
    import librosa
    audio, _ = librosa.load(audio_path, sr=16000)
    
    stream = recognizer.create_stream()
    stream.accept_waveform(16000, audio)
    recognizer.decode_stream(stream)
    result = stream.result
    
    # SenseVoice results often have 'timestamps' and 'tokens'
    # We'll need to group tokens into subtitles
    # For now, let's just print the text to see if it works
    print(f"Result Text: {result.text[:100]}...")
    
    if hasattr(result, 'timestamps'):
        print(f"Found {len(result.timestamps)} timestamps.")

if __name__ == "__main__":
    test_audio = r"c:\Users\shihu\Documents\workspace\audio2text\test_dialogue.mp3"
    transcribe_sensevoice(test_audio, "test.srt")
