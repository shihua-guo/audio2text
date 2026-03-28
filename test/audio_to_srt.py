import os
import sys
import argparse
from pathlib import Path
import librosa
import numpy as np

# --- Configuration ---
CAPS_WRITER_DIR = r"C:\Users\shihu\Documents\software\CapsWriter-Offline-20260304\CapsWriter-Offline"
INTERNAL_DIR = os.path.join(CAPS_WRITER_DIR, "internal")
for path in [CAPS_WRITER_DIR, INTERNAL_DIR]:
    if path not in sys.path:
        sys.path.append(path)

def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def group_sense_tokens(tokens, timestamps, offset, max_chars=35, max_gap=0.5, min_duration=0.8):
    subtitles = []
    if not tokens: return subtitles
    
    current_text = ""
    current_start = timestamps[0] + offset
    punctuation = set("。！？，；：.!? ,;:")
    
    for i in range(len(tokens)):
        text, start = tokens[i], timestamps[i] + offset
        end = (timestamps[i+1] + offset) if i+1 < len(timestamps) else start + 0.2
        
        # Clean special tokens
        if text.startswith('<|') and text.endswith('|>'): continue
        if text.strip() == "": continue

        should_break = False
        if i > 0 and (start - (timestamps[i-1] + offset)) > max_gap: should_break = True
        if len(current_text + text) > max_chars: should_break = True
        if i > 0 and any(p in tokens[i-1] for p in punctuation): should_break = True
        
        if should_break and current_text.strip():
            actual_end = max(start, current_start + min_duration)
            subtitles.append({'start': current_start, 'end': actual_end, 'text': current_text.strip()})
            current_text, current_start = text, start
        else:
            current_text += text
            
    if current_text.strip():
        actual_end = max(timestamps[-1] + offset + 0.2, current_start + min_duration)
        subtitles.append({'start': current_start, 'end': actual_end, 'text': current_text.strip()})
    return subtitles

def transcribe_sensevoice(audio_path):
    import sherpa_onnx
    sense_dir = os.path.join(CAPS_WRITER_DIR, "models", "SenseVoice-Small", "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17")
    model_path = os.path.join(sense_dir, "model.onnx")
    tokens_path = os.path.join(sense_dir, "tokens.txt")
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        sys.exit(1)

    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=model_path, tokens=tokens_path, use_itn=True, num_threads=4
    )
    
    print("Loading audio...")
    audio, _ = librosa.load(audio_path, sr=16000)
    total_duration = len(audio) / 16000
    
    chunk_size_sec = 30
    chunk_samples = chunk_size_sec * 16000
    all_subtitles = []
    
    print(f"Transcribing {total_duration:.2f}s in {chunk_size_sec}s chunks...")
    for start_sample in range(0, len(audio), chunk_samples):
        end_sample = min(start_sample + chunk_samples, len(audio))
        chunk = audio[start_sample:end_sample]
        offset = start_sample / 16000
        
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, chunk)
        recognizer.decode_stream(stream)
        result = stream.result
        
        if result.text.strip():
            print(f"[{format_timestamp(offset)}] {result.text[:50]}...")
            if hasattr(result, 'tokens') and result.tokens:
                subs = group_sense_tokens(result.tokens, result.timestamps, offset)
                all_subtitles.extend(subs)
            else:
                all_subtitles.append({'start': offset, 'end': offset + len(chunk)/16000, 'text': result.text.strip()})
    
    return all_subtitles

def main():
    parser = argparse.ArgumentParser(description="Convert Audio to SRT (SenseVoice Stable Edition)")
    parser.add_argument("input", help="Path to input audio file")
    parser.add_argument("--output", help="Path to output SRT file")
    args = parser.parse_args()
    
    input_path = args.input
    output_path = args.output or str(Path(input_path).with_suffix(".srt"))
    
    try:
        subtitles = transcribe_sensevoice(input_path)
        print(f"Writing {len(subtitles)} segments to {output_path}...")
        with open(output_path, "w", encoding="utf-8") as f:
            for idx, sub in enumerate(subtitles, 1):
                f.write(f"{idx}\n{format_timestamp(sub['start'])} --> {format_timestamp(sub['end'])}\n{sub['text']}\n\n")
        print("Success! SRT generation complete.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
