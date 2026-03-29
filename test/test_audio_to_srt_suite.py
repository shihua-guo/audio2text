import os
import sys
import pytest
import numpy as np
import librosa
from pathlib import Path

# Add the test directory to path so we can import audio_to_srt directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audio_to_srt

class TestUtils:
    def test_format_timestamp(self):
        assert audio_to_srt.format_timestamp(0.0) == "00:00:00,000"
        assert audio_to_srt.format_timestamp(1.5) == "00:00:01,500"
        assert audio_to_srt.format_timestamp(61.123) == "00:01:01,123"
        assert audio_to_srt.format_timestamp(3661.1234) == "01:01:01,123"
        assert audio_to_srt.format_timestamp(3600.999) == "01:00:00,999"

    def test_group_sense_tokens_empty(self):
        assert audio_to_srt.group_sense_tokens([], [], 0.0) == []

    def test_group_sense_tokens_max_chars(self):
        # Tokens that result in a long string > 35 chars
        tokens = ["我"] * 40
        timestamps = [0.1 * i for i in range(40)]
        subs = audio_to_srt.group_sense_tokens(tokens, timestamps, 0.0, max_chars=35)
        # Should be broken into at least two parts
        assert len(subs) >= 2
        assert len(subs[0]['text']) <= 35

    def test_group_sense_tokens_punctuation(self):
        tokens = ["你好", "，", "世界", "。"]
        timestamps = [0.0, 0.5, 1.0, 1.5]
        subs = audio_to_srt.group_sense_tokens(tokens, timestamps, 0.0)
        
        # Depending on the logic, '，' or '。' triggers a break for the NEXT token
        # Our logic does: if i > 0 and any(p in tokens[i-1] for p in punctuation)
        # Meaning: "你好" (0) -> current="你好"
        # "，" (1) -> current="你好，"
        # "世界" (2) -> tokens[1] is "，", so wait, it should break BEFORE adding "世界"
        # So sub 1: "你好，", sub 2: "世界。"
        assert len(subs) >= 2
        assert "你好，" in subs[0]['text']
        assert "世界。" in subs[1]['text']

    def test_group_sense_tokens_gap(self):
        tokens = ["hello", "world"]
        timestamps = [0.0, 2.0] # 2.0 second gap
        # max_gap is 0.5 default
        subs = audio_to_srt.group_sense_tokens(tokens, timestamps, 0.0)
        assert len(subs) == 2
        assert subs[0]['text'] == "hello"
        assert subs[1]['text'] == "world"


class TestPipeline:
    @pytest.fixture(scope="class")
    def short_audio_path(self, tmp_path_factory):
        # Create a tiny audio file (e.g. 1 second of silence) just for testing IO paths if needed.
        # But actually we have `test_dialogue.mp3` in the current directory maybe?
        test_dir = os.path.dirname(os.path.abspath(__file__))
        dialogue_mp3 = os.path.join(test_dir, "test_dialogue.mp3")
        
        # If it doesn't exist, create a dummy wav file
        if not os.path.exists(dialogue_mp3):
            dummy_path = tmp_path_factory.mktemp("data") / "dummy.wav"
            import soundfile as sf
            sf.write(str(dummy_path), np.zeros(16000), 16000) # 1 sec silent
            return str(dummy_path)
            
        return dialogue_mp3

    def test_transcribe_sensevoice_short(self, short_audio_path):
        # This integration test runs actual ASR
        # It relies on models being available locally
        subtitles = audio_to_srt.transcribe_sensevoice(short_audio_path)
        assert isinstance(subtitles, list)
        if len(subtitles) > 0:
            assert 'start' in subtitles[0]
            assert 'end' in subtitles[0]
            assert 'text' in subtitles[0]


class TestCLI:
    def test_main_invalid_file(self, monkeypatch, capsys):
        import sys
        
        def mock_transcribe(*args, **kwargs):
            raise Exception("Simulated crash")
            
        monkeypatch.setattr(audio_to_srt, "transcribe_sensevoice", mock_transcribe)
        
        # Mock sys.argv
        monkeypatch.setattr(sys, 'argv', ['audio_to_srt.py', 'non_existent_file.mp3'])
        
        # Should not crash the whole process with uncaught exception, main() catches it
        audio_to_srt.main()
        
        captured = capsys.readouterr()
        assert "Error: Simulated crash" in captured.out
