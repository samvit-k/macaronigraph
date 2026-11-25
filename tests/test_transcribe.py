"""Speech, silence and music, multilingual and short utterances (real MLX-Whisper)."""
import subprocess
from pathlib import Path

import numpy as np
import pytest

from backend import config, transcribe as T

def say(tmp: Path, text: str, voice: str | None = None) -> Path:
    out = tmp / "a.aiff"
    subprocess.run(["say", *(["-v", voice] if voice else []), "-o", str(out), text], check=True)
    return out

def test_norm_is_unicode_aware():
    assert T.norm("¡Hola, Señor!") == "hola señor"
    assert T.norm("こんにちは。世界") == "こんにちは 世界"
    assert T.norm("♪ ... ♪") == ""

def test_grade_segment_tiers():
    ok = {"text": "Yes.", "avg_logprob": -0.3, "no_speech_prob": 0.01, "compression_ratio": 0.5}
    assert T.grade_segment(ok) == ("confident", [])
    assert T.grade_segment({**ok, "text": "Thank you."})[0] == "uncertain"
    assert T.grade_segment({**ok, "avg_logprob": -1.3})[0] == "uncertain"
    assert T.grade_segment({**ok, "text": "♪"})[0] == "rejected"
    assert T.grade_segment({**ok, "compression_ratio": 3.1})[0] == "rejected"
    assert T.grade_segment({**ok, "no_speech_prob": 0.9, "avg_logprob": -1.5})[0] == "rejected"

def test_silent_audio_skips_whisper():
    r = T.transcribe_pcm(np.zeros(T.SR * 3, np.float32))
    assert r["audio_class"] == "silent" and r["status"] == "skipped" and r["text"] == ""

def test_missing_audio_is_error(tmp_path):
    bad = tmp_path / "x.webm"
    bad.write_bytes(b"not audio")
    r = T.transcribe_file(bad)
    assert r["audio_class"] == "missing" and r["status"] == "error"

@pytest.mark.parametrize("voice,text,lang,word", [
    ("Eddy (Spanish (Spain))", "Hola, hoy vamos a hablar sobre la inteligencia artificial.", "es", "inteligencia"),
    ("Eddy (French (France))", "Bonjour, aujourd'hui nous parlons de la photosynthèse.", "fr", "photosynth"),
    ("Eddy (Japanese (Japan))", "こんにちは、今日は人工知能について話します。", "ja", "人工知能"),
])
def test_multilingual_speech(tmp_path, voice, text, lang, word):
    r = T.transcribe_file(say(tmp_path, text, voice))
    assert r["audio_class"] == "speech" and r["language"] == lang
    assert word in r["text"].lower()

def test_single_word_utterance_is_kept(tmp_path):
    r = T.transcribe_file(say(tmp_path, "Yes."))
    assert r["audio_class"] == "speech" and "yes" in T.norm(r["text"])

def test_ambiguous_phrase_preserved_as_uncertain(tmp_path):
    r = T.transcribe_file(say(tmp_path, "Thank you."))
    assert "thank you" in T.norm(r["text"] + " " + r["uncertain_text"])

def test_music_only_is_non_speech():
    r = T.transcribe_file(config.TEST_MEDIA_DIR / "TESTMUSIC.mp4")
    assert r["audio_class"] == "non_speech" and r["text"] == ""
    assert all(s["confidence"] == "low" for s in r["segments"])

def test_silent_clip_is_silent():
    r = T.transcribe_file(config.TEST_MEDIA_DIR / "TESTSILENT.mp4")
    assert r["audio_class"] == "silent" and r["text"] == "" and r["uncertain_text"] == ""