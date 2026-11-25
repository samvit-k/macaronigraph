"""Generate self-made test media with known ground truth, then verify it.

Usage: .venv/bin/python scripts/make_test_media.py [--verify-only]
"""
from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend import config  # noqa: E402

FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FFPROBE = str(Path(config.FFMPEG).with_name("ffprobe"))
BASE = f"http://localhost:{config.PORT}/test-media/page"

SPEECH = {
    "TESTCLIP01": {
        "script": "Today we are learning how vector databases work. An embedding turns text into a list "
                  "of numbers, and similar ideas end up close together, so search can find meaning instead of keywords.",
        "cards": ["VECTOR DATABASES", "EMBEDDINGS ARE NUMBERS", "SEARCH BY MEANING"],
        "caption": "How vector search works 🔍 #ai #databases",
    },
    "TESTCLIP02": {
        "script": "Here is a quick tip for better sleep. Keep your bedroom cool and dark, stop using screens "
                  "an hour before bed, and wake up at the same time every single day.",
        "cards": ["BETTER SLEEP", "COOL AND DARK ROOM", "SAME WAKE TIME"],
        "caption": "3 sleep habits that changed my life 😴 #health",
    },
    "TESTCLIP03": {
        "script": "Let me explain compound interest. When you invest money, you earn returns on your returns, "
                  "so starting early matters far more than the amount you begin with.",
        "cards": ["COMPOUND INTEREST", "RETURNS ON RETURNS", "START EARLY"],
        "flash": "FLASH CARD",
        "caption": "Why starting early wins 💸 #finance #investing",
    },
}
SUBS = {
    "TESTSUBS": {
        "script": "Caching stores the result of an expensive computation so the next request can reuse it. "
                  "Always set an expiry time or stale data will quietly break your app.",
        "title": "CACHING IN 20 SECONDS",
        "chunk_words": 3,
        "caption": "Caching explained ⚡ #backend",
    },
}
NON_SPEECH = {
    "TESTSILENT": {"cards": ["NO AUDIO HERE", "SILENT CLIP"], "duration": 8.0, "audio": "silent",
                   "caption": "silent test clip"},
    "TESTMUSIC": {"cards": ["MUSIC ONLY", "NO WORDS SPOKEN"], "duration": 10.0, "audio": "music",
                  "caption": "music test clip 🎵"},
}

def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, **kw)

def duration(path: Path, stream: str | None = None) -> float:
    sel = ["-select_streams", stream, "-show_entries", "stream=duration"] if stream \
        else ["-show_entries", "format=duration"]
    out = run([FFPROBE, "-v", "error", *sel, "-of", "csv=p=0", str(path)], text=True)
    return float(out.stdout.strip().splitlines()[0])

def max_volume_db(path: Path, start: float, end: float) -> float:
    err = subprocess.run([config.FFMPEG, "-v", "info", "-ss", str(start), "-to", str(end), "-i", str(path),
                          "-af", "volumedetect", "-vn", "-f", "null", "-"], capture_output=True, text=True).stderr
    m = re.search(r"max_volume:\s*(-?[\d.]+|-inf) dB", err)
    return float(m.group(1)) if m and m.group(1) != "-inf" else -200.0

def card_schedule(cards: list[str], total: float) -> list[dict]:
    span = total / len(cards)
    return [{"text": t, "start": round(i * span + 0.2, 2), "end": round((i + 1) * span - 0.2, 2)}
            for i, t in enumerate(cards)]

def drawtext(card: dict, y: str = "(h-text_h)/2", size: int = 64) -> str:
    y, size = card.get("y", y), card.get("size", size)
    size = min(size, int(1000 / max(len(card["text"]), 1)))
    return (f"drawtext=fontfile='{FONT}':text='{card['text']}':fontcolor=white:fontsize={size}:"
            f"box=1:boxcolor=black@0.6:boxborderw=24:x=(w-text_w)/2:y={y}:"
            f"enable='between(t,{card['start']},{card['end']})'")

def render(out: Path, audio_args: list[str], total: float, cards: list[dict], extra: list[dict] = ()):
    filters = [drawtext(c) for c in cards] + [drawtext(c, y="h*0.2", size=56) for c in extra]
    run([config.FFMPEG, "-y", "-v", "error",
         "-f", "lavfi", "-i", f"color=c=0x1f3a5f:s=720x1280:r=30:d={total}",
         *audio_args,
         "-vf", ",".join(filters),
         "-af", "apad",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
         "-c:a", "aac", "-b:a", "128k", "-t", str(total), str(out)])

def mojibake(s: str) -> str:
    return s.encode("utf-8").decode("latin-1")

def post(reel_id: str, caption: str, ts: int) -> dict:
    return {"timestamp": ts, "media": [], "label_values": [
        {"label": "URL", "value": f"{BASE}/{reel_id}", "href": f"{BASE}/{reel_id}"},
        {"label": "Caption", "value": mojibake(caption)},
        {"label": "Title", "value": ""},
        {"dict": [{"dict": [{"label": "Name", "value": h.lstrip("#")}], "title": ""}
                  for h in caption.split() if h.startswith("#")], "title": "Hashtags"},
        {"dict": [{"dict": [{"label": "URL", "value": "https://example.com/tester"},
                            {"label": "Name", "value": "Test Creator"},
                            {"label": "Username", "value": "test_creator"}], "title": ""}], "title": "Owner"},
        {"dict": [], "title": "Brand partner"},
    ]}

def generate() -> None:
    media = config.TEST_MEDIA_DIR
    media.mkdir(parents=True, exist_ok=True)
    truth, posts, ts = {}, [], 1795000000

    for rid, spec in SPEECH.items():
        with tempfile.TemporaryDirectory() as d:
            aiff = Path(d) / "speech.aiff"
            run(["say", "-r", "170", "-o", str(aiff), spec["script"]])
            speech_end = round(duration(aiff), 2)
            total = round(speech_end + 1.0, 2)
            cards = card_schedule(spec["cards"], total)
            flash = [{"text": spec["flash"], "start": round(total / 2, 2), "end": round(total / 2 + 0.6, 2)}] \
                if "flash" in spec else []
            render(media / f"{rid}.mp4", ["-i", str(aiff)], total, cards, flash)
        truth[rid] = {"audio_class": "speech", "script": spec["script"], "cards": cards + flash,
                      "duration": total, "speech_end": speech_end}
        posts.append(post(rid, spec["caption"], ts)); ts -= 60

    for rid, spec in SUBS.items():
        with tempfile.TemporaryDirectory() as d:
            aiff = Path(d) / "speech.aiff"
            run(["say", "-r", "170", "-o", str(aiff), spec["script"]])
            speech_end = round(duration(aiff), 2)
            total = round(speech_end + 1.0, 2)
            words = [w.strip(".,").upper() for w in spec["script"].split()]
            n = spec["chunk_words"]
            chunks = [" ".join(words[i:i + n]) for i in range(0, len(words), n)]
            per_word = speech_end / len(words)
            subs, t = [], 0.0
            for c in chunks:
                dur = per_word * len(c.split())
                subs.append({"text": c, "start": round(t, 2), "end": round(t + dur - 0.02, 2),
                             "y": "h*0.72", "size": 56})
                t += dur
            title = {"text": spec["title"], "start": 0.0, "end": total, "y": "h*0.08", "size": 52}
            render(media / f"{rid}.mp4", ["-i", str(aiff)], total, [title, *subs])
        truth[rid] = {"audio_class": "speech", "script": spec["script"], "duration": total, "speech_end": speech_end,
                      "cards": [{k: title[k] for k in ("text", "start", "end")}],
                      "subtitles": [{k: s[k] for k in ("text", "start", "end")} for s in subs]}
        posts.append(post(rid, spec["caption"], ts)); ts -= 60

    for rid, spec in NON_SPEECH.items():
        total = spec["duration"]
        if spec["audio"] == "silent":
            audio = ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo"]
        else:
            audio = ["-f", "lavfi", "-i",
                     f"aevalsrc='0.25*sin(2*PI*t*(262+131*mod(floor(t*4),4)))':s=44100:d={total}"]
        cards = card_schedule(spec["cards"], total)
        render(media / f"{rid}.mp4", audio, total, cards)
        truth[rid] = {"audio_class": "silent" if spec["audio"] == "silent" else "non_speech",
                      "script": "", "cards": cards, "duration": total}
        posts.append(post(rid, spec["caption"], ts)); ts -= 60

    truth["TESTMISSING"] = {"audio_class": None, "expect_status": "failed", "script": "", "cards": []}
    posts.append(post("TESTMISSING", "this video was deleted", ts))

    config.TEST_GROUND_TRUTH.write_text(json.dumps(truth, indent=2))
    config.TEST_POSTS_PATH.write_text(json.dumps(posts, indent=1))

def ocr_at(path: Path, t: float) -> str:
    from PIL import Image
    import pytesseract

    png = run([config.FFMPEG, "-v", "error", "-ss", str(t), "-i", str(path), "-frames:v", "1",
               "-f", "image2pipe", "-vcodec", "png", "-"]).stdout
    return pytesseract.image_to_string(Image.open(io.BytesIO(png)).convert("L")).upper()

def verify() -> bool:
    config.configure_tesseract()
    truth = json.loads(config.TEST_GROUND_TRUTH.read_text())
    ok = True

    def report(cond: bool, msg: str):
        nonlocal ok
        ok &= cond
        print(f"   {'ok  ' if cond else 'FAIL'} {msg}")

    for rid, t in truth.items():
        if "duration" not in t:
            continue
        path = config.TEST_MEDIA_DIR / f"{rid}.mp4"
        print(f"{rid}")
        fmt, v, a = duration(path), duration(path, "v:0"), duration(path, "a:0")
        report(abs(v - t["duration"]) < 0.1 and abs(a - t["duration"]) < 0.15,
               f"duration expected {t['duration']}s; container {fmt:.2f} video {v:.2f} audio {a:.2f}")
        vol = max_volume_db(path, 0, t["duration"])
        if t["audio_class"] == "speech":
            report(vol > -20, f"speech audible (max {vol:.1f} dB)")
            tail = max_volume_db(path, t["speech_end"] + 0.15, t["duration"])
            report(tail < -60, f"padded tail after speech is silent (max {tail:.1f} dB)")
        elif t["audio_class"] == "silent":
            report(vol < -80, f"silent track (max {vol:.1f} dB)")
        else:
            report(vol > -20, f"music audible (max {vol:.1f} dB)")
        for c in t["cards"]:
            mid = (c["start"] + c["end"]) / 2
            report(c["text"] in ocr_at(path, mid), f"card '{c['text']}' visible at {mid:.2f}s")
            if c["start"] > 0:
                report(c["text"] not in ocr_at(path, max(0.0, c["start"] - 0.15)),
                       f"card '{c['text']}' absent just before {c['start']}s")
        for s in t.get("subtitles", [])[::3]:
            mid = (s["start"] + s["end"]) / 2
            report(s["text"] in " ".join(ocr_at(path, mid).split()), f"subtitle '{s['text']}' visible at {mid:.2f}s")
    return ok

if __name__ == "__main__":
    if "--verify-only" not in sys.argv:
        generate()
    sys.exit(0 if verify() else 1)