"""Instagram export parsing: Reel ids, mojibake repair and hashtags."""
import hashlib

from backend import config, metadata

def test_fix_text_reverses_instagram_mojibake():
    assert metadata.fix_text("ð\u009f\u0092» ARM vs x86") == "💻 ARM vs x86"
    assert metadata.fix_text("already fine — ✓") == "already fine — ✓"

def test_reel_id_from_urls():
    assert metadata.reel_id_from_url("https://www.instagram.com/reel/DtestReel01/") == "DtestReel01"
    assert metadata.reel_id_from_url("https://www.instagram.com/p/Abc_-123/") == "Abc_-123"
    assert metadata.reel_id_from_url("http://localhost:8765/test-media/page/TESTCLIP01") == "TESTCLIP01"
    assert metadata.reel_id_from_url("https://example.com/x") is None

def test_parse_fixture_preserves_raw_and_decodes():
    posts = metadata.load_posts(config.TEST_POSTS_PATH)
    p = posts[0]
    assert p["reel_id"] == "TESTCLIP01" and p["media_type"] == "reel"
    assert "🔍" in p["metadata"]["caption"]
    assert p["metadata"]["hashtags"] == ["ai", "databases"]
    assert p["metadata"]["owner"] == {"name": "Test Creator", "username": "test_creator",
                                      "url": "https://example.com/tester"}
    assert p["metadata"]["raw"]["label_values"][0]["label"] == "URL"
    assert p["saved_at"].startswith("2026-")

def test_real_export_parses_and_is_not_modified():
    path = config.INSTAGRAM_POSTS_PATH
    if not path.is_file():
        import pytest
        pytest.skip("Instagram export not present")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    posts = metadata.load_posts(path)
    assert len(posts) > 0 and all(p["source_url"] for p in posts)
    assert sum(p["reel_id"] is not None for p in posts) == len(posts)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before