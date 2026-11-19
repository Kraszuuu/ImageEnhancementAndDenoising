from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "src" / "lowlight" / "demo" / "app.py"


def test_demo_renders_without_exceptions() -> None:
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.run()

    assert not app.exception
    assert app.title[0].value == "Low-light enhancement demo"
    assert app.file_uploader[0].label == "Upload a low-light image"
    assert app.sidebar.selectbox[0].label == "Family"


def test_demo_rejects_empty_upload() -> None:
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.run()
    app.file_uploader[0].upload("empty.png", b"", "image/png").run()

    assert not app.exception
    assert any("empty" in item.value for item in app.error)
