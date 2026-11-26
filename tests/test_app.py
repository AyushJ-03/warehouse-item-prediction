from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_app_renders_sales_upload_page():
    app = AppTest.from_file(str(APP_PATH)).run()

    assert not app.exception
    assert any(
        "AI Business Operations & Profit Optimization" in item.value
        for item in app.title
    )
    assert len(app.file_uploader) == 1
