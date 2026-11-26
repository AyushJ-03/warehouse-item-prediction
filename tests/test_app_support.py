from io import BytesIO
import zipfile

import pandas as pd
import pytest

from src.app_support import build_output_archive, normalize_uploaded_columns


def test_normalize_uploaded_columns_maps_company_headers():
    raw = pd.DataFrame(
        {
            "Sales Date": ["2025-01-01"],
            "Item": ["Apples"],
            "Qty": [2],
            "Price per Unit": [10.0],
            "Total Amount": [20.0],
            "Store": ["North"],
        }
    )
    mapping = {
        "transaction_date": "Sales Date",
        "product_name": "Item",
        "quantity": "Qty",
        "unit_price": "Price per Unit",
        "final_amount": "Total Amount",
    }

    normalized = normalize_uploaded_columns(raw, mapping)

    assert set(normalized.columns) == {
        "transaction_date",
        "product_name",
        "quantity",
        "unit_price",
        "final_amount",
    }
    assert normalized.loc[0, "product_name"] == "Apples"


def test_normalize_uploaded_columns_rejects_reused_source_column():
    raw = pd.DataFrame({"date": ["2025-01-01"], "value": [1]})
    mapping = {
        "transaction_date": "date",
        "product_name": "value",
        "quantity": "value",
        "unit_price": "date",
        "final_amount": "value",
    }

    with pytest.raises(ValueError, match="different CSV column"):
        normalize_uploaded_columns(raw, mapping)


def test_build_output_archive_contains_generated_files(tmp_path):
    results = tmp_path / "results.csv"
    chart = tmp_path / "visualizations" / "overview.png"
    chart.parent.mkdir()
    results.write_text("product,value\nApples,1\n", encoding="utf-8")
    chart.write_bytes(b"png")

    archive = build_output_archive(tmp_path)

    with zipfile.ZipFile(BytesIO(archive)) as zip_file:
        assert set(zip_file.namelist()) == {"results.csv", "visualizations/overview.png"}
        assert zip_file.read("results.csv").startswith(b"product,value")
