from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd


INPUT_COLUMNS = {
    "transaction_date",
    "product_name",
    "quantity",
    "unit_price",
    "final_amount",
}


def normalize_uploaded_columns(
    raw_data: pd.DataFrame,
    column_mapping: dict[str, str],
) -> pd.DataFrame:
    missing_fields = INPUT_COLUMNS.difference(column_mapping)
    if missing_fields:
        raise ValueError(
            f"Choose a CSV column for: {', '.join(sorted(missing_fields))}."
        )

    selected_columns = [column_mapping[field] for field in sorted(INPUT_COLUMNS)]
    if len(set(selected_columns)) != len(selected_columns):
        raise ValueError("Each required field must map to a different CSV column.")

    missing_columns = set(selected_columns).difference(raw_data.columns)
    if missing_columns:
        raise ValueError(
            f"CSV columns were not found: {', '.join(sorted(missing_columns))}."
        )

    normalized = raw_data.loc[:, selected_columns].copy()
    return normalized.rename(
        columns={source: target for target, source in column_mapping.items()}
    )


def build_output_archive(output_dir: str | Path) -> bytes:
    output_dir = Path(output_dir)
    files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    if not files:
        raise ValueError("There are no generated outputs to download.")

    archive = BytesIO()
    with ZipFile(archive, mode="w", compression=ZIP_DEFLATED) as zip_file:
        for path in files:
            zip_file.write(path, arcname=path.relative_to(output_dir))
    return archive.getvalue()
