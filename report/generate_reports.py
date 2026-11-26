import html
import re
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
OUTPUT_DIR = ROOT / "data" / "processed"


def _styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=25,
            textColor=colors.HexColor("#15324b"),
            alignment=TA_CENTER,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportSubtitle",
            parent=styles["Normal"],
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#526477"),
            alignment=TA_CENTER,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportH1",
            parent=styles["Heading1"],
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#15324b"),
            spaceBefore=10,
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportH2",
            parent=styles["Heading2"],
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#176b87"),
            spaceBefore=8,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportH3",
            parent=styles["Heading3"],
            fontSize=10.5,
            leading=13,
            textColor=colors.HexColor("#176b87"),
            spaceBefore=6,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportBody",
            parent=styles["BodyText"],
            fontSize=9,
            leading=12.5,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportCell",
            parent=styles["BodyText"],
            fontSize=7.1,
            leading=8.5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportHeadCell",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7,
            leading=8,
            textColor=colors.white,
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportCode",
            parent=styles["Code"],
            fontName="Courier",
            fontSize=8,
            leading=10,
            leftIndent=8,
            rightIndent=8,
            backColor=colors.HexColor("#f1f5f8"),
            borderPadding=6,
            spaceAfter=6,
        )
    )
    return styles


STYLES = _styles()


def _inline(text: str) -> str:
    text = text.replace("<br/>", " / ")
    text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda match: (
            f'<link href="{html.escape(match.group(2), quote=True)}" '
            f'color="#176b87">{html.escape(match.group(1))}</link>'
        ),
        text,
    )
    text = html.escape(text, quote=False)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    text = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)
    return text


def _table(rows: list[list[str]]) -> Table:
    data = [
        [
            Paragraph(_inline(cell.strip()), STYLES["ReportHeadCell" if row_index == 0 else "ReportCell"])
            for cell in row
        ]
        for row_index, row in enumerate(rows)
    ]
    widths = [42 * mm, 22 * mm, 22 * mm, 22 * mm, 22 * mm, 16 * mm, 16 * mm, 16 * mm]
    if len(widths) != len(data[0]):
        widths = [174 * mm / len(data[0])] * len(data[0])
        widths[0] = 174 * mm - sum(widths[1:])
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#176b87")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f1f5f8")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.white),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _markdown_story(path: Path):
    story = []
    paragraph_lines = []
    in_code = False
    code_lines = []
    table_lines = []

    def flush_paragraph():
        if paragraph_lines:
            story.append(
                Paragraph(_inline(" ".join(paragraph_lines)), STYLES["ReportBody"])
            )
            paragraph_lines.clear()

    def flush_table():
        if not table_lines:
            return
        parsed_rows = []
        for line in table_lines:
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
                parsed_rows.append(cells)
        if parsed_rows:
            story.append(_table(parsed_rows))
            story.append(Spacer(1, 3 * mm))
        table_lines.clear()

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("```"):
            flush_paragraph()
            flush_table()
            if in_code:
                story.append(
                    Preformatted("\n".join(code_lines), STYLES["ReportCode"])
                )
                code_lines.clear()
            in_code = not in_code
            continue
        if in_code:
            code_lines.append(raw_line)
            continue
        if line.startswith("|"):
            flush_paragraph()
            table_lines.append(line)
            continue
        flush_table()
        if not line:
            flush_paragraph()
            continue
        if line.startswith("# "):
            flush_paragraph()
            story.append(Paragraph(_inline(line[2:]), STYLES["ReportTitle"]))
        elif line.startswith("## "):
            flush_paragraph()
            story.append(Paragraph(_inline(line[3:]), STYLES["ReportH1"]))
        elif line.startswith("### "):
            flush_paragraph()
            story.append(Paragraph(_inline(line[4:]), STYLES["ReportH2"]))
        elif line.startswith("- "):
            flush_paragraph()
            story.append(
                Paragraph("- " + _inline(line[2:]), STYLES["ReportBody"])
            )
        else:
            paragraph_lines.append(line)

    flush_paragraph()
    flush_table()
    return story


def _footer(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#d9e2e8"))
    canvas.line(18 * mm, 15 * mm, 192 * mm, 15 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawString(
        18 * mm, 10 * mm, "AI-Powered Business Operations and Profit Optimization System"
    )
    canvas.drawRightString(192 * mm, 10 * mm, f"Page {document.page}")
    canvas.restoreState()


def _comparison_table(comparison: pd.DataFrame) -> Table:
    headers = [
        "Model",
        "Validation<br/>MAE",
        "Validation<br/>RMSE",
        "Test<br/>MAE",
        "Test<br/>RMSE",
        "Val.<br/>wins",
        "Test<br/>wins",
        "Forecast<br/>products",
    ]
    rows = [headers]
    for _, row in comparison.iterrows():
        model = str(row["Model"]).replace(" (7-day)", "")
        rows.append(
            [
                model,
                f'{row["Mean_Validation_MAE"]:.4f}',
                f'{row["Mean_Validation_RMSE"]:.4f}',
                f'{row["Mean_Test_MAE"]:.4f}',
                f'{row["Mean_Test_RMSE"]:.4f}',
                str(int(row["Validation_Wins"])),
                str(int(row["Test_Wins"])),
                str(int(row["Forecast_Products"])),
            ]
        )
    return _table(rows)


def build_reports() -> tuple[Path, Path]:
    comparison = pd.read_csv(OUTPUT_DIR / "model_comparison.csv")
    metrics = pd.read_csv(OUTPUT_DIR / "forecast_metrics.csv")
    forecast = pd.read_csv(OUTPUT_DIR / "forecast.csv")
    production = pd.read_csv(OUTPUT_DIR / "production_plan.csv")
    profit = pd.read_csv(OUTPUT_DIR / "profit_results.csv")
    clean = pd.read_csv(OUTPUT_DIR / "clean_data.csv")

    story = _markdown_story(REPORT_DIR / "submission_report.md")
    submission_path = REPORT_DIR / "submission_report.pdf"
    SimpleDocTemplate(
        str(submission_path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=21 * mm,
        title="Submission Report - AI-Powered Business Operations and Profit Optimization System",
    ).build(story, onFirstPage=_footer, onLaterPages=_footer)

    prophet_test_mae = metrics.loc[
        metrics["Used_For_Forecast"], "Test_MAE"
    ].mean()
    mae_leader = comparison.loc[comparison["Mean_Test_MAE"].idxmin()]
    rmse_leader = comparison.loc[comparison["Mean_Test_RMSE"].idxmin()]
    test_mae_winner = comparison.loc[comparison["Test_Wins"].idxmax()]
    forecast_counts = forecast.groupby("Forecast_Model")["Product"].nunique()
    forecast_summary = ", ".join(
        f"{model} {int(count)}"
        for model, count in forecast_counts.items()
    )
    unmet_demand = production["Unmet_Demand"].sum()
    capacity_ok = (
        production["Optimal_Production"] <= production["Capacity"] + 1e-8
    ).all()
    summary_story = [
        Spacer(1, 7 * mm),
        Paragraph(
            "AI-Powered Business Operations<br/>&amp; Profit Optimization System",
            STYLES["ReportTitle"],
        ),
        Paragraph(
            "Project Summary | 16 November 2026", STYLES["ReportSubtitle"]
        ),
        Paragraph("Objective", STYLES["ReportH2"]),
        Paragraph(
            "An end-to-end analytics prototype connecting grocery transaction cleaning, "
            "demand forecasting, inventory planning, capacity-constrained production, "
            "and estimated product profit.",
            STYLES["ReportBody"],
        ),
        Paragraph("Verified execution", STYLES["ReportH2"]),
        Paragraph(
            f"The pipeline processed {len(clean):,} cleaned rows across "
            f"{forecast['Product'].nunique()} products and generated {len(forecast)} "
            f"future predictions ({forecast.groupby('Product').size().min()} days per product). "
            "Automated tests are included. The full method and comparison are in "
            "submission_report.pdf and submission_report.md.",
            STYLES["ReportBody"],
        ),
        Paragraph("Forecast model comparison", STYLES["ReportH2"]),
        _comparison_table(comparison),
        Spacer(1, 2 * mm),
        Paragraph(
            f"Prophet is used for every future product forecast; the three other "
            f"models are comparison benchmarks. Mean test MAE leader: {mae_leader['Model']} "
            f"({mae_leader['Mean_Test_MAE']:.4f}). Mean test RMSE leader: "
            f"{rmse_leader['Model']} ({rmse_leader['Mean_Test_RMSE']:.4f}). "
            f"Product-level test MAE wins: {test_mae_winner['Model']} "
            f"({int(test_mae_winner['Test_Wins'])}/{forecast['Product'].nunique()}). "
            f"Forecast model distribution: {forecast_summary}. "
            f"Prophet's mean test MAE was {prophet_test_mae:.4f}.",
            STYLES["ReportBody"],
        ),
        Paragraph("Operational output", STYLES["ReportH2"]),
        Paragraph(
            f"Production respected the modeled capacity ({capacity_ok}) and reported "
            f"{unmet_demand:.2f} units of unmet demand. Estimated total profit: "
            f"{profit['Profit'].sum():,.2f} in unspecified currency.",
            STYLES["ReportBody"],
        ),
        Paragraph("Important limitation", STYLES["ReportH2"]),
        Paragraph(
            "Costs, lead time, and capacity are illustrative assumptions or proxies "
            "because real operational data are absent. Replace them before interpreting "
            "results as business recommendations. Tableau must be refreshed manually "
            "from current CSV outputs.",
            STYLES["ReportBody"],
        ),
        Paragraph(
            "Run: <b>python -m src.run_pipeline</b> | "
            "Tests: <b>python -m pytest</b>",
            STYLES["ReportBody"],
        ),
    ]
    summary_path = REPORT_DIR / "project_summary.pdf"
    SimpleDocTemplate(
        str(summary_path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=21 * mm,
        title="Project Summary - AI-Powered Business Operations and Profit Optimization System",
    ).build(summary_story, onFirstPage=_footer, onLaterPages=_footer)

    return submission_path, summary_path


if __name__ == "__main__":
    for report_path in build_reports():
        print(f"Generated {report_path}")
