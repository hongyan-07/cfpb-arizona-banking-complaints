"""Analyze CFPB consumer banking complaints for Arizona, 2024-2025.

This project uses the CFPB Consumer Complaint Database. The source data is
public, but it is not a statistical sample of all customer experiences. The
analysis therefore describes the published complaints and does not infer
company quality or population-level complaint rates.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_DIR / "data" / "raw" / "cfpb_az_banking_complaints_2024_2025.csv"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
ANALYSIS_DIR = PROJECT_DIR / "analysis"
FIGURES_DIR = PROJECT_DIR / "figures"
REPORT_DIR = PROJECT_DIR / "report"
DATABASE_FILE = PROCESSED_DIR / "cfpb_az_banking_complaints.sqlite"

EXPECTED_COLUMNS = [
    "Date received",
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    "Consumer complaint narrative",
    "Company public response",
    "Company",
    "State",
    "ZIP code",
    "Tags",
    "Submitted via",
    "Date sent to company",
    "Company response to consumer",
    "Timely response?",
    "Complaint ID",
]

PRODUCT_LABELS = {
    "Checking or savings account": "Checking / savings",
    "Credit card": "Credit card",
    "Money transfer, virtual currency, or money service": "Money transfer / digital wallet",
    "Prepaid card": "Prepaid card",
}

COLORS = {
    "Checking / savings": "#2F6B8A",
    "Credit card": "#D97706",
    "Money transfer / digital wallet": "#3C8D7B",
    "Prepaid card": "#7C5AA6",
}


def ensure_directories() -> None:
    for directory in [PROCESSED_DIR, ANALYSIS_DIR, FIGURES_DIR, REPORT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def load_and_validate() -> tuple[pd.DataFrame, dict[str, object]]:
    raw = pd.read_csv(RAW_FILE, low_memory=False)
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(raw.columns))
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    rows_downloaded = len(raw)
    duplicate_ids_before = int(raw["Complaint ID"].duplicated().sum())

    raw["date_received"] = pd.to_datetime(raw["Date received"], utc=True, errors="coerce")
    raw["date_sent_to_company"] = pd.to_datetime(
        raw["Date sent to company"], utc=True, errors="coerce"
    )

    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2026-01-01", tz="UTC")
    outside_window = int((~raw["date_received"].between(start, end, inclusive="left")).sum())

    clean = raw.loc[raw["date_received"].between(start, end, inclusive="left")].copy()
    clean = clean.drop_duplicates(subset=["Complaint ID"], keep="first")
    clean = clean.loc[clean["State"].eq("AZ")].copy()
    clean = clean.loc[clean["Product"].isin(PRODUCT_LABELS)].copy()

    clean["year"] = clean["date_received"].dt.year.astype("int64")
    clean["month"] = (
        clean["date_received"].dt.tz_convert(None).dt.to_period("M").astype(str)
    )
    clean["product_group"] = clean["Product"].map(PRODUCT_LABELS)
    clean["has_narrative"] = clean["Consumer complaint narrative"].notna()
    clean["timely_response"] = clean["Timely response?"].str.strip().str.lower().eq("yes")
    clean["cfpb_routing_days"] = (
        clean["date_sent_to_company"] - clean["date_received"]
    ).dt.total_seconds() / 86400

    response = clean["Company response to consumer"].fillna("Missing response")
    clean["response_group"] = np.select(
        [
            response.str.contains("non-monetary relief", case=False, na=False),
            response.str.contains("monetary relief", case=False, na=False),
            response.str.contains("explanation", case=False, na=False),
            response.str.contains("in progress", case=False, na=False),
        ],
        ["Non-monetary relief", "Monetary relief", "Explanation", "In progress"],
        default="Other / missing",
    )
    clean["relief_response"] = clean["response_group"].isin(
        ["Non-monetary relief", "Monetary relief"]
    )

    clean = clean.drop(
        columns=["Date received", "Date sent to company", "Timely response?"]
    )
    clean.columns = [
        column.strip().lower().replace(" ", "_").replace("?", "").replace("-", "_")
        for column in clean.columns
    ]

    quality = {
        "rows_downloaded": rows_downloaded,
        "rows_in_2024_2025_analysis_window": int(len(clean)),
        "rows_removed_outside_window": outside_window,
        "duplicate_complaint_ids_before_deduplication": duplicate_ids_before,
        "duplicate_complaint_ids_after_deduplication": int(
            clean["complaint_id"].duplicated().sum()
        ),
        "missing_date_received": int(clean["date_received"].isna().sum()),
        "missing_issue": int(clean["issue"].isna().sum()),
        "missing_company_response": int(
            clean["company_response_to_consumer"].isna().sum()
        ),
        "products_in_scope": sorted(clean["product_group"].dropna().unique().tolist()),
        "analysis_start": clean["date_received"].min().isoformat(),
        "analysis_end": clean["date_received"].max().isoformat(),
    }
    return clean, quality


def save_processed_data(clean: pd.DataFrame, quality: dict[str, object]) -> None:
    clean_csv = PROCESSED_DIR / "complaints_clean.csv"
    clean.to_csv(clean_csv, index=False)
    (ANALYSIS_DIR / "data_quality_report.json").write_text(
        json.dumps(quality, indent=2), encoding="utf-8"
    )

    missingness = (
        clean.isna()
        .sum()
        .rename("missing_rows")
        .to_frame()
        .assign(missing_pct=lambda frame: frame["missing_rows"] / len(clean) * 100)
        .sort_values("missing_rows", ascending=False)
    )
    missingness.to_csv(ANALYSIS_DIR / "field_missingness.csv")

    with sqlite3.connect(DATABASE_FILE) as connection:
        clean.to_sql("complaints", connection, if_exists="replace", index=False)
        connection.executescript(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_complaint_id
                ON complaints(complaint_id);
            CREATE INDEX IF NOT EXISTS idx_product_year
                ON complaints(product_group, year);
            CREATE INDEX IF NOT EXISTS idx_issue
                ON complaints(issue);
            """
        )


def run_sql_analysis() -> dict[str, pd.DataFrame]:
    queries = {
        "yearly_product_volume": """
            SELECT year, product_group, COUNT(*) AS complaints
            FROM complaints
            GROUP BY year, product_group
            ORDER BY year, complaints DESC;
        """,
        "monthly_product_volume": """
            SELECT month, product_group, COUNT(*) AS complaints
            FROM complaints
            GROUP BY month, product_group
            ORDER BY month, product_group;
        """,
        "product_response_metrics": """
            SELECT
                product_group,
                COUNT(*) AS complaints,
                ROUND(100.0 * AVG(timely_response), 2) AS timely_response_pct,
                ROUND(100.0 * AVG(relief_response), 2) AS relief_response_pct,
                ROUND(100.0 * AVG(has_narrative), 2) AS narrative_available_pct
            FROM complaints
            GROUP BY product_group
            ORDER BY complaints DESC;
        """,
        "top_issues": """
            SELECT
                issue,
                COUNT(*) AS complaints,
                ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM complaints), 2) AS share_pct
            FROM complaints
            WHERE issue IS NOT NULL
            GROUP BY issue
            ORDER BY complaints DESC
            LIMIT 12;
        """,
        "top_issue_by_product": """
            WITH issue_counts AS (
                SELECT product_group, issue, COUNT(*) AS complaints
                FROM complaints
                WHERE issue IS NOT NULL
                GROUP BY product_group, issue
            ),
            ranked AS (
                SELECT
                    product_group,
                    issue,
                    complaints,
                    ROW_NUMBER() OVER (
                        PARTITION BY product_group ORDER BY complaints DESC, issue
                    ) AS issue_rank
                FROM issue_counts
            )
            SELECT product_group, issue, complaints, issue_rank
            FROM ranked
            WHERE issue_rank <= 3
            ORDER BY product_group, issue_rank;
        """,
        "response_mix": """
            SELECT product_group, response_group, COUNT(*) AS complaints
            FROM complaints
            GROUP BY product_group, response_group
            ORDER BY product_group, complaints DESC;
        """,
        "spike_company_concentration": """
            SELECT company, issue, COUNT(*) AS complaints
            FROM complaints
            WHERE month = '2025-01'
              AND product_group = 'Money transfer / digital wallet'
            GROUP BY company, issue
            ORDER BY complaints DESC;
        """,
    }

    results: dict[str, pd.DataFrame] = {}
    with sqlite3.connect(DATABASE_FILE) as connection:
        for name, query in queries.items():
            frame = pd.read_sql_query(query, connection)
            frame.to_csv(ANALYSIS_DIR / f"{name}.csv", index=False)
            results[name] = frame
    return results


def build_dashboard(clean: pd.DataFrame, results: dict[str, pd.DataFrame]) -> None:
    canvas = Image.new("RGB", (2400, 1600), "white")
    draw = ImageDraw.Draw(canvas)

    def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        candidates = (
            ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "DejaVuSans-Bold.ttf"]
            if bold else
            ["/System/Library/Fonts/Supplemental/Arial.ttf", "DejaVuSans.ttf"]
        )
        for candidate in candidates:
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
        return ImageFont.load_default(size=size)

    regular = lambda size: font(size)
    bold = lambda size: font(size, bold=True)

    ink = "#173042"
    muted = "#5D6B75"
    grid = "#D8E0E5"
    panel_fill = "#F7F9FA"
    response_colors = {
        "Explanation": "#5B7FA3",
        "Non-monetary relief": "#4C9A8A",
        "Monetary relief": "#D9A441",
        "In progress": "#9B8FB5",
        "Other / missing": "#B0B0B0",
    }

    draw.text((90, 55), "ARIZONA CONSUMER BANKING COMPLAINTS", font=bold(54), fill=ink)
    draw.text((90, 120), "CFPB published records | Calendar years 2024-2025", font=regular(28), fill=muted)

    def panel(box: tuple[int, int, int, int], title: str) -> tuple[int, int, int, int]:
        draw.rounded_rectangle(box, radius=18, fill=panel_fill, outline="#E0E6EA", width=2)
        x1, y1, x2, y2 = box
        draw.text((x1 + 34, y1 + 24), title, font=bold(30), fill=ink)
        return x1 + 50, y1 + 90, x2 - 40, y2 - 45

    def wrap(text: str, max_chars: int) -> list[str]:
        words = str(text).split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    # Panel 1: monthly volume line chart.
    plot = panel((70, 190, 1180, 830), "Monthly published complaints")
    px1, py1, px2, py2 = plot
    monthly = results["monthly_product_volume"].pivot(
        index="month", columns="product_group", values="complaints"
    ).fillna(0)
    max_value = max(float(monthly.max().max()), 1)
    for tick in range(0, 5):
        y = py2 - (py2 - py1) * tick / 4
        value = max_value * tick / 4
        draw.line((px1, y, px2, y), fill=grid, width=1)
        draw.text((px1 - 10, y - 12), f"{value:.0f}", anchor="ra", font=regular(20), fill=muted)
    month_count = max(len(monthly.index) - 1, 1)
    for column in monthly.columns:
        points = []
        for index, value in enumerate(monthly[column].tolist()):
            x = px1 + (px2 - px1) * index / month_count
            y = py2 - (py2 - py1) * float(value) / max_value
            points.append((x, y))
        draw.line(points, fill=COLORS[column], width=5, joint="curve")
        for point in points:
            draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=COLORS[column])
    for index, label in enumerate(monthly.index):
        if index % 3 == 0 or index == len(monthly.index) - 1:
            x = px1 + (px2 - px1) * index / month_count
            draw.text((x, py2 + 14), label, anchor="ma", font=regular(18), fill=muted)
    legend_x, legend_y = px1, py1 - 40
    for column in monthly.columns:
        draw.rectangle((legend_x, legend_y + 5, legend_x + 26, legend_y + 21), fill=COLORS[column])
        draw.text((legend_x + 36, legend_y), column, font=regular(18), fill=muted)
        legend_x += 240 if len(column) < 20 else 330

    # Panel 2: top issues horizontal bars.
    plot = panel((1220, 190, 2330, 830), "Most common issues")
    px1, py1, px2, py2 = plot
    top_issues = results["top_issues"].head(7).iloc[::-1]
    max_issue = max(int(top_issues["complaints"].max()), 1)
    row_height = (py2 - py1) / len(top_issues)
    label_width = 380
    for row_index, (_, row) in enumerate(top_issues.iterrows()):
        y = py1 + row_index * row_height
        label_lines = wrap(str(row["issue"]), 36)[:2]
        for line_index, line in enumerate(label_lines):
            draw.text((px1, y + line_index * 20), line, font=regular(18), fill=ink)
        bar_x = px1 + label_width
        bar_y = y + 8
        bar_width = (px2 - bar_x - 90) * int(row["complaints"]) / max_issue
        draw.rounded_rectangle((bar_x, bar_y, bar_x + bar_width, bar_y + 28), radius=6, fill="#2F6B8A")
        draw.text((bar_x + bar_width + 10, bar_y + 2), f"{int(row['complaints']):,}", font=regular(18), fill=muted)

    # Panel 3: response mix stacked bars.
    plot = panel((70, 870, 1180, 1480), "Company response categories")
    px1, py1, px2, py2 = plot
    response = results["response_mix"].pivot(
        index="product_group", columns="response_group", values="complaints"
    ).fillna(0)
    response_pct = response.div(response.sum(axis=1), axis=0) * 100
    response_order = list(response_colors)
    row_height = 80
    for row_index, (product, row) in enumerate(response_pct.iterrows()):
        y = py1 + row_index * row_height
        draw.text((px1, y + 8), product, font=regular(20), fill=ink)
        bar_x = px1 + 340
        current_x = bar_x
        available_width = px2 - bar_x
        for category in response_order:
            value = float(row.get(category, 0))
            width = available_width * value / 100
            if width > 0:
                draw.rectangle((current_x, y, current_x + width, y + 40), fill=response_colors[category])
            current_x += width
    legend_x, legend_y = px1, py2 - 75
    for category in response_order:
        draw.rectangle((legend_x, legend_y, legend_x + 22, legend_y + 18), fill=response_colors[category])
        draw.text((legend_x + 30, legend_y - 3), category, font=regular(17), fill=muted)
        legend_x += 185 if len(category) < 14 else 245

    # Panel 4: product response indicators.
    plot = panel((1220, 870, 2330, 1480), "Response indicators by product")
    px1, py1, px2, py2 = plot
    product_metrics = results["product_response_metrics"].set_index("product_group")
    chart_top = py1 + 10
    chart_bottom = py2 - 105
    for tick in [0, 25, 50, 75, 100]:
        y = chart_bottom - (chart_bottom - chart_top) * tick / 100
        draw.line((px1, y, px2, y), fill=grid, width=1)
        draw.text((px1 - 10, y), f"{tick}%", anchor="rm", font=regular(18), fill=muted)
    group_width = (px2 - px1) / len(product_metrics)
    for index, (product, row) in enumerate(product_metrics.iterrows()):
        center = px1 + group_width * (index + 0.5)
        for offset, field, color in [
            (-32, "timely_response_pct", "#2F6B8A"),
            (32, "relief_response_pct", "#D9A441"),
        ]:
            value = float(row[field])
            height = (chart_bottom - chart_top) * value / 100
            draw.rectangle((center + offset - 24, chart_bottom - height, center + offset + 24, chart_bottom), fill=color)
        for line_index, line in enumerate(wrap(product, 18)[:2]):
            draw.text((center, chart_bottom + 18 + line_index * 19), line, anchor="ma", font=regular(17), fill=ink)
    draw.rectangle((px1, py2 - 52, px1 + 22, py2 - 34), fill="#2F6B8A")
    draw.text((px1 + 32, py2 - 56), "Timely response", font=regular(17), fill=muted)
    draw.rectangle((px1 + 240, py2 - 52, px1 + 262, py2 - 34), fill="#D9A441")
    draw.text((px1 + 272, py2 - 56), "Monetary or non-monetary relief", font=regular(17), fill=muted)

    draw.text(
        (90, 1540),
        "Source: CFPB Consumer Complaint Database. Published complaints are not a statistical sample of all consumer experiences.",
        font=regular(19),
        fill=muted,
    )
    canvas.save(FIGURES_DIR / "complaint_analytics_dashboard.png", quality=95)


def write_findings(clean: pd.DataFrame, results: dict[str, pd.DataFrame]) -> dict[str, object]:
    yearly = clean.groupby("year").size()
    complaints_2024 = int(yearly.get(2024, 0))
    complaints_2025 = int(yearly.get(2025, 0))
    yoy_pct = round((complaints_2025 / complaints_2024 - 1) * 100, 1)

    top_issue = results["top_issues"].iloc[0]
    top_product = clean["product_group"].value_counts().index[0]
    timely_pct = round(clean["timely_response"].mean() * 100, 2)
    relief_pct = round(clean["relief_response"].mean() * 100, 2)
    narrative_pct = round(clean["has_narrative"].mean() * 100, 2)

    spike = clean.loc[
        clean["month"].eq("2025-01")
        & clean["product_group"].eq("Money transfer / digital wallet")
    ]
    prior_year_spike_month = clean.loc[
        clean["month"].eq("2024-01")
        & clean["product_group"].eq("Money transfer / digital wallet")
    ]
    spike_company_counts = spike["company"].value_counts()
    top_two_spike_companies = int(spike_company_counts.head(2).sum())
    top_two_spike_company_share = round(top_two_spike_companies / len(spike) * 100, 1)
    spike_yoy_delta = int(len(spike) - len(prior_year_spike_month))
    total_yoy_delta = complaints_2025 - complaints_2024
    spike_share_of_yoy_delta = round(spike_yoy_delta / total_yoy_delta * 100, 1)
    repeated_narratives_in_spike = int(
        spike["consumer_complaint_narrative"].dropna().duplicated().sum()
    )

    findings = {
        "records_analyzed": int(len(clean)),
        "fields_in_source": len(EXPECTED_COLUMNS),
        "complaints_2024": complaints_2024,
        "complaints_2025": complaints_2025,
        "year_over_year_change_pct": yoy_pct,
        "largest_product_group": top_product,
        "largest_product_group_records": int(
            clean["product_group"].value_counts().iloc[0]
        ),
        "top_issue": str(top_issue["issue"]),
        "top_issue_records": int(top_issue["complaints"]),
        "top_issue_share_pct": float(top_issue["share_pct"]),
        "timely_response_pct": timely_pct,
        "relief_response_pct": relief_pct,
        "narrative_available_pct": narrative_pct,
        "jan_2025_money_transfer_records": int(len(spike)),
        "jan_2024_money_transfer_records": int(len(prior_year_spike_month)),
        "jan_money_transfer_yoy_delta": spike_yoy_delta,
        "spike_share_of_total_yoy_increase_pct": spike_share_of_yoy_delta,
        "top_two_company_share_of_spike_pct": top_two_spike_company_share,
        "repeated_narratives_in_spike_with_unique_complaint_ids": repeated_narratives_in_spike,
    }
    (ANALYSIS_DIR / "key_findings.json").write_text(
        json.dumps(findings, indent=2), encoding="utf-8"
    )

    direction = "increased" if yoy_pct >= 0 else "decreased"
    summary = f"""# Executive findings

- Analyzed **{len(clean):,}** published CFPB complaints from Arizona across checking/savings, credit card, prepaid card, and money-transfer/digital-wallet products for calendar years 2024-2025.
- Published complaint volume **{direction} {abs(yoy_pct):.1f}%** from {complaints_2024:,} in 2024 to {complaints_2025:,} in 2025.
- **{top_product}** was the largest product group with {findings['largest_product_group_records']:,} records.
- The most common issue was **{top_issue['issue']}**, representing {int(top_issue['complaints']):,} complaints ({float(top_issue['share_pct']):.1f}% of the analyzed records).
- Companies marked **{timely_pct:.2f}%** of complaints as timely responses; {relief_pct:.2f}% were closed with monetary or non-monetary relief.
- Consumer narratives were available for {narrative_pct:.2f}% of records, limiting text-based conclusions to the opt-in subset.
- A January 2025 money-transfer/digital-wallet spike added **{spike_yoy_delta:,}** complaints versus January 2024 and represented **{spike_share_of_yoy_delta:.1f}%** of the total year-over-year increase. Two companies accounted for **{top_two_spike_company_share:.1f}%** of that month's product complaints.
- The spike contained **{repeated_narratives_in_spike:,} repeated narrative texts** attached to unique complaint IDs. The project preserved those records and flagged the repetition for review instead of assuming they were duplicate complaints.

## Operational recommendations

1. Prioritize root-cause review for the top three issues within each product instead of treating all complaints as one queue.
2. Investigate concentrated spikes and repeated narratives before interpreting a volume change as a broad market trend.
3. Track complaint volume, response category, and timeliness by product monthly; use volume as a workload indicator, not as a standalone quality ranking.
4. Add denominator data such as accounts, transactions, or market share before comparing companies or geographies.

## Interpretation limits

The CFPB states that the complaint database is not a statistical sample of all consumers' experiences. Complaint counts are influenced by company size, product usage, consumer reporting behavior, and publication rules. This project describes the published records and does not rank company quality.
"""
    (REPORT_DIR / "executive_findings.md").write_text(summary, encoding="utf-8")
    return findings


def main() -> None:
    ensure_directories()
    clean, quality = load_and_validate()
    save_processed_data(clean, quality)
    results = run_sql_analysis()
    build_dashboard(clean, results)
    findings = write_findings(clean, results)
    print(json.dumps(findings, indent=2))


if __name__ == "__main__":
    main()
