"""
E-commerce Sales Data Analysis and Visualization
================================================

Workflow (run with:  python main.py):
    1. Load dataset            6. Product analysis
    2. Validate dataset        7. Region analysis
    3. Explore dataset         8. Monthly analysis
    4. Clean dataset           9. Generate visualizations
    5. Calculate metrics      10. Generate insights and reports

Every number, insight and recommendation is calculated from data/sales_data.csv.
Nothing is hardcoded.
"""

import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Draw charts to files only (no pop-up windows, works everywhere)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

# ---------------------------------------------------------------------------
# Paths and constants (pathlib keeps everything working on Windows/Mac/Linux)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "sales_data.csv"
VISUALIZATIONS_DIR = BASE_DIR / "visualizations"
REPORT_DIR = BASE_DIR / "report"
README_PATH = BASE_DIR / "README.md"

REQUIRED_COLUMNS = [
    "Date", "Product", "Quantity", "Unit_Price",
    "Customer_ID", "Region", "Total_Sales",
]
NUMERIC_COLUMNS = ["Quantity", "Unit_Price", "Total_Sales"]
TEXT_COLUMNS = ["Product", "Customer_ID", "Region"]

CHART_FILES = {
    "product_bar": "product_sales_bar_chart.png",
    "monthly_line": "monthly_sales_line_chart.png",
    "regional_bar": "regional_sales_chart.png",
    "product_pie": "product_revenue_pie_chart.png",
}

PIE_COLORS = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
    "#EECA3B", "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC",
]


class DataValidationError(Exception):
    """Raised when the dataset is empty, malformed or unusable."""


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------
def money(value):
    """Format a number as 12,345.67."""
    return f"{value:,.2f}"


def units(value):
    """Format unit counts: whole numbers without decimals."""
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.2f}"


# ---------------------------------------------------------------------------
# 1-3. Loading, validating and exploring
# ---------------------------------------------------------------------------
def load_data(file_path):
    """Load the CSV file. Fails with a clear message if it is missing or empty."""
    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}\n"
            "Place 'sales_data.csv' inside the 'data' folder and run again."
        )

    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        raise DataValidationError("The CSV file is completely empty.")
    except pd.errors.ParserError as error:
        raise DataValidationError(f"The CSV file could not be parsed: {error}")

    df.columns = df.columns.astype(str).str.strip()  # remove accidental spaces in headers

    if df.empty:
        raise DataValidationError("The CSV file has headers but no data rows.")
    return df


def validate_dataset(df):
    """Check that we have a non-empty DataFrame with every required column."""
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError("Input must be a pandas DataFrame.")
    if df.empty:
        raise DataValidationError("The dataset is empty - nothing to analyse.")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise DataValidationError(
            f"Missing required column(s): {', '.join(missing)}.\n"
            f"Expected columns: {', '.join(REQUIRED_COLUMNS)}.\n"
            f"Found columns: {', '.join(map(str, df.columns))}."
        )
    return True


def explore_data(df):
    """Print shape, first five rows, column names and data types."""
    print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nFirst 5 rows:")
    print(df.head().to_string())
    print("\nColumn names and data types:")
    for column, dtype in df.dtypes.items():
        print(f"  - {column}: {dtype}")
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())


# ---------------------------------------------------------------------------
# 4. Cleaning
# ---------------------------------------------------------------------------
def _clean_text(value):
    """Strip spaces from text values; blank values become NaN."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, float) and value.is_integer():
        value = int(value)  # 101.0 -> 101
    text = str(value).strip()
    return text if text else np.nan


def _clean_number_text(value):
    """Remove thousands separators and spaces from text like ' 1,200 '."""
    if isinstance(value, str):
        return value.replace(",", "").strip()
    return value


def parse_dates(series):
    """Convert a column to datetime. Invalid dates become NaT instead of crashing."""
    if int(pd.__version__.split(".")[0]) >= 2:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    return pd.to_datetime(series, errors="coerce")


def clean_data(df):
    """
    Clean the dataset and return (clean_df, summary).

    Steps: tidy text -> convert types -> remove duplicates -> handle missing
    values -> remove invalid rows -> recalculate Total_Sales.
    """
    validate_dataset(df)
    df = df[REQUIRED_COLUMNS].copy()

    summary = {
        "initial_rows": len(df),
        "missing_before": {col: int(df[col].isna().sum()) for col in REQUIRED_COLUMNS},
    }

    # --- Text columns: strip spaces, blanks -> missing
    for col in TEXT_COLUMNS:
        df[col] = df[col].map(_clean_text)

    # --- Numeric columns: convert; unreadable values become NaN
    invalid_numeric = 0
    for col in NUMERIC_COLUMNS:
        original = df[col].map(_clean_number_text)
        converted = pd.to_numeric(original, errors="coerce")
        bad = int((original.notna() & converted.isna()).sum())
        if original.notna().sum() > 0 and converted.notna().sum() == 0:
            raise DataValidationError(f"Column '{col}' has no valid numeric values.")
        invalid_numeric += bad
        df[col] = converted
    summary["invalid_numeric_values"] = invalid_numeric

    # --- Date column: convert to datetime
    original_dates = df["Date"]
    dates = parse_dates(original_dates)
    if original_dates.notna().sum() > 0 and dates.notna().sum() == 0:
        raise DataValidationError("Column 'Date' has no valid dates. Use a format like YYYY-MM-DD.")
    summary["invalid_date_values"] = int((original_dates.notna() & dates.isna()).sum())
    df["Date"] = dates

    # --- Remove duplicate rows
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    summary["duplicates_removed"] = before - len(df)

    # --- Rows without a usable date or product cannot be analysed
    before = len(df)
    df = df[df["Date"].notna()].reset_index(drop=True)
    summary["rows_removed_missing_date"] = before - len(df)

    before = len(df)
    df = df[df["Product"].notna()].reset_index(drop=True)
    summary["rows_removed_missing_product"] = before - len(df)

    # --- Missing Region / Customer_ID are not critical: label them "Unknown"
    summary["filled_unknown_region"] = int(df["Region"].isna().sum())
    summary["filled_unknown_customer"] = int(df["Customer_ID"].isna().sum())
    df["Region"] = df["Region"].fillna("Unknown")
    df["Customer_ID"] = df["Customer_ID"].fillna("Unknown")

    # --- Recover a missing Quantity or Unit_Price from Total_Sales when possible
    derived = 0
    total, price = df["Total_Sales"], df["Unit_Price"]
    can_qty = df["Quantity"].isna() & price.notna() & (price > 0) & total.notna()
    estimated_qty = total[can_qty] / price[can_qty]  # price > 0, so no division by zero
    is_whole = np.isclose(estimated_qty, estimated_qty.round(), atol=1e-6)
    df.loc[estimated_qty.index[is_whole], "Quantity"] = estimated_qty[is_whole].round()
    derived += int(is_whole.sum())

    qty = df["Quantity"]
    can_price = df["Unit_Price"].isna() & qty.notna() & (qty > 0) & total.notna()
    df.loc[can_price, "Unit_Price"] = total[can_price] / qty[can_price]
    derived += int(can_price.sum())
    summary["values_recovered"] = derived

    # --- Rows still missing Quantity or Unit_Price cannot be repaired
    before = len(df)
    df = df[df["Quantity"].notna() & df["Unit_Price"].notna()].reset_index(drop=True)
    summary["rows_removed_missing_values"] = before - len(df)

    # --- Negative or zero quantity / price are invalid
    before = len(df)
    df = df[(df["Quantity"] > 0) & (df["Unit_Price"] > 0)].reset_index(drop=True)
    summary["rows_removed_non_positive"] = before - len(df)

    # --- Recalculate Total_Sales = Quantity x Unit_Price where missing or inconsistent
    expected = (df["Quantity"] * df["Unit_Price"]).round(2)
    mismatch = df["Total_Sales"].isna() | ~np.isclose(df["Total_Sales"], expected, atol=0.01)
    summary["total_sales_recalculated"] = int(mismatch.sum())
    df["Total_Sales"] = expected

    # --- Tidy types
    if len(df) > 0 and (df["Quantity"] % 1 == 0).all():
        df["Quantity"] = df["Quantity"].astype(int)

    if df.empty:
        raise DataValidationError("No valid rows are left after cleaning. Please check the dataset.")

    df = df.sort_values("Date").reset_index(drop=True)
    summary["final_rows"] = len(df)
    summary["rows_removed_total"] = summary["initial_rows"] - summary["final_rows"]
    return df, summary


def print_cleaning_summary(summary):
    """Display what the cleaning step changed."""
    print(f"Rows before cleaning:                 {summary['initial_rows']}")
    print(f"Duplicate rows removed:               {summary['duplicates_removed']}")
    print(f"Unreadable numeric values found:      {summary['invalid_numeric_values']}")
    print(f"Invalid date values found:            {summary['invalid_date_values']}")
    print(f"Rows removed (missing/invalid date):  {summary['rows_removed_missing_date']}")
    print(f"Rows removed (missing product):       {summary['rows_removed_missing_product']}")
    print(f"Region set to 'Unknown':              {summary['filled_unknown_region']}")
    print(f"Customer_ID set to 'Unknown':         {summary['filled_unknown_customer']}")
    print(f"Quantity/Unit_Price recovered:        {summary['values_recovered']}")
    print(f"Rows removed (missing qty/price):     {summary['rows_removed_missing_values']}")
    print(f"Rows removed (negative/zero values):  {summary['rows_removed_non_positive']}")
    print(f"Total_Sales recalculated:             {summary['total_sales_recalculated']}")
    print(f"Rows after cleaning:                  {summary['final_rows']}")


# ---------------------------------------------------------------------------
# 5-8. Metrics and analysis
# ---------------------------------------------------------------------------
def calculate_metrics(df):
    """Calculate the headline business metrics."""
    if df is None or df.empty:
        raise DataValidationError("Cannot calculate metrics on an empty dataset.")

    total_transactions = len(df)
    total_revenue = float(df["Total_Sales"].sum())
    total_units = float(df["Quantity"].sum())
    # Division by zero is impossible here (we checked for an empty dataset above)
    average_transaction = total_revenue / total_transactions

    highest = df.loc[df["Total_Sales"].idxmax()]
    lowest = df.loc[df["Total_Sales"].idxmin()]

    revenue_by_product = df.groupby("Product")["Total_Sales"].sum()
    units_by_product = df.groupby("Product")["Quantity"].sum()

    def describe(row):
        return {
            "value": float(row["Total_Sales"]),
            "product": row["Product"],
            "date": row["Date"].strftime("%Y-%m-%d"),
            "customer": row["Customer_ID"],
        }

    return {
        "total_revenue": total_revenue,
        "total_units": total_units,
        "total_transactions": total_transactions,
        "average_transaction": average_transaction,
        "highest_transaction": describe(highest),
        "lowest_transaction": describe(lowest),
        "best_selling_product": units_by_product.idxmax(),
        "best_selling_units": float(units_by_product.max()),
        "top_revenue_product": revenue_by_product.idxmax(),
        "top_revenue_value": float(revenue_by_product.max()),
        "unique_customers": int(df["Customer_ID"].nunique()),
    }


def _group_analysis(df, column):
    """Group by a column and summarise units, revenue, transactions and share of revenue."""
    grouped = df.groupby(column).agg(
        Units_Sold=("Quantity", "sum"),
        Revenue=("Total_Sales", "sum"),
        Transactions=("Total_Sales", "count"),
    ).reset_index()

    total_revenue = grouped["Revenue"].sum()
    if total_revenue > 0:
        grouped["Revenue_Share_Pct"] = (grouped["Revenue"] / total_revenue * 100).round(2)
    else:  # avoids division by zero
        grouped["Revenue_Share_Pct"] = 0.0

    grouped["Revenue"] = grouped["Revenue"].round(2)
    return grouped.sort_values("Revenue", ascending=False).reset_index(drop=True)


def product_analysis(df):
    """Sales by product (sorted by revenue, highest first)."""
    return _group_analysis(df, "Product")


def region_analysis(df):
    """Sales by region (sorted by revenue, highest first)."""
    return _group_analysis(df, "Region")


def monthly_analysis(df):
    """Monthly revenue, units and transactions in chronological order."""
    temp = df.copy()
    temp["Month"] = temp["Date"].dt.to_period("M").astype(str)  # e.g. 2024-03
    monthly = temp.groupby("Month").agg(
        Units_Sold=("Quantity", "sum"),
        Revenue=("Total_Sales", "sum"),
        Transactions=("Total_Sales", "count"),
    ).reset_index().sort_values("Month").reset_index(drop=True)

    monthly["Revenue"] = monthly["Revenue"].round(2)
    growth = monthly["Revenue"].pct_change() * 100
    monthly["Growth_Pct"] = growth.replace([np.inf, -np.inf], np.nan).round(2)
    return monthly


# ---------------------------------------------------------------------------
# 5. Business insights (generated from the data, never hardcoded)
# ---------------------------------------------------------------------------
def generate_insights(metrics, product_df, region_df, monthly_df):
    """Return {'insights': [...], 'recommendations': [...]} built from the results."""
    insights = []
    recs = []

    # Products
    top_product = product_df.iloc[0]
    insights.append(
        f"**{top_product['Product']}** generates the highest revenue "
        f"({money(top_product['Revenue'])}), contributing {top_product['Revenue_Share_Pct']:.1f}% of total revenue."
    )

    units_leader = product_df.sort_values("Units_Sold", ascending=False).iloc[0]
    same_leader = units_leader["Product"] == top_product["Product"]
    if same_leader:
        insights.append(
            f"**{units_leader['Product']}** is also the best-selling product by volume "
            f"({units(units_leader['Units_Sold'])} units)."
        )
    else:
        insights.append(
            f"**{units_leader['Product']}** sells the most units ({units(units_leader['Units_Sold'])}), "
            f"but **{top_product['Product']}** earns more revenue - volume and revenue leaders are different products."
        )

    if len(product_df) > 3:
        top3_share = product_df.head(3)["Revenue_Share_Pct"].sum()
        insights.append(f"The top 3 products together account for {top3_share:.1f}% of total revenue.")

    # Regions (rows with a missing region are labelled "Unknown" and left out of the comparison)
    known_regions = region_df[region_df["Region"] != "Unknown"]
    top_region = known_regions.iloc[0] if not known_regions.empty else None
    low_region = known_regions.iloc[-1] if len(known_regions) > 1 else None
    if top_region is not None:
        insights.append(
            f"**{top_region['Region']}** is the highest-revenue region ({money(top_region['Revenue'])}, "
            f"{top_region['Revenue_Share_Pct']:.1f}% of total revenue)."
        )
    if low_region is not None:
        insights.append(
            f"**{low_region['Region']}** is the lowest-revenue region ({money(low_region['Revenue'])}, "
            f"{low_region['Revenue_Share_Pct']:.1f}% of total revenue)."
        )

    # Months
    best_month = monthly_df.loc[monthly_df["Revenue"].idxmax()]
    insights.append(f"**{best_month['Month']}** is the strongest month with {money(best_month['Revenue'])} in revenue.")
    if len(monthly_df) > 1:
        worst_month = monthly_df.loc[monthly_df["Revenue"].idxmin()]
        insights.append(f"**{worst_month['Month']}** is the weakest month with {money(worst_month['Revenue'])} in revenue.")
        first, last = monthly_df.iloc[0], monthly_df.iloc[-1]
        if first["Revenue"] > 0:
            change = (last["Revenue"] - first["Revenue"]) / first["Revenue"] * 100
            direction = "up" if change >= 0 else "down"
            insights.append(
                f"Revenue in the latest month ({last['Month']}) is {direction} {abs(change):.1f}% "
                f"compared with the first month ({first['Month']})."
            )

    # Transactions
    high = metrics["highest_transaction"]
    insights.append(
        f"The average transaction value is {money(metrics['average_transaction'])}; the largest single sale was "
        f"{money(high['value'])} ({high['product']} on {high['date']})."
    )

    # Recommendations
    recs.append(
        f"Keep **{top_product['Product']}** well stocked and prominently promoted - it is the biggest revenue driver."
    )
    if not same_leader:
        recs.append(
            f"Review the price and bundling of **{units_leader['Product']}**: it sells the most units "
            f"but does not lead in revenue, so small price or bundle changes could lift income."
        )
    if len(product_df) > 1:
        weakest = product_df.iloc[-1]
        recs.append(
            f"Investigate **{weakest['Product']}** (lowest revenue, {weakest['Revenue_Share_Pct']:.1f}% share): "
            f"test a promotion or pricing change before deciding to reduce stock."
        )
    if top_region is not None and low_region is not None:
        recs.append(
            f"Study what works in **{top_region['Region']}** and try similar campaigns in **{low_region['Region']}**."
        )
        if top_region["Revenue_Share_Pct"] > 50:
            recs.append(
                f"More than half of revenue comes from **{top_region['Region']}**; "
                f"growing other regions would reduce dependence on a single market."
            )
    recs.append(
        f"Plan inventory and marketing ahead of high-demand periods such as **{best_month['Month']}**."
    )
    recs.append(
        f"Raise the average transaction value ({money(metrics['average_transaction'])}) with cross-selling, "
        f"bundles or minimum-order offers."
    )
    return {"insights": insights, "recommendations": recs}


# ---------------------------------------------------------------------------
# 9. Visualizations
# ---------------------------------------------------------------------------
def _thousands(value, _position):
    return f"{value:,.0f}"


def _save_figure(fig, output_path):
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")


def create_product_bar_chart(product_df, output_path):
    """Chart 1: revenue per product."""
    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        bars = ax.bar(product_df["Product"], product_df["Revenue"], color="#4C78A8", edgecolor="white")
        ax.bar_label(bars, labels=[f"{v:,.0f}" for v in product_df["Revenue"]], padding=3, fontsize=9)
        ax.set_title("Revenue by Product", fontsize=15, fontweight="bold")
        ax.set_xlabel("Product", fontsize=12)
        ax.set_ylabel("Revenue", fontsize=12)
        ax.yaxis.set_major_formatter(FuncFormatter(_thousands))
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        ax.margins(y=0.12)
        if len(product_df) > 5:
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        _save_figure(fig, output_path)
    finally:
        plt.close(fig)


def create_monthly_line_chart(monthly_df, output_path):
    """Chart 2: revenue per month."""
    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        ax.plot(monthly_df["Month"], monthly_df["Revenue"], marker="o", linewidth=2,
                color="#E45756", label="Monthly revenue")
        if len(monthly_df) <= 12:  # value labels only when they stay readable
            for month, revenue in zip(monthly_df["Month"], monthly_df["Revenue"]):
                ax.annotate(f"{revenue:,.0f}", (month, revenue), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=9)
        ax.set_title("Monthly Sales Trend", fontsize=15, fontweight="bold")
        ax.set_xlabel("Month", fontsize=12)
        ax.set_ylabel("Revenue", fontsize=12)
        ax.yaxis.set_major_formatter(FuncFormatter(_thousands))
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.margins(y=0.15)
        ax.legend()
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        _save_figure(fig, output_path)
    finally:
        plt.close(fig)


def create_regional_bar_chart(region_df, output_path):
    """Chart 3: revenue per region."""
    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        bars = ax.bar(region_df["Region"], region_df["Revenue"], color="#54A24B", edgecolor="white")
        ax.bar_label(bars, labels=[f"{v:,.0f}" for v in region_df["Revenue"]], padding=3, fontsize=9)
        ax.set_title("Revenue by Region", fontsize=15, fontweight="bold")
        ax.set_xlabel("Region", fontsize=12)
        ax.set_ylabel("Revenue", fontsize=12)
        ax.yaxis.set_major_formatter(FuncFormatter(_thousands))
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        ax.margins(y=0.12)
        _save_figure(fig, output_path)
    finally:
        plt.close(fig)


def create_product_pie_chart(product_df, output_path):
    """Chart 4: share of revenue per product."""
    data = product_df[product_df["Revenue"] > 0]
    if data.empty:
        raise DataValidationError("No positive revenue to show in the pie chart.")

    fig, ax = plt.subplots(figsize=(10, 7))
    try:
        colors = [PIE_COLORS[i % len(PIE_COLORS)] for i in range(len(data))]
        wedges, _texts, _autotexts = ax.pie(
            data["Revenue"],
            autopct=lambda pct: f"{pct:.1f}%" if pct >= 3 else "",  # hide labels on tiny slices
            startangle=90,
            colors=colors,
            pctdistance=0.78,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
            textprops={"fontsize": 10},
        )
        legend_labels = [
            f"{name} ({share:.1f}%)" for name, share in zip(data["Product"], data["Revenue_Share_Pct"])
        ]
        ax.legend(wedges, legend_labels, title="Product", loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set_title("Product Revenue Distribution", fontsize=15, fontweight="bold")
        ax.axis("equal")
        _save_figure(fig, output_path)
    finally:
        plt.close(fig)


def generate_visualizations(product_df, region_df, monthly_df, output_dir):
    """Create all four charts. A failed chart is reported but does not stop the others."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = {
        "product_bar": (create_product_bar_chart, product_df),
        "monthly_line": (create_monthly_line_chart, monthly_df),
        "regional_bar": (create_regional_bar_chart, region_df),
        "product_pie": (create_product_pie_chart, product_df),
    }

    results = {}
    for key, (function, data) in jobs.items():
        path = output_dir / CHART_FILES[key]
        try:
            function(data, path)
            results[key] = path
            print(f"  Saved: {path.name}")
        except Exception as error:  # chart problems should never crash the whole analysis
            results[key] = None
            print(f"  WARNING: could not create {CHART_FILES[key]}: {error}")
    return results


# ---------------------------------------------------------------------------
# 10. Reports
# ---------------------------------------------------------------------------
def _format_cell(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if isinstance(value, (bool, np.bool_)):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):,.2f}"
    return str(value).replace("|", "\\|")


def dataframe_to_markdown(df):
    """Turn a DataFrame into a Markdown table (no extra libraries needed)."""
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(_format_cell(value) for value in row) + " |")
    return "\n".join(lines)


def _chart_markdown(charts, key, alt_text, prefix):
    """Markdown for a chart image (only if the chart was actually created)."""
    if charts.get(key):
        return f"![{alt_text}]({prefix}{CHART_FILES[key]})"
    return f"_{alt_text} could not be generated._"


def _top_known_region(region_df):
    """Highest-revenue region, ignoring the 'Unknown' label used for missing regions."""
    known = region_df[region_df["Region"] != "Unknown"]
    return known.iloc[0] if not known.empty else region_df.iloc[0]


def generate_report(df, summary, metrics, product_df, region_df, monthly_df, insight_data, charts, output_path):
    """Write report/analysis_report.md (charts use relative paths)."""
    prefix = "../visualizations/"
    date_min = df["Date"].min().strftime("%Y-%m-%d")
    date_max = df["Date"].max().strftime("%Y-%m-%d")

    column_info = {
        "Date": "Date of the transaction",
        "Product": "Name of the product sold",
        "Quantity": "Units sold in the transaction",
        "Unit_Price": "Price of one unit",
        "Customer_ID": "Customer identifier",
        "Region": "Region where the sale happened",
        "Total_Sales": "Quantity x Unit_Price",
    }
    columns_df = pd.DataFrame({
        "Column": REQUIRED_COLUMNS,
        "Description": [column_info[c] for c in REQUIRED_COLUMNS],
        "Data Type (after cleaning)": [str(df[c].dtype) for c in REQUIRED_COLUMNS],
    })

    cleaning_df = pd.DataFrame([
        ("Rows before cleaning", summary["initial_rows"]),
        ("Duplicate rows removed", summary["duplicates_removed"]),
        ("Unreadable numeric values found", summary["invalid_numeric_values"]),
        ("Invalid date values found", summary["invalid_date_values"]),
        ("Rows removed (missing or invalid date)", summary["rows_removed_missing_date"]),
        ("Rows removed (missing product)", summary["rows_removed_missing_product"]),
        ("Region set to 'Unknown'", summary["filled_unknown_region"]),
        ("Customer_ID set to 'Unknown'", summary["filled_unknown_customer"]),
        ("Quantity / Unit_Price recovered from Total_Sales", summary["values_recovered"]),
        ("Rows removed (missing quantity or price)", summary["rows_removed_missing_values"]),
        ("Rows removed (negative or zero values)", summary["rows_removed_non_positive"]),
        ("Total_Sales values recalculated", summary["total_sales_recalculated"]),
        ("Rows after cleaning", summary["final_rows"]),
    ], columns=["Cleaning step", "Count"])

    high, low = metrics["highest_transaction"], metrics["lowest_transaction"]
    stats_df = pd.DataFrame([
        ("Total revenue", money(metrics["total_revenue"])),
        ("Total units sold", units(metrics["total_units"])),
        ("Total transactions", f"{metrics['total_transactions']:,}"),
        ("Average transaction value", money(metrics["average_transaction"])),
        ("Highest sales transaction", f"{money(high['value'])} ({high['product']}, {high['date']})"),
        ("Lowest sales transaction", f"{money(low['value'])} ({low['product']}, {low['date']})"),
        ("Best-selling product (units)", f"{metrics['best_selling_product']} ({units(metrics['best_selling_units'])} units)"),
        ("Most profitable product (revenue)", f"{metrics['top_revenue_product']} ({money(metrics['top_revenue_value'])})"),
        ("Unique customers", f"{metrics['unique_customers']:,}"),
    ], columns=["Metric", "Value"])

    top_product = product_df.iloc[0]
    top_region = _top_known_region(region_df)
    best_month = monthly_df.loc[monthly_df["Revenue"].idxmax()]

    lines = [
        "# E-commerce Sales Data Analysis Report",
        "",
        f"_Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')} by `main.py`. All figures come from `data/sales_data.csv`._",
        "",
        "## 1. Project Overview",
        "",
        "This project loads e-commerce sales data, cleans it, explores it, calculates business metrics, "
        "creates Matplotlib charts and produces insights and recommendations.",
        "",
        "## 2. Dataset Description",
        "",
        f"- **Period covered:** {date_min} to {date_max}",
        f"- **Transactions (after cleaning):** {len(df):,}",
        f"- **Products:** {df['Product'].nunique()}",
        f"- **Regions:** {df['Region'].nunique()}",
        f"- **Unique customers:** {metrics['unique_customers']:,}",
        "",
        dataframe_to_markdown(columns_df),
        "",
        "## 3. Data Cleaning Summary",
        "",
        dataframe_to_markdown(cleaning_df),
        "",
        "## 4. Exploratory Data Analysis",
        "",
        f"After cleaning, the dataset has {len(df):,} transactions covering {df['Product'].nunique()} products "
        f"across {df['Region'].nunique()} regions between {date_min} and {date_max}. "
        f"Total revenue is {money(metrics['total_revenue'])} from {units(metrics['total_units'])} units sold.",
        "",
        "## 5. Key Statistics",
        "",
        dataframe_to_markdown(stats_df),
        "",
        "## 6. Product-wise Analysis",
        "",
        dataframe_to_markdown(product_df),
        "",
        f"**{top_product['Product']}** leads in revenue with a {top_product['Revenue_Share_Pct']:.1f}% share.",
        "",
        "## 7. Region-wise Analysis",
        "",
        dataframe_to_markdown(region_df),
        "",
        f"**{top_region['Region']}** leads in revenue with a {top_region['Revenue_Share_Pct']:.1f}% share.",
        "",
        "## 8. Monthly Sales Analysis",
        "",
        dataframe_to_markdown(monthly_df),
        "",
        f"**{best_month['Month']}** is the strongest month ({money(best_month['Revenue'])}).",
        "",
        "## 9. Business Insights",
        "",
    ]
    lines += [f"- {text}" for text in insight_data["insights"]]
    lines += ["", "## 10. Business Recommendations", ""]
    lines += [f"{i}. {text}" for i, text in enumerate(insight_data["recommendations"], start=1)]
    lines += [
        "",
        "## 11. Visualizations",
        "",
        "### Product-wise Sales",
        _chart_markdown(charts, "product_bar", "Product sales bar chart", prefix),
        "",
        "### Monthly Sales Trend",
        _chart_markdown(charts, "monthly_line", "Monthly sales line chart", prefix),
        "",
        "### Regional Sales Comparison",
        _chart_markdown(charts, "regional_bar", "Regional sales chart", prefix),
        "",
        "### Product Revenue Distribution",
        _chart_markdown(charts, "product_pie", "Product revenue pie chart", prefix),
        "",
        "## 12. Conclusion",
        "",
        f"The cleaned dataset of {len(df):,} transactions generated {money(metrics['total_revenue'])} in revenue. "
        f"**{top_product['Product']}** is the top revenue product, **{top_region['Region']}** is the top region and "
        f"**{best_month['Month']}** is the strongest month. The recommendations above are based only on this dataset"
        + (", which is fairly small, so they should be confirmed with more data." if len(df) < 500 else "."),
        "",
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def generate_project_summary(df, summary, metrics, product_df, region_df, monthly_df, output_path):
    """Write report/project_summary.md - a one-page summary."""
    top_product = product_df.iloc[0]
    top_region = _top_known_region(region_df)
    best_month = monthly_df.loc[monthly_df["Revenue"].idxmax()]
    lines = [
        "# Project Summary",
        "",
        "**Project:** E-commerce Sales Data Analysis and Visualization",
        "",
        "**Tools:** Python, Pandas, NumPy, Matplotlib, Pytest",
        "",
        "## What was done",
        "",
        f"- Loaded and validated `sales_data.csv` ({summary['initial_rows']:,} raw rows).",
        f"- Cleaned the data ({summary['final_rows']:,} rows kept, {summary['rows_removed_total']:,} removed).",
        "- Calculated revenue, units, transaction and product/region/monthly metrics.",
        "- Created four Matplotlib charts in the `visualizations/` folder.",
        "- Generated insights, recommendations and the full report in `report/analysis_report.md`.",
        "",
        "## Headline results",
        "",
        f"- Total revenue: {money(metrics['total_revenue'])}",
        f"- Total units sold: {units(metrics['total_units'])}",
        f"- Total transactions: {metrics['total_transactions']:,}",
        f"- Average transaction value: {money(metrics['average_transaction'])}",
        f"- Top revenue product: {top_product['Product']}",
        f"- Top region: {top_region['Region']}",
        f"- Strongest month: {best_month['Month']}",
        "",
        "See `analysis_report.md` for the full analysis.",
        "",
    ]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def update_readme(insight_data, readme_path=README_PATH):
    """Fill the Key Findings and Recommendations sections of README.md automatically."""
    readme_path = Path(readme_path)
    if not readme_path.is_file():
        return False

    text = readme_path.read_text(encoding="utf-8")
    findings = "\n".join(f"- {item}" for item in insight_data["insights"])
    recommendations = "\n".join(f"{i}. {item}" for i, item in enumerate(insight_data["recommendations"], start=1))

    for name, content in (("KEY_FINDINGS", findings), ("RECOMMENDATIONS", recommendations)):
        pattern = re.compile(rf"(<!-- {name}_START -->)(.*?)(<!-- {name}_END -->)", re.DOTALL)
        text = pattern.sub(lambda m, c=content: f"{m.group(1)}\n{c}\n{m.group(3)}", text)

    readme_path.write_text(text, encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print(" E-COMMERCE SALES DATA ANALYSIS")
    print("=" * 60)

    try:
        print("\n[1/10] Loading dataset...")
        raw_df = load_data(DATA_PATH)

        print("\n[2/10] Validating dataset...")
        validate_dataset(raw_df)
        print("  All required columns are present.")

        print("\n[3/10] Exploring dataset...")
        explore_data(raw_df)

        print("\n[4/10] Cleaning dataset...")
        df, summary = clean_data(raw_df)
        print_cleaning_summary(summary)

        print("\n[5/10] Calculating metrics...")
        metrics = calculate_metrics(df)
        print(f"  Total revenue:       {money(metrics['total_revenue'])}")
        print(f"  Total units sold:    {units(metrics['total_units'])}")
        print(f"  Total transactions:  {metrics['total_transactions']:,}")
        print(f"  Average transaction: {money(metrics['average_transaction'])}")
        high, low = metrics["highest_transaction"], metrics["lowest_transaction"]
        print(f"  Highest sale:        {money(high['value'])} ({high['product']}, {high['date']})")
        print(f"  Lowest sale:         {money(low['value'])} ({low['product']}, {low['date']})")
        print(f"  Best-selling:        {metrics['best_selling_product']} ({units(metrics['best_selling_units'])} units)")
        print(f"  Top revenue product: {metrics['top_revenue_product']} ({money(metrics['top_revenue_value'])})")

        print("\n[6/10] Product analysis...")
        product_df = product_analysis(df)
        print(product_df.to_string(index=False))

        print("\n[7/10] Region analysis...")
        region_df = region_analysis(df)
        print(region_df.to_string(index=False))

        print("\n[8/10] Monthly analysis...")
        monthly_df = monthly_analysis(df)
        print(monthly_df.to_string(index=False))

        print("\n[9/10] Generating visualizations...")
        charts = generate_visualizations(product_df, region_df, monthly_df, VISUALIZATIONS_DIR)

        print("\n[10/10] Generating insights and reports...")
        insight_data = generate_insights(metrics, product_df, region_df, monthly_df)
        print("\nBusiness insights:")
        for item in insight_data["insights"]:
            print("  - " + item.replace("**", ""))
        print("\nRecommendations:")
        for i, item in enumerate(insight_data["recommendations"], start=1):
            print(f"  {i}. " + item.replace("**", ""))

        generate_report(df, summary, metrics, product_df, region_df, monthly_df,
                        insight_data, charts, REPORT_DIR / "analysis_report.md")
        generate_project_summary(df, summary, metrics, product_df, region_df, monthly_df,
                                 REPORT_DIR / "project_summary.md")
        print("\n  Saved: report/analysis_report.md")
        print("  Saved: report/project_summary.md")
        if update_readme(insight_data):
            print("  Updated: README.md (Key Findings and Recommendations)")

    except FileNotFoundError as error:
        print(f"\nERROR: {error}")
        return 1
    except DataValidationError as error:
        print(f"\nDATA ERROR: {error}")
        return 1
    except Exception as error:  # last safety net so beginners see a readable message
        print(f"\nUNEXPECTED ERROR: {type(error).__name__}: {error}")
        return 1

    print("\nDone! Open report/analysis_report.md to see the full report.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
