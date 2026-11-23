# E-commerce Sales Data Analysis and Visualization

A beginner-friendly, portfolio-ready data analysis project built with **Python, Pandas, NumPy and Matplotlib**.
It loads sales data, cleans it, explores it, calculates business metrics, draws professional charts and writes a full report - all automatically from the real dataset.

## Project Overview

The script reads `data/sales_data.csv`, checks and cleans the data, calculates revenue metrics, analyses sales by product, region and month, creates four charts, and generates insights and recommendations. Nothing is hardcoded: change the CSV, run again, and every number, chart, insight and report updates.

## Objectives

- Load and validate a CSV dataset with clear error messages
- Clean data (duplicates, missing values, invalid dates, negative values, wrong totals)
- Perform exploratory data analysis (EDA)
- Create professional Matplotlib visualizations
- Generate business insights and recommendations from the data
- Produce a complete Markdown report
- Test the logic with Pytest

## Features

- Graceful handling of missing files, empty data, wrong columns, bad dates and bad numbers
- Data cleaning summary showing exactly what changed
- `Total_Sales` recalculated as `Quantity x Unit_Price` where missing or inconsistent
- Product, region and monthly analysis with revenue share
- 4 charts saved as PNG files
- Auto-generated `analysis_report.md` and `project_summary.md`
- Key Findings and Recommendations in this README are filled in automatically
- Pytest unit tests that use sample data only

## Technologies Used

- Python 3.10+
- Pandas
- NumPy
- Matplotlib
- Pytest

## Folder Structure

```
sales-data-analysis/
|-- data/
|   `-- sales_data.csv
|-- visualizations/
|   |-- product_sales_bar_chart.png
|   |-- monthly_sales_line_chart.png
|   |-- regional_sales_chart.png
|   `-- product_revenue_pie_chart.png
|-- report/
|   |-- analysis_report.md
|   `-- project_summary.md
|-- tests/
|   `-- test_analysis.py
|-- main.py
|-- requirements.txt
|-- README.md
`-- .gitignore
```

## Dataset Information

File: `data/sales_data.csv`

| Column | Description |
| --- | --- |
| Date | Date of the transaction |
| Product | Product sold |
| Quantity | Units sold |
| Unit_Price | Price per unit |
| Customer_ID | Customer identifier |
| Region | Sales region |
| Total_Sales | Quantity x Unit_Price |

## Installation

```bash
# 1. Open a terminal in the project folder
cd sales-data-analysis

# 2. (Recommended) create a virtual environment
python -m venv venv

# 3. Activate it
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac / Linux

# 4. Install the libraries
pip install -r requirements.txt
```

## How to Run the Project

Make sure `data/sales_data.csv` exists, then:

```bash
python main.py
```

This prints the full analysis in the terminal, saves the charts to `visualizations/`, writes the reports to `report/`, and updates the Key Findings below.

## How to Run Tests

```bash
pytest
```

The tests use small sample DataFrames, so they do not need the real dataset.

## Visualization Previews

| Product Sales | Monthly Trend |
| --- | --- |
| ![Product sales](visualizations/product_sales_bar_chart.png) | ![Monthly trend](visualizations/monthly_sales_line_chart.png) |

| Regional Sales | Revenue Distribution |
| --- | --- |
| ![Regional sales](visualizations/regional_sales_chart.png) | ![Revenue distribution](visualizations/product_revenue_pie_chart.png) |

## Key Findings

<!-- KEY_FINDINGS_START -->
- **Laptop** generates the highest revenue (3,889,210.00), contributing 31.4% of total revenue.
- **Laptop** is also the best-selling product by volume (136 units).
- The top 3 products together account for 77.9% of total revenue.
- **North** is the highest-revenue region (3,983,635.00, 32.2% of total revenue).
- **West** is the lowest-revenue region (2,123,922.00, 17.2% of total revenue).
- **2024-03** is the strongest month with 4,485,006.00 in revenue.
- **2024-04** is the weakest month with 1,103,468.00 in revenue.
- Revenue in the latest month (2024-04) is down 73.2% compared with the first month (2024-01).
- The average transaction value is 123,650.48; the largest single sale was 373,932.00 (Phone on 2024-01-16).
<!-- KEY_FINDINGS_END -->

## Business Recommendations

<!-- RECOMMENDATIONS_START -->
1. Keep **Laptop** well stocked and prominently promoted - it is the biggest revenue driver.
2. Investigate **Monitor** (lowest revenue, 10.9% share): test a promotion or pricing change before deciding to reduce stock.
3. Study what works in **North** and try similar campaigns in **West**.
4. Plan inventory and marketing ahead of high-demand periods such as **2024-03**.
5. Raise the average transaction value (123,650.48) with cross-selling, bundles or minimum-order offers.
<!-- RECOMMENDATIONS_END -->

## Future Improvements

- Add a profit analysis if cost data becomes available
- Customer segmentation (repeat vs one-time buyers)
- Interactive dashboard with Streamlit or Plotly
- Command-line options for choosing the input file
- Forecasting future monthly sales
- Export the report as PDF
