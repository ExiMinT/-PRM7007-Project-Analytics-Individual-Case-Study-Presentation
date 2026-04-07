from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from scipy.stats import linregress


DOWNLOADS_DIR = Path.home() / "Downloads"
SALES_FILE = DOWNLOADS_DIR / "BF Sales.csv"
FOOTFALL_FILE = DOWNLOADS_DIR / "footfall.csv"

OUTPUT_WEEKLY_TABLE = DOWNLOADS_DIR / "bf_weekly_sales_vs_footfall_table.csv"
OUTPUT_SUMMARY_TABLE = DOWNLOADS_DIR / "bf_weekly_sales_vs_footfall_summary.csv"
OUTPUT_GRAPH = DOWNLOADS_DIR / "bf_weekly_sales_vs_footfall_graph.html"


def load_weekly_data(sales_path: Path, footfall_path: Path) -> pd.DataFrame:
    sales_df = pd.read_csv(sales_path)
    footfall_df = pd.read_csv(footfall_path)

    required_sales_cols = {"WEEK_ID", "NET_SALES"}
    required_footfall_cols = {"WEEK_ID", "SUM_FF"}

    if not required_sales_cols.issubset(sales_df.columns):
        missing = sorted(required_sales_cols - set(sales_df.columns))
        raise ValueError(f"Missing columns in sales file: {missing}")

    if not required_footfall_cols.issubset(footfall_df.columns):
        missing = sorted(required_footfall_cols - set(footfall_df.columns))
        raise ValueError(f"Missing columns in footfall file: {missing}")

    sales_df["WEEK_ID"] = pd.to_numeric(sales_df["WEEK_ID"], errors="coerce")
    sales_df["NET_SALES"] = pd.to_numeric(sales_df["NET_SALES"], errors="coerce")

    footfall_df["WEEK_ID"] = pd.to_numeric(footfall_df["WEEK_ID"], errors="coerce")
    footfall_df["SUM_FF"] = pd.to_numeric(footfall_df["SUM_FF"], errors="coerce")

    weekly_sales = (
        sales_df.dropna(subset=["WEEK_ID", "NET_SALES"])
        .groupby("WEEK_ID", as_index=False)["NET_SALES"]
        .sum()
        .rename(columns={"NET_SALES": "NET_SALES_PER_WEEK"})
    )

    weekly_footfall = (
        footfall_df.dropna(subset=["WEEK_ID", "SUM_FF"])
        .groupby("WEEK_ID", as_index=False)["SUM_FF"]
        .sum()
        .rename(columns={"SUM_FF": "FOOTFALL_PER_WEEK"})
    )

    merged = (
        weekly_sales.merge(weekly_footfall, on="WEEK_ID", how="inner")
        .sort_values("WEEK_ID")
        .reset_index(drop=True)
    )

    if len(merged) < 2:
        raise ValueError("At least 2 matching weeks are required to run regression.")

    return merged


def run_regression(weekly_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = linregress(
        weekly_df["FOOTFALL_PER_WEEK"],
        weekly_df["NET_SALES_PER_WEEK"],
    )

    weekly_df = weekly_df.copy()
    weekly_df["PREDICTED_NET_SALES"] = (
        result.intercept + result.slope * weekly_df["FOOTFALL_PER_WEEK"]
    )
    weekly_df["RESIDUAL"] = (
        weekly_df["NET_SALES_PER_WEEK"] - weekly_df["PREDICTED_NET_SALES"]
    )

    summary_df = pd.DataFrame(
        [
            {
                "WEEKS_USED": len(weekly_df),
                "GRADIENT_SLOPE": result.slope,
                "INTERCEPT": result.intercept,
                "CORRELATION_R": result.rvalue,
                "R_SQUARED": result.rvalue**2,
                "P_VALUE": result.pvalue,
                "STD_ERR": result.stderr,
            }
        ]
    )

    return weekly_df, summary_df


def build_graph(weekly_df: pd.DataFrame, summary_df: pd.DataFrame, output_path: Path) -> None:
    slope = summary_df.loc[0, "GRADIENT_SLOPE"]
    intercept = summary_df.loc[0, "INTERCEPT"]
    p_value = summary_df.loc[0, "P_VALUE"]
    r_value = summary_df.loc[0, "CORRELATION_R"]

    line_x = [
        weekly_df["FOOTFALL_PER_WEEK"].min(),
        weekly_df["FOOTFALL_PER_WEEK"].max(),
    ]
    line_y = [intercept + slope * x for x in line_x]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=weekly_df["FOOTFALL_PER_WEEK"],
            y=weekly_df["NET_SALES_PER_WEEK"],
            mode="markers",
            name="Weekly data",
            text=weekly_df["WEEK_ID"].astype(int).astype(str),
            hovertemplate="Week %{text}<br>Footfall: %{x:,}<br>Net Sales: %{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            name="Regression line",
            hovertemplate="Footfall: %{x:,}<br>Predicted Net Sales: %{y:,.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=(
            "BF Weekly Net Sales vs Footfall"
            f"<br><sup>Gradient={slope:,.6f}, r={r_value:,.4f}, p-value={p_value:.6g}</sup>"
        ),
        xaxis_title="Weekly Footfall",
        yaxis_title="Weekly Net Sales",
        template="plotly_white",
    )

    fig.write_html(output_path, include_plotlyjs="cdn")


def main() -> None:
    weekly_df = load_weekly_data(SALES_FILE, FOOTFALL_FILE)
    weekly_results_df, summary_df = run_regression(weekly_df)

    weekly_results_df.to_csv(OUTPUT_WEEKLY_TABLE, index=False)
    summary_df.to_csv(OUTPUT_SUMMARY_TABLE, index=False)
    build_graph(weekly_results_df, summary_df, OUTPUT_GRAPH)

    print("Regression complete.")
    print(f"Weekly table saved to: {OUTPUT_WEEKLY_TABLE}")
    print(f"Summary table saved to: {OUTPUT_SUMMARY_TABLE}")
    print(f"Graph saved to: {OUTPUT_GRAPH}")
    print()
    print("Regression summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
