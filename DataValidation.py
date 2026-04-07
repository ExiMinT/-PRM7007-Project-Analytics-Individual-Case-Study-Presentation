import argparse
from pathlib import Path

import numpy as np
import pandas as pd

required_columns = {"WEEK_ID", "Organisation_Code", "SUM_FF"}

parser = argparse.ArgumentParser(
    description="Validate branch footfall values for a week range and export suspicious rows to CSV."
)
parser.add_argument(
    "--input",
    type=Path,
    default=Path.home() / "Downloads" / "footfallbybrch.csv",
    help="Path to source CSV (default: ~/Downloads/footfallbybrch.csv).",
)
parser.add_argument(
    "--start-week",
    type=int,
    default=2025035,
    help="Start WEEK_ID inclusive (default: 2025035).",
)
parser.add_argument(
    "--end-week",
    type=int,
    default=2025050,
    help="End WEEK_ID inclusive (default: 2025050).",
)
parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="Output CSV path. Defaults to Tabs/footfall_validation_flags_<start>_<end>.csv.",
)

args = parser.parse_args()
input_csv = args.input
start_week = args.start_week
end_week = args.end_week
output_csv = args.output or (
    Path(__file__).resolve().parent / f"footfall_validation_flags_{start_week}_{end_week}.csv"
)

df = pd.read_csv(input_csv)
missing = required_columns - set(df.columns)
if missing:
    missing_cols = ", ".join(sorted(missing))
    raise ValueError(f"CSV missing required columns: {missing_cols}")

df = df[list(required_columns)].copy()
for col in ["WEEK_ID", "Organisation_Code", "SUM_FF"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["WEEK_ID", "Organisation_Code", "SUM_FF"]).copy()
df["WEEK_ID"] = df["WEEK_ID"].astype(int)
df["Organisation_Code"] = df["Organisation_Code"].astype(int)
df["SUM_FF"] = df["SUM_FF"].astype(float)

base = df[~df["WEEK_ID"].between(start_week, end_week)].copy()
grouped = base.groupby("Organisation_Code")["SUM_FF"]

stats = grouped.agg(
    base_n="count",
    base_mean="mean",
    base_std=lambda s: float(s.std(ddof=0)),
    base_median="median",
    base_min="min",
    base_max="max",
    q01=lambda s: float(s.quantile(0.01)),
    q99=lambda s: float(s.quantile(0.99)),
).reset_index()

mad = (
    grouped.apply(lambda s: float(np.median(np.abs(s.to_numpy() - np.median(s.to_numpy())))))
    .rename("base_mad")
    .reset_index()
)
stats = stats.merge(mad, on="Organisation_Code", how="left")

target = df[df["WEEK_ID"].between(start_week, end_week)].copy()
target = target.merge(stats, on="Organisation_Code", how="left")

prev = df[["WEEK_ID", "Organisation_Code", "SUM_FF"]].copy()
prev["WEEK_ID"] = prev["WEEK_ID"] + 1000
prev = prev.rename(columns={"SUM_FF": "prev_year_sum_ff"})
target = target.merge(prev, on=["WEEK_ID", "Organisation_Code"], how="left")

target["z_score"] = (target["SUM_FF"] - target["base_mean"]) / target["base_std"]
target.loc[(target["base_std"].isna()) | (target["base_std"] == 0), "z_score"] = np.nan

target["robust_z"] = 0.6745 * (target["SUM_FF"] - target["base_median"]) / target["base_mad"]
target.loc[(target["base_mad"].isna()) | (target["base_mad"] == 0), "robust_z"] = np.nan

target["ratio_to_median"] = target["SUM_FF"] / target["base_median"]
target.loc[target["base_median"] == 0, "ratio_to_median"] = np.nan

target["yoy_pct"] = (target["SUM_FF"] - target["prev_year_sum_ff"]) / target["prev_year_sum_ff"]
target.loc[target["prev_year_sum_ff"] == 0, "yoy_pct"] = np.nan

target["flag_stat_outlier"] = (
    (target["base_n"] >= 20)
    & (
        (target["z_score"].abs() >= 3)
        | (target["robust_z"].abs() >= 3.5)
        | (target["SUM_FF"] < target["q01"])
        | (target["SUM_FF"] > target["q99"])
    )
)

target["flag_extreme_spike"] = (
    (target["base_n"] >= 20)
    & (target["base_median"] >= 500)
    & (
        (target["ratio_to_median"] >= 5)
        | (target["SUM_FF"] >= 3 * target["q99"])
        | (target["SUM_FF"] >= 2 * target["base_max"])
    )
)

target["flag_extreme_drop"] = (
    (target["base_n"] >= 20)
    & (target["base_median"] >= 1500)
    & ((target["SUM_FF"] == 0) | (target["ratio_to_median"] <= 0.2))
)

target["flag_extreme_yoy"] = (
    (target["prev_year_sum_ff"] >= 1000) & ((target["yoy_pct"] >= 3) | (target["yoy_pct"] <= -0.8))
)

target["critical"] = target["flag_extreme_spike"] | target["flag_extreme_drop"] | target["flag_extreme_yoy"]
target["review"] = (
    target["flag_stat_outlier"] | (target["ratio_to_median"].abs() >= 2.5) | (target["yoy_pct"].abs() >= 0.5)
)

target["reason"] = ""
target.loc[target["flag_extreme_spike"], "reason"] += "extreme_spike;"
target.loc[target["flag_extreme_drop"], "reason"] += "extreme_drop;"
target.loc[target["flag_extreme_yoy"], "reason"] += "extreme_yoy;"
target.loc[target["flag_stat_outlier"], "reason"] += "stat_outlier;"
target["reason"] = target["reason"].str.rstrip(";")

target["severity"] = np.where(target["critical"], "critical", np.where(target["review"], "review", "ok"))

flagged = target[target["severity"] != "ok"].copy()
flagged = flagged.sort_values(["severity", "WEEK_ID", "Organisation_Code"], ascending=[True, True, True])

output_cols = [
    "WEEK_ID",
    "Organisation_Code",
    "SUM_FF",
    "severity",
    "reason",
    "base_n",
    "base_median",
    "base_mean",
    "base_std",
    "base_mad",
    "q01",
    "q99",
    "base_min",
    "base_max",
    "prev_year_sum_ff",
    "yoy_pct",
    "ratio_to_median",
    "z_score",
    "robust_z",
]
output_csv.parent.mkdir(parents=True, exist_ok=True)
flagged[output_cols].to_csv(output_csv, index=False)

print(f"Input: {input_csv}")
print(f"Output: {output_csv}")
print(f"Target rows assessed: {len(target)}")
print(f"Flagged rows exported: {len(flagged)}")
