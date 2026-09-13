"""Reformat each method's data.csv into a cleaner summary table.

Reads data.csv from adagrad/, adam/, gradient_descent/, and newton/, then:
  - keeps Objective Function, Start X, Start Y, Learning Rate
  - splits the 'x' column ("end_x, end_y") into End X / End Y
  - renames fun -> Function At End, nit -> Number of Iterations
  - keeps avg_runtime / stdev_runtime
  - rounds all floats to 6 decimal places

Writes data_clean.csv into each method's directory and a combined
clean_results.csv (with a Method column) in this directory.
"""

import polars as pl

METHOD_DIRS = ["adagrad", "adam", "gradient_descent", "newton"]
FLOAT_COLS = [
    "Start X",
    "Start Y",
    "Learning Rate",
    "End X",
    "End Y",
    "Function At End",
    "Avg Runtime",
    "Stdev Runtime",
]


def clean_method(method_dir: str) -> pl.DataFrame:
    df = pl.read_csv(f"{method_dir}/data.csv")

    # newton's step-size column is named decay_rate instead of lr
    lr_col = "lr" if "lr" in df.columns else "decay_rate"

    df = df.with_columns(
        pl.col("x").str.split(", ").list.get(0).cast(pl.Float64).alias("End X"),
        pl.col("x").str.split(", ").list.get(1).cast(pl.Float64).alias("End Y"),
    )

    df = df.select(
        pl.col("Objective Function"),
        pl.col("Start X"),
        pl.col("Start Y"),
        pl.col(lr_col).alias("Learning Rate"),
        pl.col("End X"),
        pl.col("End Y"),
        pl.col("fun").alias("Function At End"),
        pl.col("nit").alias("Number of Iterations"),
        pl.col("avg_runtime").alias("Avg Runtime"),
        pl.col("stdev_runtime").alias("Stdev Runtime"),
    )

    df = df.with_columns([pl.col(c).round(6) for c in FLOAT_COLS])
    return df


def main() -> None:
    combined = []
    for method_dir in METHOD_DIRS:
        df = clean_method(method_dir)
        df.write_csv(f"{method_dir}/data_clean.csv")
        combined.append(df.with_columns(pl.lit(method_dir).alias("Method")))

    combined_df = pl.concat(combined).select(
        ["Method"] + [c for c in combined[0].columns if c != "Method"]
    )
    combined_df.write_csv("clean_results.csv")
    print(f"Wrote clean_results.csv ({combined_df.height} rows)")


if __name__ == "__main__":
    main()
