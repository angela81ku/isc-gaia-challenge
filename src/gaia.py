import glob, os
import polars as pl
from concurrent.futures import ThreadPoolExecutor

DATA_DIR = "/home/irisowner/dev/data/in"
OUT_FILE = "/home/irisowner/dev/data/out/result.csv"


def _process_file(path):
    df = pl.read_csv(path, comment_prefix="#", columns=["source_id", "bp_flux", "rp_flux"])
    for col in ["bp_flux", "rp_flux"]:
        df = df.with_columns(
            pl.col(col).str.replace_all("NaN", "").str.strip_chars("[]").str.split(",")
            .list.eval(pl.element().cast(pl.Float64, strict=False)).alias(col)
        )
    return df.select([
        "source_id",
        pl.col("bp_flux").list.min().alias("bp_min"),
        pl.col("bp_flux").list.max().alias("bp_max"),
        pl.col("rp_flux").list.min().alias("rp_min"),
        pl.col("rp_flux").list.max().alias("rp_max"),
    ])


def run():
    files = sorted(
        glob.glob(os.path.join(DATA_DIR, "EpochPhotometry_*.csv.gz")),
        key=os.path.getsize, reverse=True
    )
    with ThreadPoolExecutor() as ex:
        parts = list(ex.map(_process_file, files))

    result = (
        pl.concat(parts)
        .with_columns([
            ((pl.col("bp_max") - pl.col("bp_min")) / pl.col("bp_min") * 100).alias("bp_pct"),
            ((pl.col("rp_max") - pl.col("rp_min")) / pl.col("rp_min") * 100).alias("rp_pct"),
        ])
        .with_columns(pl.max_horizontal("bp_pct", "rp_pct").fill_null(0).alias("pct"))
        .filter(pl.col("pct") > 100)
        .select(["source_id", "bp_min", "bp_max", "rp_min", "rp_max", "pct"])
    )

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    result.write_csv(OUT_FILE, include_header=False)
    print(f"Matched sources: {len(result)}")
