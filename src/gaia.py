# Gaia DR3 Variable Source Finder
# InterSystems Employee Programming Challenge #1
#
# Approach: Polars (Rust-backed DataFrames) + ThreadPoolExecutor.
# Each of the 20 gzip-compressed CSV files is processed on its own thread.
# Polars decompresses and parses natively in Rust — no Python gzip module,
# no intermediate lists. Only 3 of 48 columns are ever decoded.
#
# IRIS Embedded Python bridges ObjectScript (RunScript.mac) to this module
# via ##class(%SYS.Python).Import("gaia"). No external process, no TCP —
# the Python runtime lives inside the IRIS process.
#
# AI Hub note: ai_summary() attempts the InterSystems AI Hub path first
# (langchain_intersystems), which manages LLM configs inside IRIS via
# %ConfigStore.Configuration. In the current EAP release the SDK requires
# an iris.IRISConnection (TCP), which is not available inside Embedded Python,
# so it falls back to calling Gemini directly via langchain-google-genai.
# ai_summary() is intentionally NOT called from run() — it is a bonus step
# that runs after the benchmark timer and requires GEMINI_API_KEY.

import glob, os
import polars as pl
from concurrent.futures import ThreadPoolExecutor

DATA_DIR = "/home/irisowner/dev/data/in"
OUT_FILE = "/home/irisowner/dev/data/out/result.csv"
AI_SUMMARY_FILE = "/home/irisowner/dev/data/out/ai_summary.txt"


def _process_file(path):
    # Read only the three needed columns — Polars skips the other 45 entirely.
    df = pl.read_csv(path, comment_prefix="#", columns=["source_id", "bp_flux", "rp_flux"])
    # Each flux cell is a string like "[1820.8, 2013.8, NaN, ...]".
    # Strip brackets, split on commas, cast to Float64 (NaN → null, skipped by min/max).
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
    # Sort largest file first so the heaviest decompression starts immediately.
    files = sorted(
        glob.glob(os.path.join(DATA_DIR, "EpochPhotometry_*.csv.gz")),
        key=os.path.getsize, reverse=True
    )
    # One thread per file — I/O and Rust decompression release the GIL,
    # so threads run in parallel despite Python's GIL.
    with ThreadPoolExecutor() as ex:
        parts = list(ex.map(_process_file, files))

    # Concatenate all per-file results, compute percentage change, filter, write.
    # Formula: pct = (max - min) / min * 100, take the larger of BP and RP.
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


def ai_summary():
    # Bonus step — called separately, never inside run().
    # Reads result.csv, builds a prompt from the top 5 sources, calls Gemini.
    result = pl.read_csv(OUT_FILE, has_header=False,
                         new_columns=["source_id", "bp_min", "bp_max", "rp_min", "rp_max", "pct"])
    count = len(result)
    top = result.sort("pct", descending=True).head(5)
    top_text = "\n".join(
        f"  source_id={r['source_id']}, pct_change={r['pct']:.1f}%"
        for r in top.iter_rows(named=True)
    )
    prompt = (
        f"You analyzed {count} variable astronomical sources from the Gaia DR3 epoch "
        f"photometry archive. These sources showed BP or RP flux variability exceeding 100%.\n\n"
        f"Top 5 most variable sources:\n{top_text}\n\n"
        f"Write a concise 3-sentence scientific summary of these findings."
    )

    summary = None

    # Primary: AI Hub — reads LLM config (model, api_key) from IRIS %ConfigStore.
    # Requires iris.IRISConnection; not available in Embedded Python (EAP limitation).
    try:
        import iris
        from langchain_intersystems.chat_models import init_chat_model
        conn = iris.connect("localhost", 1972, "USER", "_SYSTEM", "SYS")
        llm = init_chat_model("gemini", conn)
        response = llm.invoke(prompt)
        summary = response.content if hasattr(response, "content") else str(response)
        print("AI summary generated via AI Hub.")
    except Exception as e:
        print(f"AI Hub unavailable ({e}), trying direct Gemini...")

    # Fallback: call Gemini directly via langchain-google-genai.
    if summary is None:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            api_key = os.environ.get("GEMINI_API_KEY", "")
            if not api_key:
                print("GEMINI_API_KEY not set, skipping AI summary.")
                return
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
            response = llm.invoke(prompt)
            summary = response.content if hasattr(response, "content") else str(response)
            print("AI summary generated via Gemini.")
        except Exception as e:
            print(f"AI summary skipped: {e}")
            return

    os.makedirs(os.path.dirname(AI_SUMMARY_FILE), exist_ok=True)
    with open(AI_SUMMARY_FILE, "w") as f:
        f.write(summary + "\n")
    print(f"AI summary written to {AI_SUMMARY_FILE}")
