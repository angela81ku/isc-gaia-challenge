import glob, os
import polars as pl
from concurrent.futures import ThreadPoolExecutor

DATA_DIR = "/home/irisowner/dev/data/in"
OUT_FILE = "/home/irisowner/dev/data/out/result.csv"
AI_SUMMARY_FILE = "/home/irisowner/dev/data/out/ai_summary.txt"


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


def _build_prompt(count: int, result: pl.DataFrame) -> str:
    top = result.sort("pct", descending=True).head(5)
    top_text = "\n".join(
        f"  source_id={r['source_id']}, pct_change={r['pct']:.1f}%"
        for r in top.iter_rows(named=True)
    )
    return (
        f"You analyzed {count} variable astronomical sources from the Gaia DR3 epoch "
        f"photometry archive. These sources showed BP or RP flux variability exceeding 100%.\n\n"
        f"Top 5 most variable sources:\n{top_text}\n\n"
        f"Write a concise 3-sentence scientific summary of these findings."
    )


def ai_summary(count: int, result: pl.DataFrame):
    """Generate an AI summary of findings using InterSystems AI Hub + Gemini."""
    summary = None

    # Primary path: AI Hub via langchain_intersystems
    try:
        import iris
        from langchain_intersystems.chat_models import init_chat_model
        llm = init_chat_model("gemini", iris.createIRIS())
        response = llm.invoke(_build_prompt(count, result))
        summary = response.content if hasattr(response, "content") else str(response)
        print("AI summary generated via AI Hub.")
    except Exception as e:
        print(f"AI Hub path failed ({e}), trying fallback...")

    # Fallback: call Gemini directly via langchain-google-genai
    if summary is None:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            api_key = os.environ.get("GEMINI_API_KEY", "")
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
            response = llm.invoke(_build_prompt(count, result))
            summary = response.content if hasattr(response, "content") else str(response)
            print("AI summary generated via Gemini.")
        except Exception as e:
            summary = f"AI summary unavailable: {e}"

    os.makedirs(os.path.dirname(AI_SUMMARY_FILE), exist_ok=True)
    with open(AI_SUMMARY_FILE, "w") as f:
        f.write(summary + "\n")
    print(f"AI summary written to {AI_SUMMARY_FILE}")


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
    ai_summary(len(result), result)
