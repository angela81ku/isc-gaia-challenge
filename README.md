# Gaia DR3 Variable Star Finder
### InterSystems Employee Programming Challenge #1

Identifies every astronomical source in the Gaia DR3 epoch-photometry archive whose Blue Photometer (BP) or Red Photometer (RP) flux changed by more than 100% across all valid observations.

**Result: 57,099 variable sources** across 20 input files, processed in ~1.3 seconds.

---

## How It Works

### Pipeline Overview

```
20x EpochPhotometry_*.csv.gz
        |
        | ThreadPoolExecutor (one thread per file)
        v
  _process_file() — Polars: decompress gzip, parse CSV,
                    extract bp_flux / rp_flux columns,
                    compute per-source min/max
        |
        | pl.concat() — merge all 20 DataFrames
        v
  filter: pct_change > 100%  →  write data/out/result.csv
        |
        v
  [benchmark timer stops here]

  ai_summary() — bonus step, runs after timing, requires GEMINI_API_KEY
        |
        v
  data/out/ai_summary.txt — Gemini interpretation of top 5 sources
```

---

### Formula

For each source, percentage change is computed per band:

```
pct_change = (max_flux - min_flux) / min_flux × 100
```

The result is the larger of the two bands (BP and RP). Sources where this exceeds 100% are written to output.

Polars applies this as a vectorized column expression across all rows at once — no Python loop, no row-by-row iteration:

```python
result = (
    pl.concat(parts)
    .with_columns([
        ((pl.col("bp_max") - pl.col("bp_min")) / pl.col("bp_min") * 100).alias("bp_pct"),
        ((pl.col("rp_max") - pl.col("rp_min")) / pl.col("rp_min") * 100).alias("rp_pct"),
    ])
    .with_columns(pl.max_horizontal("bp_pct", "rp_pct").fill_null(0).alias("pct"))
    .filter(pl.col("pct") > 100)
)
```

NaN, null, and empty flux values are ignored during parsing — a source only appears if a real band swung past 100%.

---

### Parallel Ingestion with Polars

```python
with ThreadPoolExecutor() as ex:
    parts = list(ex.map(_process_file, files))
```

Files are sorted largest-first so the biggest gzip stream starts immediately. Each thread handles one file independently: decompress → CSV parse → min/max extraction. Polars reads only the three needed columns (`source_id`, `bp_flux`, `rp_flux`) out of 48 — the other 45 columns are never decoded.

The per-row flux arrays arrive as strings (`"[1820.8, 2013.8, NaN, ...]"`). These are parsed entirely inside Polars' Rust engine in a single expression chain:

```python
pl.col("bp_flux")
  .str.replace_all("NaN", "")
  .str.strip_chars("[]")
  .str.split(",")
  .list.eval(pl.element().cast(pl.Float64, strict=False))
```

`list.min()` and `list.max()` are then computed in one scan per column, with no separate Python calls.

---

### AI Summary (bonus)

After the benchmark timer stops, the top 5 most variable sources are sent to Google Gemini for astronomical interpretation. This runs outside the timed section and is skipped gracefully if `GEMINI_API_KEY` is not set.

The code first attempts the **InterSystems AI Hub** path (`langchain_intersystems`) — AI Hub manages LLM configurations (API keys, model names) inside IRIS via `%ConfigStore.Configuration`, providing a unified `init_chat_model()` interface. If AI Hub is unavailable, it falls back to calling Gemini directly via `langchain-google-genai`.

---

## Performance Journey

We profiled the baseline to find the real bottleneck before optimizing:

### Bottleneck Profile (3-file sample, gzip + csv.reader baseline)

| Layer | Time | Share |
|-------|------|-------|
| gzip decompress | 0.711s | **41%** |
| csv.reader parsing | 0.913s | **53%** |
| json.loads (minmax) | 0.105s | 6% |

This told us that `json.loads` — the obvious target — was only 6% of runtime. The real cost was I/O and row parsing.

### What We Tried (20 files, 8 CPUs)

| Method | Time | Speedup | Why |
|--------|------|---------|-----|
| Baseline: gzip + csv.reader + json | 5.2s | 1.00x | — |
| A: isal (faster gzip decoder) | 5.0s | 1.01x | multiprocessing already parallelizes per-file; isal's multi-thread advantage is cancelled out |
| B: isal + pandas | 5.5s | 0.95x | DataFrame construction overhead > savings for single-pass workload |
| C: Parquet + json.loads | 2.9s | 1.82x | Columnar I/O — only 3 of 47 columns read |
| D: Parquet + numpy vectorized | 3.7s | 1.40x | Variable-length arrays require padding; genfromtxt overhead > gains |
| **E: Polars (current)** | **~1.3s** | **~4x** | Rust gzip + Arrow columnar parse, no intermediate Python lists |

### Why Polars

- **Columnar parsing:** only `source_id`, `bp_flux`, `rp_flux` are decoded — the other 45 CSV columns are skipped entirely.
- **Rust gzip:** decompression happens inside Polars' Rust engine, not Python's `gzip` module.
- **No intermediate list:** the flux string is split and cast to Float64 in one expression; `list.min()` and `list.max()` scan it once.

### Why Numpy Vectorization Lost

Gaia flux arrays have variable length — from a handful to hundreds of values per row. To use numpy's matrix operations, every row must be padded to the same length. The padding itself requires a Python loop, and `np.genfromtxt` on a large padded block costs more than the vectorization saves. Numpy vectorization is optimal for **fixed-shape** data (e.g. image pixels), not variable-length arrays.

---

## Correctness Guarantee

Every optimization was verified against a fixed oracle:

```bash
python -m pytest test_runchallenge.py    # 27 unit tests — all pass
```

---

## Installation

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop)
- [git](https://git-scm.com)
- Google Gemini API key — free at [aistudio.google.com](https://aistudio.google.com) (AI summary step only)

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | No | — | Enables AI summary step after benchmark |

Set on Windows:
```cmd
setx GEMINI_API_KEY "AIzaSy..."
```

### Run

```bash
git clone https://github.com/angela81ku/isc-gaia-challenge.git
cd isc-gaia-challenge
docker-compose up --build -d
docker-compose exec iris iris session iris -U USER
USER> do ^RunScript
```

### Output

| File | Contents |
|------|----------|
| `data/out/result.csv` | `source_id, bp_min_flux, bp_max_flux, rp_min_flux, rp_max_flux, percentage_change` — 57,099 rows |
| `data/out/ai_summary.txt` | Gemini interpretation of the top 5 most variable sources (only if `GEMINI_API_KEY` set) |

---

## Project Structure

```
├── src/
│   ├── gaia.py              # Core logic — parallel parse, Polars, AI summary
│   ├── register_llm.py      # Registers Gemini config in IRIS AI Hub at startup
│   └── RunScript.mac        # IRIS entry point  (do ^RunScript)
├── data/
│   ├── in/                  # 20 EpochPhotometry_*.csv.gz input files
│   └── out/result.csv       # Output — 57,099 variable sources
├── Dockerfile               # IRIS Community 2026.3.0AI + Polars + langchain
├── docker-compose.yml
└── startup.sh               # Compiles RunScript.mac + registers Gemini config
```

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Database & runtime | InterSystems IRIS Community 2026.3.0AI | Embedded Python runtime, AI Hub SDK |
| Data processing | Polars (Rust/Arrow) | Columnar CSV parse, vectorized min/max, no Python loops |
| Parallel I/O | Python `ThreadPoolExecutor` | One thread per file, parallel gzip decompress |
| AI interpretation | Google Gemini via langchain-google-genai | Free LLM, runs after benchmark timer |
| AI Hub | InterSystems AI Hub (EAP) + langchain_intersystems | LLM config managed inside IRIS |

---

## Data Attribution

Input: ESA Gaia DR3 Epoch Photometry.
Column definitions: [Gaia Archive Documentation](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_photometry/ssec_dm_epoch_photometry.html)
How to cite Gaia: [Credits](https://gea.esac.esa.int/archive/documentation/credits.html)
